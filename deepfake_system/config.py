"""Central configuration for the deepfake detection system."""
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Load the repo's `env` file before any setting below reads os.environ.
# Without this every DFD_* value in that file is inert and the app runs
# on local JSON no matter what the file says.
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from app.envfile import load as _load_env, summary as _env_summary
    ENV_INFO = _load_env()
    if os.environ.get("DFD_ENV_QUIET", "").strip().lower() not in ("1", "true"):
        print(_env_summary(ENV_INFO))
except Exception as _exc:                     # never block startup on this
    ENV_INFO = {"path": None, "found": 0, "applied": [], "error": str(_exc)}

# Where a trained run lives when the app is served. Defaults to runs/v1
# next to this file so a checkpoint copied there is found on any machine;
# override with DEEPFAKE_RUNS when the run directory is elsewhere.
_RUNS = Path(os.environ.get("DEEPFAKE_RUNS",
                            Path(__file__).resolve().parent / "runs" / "v1"))


@dataclass
class DataConfig:
    # Root that holds the four extracted datasets.
    root: Path = Path("/content/data")

    # Sub-directories after unzipping. Adjust if your folder names differ.
    sources: dict = field(default_factory=lambda: {
        "ffpp":    "FF++",
        "celebdf": "celebdf_faces",
        "df40":    "df40_frames",
        "wild":    "wild",
    })

    manifest: Path = Path("/content/manifest.csv")

    # Frames per clip fed to the temporal head. 8 rather than 16 because
    # Celeb-DF crops here are 32 frames per video: at clip_len 16 with
    # stride 3 the sampler runs off the end and pads by repeating the
    # last frame, so ~31% of every clip is one frozen image.
    clip_len: int = 8
    # Stride between sampled frames inside a video (in frames).
    frame_stride: int = 3
    # Face crop size. 224 rather than 300 to fit a 16 GB T4: train.py
    # holds the clean and degraded graphs in memory at once, so a step
    # costs 2 x batch x clip_len images of activations.
    img_size: int = 224

    # Splitting is done by video/identity, never by frame.
    val_ratio: float = 0.12
    test_ratio: float = 0.13
    seed: int = 1337

    # Sources held out entirely from training.
    #
    # Run A kept "wild" here, which is where the unseen-source number
    # (69.4% acc, AUC 0.778) comes from - that measurement is banked and
    # belongs to that checkpoint. The deployed model splits wild normally
    # instead: held-out data is reserved for measurement, not permanently
    # excluded from the product. Its test slice (~105 videos) still gives a
    # real in-the-wild number for the shipped model.
    holdout_sources: tuple = ()
    # DF40 generator families held out to measure unseen-generator behaviour.
    # Families actually present in this DF40 archive: inswap (600),
    # dit_ (600), deepfacelab (585), collabdiff (250). CollabDiff is held
    # out because it is the smallest, so the unseen-generator probe costs
    # the least training data - and DiT stays in, keeping the diffusion
    # coverage that DF40 was added for.
    # NB: matches the method name build_manifest records, which is
    # "collab" - that token is checked before "collabdiff".
    holdout_methods: tuple = ("collab",)


@dataclass
class ModelConfig:
    backbone: str = "tf_efficientnetv2_s"     # timm name
    pretrained: bool = True
    dropout: float = 0.3
    # Temporal aggregator over the clip: "attention" | "gru" | "mean"
    temporal: str = "attention"
    # Adds a parallel high-frequency (SRM) stream. Costs ~30% speed,
    # buys a few points on compressed video.
    use_srm: bool = True


@dataclass
class TrainConfig:
    # 16 with early stopping (train.py --patience) rather than a guess.
    # An epoch here is one clip per training video, so epochs are
    # smaller than the clips_per_video=2 in train.py implies - the
    # sampler's index range caps at the video count.
    epochs: int = 16
    batch_size: int = 8            # clips per batch (T4). Use 16 on A100.
    accum_steps: int = 4           # effective batch 32
    lr: float = 2.5e-4
    backbone_lr_mult: float = 0.25
    weight_decay: float = 0.02
    warmup_epochs: int = 2
    label_smoothing: float = 0.05
    amp: bool = True
    ema_decay: float = 0.999
    # Scaled to the machine. Colab free gives 2 vCPUs, Pro gives 8-12.
    # degrade_clip is ~70% of the per-item CPU cost, so on a fast GPU the
    # loader is what starves first if this is left low.
    num_workers: int = min(8, max(2, (os.cpu_count() or 4) - 2))

    # Probability that a training clip is degraded to look like a
    # messenger re-upload.
    degrade_prob: float = 0.55
    # Weight on the clean/degraded agreement loss. This is what keeps
    # accuracy from collapsing on WhatsApp video.
    consistency_weight: float = 0.6

    out_dir: Path = Path("/content/runs/v1")
    resume: str | None = None


@dataclass
class AudioConfig:
    # IR 3.4.1 FR1: multimodal analysis (voice + lip-sync).
    enabled: bool = True
    # Only the first N seconds are decoded; speech statistics converge
    # long before a full clip is read.
    max_seconds: float = 60.0


def _env(*names: str, default: str = "") -> str:
    """First of `names` that is set. Lets DFD_* (the repo's `env` file)
    and the older DF_* spellings coexist without duplicating defaults."""
    for name in names:
        value = os.environ.get(name)
        if value is not None and value.strip() != "":
            return value.strip()
    return default


def _env_bool(*names: str, default: bool = False) -> bool:
    raw = _env(*names)
    if raw == "":
        return default
    return raw.lower() in ("1", "true", "yes", "on")


@dataclass
class StoreConfig:
    # Zero-retention forensic record storage (ISO/IEC 27037 compliant).
    # The *video* is never stored either way; this only decides where the
    # verdict, the evidence hash and the account records live.
    enabled: bool = True
    local_path: Path = Path("./reports/reports.jsonl")

    # 'dynamodb' uses AWS (IR 2.4.4); anything else keeps the local JSON
    # files. Named DFD_DB_BACKEND to match the repo's `env` file.
    backend: str = _env("DFD_DB_BACKEND", "DF_DB_BACKEND",
                        default="local").strip().lower()
    region: str = _env("AWS_REGION", "AWS_DEFAULT_REGION",
                       default="ap-southeast-5")
    prefix: str = _env("DFD_TABLE_PREFIX", default="dfd_")
    billing_mode: str = _env("DFD_DYNAMODB_BILLING_MODE",
                             default="PAY_PER_REQUEST")

    @property
    def use_dynamodb(self) -> bool:
        return self.backend == "dynamodb"

    @property
    def local_dir(self) -> Path:
        return self.local_path.parent

    # Table names. These four already exist in the account; audit_log is
    # created by scripts/create_tables.py. Each is individually
    # overridable for anyone pointing at a different account.
    @property
    def users_table(self) -> str:
        return _env("DFD_TABLE_USERS", default=f"{self.prefix}users")

    @property
    def sessions_table(self) -> str:
        return _env("DFD_TABLE_SESSIONS", default=f"{self.prefix}sessions")

    @property
    def analyses_table(self) -> str:
        return _env("DFD_TABLE_ANALYSES", default=f"{self.prefix}analyses")

    @property
    def login_attempts_table(self) -> str:
        return _env("DFD_TABLE_LOGIN_ATTEMPTS",
                    default=f"{self.prefix}login_attempts")

    @property
    def audit_table(self) -> str:
        return _env("DFD_TABLE_AUDIT", default=f"{self.prefix}audit_log")


@dataclass
class AuthConfig:
    """Session auth backed by the DynamoDB users/sessions tables."""
    # PBKDF2-HMAC-SHA256. Stdlib, so no compiled dependency has to build
    # in the Railway image. OWASP's 2023 floor for this KDF is 600k.
    pbkdf2_rounds: int = int(_env("DFD_PBKDF2_ROUNDS", "DF_PBKDF2_ROUNDS",
                                  default="600000"))
    session_days: int = int(_env("DFD_SESSION_DAYS", "DF_SESSION_DAYS",
                                 default="7"))
    cookie_name: str = _env("DFD_COOKIE_NAME", "DF_COOKIE_NAME",
                            default="df_session")

    # A proxy that terminates TLS means cookies must be Secure in
    # production, and must not be over plain http on localhost. The `env`
    # file sets DFD_COOKIE_SECURE=0 for local work.
    cookie_secure: bool = _env_bool(
        "DFD_COOKIE_SECURE", "DF_COOKIE_SECURE",
        default=bool(os.environ.get("RAILWAY_ENVIRONMENT")))

    # Lock an account after this many consecutive failures.
    max_failed: int = int(_env("DFD_MAX_FAILED_LOGINS", "DF_MAX_FAILED_LOGINS",
                               default="8"))
    lockout_minutes: int = int(_env("DFD_LOCKOUT_MINUTES", "DF_LOCKOUT_MINUTES",
                                    default="15"))

    # Leave empty to let anyone register. Set to a comma-separated list of
    # domains to restrict sign-up, e.g. "apu.edu.my,staffemail.apu.edu.my".
    allowed_domains: str = _env("DFD_ALLOWED_EMAIL_DOMAINS",
                                "DF_ALLOWED_EMAIL_DOMAINS", default="")

    # When true an unauthenticated caller cannot reach /api/analyse.
    require_login: bool = _env_bool("DFD_REQUIRE_LOGIN", "DF_REQUIRE_LOGIN",
                                    default=True)

    # DFD_MAX_UPLOAD_MB in the env file; the server enforces it.
    max_upload_mb: int = int(_env("DFD_MAX_UPLOAD_MB", default="250"))


@dataclass
class InferConfig:
    # Balanced for fast real-time forensic scanning on CPU (10-15s)
    clips_per_video: int = 12

    # Score each clip twice - as-is and horizontally mirrored - and average.
    # Training already flips clips at random, so the model is flip-invariant
    # by construction and this is pure variance reduction. Costs 2x inference.
    tta: bool = True
    # Aggregation across clips: "topk" | "trimmed_mean" | "mean"
    aggregation: str = "topk"
    trim_fraction: float = 0.25
    # Decision threshold and temperature are written by evaluate.py.
    calibration_file: Path = _RUNS / "calibration.json"
    checkpoint: Path = _RUNS / "best.pt"
    min_face_conf: float = 0.9

    # IR 3.4.1 UI/UX: explainable visual cues. Each explained frame costs
    # one extra forward and backward pass, so this trades directly against
    # the "near real-time" non-functional requirement. Three is enough to
    # show a pattern without noticeably moving the 10-15 s budget; set 0
    # to turn the feature off.
    explain_frames: int = int(_env("DFD_EXPLAIN_FRAMES", default="3"))

    # IR 3.4.1 UI/UX: "a definite Real or Fake tag". The system defaults
    # to three states because refusing to guess is what keeps false
    # positives down. Set this True to force a binary call and report the
    # margin separately.
    strict_binary: bool = False


DATA = DataConfig()
MODEL = ModelConfig()
AUDIO = AudioConfig()
STORE = StoreConfig()
AUTH = AuthConfig()
TRAIN = TrainConfig()
INFER = InferConfig()
