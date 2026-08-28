"""Central configuration for the deepfake detection system."""
import os
from dataclasses import dataclass, field
from pathlib import Path

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

    # Datasets held out entirely for the "unseen source" report.
    holdout_sources: tuple = ("wild",)
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


@dataclass
class StoreConfig:
    # IR 2.4.4: AWS DynamoDB as the data store. Falls back to a local
    # JSON-lines file when boto3 or credentials are unavailable.
    enabled: bool = True
    use_dynamodb: bool = True
    table_name: str = "deepfake_reports"
    region: str = "ap-southeast-1"
    local_path: Path = Path("./reports/reports.jsonl")


@dataclass
class InferConfig:
    # Clips sampled per video at inference.
    clips_per_video: int = 12
    # Aggregation across clips: "trimmed_mean" | "mean" | "topk"
    aggregation: str = "trimmed_mean"
    trim_fraction: float = 0.2
    # Decision threshold and temperature are written by evaluate.py.
    calibration_file: Path = _RUNS / "calibration.json"
    checkpoint: Path = _RUNS / "best.pt"
    min_face_conf: float = 0.9

    # IR 3.4.1 UI/UX: "a definite Real or Fake tag". The system defaults
    # to three states because refusing to guess is what keeps false
    # positives down. Set this True to force a binary call and report the
    # margin separately.
    strict_binary: bool = False


DATA = DataConfig()
MODEL = ModelConfig()
AUDIO = AudioConfig()
STORE = StoreConfig()
TRAIN = TrainConfig()
INFER = InferConfig()
