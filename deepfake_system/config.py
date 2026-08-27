"""Central configuration for the deepfake detection system."""
from dataclasses import dataclass, field
from pathlib import Path


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

    # Frames per clip fed to the temporal head.
    clip_len: int = 16
    # Stride between sampled frames inside a video (in frames).
    frame_stride: int = 3
    # Face crop size.
    img_size: int = 300

    # Splitting is done by video/identity, never by frame.
    val_ratio: float = 0.12
    test_ratio: float = 0.13
    seed: int = 1337

    # Datasets held out entirely for the "unseen source" report.
    holdout_sources: tuple = ("wild",)
    # DF40 generator families held out to measure unseen-generator behaviour.
    holdout_methods: tuple = ("simswap", "e4s", "sd15", "sdxl", "danet")


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
    epochs: int = 22
    batch_size: int = 8            # clips per batch (T4). Use 16 on A100.
    accum_steps: int = 2
    lr: float = 2.5e-4
    backbone_lr_mult: float = 0.25
    weight_decay: float = 0.02
    warmup_epochs: int = 2
    label_smoothing: float = 0.05
    amp: bool = True
    ema_decay: float = 0.999
    num_workers: int = 4

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
    calibration_file: Path = Path("/content/runs/v1/calibration.json")
    checkpoint: Path = Path("/content/runs/v1/best.pt")
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
