"""Evaluate 200 Fake vs 200 Real Videos.

Displays the complete breakdown of False Positives, False Negatives,
Fake-to-Fake ratio (Detection Rate / Recall), and Real-to-Real ratio (Specificity).

Usage:
    # 1. Evaluate 200 fake vs 200 real from the manifest test split:
    python -m scripts.evaluate_200 --n 200

    # 2. Evaluate from custom folders of video files:
    python -m scripts.evaluate_200 --fake-dir path/to/fake_videos --real-dir path/to/real_videos --n 200

    # 3. Test under simulated WhatsApp / messenger compression:
    python -m scripts.evaluate_200 --n 200 --degraded
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import DATA, INFER, MODEL        # noqa: E402
from data.dataset import load_manifest        # noqa: E402
from evaluate import collect                  # noqa: E402
from models.net import build_model            # noqa: E402


def sample_balanced_manifest(videos: dict, n: int = 200, seed: int = 1337):
    """Pick exactly n fake and n real videos from the loaded manifest."""
    rng = random.Random(seed)
    fake_keys = sorted(k for k, v in videos.items() if v["label"] == 1)
    real_keys = sorted(k for k, v in videos.items() if v["label"] == 0)
    rng.shuffle(fake_keys)
    rng.shuffle(real_keys)

    take_f = min(n, len(fake_keys))
    take_r = min(n, len(real_keys))

    selected = fake_keys[:take_f] + real_keys[:take_r]
    return {k: videos[k] for k in selected}, take_f, take_r


def load_from_directories(fake_dir: str, real_dir: str, n: int = 200, seed: int = 1337):
    """Create a video dict from two raw video directories."""
    exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    f_path, r_path = Path(fake_dir), Path(real_dir)

    fake_files = [p for p in f_path.glob("*") if p.suffix.lower() in exts]
    real_files = [p for p in r_path.glob("*") if p.suffix.lower() in exts]

    rng = random.Random(seed)
    rng.shuffle(fake_files)
    rng.shuffle(real_files)

    fake_files = fake_files[:n]
    real_files = real_files[:n]

    videos = {}
    for p in fake_files:
        videos[f"custom_fake/{p.stem}"] = {
            "source": "custom", "identity": p.stem, "label": 1,
            "method": "custom_fake", "frames": [str(p)]
        }
    for p in real_files:
        videos[f"custom_real/{p.stem}"] = {
            "source": "custom", "identity": p.stem, "label": 0,
            "method": "real", "frames": [str(p)]
        }
    return videos, len(fake_files), len(real_files)


def print_detection_matrix(y: np.ndarray, p: np.ndarray, thr: float,
                           target_n: int, condition: str = "clean"):
    """Format and print the 200 Fake vs 200 Real detection matrix."""
    pred = (p >= thr).astype(int)

    tp = int(((pred == 1) & (y == 1)).sum())   # Fake detected as Fake
    fn = int(((pred == 0) & (y == 1)).sum())   # Fake missed as Real (FN)
    tn = int(((pred == 0) & (y == 0)).sum())   # Real cleared as Real
    fp = int(((pred == 1) & (y == 0)).sum())   # Real flagged as Fake (FP)

    n_fake = int((y == 1).sum())
    n_real = int((y == 0).sum())

    # Ratios
    fake_to_fake_ratio = tp / max(1, n_fake)   # Recall / True Positive Rate
    fake_to_real_ratio = fn / max(1, n_fake)   # Miss Rate / False Negative Rate
    real_to_real_ratio = tn / max(1, n_real)   # Specificity / True Negative Rate
    real_to_fake_ratio = fp / max(1, n_real)   # False Alarm Rate / False Positive Rate

    precision = tp / max(1, tp + fp)
    f1 = 2 * precision * fake_to_fake_ratio / max(1e-9, precision + fake_to_fake_ratio)
    balanced_acc = 0.5 * (fake_to_fake_ratio + real_to_real_ratio)
    overall_acc = (tp + tn) / max(1, len(y))

    def pbar(val, width=28):
        filled = int(round(val * width))
        return "#" * filled + "." * (width - filled)

    sep = "=" * 76
    sub_sep = "-" * 76

    print("\n" + sep)
    print(f"             200 FAKE vs 200 REAL VIDEOS DETECTION MATRIX")
    print(f"               Condition: {condition.upper()} | Threshold: {thr:.3f}")
    print(sep)

    print("\n[1] FAKE VIDEOS EVALUATION (Ground Truth = FAKE):")
    print(f"    Total Fake Videos Tested : {n_fake:4d}")
    print(f"    -> Detected as FAKE (TP) : {tp:4d} / {n_fake}  [{pbar(fake_to_fake_ratio)}]  {fake_to_fake_ratio:6.2%}  (Ratio Fake->Fake: {fake_to_fake_ratio:.4f})")
    print(f"    -> Missed as REAL   (FN) : {fn:4d} / {n_fake}  [{pbar(fake_to_real_ratio)}]  {fake_to_real_ratio:6.2%}  (Ratio Fake->Real: {fake_to_real_ratio:.4f}) [FALSE NEGATIVE]")

    print("\n[2] REAL VIDEOS EVALUATION (Ground Truth = REAL):")
    print(f"    Total Real Videos Tested : {n_real:4d}")
    print(f"    -> Cleared as REAL  (TN) : {tn:4d} / {n_real}  [{pbar(real_to_real_ratio)}]  {real_to_real_ratio:6.2%}  (Ratio Real->Real: {real_to_real_ratio:.4f})")
    print(f"    -> Flagged as FAKE  (FP) : {fp:4d} / {n_real}  [{pbar(real_to_fake_ratio)}]  {real_to_fake_ratio:6.2%}  (Ratio Real->Fake: {real_to_fake_ratio:.4f}) [FALSE POSITIVE]")

    print("\n" + sub_sep)
    print("CONFUSION MATRIX TABLE:")
    print(sub_sep)
    print(f"  {'':20s} | {'PREDICTED FAKE':^22s} | {'PREDICTED REAL':^22s} | {'TOTAL':^8s}")
    print(f"  {'-'*20}-+-{'-'*22}-+-{'-'*22}-+-{'-'*8}")
    print(f"  {'ACTUAL FAKE':20s} | {f'{tp} (True Positive)':^22s} | {f'{fn} (False Negative)':^22s} | {n_fake:^8d}")
    print(f"  {'ACTUAL REAL':20s} | {f'{fp} (False Positive)':^22s} | {f'{tn} (True Negative)':^22s} | {n_real:^8d}")
    print(f"  {'-'*20}-+-{'-'*22}-+-{'-'*22}-+-{'-'*8}")
    print(f"  {'TOTAL PREDICTED':20s} | {tp+fp:^22d} | {fn+tn:^22d} | {len(y):^8d}")

    print("\n" + sub_sep)
    print("PERFORMANCE SUMMARY & RATIOS:")
    print(sub_sep)
    print(f"  * Fake-to-Fake Detection Rate (Recall / TPR) : {fake_to_fake_ratio:7.2%}  ({tp}/{n_fake})")
    print(f"  * Fake-to-Real Miss Rate (FN Rate / FNR)      : {fake_to_real_ratio:7.2%}  ({fn}/{n_fake})")
    print(f"  * Real-to-Real Clearance Rate (Specificity)   : {real_to_real_ratio:7.2%}  ({tn}/{n_real})")
    print(f"  * Real-to-Fake False Alarm Rate (FP Rate/FPR) : {real_to_fake_ratio:7.2%}  ({fp}/{n_real})")
    print(f"  * Precision (When model flags Fake, is it?)  : {precision:7.2%}")
    print(f"  * F1 Score                                    : {f1:7.4f}")
    print(f"  * Balanced Accuracy                           : {balanced_acc:7.2%}")
    print(f"  * Overall Accuracy                            : {overall_acc:7.2%}")
    print(sep + "\n")

    return {
        "n_fake": n_fake, "n_real": n_real,
        "true_positives": tp, "false_negatives": fn,
        "true_negatives": tn, "false_positives": fp,
        "fake_to_fake_ratio": fake_to_fake_ratio,
        "fake_to_real_ratio": fake_to_real_ratio,
        "real_to_real_ratio": real_to_real_ratio,
        "real_to_fake_ratio": real_to_fake_ratio,
        "precision": precision, "f1": f1,
        "balanced_accuracy": balanced_acc,
        "accuracy": overall_acc,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate False Positives & False Negatives on 200 Fake vs 200 Real Videos")
    parser.add_argument("--checkpoint", default=str(INFER.checkpoint), help="Path to best.pt checkpoint")
    parser.add_argument("--manifest", default=str(DATA.manifest), help="Path to manifest.csv")
    parser.add_argument("--split", default="test", help="Split to sample from (test, val, etc.)")
    parser.add_argument("--n", type=int, default=200, help="Number of videos per class (default: 200)")
    parser.add_argument("--threshold", type=float, default=None, help="Decision threshold override (default: from calibration.json)")
    parser.add_argument("--fake-dir", default=None, help="Optional: directory containing raw fake video files")
    parser.add_argument("--real-dir", default=None, help="Optional: directory containing raw real video files")
    parser.add_argument("--degraded", action="store_true", help="Simulate WhatsApp/messenger transcode degradation")
    parser.add_argument("--clips", type=int, default=INFER.clips_per_video, help="Clips per video (default from config)")
    parser.add_argument("--out-json", default=None, help="Save evaluation matrix results to a JSON file")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        sys.exit(f"Error: Checkpoint not found at {ckpt_path}. Please train a model or place best.pt in runs/v1/.")

    # Load Model
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    MODEL.backbone = cfg.get("backbone", MODEL.backbone)
    MODEL.use_srm = cfg.get("use_srm", MODEL.use_srm)
    MODEL.temporal = cfg.get("temporal", MODEL.temporal)
    MODEL.pretrained = False

    model = build_model(MODEL).to(device).eval()
    model.load_state_dict(ckpt["model"])
    print(f"Loaded {MODEL.backbone} from {ckpt_path} on {device.upper()}")

    # Determine Threshold
    thr = args.threshold
    if thr is None:
        cal_path = ckpt_path.parent / "calibration.json"
        if cal_path.exists():
            cal_data = json.loads(cal_path.read_text())
            thr = float(cal_data.get("threshold", 0.5))
            crit = cal_data.get("criterion", "calibrated")
            print(f"Using calibrated decision threshold: {thr:.3f} (criterion: {crit})")
        else:
            thr = 0.50
            print("No calibration.json found - using default threshold: 0.500")

    # Load Videos
    if args.fake_dir and args.real_dir:
        print(f"Loading custom video directories: {args.fake_dir} and {args.real_dir}")
        subset, n_f, n_r = load_from_directories(args.fake_dir, args.real_dir, n=args.n)
    else:
        manifest_path = Path(args.manifest)
        if not manifest_path.exists():
            sys.exit(f"Error: Manifest not found at {manifest_path}. Please provide --manifest or --fake-dir/--real-dir.")
        videos = load_manifest(str(manifest_path), splits=(args.split,))
        if not videos:
            sys.exit(f"No videos found in manifest split '{args.split}'.")
        subset, n_f, n_r = sample_balanced_manifest(videos, n=args.n)

    condition = "degraded" if args.degraded else "clean"
    print(f"\nScoring {len(subset)} videos ({n_f} Fake, {n_r} Real) with {args.clips} clips each [{condition}]...")

    rows = collect(model, subset, device,
                   degrade_prob=1.0 if args.degraded else 0.0,
                   clips_per_video=args.clips,
                   tag=condition)

    y = np.array([r["label"] for r in rows])
    p = np.array([r["prob"] for r in rows])

    results = print_detection_matrix(y, p, thr=thr, target_n=args.n, condition=condition)

    if args.out_json:
        out_p = Path(args.out_json)
        out_p.write_text(json.dumps(results, indent=2))
        print(f"Saved detection report to {out_p}")


if __name__ == "__main__":
    main()
