"""Per-video detection report: how many fakes are caught, how many reals cleared.

    python -m scripts.detection_report --checkpoint runs/v1/best.pt --n 200

Samples an equal number of fake and real videos from a split, scores each at
video level exactly the way evaluate.py does, and prints the confusion matrix
plus a threshold sweep.

A balanced sample is deliberate. The test split is ~76% fake, so accuracy on
it flatters any model that leans toward "fake". Fixing the counts at n vs n
makes the two error types directly comparable.
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

from config import DATA, INFER          # noqa: E402
from data.dataset import load_manifest  # noqa: E402
from evaluate import collect, metrics    # noqa: E402
from models.net import build_model      # noqa: E402


def sample_balanced(videos: dict, n: int, seed: int = 1337):
    """n fake + n real videos, or as many as the split can supply."""
    rng = random.Random(seed)
    fake = sorted(k for k, v in videos.items() if v["label"] == 1)
    real = sorted(k for k, v in videos.items() if v["label"] == 0)
    rng.shuffle(fake)
    rng.shuffle(real)
    take_f, take_r = min(n, len(fake)), min(n, len(real))
    if take_f < n or take_r < n:
        print(f"  note: split holds {len(fake):,} fake / {len(real):,} real; "
              f"using {take_f} / {take_r}")
    chosen = fake[:take_f] + real[:take_r]
    return {k: videos[k] for k in chosen}


def confusion(y, p, thr):
    pred = (p >= thr).astype(int)
    return {
        "tp": int(((pred == 1) & (y == 1)).sum()),   # fake called fake
        "fn": int(((pred == 0) & (y == 1)).sum()),   # fake called real  <- miss
        "tn": int(((pred == 0) & (y == 0)).sum()),   # real called real
        "fp": int(((pred == 1) & (y == 0)).sum()),   # real called fake  <- false alarm
    }


def bar(frac, width=32):
    filled = int(round(frac * width))
    return "#" * filled + "." * (width - filled)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(INFER.checkpoint))
    ap.add_argument("--manifest", default=str(DATA.manifest))
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=200,
                    help="videos per class (default 200)")
    ap.add_argument("--clips", type=int, default=INFER.clips_per_video)
    ap.add_argument("--threshold", type=float, default=None,
                    help="override; defaults to calibration.json beside the "
                         "checkpoint, else 0.5")
    ap.add_argument("--degraded", action="store_true",
                    help="score messenger-degraded copies instead of clean")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    from config import MODEL
    MODEL.backbone = cfg.get("backbone", MODEL.backbone)
    MODEL.use_srm = cfg.get("use_srm", MODEL.use_srm)
    MODEL.temporal = cfg.get("temporal", MODEL.temporal)
    MODEL.pretrained = False
    model = build_model(MODEL).to(device).eval()
    model.load_state_dict(ckpt["model"])

    thr = args.threshold
    if thr is None:
        cal = Path(args.checkpoint).parent / "calibration.json"
        if cal.exists():
            data = json.loads(cal.read_text())
            thr = float(data["threshold"])
            crit = data.get("criterion", "unknown")
            print(f"threshold {thr:.3f} from calibration.json (criterion: {crit})")
        else:
            thr = 0.5
            print("no calibration.json found - using 0.500")
    else:
        print(f"threshold {thr:.3f} (given on the command line)")

    videos = load_manifest(args.manifest, splits=(args.split,))
    if not videos:
        sys.exit(f"no videos in split '{args.split}'")
    subset = sample_balanced(videos, args.n)

    condition = "degraded" if args.degraded else "clean"
    print(f"scoring {len(subset)} videos from '{args.split}' ({condition}, "
          f"{args.clips} clips each)...", flush=True)
    rows = collect(model, subset, device,
                   1.0 if args.degraded else 0.0, args.clips, condition)

    y = np.array([r["label"] for r in rows])
    p = np.array([r["prob"] for r in rows])
    n_fake, n_real = int((y == 1).sum()), int((y == 0).sum())
    c = confusion(y, p, thr)

    W = 66
    print()
    print("=" * W)
    print(f"DETECTION REPORT   {Path(args.checkpoint).parent.name}   "
          f"split={args.split}   {condition}")
    print("=" * W)
    print()
    print(f"  FAKE videos tested : {n_fake}")
    print(f"    detected as FAKE : {c['tp']:4d}   {bar(c['tp']/max(1,n_fake))}  "
          f"{c['tp']/max(1,n_fake):6.1%}   <- detection rate")
    print(f"    missed as REAL   : {c['fn']:4d}   {bar(c['fn']/max(1,n_fake))}  "
          f"{c['fn']/max(1,n_fake):6.1%}   <- FALSE NEGATIVES")
    print()
    print(f"  REAL videos tested : {n_real}")
    print(f"    cleared as REAL  : {c['tn']:4d}   {bar(c['tn']/max(1,n_real))}  "
          f"{c['tn']/max(1,n_real):6.1%}   <- specificity")
    print(f"    flagged as FAKE  : {c['fp']:4d}   {bar(c['fp']/max(1,n_real))}  "
          f"{c['fp']/max(1,n_real):6.1%}   <- FALSE POSITIVES")
    print()

    print("-" * W)
    print("  CONFUSION MATRIX")
    print("-" * W)
    print(f"{'':18s}{'said FAKE':>12s}{'said REAL':>12s}")
    print(f"{'actually FAKE':18s}{c['tp']:>12d}{c['fn']:>12d}")
    print(f"{'actually REAL':18s}{c['fp']:>12d}{c['tn']:>12d}")
    print()

    m = metrics(y, p, thr)
    print("-" * W)
    print("  RATES")
    print("-" * W)
    for key, label in (("recall", "recall / detection rate  (fake caught)"),
                       ("specificity", "specificity             (real cleared)"),
                       ("precision", "precision               (flags that were right)"),
                       ("fpr", "false positive rate     (real wrongly flagged)"),
                       ("f1", "F1"),
                       ("balanced_acc", "balanced accuracy"),
                       ("acc", "accuracy")):
        if key in m:
            print(f"    {label:42s} {m[key]:.4f}")
    if "auc" in m:
        print(f"    {'AUC (threshold-independent)':42s} {m['auc']:.4f}")

    print()
    print("-" * W)
    print("  THRESHOLD SWEEP")
    print("-" * W)
    print(f"    {'thr':>5s} {'fake caught':>13s} {'real cleared':>14s} "
          f"{'FP':>5s} {'FN':>5s} {'bal_acc':>8s}")
    best_t, best_b = thr, -1.0
    for t in np.arange(0.05, 0.96, 0.05):
        cc = confusion(y, p, t)
        rec = cc["tp"] / max(1, n_fake)
        spec = cc["tn"] / max(1, n_real)
        b = 0.5 * (rec + spec)
        if b > best_b:
            best_t, best_b = float(t), b
        mark = "  <- current" if abs(t - thr) < 0.025 else ""
        print(f"    {t:5.2f} {cc['tp']:5d}/{n_fake:<7d} {cc['tn']:6d}/{n_real:<7d} "
              f"{cc['fp']:5d} {cc['fn']:5d} {b:8.4f}{mark}")

    print()
    print(f"    best balanced accuracy {best_b:.4f} at threshold {best_t:.2f}")
    if abs(best_t - thr) > 0.05:
        cc = confusion(y, p, best_t)
        print(f"    moving there: {cc['tp']}/{n_fake} fakes caught, "
              f"{cc['tn']}/{n_real} reals cleared "
              f"({cc['fp']} false alarms, was {c['fp']})")

    out = Path(args.checkpoint).parent / f"detection_{args.split}_{condition}.json"
    out.write_text(json.dumps({
        "split": args.split, "condition": condition, "threshold": thr,
        "n_fake": n_fake, "n_real": n_real, "confusion": c, "rates": m,
        "best_threshold_balanced": best_t,
    }, indent=2))
    print()
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
