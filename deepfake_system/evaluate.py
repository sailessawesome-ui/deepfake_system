"""Video-level evaluation, threshold calibration, and an honest breakdown.

Reports separately for:
  - each source dataset (FF++, Celeb-DF, DF40, wild)
  - clean vs light vs typical vs harsh messenger degradation
  - held-out generator families the model never saw in training

A single headline number hides exactly the failure your examiner will
find. Report the table.

    python evaluate.py --checkpoint runs/v1/best.pt --split test
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import DATA, INFER
from data.dataset import ClipDataset, load_manifest
from models.net import build_model


def aggregate(clip_probs: np.ndarray, how: str = None,
              trim: float = None) -> float:
    how = how or INFER.aggregation
    trim = INFER.trim_fraction if trim is None else trim
    p = np.sort(clip_probs)
    if how == "mean" or len(p) < 4:
        return float(p.mean())
    if how == "topk":
        k = max(1, int(len(p) * 0.3))
        return float(p[-k:].mean())
    cut = int(len(p) * trim)
    return float(p[cut:len(p) - cut].mean()) if len(p) - 2 * cut > 0 else float(p.mean())


def metrics(y, p, thr):
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    # Specificity and FPR make the false-positive cost visible. Precision
    # hides it whenever a split is mostly fake, and every split here is.
    spec = tn / max(1, tn + fp)
    out = {"n": len(y), "acc": float((pred == y).mean()), "f1": float(f1),
           "precision": float(prec), "recall": float(rec),
           "specificity": float(spec), "fpr": float(1.0 - spec),
           "balanced_acc": float(0.5 * (rec + spec)),
           "tp": tp, "tn": tn, "fp": fp, "fn": fn}
    if len(set(y.tolist())) > 1:
        from sklearn.metrics import roc_auc_score
        out["auc"] = float(roc_auc_score(y, p))
    return out


def best_threshold(y, p, criterion="youden"):
    """Pick a decision threshold on validation.

    "f1"     maximises F1. On a split that is ~85% fake this rewards
             catching fakes and barely penalises false alarms, so it drives
             the threshold down and the false-positive rate up.
    "youden" maximises TPR - FPR, weighting a false alarm the same as a
             miss. This is what a tool claiming "high detection rate AND
             low false positives" should use.
    "target_fpr" is the strictest: the lowest threshold whose validation
             FPR stays within TARGET_FPR.
    """
    grid = np.linspace(0.05, 0.95, 181)
    rows = [(t, metrics(y, p, t)) for t in grid]

    if criterion == "f1":
        scores = [m["f1"] for _, m in rows]
    elif criterion == "target_fpr":
        ok = [(t, m) for t, m in rows if m["fpr"] <= TARGET_FPR]
        if ok:
            # Best recall inside the false-alarm budget; when recall ties,
            # take the threshold that raises fewer false alarms rather than
            # spending budget that buys nothing.
            return float(max(ok, key=lambda tm: (tm[1]["recall"],
                                                 -tm[1]["fpr"]))[0])
        scores = [-m["fpr"] for _, m in rows]
    else:                                    # youden
        scores = [m["recall"] - m["fpr"] for _, m in rows]

    return float(grid[int(np.argmax(scores))])


TARGET_FPR = 0.05


def threshold_report(y, p):
    """What each criterion would cost, so the choice is visible."""
    lines = []
    for name in ("f1", "youden", "target_fpr"):
        t = best_threshold(y, p, name)
        m = metrics(y, p, t)
        lines.append(f"  {name:11s} thr {t:.3f}  f1 {m['f1']:.4f}  "
                     f"recall {m['recall']:.4f}  fpr {m['fpr']:.4f}  "
                     f"bal_acc {m['balanced_acc']:.4f}")
    return "\n".join(lines)


def fit_temperature(y, logits):
    """One-parameter calibration so the reported confidence means something."""
    t = torch.ones(1, requires_grad=True)
    yt = torch.tensor(y, dtype=torch.float32)
    lt = torch.tensor(logits, dtype=torch.float32)
    opt = torch.optim.LBFGS([t], lr=0.05, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            lt / t.clamp(min=1e-2), yt)
        loss.backward()
        return loss
    opt.step(closure)
    return float(t.detach().clamp(min=1e-2).item())


@torch.no_grad()
def collect(model, videos, device, degrade_prob, clips_per_video, tag):
    ds = ClipDataset(videos, train=False, clips_per_video=clips_per_video,
                     paired=False, degrade_prob=degrade_prob)
    ld = DataLoader(ds, batch_size=8, shuffle=False, num_workers=4)
    per_video = defaultdict(lambda: {"probs": [], "logits": []})
    meta = {}
    for batch in ld:
        x = batch["clean"].to(device)
        with torch.autocast("cuda", enabled=torch.cuda.is_available()):
            logit, _ = model(x)
        logit = logit.float().cpu().numpy()
        prob = 1 / (1 + np.exp(-logit))
        for j, vid in enumerate(batch["video_id"]):
            per_video[vid]["probs"].append(float(prob[j]))
            per_video[vid]["logits"].append(float(logit[j]))
            meta[vid] = {"label": int(batch["label"][j].item()),
                         "source": batch["source"][j],
                         "method": batch["method"][j], "condition": tag}
    rows = []
    for vid, d in per_video.items():
        rows.append({"video_id": vid, "prob": aggregate(np.array(d["probs"])),
                     "logit": float(np.mean(d["logits"])), **meta[vid]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(INFER.checkpoint))
    ap.add_argument("--manifest", default=str(DATA.manifest))
    ap.add_argument("--split", default="test")
    ap.add_argument("--clips", type=int, default=INFER.clips_per_video)
    ap.add_argument("--out", default=None)
    ap.add_argument("--threshold-criterion", default="youden",
                    choices=["youden", "f1", "target_fpr"],
                    help="how the decision threshold is chosen on val. "
                         "youden balances misses against false alarms; f1 "
                         "is the old behaviour and inflates the FPR.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt.get("config", {})
    from config import MODEL
    MODEL.backbone = cfg.get("backbone", MODEL.backbone)
    MODEL.use_srm = cfg.get("use_srm", MODEL.use_srm)
    MODEL.temporal = cfg.get("temporal", MODEL.temporal)
    MODEL.pretrained = False
    model = build_model(MODEL).to(device).eval()
    model.load_state_dict(ckpt["model"])

    # Calibrate on val, never on test.
    val = load_manifest(args.manifest, splits=("val",))
    val_rows = collect(model, val, device, 0.0, max(4, args.clips // 2), "clean")
    y_val = np.array([r["label"] for r in val_rows])
    p_val = np.array([r["prob"] for r in val_rows])
    l_val = np.array([r["logit"] for r in val_rows])
    temp = fit_temperature(y_val, l_val)
    thr = best_threshold(y_val, p_val, args.threshold_criterion)
    print("threshold criteria on validation:")
    print(threshold_report(y_val, p_val))
    print(f"\ncalibration: temperature {temp:.3f}, threshold {thr:.3f} "
          f"(criterion: {args.threshold_criterion})")

    videos = load_manifest(args.manifest, splits=(args.split,))
    rows = []
    for tag, prob in (("clean", 0.0), ("degraded", 1.0)):
        rows += collect(model, videos, device, prob, args.clips, tag)

    y = np.array([r["label"] for r in rows])
    p = np.array([r["prob"] for r in rows])
    report = {"overall": metrics(y, p, thr),
              "threshold": thr, "temperature": temp,
              "threshold_criterion": args.threshold_criterion,
              "by_condition": {}, "by_source": {}, "by_method": {}}

    for key, field in (("by_condition", "condition"), ("by_source", "source")):
        for value in sorted({r[field] for r in rows}):
            sel = [r for r in rows if r[field] == value]
            report[key][value] = metrics(
                np.array([r["label"] for r in sel]),
                np.array([r["prob"] for r in sel]), thr)

    for value in sorted({r["method"] for r in rows}):
        sel = [r for r in rows if r["method"] == value]
        if len(sel) >= 8:
            report["by_method"][value] = metrics(
                np.array([r["label"] for r in sel]),
                np.array([r["prob"] for r in sel]), thr)

    print("\n=== overall ===")
    print(json.dumps(report["overall"], indent=2))
    print("\n=== by condition ===")
    for k, v in report["by_condition"].items():
        print(f"  {k:12s} acc {v['acc']:.4f}  f1 {v['f1']:.4f}  n={v['n']}")
    print("\n=== by source ===")
    for k, v in report["by_source"].items():
        print(f"  {k:12s} acc {v['acc']:.4f}  f1 {v['f1']:.4f}  n={v['n']}")
    print("\n=== by generator ===")
    for k, v in sorted(report["by_method"].items(), key=lambda kv: kv[1]["f1"]):
        print(f"  {k:20s} f1 {v['f1']:.4f}  recall {v['recall']:.4f}  n={v['n']}")

    out = Path(args.out or Path(args.checkpoint).parent / f"report_{args.split}.json")
    out.write_text(json.dumps(report, indent=2))
    cal = Path(args.checkpoint).parent / "calibration.json"
    cal.write_text(json.dumps({"temperature": temp, "threshold": thr,
                               "criterion": args.threshold_criterion},
                              indent=2))
    print(f"\nwrote {out} and {cal}")


if __name__ == "__main__":
    main()
