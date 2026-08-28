"""Training entry point.

    python train.py --epochs 22 --batch-size 8

The loss has three parts:

  BCE on the clip logit                    — the actual task
  BCE on per-frame logits (weight 0.3)     — denser signal, faster convergence
  agreement between the clean view and a   — this is what makes the model
  messenger-degraded view of the SAME clip   survive WhatsApp transcoding

The third term is the one worth explaining in your viva. Without it the
model learns compression artifacts as a shortcut; with it, the clean and
degraded views are pushed to the same decision, so the features that
survive are the ones that describe the face rather than the codec.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import DATA, MODEL, TRAIN
from data.dataset import ClipDataset, load_manifest, make_sampler
from models.net import build_model, param_groups


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s, p in zip(self.shadow.state_dict().values(),
                        model.state_dict().values()):
            if s.dtype.is_floating_point:
                s.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
            else:
                s.copy_(p)


def lr_lambda(epoch, warmup, total):
    if epoch < warmup:
        return (epoch + 1) / max(1, warmup)
    prog = (epoch - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * prog))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    probs, labels = [], []
    for batch in loader:
        x = batch["clean"].to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=TRAIN.amp):
            logit, _ = model(x)
        probs.append(torch.sigmoid(logit.float()).cpu())
        labels.append(batch["label"])
    p = torch.cat(probs).numpy()
    y = torch.cat(labels).numpy()
    pred = (p >= 0.5).astype(int)
    tp = ((pred == 1) & (y == 1)).sum()
    fp = ((pred == 1) & (y == 0)).sum()
    fn = ((pred == 0) & (y == 1)).sum()
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y, p)
    except Exception:
        auc = float("nan")
    return {"acc": float((pred == y).mean()), "f1": float(f1),
            "auc": float(auc), "n": int(len(y))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(DATA.manifest))
    ap.add_argument("--epochs", type=int, default=TRAIN.epochs)
    ap.add_argument("--batch-size", type=int, default=TRAIN.batch_size)
    ap.add_argument("--lr", type=float, default=TRAIN.lr)
    ap.add_argument("--out", default=str(TRAIN.out_dir))
    ap.add_argument("--backbone", default=MODEL.backbone)
    ap.add_argument("--resume", default=TRAIN.resume)
    ap.add_argument("--no-srm", action="store_true")
    ap.add_argument("--patience", type=int, default=5,
                    help="stop if val F1 has not improved for this many "
                         "epochs. 0 disables it.")
    ap.add_argument("--consistency-weight", type=float,
                    default=TRAIN.consistency_weight,
                    help="weight on the clean/degraded agreement loss. "
                         "Set 0 to ablate it.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    MODEL.backbone = args.backbone
    if args.no_srm:
        MODEL.use_srm = False
    TRAIN.consistency_weight = args.consistency_weight
    print(f"consistency weight {TRAIN.consistency_weight} | "
          f"srm {MODEL.use_srm} | backbone {MODEL.backbone}")

    train_videos = load_manifest(args.manifest, splits=("train",))
    val_videos = load_manifest(args.manifest, splits=("val",))
    print(f"train videos {len(train_videos):,} | val videos {len(val_videos):,}")

    train_ds = ClipDataset(train_videos, train=True, clips_per_video=2,
                           paired=True)
    val_ds = ClipDataset(val_videos, train=False, clips_per_video=1,
                         paired=False, degrade_prob=0.0)

    train_ld = DataLoader(train_ds, batch_size=args.batch_size,
                          sampler=make_sampler(train_videos),
                          num_workers=TRAIN.num_workers, pin_memory=True,
                          drop_last=True, persistent_workers=True)
    val_ld = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=TRAIN.num_workers, pin_memory=True)

    model = build_model(MODEL).to(device)

    opt = torch.optim.AdamW(
        param_groups(model, args.lr, TRAIN.backbone_lr_mult, TRAIN.weight_decay))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda e: lr_lambda(e, TRAIN.warmup_epochs, args.epochs))
    scaler = torch.amp.GradScaler("cuda", enabled=TRAIN.amp)
    ema = EMA(model, TRAIN.ema_decay)

    # Resuming restores the optimizer, the LR schedule position and the EMA
    # as well as the weights. Loading weights alone restarts the cosine
    # schedule from warmup and throws away Adam's moments, which costs more
    # than the epochs it appears to save. Older checkpoints that only carry
    # "model" still load — the rest is skipped.
    start_epoch, best_f1, history = 0, 0.0, []
    best_epoch, stale = -1, 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        for key, obj in (("ema", ema.shadow), ("opt", opt),
                         ("sched", sched), ("scaler", scaler)):
            if key in ckpt:
                obj.load_state_dict(ckpt[key])
        start_epoch = ckpt.get("epoch", -1) + 1
        best_f1 = ckpt.get("best_f1", 0.0)
        best_epoch = ckpt.get("best_epoch", -1)
        history = ckpt.get("history", [])
        missing = [k for k in ("ema", "opt", "sched", "scaler") if k not in ckpt]
        print(f"resumed from {args.resume} at epoch {start_epoch}"
              f" (best f1 so far {best_f1:.4f})")
        if missing:
            print(f"  note: checkpoint had no {', '.join(missing)} — "
                  f"those restart from scratch")
        if start_epoch >= args.epochs:
            print(f"  nothing to do: already finished {start_epoch} epochs. "
                  f"Raise --epochs to keep training.")
            return

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0, running = time.time(), []
        opt.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_ld):
            clean = batch["clean"].to(device, non_blocking=True)
            degraded = batch["degraded"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            y_smooth = y * (1 - TRAIN.label_smoothing) + \
                TRAIN.label_smoothing * 0.5

            with torch.autocast("cuda", enabled=TRAIN.amp):
                logit_c, frame_c = model(clean)
                logit_d, frame_d = model(degraded)

                loss_clip = 0.5 * (
                    F.binary_cross_entropy_with_logits(logit_c, y_smooth) +
                    F.binary_cross_entropy_with_logits(logit_d, y_smooth))

                y_frame = y_smooth.unsqueeze(1).expand_as(frame_c)
                loss_frame = 0.5 * (
                    F.binary_cross_entropy_with_logits(frame_c, y_frame) +
                    F.binary_cross_entropy_with_logits(frame_d, y_frame))

                # Agreement between the two views of the same clip.
                p_c = torch.sigmoid(logit_c.detach())
                loss_cons = F.binary_cross_entropy_with_logits(logit_d, p_c) + \
                    F.mse_loss(torch.sigmoid(logit_d), p_c)

                loss = (loss_clip + 0.3 * loss_frame +
                        TRAIN.consistency_weight * loss_cons)
                loss = loss / TRAIN.accum_steps

            scaler.scale(loss).backward()
            if (step + 1) % TRAIN.accum_steps == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                ema.update(model)

            running.append(loss.item() * TRAIN.accum_steps)
            if step % 50 == 0:
                print(f"  e{epoch} s{step}/{len(train_ld)} "
                      f"loss {np.mean(running[-50:]):.4f}", flush=True)

        sched.step()
        m_raw = evaluate(model, val_ld, device)
        m_ema = evaluate(ema.shadow, val_ld, device)
        best_of = m_ema if m_ema["f1"] >= m_raw["f1"] else m_raw
        tag = "ema" if m_ema["f1"] >= m_raw["f1"] else "raw"
        print(f"epoch {epoch}  loss {np.mean(running):.4f}  "
              f"val[{tag}] acc {best_of['acc']:.4f} f1 {best_of['f1']:.4f} "
              f"auc {best_of['auc']:.4f}  ({time.time() - t0:.0f}s)", flush=True)
        history.append({"epoch": epoch, "loss": float(np.mean(running)),
                        "raw": m_raw, "ema": m_ema})

        if best_of["f1"] > best_f1:
            best_f1, best_epoch, stale = best_of["f1"], epoch, 0
            state = (ema.shadow if tag == "ema" else model).state_dict()
            torch.save({"model": state, "config": {
                "backbone": MODEL.backbone, "use_srm": MODEL.use_srm,
                "temporal": MODEL.temporal, "img_size": DATA.img_size,
                "clip_len": DATA.clip_len,
                "consistency_weight": TRAIN.consistency_weight},
                "epoch": epoch, "val": best_of}, out / "best.pt")
            print(f"  saved best.pt (f1 {best_f1:.4f})")
        else:
            stale += 1
            print(f"  no improvement for {stale} epoch(s); "
                  f"best is epoch {best_epoch} at f1 {best_f1:.4f}")

        torch.save({"model": model.state_dict(),
                    "ema": ema.shadow.state_dict(),
                    "opt": opt.state_dict(),
                    "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(),
                    "epoch": epoch, "best_f1": best_f1,
                    "best_epoch": best_epoch, "history": history,
                    "config": {"backbone": MODEL.backbone,
                               "use_srm": MODEL.use_srm,
                               "temporal": MODEL.temporal,
                               "img_size": DATA.img_size,
                               "clip_len": DATA.clip_len}},
                   out / "last.pt")
        (out / "history.json").write_text(json.dumps(history, indent=2))

        if args.patience and stale >= args.patience:
            print()
            print(f"stopping early: no val F1 gain in {stale} epochs. "
                  f"The cosine schedule did not finish, so the last epochs "
                  f"ran at a higher LR than planned - best.pt still holds "
                  f"the peak-F1 weights, which is what you want.")
            break

    print()
    print(f"done. best val f1 {best_f1:.4f} at epoch {best_epoch} "
          f"(of {len(history)} epochs run). best.pt holds those weights.")


if __name__ == "__main__":
    main()
