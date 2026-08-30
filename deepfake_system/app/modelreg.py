"""Model registry and hot-swap — IR 3.4.1, non-functional requirement 4.

"The system architecture should enable the ongoing updating of its machine
learning models to keep up with the accelerating, ongoing changes in the
technologies involved in deepfake generation."

Before this, updating the model meant overwriting `runs/v1/best.pt` and
restarting the process. That fails the requirement in three ways: the
service goes down, a bad checkpoint is only discovered after it is already
live, and nothing records that the model changed — so a verdict issued
last week cannot be tied to the weights that produced it.

What this adds:

- **Discovery.** Any `*.pt` under `runs/` is a candidate, with the metrics
  the training run recorded alongside it.
- **Validation before activation.** A candidate is loaded into a *separate*
  model instance and made to score a synthetic clip. Only if that produces
  a finite probability does it replace the live one. A checkpoint that is
  truncated, built for a different architecture, or silently outputs NaN
  is rejected while the running model keeps serving.
- **Atomic swap.** The engine's attributes are replaced in one step under
  a lock, so an analysis in flight finishes on the model it started with.
- **An audit trail.** Activation is an audited event, and `model_version`
  is already written onto every stored analysis, so any past verdict can
  be traced to the weights behind it.

Activation is admin-only. A detection model is the thing the entire
forensic claim rests on; letting any account swap it would make every
report unfalsifiable.
"""
from __future__ import annotations

import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import DATA, INFER, MODEL  # noqa: E402

_swap_lock = threading.Lock()


def _runs_dir() -> Path:
    return Path(INFER.checkpoint).resolve().parent.parent


def discover() -> list[dict]:
    """Every checkpoint under runs/, newest first."""
    root = _runs_dir()
    out: list[dict] = []
    if not root.exists():
        return out

    active = Path(INFER.checkpoint).resolve()
    for path in sorted(root.rglob("*.pt")):
        try:
            stat = path.stat()
        except OSError:
            continue
        entry = {
            "id": path.relative_to(root).as_posix(),
            "path": str(path),
            "size_mb": round(stat.st_size / 1e6, 1),
            "modified": datetime.fromtimestamp(
                stat.st_mtime, timezone.utc).isoformat(),
            "active": path.resolve() == active,
        }
        entry.update(_peek(path))
        out.append(entry)
    out.sort(key=lambda e: e["modified"], reverse=True)
    return out


def _peek(path: Path) -> dict:
    """Read a checkpoint's metadata without building the model.

    `weights_only=True` matters: a .pt file is a pickle, and unpickling an
    untrusted one executes arbitrary code. Listing the available models
    must not be a way to run something.
    """
    try:
        import torch  # type: ignore
        ck = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        # Older checkpoints saved with a config dict cannot be read under
        # weights_only. Report what the filesystem knows instead of
        # falling back to an unsafe load.
        return {"readable": False, "backbone": None, "metrics": None}

    cfg = ck.get("config", {}) if isinstance(ck, dict) else {}
    val = ck.get("val", {}) if isinstance(ck, dict) else {}
    return {
        "readable": True,
        "backbone": cfg.get("backbone"),
        "epoch": ck.get("epoch") if isinstance(ck, dict) else None,
        "metrics": {k: round(float(v), 4) for k, v in val.items()
                    if isinstance(v, (int, float))} or None,
    }


def validate(path: Path) -> dict:
    """Load a candidate in isolation and make it score something.

    Returns {"ok": bool, "reason": str, ...}. Never raises, and never
    touches the running engine.
    """
    import numpy as np

    try:
        import torch  # type: ignore
        from models.net import build_model  # type: ignore
    except Exception as exc:
        return {"ok": False, "reason": f"torch unavailable: {exc}"}

    if not path.exists():
        return {"ok": False, "reason": "file does not exist"}

    try:
        ck = torch.load(path, map_location="cpu")
    except Exception as exc:
        return {"ok": False,
                "reason": f"unreadable checkpoint: {type(exc).__name__}"}
    if not isinstance(ck, dict) or "model" not in ck:
        return {"ok": False, "reason": "not a training checkpoint "
                                       "(no 'model' state dict)"}

    cfg = ck.get("config", {}) or {}
    # Build against a copy of the config so a failed candidate cannot
    # leave the global MODEL settings mutated for the live engine.
    import copy
    probe_cfg = copy.deepcopy(MODEL)
    probe_cfg.backbone = cfg.get("backbone", MODEL.backbone)
    probe_cfg.use_srm = cfg.get("use_srm", MODEL.use_srm)
    probe_cfg.temporal = cfg.get("temporal", MODEL.temporal)
    probe_cfg.pretrained = False

    try:
        model = build_model(probe_cfg).eval()
        missing, unexpected = model.load_state_dict(ck["model"], strict=False)
        if missing:
            return {"ok": False,
                    "reason": f"{len(missing)} weights missing from the "
                              "checkpoint - architecture mismatch"}
    except Exception as exc:
        return {"ok": False,
                "reason": f"will not build: {type(exc).__name__}: "
                          f"{str(exc)[:120]}"}

    # A smoke test on noise. It cannot tell us the model is *good* — only
    # that it runs and produces a usable number. A checkpoint that emits
    # NaN passes every structural check and then poisons every verdict.
    clip_len = cfg.get("clip_len", DATA.clip_len)
    size = cfg.get("img_size", DATA.img_size)
    try:
        x = torch.from_numpy(
            np.random.default_rng(0).standard_normal(
                (1, clip_len, 3, size, size)).astype("float32"))
        with torch.no_grad():
            logit, frame_logits = model(x)
        prob = float(torch.sigmoid(logit)[0])
        if not (0.0 <= prob <= 1.0) or prob != prob:      # NaN != NaN
            return {"ok": False, "reason": f"produced an unusable "
                                           f"probability ({prob})"}
        if int(frame_logits.shape[1]) != clip_len:
            return {"ok": False, "reason": "frame head shape mismatch"}
    except Exception as exc:
        return {"ok": False,
                "reason": f"inference failed: {type(exc).__name__}: "
                          f"{str(exc)[:120]}"}

    val = ck.get("val", {}) or {}
    return {
        "ok": True,
        "reason": "loaded and scored a probe clip",
        "backbone": probe_cfg.backbone,
        "clip_len": clip_len,
        "img_size": size,
        "probe_probability": round(prob, 4),
        "metrics": {k: round(float(v), 4) for k, v in val.items()
                    if isinstance(v, (int, float))} or None,
        "unexpected_keys": len(unexpected),
    }


def activate(engine, checkpoint_id: str) -> dict:
    """Validate a checkpoint, then swap it into a live engine.

    The engine keeps serving its current model throughout; if anything
    below fails, nothing about it changes.
    """
    root = _runs_dir()
    path = (root / checkpoint_id).resolve()

    # Path containment: the id comes off an HTTP request, so "../../etc"
    # must not resolve to somewhere outside the runs directory.
    try:
        path.relative_to(root.resolve())
    except ValueError:
        return {"ok": False, "reason": "checkpoint id is outside runs/"}

    report = validate(path)
    if not report.get("ok"):
        return report

    with _swap_lock:
        previous = engine.model_version
        try:
            # Engine._load rebuilds model, threshold, clip_len and
            # model_version from the checkpoint. Doing it through the
            # engine's own loader keeps one code path for "how a model is
            # brought up" rather than duplicating it here.
            engine._load(str(path))
        except Exception as exc:
            return {"ok": False,
                    "reason": f"swap failed: {type(exc).__name__}: "
                              f"{str(exc)[:120]}",
                    "still_running": previous}
        if engine.model is None or engine.mode != "model":
            return {"ok": False, "reason": "engine did not come back up "
                                           "in model mode",
                    "still_running": previous}
        INFER.checkpoint = path
        current = engine.model_version

    return {"ok": True, "previous": previous, "active": current,
            "checkpoint": checkpoint_id, "validation": report}
