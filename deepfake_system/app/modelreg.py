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
    try:
        import torch  # type: ignore
        ck = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
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

    clip_len = cfg.get("clip_len", DATA.clip_len)
    size = cfg.get("img_size", DATA.img_size)
    try:
        x = torch.from_numpy(
            np.random.default_rng(0).standard_normal(
                (1, clip_len, 3, size, size)).astype("float32"))
        with torch.no_grad():
            logit, frame_logits = model(x)
        prob = float(torch.sigmoid(logit)[0])
        if not (0.0 <= prob <= 1.0) or prob != prob:     
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
    root = _runs_dir()
    path = (root / checkpoint_id).resolve()

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
