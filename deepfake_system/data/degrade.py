from __future__ import annotations

import random

import cv2
import numpy as np

_LONG_EDGES = (480, 640, 854, 1280)


def _jpeg(frame: np.ndarray, quality: int) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", frame[:, :, ::-1],
                           [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return frame
    dec = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return dec[:, :, ::-1] if dec is not None else frame


def _chroma_subsample(frame: np.ndarray, factor: int = 2) -> np.ndarray:
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    h, w = y.shape
    small = (max(1, w // factor), max(1, h // factor))
    cr = cv2.resize(cv2.resize(cr, small, interpolation=cv2.INTER_AREA),
                    (w, h), interpolation=cv2.INTER_LINEAR)
    cb = cv2.resize(cv2.resize(cb, small, interpolation=cv2.INTER_AREA),
                    (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(cv2.merge([y, cr, cb]), cv2.COLOR_YCrCb2RGB)


def _rescale(frame: np.ndarray, long_edge: int) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = long_edge / max(h, w)
    if scale >= 1.0:
        return frame
    small = cv2.resize(frame, (max(8, int(w * scale)), max(8, int(h * scale))),
                       interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def _block_artifacts(frame: np.ndarray, strength: float) -> np.ndarray:
    h, w = frame.shape[:2]
    hh, ww = h - h % 8, w - w % 8
    if hh < 8 or ww < 8:
        return frame
    crop = frame[:hh, :ww].astype(np.float32)
    blocks = crop.reshape(hh // 8, 8, ww // 8, 8, 3)
    means = blocks.mean(axis=(1, 3), keepdims=True)
    blended = blocks * (1 - strength) + means * strength
    out = frame.copy().astype(np.float32)
    out[:hh, :ww] = blended.reshape(hh, ww, 3)
    return np.clip(out, 0, 255).astype(np.uint8)


def sample_recipe(rng: random.Random | None = None) -> dict:
    r = rng or random
    tier = r.choices(["light", "typical", "harsh"], weights=[3, 5, 2])[0]
    if tier == "light":
        return dict(long_edge=r.choice((854, 1280)), q1=r.randint(65, 85),
                    passes=1, block=r.uniform(0.0, 0.10),
                    noise=r.uniform(0.0, 1.5), blur=0.0, tier=tier)
    if tier == "typical":
        return dict(long_edge=r.choice((640, 854)), q1=r.randint(40, 65),
                    passes=r.choice((1, 2)), block=r.uniform(0.10, 0.25),
                    noise=r.uniform(0.5, 2.5), blur=r.choice((0.0, 0.6)),
                    tier=tier)
    return dict(long_edge=r.choice((480, 640)), q1=r.randint(22, 40),
                passes=r.choice((2, 3)), block=r.uniform(0.25, 0.45),
                noise=r.uniform(1.0, 4.0), blur=r.choice((0.6, 0.9)),
                tier=tier)


def apply_recipe(frame: np.ndarray, recipe: dict) -> np.ndarray:
    out = _rescale(frame, recipe["long_edge"])
    out = _chroma_subsample(out)
    if recipe["blur"] > 0:
        out = cv2.GaussianBlur(out, (0, 0), recipe["blur"])
    if recipe["block"] > 0:
        out = _block_artifacts(out, recipe["block"])
    q = recipe["q1"]
    for _ in range(recipe["passes"]):
        out = _jpeg(out, q)
        q = max(18, q - 8)         
    if recipe["noise"] > 0:
        noise = np.random.normal(0, recipe["noise"], out.shape)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return out


def degrade_clip(clip: np.ndarray, rng: random.Random | None = None
                 ) -> tuple[np.ndarray, dict]:
    recipe = sample_recipe(rng)
    out = np.stack([apply_recipe(f, recipe) for f in clip])
    return out, recipe



def looks_like_messenger_upload(video_path: str, meta: dict | None = None
                                ) -> dict:
    import os
    import re

    name = os.path.basename(video_path)
    flags = {
        "whatsapp_filename": bool(re.match(r"(VID|IMG)-\d{8}-WA\d{4}", name)),
        "telegram_filename": bool(re.match(r"video_\d{4}-\d{2}-\d{2}", name)),
        "stripped_metadata": False,
        "low_bitrate": False,
        "capped_resolution": False,
    }
    if meta:
        tags = meta.get("tags", {}) or {}
        flags["stripped_metadata"] = not any(
            k in tags for k in ("com.apple.quicktime.model", "encoder", "model"))
        br = meta.get("bit_rate")
        if br:
            flags["low_bitrate"] = int(br) < 1_600_000
        h = meta.get("height")
        if h:
            flags["capped_resolution"] = int(h) in (480, 540, 720)
    flags["likely_recompressed"] = sum(
        bool(v) for k, v in flags.items() if k != "likely_recompressed") >= 2
    return flags
