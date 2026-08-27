"""Signal-based fallback scorer.

The CNN is the real detector. But you need a running server before you
have a trained checkpoint, and a demo that returns random numbers is
worse than useless. So this module computes genuine forensic features and
combines them with fixed, hand-set weights.

It is uncalibrated and it says so everywhere it appears in the UI. Expect
roughly 60-70% accuracy from it — useful for wiring up the interface and
for the "classical baseline" row in your results table, not for a claim.

Features, all computed on the tracked face region:

  sharpness_ratio   face detail vs surrounding detail. Swapped faces are
                    often softer than the frame they were pasted into.
  boundary_energy   gradient magnitude on the ring around the face oval,
                    where blending seams live.
  temporal_flicker  frame-to-frame change in the aligned face, above what
                    the rest of the frame is doing.
  spectral_slope    high-band vs mid-band radial FFT energy. Upsampling
                    from a generator's output resolution leaves a bump.
  chroma_mismatch   colour spread inside the face vs just outside it.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

WEIGHTS = {
    "sharpness_ratio": 0.26,
    "boundary_energy": 0.22,
    "temporal_flicker": 0.20,
    "spectral_slope": 0.20,
    "chroma_mismatch": 0.12,
}

# (centre, scale, direction) for the logistic squash. direction -1 means
# a low raw value is the suspicious one.
CALIB = {
    "sharpness_ratio": (0.85, 0.30, -1),
    "boundary_energy": (14.0, 6.0, +1),
    "temporal_flicker": (0.055, 0.030, +1),
    "spectral_slope": (0.38, 0.12, +1),
    "chroma_mismatch": (7.5, 4.0, +1),
}

READABLE = {
    "sharpness_ratio": "Face detail vs surroundings",
    "boundary_energy": "Edge energy at the face boundary",
    "temporal_flicker": "Frame-to-frame instability",
    "spectral_slope": "High-frequency spectral bump",
    "chroma_mismatch": "Colour spread inside vs outside the face",
}


def _squash(value, centre, scale, direction):
    if not math.isfinite(value):
        return 0.5
    z = direction * (value - centre) / max(1e-6, scale)
    z = max(-30.0, min(30.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def _oval_masks(h, w):
    inner = np.zeros((h, w), np.uint8)
    cv2.ellipse(inner, (w // 2, int(h * 0.52)),
                (int(w * 0.34), int(h * 0.42)), 0, 0, 360, 255, -1)
    outer = np.zeros((h, w), np.uint8)
    cv2.ellipse(outer, (w // 2, int(h * 0.52)),
                (int(w * 0.44), int(h * 0.50)), 0, 0, 360, 255, -1)
    ring = cv2.subtract(outer, inner)
    background = cv2.bitwise_not(outer)
    return inner, ring, background


def _sharpness(gray, mask):
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    vals = lap[mask > 0]
    return float(vals.var()) if vals.size else 0.0


def _radial_spectrum(gray):
    g = gray.astype(np.float32) / 255.0
    g = g - g.mean()
    win = np.outer(np.hanning(g.shape[0]), np.hanning(g.shape[1]))
    mag = np.abs(np.fft.fftshift(np.fft.fft2(g * win)))
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    rmax = r.max()
    mid = mag[(r > rmax * 0.18) & (r <= rmax * 0.45)].mean()
    high = mag[r > rmax * 0.60].mean()
    return float(high / max(1e-6, mid))


def score_clip(crops: np.ndarray) -> dict:
    """crops: (T, H, W, 3) uint8 RGB face crops from one video."""
    if len(crops) == 0:
        return {"score": 0.0, "features": {}, "usable": False}

    h, w = crops[0].shape[:2]
    inner, ring, background = _oval_masks(h, w)

    sharp_ratios, boundary, spectral, chroma = [], [], [], []
    for c in crops:
        gray = cv2.cvtColor(c, cv2.COLOR_RGB2GRAY)

        s_in = _sharpness(gray, inner)
        s_out = _sharpness(gray, background)
        sharp_ratios.append(s_in / max(1e-6, s_out))

        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad = np.sqrt(gx ** 2 + gy ** 2)
        boundary.append(float(grad[ring > 0].mean()))

        spectral.append(_radial_spectrum(gray))

        ycrcb = cv2.cvtColor(c, cv2.COLOR_RGB2YCrCb)
        cr = ycrcb[:, :, 1].astype(np.float32)
        chroma.append(abs(float(cr[inner > 0].std()) -
                          float(cr[background > 0].std())))

    flicker = []
    for a, b in zip(crops[:-1], crops[1:]):
        ga = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
        gb = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
        d_face = np.abs(gb - ga)[inner > 0].mean()
        d_bg = np.abs(gb - ga)[background > 0].mean()
        flicker.append(float(d_face - 0.7 * d_bg))

    raw = {
        "sharpness_ratio": float(np.median(sharp_ratios)),
        "boundary_energy": float(np.median(boundary)),
        "temporal_flicker": float(np.median(flicker)) if flicker else 0.0,
        "spectral_slope": float(np.median(spectral)),
        "chroma_mismatch": float(np.median(chroma)),
    }

    features, total = {}, 0.0
    for k, v in raw.items():
        norm = _squash(v, *CALIB[k])
        total += WEIGHTS[k] * norm
        features[k] = {"label": READABLE[k], "raw": round(v, 4),
                       "normalised": round(norm, 4),
                       "weight": WEIGHTS[k]}

    return {"score": float(total), "features": features, "usable": True}


def per_frame_scores(crops: np.ndarray) -> list[float]:
    """A cheap per-frame proxy so the timeline has something to draw."""
    if len(crops) == 0:
        return []
    h, w = crops[0].shape[:2]
    inner, ring, background = _oval_masks(h, w)
    out = []
    for c in crops:
        gray = cv2.cvtColor(c, cv2.COLOR_RGB2GRAY)
        s_in = _sharpness(gray, inner)
        s_out = _sharpness(gray, background)
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad = np.sqrt(gx ** 2 + gy ** 2)
        s = (0.55 * _squash(s_in / max(1e-6, s_out), *CALIB["sharpness_ratio"]) +
             0.45 * _squash(float(grad[ring > 0].mean()),
                            *CALIB["boundary_energy"]))
        out.append(round(float(s), 4))
    return out
