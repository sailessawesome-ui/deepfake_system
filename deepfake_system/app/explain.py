from __future__ import annotations

import base64
from typing import Any

import cv2
import numpy as np


def _last_conv(module) -> Any:
    import torch.nn as nn  # type: ignore
    found = None
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            found = m
    return found


class GradCAM:

    def __init__(self, model, device: str = "cpu"):
        self.model = model
        self.device = device
        self.layer = _last_conv(getattr(model, "backbone", model))
        self._acts: Any = None
        self._grads: Any = None
        self._handles: list = []

    def __enter__(self):
        if self.layer is None:
            return self

        def fwd(_m, _inp, out):
            self._acts = out

        def bwd(_m, _gin, gout):
            self._grads = gout[0]

        self._handles.append(self.layer.register_forward_hook(fwd))
        self._handles.append(self.layer.register_full_backward_hook(bwd))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles.clear()
        self._acts = self._grads = None
        return False

    def cam(self, clip: Any, frame_index: int) -> np.ndarray | None:
        import torch  # type: ignore

        if self.layer is None:
            return None
        try:
            self.model.zero_grad(set_to_none=True)
            with torch.enable_grad():
                clip = clip.clone().requires_grad_(False)
                _logit, frame_logits = self.model(clip)
                idx = min(max(frame_index, 0), frame_logits.shape[1] - 1)
                target = frame_logits[0, idx]
                target.backward()

            if self._acts is None or self._grads is None:
                return None

            row = min(idx, self._acts.shape[0] - 1)
            acts = self._acts[row].detach()              
            grads = self._grads[row].detach()            

            weights = grads.mean(dim=(1, 2), keepdim=True)  
            cam = torch.relu((weights * acts).sum(dim=0))   
            cam_np = cam.float().cpu().numpy()
        except Exception as exc:
            print(f"[explain] Grad-CAM failed: {type(exc).__name__}: {exc}")
            return None
        finally:
            self.model.zero_grad(set_to_none=True)

        span = float(cam_np.max() - cam_np.min())
        if span < 1e-8:
            return None
        return (cam_np - cam_np.min()) / span


def overlay(crop_rgb: np.ndarray, cam: np.ndarray, size: int = 104,
            alpha: float = 0.45, quality: int = 72) -> str | None:
    try:
        h, w = crop_rgb.shape[:2]
        heat = cv2.resize(cam.astype(np.float32), (w, h),
                          interpolation=cv2.INTER_CUBIC)
        heat = np.clip(heat, 0.0, 1.0)

        colour = cv2.applyColorMap((heat * 255).astype(np.uint8),
                                   cv2.COLORMAP_INFERNO)[:, :, ::-1]

        a = (heat * alpha)[..., None]
        blended = crop_rgb.astype(np.float32) * (1 - a) + \
            colour.astype(np.float32) * a
        out = np.clip(blended, 0, 255).astype(np.uint8)

        small = cv2.resize(out, (size, size), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", small[:, :, ::-1],
                               [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            return None
        return "data:image/jpeg;base64," + base64.b64encode(buf).decode()
    except Exception as exc:
        print(f"[explain] overlay failed: {type(exc).__name__}: {exc}")
        return None


def describe(cam: np.ndarray) -> str:
    h, w = cam.shape
    rows = {"upper face (eyes and brow)": cam[:h // 3].mean(),
            "mid-face (nose and cheeks)": cam[h // 3:2 * h // 3].mean(),
            "lower face (mouth and jaw)": cam[2 * h // 3:].mean()}
    where = max(rows, key=lambda k: rows[k])

    peak = float(cam.max())
    spread = float((cam > 0.5).mean())
    if spread > 0.45:
        focus = ("The evidence is spread across the whole crop rather than "
                 "one region, which is weaker support for a specific "
                 "artefact.")
    else:
        focus = f"The strongest response is in the {where}."
    return f"{focus} Peak response {peak:.2f}, covering {spread * 100:.0f}% of the crop."


def explain_clip(model, crops: np.ndarray, indices: list[int],
                 device: str, clip_len: int, mean, std) -> dict[int, dict]:
    import torch  # type: ignore

    out: dict[int, dict] = {}
    if model is None or len(crops) == 0 or not indices:
        return out

    n = len(crops)
    try:
        with GradCAM(model, device) as cam_maker:
            if cam_maker.layer is None:
                return out
            for i in indices:
                if not 0 <= i < n:
                    continue
                window = crops[i:i + 1]
                arr = window.astype(np.float32) / 255.0
                arr = (arr - mean) / std
                x = torch.from_numpy(arr).permute(0, 3, 1, 2).unsqueeze(0)
                x = x.to(device)

                cam = cam_maker.cam(x, 0)
                if cam is None:
                    continue
                out[i] = {"cam": cam, "text": describe(cam)}
    except Exception as exc:
        print(f"[explain] {type(exc).__name__}: {exc}")
    return out
