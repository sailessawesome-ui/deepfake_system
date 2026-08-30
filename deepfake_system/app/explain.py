"""Grad-CAM explanations — IR 3.4.1, UI/UX requirement 3.

The report asks for "minimal visual cues (Explainable AI) to explain why
it was called out (e.g., pointing to a particular artifact on the face)",
and records it as the top priority of more than half of the 56 survey
respondents. Nothing in the system did this: it returned a number and a
label, and the analyst had to take both on trust.

Grad-CAM (Selvaraju et al., 2017) answers "which pixels moved this
verdict" by taking the gradient of the model's own decision with respect
to the last convolutional feature map. Channels whose activations push
the logit up are weighted up; the weighted sum, rectified, is a coarse map
over the face crop. It is not a segmentation of "the fake part" — it is
where the network looked, which is the honest claim to make in a viva.

Two properties that matter for a forensic tool:

- It explains *this* model's decision, not deepfakes in general. If the
  map lands on the background, that is real evidence the score is being
  driven by something other than the face, and the analyst should see it.
- It costs one extra forward and backward pass per explained frame, so
  only the few most suspicious frames are explained. Explaining all of
  them would multiply a 10-15 s analysis several times over for no
  additional insight.

Nothing here is written to disk. Overlays go back in the JSON response
next to the existing thumbnails and die with the request, which is what
IR 3.4.1's zero-retention requirement demands.
"""
from __future__ import annotations

import base64
from typing import Any

import cv2
import numpy as np


def _last_conv(module) -> Any:
    """The deepest Conv2d in the backbone.

    Grad-CAM needs a layer that still has spatial extent; by the time the
    features reach the classifier head they are a single vector and there
    is nothing left to localise. Walking for the last Conv2d works for
    both the timm backbones and MesoInception4 without hard-coding a
    layer name per architecture.
    """
    import torch.nn as nn  # type: ignore
    found = None
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            found = m
    return found


class GradCAM:
    """Hooks a layer, then turns one frame's logit into a heatmap."""

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
        # full_backward_hook is the non-deprecated spelling and fires
        # after the whole module's grads are computed.
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
        """clip: (1, T, 3, H, W) tensor. Returns an HxW map in [0, 1]."""
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

            # The model flattens (B, T) into the batch dimension before the
            # backbone, so with B=1 the row for frame t is simply row t.
            row = min(idx, self._acts.shape[0] - 1)
            acts = self._acts[row].detach()               # (C, H, W)
            grads = self._grads[row].detach()             # (C, H, W)

            weights = grads.mean(dim=(1, 2), keepdim=True)   # (C, 1, 1)
            cam = torch.relu((weights * acts).sum(dim=0))    # (H, W)
            cam_np = cam.float().cpu().numpy()
        except Exception as exc:
            print(f"[explain] Grad-CAM failed: {type(exc).__name__}: {exc}")
            return None
        finally:
            self.model.zero_grad(set_to_none=True)

        span = float(cam_np.max() - cam_np.min())
        if span < 1e-8:
            # A flat map means the gradient said nothing. Returning zeros
            # would render as a uniform wash that looks like a real result;
            # better to show no overlay at all.
            return None
        return (cam_np - cam_np.min()) / span


def overlay(crop_rgb: np.ndarray, cam: np.ndarray, size: int = 104,
            alpha: float = 0.45, quality: int = 72) -> str | None:
    """Blend a heatmap over the face crop and return a data URI."""
    try:
        h, w = crop_rgb.shape[:2]
        heat = cv2.resize(cam.astype(np.float32), (w, h),
                          interpolation=cv2.INTER_CUBIC)
        heat = np.clip(heat, 0.0, 1.0)

        # INFERNO reads as "intensity" without implying the red/green
        # semantics of a verdict, which JET would.
        colour = cv2.applyColorMap((heat * 255).astype(np.uint8),
                                   cv2.COLORMAP_INFERNO)[:, :, ::-1]

        # Weight the blend by the map itself, so cool regions stay close
        # to the original pixels and only the hot ones are tinted. A flat
        # alpha washes the whole face and hides the very artefact the
        # analyst is being pointed at.
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
    """Where the map concentrates, in words.

    The IR asks for output a non-technical user can read, so the heatmap
    ships with a sentence rather than only a picture. Thirds of the crop
    are used because face crops are aligned, so the regions correspond
    roughly to brow/eyes, nose/cheeks and mouth/jaw.
    """
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
    """Grad-CAM for a handful of frames. Never raises."""
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
                # A single frame, not the whole clip. `frame_logits[0, t]`
                # comes from `frame_head(feats[0, t])`, and `feats[0, t]`
                # depends only on `x[0, t]` — so pushing the other
                # clip_len-1 frames through the backbone would cost a
                # multiple of the time and change nothing about this
                # frame's gradient. The temporal layer still runs, over a
                # sequence of one, and its output is not backpropagated.
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
