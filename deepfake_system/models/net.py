"""Detector network.

Frames go through a shared CNN backbone; an optional SRM high-pass stream
runs beside it because residual noise patterns behave differently from
RGB content under compression. The per-frame embeddings are then pooled
across the clip with an attention head, so a video where only a few
frames are manipulated still gets flagged.
"""
from __future__ import annotations

import sys
from pathlib import Path

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import MODEL  # noqa: E402

# Three standard SRM high-pass residual kernels.
_SRM = torch.tensor([
    [[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0],
     [0, -1, 2, -1, 0], [0, 0, 0, 0, 0]],
    [[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2],
     [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]],
    [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, -2, 1, 0],
     [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
], dtype=torch.float32) / torch.tensor([4.0, 12.0, 2.0]).view(3, 1, 1)


class SRMConv(nn.Module):
    """Fixed high-pass filter bank, applied per colour channel."""

    def __init__(self):
        super().__init__()
        w = _SRM.unsqueeze(1).repeat(3, 1, 1, 1)      # (9, 1, 5, 5)
        self.register_buffer("weight", w)

    def forward(self, x):
        out = F.conv2d(x, self.weight, padding=2, groups=3)
        return torch.clamp(out, -3.0, 3.0)


class MesoInception4(nn.Module):
    """MesoNet / MesoInception-4, per Afchar et al. 2018 — the network
    cited in IR section 1.4 (objective: To Develop).

    Roughly 28k parameters against EfficientNetV2-S's 21M. It is far
    weaker in absolute accuracy, but it is the architecture the report
    specifies, it trains in minutes, and it makes an honest ablation row:
    "the cited compact network vs a modern backbone, same data, same
    protocol". Select it with --backbone mesonet.
    """

    def __init__(self, num_classes=0):
        super().__init__()
        self.num_features = 16

        def inception(cin, a, b, c, d):
            return nn.ModuleDict({
                "p1": nn.Conv2d(cin, a, 1),
                "p2": nn.Sequential(nn.Conv2d(cin, b, 1),
                                    nn.Conv2d(b, b, 3, padding=1)),
                "p3": nn.Sequential(nn.Conv2d(cin, c, 1),
                                    nn.Conv2d(c, c, 3, padding=2, dilation=2)),
                "p4": nn.Sequential(nn.Conv2d(cin, d, 1),
                                    nn.Conv2d(d, d, 3, padding=3, dilation=3)),
            })

        self.inc1 = inception(3, 1, 4, 4, 2)
        self.bn1 = nn.BatchNorm2d(11)
        self.inc2 = inception(11, 2, 4, 4, 2)
        self.bn2 = nn.BatchNorm2d(12)

        self.conv3 = nn.Conv2d(12, 16, 5, padding=2)
        self.bn3 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16, 16, 5, padding=2)
        self.bn4 = nn.BatchNorm2d(16)

        self.pool = nn.MaxPool2d(2)
        self.wide_pool = nn.MaxPool2d(4)
        self.gap = nn.AdaptiveAvgPool2d(1)

    @staticmethod
    def _run_inception(block, x):
        return torch.cat([block["p1"](x), block["p2"](x),
                          block["p3"](x), block["p4"](x)], dim=1)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self._run_inception(self.inc1, x))))
        x = self.pool(F.relu(self.bn2(self._run_inception(self.inc2, x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.wide_pool(F.relu(self.bn4(self.conv4(x))))
        return self.gap(x).flatten(1)


class TemporalAttention(nn.Module):
    def __init__(self, dim, hidden=128):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, x, return_weights=False):
        # x: (B, T, D)
        w = torch.softmax(self.score(x).squeeze(-1), dim=1)   # (B, T)
        pooled = torch.einsum("bt,btd->bd", w, x)
        return (pooled, w) if return_weights else (pooled, None)


class DeepfakeDetector(nn.Module):
    def __init__(self, cfg=MODEL):
        super().__init__()
        self.cfg = cfg
        if cfg.backbone.lower() in ("mesonet", "mesoinception4", "meso4"):
            self.backbone = MesoInception4()
        else:
            self.backbone = timm.create_model(
                cfg.backbone, pretrained=cfg.pretrained, num_classes=0)
        dim = self.backbone.num_features

        self.use_srm = cfg.use_srm
        if self.use_srm:
            self.srm = SRMConv()
            self.srm_stem = nn.Sequential(
                nn.Conv2d(9, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32),
                nn.SiLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64),
                nn.SiLU(),
                nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d(1), nn.Flatten())
            dim += 128

        if cfg.temporal == "gru":
            self.temporal = nn.GRU(dim, 256, batch_first=True,
                                   bidirectional=True)
            pooled_dim = 512
        elif cfg.temporal == "attention":
            self.temporal = TemporalAttention(dim)
            pooled_dim = dim
        else:
            self.temporal = None
            pooled_dim = dim

        self.head = nn.Sequential(
            nn.Dropout(cfg.dropout),
            nn.Linear(pooled_dim, 256), nn.SiLU(),
            nn.Dropout(cfg.dropout * 0.5),
            nn.Linear(256, 1))
        # Auxiliary per-frame head — supervising frames directly speeds up
        # convergence and gives the UI a per-frame timeline.
        self.frame_head = nn.Sequential(nn.Dropout(cfg.dropout),
                                        nn.Linear(dim, 1))

    def encode_frames(self, x):
        """x: (B, T, 3, H, W) -> (B, T, D)"""
        B, T = x.shape[:2]
        flat = x.flatten(0, 1)
        feat = self.backbone(flat)
        if self.use_srm:
            feat = torch.cat([feat, self.srm_stem(self.srm(flat))], dim=1)
        return feat.view(B, T, -1)

    def forward(self, x, return_attention=False):
        feats = self.encode_frames(x)                     # (B, T, D)
        frame_logits = self.frame_head(feats).squeeze(-1)  # (B, T)

        if isinstance(self.temporal, nn.GRU):
            out, _ = self.temporal(feats)
            pooled, attn = out.mean(dim=1), None
        elif isinstance(self.temporal, TemporalAttention):
            pooled, attn = self.temporal(feats, return_weights=True)
        else:
            pooled, attn = feats.mean(dim=1), None

        logit = self.head(pooled).squeeze(-1)              # (B,)
        if return_attention:
            return logit, frame_logits, attn
        return logit, frame_logits


def build_model(cfg=MODEL):
    return DeepfakeDetector(cfg)


def param_groups(model, base_lr, backbone_mult, weight_decay):
    backbone, rest = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (backbone if name.startswith("backbone") else rest).append(p)
    return [
        {"params": backbone, "lr": base_lr * backbone_mult,
         "weight_decay": weight_decay},
        {"params": rest, "lr": base_lr, "weight_decay": weight_decay},
    ]
