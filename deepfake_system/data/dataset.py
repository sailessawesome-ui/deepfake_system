from __future__ import annotations

import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA, TRAIN  # noqa: E402
from data.degrade import degrade_clip  # noqa: E402

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_manifest(path=None, splits=("train",)):
    import csv
    path = Path(path or DATA.manifest)
    videos = defaultdict(lambda: {"frames": [], "label": 0,
                                  "source": "", "method": ""})
    with path.open() as f:
        for row in csv.DictReader(f):
            if row["split"] not in splits:
                continue
            v = videos[row["video_id"]]
            v["frames"].append(row["frame_path"])
            v["label"] = int(row["label"])
            v["source"] = row["source"]
            v["method"] = row["method"]
    for v in videos.values():
        v["frames"].sort()
    return dict(videos)


class ClipDataset(Dataset):
    def __init__(self, videos: dict, train: bool = True,
                 clips_per_video: int = 1, degrade_prob: float | None = None,
                 paired: bool | None = None, img_size: int | None = None):
        self.ids = [k for k, v in videos.items()
                    if len(v["frames"]) >= 2]
        self.videos = videos
        self.train = train
        self.clips_per_video = clips_per_video
        self.degrade_prob = (TRAIN.degrade_prob if degrade_prob is None
                             else degrade_prob)
        self.paired = train if paired is None else paired
        self.img_size = img_size or DATA.img_size
        self.T = DATA.clip_len

    def __len__(self):
        return len(self.ids) * self.clips_per_video

    def _sample_frames(self, frames, rng):
        n = len(frames)
        need = self.T
        stride = DATA.frame_stride
        span = min(n, need * stride)
        if self.train:
            start = rng.randint(0, max(0, n - span))
        else:
            start = max(0, (n - span) // 2)
        idx = [min(n - 1, start + i * stride) for i in range(need)]
        return [frames[i] for i in idx]

    def _read(self, paths):
        out = []
        last = None
        for p in paths:
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is None:
                img = last if last is not None else np.zeros(
                    (self.img_size, self.img_size, 3), np.uint8)
            else:
                img = img[:, :, ::-1]
                img = cv2.resize(img, (self.img_size, self.img_size),
                                 interpolation=cv2.INTER_AREA)
            last = img
            out.append(img)
        return np.stack(out)

    def _spatial_aug(self, clip, rng):
        if rng.random() < 0.5:
            clip = clip[:, :, ::-1]
        if rng.random() < 0.3:                      
            ang = rng.uniform(-8, 8)
            h, w = clip.shape[1:3]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
            clip = np.stack([cv2.warpAffine(f, M, (w, h),
                                            borderMode=cv2.BORDER_REFLECT)
                             for f in clip])
        if rng.random() < 0.3:                      
            a, b = rng.uniform(0.85, 1.15), rng.uniform(-15, 15)
            clip = np.clip(clip.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
        if rng.random() < 0.15:                     
            t = rng.randrange(len(clip))
            h, w = clip.shape[1:3]
            ch, cw = rng.randint(20, h // 4), rng.randint(20, w // 4)
            y, x = rng.randint(0, h - ch), rng.randint(0, w - cw)
            clip = clip.copy()
            clip[t, y:y + ch, x:x + cw] = rng.randint(0, 255)
        return np.ascontiguousarray(clip)

    def _to_tensor(self, clip):
        x = clip.astype(np.float32) / 255.0
        x = (x - MEAN) / STD
        return torch.from_numpy(x).permute(0, 3, 1, 2)  

    def __getitem__(self, i):
        vid = self.ids[i % len(self.ids)]
        rng = random.Random() if self.train else random.Random(i)
        meta = self.videos[vid]
        clip = self._read(self._sample_frames(meta["frames"], rng))

        if self.train:
            clip = self._spatial_aug(clip, rng)

        item = {"label": torch.tensor(float(meta["label"])),
                "video_id": vid, "source": meta["source"],
                "method": meta["method"]}

        if self.paired:
            item["clean"] = self._to_tensor(clip)
            deg, recipe = degrade_clip(clip, rng)
            item["degraded"] = self._to_tensor(deg)
            item["tier"] = recipe["tier"]
        else:
            if rng.random() < self.degrade_prob:
                clip, recipe = degrade_clip(clip, rng)
                item["tier"] = recipe["tier"]
            else:
                item["tier"] = "none"
            item["clean"] = self._to_tensor(clip)
        return item


def make_sampler(videos: dict):
    from torch.utils.data import WeightedRandomSampler
    ids = [k for k, v in videos.items() if len(v["frames"]) >= 2]
    n_by_label = defaultdict(int)
    n_by_source = defaultdict(int)
    for k in ids:
        n_by_label[videos[k]["label"]] += 1
        n_by_source[videos[k]["source"]] += 1
    weights = []
    for k in ids:
        v = videos[k]
        w = (1.0 / n_by_label[v["label"]]) * (1.0 / n_by_source[v["source"]] ** 0.5)
        weights.append(w)
    return WeightedRandomSampler(torch.tensor(weights, dtype=torch.double),
                                 num_samples=len(ids), replacement=True)
