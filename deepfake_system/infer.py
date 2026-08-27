"""End-to-end inference on a real video file.

    python infer.py --video VID-20260825-WA0007.mp4

Pipeline: probe container -> decode frames -> detect and track faces ->
build clips -> run the model on several clips -> aggregate -> calibrate ->
return a verdict with a per-frame timeline and the reason the confidence
band was widened (if it was).
"""
from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np
import torch

from config import DATA, INFER, MODEL
from data.degrade import looks_like_messenger_upload
from data.dataset import MEAN, STD
from models.net import build_model


# ---------------------------------------------------------------- container

def probe(video_path: str) -> dict:
    """ffprobe metadata. Absence of metadata is itself informative."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", "-show_streams", video_path],
            capture_output=True, text=True, timeout=30).stdout
        data = json.loads(out)
    except Exception:
        return {}
    v = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), {})
    fmt = data.get("format", {})
    return {
        "codec": v.get("codec_name"), "profile": v.get("profile"),
        "width": v.get("width"), "height": v.get("height"),
        "fps": v.get("avg_frame_rate"), "duration": fmt.get("duration"),
        "bit_rate": fmt.get("bit_rate"), "tags": fmt.get("tags", {}),
        "has_audio": any(s.get("codec_type") == "audio"
                         for s in data.get("streams", [])),
    }


# -------------------------------------------------------------- face crops

class FaceExtractor:
    """MTCNN if available, MediaPipe next, Haar as the floor. Keeps the
    largest face and smooths the box over time so crops do not jitter."""

    def __init__(self, device="cpu", size=None):
        self.size = size or DATA.img_size
        self.device = device
        self.backend = None
        try:
            from facenet_pytorch import MTCNN
            self.det = MTCNN(keep_all=True, device=device, post_process=False)
            self.backend = "mtcnn"
        except Exception:
            try:
                import mediapipe as mp
                self.det = mp.solutions.face_detection.FaceDetection(
                    model_selection=1, min_detection_confidence=0.5)
                self.backend = "mediapipe"
            except Exception:
                self.det = cv2.CascadeClassifier(
                    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                self.backend = "haar"

    def _boxes(self, frame_rgb):
        h, w = frame_rgb.shape[:2]
        if self.backend == "mtcnn":
            boxes, probs = self.det.detect(frame_rgb)
            if boxes is None:
                return []
            return [(b, float(p)) for b, p in zip(boxes, probs)
                    if p is not None and p >= INFER.min_face_conf]
        if self.backend == "mediapipe":
            res = self.det.process(frame_rgb)
            if not res.detections:
                return []
            out = []
            for d in res.detections:
                r = d.location_data.relative_bounding_box
                out.append((np.array([r.xmin * w, r.ymin * h,
                                      (r.xmin + r.width) * w,
                                      (r.ymin + r.height) * h]),
                            float(d.score[0])))
            return out
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        faces = self.det.detectMultiScale(gray, 1.1, 5)
        return [(np.array([x, y, x + fw, y + fh]), 0.9) for x, y, fw, fh in faces]

    def crop(self, frame_rgb, box, margin=0.30):
        h, w = frame_rgb.shape[:2]
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        side = max(x2 - x1, y2 - y1) * (1 + margin)
        x1, y1 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
        x2, y2 = int(min(w, cx + side / 2)), int(min(h, cy + side / 2))
        if x2 - x1 < 16 or y2 - y1 < 16:
            return None
        return cv2.resize(frame_rgb[y1:y2, x1:x2], (self.size, self.size),
                          interpolation=cv2.INTER_AREA)

    def extract(self, video_path, max_frames=320, stride=None):
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or max_frames
        stride = stride or max(1, total // max_frames)
        crops, indices, prev = [], [], None
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % stride == 0:
                rgb = frame[:, :, ::-1]
                dets = self._boxes(rgb)
                if dets:
                    box = max(dets, key=lambda d: (d[0][2] - d[0][0]) *
                              (d[0][3] - d[0][1]))[0]
                    box = box if prev is None else 0.7 * box + 0.3 * prev
                    prev = box
                    c = self.crop(rgb, box)
                    if c is not None:
                        crops.append(c)
                        indices.append(i)
            i += 1
        cap.release()
        return np.array(crops) if crops else np.empty((0,)), indices


# ------------------------------------------------------------------ verdict

@dataclass
class Verdict:
    label: str
    probability: float
    confidence_band: tuple
    threshold: float
    faces_found: int
    clips_scored: int
    provenance: dict
    frame_timeline: list
    notes: list


class Detector:
    def __init__(self, checkpoint=None, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(checkpoint or INFER.checkpoint, map_location=self.device)
        cfg = ck.get("config", {})
        MODEL.backbone = cfg.get("backbone", MODEL.backbone)
        MODEL.use_srm = cfg.get("use_srm", MODEL.use_srm)
        MODEL.temporal = cfg.get("temporal", MODEL.temporal)
        MODEL.pretrained = False
        self.clip_len = cfg.get("clip_len", DATA.clip_len)
        self.model = build_model(MODEL).to(self.device).eval()
        self.model.load_state_dict(ck["model"])
        self.faces = FaceExtractor(self.device, cfg.get("img_size", DATA.img_size))
        cal = Path(INFER.calibration_file)
        c = json.loads(cal.read_text()) if cal.exists() else {}
        self.temperature = c.get("temperature", 1.0)
        self.threshold = c.get("threshold", 0.5)

    def _clips(self, crops):
        n = len(crops)
        T = self.clip_len
        if n < T:
            pad = np.repeat(crops[-1:], T - n, axis=0)
            return [np.concatenate([crops, pad])]
        starts = np.linspace(0, n - T, min(INFER.clips_per_video,
                                           max(1, n - T + 1))).astype(int)
        return [crops[s:s + T] for s in sorted(set(starts.tolist()))]

    def _tensor(self, clip):
        x = clip.astype(np.float32) / 255.0
        x = (x - MEAN) / STD
        return torch.from_numpy(x).permute(0, 3, 1, 2).unsqueeze(0)

    @torch.no_grad()
    def predict(self, video_path: str) -> Verdict:
        notes = []
        meta = probe(video_path)
        prov = looks_like_messenger_upload(video_path, meta)

        crops, indices = self.faces.extract(video_path)
        if len(crops) == 0:
            return Verdict("no_face", 0.0, (0.0, 0.0), self.threshold, 0, 0,
                           prov, [], ["No face was detected — nothing to score."])
        if len(crops) < self.clip_len:
            notes.append(f"Only {len(crops)} usable face crops; the temporal "
                         "signal is weak on clips this short.")

        logits, frame_scores = [], []
        for clip in self._clips(crops):
            x = self._tensor(clip).to(self.device)
            logit, frame_logit, attn = self.model(x, return_attention=True)
            logits.append(float(logit.item()) / self.temperature)
            frame_scores += torch.sigmoid(frame_logit[0].float()).cpu().tolist()

        arr = np.sort(1 / (1 + np.exp(-np.array(logits))))
        cut = int(len(arr) * INFER.trim_fraction)
        core = arr[cut:len(arr) - cut] if len(arr) - 2 * cut > 0 else arr
        prob = float(core.mean())
        spread = float(arr.std())

        band = 1.96 * spread / max(1, np.sqrt(len(arr)))
        if prov.get("likely_recompressed"):
            band += 0.06
            notes.append("This file looks like a messenger re-upload "
                         "(re-encoded, metadata stripped). Compression removes "
                         "evidence, so the confidence band is widened.")
        if not meta.get("has_audio"):
            notes.append("No audio track — lip-sync and voice checks are "
                         "unavailable for this file.")
        lo, hi = max(0.0, prob - band), min(1.0, prob + band)

        if lo <= self.threshold <= hi:
            label = "inconclusive"
            notes.append("The decision threshold sits inside the confidence "
                         "band. Treat this as 'needs a human', not as a clear "
                         "result.")
        else:
            label = "manipulated" if prob >= self.threshold else "authentic"

        timeline = [{"frame": indices[i] if i < len(indices) else i,
                     "score": round(s, 4)}
                    for i, s in enumerate(frame_scores[:len(indices)])]

        return Verdict(label, round(prob, 4), (round(lo, 4), round(hi, 4)),
                       self.threshold, len(crops), len(logits), prov,
                       timeline, notes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--checkpoint", default=str(INFER.checkpoint))
    args = ap.parse_args()
    v = Detector(args.checkpoint).predict(args.video)
    print(json.dumps(asdict(v), indent=2)[:4000])


if __name__ == "__main__":
    main()
