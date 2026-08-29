"""One entry point the web layer talks to.

If a trained checkpoint is present it runs the CNN from models/net.py.
If not, it falls back to the signal-based baseline so the server is
usable immediately. Which engine ran is reported in every response and
shown in the interface — the fallback never masquerades as the model.
"""
from __future__ import annotations

import base64
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import AUDIO, DATA, INFER  # type: ignore # noqa: E402
from app import audio as audio_branch  # type: ignore # noqa: E402
from app import heuristic  # type: ignore # noqa: E402
from app import provenance  # type: ignore # noqa: E402


# ---------------------------------------------------------------- container

def probe(video_path: str) -> dict[str, Any]:
    import subprocess
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", "-show_streams", video_path],
            capture_output=True, text=True, timeout=30).stdout
        data = json.loads(out)
    except Exception:
        cap = cv2.VideoCapture(video_path)
        meta: dict[str, Any] = {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or None,
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None,
            "fps": round(cap.get(cv2.CAP_PROP_FPS) or 0, 2),
            "tags": {}, "has_audio": None, "ffprobe": False
        }
        cap.release()
        return meta
    v: dict[str, Any] = next((s for s in data.get("streams", [])
                              if s.get("codec_type") == "video"), {})
    fmt: dict[str, Any] = data.get("format", {})
    fps_raw = v.get("avg_frame_rate", "0/1")
    try:
        num, den = fps_raw.split("/")
        fps: float | None = round(float(num) / max(1e-6, float(den)), 2)
    except Exception:
        fps = None
    return {
        "codec": v.get("codec_name"), "profile": v.get("profile"),
        "width": v.get("width"), "height": v.get("height"), "fps": fps,
        "duration": fmt.get("duration"), "bit_rate": fmt.get("bit_rate"),
        "tags": fmt.get("tags", {}) or {},
        "has_audio": any(s.get("codec_type") == "audio"
                         for s in data.get("streams", [])),
        "ffprobe": True
    }


def messenger_flags(filename: str, meta: dict[str, Any]) -> dict[str, Any]:
    import re
    flags: dict[str, Any] = {
        "whatsapp_filename": bool(re.match(r"(VID|IMG)-\d{8}-WA\d{4}",
                                           filename or "")),
        "telegram_filename": bool(re.match(r"video_\d{4}-\d{2}-\d{2}",
                                           filename or "")),
        "stripped_metadata": False,
        "low_bitrate": False,
        "capped_resolution": False,
    }
    tags = meta.get("tags", {}) or {}
    if meta.get("ffprobe"):
        flags["stripped_metadata"] = not any(
            k.lower() in {"com.apple.quicktime.model", "encoder", "model",
                          "make"} for k in tags)
    br = meta.get("bit_rate")
    if br:
        try:
            flags["low_bitrate"] = int(br) < 1_600_000
        except (TypeError, ValueError):
            pass
    h = meta.get("height")
    if h:
        flags["capped_resolution"] = int(h) in (360, 480, 540, 720)
    flags["likely_recompressed"] = sum(
        bool(v) for k, v in flags.items() if k != "likely_recompressed") >= 2
    return flags


# -------------------------------------------------------------- face frames

def _haar_path() -> str:
    data_dir = getattr(cv2, "data", None)
    haarcascades = getattr(data_dir, "haarcascades", "") if data_dir else ""
    return str(haarcascades) + "haarcascade_frontalface_default.xml"


class FaceExtractor:
    def __init__(self, size: int | None = None):
        self.size = size or DATA.img_size
        self.backend = "haar"
        self.det: Any = None
        self._haar: cv2.CascadeClassifier | None = None
        try:
            import torch  # type: ignore # noqa: F401
            from facenet_pytorch import MTCNN  # type: ignore
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            self.det = MTCNN(keep_all=True, device=dev, post_process=False)
            self.backend = "mtcnn"
        except Exception:
            try:
                import mediapipe as mp  # type: ignore
                self.det = mp.solutions.face_detection.FaceDetection(
                    model_selection=1, min_detection_confidence=0.5)
                self.backend = "mediapipe"
            except Exception:
                self.det = cv2.CascadeClassifier(_haar_path())
                self.backend = "haar"

    def _boxes(self, rgb: np.ndarray, min_conf: float = 0.60):
        h, w = rgb.shape[:2]
        if self.backend == "mtcnn" and self.det is not None:
            try:
                boxes, probs = self.det.detect(rgb)
                if boxes is not None:
                    valid = [b for b, p in zip(boxes, probs)
                             if p is not None and p >= min_conf]
                    if valid:
                        return valid
            except Exception:
                pass
        if self.backend == "mediapipe" and self.det is not None:
            try:
                res = self.det.process(rgb)
                if res and res.detections:
                    out = []
                    for d in res.detections:
                        r = d.location_data.relative_bounding_box
                        out.append(np.array([r.xmin * w, r.ymin * h,
                                             (r.xmin + r.width) * w,
                                             (r.ymin + r.height) * h]))
                    if out:
                        return out
            except Exception:
                pass
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        if self._haar is None:
            self._haar = cv2.CascadeClassifier(_haar_path())
        faces = self._haar.detectMultiScale(gray, 1.1, 4, minSize=(36, 36))
        return [np.array([x, y, x + fw, y + fh]) for x, y, fw, fh in faces]

    def _crop(self, rgb: np.ndarray, box: Any, margin: float = 0.32):
        h, w = rgb.shape[:2]
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        side = max(x2 - x1, y2 - y1) * (1 + margin)
        x1, y1 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
        x2, y2 = int(min(w, cx + side / 2)), int(min(h, cy + side / 2))
        if x2 - x1 < 24 or y2 - y1 < 24:
            return None
        return cv2.resize(rgb[y1:y2, x1:x2], (self.size, self.size),
                          interpolation=cv2.INTER_AREA)

    def extract(self, video_path: str, max_samples: int = 48):
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        stride = max(1, total // max_samples) if total else 5
        crops, times, prev = [], [], None
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % stride == 0:
                rgb = frame[:, :, ::-1]
                boxes = self._boxes(rgb)
                if boxes:
                    box = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
                    box = box if prev is None else 0.65 * box + 0.35 * prev
                    prev = box
                    c = self._crop(rgb, box)
                    if c is not None:
                        crops.append(c)
                        times.append(round(i / max(1e-6, fps), 2))
                if len(crops) >= max_samples:
                    break
            i += 1
        cap.release()
        return (np.array(crops) if crops else np.empty((0,))), times, total, fps


def thumb(crop: np.ndarray, size: int = 104, quality: int = 72) -> str | None:
    small = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", small[:, :, ::-1],
                           [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


# ------------------------------------------------------------------- engine

class Engine:
    def __init__(self, checkpoint: str | None = None):
        self.faces = FaceExtractor()
        self.mode = "heuristic"
        self.model: Any = None
        self.device = "cpu"
        self.threshold = 0.50
        self.temperature = 1.0
        self.backbone: str | None = None
        self.clip_len = DATA.clip_len
        self.model_version = "heuristic-1.0"
        self._load(checkpoint or INFER.checkpoint)

    def _load(self, path: str | Path | None):
        p = Path(path) if path else None
        if not p or not p.exists():
            for cand in [
                ROOT / "runs" / "v1" / "best.pt",
                ROOT / "runs" / "best.pt",
                ROOT.parent / "deepfake_system" / "runs" / "v1" / "best.pt",
                Path("runs/v1/best.pt"),
                Path("runs/best.pt")
            ]:
                if cand.exists():
                    p = cand.resolve()
                    break
        if not p or not p.exists():
            print(f"[engine] no checkpoint found at {path} - using classical baseline.")
            return
        try:
            import torch  # type: ignore
            from config import MODEL  # type: ignore
            from models.net import build_model  # type: ignore
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            ck = torch.load(p, map_location=self.device)
            cfg = ck.get("config", {})
            MODEL.backbone = cfg.get("backbone", MODEL.backbone)
            MODEL.use_srm = cfg.get("use_srm", MODEL.use_srm)
            MODEL.temporal = cfg.get("temporal", MODEL.temporal)
            MODEL.pretrained = False
            self.clip_len = cfg.get("clip_len", DATA.clip_len)
            # Crops must be served at the size the model was trained on.
            # FaceExtractor was built from DATA.img_size before the
            # checkpoint was read, so correct it here.
            self.faces.size = cfg.get("img_size", DATA.img_size)
            self.model = build_model(MODEL).to(self.device).eval()
            self.model.load_state_dict(ck["model"])
            self.backbone = MODEL.backbone
            self.mode = "model"
            val = ck.get("val", {}) or {}
            self.model_version = (
                f"{MODEL.backbone}@{p.name}"
                f"+f1={val.get('f1', 0):.4f}" if val else f"{MODEL.backbone}@{p.name}")
            cal = Path(INFER.calibration_file)
            if cal.exists():
                c = json.loads(cal.read_text())
                self.threshold = c.get("threshold", 0.5)
                self.temperature = c.get("temperature", 1.0)
        except Exception as exc:                       # pragma: no cover
            print(f"[engine] checkpoint present but did not load: {exc}")
            self.mode = "heuristic"

    def status(self) -> dict[str, Any]:
        return {"mode": self.mode, "device": self.device,
                "backbone": self.backbone, "threshold": self.threshold,
                "face_backend": self.faces.backend,
                "calibrated": self.mode == "model" and self.temperature != 1.0,
                "audio_branch": AUDIO.enabled,
                "strict_binary": INFER.strict_binary,
                "model_version": self.model_version}

    # ---------------------------------------------------------- model path
    def _model_scores(self, crops: np.ndarray):
        import torch  # type: ignore
        from data.dataset import MEAN, STD  # type: ignore
        T = self.clip_len
        n = len(crops)
        if n < T:
            crops = np.concatenate([crops, np.repeat(crops[-1:], T - n, 0)])
            n = len(crops)
        starts = np.linspace(0, n - T, min(INFER.clips_per_video,
                                           max(1, n - T + 1))).astype(int)
        logits, frame_scores = [], np.zeros(n)
        counts = np.zeros(n) + 1e-6
        if self.model is None:
            return np.array([]), []
        with torch.no_grad():
            for s in sorted(set(starts.tolist())):
                clip = crops[s:s + T].astype(np.float32) / 255.0
                clip = (clip - MEAN) / STD
                x = torch.from_numpy(clip).permute(0, 3, 1, 2).unsqueeze(0)
                x = x.to(self.device)
                assert self.model is not None
                logit, frame_logit = self.model(x)
                if INFER.tta:
                    # Fast 2-View Flip TTA for variance reduction & rapid CPU inference
                    x_flip = torch.flip(x, dims=[-1])
                    logit_f, frame_f = self.model(x_flip)
                    logit = 0.5 * (logit + logit_f)
                    frame_logit = 0.5 * (frame_logit + frame_f)
                logits.append(float(logit.item()) / self.temperature)
                fs = torch.sigmoid(frame_logit[0].float()).cpu().numpy()
                frame_scores[s:s + T] += fs
                counts[s:s + T] += 1
        return np.array(logits), (frame_scores / counts).tolist()

    # ------------------------------------------------------------- analyse
    def analyse(self, video_path: str, filename: str = "") -> dict[str, Any]:
        t0 = time.time()
        timings: dict[str, float] = {}

        meta = probe(video_path)
        prov = messenger_flags(filename, meta)
        timings["probe"] = round(time.time() - t0, 3)

        t = time.time()
        credentials = provenance.inspect(video_path, filename)
        timings["provenance"] = round(time.time() - t, 3)

        t = time.time()
        crops, times, total_frames, fps = self.faces.extract(video_path)
        timings["face_extraction"] = round(time.time() - t, 3)

        notes: list[str] = []
        features: dict[str, Any] = {}

        if len(crops) == 0:
            return {"label": "no_face", "probability": None,
                    "confidence_band": None, "threshold": self.threshold,
                    "engine": self.mode, "backbone": self.backbone,
                    "faces_found": 0, "clips_scored": 0, "frames": [],
                    "total_frames": total_frames, "fps": round(float(fps), 2),
                    "provenance": prov, "media": meta, "features": {},
                    "elapsed": round(time.time() - t0, 2),
                    "notes": ["No face was found in this video. There is "
                              "nothing for a facial-manipulation detector to "
                              "examine — try a clip where a face is visible "
                              "and reasonably large."]}

        t = time.time()
        if self.mode == "model":
            logits, frame_scores = self._model_scores(crops)
            probs = np.sort(1 / (1 + np.exp(-logits)))
            if INFER.aggregation == "topk":
                k = max(1, int(len(probs) * INFER.trim_fraction))
                prob = float(probs[-k:].mean())
            elif INFER.aggregation == "mean" or len(probs) < 4:
                prob = float(probs.mean())
            else:
                cut = int(len(probs) * INFER.trim_fraction)
                core = probs[cut:len(probs) - cut] if len(probs) - 2 * cut > 0 else probs
                prob = float(core.mean())
            spread = float(probs.std())
            clips_scored = len(logits)
        else:
            result = heuristic.score_clip(crops)
            prob = float(result["score"])
            features = result["features"]
            frame_scores = heuristic.per_frame_scores(crops)
            spread = float(np.std(frame_scores)) if frame_scores else 0.15
            clips_scored = 1
            notes.append("Running the classical baseline, not the trained "
                         "network. Scores here are uncalibrated — treat them "
                         "as an indication, not a verdict. Train a checkpoint "
                         "and drop it at runs/v1/best.pt to switch engines.")
        timings["visual_scoring"] = round(time.time() - t, 3)

        # ---- audio branch (IR 3.4.1 FR1: multimodal analysis) ----------
        if AUDIO.enabled:
            t = time.time()
            audio_report = audio_branch.analyse(video_path, crops, times,
                                                AUDIO.max_seconds)
            timings["audio"] = round(time.time() - t, 3)
        else:
            audio_report = {"available": False,
                            "reason": "The audio branch is switched off.",
                            "voice": {}, "lipsync": {}, "notes": []}

        band = 1.96 * spread / max(1.0, math.sqrt(max(1, clips_scored)))
        if self.mode == "heuristic":
            band = max(band, 0.18)
        if prov["likely_recompressed"]:
            band += 0.06
            notes.append("This file carries the signature of a messenger "
                         "re-upload — re-encoded, low bitrate, metadata "
                         "stripped. Compression destroys the traces this "
                         "detector reads, so the margin is widened.")
        if len(crops) < 12:
            band += 0.05
            notes.append(f"Only {len(crops)} usable face crops were found. "
                         "Short or intermittent face coverage weakens the "
                         "temporal signal.")
        notes.extend(audio_report.get("notes", []))
        notes.extend(credentials.get("notes", []))
        if audio_report.get("available") is False and audio_report.get("reason"):
            notes.append(audio_report["reason"])
        # Multimodal Audio-Visual Fusion (IR 3.4.1 FR1)
        if audio_report.get("available"):
            lip_reading = audio_report.get("lipsync", {}).get("reading")
            voice_synth = float(audio_report.get("voice", {}).get("synthetic_indicator", 0.0) or 0.0)
            high_flat = float(audio_report.get("voice", {}).get("high_band_flatness", 0.0) or 0.0)

            audio_anomaly = 0.0
            if lip_reading == "mismatched":
                audio_anomaly = max(audio_anomaly, 0.75)
                band += 0.04
            if voice_synth >= 0.65:
                audio_anomaly = max(audio_anomaly, voice_synth)
                band += 0.04
            if high_flat >= 0.75:
                audio_anomaly = max(audio_anomaly, high_flat * 0.7)

            if audio_anomaly > 0.5:
                # Smooth continuous fusion: 85% neural visual model, 15% audio anomaly signal
                prob = float(0.85 * prob + 0.15 * audio_anomaly)

        lo, hi = max(0.0, prob - band), min(1.0, prob + band)
        straddles = lo <= self.threshold <= hi
        if straddles and not INFER.strict_binary:
            label = "inconclusive"
            notes.insert(0, "The decision threshold falls inside the "
                            "confidence margin. This one needs a human — do "
                            "not record it as either result.")
        else:
            label = "manipulated" if prob >= self.threshold else "authentic"
            if straddles:
                notes.insert(0, "Binary mode is on, so a call was forced, but "
                                "the margin still covers the decision line. "
                                "The confidence here is low.")

        step = max(1, len(crops) // 32)
        calibrated_frame_scores = []
        if len(frame_scores) > 0:
            raw_arr = np.array(frame_scores)
            if raw_arr.max() > raw_arr.min():
                norm_var = (raw_arr - raw_arr.mean()) / (raw_arr.std() + 1e-6)
                scaled = prob + 0.08 * norm_var
                calibrated_frame_scores = np.clip(scaled, 0.0, 1.0).tolist()
            else:
                calibrated_frame_scores = [prob] * len(frame_scores)
        else:
            calibrated_frame_scores = [prob] * len(crops)

        frames = []
        for i in range(0, len(crops), step):
            frames.append({
                "t": times[i] if i < len(times) else None,
                "score": round(float(calibrated_frame_scores[i]), 4)
                if i < len(calibrated_frame_scores) else None,
                "thumb": thumb(crops[i]),
            })

        return {"label": label, "probability": round(prob, 4),
                "confidence_band": [round(lo, 4), round(hi, 4)],
                "threshold": self.threshold, "engine": self.mode,
                "backbone": self.backbone, "faces_found": len(crops),
                "clips_scored": clips_scored, "frames": frames,
                "provenance": prov, "media": meta, "features": features,
                "content_credentials": credentials, "audio": audio_report,
                "model_version": self.model_version,
                "margin_straddles_threshold": bool(straddles),
                "total_frames": total_frames, "fps": round(float(fps), 2),
                "timings": timings,
                "elapsed": round(time.time() - t0, 2), "notes": notes}
