from __future__ import annotations

import math
import subprocess
import wave

import cv2
import numpy as np

SR = 16000           
HOP = 160            
WIN = 400            



def extract_audio(video_path: str, seconds: float | None = None
                  ) -> tuple[np.ndarray | None, str | None]:
    cmd = ["ffmpeg", "-v", "quiet", "-i", video_path]
    if seconds:
        cmd += ["-t", str(seconds)]         
    cmd += ["-vn", "-ac", "1", "-ar", str(SR),
            "-acodec", "pcm_s16le", "-f", "wav", "-"]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=120)
    except FileNotFoundError:
        return None, "ffmpeg is not installed, so the audio track could not be read."
    except subprocess.TimeoutExpired:
        return None, "Audio decoding timed out."
    if proc.returncode != 0 or not proc.stdout:
        return None, "This file has no readable audio track."

    try:
        import io
        with wave.open(io.BytesIO(proc.stdout), "rb") as w:
            raw = w.readframes(w.getnframes())
            width = w.getsampwidth()
    except Exception:
        raw, width = proc.stdout[44:], 2

    dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(width, np.int16)
    audio = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    if audio.size == 0:
        return None, "The audio track decoded to nothing."
    audio /= float(np.iinfo(dtype).max)  # type: ignore
    return audio, None



def _frames(x, win=WIN, hop=HOP):
    n = 1 + max(0, (len(x) - win) // hop)
    if n <= 0:
        return np.empty((0, win), np.float32)
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]


def envelope(audio: np.ndarray) -> np.ndarray:
    fr = _frames(audio)
    if len(fr) == 0:
        return np.zeros(0, np.float32)
    rms = np.sqrt((fr ** 2).mean(axis=1) + 1e-12)
    db = 20 * np.log10(rms + 1e-9)
    lo, hi = np.percentile(db, 5), np.percentile(db, 99)
    return np.clip((db - lo) / max(1e-6, hi - lo), 0, 1)


def voice_indicators(audio: np.ndarray) -> dict:
    fr = _frames(audio)
    if len(fr) < 8:
        return {}
    win = np.hanning(fr.shape[1])
    spec = np.abs(np.fft.rfft(fr * win, axis=1)) + 1e-10
    freqs = np.fft.rfftfreq(fr.shape[1], 1 / SR)

    rms = np.sqrt((fr ** 2).mean(axis=1) + 1e-12)
    voiced = rms > max(1e-4, np.percentile(rms, 55))
    if voiced.sum() < 4:
        return {}

    high = spec[voiced][:, freqs > 4000]
    flatness = float(np.mean(
        np.exp(np.log(high).mean(axis=1)) / (high.mean(axis=1) + 1e-10)))

    floor_db = float(20 * np.log10(np.percentile(rms, 5) + 1e-9))

    env = envelope(audio)
    regularity = float(1.0 - min(1.0, np.std(np.diff(env)) * 6)) if len(env) > 2 else 0.0

    def squash(v, c, s):
        return 1 / (1 + math.exp(-max(-30, min(30, (v - c) / s))))

    score = (0.45 * squash(flatness, 0.42, 0.12) +
             0.30 * squash(floor_db, -52.0, 9.0) +
             0.25 * squash(regularity, 0.62, 0.14))

    return {
        "high_band_flatness": round(flatness, 4),
        "noise_floor_db": round(floor_db, 2),
        "envelope_regularity": round(regularity, 4),
        "synthetic_indicator": round(float(score), 4),
        "voiced_ratio": round(float(voiced.mean()), 3),
    }



def mouth_openness(crops: np.ndarray) -> np.ndarray:
    out = []
    for c in crops:
        h, w = c.shape[:2]
        roi = cv2.cvtColor(c[int(h * 0.58):int(h * 0.92),
                             int(w * 0.28):int(w * 0.72)],
                           cv2.COLOR_RGB2GRAY)
        if roi.size == 0:
            out.append(0.0)
            continue
        roi = cv2.GaussianBlur(roi, (0, 0), 1.2).astype(np.float32)
        rows = roi.mean(axis=1)
        depth = float(rows.max() - rows.min())
        dark = rows < (rows.mean() - 0.28 * (rows.std() + 1e-6))
        out.append((dark.sum() / len(rows)) * (depth / 255.0))
    arr = np.array(out, np.float32)
    if arr.std() > 1e-6:
        arr = (arr - arr.mean()) / arr.std()
    return arr


def lipsync_agreement(crops: np.ndarray, crop_times: list,
                      audio: np.ndarray) -> dict:
    if len(crops) < 8 or audio is None or len(audio) < SR // 2:
        return {}
    mouth = mouth_openness(crops)
    env = envelope(audio)
    if len(env) < 8 or np.std(mouth) < 1e-6:
        return {}

    env_t = np.arange(len(env)) * (HOP / SR)
    times = np.array([t if t is not None else i * 0.04
                      for i, t in enumerate(crop_times[:len(crops)])])
    env_at_frames = np.interp(times, env_t, env)
    if np.std(env_at_frames) < 1e-6:
        return {}
    env_at_frames = (env_at_frames - env_at_frames.mean()) / np.std(env_at_frames)

    best_r, best_lag = -1.0, 0
    span = min(10, len(mouth) // 4)
    for lag in range(-span, span + 1):
        a = mouth[max(0, lag):len(mouth) + min(0, lag)]
        b = env_at_frames[max(0, -lag):len(env_at_frames) + min(0, -lag)]
        n = min(len(a), len(b))
        if n < 6:
            continue
        r = float(np.corrcoef(a[:n], b[:n])[0, 1])
        if not math.isnan(r) and r > best_r:
            best_r, best_lag = r, lag

    if best_r < -1:
        return {}

    frame_dt = float(np.median(np.diff(times))) if len(times) > 1 else 0.04
    lag_s = best_lag * frame_dt

    if best_r >= 0.55:
        reading = "tight"
        note = ("Mouth movement and speech track each other closely. Note "
                "that talking-head generators drive the mouth from the "
                "audio, so a very tight match is not by itself reassuring.")
    elif best_r >= 0.25:
        reading = "loose"
        note = "Mouth movement and speech broadly agree, with some drift."
    else:
        reading = "mismatched"
        note = ("Mouth movement does not follow the speech. That fits "
                "dubbing, a re-voiced clip, or a face swap where the "
                "audio came from elsewhere.")

    return {
        "correlation": round(best_r, 4),
        "lag_seconds": round(lag_s, 3),
        "reading": reading,
        "note": note,
        "frames_used": int(len(mouth)),
    }



def analyse(video_path: str, crops: np.ndarray, crop_times: list,
            max_seconds: float = 60.0) -> dict:
    audio, error = extract_audio(video_path, seconds=max_seconds)
    if audio is None:
        return {"available": False, "reason": error,
                "voice": {}, "lipsync": {}, "notes": []}

    voice = voice_indicators(audio)
    sync = lipsync_agreement(crops, crop_times, audio) if len(crops) else {}

    notes = []
    if sync.get("note"):
        notes.append(sync["note"])
    if voice.get("synthetic_indicator", 0) >= 0.62:
        notes.append("The speech has an unusually flat high band and very "
                     "little noise floor, which is common in vocoded or "
                     "generated audio. This is an indicator, not a finding "
                     "— clean studio recordings can look similar.")
    if voice.get("voiced_ratio", 1) < 0.15:
        notes.append("There is very little speech in this clip, so the "
                     "audio checks carry almost no weight here.")

    return {"available": True, "reason": None, "duration_seconds":
            round(len(audio) / SR, 2), "voice": voice, "lipsync": sync,
            "notes": notes}
