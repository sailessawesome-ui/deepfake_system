"""Audio branch — the second half of "multimodal forensic analysis".

IR section 3.4.1, Functional Requirement 1: the system should examine
"audio features (e.g., voice cloning artifacts, lip-sync
synchronization)" alongside the visual ones.

Two things are computed here, and they answer different questions:

  Lip-sync agreement
      Correlation between how open the mouth is (measured from the face
      crops the visual pipeline already produced) and how loud the voice
      is (from the audio envelope). Real speech has the two moving
      together with a short, consistent lag. Puppet-style forgeries —
      Wav2Lip, SadTalker, most talking-head generators — drive the mouth
      from audio and usually score *high* here. Dubbed or re-voiced
      video scores low. So this is a mismatch detector, not a fake
      detector, and it is reported separately from the face score.

  Synthetic voice indicators
      Vocoded and diffusion-generated speech tends to have an unusually
      flat spectrum in the high band, unnaturally regular energy, and
      very little of the low-level noise floor a microphone always
      picks up. Three cheap statistics get at that: high-band spectral
      flatness, silence-floor level, and envelope regularity.

This is deliberately a set of measurements rather than a verdict. The
honest claim in your report is "the system measures audio-visual
consistency and surfaces synthetic-speech indicators", not "the system
detects cloned voices" — a trained speech-antispoofing model would be
needed for the second claim, and that is a project of its own.

Only numpy, OpenCV and ffmpeg are needed. No extra dependencies.
"""
from __future__ import annotations

import math
import subprocess
import wave

import cv2
import numpy as np

SR = 16000            # analysis sample rate
HOP = 160             # 10 ms hop
WIN = 400             # 25 ms window


# ------------------------------------------------------------------ decode

def extract_audio(video_path: str, seconds: float | None = None
                  ) -> tuple[np.ndarray | None, str | None]:
    """Decode the audio track to mono float32 at 16 kHz."""
    cmd = ["ffmpeg", "-v", "quiet", "-i", video_path]
    if seconds:
        cmd += ["-t", str(seconds)]          # must follow the input, not precede it
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
        # ffmpeg streams a wav header with an unknown size; fall back to
        # skipping the 44-byte header and reading raw PCM.
        raw, width = proc.stdout[44:], 2

    dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(width, np.int16)
    audio = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    if audio.size == 0:
        return None, "The audio track decoded to nothing."
    audio /= float(np.iinfo(dtype).max)  # type: ignore
    return audio, None


# ---------------------------------------------------------------- features

def _frames(x, win=WIN, hop=HOP):
    n = 1 + max(0, (len(x) - win) // hop)
    if n <= 0:
        return np.empty((0, win), np.float32)
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]


def envelope(audio: np.ndarray) -> np.ndarray:
    """Per-10ms RMS energy, in dB, normalised to 0..1."""
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

    # A real recording has a noise floor and an uneven spectrum. High
    # flatness + high floor + very smooth envelope is the synthetic look.
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


# ---------------------------------------------------------------- lip sync

def mouth_openness(crops: np.ndarray) -> np.ndarray:
    """Vertical dark-pixel extent in the lower-middle of each face crop.

    Deliberately landmark-free so it still works when MediaPipe and dlib
    are unavailable. The mouth is the darkest horizontal band in the
    lower third of an aligned face crop; its height tracks openness.
    """
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
    """Correlate mouth movement against the audio envelope."""
    if len(crops) < 8 or audio is None or len(audio) < SR // 2:
        return {}
    mouth = mouth_openness(crops)
    env = envelope(audio)
    if len(env) < 8 or np.std(mouth) < 1e-6:
        return {}

    # Resample the audio envelope onto the frame timestamps.
    env_t = np.arange(len(env)) * (HOP / SR)
    times = np.array([t if t is not None else i * 0.04
                      for i, t in enumerate(crop_times[:len(crops)])])
    env_at_frames = np.interp(times, env_t, env)
    if np.std(env_at_frames) < 1e-6:
        return {}
    env_at_frames = (env_at_frames - env_at_frames.mean()) / np.std(env_at_frames)

    # Best correlation across a plausible lag window (about +/- 0.4 s).
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


# ------------------------------------------------------------------ public

def analyse(video_path: str, crops: np.ndarray, crop_times: list,
            max_seconds: float = 60.0) -> dict:
    """Everything the audio branch contributes to one report."""
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
