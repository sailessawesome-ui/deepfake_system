"""Report store — Local zero-retention storage per ISO/IEC 27037 & NIST SP 800-86.

Zero retention means the video itself is never stored. The forensic verdict,
cryptographic SHA-256 evidence digest, temporal timeline, and model version
are recorded in local JSONL storage (./reports/reports.jsonl).
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Fields that must never reach persistent storage
_STRIP = {"frames", "thumb", "media_path", "tmp_path"}


def _clean(value):
    """Storage-safe copy: evidence media and raw frames stripped."""
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items() if k not in _STRIP}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def build_record(result: dict, filename: str, model_version: str,
                 user_id: str = "SailessRaj", email: str = "sailessraj149@gmail.com") -> dict:
    """Subset of forensic result safe and appropriate for ISO 27037 evidence audit."""
    now = datetime.now(timezone.utc).isoformat()
    return {
        "report_id": str(uuid.uuid4()),
        "created_at": now,
        "user_id": user_id or "SailessRaj",
        "user_email": email or "sailessraj149@gmail.com",
        "filename": filename,
        "evidence_sha256": result.get("evidence_sha256"),
        "label": result.get("label"),
        "probability": result.get("probability"),
        "confidence_band": result.get("confidence_band"),
        "threshold": result.get("threshold"),
        "engine": result.get("engine"),
        "model_version": model_version,
        "faces_found": result.get("faces_found"),
        "clips_scored": result.get("clips_scored"),
        "provenance": result.get("provenance"),
        "audio_available": (result.get("audio") or {}).get("available"),
        "lipsync": (result.get("audio") or {}).get("lipsync"),
        "media": {k: v for k, v in (result.get("media") or {}).items()
                  if k in ("codec", "width", "height", "fps", "duration", "bit_rate")},
        "timings": result.get("timings"),
    }


class ReportStore:
    def __init__(self, local_path: Path):
        self.local_path = Path(local_path)
        self.backend = "local"
        self.local_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def durable(self) -> bool:
        return True

    def save(self, record: dict) -> dict:
        record = {k: v for k, v in record.items() if v is not None}
        try:
            with self.local_path.open("a", encoding="utf8") as f:
                f.write(json.dumps(_clean(record), default=str) + "\n")
            return {"stored": True, "backend": "local", "report_id": record["report_id"]}
        except OSError as exc:
            return {"stored": False, "backend": "local", "error": str(exc)[:160]}

    def _local_rows(self, limit: int) -> list:
        if not self.local_path.exists():
            return []
        try:
            lines = self.local_path.read_text(encoding="utf8").strip().splitlines()
        except OSError:
            return []
        out = []
        for line in reversed(lines[-limit * 3:]):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if len(out) >= limit:
                break
        return out

    def recent(self, limit: int = 25) -> list:
        return self._local_rows(limit)

    def for_user(self, user_id: str, limit: int = 25) -> list:
        if not user_id:
            return self.recent(limit)
        return [r for r in self._local_rows(limit * 4)
                if r.get("user_id") == user_id][:limit]

    def find_by_hash(self, sha256: str, user_id: str = "") -> list:
        pool = self.for_user(user_id, 200) if user_id else self.recent(200)
        return [r for r in pool if r.get("evidence_sha256") == sha256]

    def status(self) -> dict:
        return {
            "backend": "local",
            "path": str(self.local_path),
            "durable": True,
            "error": None
        }
