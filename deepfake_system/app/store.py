from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app import tables

_STRIP = {"frames", "thumb", "cam", "media_path", "tmp_path"}


def _clean(value):
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items() if k not in _STRIP}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def build_record(result: dict, filename: str, model_version: str,
                 user_id: str = "", email: str = "") -> dict:
    now = datetime.now(timezone.utc)
    analysis_id = uuid.uuid4().hex[:12]
    label = result.get("label")
    probability = result.get("probability")

    verdict = {"manipulated": "The Video is Fake",
               "authentic": "The Video is Authentic"}.get(
                   str(label), "Inconclusive")

    return {
        "user_id": user_id or "anonymous",
        "analysis_id": analysis_id,
        "report_id": analysis_id,
        "detection_id": analysis_id,
        "classification": label,
        "confidence_score": probability,
        "detection_status": ("completed" if probability is not None
                             else "no_face" if label == "no_face"
                             else "inconclusive"),
        "model_version": model_version,
        "processing_timestamp": int(now.timestamp()),
        "created_at": now.timestamp(),
        "created_at_iso": now.isoformat(),
        "filename": filename,
        "file_sha256": result.get("evidence_sha256"),
        "verdict": verdict,
        "fake_probability": probability,
        "confidence": result.get("confidence"),
        "mode": result.get("engine") or "forensic",
        "user_email": email or None,
        "evidence_sha256": result.get("evidence_sha256"),
        "label": label,
        "probability": probability,
        "confidence_band": result.get("confidence_band"),
        "threshold": result.get("threshold"),
        "engine": result.get("engine"),
        "faces_found": result.get("faces_found"),
        "clips_scored": result.get("clips_scored"),
        "provenance": result.get("provenance"),
        "explanation": result.get("explanation"),
        "audio_available": (result.get("audio") or {}).get("available"),
        "lipsync": (result.get("audio") or {}).get("lipsync"),
        "voice": (result.get("audio") or {}).get("voice"),
        "media": {k: v for k, v in (result.get("media") or {}).items()
                  if k in ("codec", "width", "height", "fps", "duration",
                           "bit_rate")},
        "timings": result.get("timings"),
        "notes": result.get("notes"),
        "size_bytes": result.get("size_bytes"),
        "backbone": result.get("backbone"),
        "elapsed": result.get("elapsed"),
    }


class ReportStore:
    def __init__(self, local_path: Path, prefer_dynamo: bool = True):
        self.local_path = Path(local_path)
        self.local_path.parent.mkdir(parents=True, exist_ok=True)
        self.table: Any = tables.get("analyses",
                                     force_local=not prefer_dynamo)
        self.backend = self.table.backend
        self.error: str | None = None

    @property
    def durable(self) -> bool:
        return getattr(self.table, "durable", False)

    def save(self, record: dict) -> dict:
        record = {k: v for k, v in record.items() if v is not None}
        rid = record.get("analysis_id", "")
        if self.durable and self.table.put(_clean(record)):
            return {"stored": True, "backend": self.backend, "report_id": rid}
        if self.durable:
            self.error = getattr(self.table, "error", None)
        try:
            with self.local_path.open("a", encoding="utf8") as f:
                f.write(json.dumps(_clean(record), default=str) + "\n")
            return {"stored": True, "backend": "local", "report_id": rid,
                    "error": self.error}
        except OSError as exc:
            return {"stored": False, "backend": self.backend,
                    "error": str(exc)[:160]}

    def _local_rows(self, limit: int) -> list:
        if not self.local_path.exists():
            return []
        try:
            lines = self.local_path.read_text(
                encoding="utf8").strip().splitlines()
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

    @staticmethod
    def _sort_newest(rows: list) -> list:
        return sorted(rows, key=lambda r: float(r.get("created_at", 0) or 0),
                      reverse=True)

    def recent(self, limit: int = 25) -> list:
        if self.durable:
            return self._sort_newest(self.table.scan(limit * 3))[:limit]
        return self._local_rows(limit)

    def for_user(self, user_id: str, limit: int = 25) -> list:
        if not user_id:
            return self.recent(limit)
        if self.durable:
            rows = self.table.query(key_value=user_id, Limit=max(limit, 1))
            return self._sort_newest(rows)[:limit]
        return [r for r in self._local_rows(limit * 4)
                if r.get("user_id") == user_id][:limit]

    def find_by_hash(self, sha256: str, user_id: str = "") -> list:
        pool = self.for_user(user_id, 200) if user_id else self.recent(200)
        return [r for r in pool
                if sha256 in (r.get("file_sha256"), r.get("evidence_sha256"))]

    def status(self) -> dict:
        st = dict(self.table.status())
        st["local_fallback"] = str(self.local_path)
        if self.error:
            st["last_write_error"] = self.error
        return st
