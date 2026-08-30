"""Detection results — DynamoDB `dfd_analyses`, per IR 2.4.4 and Table 16.

Zero retention (IR 3.4.1) means the *video* is never stored. It does not
mean the verdict cannot be. What is written here is the finding, the
SHA-256 evidence digest and the model version — never the media, never the
face thumbnails.

The table is keyed `user_id` (HASH) + `analysis_id` (RANGE), which is why
"my past findings" is a Query on one partition rather than a Scan of the
whole table.

Field names follow the 39 records already in the table, which carry both
the original short names (`verdict`, `fake_probability`, `file_sha256`)
and the IR Table 16 names (`classification`, `confidence_score`,
`detection_id`, `model_version`, `processing_timestamp`). Both are written
so either reader works, and so Table 16 conformance is visible in the data
itself rather than only in the report.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app import tables

# Fields that must never reach persistent storage.
_STRIP = {"frames", "thumb", "cam", "media_path", "tmp_path"}


def _clean(value):
    """Storage-safe copy: evidence media and raw frames stripped."""
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items() if k not in _STRIP}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def build_record(result: dict, filename: str, model_version: str,
                 user_id: str = "", email: str = "") -> dict:
    """The subset of a result that is safe and useful to keep.

    `user_id` is the table's partition key and IR Table 16's second
    attribute. It defaults to "anonymous" rather than to any real person:
    stamping an unattributed analysis with a developer's identity would
    put false attribution into what is meant to be evidence.
    """
    now = datetime.now(timezone.utc)
    analysis_id = uuid.uuid4().hex[:12]
    label = result.get("label")
    probability = result.get("probability")

    # IR 3.4.1 asks for a definite tag; the engine's third state is kept
    # rather than forced, and spelled out here for the stored record.
    verdict = {"manipulated": "The Video is Fake",
               "authentic": "The Video is Authentic"}.get(
                   str(label), "Inconclusive")

    return {
        # --- keys -------------------------------------------------------
        "user_id": user_id or "anonymous",
        "analysis_id": analysis_id,
        # The API response and the front end both call this `report_id`.
        # It is the same value as the sort key; carried as an alias so the
        # rename to `analysis_id` (which the table's schema forced) does
        # not ripple out into the server and the interface.
        "report_id": analysis_id,
        # --- IR Table 16 names -----------------------------------------
        "detection_id": analysis_id,
        "classification": label,
        # Left absent rather than null when the engine never produced a
        # score — a null in an evidence record reads as a measurement that
        # failed, when in fact none was applicable.
        "confidence_score": probability,
        # "completed" would be misleading for a clip where no face was
        # found: the run finished, but nothing was scored. An examiner
        # reading the row should be able to tell those apart.
        "detection_status": ("completed" if probability is not None
                             else "no_face" if label == "no_face"
                             else "inconclusive"),
        "model_version": model_version,
        "processing_timestamp": int(now.timestamp()),
        # --- names the existing rows use -------------------------------
        "created_at": now.timestamp(),
        "created_at_iso": now.isoformat(),
        "filename": filename,
        "file_sha256": result.get("evidence_sha256"),
        "verdict": verdict,
        "fake_probability": probability,
        "confidence": result.get("confidence"),
        "mode": result.get("engine") or "forensic",
        # --- everything else worth keeping -----------------------------
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
        # The Grad-CAM narrative, not the heatmap images. Worth keeping in
        # the evidence record: "the model looked at the mid-face" is part
        # of why the verdict was reached, and it is a few hundred bytes.
        "explanation": result.get("explanation"),
        "audio_available": (result.get("audio") or {}).get("available"),
        "lipsync": (result.get("audio") or {}).get("lipsync"),
        # Acoustic descriptors only (flatness ratios, indicator flags) — a
        # few numbers, not the audio itself. Kept so a reopened report's
        # voice-synthesis panel reads the same as it did live.
        "voice": (result.get("audio") or {}).get("voice"),
        "media": {k: v for k, v in (result.get("media") or {}).items()
                  if k in ("codec", "width", "height", "fps", "duration",
                           "bit_rate")},
        "timings": result.get("timings"),
        # Plain-English notes and a few scalars needed to regenerate the
        # PDF/report view after logout — text only, no evidence media.
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

    # ------------------------------------------------------------- write
    def save(self, record: dict) -> dict:
        record = {k: v for k, v in record.items() if v is not None}
        rid = record.get("analysis_id", "")
        if self.durable and self.table.put(_clean(record)):
            return {"stored": True, "backend": self.backend, "report_id": rid}
        if self.durable:
            self.error = getattr(self.table, "error", None)
        # Append locally so a finding is never lost, whichever path failed.
        try:
            with self.local_path.open("a", encoding="utf8") as f:
                f.write(json.dumps(_clean(record), default=str) + "\n")
            return {"stored": True, "backend": "local", "report_id": rid,
                    "error": self.error}
        except OSError as exc:
            return {"stored": False, "backend": self.backend,
                    "error": str(exc)[:160]}

    # -------------------------------------------------------------- read
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
        """Every user's findings. Administrative view — a Scan."""
        if self.durable:
            return self._sort_newest(self.table.scan(limit * 3))[:limit]
        return self._local_rows(limit)

    def for_user(self, user_id: str, limit: int = 25) -> list:
        """One analyst's own findings — a Query on their partition, so
        cost and latency stay flat as the table grows."""
        if not user_id:
            return self.recent(limit)
        if self.durable:
            rows = self.table.query(key_value=user_id, Limit=max(limit, 1))
            return self._sort_newest(rows)[:limit]
        return [r for r in self._local_rows(limit * 4)
                if r.get("user_id") == user_id][:limit]

    def find_by_hash(self, sha256: str, user_id: str = "") -> list:
        """Has this exact file been checked before?

        Scoped to one user when a user_id is given, so the duplicate hint
        never reveals that a different analyst looked at the same file.
        """
        pool = self.for_user(user_id, 200) if user_id else self.recent(200)
        return [r for r in pool
                if sha256 in (r.get("file_sha256"), r.get("evidence_sha256"))]

    def status(self) -> dict:
        st = dict(self.table.status())
        st["local_fallback"] = str(self.local_path)
        if self.error:
            st["last_write_error"] = self.error
        return st
