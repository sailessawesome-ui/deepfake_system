"""Report store — AWS DynamoDB, per IR section 2.4.4.

Zero retention (IR 3.4.1, security requirement) means the *video* is
never kept. It does not mean the verdict cannot be. A forensic tool that
cannot produce the report it issued last week is not much use to the
analyst it was built for, so what gets written here is the finding, the
evidence hash, and the model version — never the media, never the face
thumbnails.

Falls back to a local JSON-lines file when boto3 or credentials are
absent, so the app runs on your laptop and on AWS without a code change.
Which backend is live is reported at /api/status.

Table (on-demand billing is fine):

    aws dynamodb create-table \
      --table-name deepfake_reports \
      --attribute-definitions AttributeName=report_id,AttributeType=S \
                              AttributeName=created_at,AttributeType=S \
      --key-schema AttributeName=report_id,KeyType=HASH \
                   AttributeName=created_at,KeyType=RANGE \
      --billing-mode PAY_PER_REQUEST
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path


# Fields that must never reach storage, whatever the caller passes.
_STRIP = {"frames", "thumb", "media_path", "tmp_path"}


def _clean(value):
    """DynamoDB rejects floats; Decimal is the documented substitute."""
    if isinstance(value, float):
        return Decimal(str(round(value, 6)))
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items() if k not in _STRIP}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def _undecimal(value):
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, dict):
        return {k: _undecimal(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_undecimal(v) for v in value]
    return value


def build_record(result: dict, filename: str, model_version: str) -> dict:
    """The subset of a result that is safe and useful to keep."""
    now = datetime.now(timezone.utc).isoformat()
    return {
        "report_id": str(uuid.uuid4()),
        "created_at": now,
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
        "content_credentials": (result.get("content_credentials") or {}).get("c2pa"),
        "audio_available": (result.get("audio") or {}).get("available"),
        "lipsync": (result.get("audio") or {}).get("lipsync"),
        "media": {k: v for k, v in (result.get("media") or {}).items()
                  if k in ("codec", "width", "height", "fps", "duration",
                           "bit_rate")},
        "timings": result.get("timings"),
    }


class ReportStore:
    def __init__(self, table_name: str, region: str, local_path: Path,
                 prefer_dynamo: bool = True):
        self.table_name = table_name
        self.region = region
        self.local_path = Path(local_path)
        self.backend = "local"
        self.table = None
        self.error = None

        if prefer_dynamo:
            try:
                import boto3
                from botocore.exceptions import BotoCoreError, ClientError
                res = boto3.resource("dynamodb", region_name=region)
                table = res.Table(table_name)
                table.load()                       # raises if absent/no creds
                self.table = table
                self.backend = "dynamodb"
            except ImportError:
                self.error = "boto3 not installed"
            except Exception as exc:               # ClientError, NoCredentials
                self.error = f"{type(exc).__name__}: {str(exc)[:120]}"

        if self.backend == "local":
            self.local_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------- write
    def save(self, record: dict) -> dict:
        record = {k: v for k, v in record.items() if v is not None}
        if self.backend == "dynamodb":
            try:
                self.table.put_item(Item=_clean(record))
                return {"stored": True, "backend": "dynamodb",
                        "report_id": record["report_id"]}
            except Exception as exc:
                self.error = str(exc)[:160]
                # Fall through to local so a finding is never lost.
        try:
            with self.local_path.open("a") as f:
                f.write(json.dumps(record, default=str) + "\n")
            return {"stored": True, "backend": "local",
                    "report_id": record["report_id"]}
        except OSError as exc:
            return {"stored": False, "backend": self.backend,
                    "error": str(exc)[:160]}

    # -------------------------------------------------------------- read
    def recent(self, limit: int = 25) -> list:
        if self.backend == "dynamodb":
            try:
                items = self.table.scan(Limit=limit * 3).get("Items", [])
                items = [_undecimal(i) for i in items]
                items.sort(key=lambda r: r.get("created_at", ""), reverse=True)
                return items[:limit]
            except Exception as exc:
                self.error = str(exc)[:160]
                return []
        if not self.local_path.exists():
            return []
        try:
            lines = self.local_path.read_text().strip().splitlines()
        except OSError:
            return []
        out = []
        for line in reversed(lines[-limit * 2:]):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if len(out) >= limit:
                break
        return out

    def find_by_hash(self, sha256: str) -> list:
        """Has this exact file been checked before? Useful in an
        investigation, and it costs nothing to answer."""
        return [r for r in self.recent(200)
                if r.get("evidence_sha256") == sha256]

    def status(self) -> dict:
        return {"backend": self.backend, "table": self.table_name,
                "region": self.region if self.backend == "dynamodb" else None,
                "error": self.error}
