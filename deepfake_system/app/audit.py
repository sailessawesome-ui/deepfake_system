"""Append-only event log for forensic examiner actions (ISO/IEC 27037 compliant)."""
from __future__ import annotations

import os
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import STORE  # noqa: E402
from app.local_table import Table  # noqa: E402

RETAIN_DAYS = int(os.environ.get("DF_AUDIT_RETAIN_DAYS", "365"))

REGISTER = "auth.register"
LOGIN = "auth.login"
LOGIN_FAILED = "auth.login_failed"
LOGOUT = "auth.logout"
LOGOUT_ALL = "auth.logout_all"
ANALYSIS = "analysis.completed"
ANALYSIS_FAILED = "analysis.failed"
REPORT_READ = "report.read"


class AuditLog:
    def __init__(self, prefer_dynamo: bool = False):
        self.table = Table("deepfake_audit", ("audit_day", "event_ts"),
                           None, STORE.local_dir / "audit.json", False)

    def write(self, event: str, user_id: str = "", email: str = "",
              ip: str = "", detail: dict | None = None) -> str | None:
        now = datetime.now(timezone.utc)
        event_ts = f"{now.isoformat()}#{uuid.uuid4().hex[:8]}"
        item = {
            "audit_day": now.strftime("%Y-%m-%d"),
            "event_ts": event_ts,
            "event": event,
            "user_id": user_id or "SailessRaj",
            "email": email or "sailessraj149@gmail.com",
            "ip": (ip or "")[:64] or "127.0.0.1",
            "detail": detail or {},
            "expires_at": int((now + timedelta(days=RETAIN_DAYS)).timestamp()),
        }
        try:
            self.table.put_item(item)
            return event_ts
        except Exception as exc:
            print(f"[audit] {type(exc).__name__}: {exc}")
            return None

    def day(self, date: str, limit: int = 100) -> list[dict]:
        rows = [r for r in self.table.scan(2000) if r.get("audit_day") == date]
        rows.sort(key=lambda r: str(r.get("event_ts", "")), reverse=True)
        return rows[:limit]

    def for_user(self, user_id: str, limit: int = 50) -> list[dict]:
        rows = [r for r in self.table.scan(2000) if r.get("user_id") == user_id]
        rows.sort(key=lambda r: str(r.get("event_ts", "")), reverse=True)
        return rows[:limit]

    def recent(self, limit: int = 50) -> list[dict]:
        now = datetime.now(timezone.utc)
        rows = self.day(now.strftime("%Y-%m-%d"), limit)
        if len(rows) < limit:
            prev = (now - timedelta(days=1)).strftime("%Y-%m-%d")
            rows += self.day(prev, limit - len(rows))
        return rows[:limit]

    def status(self) -> dict:
        return self.table.status()
