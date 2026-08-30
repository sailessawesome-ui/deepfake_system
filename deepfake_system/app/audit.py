"""Append-only event log for forensic examiner actions (ISO/IEC 27037 compliant)."""
from __future__ import annotations

import os
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import tables  # noqa: E402

RETAIN_DAYS = int(os.environ.get("DF_AUDIT_RETAIN_DAYS", "365"))

REGISTER = "auth.register"
LOGIN = "auth.login"
LOGIN_FAILED = "auth.login_failed"
LOGOUT = "auth.logout"
LOGOUT_ALL = "auth.logout_all"
PROFILE_UPDATE = "auth.profile_update"
PASSWORD_CHANGE = "auth.password_change"
ANALYSIS = "analysis.completed"
ANALYSIS_FAILED = "analysis.failed"
REPORT_READ = "report.read"
# IR 3.4.1 non-functional 4: a model change is an evidential event.
MODEL_ACTIVATED = "model.activated"
MODEL_REJECTED = "model.rejected"


class AuditLog:
    def __init__(self, prefer_dynamo: bool = True):
        self.table = tables.get("audit_log", force_local=not prefer_dynamo)

    def write(self, event: str, user_id: str = "", email: str = "",
              ip: str = "", detail: dict | None = None) -> str | None:
        now = datetime.now(timezone.utc)
        event_ts = f"{now.isoformat()}#{uuid.uuid4().hex[:8]}"
        item = {
            "audit_day": now.strftime("%Y-%m-%d"),
            "event_ts": event_ts,
            "event": event,
            # No personal default here. An event with no signed-in user is
            # genuinely anonymous, and stamping it with the developer's own
            # identity would put false attribution into an evidence log.
            "user_id": user_id or "anonymous",
            "email": email or None,
            "ip": (ip or "")[:64] or None,
            "detail": detail or {},
            "expires_at": int((now + timedelta(days=RETAIN_DAYS)).timestamp()),
        }
        try:
            if self.table.put(item):
                return event_ts
            print(f"[audit] write failed: {self.table.error}")
        except Exception as exc:                  # never break a request
            print(f"[audit] {type(exc).__name__}: {exc}")
        return None

    def day(self, date: str, limit: int = 100) -> list[dict]:
        """One UTC date, newest first — a Query on the partition key."""
        rows = self.table.query(key_value=date, Limit=max(limit, 1))
        if not rows:
            rows = [r for r in self.table.scan(2000)
                    if r.get("audit_day") == date]
        rows.sort(key=lambda r: str(r.get("event_ts", "")), reverse=True)
        return rows[:limit]

    def for_user(self, user_id: str, limit: int = 50) -> list[dict]:
        rows = self.table.query_index(tables.USER_AUDIT_INDEX, "user_id",
                                      user_id, limit=max(limit, 1))
        if not rows:
            rows = [r for r in self.table.scan(2000)
                    if r.get("user_id") == user_id]
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
