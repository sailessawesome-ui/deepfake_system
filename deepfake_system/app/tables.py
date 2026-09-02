"""One place that knows every table's name, key schema and index.

The schemas here are not invented — they are read off the four tables that
already exist in the account (`dfd_users`, `dfd_sessions`, `dfd_analyses`,
`dfd_login_attempts`), which already hold real records. The code matches
the data, not the other way round; changing a key here orphans rows.

    users            user_id (H)                     GSI EmailIndex(email)
    sessions         token_hash (H)                  GSI UserSessionsIndex(user_id)
    analyses         user_id (H) + analysis_id (R)
    login_attempts   identifier (H) + attempted_at (R, Number)
    audit_log        audit_day (H) + event_ts (R)    GSI UserAuditIndex(user_id)

`audit_log` is the only one that did not exist; scripts/create_tables.py
creates it. Everything else is used as found.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import STORE  # noqa: E402

# logical name -> (config attribute, hash key, range key, local filename)
SPECS: dict[str, tuple[str, str, str | None, str]] = {
    "users": ("users_table", "user_id", None, "users.json"),
    "sessions": ("sessions_table", "token_hash", None, "sessions.json"),
    "analyses": ("analyses_table", "user_id", "analysis_id", "analyses.json"),
    "login_attempts": ("login_attempts_table", "identifier", "attempted_at",
                       "login_attempts.json"),
    "audit_log": ("audit_table", "audit_day", "event_ts", "audit.json"),
}

# Index names as they exist in the account.
EMAIL_INDEX = "EmailIndex"
USER_SESSIONS_INDEX = "UserSessionsIndex"
USER_AUDIT_INDEX = "UserAuditIndex"

_cache: dict[str, Any] = {}


def table_name(logical: str) -> str:
    return getattr(STORE, SPECS[logical][0])


def get(logical: str, force_local: bool = False) -> Any:
    key = f"{logical}:{'local' if force_local else STORE.backend}"
    if key in _cache:
        return _cache[key]

    attr, pk, sk, filename = SPECS[logical]
    name = getattr(STORE, attr)
    local_path = STORE.local_dir / filename

    if STORE.use_dynamodb and not force_local:
        from app.dynamo_table import Table as DynamoTable
        t = DynamoTable(name, (pk, sk), STORE.region, local_path, True)
        if t.durable:
            _cache[key] = t
            return t
        print(f"[tables] {name}: DynamoDB unavailable ({t.error}) "
              "- falling back to local JSON")

    from app.local_table import Table as LocalTable
    t = LocalTable(name, (pk, sk), None, local_path, False)
    _cache[key] = t
    return t


def status() -> dict:
    """Backend and health of every table, for /api/status."""
    out = {}
    for logical in SPECS:
        try:
            out[logical] = get(logical).status()
        except Exception as exc:
            out[logical] = {"table": table_name(logical), "backend": "error",
                            "durable": False, "error": str(exc)[:140]}
    return out


def reset_cache() -> None:
    _cache.clear()
