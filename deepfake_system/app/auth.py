from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import AUTH  # noqa: E402
from app import tables  # noqa: E402

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[a-zA-Z]{2,}$")

ROLES = (
    "Individual",
    "Journalist / Fact-Checker",
    "Researcher / Analyst",
    "Business / Organisation",
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def new_user_id() -> str:
    return str(uuid.uuid4())


def hash_password(password: str, rounds: int | None = None) -> str:
    rounds = rounds or AUTH.pbkdf2_rounds
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf8"), salt, rounds)
    return f"pbkdf2_sha256${rounds}${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, rounds_s, salt_hex, hash_hex = stored.split("$")
        if algo != "pbkdf2_sha256":
            return False
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf8"),
                                 bytes.fromhex(salt_hex), int(rounds_s))
    except (ValueError, AttributeError):
        return False
    return hmac.compare_digest(dk.hex(), hash_hex)


def _token_id(token: str) -> str:
    return hashlib.sha256(token.encode("utf8")).hexdigest()


def _initials(name: str) -> str:
    parts = [p for p in re.split(r"\s+", name.strip()) if p]
    if not parts:
        return "??"
    if len(parts) == 1:
        return parts[0][:2].upper()
    return (parts[0][0] + parts[-1][0]).upper()


class AuthError(Exception):
    pass

class AuthStore:
    def __init__(self, prefer_dynamo: bool = True):
        force_local = not prefer_dynamo
        self.users = tables.get("users", force_local)
        self.sessions = tables.get("sessions", force_local)
        self.attempts = tables.get("login_attempts", force_local)

    def _check_domain(self, email: str) -> None:
        allowed = [d.strip().lower() for d in AUTH.allowed_domains.split(",")
                   if d.strip()]
        if not allowed:
            return
        domain = email.rsplit("@", 1)[-1].lower()
        if domain not in allowed:
            raise AuthError(
                "Sign-up is restricted to " + ", ".join(allowed) + " addresses.")

    def register(self, email: str, password: str, name: str,
                 student_id: str = "", role: str = ROLES[0]) -> dict:
        email = (email or "").strip().lower()
        name = (name or "").strip()

        if not _EMAIL_RE.match(email):
            raise AuthError("That does not look like an email address.")
        if len(password or "") < 8:
            raise AuthError("Use at least 8 characters for the password.")
        if not name:
            raise AuthError("A name is required.")
        self._check_domain(email)

        if self.get_by_email(email) is not None:
            raise AuthError("An account with that email already exists.")

        now = _iso(_now())
        uid = new_user_id()
        record = {
            "user_id": uid,
            "id": uid,
            "email": email,
            "name": name,
            "display_name": name,
            "student_id": (student_id or "").strip().upper(),
            "role": role if role in ROLES else ROLES[0],
            "initials": _initials(name),
            "password_hash": hash_password(password),
            "created_at": now,
            "updated_at": now,
            "last_login_at": None,
            "login_count": 0,
            "is_admin": 0,
            "account_status": "active",
        }
        if not self.users.put(record, unique=True):
            if self.users.error:
                raise AuthError("The account store is unavailable right now.")
            raise AuthError("Could not create that account. Try again.")
        return public_user(record)

    def update_profile(self, user_id: str, name: str, student_id: str = "",
                       role: str = "") -> dict:
        user = self.get_user(user_id)
        if not user:
            raise AuthError("That account no longer exists.")

        name = (name or "").strip()
        if not name:
            raise AuthError("A name is required.")
        if len(name) > 120:
            raise AuthError("That name is too long.")

        student_id = (student_id or "").strip().upper()
        if len(student_id) > 40:
            raise AuthError("That student or staff ID is too long.")

        user["name"] = name
        user["display_name"] = name         
        user["student_id"] = student_id
        user["role"] = role if role in ROLES else user.get("role", ROLES[0])
        user["initials"] = _initials(name)
        user["updated_at"] = _iso(_now())

        if not self.users.put(user):
            raise AuthError("The account store is unavailable right now.")
        return public_user(user)

    def change_password(self, user_id: str, current_password: str,
                        new_password: str) -> None:
        user = self.get_user(user_id)
        if not user:
            raise AuthError("That account no longer exists.")
        if not verify_password(current_password or "",
                               str(user.get("password_hash", ""))):
            raise AuthError("The current password is not correct.")
        if len(new_password or "") < 8:
            raise AuthError("Use at least 8 characters for the new password.")
        if verify_password(new_password, str(user.get("password_hash", ""))):
            raise AuthError("The new password must differ from the current one.")

        user["password_hash"] = hash_password(new_password)
        user["updated_at"] = _iso(_now())
        if not self.users.put(user):
            raise AuthError("The account store is unavailable right now.")

    def get_user(self, user_id: str) -> dict | None:
        return self.users.get({"user_id": user_id})

    def get_by_email(self, email: str) -> dict | None:
        email = (email or "").strip().lower()
        if not email:
            return None
        rows = self.users.query_index(tables.EMAIL_INDEX, "email", email,
                                      limit=2)
        if not rows:
            return None
        rows.sort(key=lambda r: str(r.get("created_at", "")))
        return rows[0]

    def authenticate(self, email: str, password: str) -> dict:
        email = (email or "").strip().lower()
        if not email or not password:
            raise AuthError("Email and password are required.")

        user = self.get_by_email(email)
        if user is None:
            hash_password(password)
            raise AuthError("Email or password is incorrect.")

        recent_failures = self._recent_failures(email)
        if recent_failures >= AUTH.max_failed:
            raise AuthError(
                f"Too many failed attempts. Try again in "
                f"{AUTH.lockout_minutes} minutes.")

        if str(user.get("account_status", "active")) != "active":
            raise AuthError("That account is disabled.")

        if not verify_password(password, str(user.get("password_hash", ""))):
            self._record_failure(email)
            raise AuthError("Email or password is incorrect.")

        self._clear_failures(email)
        user["last_login_at"] = _iso(_now())
        user["login_count"] = int(user.get("login_count", 0) or 0) + 1
        self.users.put(user)
        return user

    _USEC = 1_000_000

    def _window_start(self) -> int:
        return int((_now() - timedelta(minutes=AUTH.lockout_minutes))
                   .timestamp() * self._USEC)

    def _recent_failures(self, identifier: str) -> int:
        try:
            rows = self.attempts.query(key_value=identifier, Limit=100)
        except Exception:
            return 0
        cutoff = self._window_start()
        return sum(1 for r in rows
                   if int(r.get("attempted_at", 0) or 0) >= cutoff)

    def _record_failure(self, identifier: str) -> None:
        now = _now()
        micros = int(now.timestamp() * self._USEC)
        ok = self.attempts.put({
            "identifier": identifier,
            "attempted_at": micros,
            "ttl": int(now.timestamp()) + AUTH.lockout_minutes * 60 * 4,
            "outcome": "failed",
        })
        if not ok:
            print("[auth] WARNING: could not record failed attempt: "
                  f"{getattr(self.attempts, 'error', 'unknown')}")

    def _clear_failures(self, identifier: str) -> None:
        try:
            rows = self.attempts.query(key_value=identifier, Limit=100)
        except Exception:
            return
        for r in rows:
            self.attempts.delete({"identifier": identifier,
                                  "attempted_at": r.get("attempted_at")})

    def failure_count(self, identifier: str) -> int:
        return self._recent_failures((identifier or "").strip().lower())

    def create_session(self, user: dict, ip: str = "", agent: str = "") -> str:
        token = secrets.token_urlsafe(32)
        now = _now()
        expires = now + timedelta(days=AUTH.session_days)
        self.sessions.put({
            "token_hash": _token_id(token),
            "user_id": user["user_id"],
            "email": user.get("email"),
            "created_at": now.timestamp(),
            "expires_at": int(expires.timestamp()),
            "ip_address": (ip or "")[:64] or "unknown",
            "user_agent": (agent or "")[:200],
            "revoked": False,
        })
        return token

    def resolve(self, token: str) -> dict | None:
        if not token:
            return None
        sess = self.sessions.get({"token_hash": _token_id(token)})
        if not sess or sess.get("revoked"):
            return None
        try:
            if int(sess.get("expires_at", 0)) <= int(_now().timestamp()):
                return None
        except (TypeError, ValueError):
            return None
        return self.get_user(str(sess.get("user_id", "")))

    def revoke(self, token: str) -> bool:
        if not token:
            return False
        return self.sessions.delete({"token_hash": _token_id(token)})

    def revoke_all(self, user_id: str) -> int:
        rows = self.sessions.query_index(tables.USER_SESSIONS_INDEX,
                                         "user_id", user_id, limit=200)
        n = 0
        for r in rows:
            if self.sessions.delete({"token_hash": r["token_hash"]}):
                n += 1
        return n

    def revoke_others(self, user_id: str, keep_token: str) -> int:
        keep = _token_id(keep_token) if keep_token else ""
        rows = self.sessions.query_index(tables.USER_SESSIONS_INDEX,
                                         "user_id", user_id, limit=200)
        n = 0
        for r in rows:
            if r.get("token_hash") == keep:
                continue
            if self.sessions.delete({"token_hash": r["token_hash"]}):
                n += 1
        return n

    def status(self) -> dict:
        return {"users": self.users.status(),
                "sessions": self.sessions.status(),
                "login_attempts": self.attempts.status()}


def display_name_of(user: dict) -> str:
    for field in ("name", "display_name"):
        value = str(user.get(field) or "").strip()
        if value:
            return value
    email = str(user.get("email") or "")
    return email.split("@")[0] or "Examiner"


def public_user(user: dict) -> dict:
    name = display_name_of(user)
    return {
        "user_id": user.get("user_id"),
        "email": user.get("email"),
        "name": name,
        "studentId": user.get("student_id"),
        "role": user.get("role") or ROLES[0],
        "initials": user.get("initials") or _initials(name),
        "created_at": user.get("created_at"),
        "last_login_at": user.get("last_login_at"),
        "login_count": user.get("login_count", 0),
        "is_admin": bool(user.get("is_admin", 0)),
    }
