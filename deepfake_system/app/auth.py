"""Accounts and sessions on DynamoDB — the "user metadata" half of IR 2.4.4.

Three decisions worth defending in the viva:

1. `user_id` is a UUIDv5 of the normalised email, not a random UUIDv4.
   DynamoDB has no unique constraint, and a read-then-write check on a
   GSI races: two simultaneous sign-ups both see "no such email" and both
   succeed. Deriving the partition key from the email turns uniqueness
   into `attribute_not_exists(user_id)` on a conditional put, which is
   atomic. It also makes login a 1-RCU GetItem instead of a GSI query.

2. The session table stores SHA-256 of the token, never the token. The
   cookie the browser holds is the only copy of the secret, so a dump of
   the table cannot be replayed as a live session — the same reason
   password hashes exist.

3. Expiry is enforced in code as well as by DynamoDB TTL. TTL deletion is
   asynchronous and AWS documents it as typically within 48 hours, so an
   expired row can still be read back. Trusting TTL alone would extend
   every session by up to two days.

Passwords use PBKDF2-HMAC-SHA256 from the standard library. bcrypt or
argon2 would be stronger per round, but both are compiled dependencies
that have to build in the Railway image; PBKDF2 at 600k rounds is the
OWASP-documented floor and costs nothing at deploy time.
"""
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

from config import AUTH, STORE  # noqa: E402
from app.local_table import Table  # noqa: E402

# Namespace for deterministic user ids. Fixed forever — changing it
# orphans every existing account.
_NS = uuid.UUID("6ba7b811-9dad-11d1-80b4-00c04fd430c8")

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[a-zA-Z]{2,}$")

ROLES = (
    "BSc Cybersecurity Student (Final Year)",
    "Digital Forensics Researcher",
    "Project Supervisor / Examiner",
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def user_id_for(email: str) -> str:
    return str(uuid.uuid5(_NS, "mailto:" + email.strip().lower()))


# --------------------------------------------------------------- hashing
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
    # Constant time: a timing side channel here leaks the hash byte by byte.
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
    """Raised with a message safe to show the user."""


class AuthStore:
    def __init__(self, prefer_dynamo: bool = False):
        d = STORE.local_dir
        self.users = Table("deepfake_users", ("user_id", None), None,
                           d / "users.json", False)
        self.sessions = Table("deepfake_sessions", ("session_id", None), None,
                              d / "sessions.json", False)

    # ------------------------------------------------------- registration
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

        now = _iso(_now())
        record = {
            "user_id": user_id_for(email),
            "email": email,
            "name": name,
            "student_id": (student_id or "").strip().upper(),
            "role": role if role in ROLES else ROLES[0],
            "initials": _initials(name),
            "password_hash": hash_password(password),
            "created_at": now,
            "updated_at": now,
            "last_login_at": None,
            "login_count": 0,
            "failed_attempts": 0,
            "locked_until": None,
            "account_status": "active",
        }
        if not self.users.put(record, unique=True):
            if self.users.error:
                raise AuthError("The account store is unavailable right now.")
            raise AuthError("An account with that email already exists.")
        return public_user(record)

    # ------------------------------------------------------ authentication
    def get_user(self, user_id: str) -> dict | None:
        return self.users.get({"user_id": user_id})

    def get_by_email(self, email: str) -> dict | None:
        return self.get_user(user_id_for(email))

    def authenticate(self, email: str, password: str) -> dict:
        """Return the stored user record, or raise AuthError."""
        email = (email or "").strip().lower()
        if not email or not password:
            raise AuthError("Email and password are required.")

        user = self.get_by_email(email)
        # Spend the KDF even when the account is absent, so that response
        # time does not distinguish "no such user" from "wrong password".
        if user is None:
            hash_password(password)
            raise AuthError("Email or password is incorrect.")

        locked = user.get("locked_until")
        if locked and _iso(_now()) < str(locked):
            raise AuthError(
                "Too many failed attempts. Try again after "
                f"{str(locked)[11:16]} UTC.")

        if user.get("account_status") != "active":
            raise AuthError("That account is disabled.")

        if not verify_password(password, user.get("password_hash", "")):
            fails = int(user.get("failed_attempts", 0)) + 1
            user["failed_attempts"] = fails
            if fails >= AUTH.max_failed:
                user["locked_until"] = _iso(
                    _now() + timedelta(minutes=AUTH.lockout_minutes))
                user["failed_attempts"] = 0
            self.users.put(user)
            raise AuthError("Email or password is incorrect.")

        user["failed_attempts"] = 0
        user["locked_until"] = None
        user["last_login_at"] = _iso(_now())
        user["login_count"] = int(user.get("login_count", 0)) + 1
        self.users.put(user)
        return user

    # ------------------------------------------------------------ sessions
    def create_session(self, user: dict, ip: str = "", agent: str = "") -> str:
        """Return the raw token. Only the browser ever holds this value."""
        token = secrets.token_urlsafe(32)
        now = _now()
        expires = now + timedelta(days=AUTH.session_days)
        self.sessions.put({
            "session_id": _token_id(token),
            "user_id": user["user_id"],
            "email": user.get("email"),
            "created_at": _iso(now),
            "expires_at_iso": _iso(expires),
            # Numeric epoch attribute — this is what DynamoDB TTL reads.
            "expires_at": int(expires.timestamp()),
            "ip": (ip or "")[:64],
            "user_agent": (agent or "")[:200],
            "revoked": False,
        })
        return token

    def resolve(self, token: str) -> dict | None:
        """Token -> user record, or None. Enforces expiry itself."""
        if not token:
            return None
        sess = self.sessions.get({"session_id": _token_id(token)})
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
        return self.sessions.delete({"session_id": _token_id(token)})

    def revoke_all(self, user_id: str) -> int:
        """Sign out every device — the control an analyst needs when a
        laptop goes missing mid-investigation."""
        rows = self.sessions.query_index("user_id-index", "user_id",
                                         user_id, limit=200)
        n = 0
        for r in rows:
            if self.sessions.delete({"session_id": r["session_id"]}):
                n += 1
        return n

    def status(self) -> dict:
        return {"users": self.users.status(),
                "sessions": self.sessions.status()}


def public_user(user: dict) -> dict:
    """The subset safe to hand the browser. Never the hash."""
    return {
        "user_id": user.get("user_id"),
        "email": user.get("email"),
        "name": user.get("name"),
        "studentId": user.get("student_id"),
        "role": user.get("role"),
        "initials": user.get("initials") or _initials(user.get("name", "")),
        "created_at": user.get("created_at"),
        "last_login_at": user.get("last_login_at"),
        "login_count": user.get("login_count", 0),
    }
