"""Web server.

    python -m app.server            # http://127.0.0.1:8000
    uvicorn app.server:app --host 0.0.0.0 --port 8000

Zero retention: the upload is written to a temp file, scored, and deleted
in a finally block. Face thumbnails are returned to the browser in the
response and never written to disk.
"""
from __future__ import annotations

import asyncio
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

from fastapi import (APIRouter, Depends, FastAPI, File, HTTPException, Query,
                     Request, Response, UploadFile)
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import AUTH, STORE  # noqa: E402
from app import audit as audit_mod  # noqa: E402
from app.auth import AuthError, AuthStore, public_user  # noqa: E402
from app.engine import Engine  # noqa: E402
from app.store import ReportStore, build_record  # noqa: E402

STATIC = Path(__file__).parent / "static"
MAX_BYTES = 250 * 1024 * 1024
ALLOWED = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".3gp", ".m4v"}

router = APIRouter()
_engine: Engine | None = None
_store: ReportStore | None = None
_auth: AuthStore | None = None
_audit: audit_mod.AuditLog | None = None

# One analysis at a time per worker. Face detection and inference are
# CPU-bound and will thrash if several run at once; queueing keeps
# latency predictable under load (IR 3.4.1, scalability).
_slots = asyncio.Semaphore(int(os.getenv("DF_CONCURRENCY", "2")))


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = Engine()
        print(f"[engine] {_engine.status()}")
    return _engine


def get_store() -> ReportStore | None:
    global _store
    if not STORE.enabled:
        return None
    if _store is None:
        _store = ReportStore(STORE.local_path)
        print(f"[store] {_store.status()}")
    return _store


def get_auth() -> AuthStore:
    global _auth
    if _auth is None:
        _auth = AuthStore()
        print(f"[auth] {_auth.status()}")
    return _auth


def get_audit() -> audit_mod.AuditLog:
    global _audit
    if _audit is None:
        _audit = audit_mod.AuditLog()
        print(f"[audit] {_audit.status()}")
    return _audit


def client_ip(request: Request) -> str:
    """Railway sits behind a proxy, so request.client.host is the proxy.
    The left-most X-Forwarded-For entry is the original caller.
    """
    fwd = request.headers.get("x-forwarded-for", "")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else ""


# ------------------------------------------------------------- auth deps
def current_user(request: Request) -> dict | None:
    """The signed-in user, or None. Never raises — for endpoints that
    behave differently when signed in but do not demand it."""
    token = request.cookies.get(AUTH.cookie_name, "")
    if not token:
        # Lets the browser extension authenticate without a cookie jar.
        header = request.headers.get("authorization", "")
        if header.lower().startswith("bearer "):
            token = header[7:].strip()
    if not token:
        return None
    try:
        return get_auth().resolve(token)
    except Exception as exc:
        print(f"[auth] resolve failed: {type(exc).__name__}: {exc}")
        return None


def require_user(request: Request) -> dict:
    """Gate an endpoint behind a session. IR 3.4.1 restricts evidence
    ingestion to authenticated investigators; this is where that is
    actually enforced, rather than by hiding a button in the UI."""
    if not AUTH.require_login:
        return current_user(request) or {"user_id": "anonymous",
                                         "email": None, "name": "Anonymous"}
    user = current_user(request)
    if not user:
        raise HTTPException(401, "Sign in to run an analysis.")
    return user


@router.get("/status")
def status(request: Request):
    out = get_engine().status()
    store = get_store()
    user = current_user(request)

    store_st = store.status() if store else {"backend": "disabled",
                                             "durable": False}
    auth_st = get_auth().status()
    audit_st = get_audit().status()
    tables = [store_st, auth_st["users"], auth_st["sessions"], audit_st]

    out["authenticated"] = user is not None
    out["login_required"] = AUTH.require_login
    # One honest flag the deployment can be judged on: if any table is on
    # the local backend inside a container, findings do not survive a
    # redeploy. Safe to publish — it is a property, not a detail.
    out["persistence"] = {"durable": all(t.get("durable") for t in tables)}

    if user:
        # Table names, region and raw boto3 error text are operational
        # detail. They are genuinely useful when debugging a deployment
        # and have no business being readable by anyone who curls the
        # health check, so they are behind a session.
        out["store"] = store_st
        out["auth"] = auth_st
        out["audit"] = audit_st
        out["persistence"]["ephemeral_tables"] = [
            t.get("table") for t in tables if not t.get("durable")]
    else:
        out["store"] = {"backend": store_st.get("backend"),
                        "durable": store_st.get("durable")}
    return out


# ----------------------------------------------------------------- auth
class RegisterRequest(BaseModel):
    email: str
    password: str
    name: str
    studentId: str = ""
    role: str = ""


class LoginRequest(BaseModel):
    email: str
    password: str


def _set_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        AUTH.cookie_name, token,
        max_age=AUTH.session_days * 86400,
        httponly=True,                 # JS cannot read it, so XSS cannot steal it
        secure=AUTH.cookie_secure,     # HTTPS only in production
        samesite="lax",                # blocks cross-site CSRF on POST
        path="/",
    )


@router.post("/auth/register")
def auth_register(body: RegisterRequest, request: Request, response: Response):
    log = get_audit()
    ip = client_ip(request)
    try:
        user = get_auth().register(body.email, body.password, body.name,
                                   body.studentId, body.role)
    except AuthError as exc:
        log.write(audit_mod.REGISTER, email=body.email, ip=ip,
                  detail={"ok": False, "reason": str(exc)})
        raise HTTPException(400, str(exc))

    full = get_auth().get_user(user["user_id"]) or {}
    token = get_auth().create_session(full, ip,
                                      request.headers.get("user-agent", ""))
    _set_session_cookie(response, token)
    log.write(audit_mod.REGISTER, user_id=user["user_id"], email=user["email"],
              ip=ip, detail={"ok": True, "role": user.get("role")})
    return {"user": user, "token": token}


@router.post("/auth/login")
def auth_login(body: LoginRequest, request: Request, response: Response):
    log = get_audit()
    ip = client_ip(request)
    try:
        user = get_auth().authenticate(body.email, body.password)
    except AuthError as exc:
        log.write(audit_mod.LOGIN_FAILED, email=(body.email or "").lower(),
                  ip=ip, detail={"reason": str(exc),
                                 "agent": request.headers.get("user-agent", "")[:120]})
        raise HTTPException(401, str(exc))

    token = get_auth().create_session(user, ip,
                                      request.headers.get("user-agent", ""))
    _set_session_cookie(response, token)
    log.write(audit_mod.LOGIN, user_id=user["user_id"], email=user["email"],
              ip=ip, detail={"login_count": user.get("login_count", 0)})
    return {"user": public_user(user), "token": token}


@router.post("/auth/logout")
def auth_logout(request: Request, response: Response):
    token = request.cookies.get(AUTH.cookie_name, "")
    user = current_user(request)
    if token:
        get_auth().revoke(token)
    response.delete_cookie(AUTH.cookie_name, path="/")
    if user:
        get_audit().write(audit_mod.LOGOUT, user_id=user.get("user_id", ""),
                          email=user.get("email", ""), ip=client_ip(request))
    return {"ok": True}


@router.post("/auth/logout-all")
def auth_logout_all(request: Request, response: Response,
                    user: dict = Depends(require_user)):
    n = get_auth().revoke_all(user["user_id"])
    response.delete_cookie(AUTH.cookie_name, path="/")
    get_audit().write(audit_mod.LOGOUT_ALL, user_id=user["user_id"],
                      email=user.get("email", ""), ip=client_ip(request),
                      detail={"sessions_revoked": n})
    return {"ok": True, "sessions_revoked": n}


@router.get("/auth/me")
def auth_me(request: Request):
    user = current_user(request)
    if not user:
        return {"user": None}
    return {"user": public_user(user)}


# -------------------------------------------------------------- reports
@router.get("/reports")
def reports(request: Request, limit: int = Query(25, ge=1, le=100)):
    """Past findings. The videos are gone; the reports remain.

    Scoped to the caller. Chain of custody means an analyst sees the
    evidence they handled, not everyone's.
    """
    store = get_store()
    if not store:
        return {"reports": [], "backend": "disabled"}
    user = current_user(request)
    if user:
        return {"reports": store.for_user(user["user_id"], limit),
                "backend": store.backend, "scope": "user"}
    if AUTH.require_login:
        return {"reports": [], "backend": store.backend, "scope": "none"}
    return {"reports": store.recent(limit), "backend": store.backend,
            "scope": "all"}


@router.get("/audit")
def audit_feed(limit: int = Query(50, ge=1, le=200),
               user: dict = Depends(require_user)):
    """The caller's own activity trail — IR 2.4.4 "system logs"."""
    return {"events": get_audit().for_user(user["user_id"], limit)}


# Keyed by user. This was a single global, which was correct while the
# server had one operator and became a cross-account leak the moment it
# had accounts: whoever polled /reports/latest got whatever the previous
# caller had just analysed, whoever they were.
_LATEST_RESULT: dict[str, dict] = {}
_LATEST_CAP = 64


def _remember_latest(user_id: str, result: dict) -> None:
    if len(_LATEST_RESULT) >= _LATEST_CAP:
        _LATEST_RESULT.pop(next(iter(_LATEST_RESULT)), None)
    _LATEST_RESULT[user_id or "anonymous"] = result


@router.get("/reports/latest")
def latest_report(request: Request):
    user = current_user(request)
    key = user["user_id"] if user else "anonymous"
    result = _LATEST_RESULT.get(key)
    if not result:
        raise HTTPException(404, "No verification report generated yet.")
    return result


@router.get("/reports/by-hash/{sha256}")
def reports_by_hash(sha256: str, request: Request):
    if not re.fullmatch(r"[0-9a-fA-F]{64}", sha256):
        raise HTTPException(400, "That is not a SHA-256 digest.")
    store = get_store()
    if not store:
        return {"reports": [], "backend": "disabled"}
    user = current_user(request)
    if not user and AUTH.require_login:
        raise HTTPException(401, "Sign in to search past findings.")
    return {"reports": store.find_by_hash(sha256.lower(),
                                          user["user_id"] if user else "")}


class UrlRequest(BaseModel):
    url: str


def _download_stream_url(url: str, out_path: str) -> tuple[str, str]:
    import yt_dlp  # type: ignore
    base = os.path.splitext(out_path)[0]
    outtmpl = base + '.%(ext)s'
    ydl_opts: Any = {
        'format': 'bestvideo[height<=720]+bestaudio/best[height<=720]/best',
        'outtmpl': outtmpl,
        'merge_output_format': 'mp4',
        'overwrites': True,
        'postprocessor_args': ['-y'],
        'max_filesize': MAX_BYTES,
        'quiet': True,
        'no_warnings': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True) or {}
        title: str = str(info.get('title') or 'stream_video')
        actual_path = out_path
        if not os.path.exists(actual_path) or os.path.getsize(actual_path) == 0:
            for ext in [".mp4", ".mkv", ".webm"]:
                cand = base + ext
                if os.path.exists(cand) and os.path.getsize(cand) > 0:
                    actual_path = cand
                    break
        return title, actual_path


@router.post("/analyse-url")
async def analyse_url(req: UrlRequest, request: Request,
                      user: dict = Depends(require_user)):
    url = req.url.strip()
    if not url:
        raise HTTPException(400, "URL is required.")

    tmp_base = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_path = tmp_base.name
    tmp_base.close()
    if os.path.exists(tmp_path):
        os.unlink(tmp_path)

    try:
        title, final_path = await run_in_threadpool(_download_stream_url, url, tmp_path)
        if not os.path.exists(final_path) or os.path.getsize(final_path) == 0:
            raise HTTPException(400, f"Failed to download stream container from {url}.")

        import hashlib
        sha256 = hashlib.sha256()
        size = 0
        with open(final_path, "rb") as f:
            while chunk := f.read(1 << 20):
                size += len(chunk)
                sha256.update(chunk)
        digest = sha256.hexdigest()

        engine = get_engine()
        async with _slots:
            result = await run_in_threadpool(engine.analyse, final_path, f"{title}.mp4")

        result["filename"] = f"{title}.mp4"
        result["size_bytes"] = size
        result["evidence_sha256"] = digest
        result["source_url"] = url

        store = get_store()
        if store:
            record = build_record(result, result["filename"],
                                  engine.model_version,
                                  user.get("user_id", ""), user.get("email", ""))
            result["storage"] = store.save(record)
            result["report_id"] = record["report_id"]
            prior = [r for r in store.find_by_hash(digest,
                                                   user.get("user_id", ""))
                     if r.get("report_id") != record["report_id"]]
            if prior:
                result["seen_before"] = len(prior)

        get_audit().write(audit_mod.ANALYSIS, user.get("user_id", ""),
                          user.get("email", ""), client_ip(request),
                          {"source": "url", "url": url[:200],
                           "label": result.get("label"),
                           "probability": result.get("probability"),
                           "evidence_sha256": digest,
                           "report_id": result.get("report_id")})

        _remember_latest(user.get("user_id", ""), result)
        return result
    except HTTPException:
        raise
    except Exception as exc:
        get_audit().write(audit_mod.ANALYSIS_FAILED, user.get("user_id", ""),
                          user.get("email", ""), client_ip(request),
                          {"source": "url", "url": url[:200],
                           "error": f"{type(exc).__name__}: {str(exc)[:160]}"})
        raise HTTPException(400, f"Could not stream video from {url}: {exc}")
    finally:
        try:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except OSError:
            pass


@router.post("/analyse")
async def analyse(request: Request, video: UploadFile = File(...),
                  user: dict = Depends(require_user)):
    name = video.filename or ""
    ext = os.path.splitext(name)[1].lower()
    if ext not in ALLOWED:
        raise HTTPException(
            415, f"{ext or 'That file type'} is not a video this tool reads. "
                 "Use MP4, MOV, AVI, MKV, WebM or 3GP.")

    tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    try:
        import hashlib
        sha256 = hashlib.sha256()
        size = 0
        while chunk := await video.read(1 << 20):
            size += len(chunk)
            sha256.update(chunk)
            if size > MAX_BYTES:
                raise HTTPException(413, "That file is over 250 MB. Trim the "
                                         "clip and try again.")
            tmp.write(chunk)
        tmp.close()

        digest = sha256.hexdigest()
        engine = get_engine()

        # analyse() is CPU-bound. Running it directly in this coroutine
        # would block the event loop and stall every other request.
        async with _slots:
            result = await run_in_threadpool(engine.analyse, tmp.name, name)

        result["filename"] = name
        result["size_bytes"] = size
        result["evidence_sha256"] = digest

        store = get_store()
        if store:
            record = build_record(result, name, engine.model_version,
                                  user.get("user_id", ""), user.get("email", ""))
            result["storage"] = store.save(record)
            result["report_id"] = record["report_id"]
            prior = [r for r in store.find_by_hash(digest,
                                                   user.get("user_id", ""))
                     if r.get("report_id") != record["report_id"]]
            if prior:
                result["seen_before"] = len(prior)
                result["notes"].append(
                    f"You have checked this exact file {len(prior)} time"
                    f"{'s' if len(prior) > 1 else ''} before. The earlier "
                    "reports are under Past findings.")

        get_audit().write(audit_mod.ANALYSIS, user.get("user_id", ""),
                          user.get("email", ""), client_ip(request),
                          {"source": "upload", "filename": name[:160],
                           "size_bytes": size,
                           "label": result.get("label"),
                           "probability": result.get("probability"),
                           "evidence_sha256": digest,
                           "report_id": result.get("report_id")})

        _remember_latest(user.get("user_id", ""), result)
        return result
    finally:
        tmp.close()
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Deepfake Forensics", description="BSc Cybersecurity Capstone System", docs_url=None, redoc_url=None)

# `allow_origins=["*"]` together with `allow_credentials=True` is invalid
# per the CORS spec and every browser drops the response, which would have
# broken session cookies the moment this shipped. The interface is served
# from the same origin as the API and needs no CORS at all; the list below
# exists for the browser extension and for any origin you name explicitly.
_ORIGINS = [o.strip() for o in os.getenv("DF_ALLOWED_ORIGINS", "").split(",")
            if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ORIGINS,
    allow_origin_regex=r"^chrome-extension://[a-p]{32}$",
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)
app.include_router(router, prefix="/api")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.on_event("startup")
def on_startup():
    eng = get_engine()
    print(f"[DEFAULT ENGINE] Neural Deepfake Model Loaded: {eng.model_version} | Mode: {eng.mode} | Backbone: {eng.backbone}")
    get_store()
    get_auth()
    get_audit()


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


if __name__ == "__main__":
    import uvicorn
    # Railway assigns the port at runtime and routes to it; a hardcoded
    # 8000 makes the deploy fail its health check.
    uvicorn.run("app.server:app", host="0.0.0.0",
                port=int(os.getenv("PORT", "8000")), reload=False)
