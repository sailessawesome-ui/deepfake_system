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

from config import AUTH, INFER, STORE  # noqa: E402
from app import audit as audit_mod  # noqa: E402
from app import security as security_mod  # noqa: E402
from app.auth import AuthError, AuthStore, public_user  # noqa: E402
from app.engine import Engine  # noqa: E402
from app.store import ReportStore, build_record  # noqa: E402

STATIC = Path(__file__).parent / "static"
MAX_BYTES = AUTH.max_upload_mb * 1024 * 1024
ALLOWED = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".3gp", ".m4v"}

router = APIRouter()
_engine: Engine | None = None
_store: ReportStore | None = None
_auth: AuthStore | None = None
_audit: audit_mod.AuditLog | None = None

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
    fwd = request.headers.get("x-forwarded-for", "")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else ""


def current_user(request: Request) -> dict | None:
    token = request.cookies.get(AUTH.cookie_name, "")
    if not token:
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
    out["persistence"] = {"durable": all(t.get("durable") for t in tables)}

    if user:
        out["store"] = store_st
        out["auth"] = auth_st
        out["audit"] = audit_st
        out["transport"] = security_mod.status()
        out["transport"]["secure"] = security_mod.is_secure(request)
        out["persistence"]["ephemeral_tables"] = [
            t.get("table") for t in tables if not t.get("durable")]
    else:
        out["store"] = {"backend": store_st.get("backend"),
                        "durable": store_st.get("durable")}
    return out


class RegisterRequest(BaseModel):
    email: str
    password: str
    name: str
    studentId: str = ""
    role: str = ""


class LoginRequest(BaseModel):
    email: str
    password: str


class ProfileRequest(BaseModel):
    name: str
    studentId: str = ""
    role: str = ""


class PasswordRequest(BaseModel):
    currentPassword: str
    newPassword: str


def _set_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        AUTH.cookie_name, token,
        max_age=AUTH.session_days * 86400,
        httponly=True,                
        secure=AUTH.cookie_secure,    
        samesite="lax",               
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


@router.patch("/auth/profile")
def auth_update_profile(body: ProfileRequest, request: Request,
                        user: dict = Depends(require_user)):
    try:
        updated = get_auth().update_profile(user["user_id"], body.name,
                                            body.studentId, body.role)
    except AuthError as exc:
        raise HTTPException(400, str(exc))

    get_audit().write(audit_mod.PROFILE_UPDATE, user_id=user["user_id"],
                      email=user.get("email", ""), ip=client_ip(request),
                      detail={"role": updated.get("role"),
                              "student_id": updated.get("studentId")})
    return {"user": updated}


@router.post("/auth/password")
def auth_change_password(body: PasswordRequest, request: Request,
                         user: dict = Depends(require_user)):
    try:
        get_auth().change_password(user["user_id"], body.currentPassword,
                                   body.newPassword)
    except AuthError as exc:
        raise HTTPException(400, str(exc))

    token = request.cookies.get(AUTH.cookie_name, "")
    revoked = get_auth().revoke_others(user["user_id"], token)
    get_audit().write(audit_mod.PASSWORD_CHANGE, user_id=user["user_id"],
                      email=user.get("email", ""), ip=client_ip(request),
                      detail={"other_sessions_revoked": revoked})
    return {"ok": True, "other_sessions_revoked": revoked}


@router.get("/auth/me")
def auth_me(request: Request):
    user = current_user(request)
    if not user:
        return {"user": None}
    return {"user": public_user(user)}


@router.get("/reports")
def reports(request: Request, limit: int = Query(25, ge=1, le=100)):
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


def require_admin(request: Request) -> dict:
    user = require_user(request)
    if not user.get("is_admin"):
        raise HTTPException(403, "Administrator access is required to "
                                 "change the detection model.")
    return user


class ActivateRequest(BaseModel):
    checkpoint: str


@router.get("/models")
def list_models(user: dict = Depends(require_admin)):
    from app import modelreg
    engine = get_engine()
    return {"active": engine.model_version, "mode": engine.mode,
            "models": modelreg.discover()}


@router.post("/models/validate")
def validate_model(body: ActivateRequest, user: dict = Depends(require_admin)):
    from pathlib import Path as _P

    from app import modelreg
    root = _P(INFER.checkpoint).resolve().parent.parent
    path = (root / body.checkpoint).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError:
        raise HTTPException(400, "That checkpoint is outside runs/.")
    return modelreg.validate(path)


@router.post("/models/activate")
def activate_model(body: ActivateRequest, request: Request,
                   user: dict = Depends(require_admin)):
    from app import modelreg
    engine = get_engine()
    result = modelreg.activate(engine, body.checkpoint)

    get_audit().write(
        audit_mod.MODEL_ACTIVATED if result.get("ok")
        else audit_mod.MODEL_REJECTED,
        user.get("user_id", ""), user.get("email", ""), client_ip(request),
        {"checkpoint": body.checkpoint,
         "ok": bool(result.get("ok")),
         "reason": result.get("reason"),
         "previous": result.get("previous"),
         "active": result.get("active")})

    if not result.get("ok"):
        raise HTTPException(422, str(result.get("reason", "rejected")))
    return result


@router.get("/audit")
def audit_feed(limit: int = Query(50, ge=1, le=200),
               user: dict = Depends(require_user)):
    return {"events": get_audit().for_user(user["user_id"], limit)}


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
                raise HTTPException(413, f"That file is over "
                                         f"{AUTH.max_upload_mb} MB. Trim the "
                                         "clip and try again.")
            tmp.write(chunk)
        tmp.close()

        digest = sha256.hexdigest()
        engine = get_engine()

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

_ORIGINS = [o.strip() for o in
            os.getenv("DFD_ALLOWED_ORIGINS",
                      os.getenv("DF_ALLOWED_ORIGINS", "")).split(",")
            if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ORIGINS,
    allow_origin_regex=r"^chrome-extension://[a-p]{32}$",
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)
app.add_middleware(security_mod.SecurityHeadersMiddleware)
app.include_router(router, prefix="/api")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.on_event("startup")
def on_startup():
    from app.modelfetch import ensure_calibration, ensure_checkpoint
    fetch = ensure_checkpoint()
    ensure_calibration()

    eng = get_engine()
    print(f"[DEFAULT ENGINE] Neural Deepfake Model Loaded: {eng.model_version} | Mode: {eng.mode} | Backbone: {eng.backbone}")

    if eng.mode != "model":
        print("[model] " + "!" * 62)
        print(f"[model] DEGRADED: running the classical baseline, not the "
              f"trained model.\n[model] Reason: {fetch.get('reason')}\n"
              "[model] Results will NOT match the accuracy reported in the "
              "report.")
        print("[model] " + "!" * 62)

    get_store()
    get_auth()
    get_audit()

    store = get_store()
    if store and not store.durable:
        print("[store] WARNING: DynamoDB is not connected. Findings are "
              "going to a local file, which does NOT survive a redeploy. "
              f"Reason: {store.error or 'see /api/status'}")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.server:app", host="0.0.0.0",
                port=int(os.getenv("PORT", "8000")), reload=False)
