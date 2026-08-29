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

from fastapi import APIRouter, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import STORE  # noqa: E402
from app.engine import Engine  # noqa: E402
from app.store import ReportStore, build_record  # noqa: E402

STATIC = Path(__file__).parent / "static"
MAX_BYTES = 250 * 1024 * 1024
ALLOWED = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".3gp", ".m4v"}

router = APIRouter()
_engine: Engine | None = None
_store: ReportStore | None = None

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
        _store = ReportStore(STORE.table_name, STORE.region,
                             STORE.local_path, STORE.use_dynamodb)
        print(f"[store] {_store.status()}")
    return _store


@router.get("/status")
def status():
    out = get_engine().status()
    store = get_store()
    out["store"] = store.status() if store else {"backend": "disabled"}
    return out


@router.get("/reports")
def reports(limit: int = Query(25, ge=1, le=100)):
    """Past findings. The videos are gone; the reports remain."""
    store = get_store()
    if not store:
        return {"reports": [], "backend": "disabled"}
    return {"reports": store.recent(limit), "backend": store.backend}


_LATEST_RESULT: dict | None = None


@router.get("/reports/latest")
def latest_report():
    if not _LATEST_RESULT:
        raise HTTPException(404, "No verification report generated yet.")
    return _LATEST_RESULT


@router.get("/reports/by-hash/{sha256}")
def reports_by_hash(sha256: str):
    if not re.fullmatch(r"[0-9a-fA-F]{64}", sha256):
        raise HTTPException(400, "That is not a SHA-256 digest.")
    store = get_store()
    if not store:
        return {"reports": [], "backend": "disabled"}
    return {"reports": store.find_by_hash(sha256.lower())}


from pydantic import BaseModel


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
async def analyse_url(req: UrlRequest):
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
            record = build_record(result, result["filename"], engine.model_version)
            result["storage"] = store.save(record)
            result["report_id"] = record["report_id"]
            prior = [r for r in store.find_by_hash(digest)
                     if r.get("report_id") != record["report_id"]]
            if prior:
                result["seen_before"] = len(prior)

        global _LATEST_RESULT
        _LATEST_RESULT = result
        return result
    except Exception as exc:
        raise HTTPException(400, f"Could not stream video from {url}: {exc}")
    finally:
        try:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except OSError:
            pass


@router.post("/analyse")
async def analyse(video: UploadFile = File(...)):
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
            record = build_record(result, name, engine.model_version)
            result["storage"] = store.save(record)
            result["report_id"] = record["report_id"]
            prior = [r for r in store.find_by_hash(digest)
                     if r.get("report_id") != record["report_id"]]
            if prior:
                result["seen_before"] = len(prior)
                result["notes"].append(
                    f"This exact file has been checked here {len(prior)} time"
                    f"{'s' if len(prior) > 1 else ''} before. The earlier "
                    "reports are under Past findings.")

        global _LATEST_RESULT
        _LATEST_RESULT = result
        return result
    finally:
        tmp.close()
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Deepfake Forensics", description="BSc Cybersecurity Capstone System", docs_url=None, redoc_url=None)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router, prefix="/api")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.on_event("startup")
def on_startup():
    eng = get_engine()
    print(f"[DEFAULT ENGINE] Neural Deepfake Model Loaded: {eng.model_version} | Mode: {eng.mode} | Backbone: {eng.backbone}")
    get_store()


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.server:app", host="0.0.0.0", port=8000, reload=False)
