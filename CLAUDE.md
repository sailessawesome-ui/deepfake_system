# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository layout gotcha

The git repo root (this file's directory) contains one real project one level down: `deepfake_system/`. Every command below assumes you have `cd`'d into it first. This has already caused a real deployment failure once (Railway looked for `requirements-deploy.txt` at the repo root and failed) — if you're writing deploy config, a path, or a working-directory assumption, get the nesting right.

The repo root also holds a file literally named `env` (no dot) containing AWS credentials. It is gitignored. `config.py` loads it automatically via `app/envfile.py` before reading any setting — see "Configuration" below.

## Commands

Run from `deepfake_system/`.

**Start the server:**
```bash
./run.sh                    # http://127.0.0.1:8000, creates .venv, installs requirements-web.txt
./run.sh 0.0.0.0 8080       # reachable from your network
python -m uvicorn app.server:app --host 127.0.0.1 --port 8000   # manual, once deps are installed
```
Windows: `run.bat`, or `python scripts/dev_tls.py` to serve over local HTTPS (self-signed cert, for testing the Secure-cookie / HSTS path).

**Install:**
```bash
pip install -r requirements-web.txt    # serving only — no torch, runs the classical baseline
pip install -r requirements.txt        # + training/eval (torch, timm, facenet-pytorch, mediapipe)
pip install -r requirements-deploy.txt # what Railway installs — CPU-only torch, no training deps
```

**Train / evaluate:**
```bash
python -m data.build_manifest --dry-run   # check detected dataset layout first
python -m data.build_manifest             # write manifest.csv
python train.py --epochs 22 --batch-size 8
python evaluate.py --checkpoint runs/v1/best.pt --split test      # the unbiased number
python evaluate.py --checkpoint runs/v1/best.pt --split holdout   # cross-source/generator generalisation
```
Then copy `best.pt` and `calibration.json` into `runs/v1/` and restart the server — nothing else changes. `runs/` is gitignored.

**AWS / deploy:**
```bash
python scripts/create_tables.py [--status]   # idempotent; creates only dfd_audit_log, the rest already exist
python scripts/upload_model_s3.py            # uploads best.pt + calibration.json, verifies byte count, prints Railway env vars
```

**No test suite exists in this repo** (no `pytest` files, no CI config). Verification has been done by running the FastAPI app in-process via `TestClient` and hitting the live DynamoDB tables directly — there is no `npm test` / `pytest` equivalent to reach for.

## Architecture

### Configuration and environment loading

`config.py` is the single source of truth — seven dataclasses (`DATA`, `MODEL`, `AUDIO`, `STORE`, `AUTH`, `TRAIN`, `INFER`), instantiated once at import time. Before any of them are built, it imports `app/envfile.py`, which finds and loads the repo-root `env` file into `os.environ` (values already set in the environment win — this matters on Railway, where service variables must not be overridden by a stale file). Settings are read via `_env(*names, default=...)`, which checks a list of aliases in order — this is why you'll see both `DFD_*` (the `env` file's naming) and `DF_*` (an earlier naming) accepted for the same setting throughout.

### Storage: DynamoDB-backed, with a local JSON fallback

`app/tables.py` is the one place that knows every table's name, key schema, and index — `get(logical_name)` returns either a real DynamoDB-backed `Table` (`app/dynamo_table.py`) or a local-JSON one (`app/local_table.py`) with an identical method surface (`put`, `get`, `delete`, `scan`, `query`, `query_index`, `status`). Callers never branch on backend; they call `tables.get("users")` and use whatever comes back. This fallback exists for laptop development with no AWS account — it is *not* a deployment target, since a container filesystem is ephemeral.

Five tables, all in the AWS account this project is bound to (region `ap-southeast-5`, prefix `dfd_`). **Four of these already held production data before this codebase's DynamoDB integration was written — the code was built to match their existing schemas, not the reverse:**

| Table | Key | Notes |
|---|---|---|
| `dfd_users` | `user_id` (random UUIDv4) | Looked up via GSI `EmailIndex` — `user_id` is *not* derivable from email, unlike an earlier design that would have derived it deterministically |
| `dfd_sessions` | `token_hash` (SHA-256 of the session token) | GSI `UserSessionsIndex`; TTL on `expires_at`. The raw token is never stored — only its hash, so a table dump can't be replayed as a live session |
| `dfd_analyses` | `user_id` + `analysis_id` | Detection results. Carries both IR-spec field names (`detection_id`, `classification`, `confidence_score`) and shorter legacy names (`verdict`, `fake_probability`) for the same values — written by `app/store.py:build_record()` |
| `dfd_login_attempts` | `identifier` + `attempted_at` (microseconds — see below) | TTL on `ttl`. Separate from the user row so a failed attempt against a *non-existent* account is still recorded |
| `dfd_audit_log` | `audit_day` + `event_ts` | GSI `UserAuditIndex`; TTL on `expires_at`. Partitioned by UTC date so "everything on one day" is a `Query`, not a `Scan` |

`attempted_at` in `dfd_login_attempts` is stored in **microseconds**, not seconds — it's a range key, and multiple failed logins inside the same second would otherwise collide and overwrite each other, silently defeating lockout. If you touch lockout logic, preserve that.

Because email uniqueness in `dfd_users` can't be enforced atomically by DynamoDB (it's a GSI lookup, not the partition key), registration does a read-then-write that is theoretically racy under simultaneous sign-ups for the same address. Known and accepted at this scale.

### The detection engine (`app/engine.py`)

`Engine` picks between two modes at load time and exposes the same `analyse(video_path, filename) -> dict` either way:

- **`model`** — the trained CNN (`models/net.py: DeepfakeDetector`), loaded from `INFER.checkpoint` (default `runs/v1/best.pt`, gitignored — must be trained locally or fetched via `app/modelfetch.py` from S3 in deployment).
- **`heuristic`** (`app/heuristic.py`) — a classical signal-based fallback used automatically when no checkpoint is present. The interface is identical in both modes; only `model_version` and `engine` in the response differ.

`analyse()` pipeline, in order: probe container metadata → `provenance.inspect()` (C2PA/JUMBF watermark check) → `FaceExtractor.extract()` (MTCNN, with Haar/MediaPipe fallback) → clip sampling and scoring (`_model_scores`, with flip-TTA and top-k/trimmed-mean aggregation across `INFER.clips_per_video` clips) → `audio.analyse()` (lip-sync correlation + synthetic-voice indicators, independent signal — a lip-sync *mismatch* widens the confidence margin rather than pushing the verdict, because dubbing tools deliberately drive the mouth from audio) → `_explain()` (Grad-CAM on the top `INFER.explain_frames` frames only, since each explained frame costs an extra forward+backward pass).

**Verdict is three-state by design** (`authentic` / `manipulated` / `inconclusive`, plus `no_face` when no face is found), not binary — see `IR_COMPLIANCE.md` for the reasoning; `INFER.strict_binary` forces a binary call if needed. This is a deliberate, documented deviation from a literal spec requirement.

### Model hot-swap (`app/modelreg.py`)

Checkpoints under `runs/*.pt` can be listed, validated, and activated at runtime via admin-only endpoints (`/api/models`, `/api/models/validate`, `/api/models/activate`) without restarting the process. Validation loads the candidate as an *isolated* model and scores a synthetic probe clip before anything is swapped — this catches architecture mismatches and NaN-producing checkpoints that would otherwise pass every structural check and silently poison every verdict. The swap itself is atomic under a lock so an in-flight analysis finishes on the model it started with. Checkpoint IDs are path-validated against `runs/` to prevent traversal.

### Auth, sessions, and audit (`app/auth.py`, `app/audit.py`, `app/server.py`)

Session tokens are opaque random strings; only their SHA-256 hash reaches DynamoDB or the cookie-verification path never needs the raw value server-side. `require_user()` (a FastAPI dependency) gates `/api/analyse*`; `require_admin()` additionally checks `is_admin` on the user record and gates the model-swap endpoints. Every auth event and every completed/failed analysis is written to `dfd_audit_log` via `app/audit.py` — this is the "system logs" clause referenced in `IR_COMPLIANCE.md`'s IR-requirement mapping, not a general-purpose logger.

### Security headers (`app/security.py`)

A `BaseHTTPMiddleware` adds CSP, HSTS (only over a connection it detects as already secure via `X-Forwarded-Proto`), and related headers to every response. The CSP's `script-src` is `'self'` with no `unsafe-inline`/`unsafe-eval` — this is enforceable because the frontend genuinely has zero inline `<script>` tags; if you ever add one, the CSP will silently block it in the browser, not error server-side.

### Zero retention

Enforced in `app/server.py`: uploaded video goes to a temp file, is scored, and is deleted in a `finally` block. Never written elsewhere, never in the analyses table. Face crops and Grad-CAM overlays are returned to the browser as base64 data URIs in the JSON response and are explicitly stripped (`_STRIP` set in `app/store.py`) before anything is persisted.

### Frontend

`app/static/` is plain HTML/CSS/JS served directly by FastAPI's `StaticFiles` — no build step, no bundler, no framework. `app/static/app.js` talks to the API under `/api/*` using cookie-based sessions (`credentials: 'same-origin'`).

### Deployment

Railway-oriented (`nixpacks.toml`, `railway.json`, `Procfile`) — deploys the `deepfake_system/` subdirectory (Railway's "Root Directory" setting must point there). `app/modelfetch.py` downloads the checkpoint + calibration file from S3 at startup, since `runs/` is gitignored and can't ship via git. `DEPLOY_RAILWAY.md` has the full runbook, including which environment variables control what.

### Training/eval data discipline

`data/build_manifest.py` splits by **video identity**, never by frame — the most common way a deepfake-detection FYP result turns out to be meaningless is splitting frames randomly and leaking the same face's frames across train/test. `DATA.holdout_sources` / `DATA.holdout_methods` in `config.py` name entire dataset sources or DF40 generator families to exclude from training entirely, so `evaluate.py --split holdout` produces a genuine cross-distribution generalisation number rather than an in-distribution one. `data/degrade.py` simulates messenger (WhatsApp-style) re-compression during training; `TRAIN.consistency_weight` penalizes the model for disagreeing between clean and degraded views of the same clip.

## Reference documents

`IR_COMPLIANCE.md` maps every requirement from the project's Investigation Report to where it's implemented in code, and explicitly flags deliberate deviations (e.g., the three-state verdict) and known gaps. Read it before making changes that touch requirement-driven behavior (auth gating, retention, the audio/provenance branches, explainability) — it records *why* those decisions were made, not just what the code does.
