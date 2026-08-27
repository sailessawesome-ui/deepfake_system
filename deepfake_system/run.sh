#!/usr/bin/env bash
# Start the verification console.
#   ./run.sh              -> http://127.0.0.1:8000
#   ./run.sh 0.0.0.0 8080 -> reachable from your network
set -euo pipefail
HOST="${1:-127.0.0.1}"
PORT="${2:-8000}"
cd "$(dirname "$0")"
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
pip install -q -r requirements-web.txt
echo "Frame Zero on http://${HOST}:${PORT}"
exec python -m uvicorn app.server:app --host "$HOST" --port "$PORT"
