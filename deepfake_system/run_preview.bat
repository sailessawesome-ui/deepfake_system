@echo off
cd /d "%~dp0"
set DFD_DB_BACKEND=local
set DFD_ENV_QUIET=1
".venv\Scripts\python.exe" -m uvicorn app.server:app --host 127.0.0.1 --port 8000
