@echo off
setlocal
cd /d "%~dp0"
if not exist .venv (
    echo Creating virtual environment...
    py -3.12 -m venv .venv 2>nul || python -m venv .venv
)
call .venv\Scripts\activate.bat
pip install -q -r requirements-web.txt
echo Frame Zero on http://127.0.0.1:8000
python -m uvicorn app.server:app --host 127.0.0.1 --port 8000
