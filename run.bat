@echo off
REM Change to the directory of this .bat
cd /d "%~dp0"

REM If venv does not exist, we still let the user run main.py
REM but the first menu option ("Install dependencies") will create it.
set VENV_DIR=%~dp0.venv

REM Prefer venv python if it exists
if exist "%VENV_DIR%\Scripts\python.exe" (
    "%VENV_DIR%\Scripts\python.exe" main.py
) else (
    echo [INFO] No .venv found yet. We'll use system Python.
    python main.py
)

pause
