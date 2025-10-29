@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
  set "PY=.venv\Scripts\python.exe"
) else (
  set "PY=python"
)

if not exist "logs" mkdir "logs"

%PY% Scripts\retrain.py >> logs\retrain.log 2>&1
echo ExitCode=%ERRORLEVEL% >> logs\retrain.log

endlocal