@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\activate.bat" (
  echo Virtual environment not found. Run install_dependencies.bat first.
  pause
  exit /b 1
)
call ".venv\Scripts\activate.bat"
python -m pytest -q
if errorlevel 1 (
  echo Tests failed. Copy the full output for troubleshooting.
  pause
  exit /b 1
)
echo All tests passed.
pause
