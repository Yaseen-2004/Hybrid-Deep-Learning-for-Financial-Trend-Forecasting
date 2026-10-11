@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\activate.bat" (
  echo Virtual environment not found. Run install_dependencies.bat first.
  pause
  exit /b 1
)
call ".venv\Scripts\activate.bat"
python train.py --ticker AAPL --period 5y --epochs 20 --walk-forward-folds 2
if errorlevel 1 (
  echo Training did not complete. Copy the full output for troubleshooting.
  pause
  exit /b 1
)
echo Full training run completed.
pause
