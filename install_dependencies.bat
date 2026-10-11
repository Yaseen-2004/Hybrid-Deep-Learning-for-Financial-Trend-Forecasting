@echo off
setlocal
cd /d "%~dp0"
echo ===============================================
echo Financial Trend Forecasting - Setup
echo ===============================================
if not exist ".venv\Scripts\python.exe" (
  echo Creating a Python 3.11 virtual environment...
  py -3.11 -m venv .venv
  if errorlevel 1 (
    echo ERROR: Python 3.11 was not found or the virtual environment could not be created.
    echo Install 64-bit Python 3.11 and try again.
    pause
    exit /b 1
  )
)
call ".venv\Scripts\activate.bat"
if errorlevel 1 (
  echo ERROR: Could not activate .venv.
  pause
  exit /b 1
)
python -m pip install --upgrade pip
if errorlevel 1 goto failed
python -m pip install -r requirements-training.txt
if errorlevel 1 goto failed
echo.
echo Setup completed.
echo Next, run run_tests.bat or train_quick.bat.
pause
exit /b 0
:failed
echo.
echo ERROR: Dependency installation failed. Check the messages above.
pause
exit /b 1
