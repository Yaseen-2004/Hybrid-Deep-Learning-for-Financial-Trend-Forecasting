@echo off
setlocal
cd /d "%~dp0"
where docker >nul 2>&1
if errorlevel 1 (
  echo Docker was not found. Install and start Docker Desktop first.
  pause
  exit /b 1
)
docker compose up --build -d
if errorlevel 1 (
  echo Deployment failed. Review Docker output above.
  pause
  exit /b 1
)
echo.
echo Deployment started. Open http://localhost:8501
echo View logs using: docker compose logs -f financial-trend-forecasting
pause
