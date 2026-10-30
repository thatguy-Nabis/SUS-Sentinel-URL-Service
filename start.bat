@echo off
setlocal enabledelayedexpansion
title SUS Sentinel - Starting Local Servers

echo ==========================================================
echo        SUS Sentinel URL Service - Local Server Launcher
echo ==========================================================
echo.

:: Get directory of this script
set "ROOT_DIR=%~dp0"
cd /d "%ROOT_DIR%"

:: 1. Verify virtual environment exists
if not exist "%ROOT_DIR%.venv\Scripts\python.exe" (
    echo [ERROR] Python virtual environment not found in .venv!
    echo Please create it first by running:
    echo   python -m venv .venv
    echo   .venv\Scripts\pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

:: 2. Verify node_modules exists in frontend
if not exist "%ROOT_DIR%frontend\node_modules" (
    echo [*] Frontend dependencies missing. Running npm install...
    cd /d "%ROOT_DIR%frontend"
    call npm install
    cd /d "%ROOT_DIR%"
)

:: 3. Launch Backend in a dedicated terminal window
echo [*] Launching Backend API (FastAPI / Uvicorn on port 8000)...
start "SUS Sentinel - Backend API" cmd /k "title SUS Sentinel - Backend (Port 8000) && cd /d "%ROOT_DIR%" && .venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload"

:: 4. Launch Frontend in a dedicated terminal window
echo [*] Launching Frontend Web App (Vite on port 5173)...
start "SUS Sentinel - Frontend UI" cmd /k "title SUS Sentinel - Frontend (Port 5173) && cd /d "%ROOT_DIR%frontend" && npm run dev"

echo.
echo ==========================================================
echo              Servers Launched Successfully!
echo ==========================================================
echo  - Backend API:       http://localhost:8000
echo  - Interactive Docs:  http://localhost:8000/docs
echo  - Health Endpoint:   http://localhost:8000/v1/health
echo  - Frontend UI:       http://localhost:5173
echo.
echo Dedicated terminal windows have been opened for both servers.
echo You can view real-time logs in each window.
echo Press Ctrl+C in a server window to stop that service.
echo ==========================================================
echo.

:: 5. Prompt to open in browser
set /p OPEN_BROWSER="Open Frontend and Swagger Docs in your browser now? (Y/N, default Y): "
if /i "%OPEN_BROWSER%"=="" set OPEN_BROWSER=Y
if /i "%OPEN_BROWSER%"=="Y" (
    timeout /t 2 /nobreak >nul
    start http://localhost:5173
    start http://localhost:8000/docs
)

echo.
echo Launcher finished. You can close this window.
timeout /t 5 >nul
