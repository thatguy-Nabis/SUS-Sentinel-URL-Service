@echo off
title SUS Sentinel - Stopping Local Servers

echo ==========================================================
echo         SUS Sentinel - Stopping Local Servers
echo ==========================================================
echo.

echo [*] Terminating process on Port 8000 (Backend API)...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8000" ^| findstr "LISTENING"') do (
    taskkill /F /PID %%a 2>nul
)

echo [*] Terminating process on Port 5173 (Frontend Vite UI)...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":5173" ^| findstr "LISTENING"') do (
    taskkill /F /PID %%a 2>nul
)

echo.
echo [*] Done! Local servers stopped.
echo ==========================================================
timeout /t 3 >nul
