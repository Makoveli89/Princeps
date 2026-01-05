@echo off
setlocal

echo Princeps Platform Activation Sequence Initiated...

REM Check for Python (prefer python3, fallback to python)
python3 --version >nul 2>&1
if %errorlevel% equ 0 (
    set PYTHON_CMD=python3
) else (
    python --version >nul 2>&1
    if %errorlevel% equ 0 (
        set PYTHON_CMD=python
    ) else (
        echo Error: python is not installed or not in your PATH.
        pause
        exit /b 1
    )
)

REM Check for Node/npm
npm --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: npm is not installed. Please install Node.js.
    pause
    exit /b 1
)

REM Check for Docker (optional, but recommended for local PostgreSQL)
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Warning: Docker is not installed or not in your PATH.
    echo A running PostgreSQL instance is required for Princeps.
    echo Please ensure PostgreSQL is available before starting the platform.
    echo.
)

echo Installing Backend Dependencies...
%PYTHON_CMD% -m pip install -e ".[all]"
if %errorlevel% neq 0 (
    echo Error installing backend dependencies.
    pause
    exit /b 1
)
%PYTHON_CMD% -m pip install fastapi uvicorn
if %errorlevel% neq 0 (
    echo Error installing server dependencies.
    pause
    exit /b 1
)

echo Installing Frontend Dependencies...
cd apps\console
call npm install
if %errorlevel% neq 0 (
    echo Error installing frontend dependencies.
    cd ..\..
    pause
    exit /b 1
)
cd ..\..

echo Launching Services...

REM Start Backend in a new window
echo    - Starting Backend Server (Port 8000)...
start "Princeps Backend" cmd /k "%PYTHON_CMD% server.py"

echo    - Note: Ensure PostgreSQL is running. If the backend window shows
echo      database connection errors, start PostgreSQL and rerun this script.

REM Start Frontend in this window
echo    - Starting Frontend Console (Port 5173)...
echo    - Access the console at: http://localhost:5173
cd apps\console
call npm run dev

REM When frontend exits, attempt to stop the backend server.
echo.
echo Frontend stopped. Shutting down backend server window...
taskkill /FI "WINDOWTITLE eq Princeps Backend" /T /F >nul 2>&1
echo Backend shutdown complete. You can now safely close this window.
pause
