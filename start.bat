@echo off
setlocal

echo Princeps Platform Activation Sequence Initiated...

REM Check for Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: python is not installed or not in your PATH.
    pause
    exit /b 1
)

REM Check for Node/npm
npm --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: npm is not installed. Please install Node.js.
    pause
    exit /b 1
)

echo Installing Backend Dependencies...
pip install -e ".[all]"
if %errorlevel% neq 0 (
    echo Error installing backend dependencies.
    pause
    exit /b 1
)
pip install fastapi uvicorn
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
start "Princeps Backend" cmd /k "python server.py"

REM Start Frontend in this window
echo    - Starting Frontend Console (Port 5173)...
echo    - Access the console at: http://localhost:5173
cd apps\console
call npm run dev

REM When frontend exits, we are done.
echo.
echo Frontend stopped.
echo You may need to manually close the Backend window.
pause
