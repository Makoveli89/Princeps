#!/bin/bash
set -e

echo "🔮 Princeps Platform Activation Sequence Initiated..."

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 is not installed."
    exit 1
fi

# Check for Node/npm
if ! command -v npm &> /dev/null; then
    echo "❌ Error: npm is not installed. Please install Node.js."
    exit 1
fi

echo "📦 Installing Backend Dependencies..."
pip install fastapi uvicorn

echo "📦 Installing Frontend Dependencies..."
cd apps/console
npm install
cd ../..

echo "🚀 Launching Services..."

# Start Backend in background
echo "   - Starting Backend Server (Port 8000)..."
python3 server.py > backend.log 2>&1 &
BACKEND_PID=$!

# Start Frontend
echo "   - Starting Frontend Console (Port 5173)..."
echo "   - Access the console at: http://localhost:5173"
cd apps/console
npm run dev

# Cleanup on exit
kill $BACKEND_PID
