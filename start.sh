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

# Check for Docker (Optional but recommended for DB)
if ! command -v docker &> /dev/null; then
    echo "⚠️  Warning: Docker is not installed. You will need a running Postgres instance."
else
    # Check if a postgres container is running?
    # For now, just warn if DB is not accessible.
    pass=true
fi

echo "📦 Installing Backend Dependencies..."
# We install the package in editable mode to ensure all deps (brain, framework) are available
pip install -e ".[all]"
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

# Wait a moment for backend to maybe print DB error
sleep 2
if grep -q "Database initialization failed" backend.log; then
    echo "⚠️  Backend reported database issues. Check backend.log."
    echo "   Ensure you have a PostgreSQL database running and DATABASE_URL set."
    echo "   Example: export DATABASE_URL=postgresql://user:pass@localhost:5432/princeps"
fi

# Start Frontend
echo "   - Starting Frontend Console (Port 5173)..."
echo "   - Access the console at: http://localhost:5173"
cd apps/console
npm run dev

# Cleanup on exit
kill $BACKEND_PID
