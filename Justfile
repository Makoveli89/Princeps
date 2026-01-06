# Justfile for Princeps Brain V2
# Modern task runner to replace start.sh/start.bat
# Documentation: https://github.com/casey/just

# Load environment variables from .env if it exists
set dotenv-load

# Default recipe - list available commands
default:
    @just --list

# Install all dependencies (backend and frontend)
install:
    @echo "📦 Installing Backend Dependencies..."
    pip install -e ".[all]"
    pip install fastapi uvicorn
    @echo "📦 Installing Frontend Dependencies..."
    cd apps/console && npm install

# Run the full stack in parallel
# Uses just's parallel execution (-j2) to run backend and frontend simultaneously
run:
    @echo "🚀 Launching Services..."
    just -j2 run-backend run-frontend

# Run the backend server only
run-backend:
    @echo "   - Starting Backend (Port 8000)..."
    python server.py

# Run the frontend console only
run-frontend:
    @echo "   - Starting Frontend (Port 5173)..."
    cd apps/console && npm run dev

# Run all tests (currently backend only)
test:
    @echo "🧪 Running Tests..."
    pytest

# Run linters and formatters
lint:
    @echo "🧹 Running Linters..."
    ruff check .
    black --check .

# Apply formatting fixes
format:
    @echo "✨ Formatting Code..."
    ruff check --fix .
    black .
