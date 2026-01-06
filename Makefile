.PHONY: install run run-backend run-frontend test lint format clean help

# Default target
.DEFAULT_GOAL := help

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

install: ## Install all dependencies (backend and frontend)
	@echo "📦 Installing Backend Dependencies..."
	pip install -e ".[all]"
	pip install fastapi uvicorn
	@echo "📦 Installing Frontend Dependencies..."
	cd apps/console && npm install

run: ## Run the full stack (backend and frontend). Use 'make -j2 run' for parallel execution.
	@echo "🚀 Launching Services..."
	@$(MAKE) -j2 run-backend run-frontend

run-backend: ## Run the backend server only
	@echo "   - Starting Backend (Port 8000)..."
	python server.py

run-frontend: ## Run the frontend console only
	@echo "   - Starting Frontend (Port 5173)..."
	cd apps/console && npm run dev

test: ## Run tests
	@echo "🧪 Running Tests..."
	pytest

lint: ## Run linters
	@echo "🧹 Running Linters..."
	ruff check .
	black --check .

format: ## Run formatters
	@echo "✨ Formatting Code..."
	ruff check --fix .
	black .

clean: ## Clean up artifacts
	@echo "🗑️ Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
