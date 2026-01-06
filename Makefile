.PHONY: help install lint format test clean run run-docker

PYTHON := python
PIP := pip
UVICORN := uvicorn

help:  ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	$(PIP) install -r requirements.txt
	$(PIP) install pre-commit
	pre-commit install

lint: ## Run linting (Ruff)
	ruff check .

format: ## Run formatting (Ruff)
	ruff check --fix .
	ruff format .

test: ## Run tests
	pytest

clean: ## Clean up cache and build artifacts
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache

run: ## Run the backend server locally
	$(UVICORN) server:app --host 0.0.0.0 --port 8000 --reload

run-docker: ## Build and run the docker container
	docker build -t princeps-brain .
	docker run -p 8000:8000 princeps-brain
