# Makefile for Quant Trading Portfolio Project

.PHONY: help install clean test run-backtest run-track-a run-track-b docs build

# Default target
help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

# Installation
install: ## Install dependencies
	pip install -e .

install-dev: ## Install development dependencies
	pip install -e ".[dev]"

# Cleaning
clean: ## Clean cache files and temporary data
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +

# Testing
test: ## Run tests
	pytest tests/ -v

test-smoke: ## Run smoke tests (quick validation)
	python -c "import src; print('✅ Package imports successfully')"
	python -c "from src.utils.config import load_config; cfg = load_config(); print('✅ Config loads successfully')"

# Running
run-track-a: ## Run Track A (Ranking Engine)
	python -m src.cli track-a --config configs/config.yaml

run-track-b: ## Run Track B (Backtest)
	python -m src.cli track-b --config configs/config.yaml --strategy bt120_long

run-backtest: ## Run full backtest pipeline
	python -m src.cli run --config configs/config.yaml

# Documentation
docs: ## Generate documentation
	@echo "Documentation commands:"
	@echo "  - View README.md for project overview"
	@echo "  - View docs/ folder for detailed documentation"
	@echo "  - Run reports: python scripts/generate_reports.py"

# Build and deploy
build: ## Build package
	python -m build

# Development helpers
lint: ## Run linting
	flake8 src/ tests/ --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 src/ tests/ --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

format: ## Format code with black
	black src/ tests/

check-format: ## Check code formatting
	black --check src/ tests/

# Data management
download-data: ## Download raw data
	python -m src.cli data-download --config configs/config.yaml

validate-data: ## Validate data integrity
	python -m src.cli data-validate --config configs/config.yaml