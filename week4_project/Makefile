# ============================================================
# FedTB-Nigeria Makefile
# Usage:  make <target>
# ============================================================

.PHONY: help env download mock test lint clean notebooks run_central run_federated

help:
	@echo "FedTB-Nigeria — available targets:"
	@echo "  env           Create conda environment"
	@echo "  download      Download real datasets (interactive)"
	@echo "  mock          Generate mock data for testing"
	@echo "  test          Run all unit tests"
	@echo "  lint          Run ruff linter on src/"
	@echo "  notebooks     Launch JupyterLab"
	@echo "  run_central   Train centralised baseline"
	@echo "  run_federated Train federated model"
	@echo "  clean         Remove __pycache__ and .pyc files"

env:
	conda env create -f environment.yml
	@echo "Run: conda activate fedtb_nigeria"

download:
	python scripts/download_data.py

mock:
	python scripts/generate_mock_data.py

test:
	pytest tests/ -v --tb=short

lint:
	ruff check src/

notebooks:
	jupyter lab notebooks/

run_central:
	python scripts/run_centralised.py

run_federated:
	python scripts/run_federated.py

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleaned."
