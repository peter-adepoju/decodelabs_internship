# ─── E-Commerce Customer Intelligence ──────────────────────────────────────────
# Usage:
#   make setup       — create environment and install dependencies
#   make download    — download the Olist dataset from Kaggle
#   make notebooks   — run all notebooks in order (requires data to be downloaded)
#   make test        — run all tests with coverage
#   make dashboard   — launch the Streamlit dashboard
#   make clean       — remove generated artifacts (figures, tables, models)
#   make lint        — run black + ruff
#   make all         — setup + download + notebooks + test

.PHONY: setup download notebooks test dashboard clean lint all help

PYTHON := python
JUPYTER := jupyter
NOTEBOOKS_DIR := notebooks
SCRIPTS_DIR := scripts

help:
	@echo "Available targets:"
	@echo "  setup      — install dependencies"
	@echo "  download   — download Olist dataset (requires Kaggle API key)"
	@echo "  notebooks  — execute all notebooks in order"
	@echo "  test       — run pytest with coverage"
	@echo "  dashboard  — launch Streamlit app"
	@echo "  clean      — remove generated artifacts"
	@echo "  lint       — run black + ruff"
	@echo "  all        — setup + download + notebooks + test"

setup:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e .
	@echo "Setup complete. Activate your environment and run: make download"

download:
	$(PYTHON) $(SCRIPTS_DIR)/download_data.py

notebooks:
	@echo "Running notebooks in order..."
	$(PYTHON) $(SCRIPTS_DIR)/run_all.py

test:
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html:reports/coverage

dashboard:
	streamlit run website_or_demo/app.py

lint:
	black src/ tests/ scripts/ website_or_demo/
	ruff check src/ tests/ scripts/ website_or_demo/ --fix

clean:
	find reports/figures -name "*.png" -delete 2>/dev/null || true
	find reports/figures -name "*.pdf" -delete 2>/dev/null || true
	find reports/tables -name "*.csv" -delete 2>/dev/null || true
	find reports/tables -name "*.xlsx" -delete 2>/dev/null || true
	find models -name "*.joblib" -delete 2>/dev/null || true
	find data/interim -name "*.parquet" -delete 2>/dev/null || true
	find data/processed -name "*.parquet" -delete 2>/dev/null || true
	find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	@echo "Clean complete."

all: setup download notebooks test
	@echo "Full pipeline complete."
