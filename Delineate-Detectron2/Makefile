.PHONY: help install install-train install-dev lint format test docker-build docker-run clean

PYTHON ?= python3
IMAGE_NAME ?= field-delineation
IMAGE_TAG  ?= latest

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ── Installation ─────────────────────────────────────────────
install:  ## Install inference-only deps
	$(PYTHON) -m pip install -e .

install-train:  ## Install with training deps
	$(PYTHON) -m pip install -e ".[train]"

install-dev:  ## Install with dev + train deps
	$(PYTHON) -m pip install -e ".[train,dev]"
	pre-commit install || true

# ── Quality ──────────────────────────────────────────────────
lint:  ## Run linter
	$(PYTHON) -m ruff check src/ scripts/ tests/

format:  ## Auto-format code
	$(PYTHON) -m ruff format src/ scripts/ tests/

test:  ## Run tests
	$(PYTHON) -m pytest tests/ -v --tb=short

# ── Training ─────────────────────────────────────────────────
train:  ## Run training
	$(PYTHON) scripts/train.py -c configs/train.yaml

# ── Inference ────────────────────────────────────────────────
infer:  ## Run single-folder inference
	$(PYTHON) scripts/infer.py -c configs/inference.yaml

batch:  ## Run batch inference
	$(PYTHON) scripts/infer.py -b configs/batch.yaml

# ── Docker ───────────────────────────────────────────────────
docker-build:  ## Build production Docker image
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) .

docker-run:  ## Run inference in Docker
	docker run --gpus all -v $$(pwd)/data:/app/data $(IMAGE_NAME):$(IMAGE_TAG)

# ── Cleanup ──────────────────────────────────────────────────
clean:  ## Remove build artifacts
	rm -rf build/ dist/ *.egg-info __pycache__ .pytest_cache .ruff_cache
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
