# ── Stage 1: GDAL + Python base ──────────────────────────────
FROM ghcr.io/osgeo/gdal:ubuntu-full-3.8.5 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FIELD_ENV=prod \
    FIELD_LOG_LEVEL=INFO

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-dev libgl1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Stage 2: Install deps ───────────────────────────────────
COPY pyproject.toml ./
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir 'git+https://github.com/facebookresearch/detectron2.git' && \
    pip install --no-cache-dir .

# ── Stage 3: Copy inference code only (no training) ─────────
COPY src/data/       src/data/
COPY src/inference/  src/inference/
COPY src/export/     src/export/
COPY src/monitoring/ src/monitoring/
COPY src/__init__.py src/__init__.py
COPY scripts/infer.py scripts/
COPY configs/ configs/

# Health check
HEALTHCHECK --interval=30s --timeout=5s \
    CMD python3 -c "from src.monitoring.health import check; check()" || exit 1

ENTRYPOINT ["python3", "scripts/infer.py"]
CMD ["-b", "configs/batch.yaml"]
