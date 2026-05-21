# Field Delineation Pipeline

End-to-end field boundary delineation using Detectron2 Mask R-CNN with multi-GPU inference, LCLU integration, and production monitoring.

## Project Structure

```
├── configs/                     ← All configuration
│   ├── base.yaml                   Base (data loader, filtering, output)
│   ├── train.yaml                  Training hyperparameters
│   ├── inference.yaml              Inference + LCLU + H100 tuning
│   └── batch.yaml                  Batch processing config
│
├── data/
│   ├── training/                ← Training data (Detectron2 / YOLO)
│   │   ├── images/                 train/ and val/ image tiles
│   │   ├── labels/                 YOLO-format .txt labels
│   │   └── annotations/           COCO-format JSONs
│   │
│   └── delineation/             ← Inference / production data
│       ├── images/                 Input GeoTIFFs (subfolder per area)
│       ├── masks/                  LCLU rasters (auto-matched by name)
│       ├── delineated/             Output GeoPackages
│       └── temp/                   Warped LCLU cache
│
├── scripts/                     ← CLI entry points
│   ├── train.py                    Training CLI
│   ├── infer.py                    Inference CLI (multi-GPU + LCLU)
│   └── evaluate.py                 Evaluation CLI
│
├── src/                         ← Core library
│   ├── data/                       Data loading, analysis, LCLU
│   │   ├── analyser.py
│   │   ├── loader.py               GDAL loader + LCLU clip/filter
│   │   ├── background_loader.py    LCLU background painting
│   │   └── converter.py            YOLO ↔ COCO format
│   │
│   ├── inference/                  Inference pipeline
│   │   ├── predictor.py            Multi-GPU Detectron2 (ThreadPool)
│   │   ├── postprocessor.py        Tile merging + background apply
│   │   ├── polygonizer.py          Raster → vector conversion
│   │   ├── filtering.py            Area/geometry filtering
│   │   ├── planner.py              Region/tile planning
│   │   └── id_mapper.py            Union-Find for cross-tile IDs
│   │
│   ├── training/                   Training utilities
│   ├── export/                     GeoJSON/Shapefile export
│   └── monitoring/                 Production monitoring
│       ├── metrics.py              Prometheus metrics
│       ├── logger.py               Structured JSON logging
│       └── health.py               GPU/disk/model health checks
│
├── pipeline/                    ← Legacy pipeline (kept for compat)
├── notebooks/                   ← Exploratory notebooks
└── tests/                       ← Unit tests
```

## Quick Start

### Training
```bash
# Prepare data in data/training/{images,annotations}/
python scripts/train.py -c configs/train.yaml
```

### Inference (single folder)
```bash
python scripts/infer.py -c configs/inference.yaml \
  -i data/delineation/images/Region_A \
  -o data/delineation/delineated/Region_A.gpkg
```

### Inference with LCLU
```bash
# Auto-discovery: place data/delineation/masks/Region_A.tif
python scripts/infer.py -c configs/inference.yaml \
  -i data/delineation/images/Region_A \
  -o data/delineation/delineated/Region_A.gpkg

# Or explicit:
python scripts/infer.py -c configs/inference.yaml \
  -i data/delineation/images/Region_A \
  -o data/delineation/delineated/Region_A.gpkg \
  --lclu /path/to/lclu.tif
```

### Batch (all folders)
```bash
python scripts/infer.py -b configs/batch.yaml
```

## Hardware Optimization

Tuned for **8× NVIDIA H100 80GB + 224 CPU cores + 2TB RAM**:
- Multi-GPU: ThreadPoolExecutor for truly parallel inference
- FP16 autocast (~2× throughput)
- 16K×16K regions, batch 64
- GDAL 4GB cache + async prefetch
- Per-stage timing breakdown in logs

## LCLU Land Cover Classes (9-class)

| ID | Class              | Pipeline Role       |
|----|--------------------|---------------------|
| 0  | Water              | Clip boundary       |
| 1  | Trees              | Filter + Background |
| 2  | Grass              | Background fill     |
| 3  | Flooded vegetation | Background fill     |
| 4  | **Crops**          | **Target fields**   |
| 5  | Shrub & Scrub      | Background fill     |
| 6  | Built Area         | Clip + Filter       |
| 7  | Bare ground        | Background fill     |
| 8  | Snow & Ice         | Clip + Filter       |

## Monitoring

- **Prometheus**: tiles processed, regions completed, GPU memory
- **Structured logging**: JSON format with run IDs
- **Health checks**: GPU availability, disk space, model files
