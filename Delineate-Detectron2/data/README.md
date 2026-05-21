# Data Directory — Field Delineation Pipeline

## Structure

```
data/
├── training/                    ← Training data (Detectron2 / YOLO)
│   ├── images/                  ← Training image tiles
│   │   ├── train/
│   │   └── val/
│   ├── labels/                  ← YOLO-format labels (.txt)
│   │   ├── train/
│   │   └── val/
│   └── annotations/             ← COCO-format JSON annotations
│       ├── train.json
│       └── val.json
│
└── delineation/                 ← Inference / production data
    ├── images/                  ← Input GeoTIFFs (one subfolder per area)
    │   ├── Region_A/
    │   │   ├── part_1.tif
    │   │   └── part_2.tif
    │   └── Region_B/
    │       └── image.tif
    │
    ├── masks/                   ← LCLU masks (auto-matched by folder name)
    │   ├── Region_A.tif         ← matches images/Region_A/
    │   └── Region_B.tif         ← matches images/Region_B/
    │
    ├── delineated/              ← Output GeoPackages
    │   ├── Region_A.gpkg
    │   └── Region_B.gpkg
    │
    └── temp/                    ← Warped LCLU cache (auto-generated)
```

## LCLU Auto-Discovery

The pipeline finds masks by matching input folder names under `masks/`:

1. `masks/<name>.tif`
2. `masks/<name>.tiff`
3. `masks/<name>/<name>.tif`
4. `masks/<name>/lclu.tif`

## Land Cover Classes (9-class)

| ID | Class              | Role in Delineation     |
|----|--------------------|-------------------------|
| 0  | Water              | Clip boundary           |
| 1  | Trees              | Filter + Background     |
| 2  | Grass              | Background fill         |
| 3  | Flooded vegetation | Background fill         |
| 4  | **Crops**          | **Target fields**       |
| 5  | Shrub & Scrub      | Background fill         |
| 6  | Built Area         | Clip + Filter           |
| 7  | Bare ground        | Background fill         |
| 8  | Snow & Ice         | Clip + Filter           |
