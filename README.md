# Field Delineation Pipeline

This folder contains one orchestrated pipeline for the workflow you were running manually:

1. find Sentinel-2 MGRS tiles intersecting an AOI
2. build S2 mosaics for a date range with `S2Mosaic`
3. clip mosaics to the AOI by default
4. download Dynamic World LCLU masks with Earth Engine
5. run `opensr-model` super-resolution
6. stage `sr.tif` and masks into `Delineate-Anything/data`
7. run `Delineate-Anything` batch inference

The main entrypoint is [pipeline.py](/Users/houcine/Desktop/from_oci/field_delineation_pipeline/pipeline.py).

## Quick Start

First validate that your AOI intersects the expected tile:

```bash
cd /Users/houcine/Desktop/from_oci
./.venv/bin/python field_delineation_pipeline/pipeline.py \
  --aoi /path/to/small_aoi.geojson \
  --start-date 2025-07-01 \
  --end-date 2025-12-31 \
  --run-name small_aoi_tiles \
  --max-tiles 1 \
  --tiles-only
```

Then run the full single-tile test:

```bash
./.venv/bin/python field_delineation_pipeline/pipeline.py \
  --aoi /path/to/small_aoi.geojson \
  --start-date 2025-07-01 \
  --end-date 2025-12-31 \
  --run-name small_aoi_full \
  --max-tiles 1 \
  --ee-project agriculture-486211 \
  --gpus 0,1,2,3,4,5,6,7 \
  --overwrite
```

If Earth Engine authentication is not already configured on the machine, add `--ee-authenticate` once.

## Outputs

Each run writes a self-contained run folder:

```text
field_delineation_pipeline/runs/<run_name>/
  00_aoi/aoi.geojson
  01_mosaics/<tile_id>/*.tif
  02_clipped_mosaics/<tile_id>.tif
  03_lclu_masks/<tile_id>.tif
  04_super_resolution/<tile_id>/sr.tif
  05_delineate_configs/conf_pipeline.yaml
  05_delineate_configs/batch_pipeline.yaml
  06_delineated/<tile_id>.gpkg
  logs/pipeline.log
  manifests/intersecting_tiles.geojson
  manifests/intersecting_tiles.csv
  manifests/staged_inputs.json
  manifests/run_summary.json
```

The Delineate-Anything staging paths are also created exactly as its batch config expects:

```text
Delineate-Anything/data/images/<tile_id>/sr.tif
Delineate-Anything/data/masks/<tile_id>.tif
```

Use `--stage-mode symlink` for large country runs if you want to avoid duplicating `sr.tif` into `Delineate-Anything/data`.

## Main Functions

`load_aoi`: reads GeoJSON, Shapefile, GPKG, or WKT AOIs and normalizes CRS.

`discover_intersecting_tiles`: uses the Sentinel-2 grid and writes tile manifests. The default grid is `S2Mosaic/s2mosaic/sentinel_2_index.gpkg`.

`run_mosaic`: calls the local `S2Mosaic.mosaic` function with date range, tile ID, bands, cloud settings, and GPU-friendly OmniCloudMask settings.

`clip_raster_to_aoi`: saves an AOI-clipped mosaic per tile before OpenSR.

`download_dynamic_world_mask`: downloads Dynamic World `label` mode for the AOI/tile intersection.

`OpenSRRunner.run`: loads OpenSR once, uses CUDA when available, and writes `04_super_resolution/<tile_id>/sr.tif`.

`stage_tile_for_delineation`: stages `sr.tif` and `<tile_id>.tif` masks into `Delineate-Anything/data`.

`write_delineate_configs` and `run_delineate`: generate the batch/config YAML files and execute `delineate.py`.

## GPU Use

`S2Mosaic` uses OmniCloudMask, which will use GPU acceleration when the environment supports it. The pipeline defaults to `--ocm-batch-size 32` and `--ocm-inference-dtype bf16`, which is appropriate for H100-class GPUs.

`opensr-model` uses CUDA if PyTorch sees it. The default `--gpus 0,1,2,3,4,5,6,7` passes all eight GPUs to `opensr-utils`.

`Delineate-Anything` chooses CUDA automatically when `torch.cuda.is_available()` is true. Its batch size defaults to `-1`, so the Delineate-Anything code estimates a batch size from free GPU memory.

## LCLU Defaults

The pipeline uses Dynamic World labels by default:

```text
0 water, 1 trees, 2 grass, 3 flooded vegetation, 4 crops,
5 shrub/scrub, 6 built, 7 bare, 8 snow/ice
```

Default delineation mask settings are:

```text
--mask-range 9
--mask-filter-classes 0,1,2,3,5,6,7,8
--mask-clip-classes 0,6,7,8
```

That means crop class `4` is the only class not filtered. Adjust these if you switch from Dynamic World to another LCLU source.

## Helper Scripts

The old prototype scripts are now reusable CLI helpers:

```bash
./.venv/bin/python field_delineation_pipeline/tiles_downloader/script.py \
  --aoi /path/to/aoi.geojson \
  --output-dir /tmp/tiles

./.venv/bin/python field_delineation_pipeline/lclu_downloader/script.py \
  --aoi /path/to/aoi.geojson \
  --output /tmp/mask.tif \
  --start-date 2025-07-01 \
  --end-date 2025-12-31 \
  --ee-project agriculture-486211
```

## Full-Country Feasibility

It can work for a full country, but it should be run as a tile-sharded production job, not as one giant monolithic process.

Main blockers:

- S2 tile count: country AOIs can intersect tens to hundreds of MGRS tiles.
- S2Mosaic memory and IO: the mosaic code works at full Sentinel-2 tile dimensions before the AOI clip, so each tile can be large even if only part intersects the AOI.
- Planetary Computer/STAC throughput: full-country runs can hit download speed, transient HTTP failures, and rate limits.
- Earth Engine export limits: Dynamic World masks should stay per tile or smaller AOI chunk.
- OpenSR expansion: 4x super-resolution increases pixel count by 16x, which is usually the largest disk/time multiplier.
- Delineate-Anything postprocessing: polygonization, simplification, and GeoPackage writes can dominate on large dense agricultural areas.
- Model/checkpoint downloads: Hugging Face and OpenSR checkpoints should be pre-cached before long runs.
- Multi-GPU utilization: OpenSR can use multiple GPUs, but Delineate-Anything is best scaled by launching independent tile jobs across GPUs.
- AOI complexity: very detailed country boundary files should be dissolved/simplified for tile discovery and clipping.

Recommended scaling path:

1. Run `--tiles-only` and inspect `manifests/intersecting_tiles.csv`.
2. Run one known small AOI with `--max-tiles 1`.
3. Run several tiles with `--include-tiles TILE1,TILE2,...`.
4. For country runs, submit independent tile batches and keep `--resume` enabled.
5. Use `--stage-mode symlink` and compress/archive intermediate rasters after validation.
