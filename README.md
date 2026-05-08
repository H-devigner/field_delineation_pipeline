# Field Delineation Pipeline

This folder contains one orchestrated pipeline for the workflow you were running manually:

1. find Sentinel-2 MGRS tiles intersecting an AOI, or ingest XYZ basemap image tiles
2. build S2 mosaics for a date range with `S2Mosaic`, or convert basemap PNG/JPEG tiles to GeoTIFF
3. clip mosaics to the AOI by default
4. download Dynamic World LCLU masks with Earth Engine
5. run `opensr-model` super-resolution
6. stage `sr.tif` and masks into `Delineate-Anything/data`
7. run `Delineate-Anything` batch inference

The main entrypoint is [pipeline.py](/Users/houcine/Desktop/from_oci/field_delineation_pipeline/pipeline.py).

## Component Repos After Cloning

`S2Mosaic`, `opensr-model`, and `Delineate-Anything` are separate Git repositories. A parent Git repository will not automatically copy their contents unless you either clone submodules or let the pipeline clone them.

If they were added as Git submodules, clone with:

```bash
git clone --recurse-submodules <YOUR_PIPELINE_REPO_URL>
```

For an already-cloned repo:

```bash
git submodule update --init --recursive
```

If you did not configure submodules, let the pipeline fill missing or empty component folders. By default it clones `S2Mosaic` from `https://github.com/H-devigner/S2Mosaic.git` on branch `field-delineation-pipeline`, and the other two component repos from their upstream defaults:

```bash
./.venv/bin/python field_delineation_pipeline/pipeline.py \
  --aoi /path/to/small_aoi.geojson \
  --start-date 2025-07-01 \
  --end-date 2025-12-31 \
  --run-name repo_setup_check \
  --tiles-only \
  --clone-missing
```

Equivalent manual clone commands:

```bash
cd field_delineation_pipeline
git clone --branch field-delineation-pipeline --single-branch https://github.com/H-devigner/S2Mosaic.git
git clone https://github.com/ESAOpenSR/opensr-model.git
git clone https://github.com/Lavreniuk/Delineate-Anything.git
```

## Conda Environment

Run the pipeline from one shared conda environment. The orchestrator imports S2Mosaic, Earth Engine/geemap, OpenSR, and Delineate-Anything code, so separate per-repo environments will fail when a later step imports a package that is missing from the active environment.

On corporate networks, prefer the two-step setup. Conda installs only GDAL/geospatial packages, then pip installs the Python/ML stack through the proxy.

Set proxy variables:

```bash
export HTTP_PROXY=http://10.68.69.53:80/
export HTTPS_PROXY=http://10.68.69.53:80/
export PIP_PROXY=http://10.68.69.53:80/
export PIP_INDEX_URL=https://pypi.org/simple
export PIP_TRUSTED_HOST="pypi.org files.pythonhosted.org"
export CONDA_NO_PLUGINS=true
```

Create the conda base:

```bash
conda env create -f environment-conda.yml -n field-delineation --solver=libmamba
conda activate field-delineation
```

Install the pip dependencies through the proxy:

```bash
python -m pip install -r requirements-pipeline.txt \
  --index-url https://pypi.org/simple \
  --trusted-host pypi.org \
  --trusted-host files.pythonhosted.org \
  --proxy http://10.68.69.53:80/
```

If `--solver=libmamba` is unavailable, omit it:

```bash
conda env create -f environment-conda.yml -n field-delineation
```

The older all-in-one environment file is still available:

```bash
conda env create -f field_delineation_pipeline/environment.yml
conda activate field-delineation
```

Verify CUDA and core packages:

```bash
python - <<'PY'
from osgeo import gdal
import geemap
import ee
import torch
import opensr_utils
import omegaconf
print("GDAL:", gdal.VersionInfo())
print("geemap:", geemap.__version__)
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available(), torch.cuda.device_count())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
print("opensr_utils ok")
print("omegaconf ok")
PY
```

If you already have a conda env with GDAL, install the missing LCLU packages directly:

```bash
conda activate geo_clean
conda install -c conda-forge geemap earthengine-api -y
```

If conda cannot solve quickly, use pip inside the active conda env:

```bash
python -m pip install geemap earthengine-api
```

Authenticate Earth Engine once if needed:

```bash
python - <<'PY'
import ee
ee.Authenticate(auth_mode="notebook")
ee.Initialize(project="agriculture-486211")
print("earth engine ok")
PY
```

On a workstation with a browser, `--ee-auth-mode localhost` is also fine. On a headless server without `gcloud`, use `notebook`.

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

## Basemap / XYZ Tile Mode

Use this mode when you already have high-resolution basemap tiles, or when you have permission to download them from an XYZ tile endpoint. The pipeline mosaics the image tiles into an EPSG:3857 GeoTIFF, clips it to the AOI, stages it as `sr.tif`, and then continues into LCLU masks, Delineate-Anything, and exports.

Local tile folder layouts supported:

```text
tiles/<z>/<x>/<y>.png
tiles/<x>/<y>.png   # pass --xyz-zoom
```

Convert local tiles without the full pipeline:

```bash
python basemap_tiles.py convert \
  --tiles-root /path/to/tiles \
  --zoom 18 \
  --output runs/basemap_debug/basemap_3857.tif \
  --overwrite
```

Run the full pipeline from local XYZ tiles:

```bash
python pipeline.py \
  --input-mode xyz_tiles \
  --aoi /path/to/aoi.geojson \
  --start-date 2025-07-01 \
  --end-date 2025-12-31 \
  --run-name basemap_aoi_full \
  --xyz-tiles-root /path/to/tiles \
  --xyz-zoom 18 \
  --xyz-name basemap_aoi \
  --ee-project agriculture-486211 \
  --gpus 0,1,2,3,4,5,6,7 \
  --resume
```

Download OpenAerialMap tiles for an AOI and convert them. Coverage depends on whether OpenAerialMap has open imagery for the area. If the network blocks `apps.kontur.io`, use `eox_s2cloudless_2024` as a global Sentinel-2 cloudless fallback. EOX is not high-resolution aerial imagery, but it is useful for testing the basemap pipeline through a different host.

Create a very small centered AOI first for smoke tests:

```bash
python basemap_tiles.py tiny-aoi \
  --aoi /path/to/aoi.geojson \
  --size-meters 250 \
  --output runs/basemap_download/tiny_aoi_250m.geojson
```

```bash
python basemap_tiles.py download \
  --provider openaerialmap \
  --aoi runs/basemap_download/tiny_aoi_250m.geojson \
  --zoom 18 \
  --output-root runs/basemap_download/tiles \
  --max-tiles 2000 \
  --sleep-seconds 0.05 \
  --no-verify-ssl \
  --no-proxy \
  --convert-output runs/basemap_download/basemap_3857.tif
```

If a corporate proxy intercepts HTTPS and the machine does not have the corporate CA installed, prefer passing a PEM file with `--ca-bundle /path/to/corp-ca.pem`. For a quick trusted-network smoke test only, use `--no-verify-ssl`. If the proxy returns `502 cannotconnect`, try `--no-proxy`; if direct internet is blocked, use `--proxy http://host:port/` and ask IT to allowlist the tile host.

Probe one tile before starting a larger download:

```bash
python basemap_tiles.py probe \
  --provider openaerialmap \
  --aoi runs/basemap_download/tiny_aoi_250m.geojson \
  --zoom 18 \
  --no-verify-ssl \
  --no-proxy
```

Probe the EOX fallback host:

```bash
python basemap_tiles.py probe \
  --provider eox_s2cloudless_2024 \
  --aoi runs/basemap_download/tiny_aoi_250m.geojson \
  --zoom 16 \
  --no-verify-ssl \
  --proxy http://10.68.69.53:80/
```

Run the full orchestrator by letting it download OpenAerialMap tiles first:

```bash
python pipeline.py \
  --input-mode xyz_tiles \
  --aoi /path/to/aoi.geojson \
  --start-date 2025-07-01 \
  --end-date 2025-12-31 \
  --run-name oam_basemap_aoi_full \
  --xyz-provider openaerialmap \
  --xyz-zoom 18 \
  --xyz-name oam_basemap \
  --xyz-max-download-tiles 2000 \
  --xyz-sleep-seconds 0.05 \
  --xyz-no-verify-ssl \
  --xyz-no-proxy \
  --ee-project agriculture-486211 \
  --resume
```

For a custom licensed endpoint, pass a URL template directly. Check the provider's usage terms first; many basemap endpoints restrict bulk downloading.

```bash
python basemap_tiles.py download \
  --url-template 'https://tiles.example.com/{z}/{x}/{y}.png' \
  --aoi /path/to/aoi.geojson \
  --zoom 18 \
  --output-root runs/basemap_download/tiles \
  --max-tiles 2000 \
  --sleep-seconds 0.05 \
  --convert-output runs/basemap_download/basemap_3857.tif
```

The same custom download can be embedded in the orchestrator by passing `--xyz-url-template`. Basemap inputs skip OpenSR by default because they are already high resolution; pass `--basemap-run-super-resolution` only for experiments.

## Outputs

Each run writes a self-contained run folder:

```text
field_delineation_pipeline/runs/<run_name>/
  00_aoi/aoi.geojson
  00_basemap_tiles/                 # only when --xyz-url-template downloads tiles
  01_mosaics/<tile_id>/*.tif
  01_basemap_geotiff/<xyz_name>.tif # only for --input-mode xyz_tiles
  02_clipped_mosaics/<tile_id>.tif
  03_lclu_masks/<tile_id>.tif
  04_super_resolution/<tile_id>/sr.tif
  05_delineate_configs/conf_pipeline.yaml
  05_delineate_configs/batch_pipeline.yaml
  06_delineated/<tile_id>.gpkg
  06_instance_rasters/<tile_id>/<tile_id>.instances.tif # only with --save-instance-rasters
  07_exports/
    exports_summary.csv
    exports_summary.json
    index.html
    <tile_id>/<result_name>/*.geojson
    <tile_id>/<result_name>/*.kml
    <tile_id>/<result_name>/*.boundaries.png
    <tile_id>/<result_name>/*.sr_overlay.png
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

The pipeline defaults to `--delineate-bands 1,2,3` for Delineate-Anything because the staged `sr.tif` files are already in RGB order. Override this only if your staged imagery has a different band layout.

Use `--save-instance-rasters` when you want to preserve Delineate-Anything's postprocessed instance-ID raster before polygonization. Positive values are field instance IDs, negative values are background IDs, and `0` is nodata/background. These rasters are intended for later cross-tile seam merging.

Standalone instance-raster postprocessing:

```bash
python postprocess_instance_rasters.py \
  --instance-root "$RUNS/$RUN/06_instance_rasters" \
  --output-dir "$RUNS/$RUN/08_instance_postprocess" \
  --output-name "$RUN" \
  --merge-touching
```

This writes `<run>.merged_fields.gpkg`, `<run>.merged_fields.geojson`, `<run>.merged_fields.png`, and `<run>.merge_summary.json`. By default overlapping IDs from different rasters are merged. `--merge-touching` also merges IDs that touch or nearly touch across artificial tile seams; inspect the quicklook and summary after the first run.

For interactive exploration of huge merged outputs, build vector tiles and serve a browser map:

```bash
python vector_tile_viewer.py build \
  --input "$RUNS/$RUN/08_instance_postprocess/$RUN.merged_fields.gpkg" \
  --output-dir "$RUNS/$RUN/09_field_viewer" \
  --name "$RUN" \
  --maxzoom 16 \
  --work-dir "$RUNS/$RUN/09_field_viewer/tmp"

python vector_tile_viewer.py serve \
  --viewer-dir "$RUNS/$RUN/09_field_viewer" \
  --host 0.0.0.0 \
  --port 8088
```

The build step requires `tippecanoe` and GDAL command-line tools. In conda, install them with `conda install -c conda-forge tippecanoe gdal`.

The generated viewer includes a live `Min Area` filter when the `area` attribute is present in the vector tiles. The default build keeps `area`, so you can test candidate thresholds in the browser without generating multiple filtered GPKG/GeoJSON files. The current threshold is also reflected in the URL as `?min_area=...` so you can bookmark or share it.

## Main Functions

`load_aoi`: reads GeoJSON, Shapefile, GPKG, or WKT AOIs and normalizes CRS.

`discover_intersecting_tiles`: uses the Sentinel-2 grid and writes tile manifests. The default grid is `S2Mosaic/s2mosaic/sentinel_2_index.gpkg`.

`basemap_tiles`: downloads XYZ tiles for an AOI and mosaics local PNG/JPEG/WebP/TIFF tiles into EPSG:3857 GeoTIFFs.

`run_mosaic`: calls the local `S2Mosaic.mosaic` function with date range, tile ID, bands, cloud settings, and GPU-friendly OmniCloudMask settings.

`clip_raster_to_aoi`: saves an AOI-clipped mosaic per tile before OpenSR.

`download_dynamic_world_mask`: downloads Dynamic World `label` mode for the AOI/tile intersection.

`OpenSRRunner.run`: loads OpenSR once, uses CUDA when available, and writes `04_super_resolution/<tile_id>/sr.tif`.

`stage_tile_for_delineation`: stages `sr.tif` and `<tile_id>.tif` masks into `Delineate-Anything/data`.

`write_delineate_configs` and `run_delineate`: generate the batch/config YAML files and execute `delineate.py`.

`--save-instance-rasters`: writes `06_instance_rasters/<tile_id>/*.instances.tif` from Delineate-Anything immediately before polygonization.

`postprocess_instance_rasters`: polygonizes positive instance IDs, reconciles cross-raster seam IDs, dissolves merged fields, and writes global GPKG/GeoJSON/PNG outputs.

`vector_tile_viewer`: packages large GPKG/GeoJSON outputs into MBTiles vector tiles and serves a MapLibre basemap viewer.

`export_results`: converts final GeoPackages into WGS84 GeoJSON/KML, boundary quicklook PNGs, optional SR-overlay PNGs, summary CSV/JSON, and an `index.html` gallery.

Standalone export command:

```bash
python export_results.py runs/small_aoi_test/06_delineated \
  --outdir runs/small_aoi_test/07_exports_manual \
  --formats geojson,kml,png
```

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

For unstable corporate proxy connections to Earth Engine signed download URLs, keep the direct backend but add retries and explicit proxy settings:

```bash
--lclu-backend direct \
--lclu-retries 8 \
--lclu-retry-sleep-seconds 20 \
--lclu-request-timeout 600 \
--lclu-proxy http://10.68.69.53:80/
```

If the proxy itself is the failure point and direct internet is allowed, replace `--lclu-proxy ...` with `--lclu-no-proxy`. If TLS is intercepted, prefer `--lclu-ca-bundle /path/to/corp-ca.pem`; use `--lclu-no-verify-ssl` only for trusted-network smoke tests.

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
