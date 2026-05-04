#!/usr/bin/env python3
"""
End-to-end field delineation pipeline orchestrator.

The pipeline is intentionally modular:
1. find Sentinel-2 MGRS tiles intersecting an AOI, or ingest XYZ basemap image tiles
2. mosaic each Sentinel-2 tile with S2Mosaic, or convert basemap tiles to GeoTIFF
3. optionally clip mosaics to the AOI
4. download Dynamic World LCLU masks
5. run OpenSR super-resolution or stage the mosaic directly
6. stage inputs for Delineate-Anything and run its batch CLI
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as dt
import importlib.util
import json
import logging
import math
import os
import sqlite3
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path
from typing import Any, Iterable


PIPELINE_ROOT = Path(__file__).resolve().parent
S2MOSAIC_ROOT = PIPELINE_ROOT / "S2Mosaic"
OPENSR_ROOT = PIPELINE_ROOT / "opensr-model"
DELINEATE_ROOT = PIPELINE_ROOT / "Delineate-Anything"
DEFAULT_TILE_GRID = S2MOSAIC_ROOT / "s2mosaic" / "sentinel_2_index.gpkg"

REPO_SPECS = {
    "S2Mosaic": {
        "url": "https://github.com/H-devigner/S2Mosaic.git",
        "branch": "field-delineation-pipeline",
    },
    "opensr-model": {
        "url": "https://github.com/ESAOpenSR/opensr-model.git",
        "branch": None,
    },
    "Delineate-Anything": {
        "url": "https://github.com/Lavreniuk/Delineate-Anything.git",
        "branch": None,
    },
}

REPO_MARKERS = {
    "S2Mosaic": "s2mosaic/__init__.py",
    "opensr-model": "opensr_model/__init__.py",
    "Delineate-Anything": "delineate.py",
}

LOGGER = logging.getLogger("field_delineation_pipeline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Sentinel-2 or basemap-tile ingestion, LCLU mask, OpenSR/staging, and Delineate-Anything as one pipeline."
    )
    parser.add_argument(
        "--input-mode",
        default="sentinel2",
        choices=["sentinel2", "xyz_tiles"],
        help="Use Sentinel-2/S2Mosaic inputs or XYZ basemap image tiles converted to GeoTIFF.",
    )
    parser.add_argument("--aoi", required=True, type=Path, help="AOI file: GeoJSON/Shapefile/GPKG or WKT text.")
    parser.add_argument("--start-date", required=True, help="Inclusive start date, YYYY-MM-DD.")
    parser.add_argument("--end-date", required=True, help="Exclusive end date, YYYY-MM-DD.")
    parser.add_argument("--run-name", default=None, help="Run folder name. Defaults to timestamp plus AOI stem.")
    parser.add_argument("--output-root", default=PIPELINE_ROOT / "runs", type=Path, help="Root folder for run outputs.")
    parser.add_argument("--tile-grid", default=DEFAULT_TILE_GRID, type=Path, help="Sentinel-2 tiling grid file.")
    parser.add_argument("--tile-id-column", default=None, help="Tile ID column in the grid. Auto-detected if omitted.")
    parser.add_argument("--include-tiles", default=None, help="Comma-separated tile IDs to keep after AOI intersection.")
    parser.add_argument("--max-tiles", default=None, type=int, help="Limit number of intersecting tiles for smoke tests.")
    parser.add_argument("--clone-missing", action="store_true", help="Clone missing component repos into this folder.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing intermediate outputs.")
    parser.add_argument("--resume", action="store_true", help="Reuse existing intermediates when possible.")
    parser.add_argument("--tiles-only", action="store_true", help="Stop after AOI/tile intersection manifests are written.")

    basemap = parser.add_argument_group("XYZ basemap tiles")
    basemap.add_argument("--xyz-name", default="basemap", help="Pseudo tile ID used for basemap pipeline outputs.")
    basemap.add_argument("--xyz-tiles-root", default=None, type=Path, help="Local XYZ tile root. Supports z/x/y.png or x/y.png with --xyz-zoom.")
    basemap.add_argument("--xyz-provider", default=None, choices=["openaerialmap"], help="Built-in free/open XYZ imagery provider preset.")
    basemap.add_argument("--xyz-url-template", default=None, help="Optional XYZ URL template, e.g. https://server/{z}/{x}/{y}.png")
    basemap.add_argument("--xyz-zoom", default=None, type=int, help="XYZ zoom to download or select from the local tile root.")
    basemap.add_argument("--xyz-extension", default="png", help="Downloaded tile extension when --xyz-url-template is used.")
    basemap.add_argument("--xyz-timeout", default=60, type=int, help="HTTP timeout per downloaded basemap tile.")
    basemap.add_argument("--xyz-sleep-seconds", default=0.0, type=float, help="Delay between downloaded basemap tile requests.")
    basemap.add_argument("--xyz-user-agent", default="field-delineation-pipeline/1.0", help="User-Agent for basemap tile requests.")
    basemap.add_argument("--xyz-ca-bundle", default=None, type=Path, help="Optional PEM CA bundle for corporate TLS interception.")
    basemap.add_argument("--xyz-no-verify-ssl", action="store_true", help="Disable SSL verification for tile downloads. Use only on trusted networks.")
    basemap.add_argument("--xyz-proxy", default=None, help="Explicit HTTP/HTTPS proxy URL for basemap tile downloads.")
    basemap.add_argument("--xyz-no-proxy", action="store_true", help="Ignore HTTP_PROXY/HTTPS_PROXY environment variables for basemap tile downloads.")
    basemap.add_argument("--xyz-retries", default=2, type=int)
    basemap.add_argument("--xyz-retry-sleep-seconds", default=2.0, type=float)
    basemap.add_argument("--xyz-skip-failed", action="store_true", help="Continue when individual basemap tiles fail; missing pixels remain blank.")
    basemap.add_argument(
        "--xyz-max-download-tiles",
        default=5000,
        type=int,
        help="Safety cap for basemap tile downloads. Increase only after checking provider terms and storage.",
    )
    basemap.add_argument(
        "--basemap-run-super-resolution",
        action="store_true",
        help="Experimental: run OpenSR on basemap GeoTIFFs. By default basemap inputs skip OpenSR because they are already high resolution.",
    )

    mosaic = parser.add_argument_group("S2Mosaic")
    mosaic.add_argument("--mosaic-method", default="max_ndvi", choices=["mean", "first", "median", "percentile", "max_ndvi"])
    mosaic.add_argument("--sort-method", default="valid_data", choices=["valid_data", "oldest", "newest"])
    mosaic.add_argument("--bands", default="B04,B03,B02,B08", help="Comma-separated S2 bands for mosaics.")
    mosaic.add_argument("--no-data-threshold", default=0.001, type=float)
    mosaic.add_argument("--max-cloud-cover", default=100.0, type=float)
    mosaic.add_argument("--percentile-value", default=None, type=float)
    mosaic.add_argument("--ocm-batch-size", default=32, type=int)
    mosaic.add_argument("--ocm-inference-dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    mosaic.add_argument("--debug-cache", action="store_true")
    mosaic.add_argument("--no-clip-to-aoi", dest="clip_to_aoi", action="store_false")
    mosaic.set_defaults(clip_to_aoi=True)

    lclu = parser.add_argument_group("LCLU masks")
    lclu.add_argument("--skip-lclu", action="store_true", help="Run delineation without LCLU masks.")
    lclu.add_argument("--ee-project", default="agriculture-486211", help="Google Earth Engine project.")
    lclu.add_argument("--ee-authenticate", action="store_true", help="Run ee.Authenticate() before ee.Initialize().")
    lclu.add_argument(
        "--ee-auth-mode",
        default=None,
        choices=["notebook", "localhost", "gcloud", "appdefault", "colab"],
        help="Earth Engine auth mode used with --ee-authenticate. Use 'notebook' on headless/remote servers without gcloud.",
    )
    lclu.add_argument("--lclu-scale", default=10, type=int)
    lclu.add_argument("--lclu-crs", default="EPSG:4326")
    lclu.add_argument("--lclu-collection", default="GOOGLE/DYNAMICWORLD/V1")
    lclu.add_argument("--lclu-backend", default="direct", choices=["direct", "geemap"], help="LCLU download backend.")
    lclu.add_argument("--lclu-direct-tile-degrees", default=0.15, type=float, help="Chunk size in degrees for direct EE LCLU downloads.")
    lclu.add_argument("--lclu-request-timeout", default=300, type=int, help="HTTP timeout in seconds for direct EE LCLU downloads.")
    lclu.add_argument("--lclu-num-threads", default=1, type=int, help="Threads for geemap/geedim LCLU download.")
    lclu.add_argument("--lclu-max-tile-size", default=16, type=int, help="Max geedim tile size in MB.")
    lclu.add_argument("--lclu-max-tile-dim", default=1024, type=int, help="Max geedim tile width/height in pixels.")

    sr = parser.add_argument_group("OpenSR")
    sr.add_argument("--skip-super-resolution", action="store_true", help="Stage the mosaic as sr.tif without OpenSR.")
    sr.add_argument("--opensr-config", default=OPENSR_ROOT / "opensr_model" / "configs" / "config_10m.yaml", type=Path)
    sr.add_argument("--opensr-window-size", default="128,128")
    sr.add_argument("--opensr-factor", default=4, type=int)
    sr.add_argument("--opensr-overlap", default=12, type=int)
    sr.add_argument("--opensr-eliminate-border-px", default=2, type=int)
    sr.add_argument("--opensr-batch-size", default=2, type=int)
    sr.add_argument("--gpus", default="0,1,2,3,4,5,6,7", help="GPU IDs for OpenSR and CUDA_VISIBLE_DEVICES.")

    da = parser.add_argument_group("Delineate-Anything")
    da.add_argument("--skip-delineation", action="store_true")
    da.add_argument("--delineate-models", default="large", help="Comma-separated Delineate-Anything models.")
    da.add_argument("--delineate-bands", default="3,2,1", help="GDAL band indexes passed to Delineate-Anything.")
    da.add_argument("--delineate-batch-size", default=-1, type=int, help="-1 lets Delineate-Anything auto-select.")
    da.add_argument("--mask-range", default=9, type=int, help="Mask class range. Dynamic World label is 0..8.")
    da.add_argument("--mask-filter-classes", default="0,1,2,3,5,6,7,8")
    da.add_argument("--mask-clip-classes", default="0,6,7,8")
    da.add_argument("--delineate-output-root", default=None, type=Path)
    da.add_argument("--keep-delineate-temp", action="store_true")
    da.add_argument("--stage-mode", default="copy", choices=["copy", "symlink"], help="How to stage SR/masks into Delineate-Anything/data.")
    da.add_argument("--python-executable", default=sys.executable, help="Python executable for Delineate-Anything CLI.")
    da.add_argument("--verbose-delineate", action="store_true")

    exports = parser.add_argument_group("Exports")
    exports.add_argument("--skip-exports", action="store_true", help="Skip GeoJSON/KML/PNG export step.")
    exports.add_argument("--export-layer", default=None, help="Optional GPKG layer name. Defaults to first layer.")
    exports.add_argument("--export-formats", default="geojson,kml,png", help="Comma-separated export formats.")
    exports.add_argument("--export-assumed-epsg", default=None, type=int, help="Fallback EPSG if a GPKG has no CRS.")
    exports.add_argument("--quicklook-dpi", default=220, type=int)
    exports.add_argument("--quicklook-max-pixels", default=1800, type=int)
    return parser.parse_args()


def setup_logging(run_dir: Path) -> Path:
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "pipeline.log"

    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    LOGGER.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)
    return log_path


def parse_date(value: str) -> dt.date:
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid date '{value}', expected YYYY-MM-DD.") from exc


def parse_csv(value: str | None) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_csv(value: str | None) -> list[int]:
    return [int(item) for item in parse_csv(value)]


def parse_window_size(value: str) -> tuple[int, int]:
    parts = parse_int_csv(value)
    if len(parts) != 2 or min(parts) <= 0:
        raise ValueError("--opensr-window-size must have two positive integers, e.g. 128,128.")
    return parts[0], parts[1]


def parse_gpus(value: str) -> int | list[int]:
    parts = parse_int_csv(value)
    if not parts:
        return 0
    if len(parts) == 1:
        return parts[0]
    return parts


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def write_summary(run_dir: Path, summary: dict[str, Any]) -> None:
    dump_json(run_dir / "manifests" / "run_summary.json", summary)


@contextlib.contextmanager
def timed_step(summary: dict[str, Any], run_dir: Path, name: str) -> Iterable[None]:
    LOGGER.info("START %s", name)
    start = time.monotonic()
    step: dict[str, Any] = {
        "name": name,
        "status": "running",
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    summary.setdefault("steps", []).append(step)
    write_summary(run_dir, summary)
    try:
        yield
    except Exception as exc:
        elapsed = time.monotonic() - start
        step.update(
            {
                "status": "failed",
                "duration_seconds": round(elapsed, 3),
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        write_summary(run_dir, summary)
        LOGGER.exception("FAILED %s after %.1fs", name, elapsed)
        raise
    else:
        elapsed = time.monotonic() - start
        step.update(
            {
                "status": "completed",
                "duration_seconds": round(elapsed, 3),
                "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            }
        )
        write_summary(run_dir, summary)
        LOGGER.info("DONE %s in %.1fs", name, elapsed)


def ensure_repositories(clone_missing: bool, required_repositories: set[str] | None = None) -> None:
    for repo_name, repo_spec in REPO_SPECS.items():
        if required_repositories is not None and repo_name not in required_repositories:
            continue
        repo_url = repo_spec["url"]
        repo_branch = repo_spec["branch"]
        repo_path = PIPELINE_ROOT / repo_name
        marker_path = repo_path / REPO_MARKERS[repo_name]
        if marker_path.exists():
            if repo_branch:
                ensure_repository_branch(
                    repo_name=repo_name,
                    repo_path=repo_path,
                    repo_url=repo_url,
                    repo_branch=repo_branch,
                    allow_checkout=clone_missing,
                )
            continue

        if not clone_missing:
            raise FileNotFoundError(
                f"Missing or incomplete {repo_name} at {repo_path}. "
                f"Re-run with --clone-missing to clone {repo_url}"
                f"{' branch ' + repo_branch if repo_branch else ''}, or initialize Git submodules."
            )

        if repo_path.exists() and not repo_path.is_dir():
            raise FileExistsError(f"Expected {repo_path} to be a directory, but it is a file.")

        if repo_path.exists() and any(repo_path.iterdir()):
            raise FileExistsError(
                f"{repo_path} exists but does not look like a complete {repo_name} checkout. "
                "It is not empty, so the pipeline will not overwrite it. "
                "Initialize submodules or move/remove that directory manually."
            )

        command = ["git", "clone"]
        if repo_branch:
            command.extend(["--branch", repo_branch, "--single-branch"])
        command.extend([repo_url, str(repo_path)])
        LOGGER.info(
            "Cloning %s from %s%s",
            repo_name,
            repo_url,
            f" branch {repo_branch}" if repo_branch else "",
        )
        subprocess.run(command, check=True)


def ensure_repository_branch(
    *,
    repo_name: str,
    repo_path: Path,
    repo_url: str,
    repo_branch: str,
    allow_checkout: bool,
) -> None:
    branch_result = subprocess.run(
        ["git", "-C", str(repo_path), "branch", "--show-current"],
        capture_output=True,
        text=True,
        check=False,
    )
    current_branch = branch_result.stdout.strip()
    if branch_result.returncode != 0 or current_branch == repo_branch:
        return

    if not allow_checkout:
        LOGGER.warning(
            "%s exists on branch '%s'; expected '%s'. Existing checkout will be used.",
            repo_name,
            current_branch or "<detached>",
            repo_branch,
        )
        return

    status_result = subprocess.run(
        ["git", "-C", str(repo_path), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
    )
    if status_result.stdout.strip():
        raise RuntimeError(
            f"{repo_name} at {repo_path} has uncommitted changes and is on branch "
            f"'{current_branch}', but expected '{repo_branch}'. Commit/stash those changes "
            "or switch the branch manually."
        )

    remotes_result = subprocess.run(
        ["git", "-C", str(repo_path), "remote"],
        capture_output=True,
        text=True,
        check=True,
    )
    remotes = {line.strip() for line in remotes_result.stdout.splitlines() if line.strip()}
    remote_name = "pipeline"
    if remote_name not in remotes:
        subprocess.run(
            ["git", "-C", str(repo_path), "remote", "add", remote_name, repo_url],
            check=True,
        )

    LOGGER.info(
        "%s exists on branch '%s'; switching to %s/%s.",
        repo_name,
        current_branch or "<detached>",
        remote_name,
        repo_branch,
    )
    subprocess.run(
        ["git", "-C", str(repo_path), "fetch", remote_name, repo_branch],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo_path), "switch", "-C", repo_branch, f"{remote_name}/{repo_branch}"],
        check=True,
    )


def required_component_repositories(args: argparse.Namespace) -> set[str]:
    required: set[str] = set()
    if args.input_mode == "sentinel2":
        required.add("S2Mosaic")
    if not args.tiles_only:
        required.add("Delineate-Anything")
    if not args.skip_super_resolution:
        required.add("opensr-model")
    return required


def check_imports(requirements: dict[str, str]) -> list[str]:
    missing = []
    for import_name, package_name in requirements.items():
        if importlib.util.find_spec(import_name) is None:
            missing.append(package_name)
    return sorted(set(missing))


def check_python_environment(args: argparse.Namespace) -> None:
    requirements = {
        "geopandas": "geopandas",
        "numpy": "numpy",
        "rasterio": "rasterio",
        "shapely": "shapely",
        "yaml": "PyYAML",
    }
    if args.input_mode == "xyz_tiles" and (args.xyz_url_template or args.xyz_provider):
        requirements["requests"] = "requests"
    if not args.skip_lclu:
        requirements.update(
            {
                "ee": "earthengine-api",
                "geemap": "geemap",
                "requests": "requests",
            }
        )
    if not args.skip_super_resolution:
        if str(OPENSR_ROOT) not in sys.path:
            sys.path.insert(0, str(OPENSR_ROOT))
        requirements.update(
            {
                "omegaconf": "omegaconf",
                "opensr_model": "opensr-model",
                "opensr_utils": "opensr-utils",
                "torch": "torch",
            }
        )
    if not args.skip_delineation:
        if str(DELINEATE_ROOT) not in sys.path:
            sys.path.insert(0, str(DELINEATE_ROOT))
        requirements.update(
            {
                "cv2": "opencv-python",
                "huggingface_hub": "huggingface-hub",
                "osgeo": "gdal",
                "psutil": "psutil",
                "torch": "torch",
                "tqdm": "tqdm",
                "ultralytics": "ultralytics",
            }
        )
    if not args.skip_exports:
        requirements.update(
            {
                "matplotlib": "matplotlib",
            }
        )

    missing = check_imports(requirements)
    if missing:
        raise ImportError(
            "Missing Python packages in the environment running pipeline.py: "
            + ", ".join(missing)
            + ". Activate the shared conda environment or install these packages before running."
        )


def import_geospatial_stack() -> tuple[Any, Any, Any]:
    try:
        import geopandas as gpd
        import shapely
        from shapely import wkt
    except ImportError as exc:
        raise ImportError("Install geopandas and shapely before running the pipeline.") from exc
    return gpd, shapely, wkt


def union_geometry(gdf: Any) -> Any:
    if hasattr(gdf.geometry, "union_all"):
        return gdf.geometry.union_all()
    return gdf.geometry.unary_union


def force_2d_geometry(geometry: Any) -> Any:
    try:
        import shapely

        if hasattr(shapely, "force_2d"):
            return shapely.force_2d(geometry)
    except Exception:
        pass

    from shapely.ops import transform

    return transform(lambda x, y, *args: (x, y), geometry)


def make_valid_geometry(geometry: Any) -> Any:
    try:
        import shapely

        if hasattr(shapely, "make_valid"):
            return shapely.make_valid(geometry)
    except Exception:
        pass

    return geometry.buffer(0)


def extract_polygonal_geometry(geometry: Any) -> Any:
    from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
    from shapely.ops import unary_union

    if isinstance(geometry, (Polygon, MultiPolygon)):
        return geometry

    if isinstance(geometry, GeometryCollection):
        polygons = []
        for part in geometry.geoms:
            if isinstance(part, Polygon):
                polygons.append(part)
            elif isinstance(part, MultiPolygon):
                polygons.extend(part.geoms)
            elif isinstance(part, GeometryCollection):
                nested = extract_polygonal_geometry(part)
                if isinstance(nested, Polygon):
                    polygons.append(nested)
                elif isinstance(nested, MultiPolygon):
                    polygons.extend(nested.geoms)
        if polygons:
            return unary_union(polygons)

    raise ValueError(f"Expected polygonal geometry for Earth Engine, got {geometry.geom_type}.")


def earth_engine_geojson_geometry(geometry: Any) -> dict[str, Any]:
    from shapely.geometry import mapping

    geometry = force_2d_geometry(geometry)
    if not geometry.is_valid:
        geometry = make_valid_geometry(geometry)
    geometry = extract_polygonal_geometry(geometry)
    if geometry.is_empty:
        raise ValueError("Earth Engine geometry is empty after cleanup.")

    # JSON round-trip converts tuples and numpy scalar values into plain GeoJSON lists/numbers.
    return json.loads(json.dumps(mapping(geometry)))


def load_aoi(aoi_path: Path) -> Any:
    gpd, _, wkt = import_geospatial_stack()
    aoi_path = Path(aoi_path).expanduser().resolve()
    if not aoi_path.exists():
        raise FileNotFoundError(f"AOI not found: {aoi_path}")
    if aoi_path.is_dir():
        raise IsADirectoryError(
            f"AOI path points to a directory, not a vector file: {aoi_path}. "
            "Set --aoi to a GeoJSON/Shapefile/GPKG/WKT file."
        )

    suffix = aoi_path.suffix.lower()
    if suffix in {".wkt", ".txt"}:
        geom = wkt.loads(aoi_path.read_text(encoding="utf-8").strip())
        aoi = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")
    else:
        aoi = gpd.read_file(aoi_path)
        if aoi.empty:
            raise ValueError(f"AOI file contains no features: {aoi_path}")
        if aoi.crs is None:
            LOGGER.warning("AOI has no CRS; assuming EPSG:4326.")
            aoi = aoi.set_crs("EPSG:4326")

    aoi = aoi[["geometry"]].copy()
    aoi = aoi[aoi.geometry.notna() & ~aoi.geometry.is_empty]
    if aoi.empty:
        raise ValueError(f"AOI has no valid geometries: {aoi_path}")
    return aoi


def detect_tile_column(columns: Iterable[str], requested: str | None) -> str:
    columns_list = list(columns)
    if requested:
        if requested not in columns_list:
            raise ValueError(f"Tile column '{requested}' not found. Available columns: {columns_list}")
        return requested
    candidates = ["Name", "name", "tile_id", "tile", "MGRS_TILE", "mgrs_tile", "s2:mgrs_tile"]
    for candidate in candidates:
        if candidate in columns_list:
            return candidate
    raise ValueError(f"Could not auto-detect tile ID column. Available columns: {columns_list}")


def quote_sql_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def gpkg_geometry_to_shape(blob: bytes) -> Any:
    from shapely import wkb

    flags = blob[3]
    envelope_code = (flags >> 1) & 0b111
    envelope_sizes = {
        0: 0,
        1: 32,  # XY
        2: 48,  # XYZ
        3: 48,  # XYM
        4: 64,  # XYZM
    }
    envelope_size = envelope_sizes.get(envelope_code)
    if envelope_size is None:
        raise ValueError(f"Unsupported GeoPackage envelope code: {envelope_code}")
    return wkb.loads(bytes(blob[8 + envelope_size :]))


def read_gpkg_tiles_bbox_fast(tile_grid_path: Path, bbox: tuple[float, float, float, float], requested_tile_col: str | None) -> Any | None:
    gpd, _, _ = import_geospatial_stack()
    minx, miny, maxx, maxy = bbox
    with sqlite3.connect(tile_grid_path) as connection:
        content = connection.execute(
            "select table_name from gpkg_contents where data_type='features' limit 1"
        ).fetchone()
        geom_info = connection.execute(
            "select table_name, column_name, srs_id from gpkg_geometry_columns limit 1"
        ).fetchone()
        if content is None or geom_info is None:
            return None

        table_name = str(geom_info[0])
        geom_col = str(geom_info[1])
        srs_id = int(geom_info[2])
        rtree_name = f"rtree_{table_name}_{geom_col}"
        has_rtree = connection.execute(
            "select 1 from sqlite_master where type='table' and name=?",
            (rtree_name,),
        ).fetchone()
        if has_rtree is None:
            return None

        table_columns = [row[1] for row in connection.execute(f"pragma table_info({quote_sql_identifier(table_name)})")]
        tile_col = detect_tile_column(table_columns, requested_tile_col)

        query = f"""
            select t.{quote_sql_identifier(tile_col)}, t.{quote_sql_identifier(geom_col)}
            from {quote_sql_identifier(table_name)} t
            join {quote_sql_identifier(rtree_name)} r on t.fid = r.id
            where r.maxx >= ? and r.minx <= ? and r.maxy >= ? and r.miny <= ?
        """
        rows = connection.execute(query, (minx, maxx, miny, maxy)).fetchall()

    if not rows:
        return gpd.GeoDataFrame({tile_col: []}, geometry=[], crs=f"EPSG:{srs_id}")
    tile_ids = [row[0] for row in rows]
    geometries = [gpkg_geometry_to_shape(row[1]) for row in rows]
    return gpd.GeoDataFrame({tile_col: tile_ids}, geometry=geometries, crs=f"EPSG:{srs_id}")


def discover_intersecting_tiles(
    *,
    aoi: Any,
    tile_grid_path: Path,
    tile_id_column: str | None,
    include_tiles: list[str],
    max_tiles: int | None,
    output_dir: Path,
) -> Any:
    gpd, _, _ = import_geospatial_stack()
    tile_grid_path = Path(tile_grid_path).expanduser().resolve()
    if not tile_grid_path.exists():
        raise FileNotFoundError(f"Tile grid not found: {tile_grid_path}")

    aoi_wgs84 = aoi.to_crs("EPSG:4326")
    bbox_wgs84 = tuple(aoi_wgs84.total_bounds)
    tiles = None
    if tile_grid_path.suffix.lower() == ".gpkg":
        try:
            tiles = read_gpkg_tiles_bbox_fast(tile_grid_path, bbox_wgs84, tile_id_column)
        except Exception as exc:
            LOGGER.warning("Fast GeoPackage bbox read failed (%s); using GeoPandas.", exc)
            tiles = None
    if tiles is None:
        try:
            tiles = gpd.read_file(tile_grid_path, bbox=bbox_wgs84)
            if tiles.empty:
                LOGGER.warning("BBox read returned no tiles; reading the full tile grid.")
                tiles = gpd.read_file(tile_grid_path)
        except Exception as exc:
            LOGGER.warning("BBox read failed (%s); reading the full tile grid.", exc)
            tiles = gpd.read_file(tile_grid_path)
    if tiles.crs is None:
        tiles = tiles.set_crs("EPSG:4326")
    tile_col = detect_tile_column(tiles.columns, tile_id_column)
    aoi_in_grid_crs = aoi.to_crs(tiles.crs)
    aoi_geom = union_geometry(aoi_in_grid_crs)

    try:
        candidate_idx = tiles.sindex.query(aoi_geom, predicate="intersects")
        intersecting = tiles.iloc[candidate_idx].copy()
    except Exception as exc:
        LOGGER.warning("Spatial index query failed (%s); falling back to bounding-box prefilter.", exc)
        minx, miny, maxx, maxy = aoi_geom.bounds
        candidates = tiles.cx[minx:maxx, miny:maxy].copy()
        intersecting = candidates[candidates.geometry.intersects(aoi_geom)].copy()
    intersecting["tile_id"] = intersecting[tile_col].astype(str)
    intersecting = intersecting.sort_values("tile_id").drop_duplicates("tile_id")

    if include_tiles:
        include_set = {tile.upper() for tile in include_tiles}
        intersecting = intersecting[intersecting["tile_id"].str.upper().isin(include_set)].copy()

    if max_tiles is not None:
        if max_tiles <= 0:
            raise ValueError("--max-tiles must be positive.")
        intersecting = intersecting.head(max_tiles).copy()

    if intersecting.empty:
        raise ValueError("No Sentinel-2 tiles intersect the AOI after filters.")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_geojson = output_dir / "intersecting_tiles.geojson"
    manifest_csv = output_dir / "intersecting_tiles.csv"
    intersecting.to_crs("EPSG:4326").to_file(manifest_geojson, driver="GeoJSON")
    with manifest_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["tile_id"])
        writer.writeheader()
        for tile_id in intersecting["tile_id"].tolist():
            writer.writerow({"tile_id": tile_id})

    LOGGER.info("AOI intersects %d tile(s): %s", len(intersecting), ", ".join(intersecting["tile_id"]))
    return intersecting


def discover_basemap_input(
    *,
    aoi: Any,
    tile_id: str,
    output_dir: Path,
) -> Any:
    """Create a one-row pseudo tile manifest for basemap input mode."""

    gpd, _, _ = import_geospatial_stack()
    aoi_wgs84 = aoi.to_crs("EPSG:4326")
    geometry = union_geometry(aoi_wgs84)
    basemap_tile = gpd.GeoDataFrame({"tile_id": [tile_id]}, geometry=[geometry], crs="EPSG:4326")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_geojson = output_dir / "intersecting_tiles.geojson"
    manifest_csv = output_dir / "intersecting_tiles.csv"
    basemap_tile.to_file(manifest_geojson, driver="GeoJSON")
    with manifest_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["tile_id"])
        writer.writeheader()
        writer.writerow({"tile_id": tile_id})

    LOGGER.info("Basemap input mode uses pseudo tile '%s' for the AOI.", tile_id)
    return basemap_tile


def import_s2mosaic() -> Any:
    if str(S2MOSAIC_ROOT) not in sys.path:
        sys.path.insert(0, str(S2MOSAIC_ROOT))
    try:
        from s2mosaic import mosaic
    except ImportError as exc:
        raise ImportError("Could not import local S2Mosaic. Install its dependencies first.") from exc
    return mosaic


def run_mosaic(
    *,
    tile_id: str,
    start_date: dt.date,
    end_date: dt.date,
    output_dir: Path,
    args: argparse.Namespace,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(output_dir.glob("*.tif"))
    if existing and not args.overwrite and args.resume:
        LOGGER.info("Reusing existing mosaic for %s: %s", tile_id, existing[0])
        return existing[0]

    duration_days = (end_date - start_date).days
    if duration_days <= 0:
        raise ValueError("--end-date must be after --start-date.")

    mosaic = import_s2mosaic()
    additional_query: dict[str, Any] = {}
    if args.max_cloud_cover is not None:
        additional_query["eo:cloud_cover"] = {"lt": args.max_cloud_cover}

    result = mosaic(
        grid_id=tile_id,
        start_year=start_date.year,
        start_month=start_date.month,
        start_day=start_date.day,
        duration_days=duration_days,
        output_dir=output_dir,
        sort_method=args.sort_method,
        mosaic_method=args.mosaic_method,
        required_bands=parse_csv(args.bands),
        no_data_threshold=args.no_data_threshold,
        ocm_batch_size=args.ocm_batch_size,
        ocm_inference_dtype=args.ocm_inference_dtype,
        overwrite=args.overwrite,
        debug_cache=args.debug_cache,
        additional_query=additional_query,
        percentile_value=args.percentile_value,
    )
    return Path(result)


def prepare_basemap_geotiff(
    *,
    aoi: Any,
    output_tif: Path,
    run_dir: Path,
    args: argparse.Namespace,
) -> Path:
    if output_tif.exists() and not args.overwrite and args.resume:
        LOGGER.info("Reusing basemap GeoTIFF: %s", output_tif)
        return output_tif

    try:
        from basemap_tiles import (
            convert_xyz_tiles_to_geotiff,
            download_xyz_tiles,
            resolve_requests_verify,
            resolve_xyz_url_template,
        )
    except ImportError as exc:
        raise ImportError("Could not import basemap_tiles.py from the pipeline folder.") from exc

    tiles_root = args.xyz_tiles_root
    url_template = resolve_xyz_url_template(args.xyz_provider, args.xyz_url_template)
    if url_template:
        if args.xyz_zoom is None:
            raise ValueError("--xyz-zoom is required when --xyz-provider or --xyz-url-template is used.")
        if tiles_root is None:
            tiles_root = run_dir / "00_basemap_tiles"
        bounds = tuple(float(value) for value in aoi.to_crs("EPSG:4326").total_bounds)
        download_xyz_tiles(
            url_template=url_template,
            output_root=tiles_root,
            bounds_wgs84=bounds,
            zoom=args.xyz_zoom,
            provider=args.xyz_provider,
            extension=args.xyz_extension,
            timeout=args.xyz_timeout,
            sleep_seconds=args.xyz_sleep_seconds,
            overwrite=args.overwrite,
            user_agent=args.xyz_user_agent,
            max_tiles=args.xyz_max_download_tiles,
            verify_ssl=resolve_requests_verify(args.xyz_no_verify_ssl, args.xyz_ca_bundle),
            proxy=args.xyz_proxy,
            no_proxy=args.xyz_no_proxy,
            retries=args.xyz_retries,
            retry_sleep_seconds=args.xyz_retry_sleep_seconds,
            skip_failed=args.xyz_skip_failed,
        )

    if tiles_root is None:
        raise ValueError("Basemap input mode requires --xyz-tiles-root or --xyz-url-template.")

    return convert_xyz_tiles_to_geotiff(
        tiles_root=tiles_root,
        output_tif=output_tif,
        zoom=args.xyz_zoom,
        overwrite=args.overwrite,
    )


def clip_raster_to_aoi(
    *,
    input_tif: Path,
    output_tif: Path,
    aoi: Any,
    overwrite: bool,
) -> Path:
    if output_tif.exists() and not overwrite:
        LOGGER.info("Reusing clipped mosaic: %s", output_tif)
        return output_tif

    try:
        import rasterio
        from rasterio.mask import mask
        from shapely.geometry import mapping
    except ImportError as exc:
        raise ImportError("Install rasterio before clipping mosaics.") from exc

    output_tif.parent.mkdir(parents=True, exist_ok=True)
    tmp_tif = output_tif.with_suffix(".tmp.tif")
    tmp_tif.unlink(missing_ok=True)
    if output_tif.exists() and overwrite:
        output_tif.unlink()
    with rasterio.open(input_tif) as src:
        aoi_src = aoi.to_crs(src.crs)
        geometry = union_geometry(aoi_src)
        data, transform = mask(src, [mapping(geometry)], crop=True, nodata=src.nodata if src.nodata is not None else 0)
        profile = src.profile.copy()
        profile.pop("blockxsize", None)
        profile.pop("blockysize", None)
        profile.update(
            height=data.shape[1],
            width=data.shape[2],
            transform=transform,
            compress="lzw",
            BIGTIFF="IF_SAFER",
        )
        if data.shape[1] >= 256 and data.shape[2] >= 256:
            profile.update(tiled=True, blockxsize=256, blockysize=256)
        else:
            profile.update(tiled=False)
        descriptions = src.descriptions
        with rasterio.open(tmp_tif, "w", **profile) as dst:
            dst.write(data)
            if descriptions and any(descriptions):
                dst.descriptions = descriptions
    tmp_tif.replace(output_tif)
    return output_tif


def init_earth_engine(project: str, authenticate: bool, auth_mode: str | None) -> Any:
    try:
        import ee
    except ImportError as exc:
        raise ImportError("Install earthengine-api and geemap before downloading LCLU masks.") from exc

    if authenticate:
        if auth_mode:
            ee.Authenticate(auth_mode=auth_mode)
        else:
            ee.Authenticate()
    try:
        ee.Initialize(project=project)
    except Exception as exc:
        raise RuntimeError(
            "Earth Engine is not initialized. Run once with --ee-authenticate, "
            "or authenticate outside the pipeline with earthengine authenticate."
        ) from exc
    return ee


def download_dynamic_world_mask(
    *,
    geometry_wgs84: Any,
    output_tif: Path,
    start_date: dt.date,
    end_date: dt.date,
    args: argparse.Namespace,
) -> Path:
    if output_tif.exists() and not args.overwrite:
        LOGGER.info("Reusing LCLU mask: %s", output_tif)
        return output_tif

    try:
        import geemap
    except ImportError as exc:
        raise ImportError("Install geemap before downloading LCLU masks.") from exc

    ee = init_earth_engine(args.ee_project, args.ee_authenticate, args.ee_auth_mode)
    output_tif.parent.mkdir(parents=True, exist_ok=True)
    tmp_tif = output_tif.with_suffix(".tmp.tif")
    if tmp_tif.exists():
        tmp_tif.unlink()

    clean_geometry = force_2d_geometry(geometry_wgs84)
    if not clean_geometry.is_valid:
        clean_geometry = make_valid_geometry(clean_geometry)
    clean_geometry = extract_polygonal_geometry(clean_geometry)
    geometry_geojson = earth_engine_geojson_geometry(clean_geometry)
    LOGGER.info(
        "Downloading LCLU mask with EE geometry type=%s bounds=%s",
        geometry_geojson.get("type"),
        clean_geometry.bounds,
    )
    geometry = ee.Geometry(geometry_geojson)
    dw_collection = (
        ee.ImageCollection(args.lclu_collection)
        .filterBounds(geometry)
        .filterDate(start_date.isoformat(), end_date.isoformat())
    )
    dw_label = dw_collection.select("label").mode().clip(geometry).toUint8()

    if args.lclu_backend == "direct":
        download_ee_image_direct_tiled(
            image=dw_label,
            geometry=clean_geometry,
            output_tif=output_tif,
            scale=args.lclu_scale,
            crs=args.lclu_crs,
            tile_degrees=args.lclu_direct_tile_degrees,
            timeout=args.lclu_request_timeout,
        )
    else:
        geemap.download_ee_image(
            dw_label,
            filename=str(tmp_tif),
            scale=args.lclu_scale,
            region=geometry,
            crs=args.lclu_crs,
            num_threads=args.lclu_num_threads,
            max_tile_size=args.lclu_max_tile_size,
            max_tile_dim=args.lclu_max_tile_dim,
        )
        tmp_tif.replace(output_tif)
    return output_tif


def copy_or_symlink(src: Path, dst: Path, mode: str, overwrite: bool) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            LOGGER.info("Reusing staged file: %s", dst)
            return dst
        if dst.is_dir():
            raise IsADirectoryError(f"Expected file path, got directory: {dst}")
        dst.unlink()

    if mode == "symlink":
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)
    return dst


def download_url_to_file(url: str, output_path: Path, timeout: int) -> None:
    import requests

    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with output_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)


def extract_tif_if_zip(download_path: Path, output_tif: Path) -> Path:
    if not zipfile.is_zipfile(download_path):
        if download_path != output_tif:
            download_path.replace(output_tif)
        return output_tif

    with zipfile.ZipFile(download_path) as archive:
        tif_names = [name for name in archive.namelist() if name.lower().endswith((".tif", ".tiff"))]
        if not tif_names:
            raise FileNotFoundError(f"No GeoTIFF found inside {download_path}")
        with archive.open(tif_names[0]) as src, output_tif.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    download_path.unlink(missing_ok=True)
    return output_tif


def download_ee_image_direct(
    *,
    image: Any,
    ee_geometry: Any,
    output_tif: Path,
    scale: int,
    crs: str,
    timeout: int,
) -> Path:
    download_path = output_tif.with_suffix(".download")
    download_path.unlink(missing_ok=True)
    url = image.getDownloadURL(
        {
            "scale": scale,
            "region": ee_geometry,
            "crs": crs,
            "format": "GEO_TIFF",
        }
    )
    download_url_to_file(url, download_path, timeout)
    extract_tif_if_zip(download_path, output_tif)
    if not output_tif.exists() or output_tif.stat().st_size == 0:
        raise RuntimeError(f"Direct EE download produced an empty file: {output_tif}")
    return output_tif


def download_ee_image_direct_tiled(
    *,
    image: Any,
    geometry: Any,
    output_tif: Path,
    scale: int,
    crs: str,
    tile_degrees: float,
    timeout: int,
) -> Path:
    import ee
    import rasterio
    from shapely.geometry import mapping
    from rasterio.merge import merge
    from shapely.geometry import box

    if tile_degrees <= 0:
        raise ValueError("--lclu-direct-tile-degrees must be positive")

    output_tif.parent.mkdir(parents=True, exist_ok=True)
    chunks_dir = output_tif.parent / f"{output_tif.stem}.chunks"
    if chunks_dir.exists():
        shutil.rmtree(chunks_dir)
    chunks_dir.mkdir(parents=True)

    minx, miny, maxx, maxy = geometry.bounds
    nx = max(1, math.ceil((maxx - minx) / tile_degrees))
    ny = max(1, math.ceil((maxy - miny) / tile_degrees))
    chunk_paths: list[Path] = []
    total = nx * ny
    LOGGER.info("Direct EE LCLU download split into up to %d chunks (%dx%d).", total, nx, ny)

    for ix in range(nx):
        x0 = minx + ix * tile_degrees
        x1 = min(maxx, x0 + tile_degrees)
        for iy in range(ny):
            y0 = miny + iy * tile_degrees
            y1 = min(maxy, y0 + tile_degrees)
            raw_chunk_geom = force_2d_geometry(geometry.intersection(box(x0, y0, x1, y1)))
            if raw_chunk_geom.is_empty:
                continue
            try:
                chunk_geom = extract_polygonal_geometry(raw_chunk_geom)
            except ValueError:
                continue
            if chunk_geom.is_empty:
                continue

            chunk_geojson = json.loads(json.dumps(mapping(chunk_geom)))
            ee_chunk_geom = ee.Geometry(chunk_geojson)
            chunk_tif = chunks_dir / f"chunk_{ix:04d}_{iy:04d}.tif"
            LOGGER.info("Downloading LCLU chunk %d/%d -> %s", len(chunk_paths) + 1, total, chunk_tif.name)
            download_ee_image_direct(
                image=image,
                ee_geometry=ee_chunk_geom,
                output_tif=chunk_tif,
                scale=scale,
                crs=crs,
                timeout=timeout,
            )
            chunk_paths.append(chunk_tif)

    if not chunk_paths:
        raise RuntimeError("No non-empty chunks were generated for direct LCLU download.")

    datasets = [rasterio.open(path) for path in chunk_paths]
    try:
        mosaic, transform = merge(datasets)
        profile = datasets[0].profile.copy()
        profile.update(
            height=mosaic.shape[1],
            width=mosaic.shape[2],
            transform=transform,
            count=mosaic.shape[0],
            compress="lzw",
            tiled=True,
            BIGTIFF="IF_SAFER",
        )
        tmp_tif = output_tif.with_suffix(".tmp.tif")
        tmp_tif.unlink(missing_ok=True)
        with rasterio.open(tmp_tif, "w", **profile) as dst:
            dst.write(mosaic)
        tmp_tif.replace(output_tif)
    finally:
        for dataset in datasets:
            dataset.close()
        shutil.rmtree(chunks_dir, ignore_errors=True)

    if not output_tif.exists() or output_tif.stat().st_size == 0:
        raise RuntimeError(f"Direct tiled EE download produced an empty file: {output_tif}")
    return output_tif


@contextlib.contextmanager
def pushd(path: Path) -> Iterable[None]:
    old_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_cwd)


class OpenSRRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.model: Any = None
        self.device: str | None = None
        self.opensr_utils: Any = None

    def _load(self) -> None:
        if self.model is not None:
            return

        gpu_ids = parse_csv(self.args.gpus)
        if gpu_ids:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)

        if str(OPENSR_ROOT) not in sys.path:
            sys.path.insert(0, str(OPENSR_ROOT))

        try:
            import opensr_model
            import opensr_utils
            import torch
            from omegaconf import OmegaConf
        except ImportError as exc:
            raise ImportError(
                "OpenSR dependencies are missing. Install opensr-model, opensr-utils, torch, and omegaconf, "
                "or run with --skip-super-resolution for a pipeline smoke test."
            ) from exc

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        LOGGER.info("OpenSR device=%s visible_gpus=%s", self.device, torch.cuda.device_count() if self.device == "cuda" else 0)
        config = OmegaConf.load(str(self.args.opensr_config))
        self.model = opensr_model.SRLatentDiffusion(config, device=self.device)
        self.model.load_pretrained(config.ckpt_version)
        self.opensr_utils = opensr_utils

    def run(self, input_tif: Path, output_sr: Path) -> Path:
        if output_sr.exists() and not self.args.overwrite:
            LOGGER.info("Reusing SR output: %s", output_sr)
            return output_sr

        output_sr.parent.mkdir(parents=True, exist_ok=True)
        if self.args.skip_super_resolution:
            LOGGER.info("Skipping OpenSR; staging mosaic as sr.tif for %s", input_tif)
            copy_or_symlink(input_tif, output_sr, "copy", overwrite=True)
            return output_sr

        self._load()
        assert self.model is not None
        assert self.device is not None
        assert self.opensr_utils is not None

        start_time = time.time()
        window_size = parse_window_size(self.args.opensr_window_size)
        opensr_gpus = parse_gpus(self.args.gpus)
        if isinstance(opensr_gpus, list):
            LOGGER.warning(
                "OpenSR multi-GPU launches Lightning child processes and is unsafe inside the orchestrator. "
                "Using visible GPU %s for this tile. Run multiple tile-level pipeline jobs for multi-GPU throughput.",
                opensr_gpus[0],
            )
            opensr_gpus = opensr_gpus[0]

        runner_kwargs = {
            "root": str(input_tif),
            "model": self.model,
            "window_size": window_size,
            "factor": self.args.opensr_factor,
            "overlap": self.args.opensr_overlap,
            "eliminate_border_px": self.args.opensr_eliminate_border_px,
            "device": self.device,
            "gpus": opensr_gpus,
        }
        try:
            sr_object = self.opensr_utils.large_file_processing(
                **runner_kwargs,
                batch_size=self.args.opensr_batch_size,
            )
        except TypeError as exc:
            if "batch_size" not in str(exc):
                raise
            LOGGER.warning("opensr-utils does not expose batch_size; running with its default dataloader batch size.")
            sr_object = self.opensr_utils.large_file_processing(**runner_kwargs)

        candidate = self._resolve_sr_output(sr_object, input_tif, output_sr.parent, start_time)
        if candidate.resolve() != output_sr.resolve():
            copy_or_symlink(candidate, output_sr, "copy", overwrite=True)
        if not output_sr.exists():
            raise FileNotFoundError(f"OpenSR did not produce expected output: {output_sr}")
        return output_sr

    @staticmethod
    def _resolve_sr_output(sr_object: Any, input_tif: Path, output_dir: Path, start_time: float) -> Path:
        candidates: list[Path] = []
        if isinstance(sr_object, (str, Path)):
            candidates.append(Path(sr_object))
        for attr in ["final_sr_path", "output_path", "sr_path", "path"]:
            value = getattr(sr_object, attr, None)
            if value:
                candidates.append(Path(value))

        search_roots = [output_dir, input_tif.parent, OPENSR_ROOT]
        for root in search_roots:
            if root.exists():
                candidates.extend(path for path in root.rglob("*.tif") if path.stat().st_mtime >= start_time - 5)

        existing = [path for path in candidates if path.exists() and path.is_file()]
        if existing:
            return sorted(existing, key=lambda path: path.stat().st_mtime, reverse=True)[0]
        raise FileNotFoundError("OpenSR completed but no GeoTIFF output could be located.")


def write_delineate_configs(
    *,
    args: argparse.Namespace,
    include_tiles: list[str],
    run_dir: Path,
) -> tuple[Path, Path, Path]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("Install PyYAML before writing Delineate-Anything configs.") from exc

    config_dir = run_dir / "05_delineate_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    base_config_path = DELINEATE_ROOT / "conf_sample.yaml"
    with base_config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    config["model"] = parse_csv(args.delineate_models)
    config["mask_info"]["range"] = args.mask_range
    config["mask_info"]["filter_classes"] = parse_int_csv(args.mask_filter_classes)
    config["mask_info"]["clip_classes"] = parse_int_csv(args.mask_clip_classes)
    config["data_loader"]["bands"] = parse_int_csv(args.delineate_bands)
    for pass_config in config["passes"]:
        pass_config["batch_size"] = args.delineate_batch_size
        for model_args in pass_config.get("model_args", []):
            model_args["use_half"] = True

    pipeline_config_path = config_dir / "conf_pipeline.yaml"
    with pipeline_config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    output_root = args.delineate_output_root or (run_dir / "06_delineated")
    temp_root = run_dir / "05_delineate_temp"
    if args.skip_lclu:
        mask_root = config_dir / "empty_masks"
        mask_root.mkdir(parents=True, exist_ok=True)
    else:
        mask_root = DELINEATE_ROOT / "data" / "masks"
    batch_config = {
        "base_config": str(pipeline_config_path.resolve()),
        "data_root": str((DELINEATE_ROOT / "data" / "images").resolve()),
        "output_root": str(output_root.resolve()),
        "temp_root": str(temp_root.resolve()),
        "keep_temp": bool(args.keep_delineate_temp),
        "mask_root": str(mask_root.resolve()),
        "include": include_tiles,
        "exclude": None,
        "override": None,
    }
    batch_config_path = config_dir / "batch_pipeline.yaml"
    with batch_config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(batch_config, handle, sort_keys=False)
    return pipeline_config_path, batch_config_path, output_root


def run_delineate(batch_config: Path, args: argparse.Namespace) -> None:
    command = [args.python_executable, "delineate.py", "-b", str(batch_config.resolve())]
    if args.verbose_delineate:
        command.append("--verbose")
    env = os.environ.copy()
    gpu_ids = parse_csv(args.gpus)
    if gpu_ids:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)

    LOGGER.info("Running Delineate-Anything: %s", " ".join(command))
    process = subprocess.Popen(
        command,
        cwd=DELINEATE_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        LOGGER.info("[delineate] %s", line.rstrip())
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Delineate-Anything failed with exit code {return_code}.")


def run_exports(
    *,
    run_dir: Path,
    delineate_output_root: Path,
    staged_manifest: dict[str, Any],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    from export_results import export_gpkg, write_gallery, write_summary_csv

    export_root = run_dir / "07_exports"
    export_root.mkdir(parents=True, exist_ok=True)
    gpkg_paths = sorted(Path(delineate_output_root).glob("*.gpkg"))
    if not gpkg_paths:
        LOGGER.warning("No GPKG files found for export in %s", delineate_output_root)
        return []

    formats = {item.strip().lower() for item in args.export_formats.split(",") if item.strip()}
    rows: list[dict[str, Any]] = []
    for gpkg_path in gpkg_paths:
        tile_id = gpkg_path.stem.replace(".simp", "")
        raster_path = None
        if tile_id in staged_manifest and staged_manifest[tile_id].get("sr"):
            raster_path = Path(staged_manifest[tile_id]["sr"])

        LOGGER.info("Exporting delineation result: %s", gpkg_path)
        row = export_gpkg(
            gpkg_path,
            outdir=export_root / tile_id / gpkg_path.stem,
            layer=args.export_layer,
            assumed_epsg=args.export_assumed_epsg,
            raster_path=raster_path,
            formats=formats,
            dpi=args.quicklook_dpi,
            max_pixels=args.quicklook_max_pixels,
        )
        rows.append(row)

    write_summary_csv(rows, export_root / "exports_summary.csv")
    write_gallery(rows, export_root / "index.html")
    dump_json(export_root / "exports_summary.json", {"exports": rows})
    LOGGER.info("Exported %d GPKG file(s) to %s", len(rows), export_root)
    return rows


def stage_tile_for_delineation(
    *,
    tile_id: str,
    sr_tif: Path,
    mask_tif: Path | None,
    args: argparse.Namespace,
) -> dict[str, str | None]:
    image_dir = DELINEATE_ROOT / "data" / "images" / tile_id
    staged_sr = image_dir / "sr.tif"
    staged_mask = DELINEATE_ROOT / "data" / "masks" / f"{tile_id}.tif"
    copy_or_symlink(sr_tif, staged_sr, args.stage_mode, args.overwrite)

    staged_mask_value = None
    if mask_tif is not None:
        copy_or_symlink(mask_tif, staged_mask, args.stage_mode, args.overwrite)
        staged_mask_value = str(staged_mask)
    return {"image": str(staged_sr), "mask": staged_mask_value}


def save_aoi_copy(aoi: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    aoi.to_crs("EPSG:4326").to_file(output_path, driver="GeoJSON")


def main() -> None:
    args = parse_args()
    if args.input_mode == "xyz_tiles" and not args.skip_super_resolution and not args.basemap_run_super_resolution:
        args.skip_super_resolution = True
        args.basemap_auto_skipped_super_resolution = True
    else:
        args.basemap_auto_skipped_super_resolution = False

    if args.input_mode == "xyz_tiles" and args.xyz_zoom is not None and args.xyz_zoom < 0:
        raise ValueError("--xyz-zoom must be non-negative.")

    start_date = parse_date(args.start_date)
    end_date = parse_date(args.end_date)
    if end_date <= start_date:
        raise ValueError("--end-date must be after --start-date.")

    run_name = args.run_name
    if run_name is None:
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{timestamp}_{Path(args.aoi).stem}"
    run_dir = Path(args.output_root).expanduser().resolve() / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = setup_logging(run_dir)

    summary: dict[str, Any] = {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "log_path": str(log_path),
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "args": {key: str(value) for key, value in vars(args).items()},
    }
    write_summary(run_dir, summary)
    dump_json(run_dir / "manifests" / "args.json", summary["args"])

    with timed_step(summary, run_dir, "preflight"):
        ensure_repositories(args.clone_missing, required_component_repositories(args))
        check_python_environment(args)
        if args.basemap_auto_skipped_super_resolution:
            LOGGER.info("Basemap input is already high-resolution imagery; OpenSR is skipped unless --basemap-run-super-resolution is set.")
        LOGGER.info("Pipeline root: %s", PIPELINE_ROOT)
        LOGGER.info("Run directory: %s", run_dir)

    with timed_step(summary, run_dir, "aoi_and_tile_discovery"):
        aoi = load_aoi(args.aoi)
        save_aoi_copy(aoi, run_dir / "00_aoi" / "aoi.geojson")
        if args.input_mode == "sentinel2":
            tiles = discover_intersecting_tiles(
                aoi=aoi,
                tile_grid_path=args.tile_grid,
                tile_id_column=args.tile_id_column,
                include_tiles=parse_csv(args.include_tiles),
                max_tiles=args.max_tiles,
                output_dir=run_dir / "manifests",
            )
        else:
            if parse_csv(args.include_tiles) and args.xyz_name not in parse_csv(args.include_tiles):
                raise ValueError(f"--include-tiles was provided, but basemap pseudo tile is '{args.xyz_name}'.")
            tiles = discover_basemap_input(
                aoi=aoi,
                tile_id=args.xyz_name,
                output_dir=run_dir / "manifests",
            )

    if args.tiles_only:
        summary["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        summary["tiles_only"] = True
        write_summary(run_dir, summary)
        LOGGER.info("Tiles-only run complete. Intersections: %s", run_dir / "manifests" / "intersecting_tiles.geojson")
        return

    staged_tiles: list[str] = []
    staged_manifest: dict[str, Any] = {}
    opensr_runner = OpenSRRunner(args)
    aoi_wgs84 = aoi.to_crs("EPSG:4326")
    aoi_geom_wgs84 = union_geometry(aoi_wgs84)

    for _, tile in tiles.to_crs("EPSG:4326").iterrows():
        tile_id = str(tile["tile_id"])
        tile_geom_wgs84 = tile.geometry
        lclu_geom_wgs84 = tile_geom_wgs84.intersection(aoi_geom_wgs84)
        if lclu_geom_wgs84.is_empty:
            lclu_geom_wgs84 = aoi_geom_wgs84

        with timed_step(summary, run_dir, f"mosaic_{tile_id}"):
            if args.input_mode == "sentinel2":
                mosaic_path = run_mosaic(
                    tile_id=tile_id,
                    start_date=start_date,
                    end_date=end_date,
                    output_dir=run_dir / "01_mosaics" / tile_id,
                    args=args,
                )
            else:
                mosaic_path = prepare_basemap_geotiff(
                    aoi=aoi,
                    output_tif=run_dir / "01_basemap_geotiff" / f"{tile_id}.tif",
                    run_dir=run_dir,
                    args=args,
                )

        with timed_step(summary, run_dir, f"clip_{tile_id}"):
            if args.clip_to_aoi:
                model_input_path = clip_raster_to_aoi(
                    input_tif=mosaic_path,
                    output_tif=run_dir / "02_clipped_mosaics" / f"{tile_id}.tif",
                    aoi=aoi,
                    overwrite=args.overwrite,
                )
            else:
                model_input_path = mosaic_path

        mask_path: Path | None = None
        with timed_step(summary, run_dir, f"lclu_{tile_id}"):
            if args.skip_lclu:
                LOGGER.info("Skipping LCLU mask for %s", tile_id)
            else:
                mask_path = download_dynamic_world_mask(
                    geometry_wgs84=lclu_geom_wgs84,
                    output_tif=run_dir / "03_lclu_masks" / f"{tile_id}.tif",
                    start_date=start_date,
                    end_date=end_date,
                    args=args,
                )

        with timed_step(summary, run_dir, f"super_resolution_{tile_id}"):
            sr_path = opensr_runner.run(
                input_tif=model_input_path,
                output_sr=run_dir / "04_super_resolution" / tile_id / "sr.tif",
            )

        with timed_step(summary, run_dir, f"stage_delineate_{tile_id}"):
            staged = stage_tile_for_delineation(
                tile_id=tile_id,
                sr_tif=sr_path,
                mask_tif=mask_path,
                args=args,
            )
            staged_tiles.append(tile_id)
            staged_manifest[tile_id] = {
                "mosaic": str(mosaic_path),
                "model_input": str(model_input_path),
                "mask": str(mask_path) if mask_path else None,
                "sr": str(sr_path),
                **staged,
            }
            dump_json(run_dir / "manifests" / "staged_inputs.json", staged_manifest)

    with timed_step(summary, run_dir, "write_delineate_configs"):
        _, batch_config, delineate_output_root = write_delineate_configs(
            args=args,
            include_tiles=staged_tiles,
            run_dir=run_dir,
        )
        summary["delineate_batch_config"] = str(batch_config)
        summary["delineate_output_root"] = str(delineate_output_root)
        write_summary(run_dir, summary)

    if args.skip_delineation:
        LOGGER.info("Skipping Delineate-Anything execution. Inputs and configs are ready.")
    else:
        with timed_step(summary, run_dir, "delineation"):
            run_delineate(batch_config, args)

    if args.skip_exports:
        LOGGER.info("Skipping result exports.")
    else:
        with timed_step(summary, run_dir, "exports"):
            export_rows = run_exports(
                run_dir=run_dir,
                delineate_output_root=delineate_output_root,
                staged_manifest=staged_manifest,
                args=args,
            )
            summary["exports"] = export_rows
            summary["export_root"] = str(run_dir / "07_exports")
            write_summary(run_dir, summary)

    summary["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    write_summary(run_dir, summary)
    LOGGER.info("Pipeline complete. Summary: %s", run_dir / "manifests" / "run_summary.json")


if __name__ == "__main__":
    main()
