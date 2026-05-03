#!/usr/bin/env python3
"""
End-to-end field delineation pipeline orchestrator.

The pipeline is intentionally modular:
1. find Sentinel-2 MGRS tiles intersecting an AOI
2. mosaic each tile with S2Mosaic
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
import json
import logging
import os
import sqlite3
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterable


PIPELINE_ROOT = Path(__file__).resolve().parent
S2MOSAIC_ROOT = PIPELINE_ROOT / "S2Mosaic"
OPENSR_ROOT = PIPELINE_ROOT / "opensr-model"
DELINEATE_ROOT = PIPELINE_ROOT / "Delineate-Anything"
DEFAULT_TILE_GRID = S2MOSAIC_ROOT / "s2mosaic" / "sentinel_2_index.gpkg"

REPO_SPECS = {
    "S2Mosaic": "https://github.com/DPIRD-DMA/S2Mosaic.git",
    "opensr-model": "https://github.com/ESAOpenSR/opensr-model.git",
    "Delineate-Anything": "https://github.com/Lavreniuk/Delineate-Anything.git",
}

REPO_MARKERS = {
    "S2Mosaic": "s2mosaic/__init__.py",
    "opensr-model": "opensr_model/__init__.py",
    "Delineate-Anything": "delineate.py",
}

LOGGER = logging.getLogger("field_delineation_pipeline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Sentinel-2 mosaic, LCLU mask, OpenSR, and Delineate-Anything as one pipeline."
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
    lclu.add_argument("--lclu-scale", default=10, type=int)
    lclu.add_argument("--lclu-crs", default="EPSG:4326")
    lclu.add_argument("--lclu-collection", default="GOOGLE/DYNAMICWORLD/V1")

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


def ensure_repositories(clone_missing: bool) -> None:
    for repo_name, repo_url in REPO_SPECS.items():
        repo_path = PIPELINE_ROOT / repo_name
        marker_path = repo_path / REPO_MARKERS[repo_name]
        if marker_path.exists():
            continue

        if not clone_missing:
            raise FileNotFoundError(
                f"Missing or incomplete {repo_name} at {repo_path}. "
                f"Re-run with --clone-missing to clone {repo_url}, or initialize Git submodules."
            )

        if repo_path.exists() and not repo_path.is_dir():
            raise FileExistsError(f"Expected {repo_path} to be a directory, but it is a file.")

        if repo_path.exists() and any(repo_path.iterdir()):
            raise FileExistsError(
                f"{repo_path} exists but does not look like a complete {repo_name} checkout. "
                "It is not empty, so the pipeline will not overwrite it. "
                "Initialize submodules or move/remove that directory manually."
            )

        LOGGER.info("Cloning %s from %s", repo_name, repo_url)
        subprocess.run(["git", "clone", repo_url, str(repo_path)], check=True)


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


def load_aoi(aoi_path: Path) -> Any:
    gpd, _, wkt = import_geospatial_stack()
    aoi_path = Path(aoi_path).expanduser().resolve()
    if not aoi_path.exists():
        raise FileNotFoundError(f"AOI not found: {aoi_path}")

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
    with rasterio.open(input_tif) as src:
        aoi_src = aoi.to_crs(src.crs)
        geometry = union_geometry(aoi_src)
        data, transform = mask(src, [mapping(geometry)], crop=True, nodata=src.nodata if src.nodata is not None else 0)
        profile = src.profile.copy()
        profile.update(
            height=data.shape[1],
            width=data.shape[2],
            transform=transform,
            compress="lzw",
            tiled=True,
            BIGTIFF="IF_SAFER",
        )
        descriptions = src.descriptions
        with rasterio.open(output_tif, "w", **profile) as dst:
            dst.write(data)
            if descriptions and any(descriptions):
                dst.descriptions = descriptions
    return output_tif


def init_earth_engine(project: str, authenticate: bool) -> Any:
    try:
        import ee
    except ImportError as exc:
        raise ImportError("Install earthengine-api and geemap before downloading LCLU masks.") from exc

    if authenticate:
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
        from shapely.geometry import mapping
    except ImportError as exc:
        raise ImportError("Install geemap before downloading LCLU masks.") from exc

    ee = init_earth_engine(args.ee_project, args.ee_authenticate)
    output_tif.parent.mkdir(parents=True, exist_ok=True)
    tmp_tif = output_tif.with_suffix(".tmp.tif")
    if tmp_tif.exists():
        tmp_tif.unlink()

    geometry = ee.Geometry(mapping(geometry_wgs84))
    dw_collection = (
        ee.ImageCollection(args.lclu_collection)
        .filterBounds(geometry)
        .filterDate(start_date.isoformat(), end_date.isoformat())
    )
    dw_label = dw_collection.select("label").mode().clip(geometry).toUint8()

    geemap.download_ee_image(
        dw_label,
        filename=str(tmp_tif),
        scale=args.lclu_scale,
        region=geometry,
        crs=args.lclu_crs,
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
        with pushd(OPENSR_ROOT):
            runner_kwargs = {
                "root": str(input_tif),
                "model": self.model,
                "window_size": window_size,
                "factor": self.args.opensr_factor,
                "overlap": self.args.opensr_overlap,
                "eliminate_border_px": self.args.opensr_eliminate_border_px,
                "device": self.device,
                "gpus": parse_gpus(self.args.gpus),
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
        ensure_repositories(args.clone_missing)
        LOGGER.info("Pipeline root: %s", PIPELINE_ROOT)
        LOGGER.info("Run directory: %s", run_dir)

    with timed_step(summary, run_dir, "aoi_and_tile_discovery"):
        aoi = load_aoi(args.aoi)
        save_aoi_copy(aoi, run_dir / "00_aoi" / "aoi.geojson")
        tiles = discover_intersecting_tiles(
            aoi=aoi,
            tile_grid_path=args.tile_grid,
            tile_id_column=args.tile_id_column,
            include_tiles=parse_csv(args.include_tiles),
            max_tiles=args.max_tiles,
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
            mosaic_path = run_mosaic(
                tile_id=tile_id,
                start_date=start_date,
                end_date=end_date,
                output_dir=run_dir / "01_mosaics" / tile_id,
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

    summary["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    write_summary(run_dir, summary)
    LOGGER.info("Pipeline complete. Summary: %s", run_dir / "manifests" / "run_summary.json")


if __name__ == "__main__":
    main()
