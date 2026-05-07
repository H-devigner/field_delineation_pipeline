#!/usr/bin/env python3
"""Postprocess Delineate-Anything instance rasters across tiles.

The script polygonizes positive instance IDs, reconciles IDs across raster
seams, dissolves merged fields, and writes a global GPKG/GeoJSON/PNG quicklook.
It is intended for rasters produced with pipeline.py --save-instance-rasters.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape


LOGGER = logging.getLogger("instance_raster_postprocess")


@dataclass(frozen=True)
class RasterInfo:
    path: Path
    source_index: int
    tile_id: str
    crs: Any
    pixel_width: float
    pixel_height: float

    @property
    def pixel_area(self) -> float:
        return abs(self.pixel_width * self.pixel_height)

    @property
    def pixel_size(self) -> float:
        return (abs(self.pixel_width) + abs(self.pixel_height)) / 2.0


class UnionFind:
    def __init__(self, values: list[int]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: int) -> int:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, a: int, b: int) -> None:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return
        if root_b < root_a:
            root_a, root_b = root_b, root_a
        self.parent[root_b] = root_a


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance-root", required=True, type=Path, help="Root containing *.instances.tif files.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for merged outputs.")
    parser.add_argument("--output-name", default="merged_instance_fields", help="Base name for output files.")
    parser.add_argument(
        "--glob",
        default="**/*.instances.tif",
        help="Glob under --instance-root. Use '*.tif' if your files are flat.",
    )
    parser.add_argument("--layer-name", default="fields", help="GPKG layer name.")
    parser.add_argument("--connectivity", default=4, type=int, choices=[4, 8], help="Raster polygonization connectivity.")
    parser.add_argument("--min-area", default=0.0, type=float, help="Drop merged polygons below this area in source CRS units.")
    parser.add_argument(
        "--min-overlap-ratio",
        default=0.15,
        type=float,
        help="Merge cross-raster IDs when overlap area / smaller area is at least this value.",
    )
    parser.add_argument(
        "--min-overlap-pixels",
        default=16.0,
        type=float,
        help="Merge cross-raster IDs when overlap area is at least this many pixels.",
    )
    parser.add_argument(
        "--merge-touching",
        action="store_true",
        help="Also merge IDs from different rasters that touch or nearly touch across a seam.",
    )
    parser.add_argument(
        "--touch-distance-pixels",
        default=1.5,
        type=float,
        help="Search distance for --merge-touching, expressed in pixels.",
    )
    parser.add_argument(
        "--min-touch-pixels",
        default=8.0,
        type=float,
        help="Minimum boundary contact length for --merge-touching, expressed in pixels.",
    )
    parser.add_argument("--write-raw", action="store_true", help="Write polygonized pre-merge instances for debugging.")
    parser.add_argument("--skip-geojson", action="store_true", help="Skip WGS84 GeoJSON export.")
    parser.add_argument("--skip-quicklook", action="store_true", help="Skip PNG quicklook export.")
    parser.add_argument("--quicklook-dpi", default=220, type=int)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def discover_rasters(instance_root: Path, pattern: str) -> list[Path]:
    paths = sorted(path for path in instance_root.glob(pattern) if path.is_file())
    if not paths:
        raise FileNotFoundError(f"No instance rasters found under {instance_root} with glob {pattern!r}")
    return paths


def tile_id_from_path(path: Path, instance_root: Path) -> str:
    if path.parent.resolve() != instance_root.resolve() and path.parent.name:
        return path.parent.name
    return path.name.split(".instances")[0]


def make_valid_geometry(geom: Any) -> Any:
    if geom is None or geom.is_empty:
        return geom
    try:
        from shapely import make_valid

        return make_valid(geom)
    except Exception:
        return geom.buffer(0)


def polygonize_instance_raster(info: RasterInfo, connectivity: int) -> gpd.GeoDataFrame:
    rows: list[dict[str, Any]] = []
    LOGGER.info("Polygonizing %s", info.path)
    with rasterio.open(info.path) as src:
        if src.crs != info.crs:
            raise ValueError(f"Unexpected CRS drift in {info.path}: {src.crs} != {info.crs}")
        for geom_mapping, value in shapes(rasterio.band(src, 1), connectivity=connectivity, transform=src.transform):
            instance_id = int(value)
            if instance_id <= 0:
                continue
            geom = make_valid_geometry(shape(geom_mapping))
            if geom is None or geom.is_empty:
                continue
            rows.append(
                {
                    "source_index": info.source_index,
                    "tile_id": info.tile_id,
                    "local_id": instance_id,
                    "global_id": (info.source_index + 1) * 10_000_000_000 + instance_id,
                    "source_path": str(info.path),
                    "geometry": geom,
                }
            )
            if len(rows) % 10000 == 0:
                LOGGER.info("  %s positive raster parts polygonized from %s", len(rows), info.path.name)

    if not rows:
        LOGGER.warning("No positive instance IDs found in %s", info.path)
        return gpd.GeoDataFrame(
            columns=["source_index", "tile_id", "local_id", "global_id", "source_path", "geometry"],
            geometry="geometry",
            crs=info.crs,
        )
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=info.crs)


def dissolve_local_instances(parts: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if parts.empty:
        return parts
    LOGGER.info("Dissolving polygon fragments by source/local instance ID")
    dissolved = parts.dissolve(
        by=["source_index", "tile_id", "local_id", "global_id", "source_path"],
        as_index=False,
    )
    dissolved["geometry"] = dissolved.geometry.map(make_valid_geometry)
    dissolved = dissolved[~dissolved.geometry.is_empty].copy()
    dissolved["area"] = dissolved.geometry.area
    return dissolved


def representative_pixel_metrics(rasters: list[RasterInfo]) -> tuple[float, float]:
    pixel_sizes = [info.pixel_size for info in rasters if info.pixel_size > 0]
    pixel_areas = [info.pixel_area for info in rasters if info.pixel_area > 0]
    if not pixel_sizes or not pixel_areas:
        raise ValueError("Could not determine positive pixel size/area from instance rasters.")
    return float(sorted(pixel_sizes)[len(pixel_sizes) // 2]), float(sorted(pixel_areas)[len(pixel_areas) // 2])


def find_merge_pairs(
    instances: gpd.GeoDataFrame,
    *,
    pixel_size: float,
    pixel_area: float,
    min_overlap_ratio: float,
    min_overlap_pixels: float,
    merge_touching: bool,
    touch_distance_pixels: float,
    min_touch_pixels: float,
) -> tuple[list[tuple[int, int]], dict[str, Any]]:
    if instances.empty:
        return [], {}

    min_overlap_area = min_overlap_pixels * pixel_area
    touch_distance = max(0.0, touch_distance_pixels * pixel_size)
    min_touch_length = min_touch_pixels * pixel_size

    LOGGER.info("Finding seam merge candidates")
    LOGGER.info("  min_overlap_area=%s, min_overlap_ratio=%s", min_overlap_area, min_overlap_ratio)
    if merge_touching:
        LOGGER.info("  touch_distance=%s, min_touch_length=%s", touch_distance, min_touch_length)

    pairs: list[tuple[int, int]] = []
    stats = {
        "candidate_pairs": 0,
        "overlap_pairs": 0,
        "touch_pairs": 0,
    }
    spatial_index = instances.sindex
    geoms = instances.geometry
    source_indices = instances["source_index"].to_numpy()
    global_ids = instances["global_id"].to_numpy()
    areas = instances["area"].to_numpy()

    for pos, geom in enumerate(geoms):
        if geom is None or geom.is_empty:
            continue
        query_geom = geom.buffer(touch_distance) if merge_touching and touch_distance > 0 else geom
        candidate_positions = spatial_index.query(query_geom, predicate="intersects")
        for other_pos in candidate_positions:
            other_pos = int(other_pos)
            if other_pos <= pos:
                continue
            if source_indices[pos] == source_indices[other_pos]:
                continue

            other = geoms.iloc[other_pos]
            if other is None or other.is_empty:
                continue

            stats["candidate_pairs"] += 1
            inter_area = geom.intersection(other).area
            if inter_area > 0:
                smaller_area = max(min(float(areas[pos]), float(areas[other_pos])), 1e-9)
                overlap_ratio = inter_area / smaller_area
                if inter_area >= min_overlap_area or overlap_ratio >= min_overlap_ratio:
                    pairs.append((int(global_ids[pos]), int(global_ids[other_pos])))
                    stats["overlap_pairs"] += 1
                    continue

            if merge_touching:
                distance = geom.distance(other)
                if distance <= touch_distance:
                    boundary_contact = geom.boundary.intersection(other.boundary.buffer(touch_distance)).length
                    if boundary_contact >= min_touch_length:
                        pairs.append((int(global_ids[pos]), int(global_ids[other_pos])))
                        stats["touch_pairs"] += 1

    LOGGER.info(
        "Merge candidates: %d accepted from %d checked pairs (%d overlap, %d touching)",
        len(pairs),
        stats["candidate_pairs"],
        stats["overlap_pairs"],
        stats["touch_pairs"],
    )
    return pairs, stats


def apply_merges(instances: gpd.GeoDataFrame, pairs: list[tuple[int, int]], min_area: float) -> gpd.GeoDataFrame:
    if instances.empty:
        return instances

    uf = UnionFind([int(value) for value in instances["global_id"].tolist()])
    for a, b in pairs:
        uf.union(a, b)

    root_to_merged: dict[int, int] = {}
    merged_ids: list[int] = []
    for global_id in instances["global_id"]:
        root = uf.find(int(global_id))
        if root not in root_to_merged:
            root_to_merged[root] = len(root_to_merged) + 1
        merged_ids.append(root_to_merged[root])

    merged = instances.copy()
    merged["merged_id"] = merged_ids
    merged["source_count"] = 1
    LOGGER.info("Dissolving %d local instances into %d merged fields", len(merged), len(root_to_merged))
    merged = merged.dissolve(
        by="merged_id",
        as_index=False,
        aggfunc={
            "tile_id": lambda values: ",".join(sorted(set(map(str, values)))),
            "source_path": "first",
            "source_count": "sum",
        },
    )
    merged["geometry"] = merged.geometry.map(make_valid_geometry)
    merged = merged[~merged.geometry.is_empty].copy()
    merged["area"] = merged.geometry.area
    if min_area > 0:
        before = len(merged)
        merged = merged[merged["area"] >= min_area].copy()
        LOGGER.info("Dropped %d merged fields below min_area=%s", before - len(merged), min_area)
    merged = merged[["merged_id", "tile_id", "source_count", "area", "geometry"]]
    return merged


def write_quicklook(gdf: gpd.GeoDataFrame, output_path: Path, dpi: int) -> None:
    if gdf.empty:
        LOGGER.warning("Skipping quicklook because merged output is empty.")
        return
    fig, ax = plt.subplots(figsize=(12, 12))
    gdf.boundary.plot(ax=ax, color="black", linewidth=0.25)
    ax.set_axis_off()
    ax.set_aspect("equal")
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.02, dpi=dpi)
    plt.close(fig)


def write_outputs(
    merged: gpd.GeoDataFrame,
    raw: gpd.GeoDataFrame,
    output_dir: Path,
    output_name: str,
    layer_name: str,
    write_raw: bool,
    skip_geojson: bool,
    skip_quicklook: bool,
    quicklook_dpi: int,
    summary: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    gpkg_path = output_dir / f"{output_name}.merged_fields.gpkg"
    merged.to_file(gpkg_path, layer=layer_name, driver="GPKG")
    LOGGER.info("Wrote %s", gpkg_path)

    if write_raw:
        raw_path = output_dir / f"{output_name}.raw_instances.gpkg"
        raw.to_file(raw_path, layer="raw_instances", driver="GPKG")
        LOGGER.info("Wrote %s", raw_path)
        summary["raw_gpkg"] = str(raw_path)

    if not skip_geojson:
        geojson_path = output_dir / f"{output_name}.merged_fields.geojson"
        merged.to_crs("EPSG:4326").to_file(geojson_path, driver="GeoJSON")
        LOGGER.info("Wrote %s", geojson_path)
        summary["geojson"] = str(geojson_path)

    if not skip_quicklook:
        quicklook_path = output_dir / f"{output_name}.merged_fields.png"
        write_quicklook(merged, quicklook_path, quicklook_dpi)
        summary["quicklook_png"] = str(quicklook_path)
        LOGGER.info("Wrote %s", quicklook_path)

    summary["gpkg"] = str(gpkg_path)
    summary_path = output_dir / f"{output_name}.merge_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOGGER.info("Wrote %s", summary_path)


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    raster_paths = discover_rasters(args.instance_root.expanduser().resolve(), args.glob)
    infos: list[RasterInfo] = []
    first_crs = None
    for source_index, path in enumerate(raster_paths):
        with rasterio.open(path) as src:
            if first_crs is None:
                first_crs = src.crs
            elif src.crs != first_crs:
                raise ValueError(f"All instance rasters must share one CRS. {path} has {src.crs}, expected {first_crs}.")
            infos.append(
                RasterInfo(
                    path=path,
                    source_index=source_index,
                    tile_id=tile_id_from_path(path, args.instance_root.expanduser().resolve()),
                    crs=src.crs,
                    pixel_width=float(src.transform.a),
                    pixel_height=float(src.transform.e),
                )
            )

    pixel_size, pixel_area = representative_pixel_metrics(infos)
    LOGGER.info("Found %d instance raster(s); representative pixel_size=%s pixel_area=%s", len(infos), pixel_size, pixel_area)

    parts = [polygonize_instance_raster(info, args.connectivity) for info in infos]
    raw = gpd.GeoDataFrame(
        pd.concat(parts, ignore_index=True),
        geometry="geometry",
        crs=first_crs,
    )
    if raw.empty:
        raise RuntimeError("No positive instance polygons were created from the input rasters.")

    instances = dissolve_local_instances(raw)
    pairs, merge_stats = find_merge_pairs(
        instances,
        pixel_size=pixel_size,
        pixel_area=pixel_area,
        min_overlap_ratio=args.min_overlap_ratio,
        min_overlap_pixels=args.min_overlap_pixels,
        merge_touching=args.merge_touching,
        touch_distance_pixels=args.touch_distance_pixels,
        min_touch_pixels=args.min_touch_pixels,
    )
    merged = apply_merges(instances, pairs, args.min_area)

    summary = {
        "instance_root": str(args.instance_root),
        "rasters": [str(path) for path in raster_paths],
        "raster_count": len(raster_paths),
        "raw_polygon_parts": int(len(raw)),
        "local_instances": int(len(instances)),
        "merged_fields": int(len(merged)),
        "merge_pairs": int(len(pairs)),
        "merge_stats": merge_stats,
        "merge_touching": bool(args.merge_touching),
        "min_overlap_ratio": args.min_overlap_ratio,
        "min_overlap_pixels": args.min_overlap_pixels,
        "touch_distance_pixels": args.touch_distance_pixels,
        "min_touch_pixels": args.min_touch_pixels,
    }
    write_outputs(
        merged=merged,
        raw=instances,
        output_dir=args.output_dir.expanduser().resolve(),
        output_name=args.output_name,
        layer_name=args.layer_name,
        write_raw=args.write_raw,
        skip_geojson=args.skip_geojson,
        skip_quicklook=args.skip_quicklook,
        quicklook_dpi=args.quicklook_dpi,
        summary=summary,
    )


if __name__ == "__main__":
    main()
