#!/usr/bin/env python3
"""Find Sentinel-2 MGRS tiles intersecting an AOI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from pipeline import DEFAULT_TILE_GRID, discover_intersecting_tiles, load_aoi, parse_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Write Sentinel-2 tile intersections for an AOI.")
    parser.add_argument("--aoi", required=True, type=Path, help="AOI file: GeoJSON/Shapefile/GPKG or WKT text.")
    parser.add_argument("--tile-grid", default=DEFAULT_TILE_GRID, type=Path)
    parser.add_argument("--tile-id-column", default=None)
    parser.add_argument("--include-tiles", default=None, help="Comma-separated tile IDs to keep.")
    parser.add_argument("--max-tiles", default=None, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    aoi = load_aoi(args.aoi)
    tiles = discover_intersecting_tiles(
        aoi=aoi,
        tile_grid_path=args.tile_grid,
        tile_id_column=args.tile_id_column,
        include_tiles=parse_csv(args.include_tiles),
        max_tiles=args.max_tiles,
        output_dir=args.output_dir,
    )
    print("\n".join(tiles["tile_id"].astype(str).tolist()))


if __name__ == "__main__":
    main()
