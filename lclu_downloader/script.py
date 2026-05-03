#!/usr/bin/env python3
"""Download a Dynamic World label mask for an AOI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from pipeline import download_dynamic_world_mask, load_aoi, parse_date, union_geometry


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Dynamic World LCLU mask for an AOI.")
    parser.add_argument("--aoi", required=True, type=Path, help="AOI file: GeoJSON/Shapefile/GPKG or WKT text.")
    parser.add_argument("--output", required=True, type=Path, help="Output GeoTIFF path.")
    parser.add_argument("--start-date", required=True, help="Inclusive start date, YYYY-MM-DD.")
    parser.add_argument("--end-date", required=True, help="Exclusive end date, YYYY-MM-DD.")
    parser.add_argument("--ee-project", default="agriculture-486211")
    parser.add_argument("--ee-authenticate", action="store_true")
    parser.add_argument("--lclu-scale", default=10, type=int)
    parser.add_argument("--lclu-crs", default="EPSG:4326")
    parser.add_argument("--lclu-collection", default="GOOGLE/DYNAMICWORLD/V1")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    aoi = load_aoi(args.aoi).to_crs("EPSG:4326")
    output = download_dynamic_world_mask(
        geometry_wgs84=union_geometry(aoi),
        output_tif=args.output,
        start_date=parse_date(args.start_date),
        end_date=parse_date(args.end_date),
        args=args,
    )
    print(output)


if __name__ == "__main__":
    main()
