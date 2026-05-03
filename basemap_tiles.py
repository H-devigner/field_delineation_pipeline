#!/usr/bin/env python3
"""Download and mosaic XYZ basemap tiles into georeferenced GeoTIFFs."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


WEB_MERCATOR_LIMIT = 85.0511287798066
WEB_MERCATOR_RADIUS = 6378137.0
WEB_MERCATOR_ORIGIN_SHIFT = math.pi * WEB_MERCATOR_RADIUS
SUPPORTED_TILE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}
XYZ_PROVIDER_SPECS: dict[str, dict[str, str]] = {
    "openaerialmap": {
        "url_template": "https://apps.kontur.io/raster-tiler/oam/mosaic/{z}/{x}/{y}.png",
        "attribution": "OpenAerialMap / Open Imagery Network contributors; mosaic tiles by Kontur.",
        "notes": "Open imagery coverage varies by AOI. Use small AOIs and keep attribution with derived outputs.",
    },
}

LOGGER = logging.getLogger("basemap_tiles")


@dataclass(frozen=True)
class XYZTile:
    z: int
    x: int
    y: int
    path: Path


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def resolve_xyz_url_template(provider: str | None = None, url_template: str | None = None) -> str | None:
    if url_template:
        return url_template
    if provider is None:
        return None
    if provider not in XYZ_PROVIDER_SPECS:
        raise ValueError(f"Unknown XYZ provider '{provider}'. Choose one of: {', '.join(sorted(XYZ_PROVIDER_SPECS))}")
    return XYZ_PROVIDER_SPECS[provider]["url_template"]


def xyz_provider_attribution(provider: str | None) -> str | None:
    if provider is None:
        return None
    return XYZ_PROVIDER_SPECS.get(provider, {}).get("attribution")


def lonlat_to_xyz(lon: float, lat: float, zoom: int) -> tuple[int, int]:
    """Return the XYZ tile index containing a longitude/latitude coordinate."""

    lat = clamp(lat, -WEB_MERCATOR_LIMIT, WEB_MERCATOR_LIMIT)
    lon = clamp(lon, -180.0, 180.0)
    n = 2**zoom
    lat_rad = math.radians(lat)
    x = int(math.floor((lon + 180.0) / 360.0 * n))
    y = int(math.floor((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n))
    return clamp(x, 0, n - 1), clamp(y, 0, n - 1)


def xyz_range_for_bounds(bounds_wgs84: tuple[float, float, float, float], zoom: int) -> tuple[range, range]:
    """Return inclusive x/y XYZ ranges for WGS84 bounds."""

    min_lon, min_lat, max_lon, max_lat = bounds_wgs84
    if zoom < 0:
        raise ValueError("Zoom must be non-negative.")
    x_west, y_north = lonlat_to_xyz(min_lon, max_lat, zoom)
    x_east, y_south = lonlat_to_xyz(max_lon, min_lat, zoom)
    x0, x1 = sorted((x_west, x_east))
    y0, y1 = sorted((y_north, y_south))
    return range(x0, x1 + 1), range(y0, y1 + 1)


def xyz_tile_bounds_mercator(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    """Return tile bounds as (minx, miny, maxx, maxy) in EPSG:3857."""

    n = 2**z
    if z < 0 or not (0 <= x < n) or not (0 <= y < n):
        raise ValueError(f"Invalid XYZ tile index z={z} x={x} y={y}.")
    tile_span = 2 * WEB_MERCATOR_ORIGIN_SHIFT / n
    minx = -WEB_MERCATOR_ORIGIN_SHIFT + x * tile_span
    maxx = minx + tile_span
    maxy = WEB_MERCATOR_ORIGIN_SHIFT - y * tile_span
    miny = maxy - tile_span
    return minx, miny, maxx, maxy


def parse_int(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:
        return None


def parse_xyz_tile_path(path: Path, root: Path, zoom: int | None) -> XYZTile | None:
    """Parse common XYZ layouts: root/z/x/y.png or root/x/y.png with --zoom."""

    try:
        parts = path.relative_to(root).parts
    except ValueError:
        return None
    if not parts or path.suffix.lower() not in SUPPORTED_TILE_SUFFIXES:
        return None

    stem_y = parse_int(Path(parts[-1]).stem)
    if stem_y is None:
        return None

    if len(parts) >= 3:
        maybe_z = parse_int(parts[-3])
        maybe_x = parse_int(parts[-2])
        if maybe_z is not None and maybe_x is not None:
            if zoom is not None and maybe_z != zoom:
                return None
            n = 2**maybe_z
            if maybe_z >= 0 and 0 <= maybe_x < n and 0 <= stem_y < n:
                return XYZTile(z=maybe_z, x=maybe_x, y=stem_y, path=path)
            return None

    if zoom is not None and len(parts) >= 2:
        maybe_x = parse_int(parts[-2])
        if maybe_x is not None:
            n = 2**zoom
            if zoom >= 0 and 0 <= maybe_x < n and 0 <= stem_y < n:
                return XYZTile(z=zoom, x=maybe_x, y=stem_y, path=path)
    return None


def discover_xyz_tiles(tiles_root: Path, zoom: int | None = None) -> list[XYZTile]:
    tiles_root = Path(tiles_root).expanduser().resolve()
    if not tiles_root.exists():
        raise FileNotFoundError(f"XYZ tiles root not found: {tiles_root}")

    tiles: list[XYZTile] = []
    for path in sorted(tiles_root.rglob("*")):
        if not path.is_file():
            continue
        tile = parse_xyz_tile_path(path, tiles_root, zoom)
        if tile is not None:
            tiles.append(tile)

    if not tiles:
        suffixes = ", ".join(sorted(SUPPORTED_TILE_SUFFIXES))
        raise FileNotFoundError(f"No XYZ tile images ({suffixes}) found under {tiles_root}")

    zooms = {tile.z for tile in tiles}
    if len(zooms) > 1:
        raise ValueError(f"Found multiple zoom levels {sorted(zooms)}. Pass --zoom to select one.")
    return tiles


def convert_xyz_tiles_to_geotiff(
    *,
    tiles_root: Path,
    output_tif: Path,
    zoom: int | None = None,
    overwrite: bool = False,
) -> Path:
    """Mosaic local XYZ image tiles into one EPSG:3857 GeoTIFF."""

    import numpy as np
    import rasterio
    from rasterio.errors import NotGeoreferencedWarning
    from rasterio.transform import from_origin

    output_tif = Path(output_tif).expanduser().resolve()
    if output_tif.exists() and not overwrite:
        LOGGER.info("Reusing existing basemap GeoTIFF: %s", output_tif)
        return output_tif

    tiles = discover_xyz_tiles(tiles_root, zoom=zoom)
    zoom_level = tiles[0].z
    tile_by_index = {(tile.x, tile.y): tile for tile in tiles}
    xs = [tile.x for tile in tiles]
    ys = [tile.y for tile in tiles]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    first_tile = tiles[0]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
        with rasterio.open(first_tile.path) as src:
            first = src.read()
            band_count, tile_height, tile_width = first.shape
            dtype = first.dtype

    if tile_width != tile_height:
        raise ValueError(f"Expected square XYZ tiles, got {tile_width}x{tile_height} for {first_tile.path}")
    if band_count not in {1, 3, 4}:
        raise ValueError(f"Expected 1, 3, or 4 bands in basemap tiles, got {band_count} for {first_tile.path}")

    mosaic_width = (max_x - min_x + 1) * tile_width
    mosaic_height = (max_y - min_y + 1) * tile_height
    mosaic = np.zeros((band_count, mosaic_height, mosaic_width), dtype=dtype)

    for (x, y), tile in sorted(tile_by_index.items()):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
            with rasterio.open(tile.path) as src:
                data = src.read()
                if data.shape != first.shape:
                    raise ValueError(
                        f"Tile shape mismatch for {tile.path}: expected {first.shape}, got {data.shape}."
                    )
                if data.dtype != dtype:
                    data = data.astype(dtype, copy=False)
        row = (y - min_y) * tile_height
        col = (x - min_x) * tile_width
        mosaic[:, row : row + tile_height, col : col + tile_width] = data

    expected_tiles = (max_x - min_x + 1) * (max_y - min_y + 1)
    if expected_tiles != len(tile_by_index):
        LOGGER.warning(
            "XYZ tile mosaic has %d missing tile(s) inside the %dx%d tile extent; missing pixels stay at 0.",
            expected_tiles - len(tile_by_index),
            max_x - min_x + 1,
            max_y - min_y + 1,
        )

    left, _, _, top = xyz_tile_bounds_mercator(zoom_level, min_x, min_y)
    _, bottom, right, _ = xyz_tile_bounds_mercator(zoom_level, max_x, max_y)
    pixel_width = (right - left) / mosaic_width
    pixel_height = (top - bottom) / mosaic_height
    transform = from_origin(left, top, pixel_width, pixel_height)

    profile: dict[str, Any] = {
        "driver": "GTiff",
        "height": mosaic_height,
        "width": mosaic_width,
        "count": band_count,
        "dtype": str(dtype),
        "crs": "EPSG:3857",
        "transform": transform,
        "compress": "lzw",
        "BIGTIFF": "IF_SAFER",
    }
    if mosaic_width >= 256 and mosaic_height >= 256:
        profile.update(tiled=True, blockxsize=256, blockysize=256)

    output_tif.parent.mkdir(parents=True, exist_ok=True)
    tmp_tif = output_tif.with_suffix(".tmp.tif")
    tmp_tif.unlink(missing_ok=True)
    with rasterio.open(tmp_tif, "w", **profile) as dst:
        dst.write(mosaic)
        if band_count == 3:
            dst.set_band_description(1, "red")
            dst.set_band_description(2, "green")
            dst.set_band_description(3, "blue")
        elif band_count == 4:
            dst.set_band_description(1, "red")
            dst.set_band_description(2, "green")
            dst.set_band_description(3, "blue")
            dst.set_band_description(4, "alpha")
    tmp_tif.replace(output_tif)

    LOGGER.info(
        "Wrote %s from %d XYZ tile(s), z=%d, x=%d..%d, y=%d..%d, bands=%d, size=%dx%d.",
        output_tif,
        len(tiles),
        zoom_level,
        min_x,
        max_x,
        min_y,
        max_y,
        band_count,
        mosaic_width,
        mosaic_height,
    )
    return output_tif


def load_aoi_bounds_wgs84(aoi_path: Path) -> tuple[float, float, float, float]:
    import geopandas as gpd
    from shapely import wkt

    aoi_path = Path(aoi_path).expanduser().resolve()
    if not aoi_path.exists():
        raise FileNotFoundError(f"AOI file not found: {aoi_path}")
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
        if aoi.crs is None:
            aoi = aoi.set_crs("EPSG:4326")
    return tuple(float(value) for value in aoi.to_crs("EPSG:4326").total_bounds)


def write_tiny_aoi(
    *,
    aoi_path: Path,
    output_path: Path,
    size_meters: float,
) -> Path:
    import geopandas as gpd
    from shapely.geometry import box
    from shapely import wkt

    if size_meters <= 0:
        raise ValueError("--size-meters must be positive.")

    aoi_path = Path(aoi_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    if not aoi_path.exists():
        raise FileNotFoundError(f"AOI file not found: {aoi_path}")
    if aoi_path.is_dir():
        raise IsADirectoryError(f"AOI path points to a directory, not a vector file: {aoi_path}")

    suffix = aoi_path.suffix.lower()
    if suffix in {".wkt", ".txt"}:
        geom = wkt.loads(aoi_path.read_text(encoding="utf-8").strip())
        aoi = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")
    else:
        aoi = gpd.read_file(aoi_path)
        if aoi.crs is None:
            aoi = aoi.set_crs("EPSG:4326")

    aoi_3857 = aoi.to_crs("EPSG:3857")
    if hasattr(aoi_3857.geometry, "union_all"):
        geometry = aoi_3857.geometry.union_all()
    else:
        geometry = aoi_3857.geometry.unary_union
    center = geometry.centroid
    half = size_meters / 2
    tiny = gpd.GeoDataFrame(
        {"name": [f"tiny_{int(size_meters)}m"]},
        geometry=[box(center.x - half, center.y - half, center.x + half, center.y + half)],
        crs="EPSG:3857",
    ).to_crs("EPSG:4326")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tiny.to_file(output_path, driver="GeoJSON")
    LOGGER.info("Wrote tiny AOI %.1fm x %.1fm: %s", size_meters, size_meters, output_path)
    return output_path


def resolve_requests_verify(no_verify_ssl: bool = False, ca_bundle: Path | None = None) -> bool | str:
    if no_verify_ssl:
        return False
    if ca_bundle is None:
        return True
    ca_bundle = Path(ca_bundle).expanduser().resolve()
    if not ca_bundle.exists():
        raise FileNotFoundError(f"CA bundle not found: {ca_bundle}")
    return str(ca_bundle)


def download_xyz_tiles(
    *,
    url_template: str,
    output_root: Path,
    bounds_wgs84: tuple[float, float, float, float],
    zoom: int,
    provider: str | None = None,
    extension: str = "png",
    timeout: int = 60,
    sleep_seconds: float = 0.0,
    overwrite: bool = False,
    user_agent: str = "field-delineation-pipeline/1.0",
    max_tiles: int | None = None,
    verify_ssl: bool | str = True,
) -> list[Path]:
    """Download XYZ tiles for WGS84 bounds from a URL template."""

    import requests

    x_range, y_range = xyz_range_for_bounds(bounds_wgs84, zoom)
    planned = [(x, y) for x in x_range for y in y_range]
    if max_tiles is not None and len(planned) > max_tiles:
        raise RuntimeError(
            f"Refusing to download {len(planned)} XYZ tiles because --xyz-max-download-tiles={max_tiles}. "
            "Increase the limit after checking provider terms and expected storage."
        )

    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    extension = extension.lstrip(".")
    session = requests.Session()
    session.headers.update({"User-Agent": user_agent})
    if verify_ssl is False:
        try:
            import urllib3
            from urllib3.exceptions import InsecureRequestWarning

            urllib3.disable_warnings(InsecureRequestWarning)
        except Exception:
            pass
        LOGGER.warning("SSL certificate verification is disabled for XYZ tile downloads.")

    downloaded: list[Path] = []
    manifest_rows: list[dict[str, Any]] = []
    for index, (x, y) in enumerate(planned, start=1):
        url = url_template.format(z=zoom, x=x, y=y)
        output_path = output_root / str(zoom) / str(x) / f"{y}.{extension}"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists() and not overwrite:
            LOGGER.info("Reusing XYZ tile %d/%d: %s", index, len(planned), output_path)
            downloaded.append(output_path)
            manifest_rows.append({"z": zoom, "x": x, "y": y, "path": str(output_path), "url": url, "status": "reused"})
            continue

        LOGGER.info("Downloading XYZ tile %d/%d: z=%d x=%d y=%d", index, len(planned), zoom, x, y)
        with session.get(url, stream=True, timeout=timeout, verify=verify_ssl) as response:
            response.raise_for_status()
            with output_path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 256):
                    if chunk:
                        handle.write(chunk)
        downloaded.append(output_path)
        manifest_rows.append({"z": zoom, "x": x, "y": y, "path": str(output_path), "url": url, "status": "downloaded"})
        if sleep_seconds > 0 and index < len(planned):
            time.sleep(sleep_seconds)

    with (output_root / "download_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["z", "x", "y", "path", "url", "status"])
        writer.writeheader()
        writer.writerows(manifest_rows)
    with (output_root / "download_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "provider": provider,
                "attribution": xyz_provider_attribution(provider),
                "url_template": url_template,
                "verify_ssl": verify_ssl,
                "tiles": manifest_rows,
            },
            handle,
            indent=2,
        )

    return downloaded


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download and convert XYZ basemap image tiles.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert = subparsers.add_parser("convert", help="Mosaic local XYZ tiles into one EPSG:3857 GeoTIFF.")
    convert.add_argument("--tiles-root", required=True, type=Path)
    convert.add_argument("--output", required=True, type=Path)
    convert.add_argument("--zoom", default=None, type=int)
    convert.add_argument("--overwrite", action="store_true")

    download = subparsers.add_parser("download", help="Download XYZ tiles for an AOI.")
    download.add_argument("--provider", default=None, choices=sorted(XYZ_PROVIDER_SPECS), help="Built-in free/open imagery provider preset.")
    download.add_argument("--url-template", default=None, help="Example: https://server/{z}/{x}/{y}.png")
    download.add_argument("--aoi", required=True, type=Path)
    download.add_argument("--zoom", required=True, type=int)
    download.add_argument("--output-root", required=True, type=Path)
    download.add_argument("--extension", default="png")
    download.add_argument("--timeout", default=60, type=int)
    download.add_argument("--sleep-seconds", default=0.0, type=float)
    download.add_argument("--overwrite", action="store_true")
    download.add_argument("--user-agent", default="field-delineation-pipeline/1.0")
    download.add_argument("--max-tiles", default=5000, type=int)
    download.add_argument("--ca-bundle", default=None, type=Path, help="Optional PEM CA bundle for corporate TLS interception.")
    download.add_argument("--no-verify-ssl", action="store_true", help="Disable SSL verification for tile downloads. Use only on trusted networks.")
    download.add_argument("--convert-output", default=None, type=Path, help="Optional GeoTIFF to create after download.")

    tiny_aoi = subparsers.add_parser("tiny-aoi", help="Create a small centered square AOI for basemap smoke tests.")
    tiny_aoi.add_argument("--aoi", required=True, type=Path)
    tiny_aoi.add_argument("--output", required=True, type=Path)
    tiny_aoi.add_argument("--size-meters", default=500.0, type=float, help="Square side length in meters.")

    providers = subparsers.add_parser("providers", help="List built-in XYZ provider presets.")
    providers.set_defaults(command="providers")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "convert":
        convert_xyz_tiles_to_geotiff(
            tiles_root=args.tiles_root,
            output_tif=args.output,
            zoom=args.zoom,
            overwrite=args.overwrite,
        )
        return

    if args.command == "providers":
        for provider, spec in sorted(XYZ_PROVIDER_SPECS.items()):
            print(f"{provider}")
            print(f"  url_template: {spec['url_template']}")
            print(f"  attribution: {spec['attribution']}")
            print(f"  notes: {spec['notes']}")
        return

    if args.command == "tiny-aoi":
        output = write_tiny_aoi(
            aoi_path=args.aoi,
            output_path=args.output,
            size_meters=args.size_meters,
        )
        print(output)
        return

    bounds = load_aoi_bounds_wgs84(args.aoi)
    url_template = resolve_xyz_url_template(args.provider, args.url_template)
    if url_template is None:
        raise ValueError("Download mode requires either --provider or --url-template.")
    download_xyz_tiles(
        url_template=url_template,
        output_root=args.output_root,
        bounds_wgs84=bounds,
        zoom=args.zoom,
        provider=args.provider,
        extension=args.extension,
        timeout=args.timeout,
        sleep_seconds=args.sleep_seconds,
        overwrite=args.overwrite,
        user_agent=args.user_agent,
        max_tiles=args.max_tiles,
        verify_ssl=resolve_requests_verify(args.no_verify_ssl, args.ca_bundle),
    )
    if args.convert_output:
        convert_xyz_tiles_to_geotiff(
            tiles_root=args.output_root,
            output_tif=args.convert_output,
            zoom=args.zoom,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
