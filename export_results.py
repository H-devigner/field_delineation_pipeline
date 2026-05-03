#!/usr/bin/env python3
"""Export delineation GeoPackages to web/vector formats and quicklook images."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any

import geopandas as gpd


LOGGER = logging.getLogger("field_delineation_exports")


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def safe_layer_name(path: Path, layer: str | None) -> str:
    if layer:
        return f"{path.stem}_{layer}"
    return path.stem


def read_gpkg(gpkg_path: Path, layer: str | None = None, assumed_epsg: int | None = None) -> gpd.GeoDataFrame:
    try:
        gdf = gpd.read_file(gpkg_path, layer=layer)
    except Exception as exc:
        raise RuntimeError(f"Error reading {gpkg_path}: {exc}") from exc

    if gdf.empty:
        raise ValueError(f"{gpkg_path} is empty.")

    if gdf.crs is None:
        if assumed_epsg is None:
            raise ValueError(f"{gpkg_path} has no CRS and no assumed EPSG was provided.")
        LOGGER.warning("%s CRS missing, assuming EPSG:%s", gpkg_path, assumed_epsg)
        gdf = gdf.set_crs(epsg=assumed_epsg)

    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    if gdf.empty:
        raise ValueError(f"{gpkg_path} has no non-empty geometries.")
    return gdf


def add_area_columns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = gdf.copy()
    try:
        area = out.to_crs("EPSG:6933").geometry.area
        out["area_m2_calc"] = area
        out["area_ha_calc"] = area / 10000.0
    except Exception as exc:
        LOGGER.warning("Could not calculate equal-area fields: %s", exc)
    return out


def export_geojson(gdf: gpd.GeoDataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    add_area_columns(gdf).to_crs("EPSG:4326").to_file(output_path, driver="GeoJSON")
    LOGGER.info("Saved GeoJSON: %s", output_path)
    return output_path


def export_kml(gdf: gpd.GeoDataFrame, output_path: Path) -> Path | None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        add_area_columns(gdf).to_crs("EPSG:4326").to_file(output_path, driver="KML")
    except Exception as exc:
        LOGGER.warning("Could not save KML %s: %s", output_path, exc)
        return None
    LOGGER.info("Saved KML: %s", output_path)
    return output_path


def plot_boundaries(gdf: gpd.GeoDataFrame, output_path: Path, title: str | None = None, dpi: int = 220) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 10))
    gdf.plot(ax=ax, edgecolor="#111111", facecolor="#52b78822", linewidth=0.35)
    gdf.boundary.plot(ax=ax, color="#111111", linewidth=0.3)
    if title:
        ax.set_title(title, fontsize=11)
    ax.set_axis_off()
    ax.set_aspect("equal")
    fig.savefig(output_path, bbox_inches="tight", dpi=dpi, facecolor="white")
    plt.close(fig)
    LOGGER.info("Saved boundary PNG: %s", output_path)
    return output_path


def contrast_stretch(rgb: Any) -> Any:
    import numpy as np

    arr = rgb.astype("float32", copy=False)
    out = np.zeros_like(arr, dtype="float32")
    for idx in range(arr.shape[0]):
        band = arr[idx]
        valid = band[np.isfinite(band)]
        valid = valid[valid > 0]
        if valid.size == 0:
            continue
        lo, hi = np.percentile(valid, [2, 98])
        if hi <= lo:
            hi = lo + 1
        out[idx] = np.clip((band - lo) / (hi - lo), 0, 1)
    return out


def plot_overlay(
    gdf: gpd.GeoDataFrame,
    raster_path: Path,
    output_path: Path,
    title: str | None = None,
    dpi: int = 220,
    max_pixels: int = 1800,
) -> Path | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import rasterio
    from rasterio.plot import plotting_extent

    if raster_path is None or not raster_path.exists():
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(raster_path) as src:
        if src.count < 3:
            LOGGER.warning("Skipping overlay, raster has fewer than 3 bands: %s", raster_path)
            return None
        scale = max(src.width / max_pixels, src.height / max_pixels, 1)
        out_width = max(1, int(src.width / scale))
        out_height = max(1, int(src.height / scale))
        rgb = src.read([1, 2, 3], out_shape=(3, out_height, out_width), masked=True).filled(0)
        transform = src.transform * src.transform.scale(src.width / out_width, src.height / out_height)
        extent = plotting_extent(np.moveaxis(rgb, 0, -1), transform)
        plot_gdf = gdf.to_crs(src.crs)

    rgb = np.moveaxis(contrast_stretch(rgb), 0, -1)
    fig, ax = plt.subplots(figsize=(11, 11))
    ax.imshow(rgb, extent=extent)
    plot_gdf.boundary.plot(ax=ax, color="#ff2d2d", linewidth=0.45)
    if title:
        ax.set_title(title, fontsize=11)
    ax.set_axis_off()
    ax.set_aspect("equal")
    fig.savefig(output_path, bbox_inches="tight", dpi=dpi, facecolor="white")
    plt.close(fig)
    LOGGER.info("Saved overlay PNG: %s", output_path)
    return output_path


def summarize_gdf(gdf: gpd.GeoDataFrame, gpkg_path: Path, layer: str | None) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "source": str(gpkg_path),
        "layer": layer,
        "crs": str(gdf.crs),
        "feature_count": int(len(gdf)),
        "bounds": [float(value) for value in gdf.total_bounds],
    }
    try:
        area_ha = gdf.to_crs("EPSG:6933").geometry.area / 10000.0
        summary["total_area_ha"] = float(area_ha.sum())
        summary["median_area_ha"] = float(area_ha.median())
    except Exception as exc:
        summary["area_error"] = str(exc)
    return summary


def export_gpkg(
    gpkg_path: Path,
    *,
    outdir: Path,
    layer: str | None = None,
    assumed_epsg: int | None = None,
    raster_path: Path | None = None,
    formats: set[str] | None = None,
    dpi: int = 220,
    max_pixels: int = 1800,
) -> dict[str, Any]:
    if formats is None:
        formats = {"geojson", "kml", "png"}

    gpkg_path = Path(gpkg_path)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    base = safe_layer_name(gpkg_path, layer)
    gdf = read_gpkg(gpkg_path, layer=layer, assumed_epsg=assumed_epsg)

    outputs: dict[str, Any] = summarize_gdf(gdf, gpkg_path, layer)
    outputs["exports"] = {}

    if "geojson" in formats:
        outputs["exports"]["geojson"] = str(export_geojson(gdf, outdir / f"{base}.geojson"))
    if "kml" in formats:
        kml_path = export_kml(gdf, outdir / f"{base}.kml")
        if kml_path is not None:
            outputs["exports"]["kml"] = str(kml_path)
    if "png" in formats:
        outputs["exports"]["png_boundaries"] = str(
            plot_boundaries(gdf, outdir / f"{base}.boundaries.png", title=base, dpi=dpi)
        )
        if raster_path:
            overlay_path = plot_overlay(
                gdf,
                raster_path=Path(raster_path),
                output_path=outdir / f"{base}.sr_overlay.png",
                title=f"{base} on SR",
                dpi=dpi,
                max_pixels=max_pixels,
            )
            if overlay_path is not None:
                outputs["exports"]["png_sr_overlay"] = str(overlay_path)

    summary_path = outdir / f"{base}.summary.json"
    summary_path.write_text(json.dumps(outputs, indent=2), encoding="utf-8")
    outputs["exports"]["summary_json"] = str(summary_path)
    return outputs


def write_summary_csv(rows: list[dict[str, Any]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["source", "layer", "crs", "feature_count", "total_area_ha", "median_area_ha"]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return output_path


def write_gallery(rows: list[dict[str, Any]], output_path: Path) -> Path:
    cards = []
    for row in rows:
        exports = row.get("exports", {})
        title = Path(row["source"]).stem
        image = exports.get("png_sr_overlay") or exports.get("png_boundaries")
        links = []
        for label, key in [("GeoJSON", "geojson"), ("KML", "kml"), ("Summary", "summary_json")]:
            path = exports.get(key)
            if path:
                rel = os.path.relpath(path, output_path.parent)
                links.append(f'<a href="{rel}">{label}</a>')
        image_html = ""
        if image:
            rel_img = os.path.relpath(image, output_path.parent)
            image_html = f'<img src="{rel_img}" alt="{title} quicklook">'
        cards.append(
            "\n".join(
                [
                    '<section class="card">',
                    f"<h2>{title}</h2>",
                    image_html,
                    f"<p>{row.get('feature_count', 0):,} features"
                    + (
                        f" | {row.get('total_area_ha', 0):,.1f} ha"
                        if "total_area_ha" in row
                        else ""
                    )
                    + "</p>",
                    f"<p>{' | '.join(links)}</p>",
                    "</section>",
                ]
            )
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Delineation Exports</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; background: #f7f7f4; color: #1f2933; }}
    h1 {{ font-size: 24px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 18px; }}
    .card {{ background: white; border: 1px solid #d8ded8; border-radius: 8px; padding: 14px; }}
    .card h2 {{ font-size: 16px; margin: 0 0 10px; }}
    img {{ width: 100%; height: auto; border: 1px solid #e5e7eb; }}
    a {{ color: #0f766e; }}
  </style>
</head>
<body>
  <h1>Delineation Exports</h1>
  <div class="grid">
    {''.join(cards)}
  </div>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")
    return output_path


def expand_inputs(inputs: list[str]) -> list[Path]:
    gpkg_files: set[Path] = set()
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            gpkg_files.update(path.glob("*.gpkg"))
        else:
            gpkg_files.update(Path(match) for match in glob.glob(item))
    return sorted(gpkg_files)


def process_one(args: tuple[Path, Path, str | None, int | None, Path | None, set[str], int, int]) -> dict[str, Any]:
    gpkg_path, outdir, layer, assumed_epsg, raster_path, formats, dpi, max_pixels = args
    return export_gpkg(
        gpkg_path,
        outdir=outdir,
        layer=layer,
        assumed_epsg=assumed_epsg,
        raster_path=raster_path,
        formats=formats,
        dpi=dpi,
        max_pixels=max_pixels,
    )


def process_all(
    gpkg_files: list[Path],
    *,
    outdir: Path,
    layer: str | None,
    assumed_epsg: int | None,
    raster_path: Path | None,
    formats: set[str],
    parallel: bool,
    dpi: int,
    max_pixels: int,
) -> list[dict[str, Any]]:
    if not gpkg_files:
        raise FileNotFoundError("No GPKG files found.")

    tasks = [
        (
            gpkg,
            outdir / gpkg.stem,
            layer,
            assumed_epsg,
            raster_path,
            formats,
            dpi,
            max_pixels,
        )
        for gpkg in gpkg_files
    ]

    rows: list[dict[str, Any]] = []
    if parallel and len(tasks) > 1:
        workers = min(cpu_count(), len(tasks))
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(process_one, task) for task in tasks]
            for future in as_completed(futures):
                rows.append(future.result())
    else:
        for task in tasks:
            rows.append(process_one(task))

    write_summary_csv(rows, outdir / "exports_summary.csv")
    write_gallery(rows, outdir / "index.html")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export delineation GeoPackages.")
    parser.add_argument("inputs", nargs="*", help="GPKG files, folders, or glob patterns. Also accepts stdin.")
    parser.add_argument("--layer", default=os.environ.get("GPKG_LAYER"))
    parser.add_argument("--outdir", default=os.environ.get("GPKG_OUTDIR"), type=Path)
    parser.add_argument("--assumed-epsg", default=None, type=int)
    parser.add_argument("--raster", default=None, type=Path, help="Optional raster backdrop for overlay PNGs.")
    parser.add_argument("--formats", default="geojson,kml,png")
    parser.add_argument("--dpi", default=220, type=int)
    parser.add_argument("--max-pixels", default=1800, type=int)
    parser.add_argument("--no-parallel", action="store_true")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    inputs = list(args.inputs)
    if not sys.stdin.isatty():
        inputs.extend(line.strip() for line in sys.stdin if line.strip())
    if not inputs:
        raise SystemExit(
            "Usage: python export_results.py <gpkg | folder | glob>\n"
            "       find . -name '*.gpkg' | python export_results.py"
        )
    outdir = args.outdir or Path("exports")
    formats = {item.strip().lower() for item in args.formats.split(",") if item.strip()}
    rows = process_all(
        expand_inputs(inputs),
        outdir=outdir,
        layer=args.layer,
        assumed_epsg=args.assumed_epsg,
        raster_path=args.raster,
        formats=formats,
        parallel=not args.no_parallel,
        dpi=args.dpi,
        max_pixels=args.max_pixels,
    )
    LOGGER.info("Exported %d GeoPackage(s) to %s", len(rows), outdir)


if __name__ == "__main__":
    main()
