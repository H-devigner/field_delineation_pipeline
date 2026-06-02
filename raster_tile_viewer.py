#!/usr/bin/env python3
"""Build and serve a browser viewer for large GeoTIFF imagery.

The build step indexes one GeoTIFF or a folder of GeoTIFFs and writes a small
viewer package. The serve step exposes dynamic XYZ PNG tiles, reading only the
windows needed for the current map view.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import math
import re
import threading
import urllib.parse
from dataclasses import dataclass
from functools import partial
from html import escape
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger("raster_tile_viewer")
WEB_MERCATOR_LIMIT = 20037508.342789244
WEB_MERCATOR_INITIAL_RESOLUTION = 156543.03392804097
EMPTY_TILE_CACHE: dict[int, bytes] = {}


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>__TITLE__</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>
    html, body, #map {
      height: 100%;
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: #141414;
    }
    .panel {
      position: absolute;
      top: 12px;
      left: 12px;
      z-index: 1000;
      width: min(380px, calc(100vw - 24px));
      background: rgba(255, 255, 255, 0.94);
      border: 1px solid rgba(0, 0, 0, 0.14);
      border-radius: 8px;
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.18);
      padding: 12px;
    }
    .title {
      font-size: 14px;
      font-weight: 700;
      margin-bottom: 8px;
      overflow-wrap: anywhere;
    }
    .meta {
      font-size: 12px;
      color: #424242;
      line-height: 1.35;
      margin-bottom: 10px;
    }
    .row {
      display: grid;
      grid-template-columns: 92px 1fr;
      gap: 10px;
      align-items: center;
      font-size: 12px;
      margin-top: 8px;
    }
    select, input[type="range"] {
      width: 100%;
      min-width: 0;
    }
    select {
      border: 1px solid rgba(0, 0, 0, 0.28);
      border-radius: 6px;
      padding: 6px 7px;
      background: white;
      font: inherit;
    }
    button {
      border: 1px solid #1f2937;
      background: #1f2937;
      color: white;
      border-radius: 6px;
      padding: 7px 10px;
      cursor: pointer;
      font-size: 12px;
      margin-right: 6px;
    }
    button.secondary {
      background: white;
      color: #1f2937;
    }
    .status {
      margin-top: 8px;
      font-size: 11px;
      color: #555;
      line-height: 1.45;
      overflow-wrap: anywhere;
    }
    .coords {
      position: absolute;
      right: 12px;
      bottom: 18px;
      z-index: 1000;
      padding: 5px 7px;
      border-radius: 6px;
      background: rgba(255, 255, 255, 0.88);
      border: 1px solid rgba(0, 0, 0, 0.14);
      font: 11px/1.2 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      color: #222;
    }
    .leaflet-container {
      background: #111;
    }
  </style>
</head>
<body>
  <div id="map"></div>
  <div class="panel">
    <div class="title">__TITLE__</div>
    <div class="meta">
      <span id="count"></span><br>
      Zoom <span id="zoom-meta"></span>
    </div>
    <div class="row">
      <label for="layer">Layer</label>
      <select id="layer"></select>
    </div>
    <div class="row">
      <label for="opacity">Opacity</label>
      <input id="opacity" type="range" min="0" max="1" value="1" step="0.02" />
    </div>
    <button id="fit">Fit</button>
    <button id="open-config" class="secondary">Config</button>
    <div id="details" class="status"></div>
  </div>
  <div id="coords" class="coords"></div>
  <script>
    const config = __CONFIG_JSON__;
    const rastersById = new Map(config.rasters.map((item) => [item.id, item]));
    const mapBounds = [[config.bounds[1], config.bounds[0]], [config.bounds[3], config.bounds[2]]];
    const maxZoom = Math.min(24, Number(config.maxzoom || 18) + 2);

    const map = L.map("map", {
      preferCanvas: true,
      zoomControl: true,
      maxZoom
    });
    L.control.scale({ metric: true, imperial: false }).addTo(map);

    let tileLayer = L.tileLayer("/tiles/mosaic/{z}/{x}/{y}.png", {
      tileSize: config.tileSize || 256,
      minZoom: config.minzoom || 0,
      maxZoom,
      maxNativeZoom: maxZoom,
      noWrap: true,
      attribution: "GeoTIFF imagery"
    }).addTo(map);

    function layerBounds(layerId) {
      if (layerId === "mosaic") return mapBounds;
      const raster = rastersById.get(layerId);
      return raster ? [[raster.bounds4326[1], raster.bounds4326[0]], [raster.bounds4326[3], raster.bounds4326[2]]] : mapBounds;
    }

    function selectedLayerId() {
      return document.getElementById("layer").value || "mosaic";
    }

    function updateDetails(layerId) {
      const details = document.getElementById("details");
      if (layerId === "mosaic") {
        details.textContent = `Mosaic layer from ${config.rasters.length} raster(s).`;
        return;
      }
      const raster = rastersById.get(layerId);
      if (!raster) {
        details.textContent = "";
        return;
      }
      const overviewText = raster.hasOverviews ? "overviews yes" : "overviews no";
      const resText = raster.resolution3857
        ? `res ${raster.resolution3857[0].toFixed(2)} x ${raster.resolution3857[1].toFixed(2)} m`
        : "res unknown";
      details.textContent = `${raster.name} | ${raster.width} x ${raster.height} | ${raster.crs || "no CRS"} | ${resText} | ${overviewText}`;
    }

    function setTileUrl(layerId, fit) {
      tileLayer.setUrl(`/tiles/${encodeURIComponent(layerId)}/{z}/{x}/{y}.png`, false);
      tileLayer.redraw();
      updateDetails(layerId);
      if (fit) map.fitBounds(layerBounds(layerId), { padding: [30, 30] });
    }

    const select = document.getElementById("layer");
    const mosaicOption = document.createElement("option");
    mosaicOption.value = "mosaic";
    mosaicOption.textContent = "Mosaic: all rasters";
    select.appendChild(mosaicOption);
    for (const raster of config.rasters) {
      const option = document.createElement("option");
      option.value = raster.id;
      option.textContent = raster.name;
      select.appendChild(option);
    }
    select.addEventListener("change", () => setTileUrl(selectedLayerId(), true));

    document.getElementById("opacity").addEventListener("input", (event) => {
      tileLayer.setOpacity(Number(event.target.value));
    });
    document.getElementById("fit").addEventListener("click", () => {
      map.fitBounds(layerBounds(selectedLayerId()), { padding: [30, 30] });
    });
    document.getElementById("open-config").addEventListener("click", () => {
      window.open("/viewer_config.json", "_blank");
    });

    document.getElementById("count").textContent = `${config.rasters.length} raster(s)`;
    document.getElementById("zoom-meta").textContent = `${config.minzoom || 0}-${maxZoom}`;
    updateDetails("mosaic");
    map.fitBounds(mapBounds, { padding: [30, 30] });

    map.on("mousemove", (event) => {
      document.getElementById("coords").textContent =
        `${event.latlng.lat.toFixed(6)}, ${event.latlng.lng.toFixed(6)} | z ${map.getZoom().toFixed(2)}`;
    });
  </script>
</body>
</html>
"""


@dataclass
class RasterEntry:
    id: str
    path: Path
    bands: list[int]
    stats: list[dict[str, float]]
    bounds3857: list[float]


class DatasetCache:
    def __init__(self) -> None:
        self._items: dict[str, tuple[Any, threading.Lock]] = {}
        self._lock = threading.Lock()

    def get(self, entry: RasterEntry) -> tuple[Any, threading.Lock]:
        key = str(entry.path)
        with self._lock:
            if key not in self._items:
                import rasterio

                self._items[key] = (rasterio.open(entry.path), threading.Lock())
            return self._items[key]

    def close(self) -> None:
        with self._lock:
            for dataset, _ in self._items.values():
                with contextlib.suppress(Exception):
                    dataset.close()
            self._items.clear()


class TileServerState:
    def __init__(self, viewer_dir: Path) -> None:
        config_path = viewer_dir / "viewer_config.json"
        self.viewer_dir = viewer_dir
        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        self.tile_size = int(self.config.get("tileSize", 256))
        self.cache = DatasetCache()
        self.entries = [
            RasterEntry(
                id=item["id"],
                path=Path(item["path"]),
                bands=[int(band) for band in item["bands"]],
                stats=item["stats"],
                bounds3857=[float(value) for value in item["bounds3857"]],
            )
            for item in self.config.get("rasters", [])
        ]
        self.by_id = {entry.id: entry for entry in self.entries}

    def close(self) -> None:
        self.cache.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Index GeoTIFF imagery and create a viewer package.")
    build.add_argument("--input", required=True, type=Path, help="GeoTIFF file or directory of GeoTIFFs.")
    build.add_argument("--output-dir", required=True, type=Path, help="Viewer package output directory.")
    build.add_argument("--name", default=None, help="Viewer title. Defaults to the input folder/file stem.")
    build.add_argument("--bands", default="1,2,3", help="1-based bands to display as RGB. A single band is shown as grayscale.")
    build.add_argument("--minzoom", default=0, type=int)
    build.add_argument("--maxzoom", default=None, type=int, help="Defaults to an estimate from source resolution.")
    build.add_argument("--tile-size", default=256, type=int)
    build.add_argument("--percentiles", default="2,98", help="Low,high percentiles for display stretch.")
    build.add_argument("--max-sample-pixels", default=1_000_000, type=int)
    build.add_argument("--assume-epsg", default=None, type=int, help="Fallback EPSG for rasters without CRS.")
    build.add_argument("--no-recursive", action="store_true", help="Do not scan input directories recursively.")
    build.add_argument("--verbose", action="store_true")

    serve = subparsers.add_parser("serve", help="Serve an existing viewer package.")
    serve.add_argument("--viewer-dir", required=True, type=Path)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", default=8090, type=int)
    serve.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_int_csv(value: str) -> list[int]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("Expected at least one band index.")
    return [int(item) for item in items]


def parse_float_pair(value: str) -> tuple[float, float]:
    parts = [float(item.strip()) for item in value.split(",") if item.strip()]
    if len(parts) != 2 or parts[0] >= parts[1]:
        raise ValueError("Expected two increasing values, for example 2,98.")
    return parts[0], parts[1]


def safe_id(value: str, existing: set[str]) -> str:
    base = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-") or "raster"
    candidate = base
    suffix = 2
    while candidate in existing:
        candidate = f"{base}_{suffix}"
        suffix += 1
    existing.add(candidate)
    return candidate


def find_rasters(input_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in {".tif", ".tiff"}:
            raise ValueError(f"Input file is not a GeoTIFF: {input_path}")
        return [input_path.resolve()]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    pattern = "**/*" if recursive else "*"
    rasters = sorted(
        path.resolve()
        for path in input_path.glob(pattern)
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
    )
    if not rasters:
        raise FileNotFoundError(f"No .tif/.tiff files found under {input_path}")
    return rasters


def transform_bounds(bounds: tuple[float, float, float, float], src_crs: Any, dst_crs: str) -> list[float]:
    from rasterio.warp import transform_bounds as rio_transform_bounds

    return list(rio_transform_bounds(src_crs, dst_crs, *bounds, densify_pts=21))


def estimate_maxzoom(bounds3857: list[float], width: int, height: int) -> int:
    if width <= 0 or height <= 0:
        return 18
    xres = abs(bounds3857[2] - bounds3857[0]) / width
    yres = abs(bounds3857[3] - bounds3857[1]) / height
    resolution = max(min(xres, yres), 0.001)
    zoom = math.ceil(math.log2(WEB_MERCATOR_INITIAL_RESOLUTION / resolution)) + 1
    return max(0, min(24, int(zoom)))


def sample_band_stats(src: Any, bands: list[int], percentiles: tuple[float, float], max_sample_pixels: int) -> list[dict[str, float]]:
    import numpy as np

    sample_scale = min(1.0, math.sqrt(max_sample_pixels / max(1, src.width * src.height)))
    out_width = max(1, int(src.width * sample_scale))
    out_height = max(1, int(src.height * sample_scale))
    stats = []
    for band in bands:
        data = src.read(band, out_shape=(out_height, out_width), masked=True)
        values = data.compressed() if np.ma.isMaskedArray(data) else data.reshape(-1)
        values = values[np.isfinite(values)]
        if values.size == 0:
            lo, hi = 0.0, 1.0
        else:
            lo, hi = np.percentile(values, percentiles)
            if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                lo = float(np.nanmin(values))
                hi = float(np.nanmax(values))
            if lo == hi:
                hi = lo + 1.0
        stats.append({"band": int(band), "min": float(lo), "max": float(hi)})
    return stats


def build_viewer(args: argparse.Namespace) -> None:
    import rasterio
    from rasterio.crs import CRS

    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    requested_bands = parse_int_csv(args.bands)
    percentiles = parse_float_pair(args.percentiles)
    raster_paths = find_rasters(input_path, recursive=not args.no_recursive)
    title = args.name or input_path.stem
    existing_ids: set[str] = set()
    rasters = []
    union_bounds4326: list[float] | None = None
    inferred_maxzoom = args.minzoom

    for path in raster_paths:
        LOGGER.info("Indexing %s", path)
        with rasterio.open(path) as src:
            crs = src.crs
            if crs is None and args.assume_epsg is not None:
                crs = CRS.from_epsg(args.assume_epsg)
                LOGGER.warning("Assuming EPSG:%s for %s", args.assume_epsg, path)
            if crs is None:
                raise ValueError(f"Raster has no CRS: {path}. Pass --assume-epsg if appropriate.")

            bands = [band for band in requested_bands if 1 <= band <= src.count]
            if not bands:
                raise ValueError(f"Requested bands {requested_bands} are not available in {path} with {src.count} band(s).")
            if len(bands) not in {1, 3}:
                raise ValueError("--bands must resolve to either one band or three bands for display.")

            bounds = tuple(src.bounds)
            bounds4326 = transform_bounds(bounds, crs, "EPSG:4326")
            bounds3857 = transform_bounds(bounds, crs, "EPSG:3857")
            stats = sample_band_stats(src, bands, percentiles, args.max_sample_pixels)
            overviews = {str(band): src.overviews(band) for band in bands}
            has_overviews = any(bool(values) for values in overviews.values())
            if not has_overviews:
                LOGGER.warning("No internal overviews found for %s. Deep zoom may be slower.", path)

            rel_name = str(path.relative_to(input_path)) if input_path.is_dir() and path.is_relative_to(input_path) else path.name
            raster_id = safe_id(rel_name.replace("/", "_"), existing_ids)
            resolution3857 = [
                abs(bounds3857[2] - bounds3857[0]) / max(1, src.width),
                abs(bounds3857[3] - bounds3857[1]) / max(1, src.height),
            ]
            inferred_maxzoom = max(inferred_maxzoom, estimate_maxzoom(bounds3857, src.width, src.height))

            if union_bounds4326 is None:
                union_bounds4326 = bounds4326
            else:
                union_bounds4326 = [
                    min(union_bounds4326[0], bounds4326[0]),
                    min(union_bounds4326[1], bounds4326[1]),
                    max(union_bounds4326[2], bounds4326[2]),
                    max(union_bounds4326[3], bounds4326[3]),
                ]

            rasters.append(
                {
                    "id": raster_id,
                    "name": rel_name,
                    "path": str(path),
                    "width": int(src.width),
                    "height": int(src.height),
                    "count": int(src.count),
                    "crs": str(crs),
                    "bounds4326": bounds4326,
                    "bounds3857": bounds3857,
                    "bands": bands,
                    "stats": stats,
                    "overviews": overviews,
                    "hasOverviews": has_overviews,
                    "resolution3857": resolution3857,
                    "nodata": src.nodata,
                    "dtype": src.dtypes[0] if src.dtypes else None,
                }
            )

    if union_bounds4326 is None:
        raise RuntimeError("No rasters were indexed.")

    maxzoom = args.maxzoom if args.maxzoom is not None else inferred_maxzoom
    config = {
        "name": title,
        "createdBy": "raster_tile_viewer.py",
        "tileSize": int(args.tile_size),
        "minzoom": int(args.minzoom),
        "maxzoom": int(maxzoom),
        "bounds": union_bounds4326,
        "center": [(union_bounds4326[0] + union_bounds4326[2]) / 2.0, (union_bounds4326[1] + union_bounds4326[3]) / 2.0],
        "rasters": rasters,
    }

    (output_dir / "viewer_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    html = HTML_TEMPLATE.replace("__TITLE__", escape(title)).replace("__CONFIG_JSON__", json.dumps(config))
    (output_dir / "index.html").write_text(html, encoding="utf-8")
    LOGGER.info("Viewer written to %s", output_dir)
    LOGGER.info("Serve with: python raster_tile_viewer.py serve --viewer-dir %s --host 0.0.0.0 --port 8090", output_dir)


def tile_bounds_3857(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    tiles = 2**z
    tile_span = 2 * WEB_MERCATOR_LIMIT / tiles
    minx = -WEB_MERCATOR_LIMIT + x * tile_span
    maxx = minx + tile_span
    maxy = WEB_MERCATOR_LIMIT - y * tile_span
    miny = maxy - tile_span
    return minx, miny, maxx, maxy


def bounds_intersect(a: list[float] | tuple[float, float, float, float], b: list[float] | tuple[float, float, float, float]) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def empty_tile(tile_size: int) -> bytes:
    from PIL import Image

    if tile_size not in EMPTY_TILE_CACHE:
        image = Image.new("RGBA", (tile_size, tile_size), (0, 0, 0, 0))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=True)
        EMPTY_TILE_CACHE[tile_size] = buffer.getvalue()
    return EMPTY_TILE_CACHE[tile_size]


def read_tile_rgba(state: TileServerState, entry: RasterEntry, z: int, x: int, y: int) -> Image.Image | None:
    import numpy as np
    from PIL import Image
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.windows import from_bounds
    from rasterio.warp import transform_bounds as rio_transform_bounds

    tile_bounds = tile_bounds_3857(z, x, y)
    if not bounds_intersect(entry.bounds3857, tile_bounds):
        return None

    dataset, lock = state.cache.get(entry)
    with lock:
        source_bounds = rio_transform_bounds("EPSG:3857", dataset.crs, *tile_bounds, densify_pts=21)
        window = from_bounds(*source_bounds, transform=dataset.transform)
        data = dataset.read(
            entry.bands,
            window=window,
            out_shape=(len(entry.bands), state.tile_size, state.tile_size),
            resampling=Resampling.bilinear,
            boundless=True,
            masked=True,
        )

    if data.size == 0:
        return None

    mask = np.ma.getmaskarray(data)
    if mask.ndim == 3:
        valid = ~np.all(mask, axis=0)
    else:
        valid = ~mask
    if not np.any(valid):
        return None

    filled = data.astype("float32").filled(np.nan)
    channels = []
    for idx, stats in enumerate(entry.stats):
        lo = float(stats["min"])
        hi = float(stats["max"])
        if hi <= lo:
            hi = lo + 1.0
        channel = np.clip((filled[idx] - lo) / (hi - lo), 0, 1) * 255.0
        channels.append(np.nan_to_num(channel, nan=0.0).astype("uint8"))

    if len(channels) == 1:
        rgb = np.stack([channels[0], channels[0], channels[0]], axis=-1)
    else:
        rgb = np.stack(channels[:3], axis=-1)

    alpha = (valid.astype("uint8") * 255).reshape(state.tile_size, state.tile_size, 1)
    rgba = np.concatenate([rgb, alpha], axis=-1)
    return Image.fromarray(rgba, "RGBA")


def render_tile(state: TileServerState, layer_id: str, z: int, x: int, y: int) -> bytes:
    from PIL import Image

    if layer_id == "mosaic":
        entries = [entry for entry in state.entries if bounds_intersect(entry.bounds3857, tile_bounds_3857(z, x, y))]
    else:
        entry = state.by_id.get(layer_id)
        entries = [entry] if entry else []

    if not entries:
        return empty_tile(state.tile_size)

    base = Image.new("RGBA", (state.tile_size, state.tile_size), (0, 0, 0, 0))
    wrote = False
    for entry in entries:
        image = read_tile_rgba(state, entry, z, x, y)
        if image is None:
            continue
        base.alpha_composite(image)
        wrote = True

    if not wrote:
        return empty_tile(state.tile_size)

    buffer = io.BytesIO()
    base.save(buffer, format="PNG")
    return buffer.getvalue()


class RasterTileHandler(SimpleHTTPRequestHandler):
    state: TileServerState

    def log_message(self, fmt: str, *args: Any) -> None:
        LOGGER.debug(fmt, *args)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        parts = [urllib.parse.unquote(part) for part in parsed.path.split("/") if part]
        if len(parts) == 5 and parts[0] == "tiles" and parts[4].endswith(".png"):
            try:
                layer_id = parts[1]
                z = int(parts[2])
                x = int(parts[3])
                y = int(parts[4][:-4])
                payload = render_tile(self.state, layer_id, z, x, y)
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Cache-Control", "public, max-age=3600")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            except Exception as exc:
                LOGGER.exception("Tile request failed: %s", parsed.path)
                self.send_error(500, str(exc))
            return

        return super().do_GET()


def serve_viewer(args: argparse.Namespace) -> None:
    viewer_dir = args.viewer_dir.expanduser().resolve()
    if not (viewer_dir / "viewer_config.json").exists():
        raise FileNotFoundError(f"Viewer config not found: {viewer_dir / 'viewer_config.json'}")

    state = TileServerState(viewer_dir)

    class Handler(RasterTileHandler):
        pass

    Handler.state = state
    handler = partial(Handler, directory=str(viewer_dir))
    server = ThreadingHTTPServer((args.host, args.port), handler)
    try:
        LOGGER.info("Serving raster viewer at http://%s:%s", args.host, args.port)
        LOGGER.info("Viewer directory: %s", viewer_dir)
        server.serve_forever()
    finally:
        state.close()
        server.server_close()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    if args.command == "build":
        build_viewer(args)
    elif args.command == "serve":
        serve_viewer(args)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
