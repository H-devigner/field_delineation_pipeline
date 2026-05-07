#!/usr/bin/env python3
"""Build and serve a browser viewer for large field polygon outputs.

The build step converts a GPKG/GeoJSON into vector tiles with tippecanoe and
creates a MapLibre viewer. The serve step exposes the MBTiles file through a
small local tile endpoint so the browser only loads the tiles it needs.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import urllib.parse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger("field_vector_tile_viewer")


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <link href="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.css" rel="stylesheet" />
  <script src="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.js"></script>
  <style>
    html, body, #map {{
      height: 100%;
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    .panel {{
      position: absolute;
      top: 12px;
      left: 12px;
      z-index: 2;
      width: min(360px, calc(100vw - 24px));
      background: rgba(255, 255, 255, 0.94);
      border: 1px solid rgba(0, 0, 0, 0.14);
      border-radius: 8px;
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.18);
      padding: 12px;
      color: #141414;
    }}
    .title {{
      font-size: 14px;
      font-weight: 700;
      margin-bottom: 8px;
    }}
    .meta {{
      font-size: 12px;
      color: #474747;
      line-height: 1.35;
      margin-bottom: 10px;
    }}
    .row {{
      display: grid;
      grid-template-columns: 92px 1fr;
      gap: 10px;
      align-items: center;
      font-size: 12px;
      margin-top: 8px;
    }}
    input[type="range"] {{
      width: 100%;
    }}
    button {{
      border: 1px solid #1f2937;
      background: #1f2937;
      color: white;
      border-radius: 6px;
      padding: 7px 10px;
      cursor: pointer;
      font-size: 12px;
      margin-right: 6px;
    }}
    button.secondary {{
      background: white;
      color: #1f2937;
    }}
    .maplibregl-popup-content {{
      max-width: 320px;
      font: 12px/1.4 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      white-space: pre-wrap;
    }}
  </style>
</head>
<body>
  <div id="map"></div>
  <div class="panel">
    <div class="title">{title}</div>
    <div class="meta">
      Vector tiles: zoom {minzoom}-{maxzoom}<br>
      Click a field to inspect available attributes.
    </div>
    <button id="fit">Fit Extent</button>
    <button id="toggle-fill" class="secondary">Toggle Fill</button>
    <div class="row">
      <label for="opacity">Fill Opacity</label>
      <input id="opacity" type="range" min="0" max="0.8" value="0.18" step="0.02" />
    </div>
    <div class="row">
      <label for="line-width">Line Width</label>
      <input id="line-width" type="range" min="0.1" max="2.5" value="0.8" step="0.1" />
    </div>
  </div>
  <script>
    const config = {config_json};
    const bounds = config.bounds;

    const map = new maplibregl.Map({{
      container: "map",
      style: {{
        version: 8,
        sources: {{
          basemap: {{
            type: "raster",
            tiles: [config.basemapUrl],
            tileSize: 256,
            attribution: config.basemapAttribution
          }},
          fields: {{
            type: "vector",
            tiles: [window.location.origin + "/tiles/{{z}}/{{x}}/{{y}}.pbf"],
            minzoom: config.minzoom,
            maxzoom: config.maxzoom
          }}
        }},
        layers: [
          {{
            id: "basemap",
            type: "raster",
            source: "basemap"
          }},
          {{
            id: "fields-fill",
            type: "fill",
            source: "fields",
            "source-layer": config.sourceLayer,
            paint: {{
              "fill-color": [
                "case",
                ["has", "tile_id"],
                ["rgb", 46, 125, 50],
                ["rgb", 14, 116, 144]
              ],
              "fill-opacity": 0.18
            }}
          }},
          {{
            id: "fields-line",
            type: "line",
            source: "fields",
            "source-layer": config.sourceLayer,
            paint: {{
              "line-color": "#111111",
              "line-opacity": 0.85,
              "line-width": [
                "interpolate",
                ["linear"],
                ["zoom"],
                8, 0.2,
                13, 0.7,
                16, 1.1
              ]
            }}
          }}
        ]
      }},
      center: config.center,
      zoom: config.initialZoom
    }});

    map.addControl(new maplibregl.NavigationControl({{ visualizePitch: true }}), "top-right");
    map.addControl(new maplibregl.ScaleControl({{ unit: "metric" }}), "bottom-left");

    map.on("load", () => {{
      map.fitBounds(bounds, {{ padding: 40, duration: 0 }});
    }});

    map.on("click", "fields-fill", (event) => {{
      const feature = event.features && event.features[0];
      if (!feature) return;
      const props = feature.properties || {{}};
      const text = Object.keys(props).length
        ? JSON.stringify(props, null, 2)
        : "No attributes were kept in the vector tiles.";
      new maplibregl.Popup()
        .setLngLat(event.lngLat)
        .setText(text)
        .addTo(map);
    }});

    map.on("mouseenter", "fields-fill", () => map.getCanvas().style.cursor = "pointer");
    map.on("mouseleave", "fields-fill", () => map.getCanvas().style.cursor = "");

    document.getElementById("fit").addEventListener("click", () => {{
      map.fitBounds(bounds, {{ padding: 40, duration: 350 }});
    }});

    document.getElementById("toggle-fill").addEventListener("click", () => {{
      const current = map.getLayoutProperty("fields-fill", "visibility");
      map.setLayoutProperty("fields-fill", "visibility", current === "none" ? "visible" : "none");
    }});

    document.getElementById("opacity").addEventListener("input", (event) => {{
      map.setPaintProperty("fields-fill", "fill-opacity", Number(event.target.value));
    }});

    document.getElementById("line-width").addEventListener("input", (event) => {{
      map.setPaintProperty("fields-line", "line-width", Number(event.target.value));
    }});
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Build MBTiles and the static viewer files.")
    build.add_argument("--input", required=True, type=Path, help="Input GPKG/GeoJSON field polygons.")
    build.add_argument("--output-dir", required=True, type=Path, help="Viewer package output directory.")
    build.add_argument("--name", default=None, help="Output package name. Defaults to input file stem.")
    build.add_argument("--input-layer", default=None, help="Input layer name. Defaults to the first layer.")
    build.add_argument("--tile-layer", default="fields", help="Vector tile source-layer name.")
    build.add_argument("--minzoom", default=5, type=int)
    build.add_argument("--maxzoom", default=16, type=int)
    build.add_argument(
        "--include-properties",
        default="merged_id,tile_id,area,source_count",
        help="Comma-separated properties to keep. Use '' to keep no attributes.",
    )
    build.add_argument(
        "--keep-all",
        action="store_true",
        help="Ask tippecanoe not to drop features. Produces larger tiles but preserves more detail.",
    )
    build.add_argument("--where", default=None, help="Optional OGR SQL WHERE filter.")
    build.add_argument(
        "--work-dir",
        default=None,
        type=Path,
        help="Directory for temporary GeoJSONSeq files. Defaults to <output-dir>/tmp.",
    )
    build.add_argument(
        "--basemap-url",
        default="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        help="XYZ raster basemap URL template.",
    )
    build.add_argument(
        "--basemap-attribution",
        default="OpenStreetMap contributors",
        help="Basemap attribution shown by MapLibre.",
    )
    build.add_argument("--keep-geojsonseq", action="store_true", help="Keep temporary GeoJSONSeq next to MBTiles.")
    build.add_argument("--verbose", action="store_true")

    serve = subparsers.add_parser("serve", help="Serve an existing viewer package.")
    serve.add_argument("--viewer-dir", required=True, type=Path, help="Directory created by the build command.")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", default=8088, type=int)
    serve.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def run(command: list[str], *, cwd: Path | None = None) -> None:
    LOGGER.info("Running: %s", " ".join(command))
    subprocess.run(command, cwd=str(cwd) if cwd else None, check=True)


def require_tool(name: str, install_hint: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(f"Required command not found: {name}\n{install_hint}")
    return path


def parse_property_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def get_layer_info(vector_path: Path, layer_name: str | None) -> dict[str, Any]:
    try:
        from osgeo import ogr, osr
    except ImportError as exc:
        raise ImportError("Install GDAL Python bindings in the active environment before building a viewer.") from exc

    ogr.UseExceptions()
    ds = ogr.Open(str(vector_path))
    if ds is None:
        raise FileNotFoundError(f"Could not open vector dataset: {vector_path}")

    layer = ds.GetLayerByName(layer_name) if layer_name else ds.GetLayer(0)
    if layer is None:
        available = [ds.GetLayer(i).GetName() for i in range(ds.GetLayerCount())]
        raise ValueError(f"Layer {layer_name!r} not found. Available layers: {available}")

    extent = layer.GetExtent()
    if extent is None:
        raise ValueError(f"Could not read extent from {vector_path}")
    minx, maxx, miny, maxy = extent

    source_srs = layer.GetSpatialRef()
    if source_srs is None:
        LOGGER.warning("Input layer has no CRS; assuming EPSG:4326.")
        source_srs = osr.SpatialReference()
        source_srs.ImportFromEPSG(4326)

    try:
        source_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    except AttributeError:
        pass

    target_srs = osr.SpatialReference()
    target_srs.ImportFromEPSG(4326)
    try:
        target_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    except AttributeError:
        pass

    transform = osr.CoordinateTransformation(source_srs, target_srs)
    corners = [
        transform.TransformPoint(minx, miny),
        transform.TransformPoint(minx, maxy),
        transform.TransformPoint(maxx, miny),
        transform.TransformPoint(maxx, maxy),
    ]
    lons = [point[0] for point in corners]
    lats = [point[1] for point in corners]
    bounds = [min(lons), min(lats), max(lons), max(lats)]
    center = [(bounds[0] + bounds[2]) / 2.0, (bounds[1] + bounds[3]) / 2.0]

    return {
        "input_layer": layer.GetName(),
        "feature_count": int(layer.GetFeatureCount()),
        "bounds": bounds,
        "center": center,
    }


def export_geojsonseq(
    *,
    ogr2ogr: str,
    input_path: Path,
    input_layer: str,
    output_path: Path,
    where: str | None,
) -> None:
    command = [
        ogr2ogr,
        "-f",
        "GeoJSONSeq",
        "-t_srs",
        "EPSG:4326",
        "-lco",
        "RS=NO",
        "-nlt",
        "PROMOTE_TO_MULTI",
    ]
    if where:
        command.extend(["-where", where])
    command.extend([str(output_path), str(input_path), input_layer])
    run(command)


def build_mbtiles(
    *,
    tippecanoe: str,
    geojsonseq_path: Path,
    mbtiles_path: Path,
    tile_layer: str,
    minzoom: int,
    maxzoom: int,
    properties: list[str],
    keep_all: bool,
) -> None:
    command = [
        tippecanoe,
        "-f",
        "-o",
        str(mbtiles_path),
        "-l",
        tile_layer,
        "-Z",
        str(minzoom),
        "-z",
        str(maxzoom),
        "--detect-shared-borders",
        "--read-parallel",
    ]
    if keep_all:
        command.extend(["--no-feature-limit", "--no-tile-size-limit"])
    else:
        command.extend(["--drop-densest-as-needed", "--extend-zooms-if-still-dropping"])

    if properties:
        for prop in properties:
            command.extend(["-y", prop])
    else:
        command.append("--exclude-all")

    command.append(str(geojsonseq_path))
    run(command)


def write_viewer(output_dir: Path, config: dict[str, Any]) -> None:
    viewer_dir = output_dir / "viewer"
    viewer_dir.mkdir(parents=True, exist_ok=True)
    html = HTML_TEMPLATE.format(
        title=config["title"],
        minzoom=config["minzoom"],
        maxzoom=config["maxzoom"],
        config_json=json.dumps(config),
    )
    (viewer_dir / "index.html").write_text(html, encoding="utf-8")
    (output_dir / "viewer_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def build(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tiles_dir = output_dir / "tiles"
    tiles_dir.mkdir(exist_ok=True)

    input_path = args.input.expanduser().resolve()
    name = args.name or input_path.stem
    mbtiles_path = tiles_dir / f"{name}.mbtiles"

    layer_info = get_layer_info(input_path, args.input_layer)
    input_layer = layer_info["input_layer"]
    LOGGER.info("Input layer=%s features=%s", input_layer, layer_info["feature_count"])

    ogr2ogr = require_tool("ogr2ogr", "Install GDAL CLI tools, e.g. conda install -c conda-forge gdal.")
    tippecanoe = require_tool("tippecanoe", "Install tippecanoe, e.g. conda install -c conda-forge tippecanoe.")

    if args.keep_geojsonseq:
        geojsonseq_path = tiles_dir / f"{name}.geojsonseq"
        export_geojsonseq(
            ogr2ogr=ogr2ogr,
            input_path=input_path,
            input_layer=input_layer,
            output_path=geojsonseq_path,
            where=args.where,
        )
        build_mbtiles(
            tippecanoe=tippecanoe,
            geojsonseq_path=geojsonseq_path,
            mbtiles_path=mbtiles_path,
            tile_layer=args.tile_layer,
            minzoom=args.minzoom,
            maxzoom=args.maxzoom,
            properties=parse_property_list(args.include_properties),
            keep_all=args.keep_all,
        )
    else:
        work_dir = (args.work_dir.expanduser().resolve() if args.work_dir else output_dir / "tmp")
        work_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="field_viewer_", dir=work_dir) as tmpdir:
            geojsonseq_path = Path(tmpdir) / f"{name}.geojsonseq"
            export_geojsonseq(
                ogr2ogr=ogr2ogr,
                input_path=input_path,
                input_layer=input_layer,
                output_path=geojsonseq_path,
                where=args.where,
            )
            build_mbtiles(
                tippecanoe=tippecanoe,
                geojsonseq_path=geojsonseq_path,
                mbtiles_path=mbtiles_path,
                tile_layer=args.tile_layer,
                minzoom=args.minzoom,
                maxzoom=args.maxzoom,
                properties=parse_property_list(args.include_properties),
                keep_all=args.keep_all,
            )

    config = {
        "title": name,
        "mbtiles": str(mbtiles_path.relative_to(output_dir)),
        "sourceLayer": args.tile_layer,
        "bounds": [[layer_info["bounds"][0], layer_info["bounds"][1]], [layer_info["bounds"][2], layer_info["bounds"][3]]],
        "center": layer_info["center"],
        "initialZoom": max(args.minzoom, min(args.maxzoom, 10)),
        "minzoom": args.minzoom,
        "maxzoom": args.maxzoom,
        "basemapUrl": args.basemap_url,
        "basemapAttribution": args.basemap_attribution,
        "input": str(input_path),
        "inputLayer": input_layer,
        "featureCount": layer_info["feature_count"],
    }
    write_viewer(output_dir, config)
    LOGGER.info("Viewer package written to %s", output_dir)
    LOGGER.info("Serve with: python vector_tile_viewer.py serve --viewer-dir %s", output_dir)


class TileViewerHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, viewer_dir: Path, config: dict[str, Any], **kwargs: Any) -> None:
        self.viewer_dir = viewer_dir
        self.config = config
        self.mbtiles_path = viewer_dir / config["mbtiles"]
        super().__init__(*args, directory=str(viewer_dir), **kwargs)

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/":
            self.path = "/viewer/index.html"
            return super().do_GET()
        if parsed.path.startswith("/tiles/"):
            return self.serve_tile(parsed.path)
        return super().do_GET()

    def serve_tile(self, path: str) -> None:
        parts = path.strip("/").split("/")
        if len(parts) != 4 or parts[0] != "tiles" or not parts[3].endswith(".pbf"):
            self.send_error(404)
            return
        try:
            z = int(parts[1])
            x = int(parts[2])
            y = int(parts[3].removesuffix(".pbf"))
        except ValueError:
            self.send_error(400)
            return

        tile_row = (1 << z) - 1 - y
        with sqlite3.connect(self.mbtiles_path) as conn:
            row = conn.execute(
                "select tile_data from tiles where zoom_level = ? and tile_column = ? and tile_row = ?",
                (z, x, tile_row),
            ).fetchone()

        if row is None:
            self.send_error(404)
            return

        data = row[0]
        self.send_response(200)
        self.send_header("Content-Type", "application/vnd.mapbox-vector-tile")
        self.send_header("Cache-Control", "public, max-age=86400")
        if len(data) >= 2 and data[:2] == b"\x1f\x8b":
            self.send_header("Content-Encoding", "gzip")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def serve(args: argparse.Namespace) -> None:
    viewer_dir = args.viewer_dir.expanduser().resolve()
    config_path = viewer_dir / "viewer_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing viewer config: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    mbtiles_path = viewer_dir / config["mbtiles"]
    if not mbtiles_path.exists():
        raise FileNotFoundError(f"Missing MBTiles file: {mbtiles_path}")

    handler = partial(TileViewerHandler, viewer_dir=viewer_dir, config=config)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    url_host = "localhost" if args.host in {"0.0.0.0", "::"} else args.host
    LOGGER.info("Serving %s", viewer_dir)
    LOGGER.info("Open: http://%s:%s/", url_host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("Stopping server")
    finally:
        server.server_close()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    try:
        if args.command == "build":
            build(args)
        elif args.command == "serve":
            serve(args)
        else:
            raise ValueError(args.command)
    except Exception as exc:
        LOGGER.error("%s", exc)
        if getattr(args, "verbose", False):
            raise
        sys.exit(1)


if __name__ == "__main__":
    main()
