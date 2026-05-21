"""
GeoJSON Export — RFC 7946 compliant, Kepler.gl compatible.

Responsibilities:
  - Read polygons from GeoPackage
  - Reproject to EPSG:4326 (WGS84) if needed
  - Write RFC 7946 GeoJSON FeatureCollection (no 'crs' field)
"""

import json
import logging
from pathlib import Path
from typing import Optional

from osgeo import ogr, osr
from pyproj import Transformer

logger = logging.getLogger(__name__)


def export_geojson(
    gpkg_path: str,
    layer_name: str = "fields",
    output_path: Optional[str] = None,
) -> str:
    """
    Convert a GeoPackage layer to RFC 7946 GeoJSON (WGS84).

    Parameters
    ----------
    gpkg_path : str
        Input GeoPackage path.
    layer_name : str
        Layer name to export.
    output_path : str or None
        Output GeoJSON path. If None, replaces .gpkg → .geojson.

    Returns
    -------
    str
        Path to the written GeoJSON file.
    """
    if output_path is None:
        output_path = str(Path(gpkg_path).with_suffix(".geojson"))

    gpkg = ogr.Open(gpkg_path, 0)
    if gpkg is None:
        raise FileNotFoundError(f"Cannot open: {gpkg_path}")

    layer = gpkg.GetLayerByName(layer_name)
    if layer is None:
        raise ValueError(f"Layer '{layer_name}' not found in {gpkg_path}")

    # Determine source CRS
    src_srs = layer.GetSpatialRef()
    src_epsg = src_srs.GetAttrValue("AUTHORITY", 1) if src_srs else None

    need_reproject = src_epsg != "4326"
    if need_reproject and src_srs is not None:
        # Build pyproj transformer for accurate reprojection
        src_wkt = src_srs.ExportToWkt()
        transformer = Transformer.from_crs(
            f"EPSG:{src_epsg}" if src_epsg else src_wkt,
            "EPSG:4326",
            always_xy=True,
        )
        logger.info(f"Reprojecting from EPSG:{src_epsg} → EPSG:4326")
    else:
        logger.info("Source already in EPSG:4326")

    features = []
    for feature in layer:
        geom = feature.GetGeometryRef()
        if geom is None or geom.IsEmpty():
            continue

        field_id = feature.GetField("id")
        area = feature.GetField("area")
        bg = feature.GetField("bg")

        # Convert OGR geometry to GeoJSON
        geom_json = json.loads(geom.ExportToJson())

        # Reproject coordinates if needed
        if need_reproject:
            geom_json = _reproject_geojson_geometry(geom_json, transformer)

        features.append({
            "type": "Feature",
            "properties": {
                "id": field_id,
                "area_m2": round(float(area), 2) if area else 0,
                "class": "land-Sx1C",
                "background": bool(bg),
            },
            "geometry": geom_json,
        })

    gpkg = None

    # RFC 7946: no 'crs' field, coordinates must be WGS84
    geojson = {
        "type": "FeatureCollection",
        "features": features,
    }

    with open(output_path, "w") as f:
        json.dump(geojson, f)

    size_kb = Path(output_path).stat().st_size / 1024
    logger.info(f"Exported {len(features)} fields → {output_path} ({size_kb:.0f} KB)")

    return output_path


def _reproject_geojson_geometry(geom_json: dict, transformer) -> dict:
    """Reproject GeoJSON geometry coordinates using pyproj Transformer."""
    geom_type = geom_json["type"]

    if geom_type == "Polygon":
        geom_json["coordinates"] = [
            _reproject_ring(ring, transformer)
            for ring in geom_json["coordinates"]
        ]
    elif geom_type == "MultiPolygon":
        geom_json["coordinates"] = [
            [_reproject_ring(ring, transformer) for ring in polygon]
            for polygon in geom_json["coordinates"]
        ]

    return geom_json


def _reproject_ring(ring, transformer):
    """Reproject a coordinate ring [(x,y), ...] to EPSG:4326."""
    coords = []
    for point in ring:
        x, y = point[0], point[1]
        lon, lat = transformer.transform(x, y)
        coords.append([round(lon, 8), round(lat, 8)])
    return coords
