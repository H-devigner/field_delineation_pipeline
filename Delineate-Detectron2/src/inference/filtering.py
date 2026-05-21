"""
Filtering — Post-delineation merge, area filtering, simplification.

Enhanced from Delineate-Anything inference.py post-delineation merge
for production quality on large datasets.

Handles:
  - Merging fragmented polygons across region boundaries (negative IDs)
  - Geometry repair (Buffer(0), union, decompose MultiPolygon)
  - Small-hole removal using equal-area projection (EPSG:6933)
  - Optional Douglas-Peucker simplification
  - Progress tracking and error resilience
"""

import logging
from typing import Optional

import numpy as np
from osgeo import ogr, osr
from tqdm import tqdm

logger = logging.getLogger(__name__)


def postdelineation_merge(
    gpkg_path: str,
    layer_name: str,
    min_area_m2: float = 2500,
    min_hole_area_m2: float = 2500,
):
    """
    Merge fragmented polygons that span processing-region boundaries.

    Polygons with negative IDs are fragments that need merging.
    Positive IDs are already final.

    Steps:
      1. Scan features: positive IDs get kept as-is (FID becomes new id)
         negative IDs are grouped for union
      2. Delete fragment features
      3. Merge fragments: Buffer(0) + Union + decompose MultiPolygon
      4. Remove small holes, apply area filter
      5. Write merged polygons back
    """
    gpkg = ogr.Open(gpkg_path, 1)
    if gpkg is None:
        logger.error(f"Cannot open GeoPackage: {gpkg_path}")
        return
    layer = gpkg.GetLayerByName(layer_name)
    if layer is None:
        logger.error(f"Layer '{layer_name}' not found in {gpkg_path}")
        gpkg = None
        return

    # Build equal-area transform for accurate area
    src_srs = layer.GetSpatialRef()
    dst_srs = osr.SpatialReference()
    dst_srs.ImportFromEPSG(6933)
    transform = osr.CoordinateTransformation(src_srs, dst_srs)

    max_id = 0
    field_parts = {}
    features_to_delete = []
    n_final = 0

    logger.info("Scanning for fragments to merge...")
    layer.StartTransaction()

    try:
        # ── Step 1: Classify features ────────────────────────
        layer.ResetReading()
        for feature in layer:
            fid = feature.GetFID()
            field_id = feature.GetField("id")
            max_id = max(max_id, fid)

            # Positive ID = already final
            if field_id > 0:
                feature.SetField("id", fid)
                layer.SetFeature(feature)
                n_final += 1
                continue

            # Negative ID = fragment needing merge
            bg = int(feature.GetField("bg"))
            orig_geom = feature.GetGeometryRef()
            if orig_geom is None or orig_geom.IsEmpty():
                features_to_delete.append(fid)
                continue

            geom_clone = orig_geom.Clone()

            key = (field_id, bg)
            if key in field_parts:
                field_parts[key].append(geom_clone)
            else:
                field_parts[key] = [geom_clone]

            features_to_delete.append(fid)

        logger.info(f"  Final polygons: {n_final}")
        logger.info(f"  Fragment groups: {len(field_parts)}")
        logger.info(f"  Fragment features to merge: {len(features_to_delete)}")

        # ── Step 2: Delete fragment features ─────────────────
        for fid in features_to_delete:
            layer.DeleteFeature(fid)

        # ── Step 3: Merge and re-create ──────────────────────
        n_merged = 0
        n_dropped = 0

        for key in tqdm(field_parts.keys(), desc="Merging fragments", unit="group"):
            field_id, bg = key
            parts = field_parts[key]

            # Buffer(0) to repair invalid geometries, then union
            cleaned = []
            for g in parts:
                if g is None or g.IsEmpty():
                    continue
                repaired = g.Buffer(0)
                if repaired is not None and not repaired.IsEmpty():
                    cleaned.append(repaired)

            if not cleaned:
                n_dropped += 1
                continue

            # Progressive union (more robust than chained Union)
            merged = cleaned[0]
            for g in cleaned[1:]:
                try:
                    merged = merged.Union(g)
                except Exception:
                    # If union fails, try Buffer(0) on both and retry
                    try:
                        merged = merged.Buffer(0).Union(g.Buffer(0))
                    except Exception as e:
                        logger.warning(f"Union failed for group {key}: {e}")
                        continue

            if merged is None or merged.IsEmpty():
                n_dropped += 1
                continue

            # Decompose into individual Polygon features
            geom_type = merged.GetGeometryType()
            if geom_type == ogr.wkbPolygon:
                sub_geoms = [merged]
            elif geom_type == ogr.wkbMultiPolygon:
                sub_geoms = [
                    merged.GetGeometryRef(i).Clone()
                    for i in range(merged.GetGeometryCount())
                ]
            elif geom_type == ogr.wkbGeometryCollection:
                # Extract polygons from geometry collection
                sub_geoms = []
                for i in range(merged.GetGeometryCount()):
                    sub = merged.GetGeometryRef(i)
                    if sub.GetGeometryType() == ogr.wkbPolygon:
                        sub_geoms.append(sub.Clone())
            else:
                logger.debug(f"Skipping unexpected geometry: {merged.GetGeometryName()}")
                n_dropped += 1
                continue

            for sub_geom in sub_geoms:
                geom, area = _remove_holes(sub_geom, min_hole_area_m2, transform)
                if area < min_area_m2:
                    n_dropped += 1
                    continue

                max_id += 1
                feat = ogr.Feature(layer.GetLayerDefn())
                feat.SetFID(-1)
                feat.SetGeometry(geom)
                feat.SetField("id", max_id)
                feat.SetField("bg", bg)
                feat.SetField("area", float(area))
                layer.CreateFeature(feat)
                feat = None
                n_merged += 1

        layer.CommitTransaction()
        logger.info(f"  Merged: {n_merged}, Dropped: {n_dropped}")

    except Exception:
        layer.RollbackTransaction()
        raise

    # Compact
    try:
        gpkg.ExecuteSQL("VACUUM")
    except Exception:
        pass

    gpkg = None
    logger.info(f"Post-delineation merge complete: {gpkg_path}")


def simplify_geopackage(
    gpkg_path: str,
    layer_name: str,
    tolerance: float = 1.0,
):
    """
    Apply Douglas-Peucker simplification to all polygons.

    Parameters
    ----------
    tolerance : float
        Simplification tolerance in CRS units.
    """
    gpkg = ogr.Open(gpkg_path, 1)
    if gpkg is None:
        return

    layer = gpkg.GetLayerByName(layer_name)
    if layer is None:
        gpkg = None
        return

    n_simplified = 0
    layer.StartTransaction()
    try:
        layer.ResetReading()
        for feature in tqdm(layer, desc="Simplifying", unit="poly"):
            geom = feature.GetGeometryRef()
            if geom is None:
                continue
            simplified = geom.Simplify(tolerance)
            if simplified and not simplified.IsEmpty():
                # Verify simplified is still valid
                if simplified.IsValid():
                    feature.SetGeometry(simplified)
                    layer.SetFeature(feature)
                    n_simplified += 1
                else:
                    # Try buffer repair after simplification
                    repaired = simplified.Buffer(0)
                    if repaired and not repaired.IsEmpty() and repaired.IsValid():
                        feature.SetGeometry(repaired)
                        layer.SetFeature(feature)
                        n_simplified += 1

        layer.CommitTransaction()
    except Exception:
        layer.RollbackTransaction()
        raise

    gpkg = None
    logger.info(f"Simplified {n_simplified} polygons in: {gpkg_path}")


def filter_by_area(
    gpkg_path: str,
    layer_name: str,
    min_area_m2: float = 2500,
    max_area_m2: Optional[float] = None,
):
    """
    Remove polygons outside area thresholds.

    Parameters
    ----------
    min_area_m2 : float
        Minimum area in m² to keep.
    max_area_m2 : float or None
        Maximum area in m² to keep (None = no upper limit).
    """
    gpkg = ogr.Open(gpkg_path, 1)
    if gpkg is None:
        return

    layer = gpkg.GetLayerByName(layer_name)
    if layer is None:
        gpkg = None
        return

    to_delete = []
    layer.ResetReading()
    for feature in layer:
        area = feature.GetField("area")
        if area is None or area < min_area_m2:
            to_delete.append(feature.GetFID())
        elif max_area_m2 is not None and area > max_area_m2:
            to_delete.append(feature.GetFID())

    layer.StartTransaction()
    for fid in to_delete:
        layer.DeleteFeature(fid)
    layer.CommitTransaction()

    try:
        gpkg.ExecuteSQL("VACUUM")
    except Exception:
        pass

    gpkg = None
    logger.info(f"Removed {len(to_delete)} polygons outside area range")


def _remove_holes(geom, min_hole_area_m2: float, transform):
    """Remove interior rings smaller than threshold. Returns (geom, area_m2)."""
    area_geom = geom.Clone()

    try:
        area_geom.Transform(transform)
    except Exception:
        # If transform fails, estimate area from geometry
        return geom, abs(geom.GetArea())

    n_parts = area_geom.GetGeometryCount()
    if n_parts == 0:
        return geom, 0

    parts_area = np.array([
        abs(area_geom.GetGeometryRef(i).GetArea())
        for i in range(n_parts)
    ])
    total_area = parts_area[0]

    if n_parts == 1:
        return geom, total_area

    # Keep only holes larger than threshold
    new_poly = ogr.Geometry(ogr.wkbPolygon)
    new_poly.AddGeometry(geom.GetGeometryRef(0))  # exterior ring

    for i in range(1, n_parts):
        if parts_area[i] >= min_hole_area_m2:
            total_area -= parts_area[i]
            new_poly.AddGeometry(geom.GetGeometryRef(i))

    return new_poly, total_area
