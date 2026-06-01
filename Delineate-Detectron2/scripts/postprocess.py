"""
postprocess.py — Standalone post-processor for field delineation GeoPackages.

Operates on .gpkg files independently of the inference pipeline.
Supports: merging multiple GPKGs, dissolving overlaps, area filtering,
hole removal, simplification, and LCLU background masking.

Usage:
    # Process a single GPKG
    python scripts/postprocess.py -i fields.gpkg -o fields_clean.gpkg

    # Merge multiple GPKGs from separate runs
    python scripts/postprocess.py -i run1.gpkg run2.gpkg run3.gpkg -o merged.gpkg

    # Custom thresholds
    python scripts/postprocess.py -i fields.gpkg -o clean.gpkg \
        --min-area 5000 --min-hole 2500 --simplify 1.5

    # Apply LCLU mask
    python scripts/postprocess.py -i fields.gpkg -o clean.gpkg \
        --lclu lclu.tif --bg-classes 1 2 3 5 7

    # Dissolve overlapping polygons across tiles
    python scripts/postprocess.py -i fields.gpkg -o clean.gpkg --dissolve
"""

import os
import sys
import argparse
import logging
import time
from copy import deepcopy

import numpy as np
from osgeo import ogr, osr, gdal
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-20s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("postprocess")

ogr.UseExceptions()
gdal.UseExceptions()


def compact_layer(ds, layer_name):
    """Best-effort GeoPackage compaction; never fail postprocessing for cleanup."""
    commands = [
        (f"REPACK {layer_name}", "OGRSQL"),
        ("VACUUM", "SQLITE"),
    ]
    errors = []

    for sql, dialect in commands:
        try:
            result = ds.ExecuteSQL(sql, None, dialect)
            if result is not None:
                ds.ReleaseResultSet(result)
            return
        except RuntimeError as exc:
            errors.append(f"{sql} ({dialect}): {exc}")

    logger.warning("Skipping compaction; unsupported by this GDAL/driver combination: %s", "; ".join(errors))


# ═══════════════════════════════════════════════════════════════
# Core operations
# ═══════════════════════════════════════════════════════════════

def merge_gpkgs(input_paths, output_path, layer_name="fields"):
    """
    Merge multiple GeoPackage files into one, copying all features.
    """
    logger.info(f"Merging {len(input_paths)} GPKG(s) → {output_path}")

    driver = ogr.GetDriverByName("GPKG")
    if os.path.exists(output_path):
        os.remove(output_path)
    out_ds = driver.CreateDataSource(output_path)

    out_layer = None
    total = 0

    for path in input_paths:
        if not os.path.exists(path):
            logger.warning(f"  Skipping (not found): {path}")
            continue

        src_ds = ogr.Open(path, 0)
        if src_ds is None:
            logger.warning(f"  Skipping (cannot open): {path}")
            continue

        src_layer = src_ds.GetLayerByName(layer_name)
        if src_layer is None:
            # Try first layer
            src_layer = src_ds.GetLayer(0)
            if src_layer is None:
                logger.warning(f"  Skipping (no layers): {path}")
                src_ds = None
                continue

        # Create output layer from first input
        if out_layer is None:
            srs = src_layer.GetSpatialRef()
            out_layer = out_ds.CreateLayer(layer_name, srs, ogr.wkbPolygon)
            src_defn = src_layer.GetLayerDefn()
            for i in range(src_defn.GetFieldCount()):
                out_layer.CreateField(src_defn.GetFieldDefn(i))

        # Copy features
        out_layer.StartTransaction()
        out_defn = out_layer.GetLayerDefn()
        count = 0
        src_layer.ResetReading()
        for feat in src_layer:
            geom = feat.GetGeometryRef()
            if geom is None or geom.IsEmpty():
                continue
            out_feat = ogr.Feature(out_defn)
            out_feat.SetGeometry(geom.Clone())
            for i in range(out_defn.GetFieldCount()):
                fname = out_defn.GetFieldDefn(i).GetName()
                try:
                    out_feat.SetField(fname, feat.GetField(fname))
                except Exception:
                    pass
            out_layer.CreateFeature(out_feat)
            count += 1
        out_layer.CommitTransaction()
        total += count
        logger.info(f"  {path}: {count} features")
        src_ds = None

    out_ds = None
    logger.info(f"  Total: {total} features → {output_path}")
    return output_path


def dissolve_overlaps(gpkg_path, layer_name="fields", iou_threshold=0.3):
    """
    Dissolve overlapping polygons using spatial index + IoU merge.
    Polygons overlapping above the IoU threshold are unioned.
    """
    logger.info(f"Dissolving overlaps (IoU > {iou_threshold})...")

    ds = ogr.Open(gpkg_path, 1)
    layer = ds.GetLayerByName(layer_name)
    if layer is None:
        layer = ds.GetLayer(0)

    srs = layer.GetSpatialRef()
    ea_srs = osr.SpatialReference()
    ea_srs.ImportFromEPSG(6933)
    transform = osr.CoordinateTransformation(srs, ea_srs)

    # Load all geometries into memory
    features_data = []
    layer.ResetReading()
    for feat in layer:
        geom = feat.GetGeometryRef()
        if geom is None or geom.IsEmpty():
            continue
        fid = feat.GetFID()
        geom_clone = geom.Clone()
        bg = 0
        try:
            bg = feat.GetField("bg")
        except Exception:
            pass
        features_data.append({"fid": fid, "geom": geom_clone, "bg": bg, "merged": False})

    logger.info(f"  Loaded {len(features_data)} features")

    # Build merge groups using pairwise IoU
    merge_groups = []
    n = len(features_data)

    for i in tqdm(range(n), desc="  Finding overlaps", unit="poly"):
        if features_data[i]["merged"]:
            continue

        group = [i]
        features_data[i]["merged"] = True

        # Check against remaining
        queue = [i]
        while queue:
            ci = queue.pop()
            geom_a = features_data[ci]["geom"]
            env_a = geom_a.GetEnvelope()

            for j in range(ci + 1, n):
                if features_data[j]["merged"]:
                    continue
                geom_b = features_data[j]["geom"]
                env_b = geom_b.GetEnvelope()

                # Quick envelope check
                if env_a[1] < env_b[0] or env_b[1] < env_a[0]:
                    continue
                if env_a[3] < env_b[2] or env_b[3] < env_a[2]:
                    continue

                if not geom_a.Intersects(geom_b):
                    continue

                try:
                    inter = geom_a.Intersection(geom_b)
                    if inter is None or inter.IsEmpty():
                        continue
                    inter_area = inter.GetArea()
                    area_a = geom_a.GetArea()
                    area_b = geom_b.GetArea()
                    union_area = area_a + area_b - inter_area
                    if union_area <= 0:
                        continue
                    iou = inter_area / union_area
                    if iou >= iou_threshold:
                        features_data[j]["merged"] = True
                        group.append(j)
                        queue.append(j)
                except Exception:
                    continue

        merge_groups.append(group)

    # Rebuild layer
    n_merged = sum(1 for g in merge_groups if len(g) > 1)
    logger.info(f"  Merge groups: {len(merge_groups)} ({n_merged} with overlaps)")

    # Delete all features and rewrite
    layer.StartTransaction()
    layer.ResetReading()
    fids_to_delete = []
    for feat in layer:
        fids_to_delete.append(feat.GetFID())
    for fid in fids_to_delete:
        layer.DeleteFeature(fid)
    layer.CommitTransaction()

    compact_layer(ds, layer_name)

    # Write merged geometries
    layer.StartTransaction()
    defn = layer.GetLayerDefn()
    new_id = 1

    for group in tqdm(merge_groups, desc="  Writing merged", unit="group"):
        geoms = [features_data[i]["geom"] for i in group]
        bg = max(features_data[i]["bg"] for i in group)

        if len(geoms) == 1:
            merged = geoms[0]
        else:
            merged = geoms[0].Clone()
            for g in geoms[1:]:
                try:
                    merged = merged.Union(g)
                except Exception:
                    try:
                        merged = merged.Buffer(0).Union(g.Buffer(0))
                    except Exception:
                        continue

        if merged is None or merged.IsEmpty():
            continue

        # Decompose MultiPolygon
        sub_geoms = _decompose(merged)

        for sg in sub_geoms:
            feat = ogr.Feature(defn)
            feat.SetGeometry(sg)
            try:
                feat.SetField("id", new_id)
            except Exception:
                pass
            try:
                # Compute area in equal-area projection
                ea = sg.Clone()
                ea.Transform(transform)
                feat.SetField("area", ea.GetArea())
            except Exception:
                pass
            try:
                feat.SetField("bg", bg)
            except Exception:
                pass
            layer.CreateFeature(feat)
            new_id += 1

    layer.CommitTransaction()
    ds = None
    logger.info(f"  Result: {new_id - 1} polygons")


def filter_by_area(gpkg_path, layer_name="fields",
                   min_area_m2=2500, max_area_m2=None,
                   min_hole_area_m2=2500):
    """
    Remove polygons below/above area thresholds and small holes.
    Areas are computed in EPSG:6933 (World Cylindrical Equal Area).
    """
    logger.info(f"Filtering: min={min_area_m2}m², max={max_area_m2}m², min_hole={min_hole_area_m2}m²")

    ds = ogr.Open(gpkg_path, 1)
    layer = ds.GetLayerByName(layer_name)
    if layer is None:
        layer = ds.GetLayer(0)

    srs = layer.GetSpatialRef()
    ea_srs = osr.SpatialReference()
    ea_srs.ImportFromEPSG(6933)
    transform = osr.CoordinateTransformation(srs, ea_srs)

    to_delete = []
    to_update = []
    n_total = layer.GetFeatureCount()

    layer.ResetReading()
    for feat in tqdm(layer, desc="  Filtering", total=n_total, unit="poly"):
        geom = feat.GetGeometryRef()
        if geom is None or geom.IsEmpty():
            to_delete.append(feat.GetFID())
            continue

        ea_geom = geom.Clone()
        ea_geom.Transform(transform)
        area = ea_geom.GetArea()

        if area < min_area_m2:
            to_delete.append(feat.GetFID())
            continue
        if max_area_m2 is not None and area > max_area_m2:
            to_delete.append(feat.GetFID())
            continue

        # Remove small holes
        new_geom, new_area = _remove_holes(geom, min_hole_area_m2, transform)
        if new_geom is not None:
            to_update.append((feat.GetFID(), new_geom, new_area))

    layer.StartTransaction()
    for fid in to_delete:
        layer.DeleteFeature(fid)
    for fid, geom, area in to_update:
        feat = layer.GetFeature(fid)
        feat.SetGeometry(geom)
        try:
            feat.SetField("area", area)
        except Exception:
            pass
        layer.SetFeature(feat)
    layer.CommitTransaction()

    compact_layer(ds, layer_name)
    remaining = layer.GetFeatureCount()
    ds = None
    logger.info(f"  Removed {len(to_delete)}, updated {len(to_update)}, remaining: {remaining}")


def simplify_geometries(gpkg_path, layer_name="fields", tolerance=1.0):
    """Apply Douglas-Peucker simplification to all polygons."""
    logger.info(f"Simplifying (tolerance={tolerance})...")

    ds = ogr.Open(gpkg_path, 1)
    layer = ds.GetLayerByName(layer_name)
    if layer is None:
        layer = ds.GetLayer(0)

    n_total = layer.GetFeatureCount()
    layer.StartTransaction()
    layer.ResetReading()

    for feat in tqdm(layer, desc="  Simplifying", total=n_total, unit="poly"):
        geom = feat.GetGeometryRef()
        if geom is None:
            continue
        simplified = geom.Simplify(tolerance)
        if simplified is not None and not simplified.IsEmpty():
            feat.SetGeometry(simplified)
            layer.SetFeature(feat)

    layer.CommitTransaction()
    ds = None
    logger.info(f"  Simplified {n_total} polygons")


def apply_lclu_mask(gpkg_path, lclu_path, layer_name="fields",
                    bg_classes=None, clip_classes=None):
    """
    Tag or remove polygons based on LCLU raster values.

    bg_classes: polygons mostly overlapping these classes get bg=1
    clip_classes: polygons mostly overlapping these classes get removed
    """
    if bg_classes is None and clip_classes is None:
        return

    logger.info(f"Applying LCLU mask: {lclu_path}")
    logger.info(f"  background classes: {bg_classes}")
    logger.info(f"  clip classes: {clip_classes}")

    lclu_ds = gdal.Open(lclu_path)
    if lclu_ds is None:
        logger.error(f"Cannot open LCLU: {lclu_path}")
        return

    lclu_band = lclu_ds.GetRasterBand(1)
    lclu_gt = lclu_ds.GetGeoTransform()
    lclu_w, lclu_h = lclu_ds.RasterXSize, lclu_ds.RasterYSize

    ds = ogr.Open(gpkg_path, 1)
    layer = ds.GetLayerByName(layer_name)
    if layer is None:
        layer = ds.GetLayer(0)

    to_delete = []
    to_tag_bg = []

    layer.ResetReading()
    for feat in tqdm(layer, desc="  LCLU check", unit="poly"):
        geom = feat.GetGeometryRef()
        if geom is None:
            continue

        env = geom.GetEnvelope()  # minX, maxX, minY, maxY

        # Convert envelope to pixel coords
        px_min = int((env[0] - lclu_gt[0]) / lclu_gt[1])
        px_max = int((env[1] - lclu_gt[0]) / lclu_gt[1]) + 1
        py_min = int((env[3] - lclu_gt[3]) / lclu_gt[5])
        py_max = int((env[2] - lclu_gt[3]) / lclu_gt[5]) + 1

        px_min = max(0, min(px_min, lclu_w - 1))
        px_max = max(0, min(px_max, lclu_w))
        py_min = max(0, min(py_min, lclu_h - 1))
        py_max = max(0, min(py_max, lclu_h))

        if px_max <= px_min or py_max <= py_min:
            continue

        chunk = lclu_band.ReadAsArray(px_min, py_min, px_max - px_min, py_max - py_min)
        if chunk is None or chunk.size == 0:
            continue

        total_pixels = chunk.size

        # Check clip classes
        if clip_classes:
            clip_mask = np.isin(chunk, clip_classes)
            clip_ratio = clip_mask.sum() / total_pixels
            if clip_ratio > 0.5:
                to_delete.append(feat.GetFID())
                continue

        # Check background classes
        if bg_classes:
            bg_mask = np.isin(chunk, bg_classes)
            bg_ratio = bg_mask.sum() / total_pixels
            if bg_ratio > 0.5:
                to_tag_bg.append(feat.GetFID())

    layer.StartTransaction()
    for fid in to_delete:
        layer.DeleteFeature(fid)
    for fid in to_tag_bg:
        feat = layer.GetFeature(fid)
        try:
            feat.SetField("bg", 1)
        except Exception:
            pass
        layer.SetFeature(feat)
    layer.CommitTransaction()

    if to_delete:
        compact_layer(ds, layer_name)

    remaining = layer.GetFeatureCount()
    ds = None
    lclu_ds = None
    logger.info(f"  Clipped: {len(to_delete)}, tagged bg: {len(to_tag_bg)}, remaining: {remaining}")


def add_statistics(gpkg_path, layer_name="fields"):
    """
    Recompute area (m²) and perimeter (m) for all polygons using EPSG:6933.
    """
    logger.info("Recomputing statistics...")

    ds = ogr.Open(gpkg_path, 1)
    layer = ds.GetLayerByName(layer_name)
    if layer is None:
        layer = ds.GetLayer(0)

    srs = layer.GetSpatialRef()
    ea_srs = osr.SpatialReference()
    ea_srs.ImportFromEPSG(6933)
    transform = osr.CoordinateTransformation(srs, ea_srs)

    # Ensure fields exist
    defn = layer.GetLayerDefn()
    field_names = [defn.GetFieldDefn(i).GetName() for i in range(defn.GetFieldCount())]
    if "area" not in field_names:
        layer.CreateField(ogr.FieldDefn("area", ogr.OFTReal))
    if "perimeter" not in field_names:
        layer.CreateField(ogr.FieldDefn("perimeter", ogr.OFTReal))
    if "compactness" not in field_names:
        layer.CreateField(ogr.FieldDefn("compactness", ogr.OFTReal))

    layer.StartTransaction()
    layer.ResetReading()
    count = 0
    for feat in layer:
        geom = feat.GetGeometryRef()
        if geom is None:
            continue
        ea = geom.Clone()
        ea.Transform(transform)
        area = ea.GetArea()
        boundary = ea.Boundary()
        perimeter = boundary.Length() if boundary else 0

        feat.SetField("area", round(area, 1))
        feat.SetField("perimeter", round(perimeter, 1))

        # Polsby-Popper compactness: 4π × area / perimeter²
        if perimeter > 0:
            compactness = (4 * 3.14159265 * area) / (perimeter * perimeter)
            feat.SetField("compactness", round(compactness, 4))

        layer.SetFeature(feat)
        count += 1

    layer.CommitTransaction()
    ds = None
    logger.info(f"  Updated {count} features (area, perimeter, compactness)")


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def _remove_holes(geom, min_hole_area_m2, transform):
    """Remove interior rings smaller than threshold. Returns (geom, area)."""
    geom_type = geom.GetGeometryType()
    if geom_type != ogr.wkbPolygon:
        ea = geom.Clone()
        ea.Transform(transform)
        return None, ea.GetArea()

    n_rings = geom.GetGeometryCount()
    if n_rings <= 1:
        return None, 0

    ea = geom.Clone()
    ea.Transform(transform)

    new_poly = ogr.Geometry(ogr.wkbPolygon)
    new_poly.AddGeometry(geom.GetGeometryRef(0))  # exterior
    total_area = ea.GetGeometryRef(0).GetArea() if ea.GetGeometryCount() > 0 else 0

    changed = False
    for i in range(1, n_rings):
        hole_area = abs(ea.GetGeometryRef(i).GetArea())
        if hole_area >= min_hole_area_m2:
            new_poly.AddGeometry(geom.GetGeometryRef(i))
            total_area -= hole_area
        else:
            changed = True

    if changed:
        return new_poly, abs(total_area)
    return None, abs(total_area)


def _decompose(geom):
    """Decompose a geometry into individual Polygons."""
    gt = geom.GetGeometryType()
    if gt == ogr.wkbPolygon:
        return [geom]
    elif gt in (ogr.wkbMultiPolygon, ogr.wkbGeometryCollection):
        result = []
        for i in range(geom.GetGeometryCount()):
            sub = geom.GetGeometryRef(i)
            if sub.GetGeometryType() == ogr.wkbPolygon:
                result.append(sub.Clone())
        return result
    return []


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Standalone post-processor for field delineation GeoPackages",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file — filter + simplify
  python scripts/postprocess.py -i fields.gpkg -o clean.gpkg

  # Merge multiple runs
  python scripts/postprocess.py -i run1.gpkg run2.gpkg -o merged.gpkg

  # Full pipeline: merge + dissolve + filter + simplify + LCLU
  python scripts/postprocess.py -i run1.gpkg run2.gpkg -o final.gpkg \\
    --dissolve --min-area 5000 --simplify 1.5 \\
    --lclu lclu.tif --bg-classes 1 2 3 5 7 --clip-classes 0 6 8
        """,
    )
    parser.add_argument("-i", "--input", nargs="+", required=True,
                        help="Input GPKG file(s)")
    parser.add_argument("-o", "--output", required=True,
                        help="Output GPKG path")
    parser.add_argument("--layer", default="fields",
                        help="Layer name (default: fields)")

    # Operations
    parser.add_argument("--dissolve", action="store_true",
                        help="Dissolve overlapping polygons")
    parser.add_argument("--iou", type=float, default=0.3,
                        help="IoU threshold for dissolve (default: 0.3)")
    parser.add_argument("--min-area", type=float, default=2500,
                        help="Min polygon area in m² (default: 2500)")
    parser.add_argument("--max-area", type=float, default=None,
                        help="Max polygon area in m² (default: none)")
    parser.add_argument("--min-hole", type=float, default=2500,
                        help="Min hole area in m² to keep (default: 2500)")
    parser.add_argument("--simplify", type=float, default=None,
                        help="Douglas-Peucker tolerance in CRS units")
    parser.add_argument("--no-stats", action="store_true",
                        help="Skip recomputing area/perimeter/compactness")

    # LCLU
    parser.add_argument("--lclu", default=None,
                        help="LCLU raster path")
    parser.add_argument("--bg-classes", nargs="+", type=int, default=None,
                        help="LCLU background classes (e.g. 1 2 3 5 7)")
    parser.add_argument("--clip-classes", nargs="+", type=int, default=None,
                        help="LCLU clip classes (e.g. 0 6 8)")

    args = parser.parse_args()

    t_start = time.time()

    # ── Step 1: Merge (or copy) inputs ───────────────────────
    if len(args.input) > 1 or args.input[0] != args.output:
        merge_gpkgs(args.input, args.output, args.layer)
    else:
        logger.info("In-place processing")

    # ── Step 2: Dissolve overlaps ────────────────────────────
    if args.dissolve:
        dissolve_overlaps(args.output, args.layer, args.iou)

    # ── Step 3: Area filter + hole removal ───────────────────
    filter_by_area(
        args.output, args.layer,
        min_area_m2=args.min_area,
        max_area_m2=args.max_area,
        min_hole_area_m2=args.min_hole,
    )

    # ── Step 4: LCLU mask ────────────────────────────────────
    if args.lclu:
        apply_lclu_mask(
            args.output, args.lclu, args.layer,
            bg_classes=args.bg_classes,
            clip_classes=args.clip_classes,
        )

    # ── Step 5: Simplify ─────────────────────────────────────
    if args.simplify:
        simplify_geometries(args.output, args.layer, args.simplify)

    # ── Step 6: Statistics ───────────────────────────────────
    if not args.no_stats:
        add_statistics(args.output, args.layer)

    elapsed = time.time() - t_start
    logger.info(f"Done in {elapsed:.1f}s → {args.output}")


if __name__ == "__main__":
    main()
