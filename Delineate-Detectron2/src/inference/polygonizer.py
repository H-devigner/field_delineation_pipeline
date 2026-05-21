"""
Polygonizer — Vectorize the instance raster into polygons.

Adapted from Delineate-Anything/methods/main/PolygonizationWorker.py
Responsibilities:
  - Convert instance raster → vector polygons via rasterio.features.shapes
  - Apply affine transform (pixel coords → CRS coords)
  - Remove small holes using equal-area projection (EPSG:6933) for m² accuracy
  - Write polygons to GeoPackage (OGR) or return as Shapely geometries
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from affine import Affine
from osgeo import gdal, ogr, osr
from rasterio import features
from shapely.affinity import affine_transform as shapely_affine_transform
from shapely.geometry import box, shape

logger = logging.getLogger(__name__)


class Polygonizer:
    """Vectorize an instance raster into geo-referenced polygons."""

    def __init__(self, srs_wkt: str, config: dict):
        """
        Parameters
        ----------
        srs_wkt : str
            WKT of the raster's spatial reference system.
        config : dict
            Keys: minimum_area_m2, minimum_hole_area_m2, minimum_part_area_m2
        """
        self.srs_wkt = srs_wkt
        self.min_area_m2 = config.get("minimum_area_m2", 2500)
        self.min_hole_area_m2 = config.get("minimum_hole_area_m2", 2500)
        self.min_part_area_m2 = config.get("minimum_part_area_m2", 0)

        # Build equal-area transform for m² calculations
        self.srs = osr.SpatialReference()
        self.srs.ImportFromWkt(srs_wkt)

        self.equal_area_srs = osr.SpatialReference()
        self.equal_area_srs.ImportFromEPSG(6933)  # World Cylindrical Equal Area
        self.coord_transform = osr.CoordinateTransformation(self.srs, self.equal_area_srs)

    def polygonize_region(
        self,
        instances: np.ndarray,
        geotransform: Tuple,
    ) -> List[Dict]:
        """
        Vectorize an instance raster into polygons.

        Parameters
        ----------
        instances : np.ndarray, shape (H, W), int32
            Instance IDs. 0=background, >1=field IDs.
        geotransform : tuple
            GDAL geotransform for this region.

        Returns
        -------
        list of dict
            Each dict has keys: geometry (WKB), area_m2, id, is_background
        """
        affine = Affine.from_gdal(*geotransform)
        a, b, c, d, e, f = affine.a, affine.b, affine.c, affine.d, affine.e, affine.f
        shapely_params = (a, b, d, e, c, f)

        height, width = instances.shape
        image_bbox = box(0, 0, width, height)

        # Generate pixel-space polygons from the instance raster
        # Mask: only vectorize cells with field IDs (>1) or negative (background fields)
        mask = (instances < 0) | (instances > 1)
        shapes_gen = list(features.shapes(instances, mask=mask))

        results = []

        for geom_json, value in shapes_gen:
            if 0 <= value < 2:
                continue

            geom = shape(geom_json)

            # Check if polygon touches region edge
            poly_bbox = box(*geom.bounds)
            touches_edge = poly_bbox.intersects(image_bbox.boundary)

            # Transform to CRS coordinates
            geom_geo = shapely_affine_transform(geom, shapely_params)
            if geom_geo.is_empty or not geom_geo.is_valid:
                continue

            wkb = geom_geo.wkb
            if not wkb:
                continue

            # Compute area in equal-area projection
            ogr_geom = ogr.CreateGeometryFromWkb(wkb)
            if ogr_geom is None or ogr_geom.IsEmpty():
                continue

            is_background = value < 0
            ogr_geom, area_m2 = self._remove_small_holes(ogr_geom, self.min_hole_area_m2)

            # Area filtering
            if not is_background:
                if touches_edge and area_m2 < self.min_part_area_m2:
                    continue
                if not touches_edge and area_m2 < self.min_area_m2:
                    continue

            # Negative ID = needs merging in post-delineation; positive = final
            result_id = -int(value) if touches_edge else int(value)
            if is_background:
                result_id = -result_id

            results.append({
                "geometry_wkb": wkb,
                "area_m2": float(area_m2),
                "id": result_id,
                "is_background": int(is_background),
            })

        logger.debug(f"Polygonized region: {len(results)} polygons")
        return results

    def write_to_geopackage(
        self,
        polygons: List[Dict],
        gpkg_path: str,
        layer_name: str = "fields",
    ):
        """
        Write polygons to a GeoPackage file.

        Parameters
        ----------
        polygons : list of dict
            From polygonize_region().
        gpkg_path : str
            Output GeoPackage path.
        layer_name : str
            Layer name within the GeoPackage.
        """
        driver = ogr.GetDriverByName("GPKG")

        # Create or open GeoPackage
        import os
        if os.path.exists(gpkg_path):
            ds = ogr.Open(gpkg_path, 1)
            layer = ds.GetLayerByName(layer_name)
            if layer is None:
                layer = self._create_layer(ds, layer_name)
        else:
            ds = driver.CreateDataSource(gpkg_path)
            layer = self._create_layer(ds, layer_name)

        layer.StartTransaction()
        try:
            layer_defn = layer.GetLayerDefn()
            for poly in polygons:
                feature = ogr.Feature(layer_defn)
                geom = ogr.CreateGeometryFromWkb(poly["geometry_wkb"])
                feature.SetGeometry(geom)
                feature.SetField("id", poly["id"])
                feature.SetField("area", poly["area_m2"])
                feature.SetField("bg", poly["is_background"])
                layer.CreateFeature(feature)
                feature = None
            layer.CommitTransaction()
        except Exception:
            layer.RollbackTransaction()
            raise

        ds = None
        logger.debug(f"Wrote {len(polygons)} polygons to {gpkg_path}")

    def _create_layer(self, ds, layer_name: str):
        """Create a new layer in the GeoPackage."""
        layer = ds.CreateLayer(layer_name, self.srs, ogr.wkbPolygon)
        layer.CreateField(ogr.FieldDefn("id", ogr.OFTInteger))
        layer.CreateField(ogr.FieldDefn("area", ogr.OFTReal))
        layer.CreateField(ogr.FieldDefn("bg", ogr.OFTInteger))
        return layer

    def _remove_small_holes(self, ogr_geom, min_hole_area_m2: float):
        """Remove interior rings smaller than threshold (in m²)."""
        area_geom = ogr_geom.Clone()
        area_geom.Transform(self.coord_transform)

        n_parts = area_geom.GetGeometryCount()
        parts_area = np.array([area_geom.GetGeometryRef(i).GetArea() for i in range(n_parts)])

        total_area = parts_area[0] if n_parts > 0 else 0

        if n_parts <= 1:
            return ogr_geom, total_area

        # Keep only holes larger than threshold
        new_poly = ogr.Geometry(ogr.wkbPolygon)
        new_poly.AddGeometry(ogr_geom.GetGeometryRef(0))  # exterior ring

        for i in range(1, n_parts):
            if parts_area[i] >= min_hole_area_m2:
                total_area -= parts_area[i]
                new_poly.AddGeometry(ogr_geom.GetGeometryRef(i))

        return new_poly, total_area
