"""
DataAnalyser — Validate and characterise input GeoTIFFs.

Adapted from Delineate-Anything/methods/main/DataAnalyser.py
Responsibilities:
  - Validate that all TIFFs share the same CRS, pixel size, and data type
  - Compute total geographic extent across all input files
  - Estimate per-band normalization bounds (p1/p99 percentiles)
  - Auto-detect pixel resolution for tile sizing
"""

import math
import logging
from typing import List, Optional, Tuple

import numpy as np
from osgeo import gdal, osr
from tqdm import tqdm

logger = logging.getLogger(__name__)

gdal.UseExceptions()


class DataAnalyser:
    """Analyse a set of GeoTIFFs for compatibility and normalization."""

    def __init__(self, tiffs: List[str], bands: List[int], super_resolution: Optional[int] = None):
        """
        Parameters
        ----------
        tiffs : list of str
            Paths to input GeoTIFF files.
        bands : list of int
            Band indices to use (1-based GDAL indexing), e.g. [1, 2, 3] for RGB.
        super_resolution : int or None
            Scale factor for super-resolution. None = auto-detect.
        """
        self.tiffs = tiffs
        self.bands = bands
        self.sr = super_resolution

        # Set after is_compatible()
        self.projection = None
        self.data_type = None
        self.pixel_size_x = None
        self.pixel_size_y = None
        self.total_bounds = None
        self.tile_size = None
        self.scale = None

        # Set after calc_normalization_bounds()
        self.min_vals = None
        self.max_vals = None

    def is_compatible(self) -> bool:
        """Check that all TIFFs share the same CRS, pixel size, and data type."""
        projections = []
        pixel_sizes_x = []
        pixel_sizes_y = []
        data_types = []

        for path in self.tiffs:
            ds = gdal.Open(path)
            if ds is None:
                logger.error(f"Cannot open: {path}")
                return False
            projections.append(ds.GetProjection())
            gt = ds.GetGeoTransform()
            pixel_sizes_x.append(gt[1])
            pixel_sizes_y.append(gt[5])
            data_types.append(ds.GetRasterBand(1).DataType)
            ds = None

        if len(projections) == 0:
            return False

        self.projection = projections[0]
        self.data_type = data_types[0]
        self.pixel_size_x = pixel_sizes_x[0]
        self.pixel_size_y = pixel_sizes_y[0]

        compatible = (
            all(p == projections[0] for p in projections)
            and all(abs(p - pixel_sizes_x[0]) < 1e-6 for p in pixel_sizes_x)
            and all(abs(p - pixel_sizes_y[0]) < 1e-6 for p in pixel_sizes_y)
            and all(t == data_types[0] for t in data_types)
        )

        if compatible:
            self.total_bounds = self._get_total_bounds()

            if self.sr is None:
                pixel_m = self._get_pixel_size_meters(self.tiffs[0])
                self.tile_size = 640 if pixel_m < 5 else 320
                logger.info(f"Pixel size: {pixel_m:.2f} m → tile_size={self.tile_size}")
            else:
                self.tile_size = 640 // self.sr

            self.scale = 640 // self.tile_size

        return compatible

    def calc_normalization_bounds(self):
        """Compute per-band p1/p99 percentile bounds for normalization."""
        mins = [[] for _ in self.bands]
        maxs = [[] for _ in self.bands]

        for path in tqdm(self.tiffs, desc="Normalization bounds"):
            ds = gdal.Open(path, gdal.GA_ReadOnly)

            for i, band_idx in enumerate(self.bands):
                rb = ds.GetRasterBand(band_idx)

                # Byte data → already 0–255
                if rb.DataType == gdal.GDT_Byte:
                    self.min_vals = [0] * len(self.bands)
                    self.max_vals = [255] * len(self.bands)
                    ds = None
                    logger.info("Byte data detected — normalization: 0–255")
                    return

                data = rb.ReadAsArray()
                valid = data[data > 0]
                if len(valid) == 0:
                    continue
                p1, p99 = np.percentile(valid, [1, 99])
                mins[i].append(p1)
                maxs[i].append(p99)

            ds = None

        self.min_vals = [np.mean(m) if m else 0 for m in mins]
        self.max_vals = [np.mean(m) if m else 255 for m in maxs]
        logger.info(f"Normalization bounds: min={self.min_vals}, max={self.max_vals}")

    def get_pixel_offset(self, ds) -> Tuple[int, int]:
        """Compute pixel offset of a dataset relative to total_bounds."""
        gt = ds.GetGeoTransform()
        offset_x = (gt[0] - self.total_bounds[0]) / gt[1]
        offset_y = (self.total_bounds[3] - gt[3]) / abs(gt[5])
        return int(round(offset_x)), int(round(offset_y))

    # ── Private helpers ──────────────────────────────────────

    def _get_bounds(self, ds) -> Tuple[float, float, float, float]:
        gt = ds.GetGeoTransform()
        cols, rows = ds.RasterXSize, ds.RasterYSize
        minx = gt[0]
        maxx = gt[0] + cols * gt[1]
        miny = gt[3] + rows * gt[5]
        maxy = gt[3]
        return (minx, miny, maxx, maxy)

    def _get_total_bounds(self) -> Tuple[float, float, float, float]:
        extents = []
        for path in self.tiffs:
            ds = gdal.Open(path)
            extents.append(self._get_bounds(ds))
            ds = None

        return (
            min(e[0] for e in extents),
            min(e[1] for e in extents),
            max(e[2] for e in extents),
            max(e[3] for e in extents),
        )

    def _get_pixel_size_meters(self, tiff_path: str) -> float:
        ds = gdal.Open(tiff_path)
        gt = ds.GetGeoTransform()
        width_units = abs(gt[1])
        height_units = abs(gt[5])

        width = ds.RasterXSize
        height = ds.RasterYSize
        center_y = gt[3] + (width / 2) * gt[4] + (height / 2) * gt[5]

        srs = osr.SpatialReference()
        srs.ImportFromWkt(ds.GetProjection())
        ds = None

        if srs.IsProjected():
            scale = srs.GetLinearUnits()
            return 0.5 * (width_units * scale + height_units * scale)
        else:
            scale = srs.GetAngularUnits()
            lat_rad = center_y * scale
            earth_radius = 6_371_000
            lat_m = earth_radius * scale
            lon_m = lat_m * math.cos(lat_rad)
            return 0.5 * (width_units * lon_m + height_units * lat_m)
