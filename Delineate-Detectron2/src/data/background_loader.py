"""
BackgroundLoader — Load LCLU (Land Cover Land Use) maps to identify non-field areas.

Ported from Delineate-Anything/methods/main/BackgroundLoader.py.

After all tiles in a region are processed, this module reads the LCLU map
for the region and paints non-field pixels (water, forest, urban) as
*negative* IDs in the instance raster. This allows the polygonizer to:
  - Output background polygons with `bg=1` flag
  - Apply different area thresholds for background vs foreground fields
  - Fill gaps where the model missed detections in known agricultural areas
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
from osgeo import gdal, ogr

logger = logging.getLogger(__name__)

try:
    import rasterio
    from rasterio.warp import reproject, Resampling
    from affine import Affine
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    logger.debug("rasterio not available — LCLU reprojection disabled")


class BackgroundLoader:
    """
    Load LCLU raster and identify background (non-field) areas.

    Background classes from the LCLU map are painted as negative IDs
    in the instance raster, so the polygonizer can distinguish them
    from model-predicted fields (positive IDs).
    """

    def __init__(self, background_config: dict, lclu_path: Optional[str], lclu_range: Optional[int]):
        """
        Parameters
        ----------
        background_config : dict
            Keys:
              background_classes_from_mask: list[int] — LCLU class values to treat as background
              additional_source: [path, layer_name] or null — optional vector source
        lclu_path : str or None
            Path to the warped LCLU GeoTIFF (same CRS and resolution as imagery).
        lclu_range : int or None
            Total number of classes in the LCLU map (for fast LUT lookup).
            If None, uses np.isin (slower but works with any values).
        """
        self.lclu_path = lclu_path
        self.lclu_range = lclu_range

        bg_classes = background_config.get("background_classes_from_mask", [])
        self.lclu_background_classes = bg_classes if bg_classes else None

        self.additional_source = background_config.get("additional_source")

        if self.lclu_background_classes is not None:
            self.offset = int(np.max(self.lclu_background_classes))
        else:
            self.offset = 1

    def get_background(self, geotransform, width: int, height: int, srs_wkt: str) -> Optional[np.ndarray]:
        """
        Load LCLU background for a processing region.

        Parameters
        ----------
        geotransform : tuple
            GDAL-style (originX, pixelW, 0, originY, 0, pixelH)
        width, height : int
            Region size in pixels
        srs_wkt : str
            WKT of the region CRS

        Returns
        -------
        np.ndarray or None
            int32 array (height, width) with LCLU class values for background pixels,
            0 for non-background. Returns None if no LCLU is configured.
        """
        dst_array = None

        if self.lclu_path is not None and self.lclu_background_classes is not None:
            dst_array = self._load_from_raster(geotransform, width, height, srs_wkt)

            if dst_array is not None:
                if self.lclu_range is not None:
                    # Fast LUT approach
                    lut = np.zeros(self.lclu_range, dtype=np.int32)
                    for val in self.lclu_background_classes:
                        if val < self.lclu_range:
                            lut[val] = val
                    # Clip to valid range
                    clipped = np.clip(dst_array, 0, self.lclu_range - 1)
                    dst_array = lut[clipped]
                else:
                    # General approach
                    mask = np.isin(dst_array, self.lclu_background_classes)
                    dst_array = np.where(mask, dst_array, 0)

        # Optional: additional vector source (e.g., previous year's delineation)
        if self.additional_source is not None:
            vector_path = self.additional_source[0]
            dst_vector = self._load_from_vector(vector_path, geotransform, width, height, srs_wkt)
            if dst_vector is not None:
                if dst_array is not None:
                    mask = dst_vector > 0
                    dst_array[mask] = self.offset + dst_vector[mask]
                else:
                    dst_array = dst_vector

        return dst_array

    def _load_from_raster(self, geotransform, width, height, srs_wkt) -> Optional[np.ndarray]:
        """Load LCLU raster, reprojecting if needed."""
        if not HAS_RASTERIO:
            # Fallback: direct GDAL read (assumes same CRS/resolution)
            return self._load_from_raster_gdal(geotransform, width, height)

        gt = Affine.from_gdal(*geotransform)
        crs = rasterio.crs.CRS.from_wkt(srs_wkt)

        dst_array = np.zeros((height, width), dtype=np.int32)
        try:
            with rasterio.open(self.lclu_path) as src:
                reproject(
                    source=rasterio.band(src, 1),
                    destination=dst_array,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=gt,
                    dst_crs=crs,
                    resampling=Resampling.nearest,
                )
        except Exception as e:
            logger.warning(f"LCLU rasterio reproject failed: {e}, trying GDAL")
            return self._load_from_raster_gdal(geotransform, width, height)

        return dst_array

    def _load_from_raster_gdal(self, geotransform, width, height) -> Optional[np.ndarray]:
        """Direct GDAL read (assumes LCLU already matches CRS/resolution)."""
        ds = gdal.Open(self.lclu_path, gdal.GF_Read)
        if ds is None:
            logger.warning(f"Cannot open LCLU: {self.lclu_path}")
            return None

        # Calculate pixel coords from geotransform
        gt = ds.GetGeoTransform()
        ox, pw, _, oy, _, ph = gt
        roi_ox, roi_pw, _, roi_oy, _, roi_ph = geotransform

        # Source pixel coords
        sx = int((roi_ox - ox) / pw)
        sy = int((roi_oy - oy) / ph)

        # Clamp to raster bounds
        rx = max(0, sx)
        ry = max(0, sy)
        rw = min(ds.RasterXSize - rx, width - max(0, -sx))
        rh = min(ds.RasterYSize - ry, height - max(0, -sy))

        if rw <= 0 or rh <= 0:
            ds = None
            return None

        band = ds.GetRasterBand(1)
        data = band.ReadAsArray(rx, ry, rw, rh)

        result = np.zeros((height, width), dtype=np.int32)
        dx = max(0, -sx)
        dy = max(0, -sy)
        result[dy:dy + rh, dx:dx + rw] = data

        ds = None
        return result

    @staticmethod
    def _load_from_vector(vector_path, geotransform, width, height, srs_wkt) -> Optional[np.ndarray]:
        """Rasterize a vector source (e.g., previous delineation)."""
        try:
            mem_drv = gdal.GetDriverByName("MEM")
            dst_ds = mem_drv.Create("", width, height, 1, gdal.GDT_Int32)
            dst_ds.SetGeoTransform(geotransform)
            dst_ds.SetProjection(srs_wkt)

            src_ds = ogr.Open(vector_path)
            if src_ds is None:
                return None
            layer = src_ds.GetLayer()

            gdal.RasterizeLayer(dst_ds, [1], layer, options=["ATTRIBUTE=id"])
            arr = dst_ds.ReadAsArray()

            dst_ds = None
            src_ds = None
            return np.abs(arr)
        except Exception as e:
            logger.warning(f"Vector background load failed: {e}")
            return None


def warp_lclu(
    src_path: str,
    dst_path: str,
    sample_tiff: str,
    total_bounds: Tuple[float, float, float, float],
    pixel_size: Tuple[float, float],
) -> Optional[str]:
    """
    Reproject and resample an LCLU raster to match the imagery.

    Parameters
    ----------
    src_path : str or None
        Path to the source LCLU raster (any CRS).
    dst_path : str
        Output path for the warped LCLU GeoTIFF.
    sample_tiff : str
        A sample GeoTIFF from the imagery (for target CRS).
    total_bounds : (minx, miny, maxx, maxy)
        Bounding box in target CRS.
    pixel_size : (px, py)
        Target pixel size.

    Returns
    -------
    str or None
        Path to the warped LCLU, or None if no source provided.
    """
    import math

    if src_path is None:
        return None

    # Reuse existing warp if it exists
    import os
    if os.path.exists(dst_path):
        logger.info(f"Using existing warped LCLU: {dst_path}")
        return dst_path

    sample_ds = gdal.Open(sample_tiff)
    if sample_ds is None:
        logger.error(f"Cannot open sample TIFF: {sample_tiff}")
        return None
    target_proj = sample_ds.GetProjection()
    sample_ds = None

    minx, miny, maxx, maxy = total_bounds
    cols = int(math.ceil((maxx - minx) / pixel_size[0]))
    rows = int(math.ceil((maxy - miny) / abs(pixel_size[1])))

    logger.info(f"Warping LCLU: {src_path} → {dst_path} ({cols}×{rows})")

    try:
        gdal.Warp(
            dst_path,
            src_path,
            format="GTiff",
            dstSRS=target_proj,
            outputBounds=total_bounds,
            width=cols,
            height=rows,
            resampleAlg="nearest",
            creationOptions=[
                "BIGTIFF=YES",
                "COMPRESS=ZSTD",
                "ZSTD_LEVEL=2",
                "TILED=YES",
                "NUM_THREADS=ALL_CPUS",
            ],
        )
        logger.info("LCLU warp complete")
        return dst_path
    except Exception as e:
        logger.error(f"LCLU warp failed: {e}")
        return None
