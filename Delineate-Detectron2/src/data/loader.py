"""
DataLoader — Async, high-throughput GeoTIFF tile loader with LCLU support.

Optimized for 8× H100 + 224 cores + 2TB RAM:
  1. Background thread pre-fetches next region while current is processing
  2. Large tile cache (2TB RAM → entire region + next in memory)
  3. Vectorized normalization with numpy
  4. GDAL block cache tuned for high-throughput reads
  5. LCLU clip/filter masks for land cover integration
"""

import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from osgeo import gdal

logger = logging.getLogger(__name__)


def configure_gdal(cache_mb: int = 4096):
    """Set GDAL global options for high-throughput reads."""
    gdal.SetConfigOption("GDAL_CACHEMAX", str(cache_mb))
    gdal.SetConfigOption("GDAL_NUM_THREADS", "ALL_CPUS")
    gdal.SetConfigOption("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    gdal.SetConfigOption("VSI_CACHE", "TRUE")
    gdal.SetConfigOption("VSI_CACHE_SIZE", str(cache_mb * 1024 * 1024))
    logger.debug(f"GDAL configured: cache={cache_mb}MB, threads=ALL_CPUS")


class DataLoader:
    """
    High-throughput tile loader with LCLU clip/filter support.

    Produces (image, clip_mask, filter_mask, bounds) per tile where:
      - clip_mask: 1=valid, 0=clip (nodata OR LCLU clip classes)
      - filter_mask: 1=valid, 0=filter (nodata OR LCLU filter classes)
    """

    def __init__(self, plan: dict, config: dict, batch_size: int,
                 lclu_path: Optional[str] = None, lclu_config: Optional[dict] = None):
        self.bands = config["bands"]
        self.nodata_value = config.get("nodata_value", [0, 0, 0])
        self.min_vals = config["min"]
        self.max_vals = config["max"]
        self.skip = config.get("skip", False)

        # LCLU config
        self.lclu_path = lclu_path
        self.lclu_range = lclu_config.get("range") if lclu_config else None
        self.lclu_clip_classes = lclu_config.get("clip_classes", []) if lclu_config else []
        self.lclu_filter_classes = lclu_config.get("filter_classes", []) if lclu_config else []

        self.init_plan = plan
        self.plan = plan
        self.batch_size = batch_size

        self.pos = plan["infile_begin"].copy()
        self.offset = plan["infile_begin"].copy()

        self._image_cache = None
        self._clip_cache = None    # nodata + LCLU clip
        self._filter_cache = None  # nodata + LCLU filter
        if not self.skip:
            self._load_image()
            self._load_lclu()

    def is_compatible(self, plan: dict) -> bool:
        if self.skip:
            return True
        if self.init_plan["file"] != plan["file"]:
            return False
        if (self.init_plan["infile_begin"][0] > plan["infile_begin"][0]
                or self.init_plan["infile_begin"][1] > plan["infile_begin"][1]):
            return False
        if (self.init_plan["infile_end"][0] != plan["infile_end"][0]
                or self.init_plan["infile_end"][1] != plan["infile_end"][1]):
            return False
        return True

    def set_plan(self, plan: dict):
        self.plan = plan
        self.pos = plan["infile_begin"].copy()

    def get_batch(self) -> Tuple[Optional[List[np.ndarray]], Optional[List[np.ndarray]], Optional[List[dict]]]:
        """
        Fetch the next batch of tiles.

        Returns
        -------
        (images, nodata_masks, bounds) or (None, None, None) if exhausted.

        nodata_masks here is the clip_mask (1=valid, 0=clip).
        The filter_mask is embedded in bounds["filter_mask"] for use by postprocessor.
        """
        if self.skip:
            return None, None, None

        batch_images = []
        batch_nodata = []
        batch_bounds = []

        tile_size = self.plan["infile_size"]
        tile_step = self.plan["infile_step"]
        infile_begin = self.plan["infile_begin"]
        infile_end = self.plan["infile_end"]
        inregion_begin = self.plan["inregion_begin"]
        global_begin = self.plan["global_begin"]
        scale = self.plan["scale"]

        cache_h, cache_w = self._image_cache.shape[:2]

        while self.pos[1] < infile_end[1]:
            while self.pos[0] < infile_end[0]:
                px, py = self.pos[0], self.pos[1]
                self.pos[0] += tile_step[0]

                bx = px - self.offset[0]
                by = py - self.offset[1]
                ex = bx + tile_size[0]
                ey = by + tile_size[1]

                if bx < 0 or by < 0 or ex > cache_w or ey > cache_h:
                    continue

                clip = self._clip_cache[by:ey, bx:ex]
                if np.all(clip == 0):
                    continue

                image = self._image_cache[by:ey, bx:ex].copy()
                filter_mask = self._filter_cache[by:ey, bx:ex].copy()

                if image.shape[0] != 640 or image.shape[1] != 640:
                    image = cv2.resize(image, (640, 640), interpolation=cv2.INTER_CUBIC)
                    clip = cv2.resize(clip, (640, 640), interpolation=cv2.INTER_NEAREST)
                    filter_mask = cv2.resize(filter_mask, (640, 640), interpolation=cv2.INTER_NEAREST)

                image_bgr = image[:, :, ::-1].copy()

                bounds = {
                    "global": (
                        global_begin[0] + scale * (px - infile_begin[0]),
                        global_begin[1] + scale * (py - infile_begin[1]),
                        scale * tile_size[0], scale * tile_size[1],
                    ),
                    "inregion": (
                        inregion_begin[0] + scale * (px - infile_begin[0]),
                        inregion_begin[1] + scale * (py - infile_begin[1]),
                        scale * tile_size[0], scale * tile_size[1],
                    ),
                    "infile": (px, py, tile_size[0], tile_size[1]),
                    "filename": self.plan["file"],
                    "filter_mask": filter_mask,  # for postprocessor remaining_area calc
                }

                batch_images.append(image_bgr)
                batch_nodata.append(clip)
                batch_bounds.append(bounds)

                if len(batch_images) == self.batch_size:
                    return batch_images, batch_nodata, batch_bounds

            self.pos[1] += tile_step[1]
            self.pos[0] = infile_begin[0]

        if batch_images:
            return batch_images, batch_nodata, batch_bounds
        return None, None, None

    def _load_image(self):
        """Load and normalize the image region into cache (vectorized)."""
        ds = gdal.Open(self.plan["file"], gdal.GF_Read)
        if ds is None:
            raise FileNotFoundError(f"Cannot open: {self.plan['file']}")

        begin = self.plan["infile_begin"]
        end = self.plan["infile_end"]
        end_padded = [end[0] + self.plan["infile_size"][0],
                      end[1] + self.plan["infile_size"][1]]
        size = [end_padded[0] - begin[0], end_padded[1] - begin[1]]

        n_bands = len(self.bands)
        self._image_cache = np.zeros((size[1], size[0], n_bands), dtype=np.uint8)
        self._clip_cache = np.ones((size[1], size[0]), dtype=np.uint8)
        self._filter_cache = np.ones((size[1], size[0]), dtype=np.uint8)

        x_begin = max(-begin[0], 0)
        x_end = min(size[0], ds.RasterXSize - begin[0])
        y_begin = max(-begin[1], 0)
        y_end = min(size[1], ds.RasterYSize - begin[1])

        if x_end <= x_begin or y_end <= y_begin:
            self._clip_cache[:] = 0
            self._filter_cache[:] = 0
            ds = None
            return

        read_w = x_end - x_begin
        read_h = y_end - y_begin
        read_x = max(0, begin[0])
        read_y = max(0, begin[1])

        for i, band_idx in enumerate(self.bands):
            band = ds.GetRasterBand(band_idx)
            data = band.ReadAsArray(read_x, read_y, read_w, read_h)

            mn, mx = self.min_vals[i], self.max_vals[i]
            if mx > mn:
                scale = 255.0 / (mx - mn)
                normalized = np.clip((data - mn) * scale, 0, 255).astype(np.uint8)
            else:
                normalized = np.zeros_like(data, dtype=np.uint8)

            self._image_cache[y_begin:y_end, x_begin:x_end, i] = normalized

            if self.nodata_value is not None and i < len(self.nodata_value):
                is_nodata = (data == self.nodata_value[i]).astype(np.uint8)
                self._clip_cache[y_begin:y_end, x_begin:x_end] &= (1 - is_nodata)

        # clip_mask = 1 means valid. Erode slightly for edge artifacts
        self._clip_cache = cv2.erode(self._clip_cache, np.ones((3, 3), dtype=np.uint8))
        # filter_mask starts as eroded clip (more conservative)
        self._filter_cache = cv2.erode(self._clip_cache, np.ones((5, 5), dtype=np.uint8))

        ds = None
        logger.debug(f"Loaded {self.plan['file']}: {read_w}×{read_h}")

    def _load_lclu(self):
        """Apply LCLU clip/filter classes to the nodata masks."""
        if self.skip or self.lclu_path is None:
            return
        if not self.lclu_clip_classes and not self.lclu_filter_classes:
            return

        lclu_ds = gdal.Open(self.lclu_path, gdal.GF_Read)
        if lclu_ds is None:
            logger.warning(f"Cannot open LCLU: {self.lclu_path}")
            return

        begin = self.plan["infile_begin"]
        end = self.plan["infile_end"]
        end_padded = [end[0] + self.plan["infile_size"][0],
                      end[1] + self.plan["infile_size"][1]]
        size = [end_padded[0] - begin[0], end_padded[1] - begin[1]]

        scale = self.plan.get("scale", 1)
        begin_offset = [v // scale for v in self.plan.get("global_begin", begin)]
        end_offset = [(v + self.plan["infile_size"][0]) // scale
                      for v in self.plan.get("global_end", end)]

        x_begin = max(-begin_offset[0], 0)
        x_end = min(size[0], lclu_ds.RasterXSize - begin_offset[0])
        y_begin = max(-begin_offset[1], 0)
        y_end = min(size[1], lclu_ds.RasterYSize - begin_offset[1])

        if x_end <= x_begin or y_end <= y_begin:
            lclu_ds = None
            return

        lclu_band = lclu_ds.GetRasterBand(1)
        lclu = lclu_band.ReadAsArray(
            max(0, begin_offset[0]), max(0, begin_offset[1]),
            x_end - x_begin, y_end - y_begin,
        )

        if self.lclu_range is not None:
            # Fast LUT approach
            clip_lut = np.ones(self.lclu_range, dtype=np.uint8)
            filter_lut = np.ones(self.lclu_range, dtype=np.uint8)

            for val in self.lclu_clip_classes:
                if val < self.lclu_range:
                    clip_lut[val] = 0
            for val in self.lclu_filter_classes:
                if val < self.lclu_range:
                    filter_lut[val] = 0

            lclu_clipped = np.clip(lclu, 0, self.lclu_range - 1)
            self._clip_cache[y_begin:y_end, x_begin:x_end] &= clip_lut[lclu_clipped]
            self._filter_cache[y_begin:y_end, x_begin:x_end] &= filter_lut[lclu_clipped]
        else:
            # General approach
            for val in self.lclu_clip_classes:
                self._clip_cache[y_begin:y_end, x_begin:x_end] &= (lclu != val).astype(np.uint8)
            for val in self.lclu_filter_classes:
                self._filter_cache[y_begin:y_end, x_begin:x_end] &= (lclu != val).astype(np.uint8)

        lclu_ds = None
        logger.debug(f"LCLU masks applied: clip={self.lclu_clip_classes}, filter={self.lclu_filter_classes}")


class AsyncPrefetchLoader:
    """
    Wraps DataLoader with background prefetch for the next tile plan.

    Usage:
        prefetcher = AsyncPrefetchLoader(config, batch_size)
        prefetcher.submit(plan)        # start loading in background
        loader = prefetcher.get()       # blocks until ready, returns DataLoader
    """

    def __init__(self, config: dict, batch_size: int, max_workers: int = 2,
                 lclu_path: Optional[str] = None, lclu_config: Optional[dict] = None):
        self.config = config
        self.batch_size = batch_size
        self.lclu_path = lclu_path
        self.lclu_config = lclu_config
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="prefetch")
        self._future = None
        self._current_loader = None

    def submit(self, plan: dict):
        self._future = self._executor.submit(self._load, plan)

    def get(self) -> Optional['DataLoader']:
        if self._future is None:
            return None
        loader = self._future.result()
        self._future = None
        self._current_loader = loader
        return loader

    def get_or_create(self, plan: dict) -> 'DataLoader':
        if self._current_loader is not None and self._current_loader.is_compatible(plan):
            self._current_loader.set_plan(plan)
            return self._current_loader

        if self._future is not None:
            loader = self._future.result()
            self._future = None
            if loader.is_compatible(plan):
                loader.set_plan(plan)
                self._current_loader = loader
                return loader

        loader = DataLoader(plan, self.config, self.batch_size,
                            self.lclu_path, self.lclu_config)
        self._current_loader = loader
        return loader

    def _load(self, plan: dict) -> 'DataLoader':
        return DataLoader(plan, self.config, self.batch_size,
                          self.lclu_path, self.lclu_config)

    def shutdown(self):
        self._executor.shutdown(wait=False)
