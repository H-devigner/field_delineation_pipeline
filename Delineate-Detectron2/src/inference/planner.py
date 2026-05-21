"""
ExecutionPlanner — Divide raster into processing regions and tile plans.

Adapted from Delineate-Anything/methods/main/ExecutionPlanner.py
Responsibilities:
  - Compute total raster size in pixels (accounting for super-resolution scale)
  - Divide into processing regions (default 4096×4096)
  - For each region, generate a tile plan with overlapping tiles for inference
  - Provide geotransform for each region
"""

import math
import logging
from typing import Dict, List, Optional, Tuple

from osgeo import gdal

logger = logging.getLogger(__name__)


class ExecutionPlanner:
    """Plan how to process a large raster in manageable regions and tiles."""

    def __init__(self, analyser, planner_config: dict):
        """
        Parameters
        ----------
        analyser : DataAnalyser
            Must have called is_compatible() first.
        planner_config : dict
            Keys: region_width, region_height, pixel_offset
        """
        self.analyser = analyser

        # Total raster size in scaled pixels
        bounds = analyser.total_bounds
        self.region_size_full = [
            analyser.scale * int(math.ceil(
                (bounds[2] - bounds[0]) / analyser.pixel_size_x
            )),
            analyser.scale * int(math.ceil(
                (bounds[3] - bounds[1]) / abs(analyser.pixel_size_y)
            )),
        ]

        self.region_size = [
            planner_config.get("region_width", 4096),
            planner_config.get("region_height", 4096),
        ]

        self.pixel_offset = planner_config.get("pixel_offset", [0, 0])
        self.current_region = None

    def get_num_regions(self) -> int:
        """Total number of processing regions."""
        nx = self.region_size_full[0] // self.region_size[0]
        if self.region_size_full[0] % self.region_size[0] != 0:
            nx += 1
        ny = self.region_size_full[1] // self.region_size[1]
        if self.region_size_full[1] % self.region_size[1] != 0:
            ny += 1
        return nx * ny

    def get_geotransform(self) -> Tuple:
        """GDAL geotransform for the current region."""
        minx, _, _, maxy = self.analyser.total_bounds
        pw = self.analyser.pixel_size_x / self.analyser.scale
        ph = self.analyser.pixel_size_y / self.analyser.scale
        return (
            minx + self.current_region[0] * pw,
            pw,
            0,
            maxy + self.current_region[1] * ph,
            0,
            ph,
        )

    def move_to_next_region(self) -> bool:
        """Advance to the next region. Returns False when exhausted."""
        if self.current_region is None:
            self.current_region = [0, 0]
        else:
            self.current_region[0] += self.region_size[0]
            if self.current_region[0] >= self.region_size_full[0]:
                self.current_region[0] = 0
                self.current_region[1] += self.region_size[1]

        return (
            self.current_region[0] < self.region_size_full[0]
            and self.current_region[1] < self.region_size_full[1]
        )

    def get_plan(self, tile_size: Optional[int], tile_step_ratio: float) -> List[Dict]:
        """
        Generate tile plans for the current region.

        Parameters
        ----------
        tile_size : int or None
            Tile size in pixels. None = use analyser.tile_size.
        tile_step_ratio : float
            Overlap ratio, e.g. 0.5 = 50% overlap.

        Returns
        -------
        list of dict
            Each dict describes one tile sweep pattern with fields:
            file, infile_begin, infile_end, infile_size, infile_step,
            inregion_begin, inregion_end, global_begin, global_end, scale
        """
        if self.current_region is None:
            return []

        ts = self.analyser.tile_size if tile_size is None else tile_size
        tile_step = int(ts * tile_step_ratio)
        steps_per_non_overlap = (ts // tile_step) + (1 if ts % tile_step != 0 else 0)

        plans = []
        for tiff in self.analyser.tiffs:
            ds = gdal.Open(tiff)
            ds_size = [ds.RasterXSize, ds.RasterYSize]
            ds_offset = self.analyser.get_pixel_offset(ds)
            ds = None

            left = ds_offset[0]
            right = left + ds_size[0]
            top = ds_offset[1]
            bottom = top + ds_size[1]

            scale = self.analyser.scale
            rl = self.current_region[0] // scale
            rr = (self.current_region[0] + self.region_size[0]) // scale
            rt = self.current_region[1] // scale
            rb = (self.current_region[1] + self.region_size[1]) // scale

            # Skip if no intersection
            if not (left < rr and right > rl and top < rb and bottom > rt):
                continue

            begin_x = ((max(left, rl) - (ts - tile_step)) // tile_step) * tile_step - left
            begin_y = ((max(top, rt) - (ts - tile_step)) // tile_step) * tile_step - top
            end_x = min(right, rr) - left
            end_y = min(bottom, rb) - top

            for i in range(steps_per_non_overlap):
                for j in range(steps_per_non_overlap):
                    bx = begin_x + i * tile_step + self.pixel_offset[0]
                    by = begin_y + j * tile_step + self.pixel_offset[1]
                    ex = end_x + self.pixel_offset[0]
                    ey = end_y + self.pixel_offset[1]

                    if bx >= ex or by >= ey:
                        continue

                    irb_x = self.current_region[0] // scale
                    irb_y = self.current_region[1] // scale

                    plans.append({
                        "file": tiff,
                        "infile_begin": [bx, by],
                        "infile_end": [ex, ey],
                        "infile_size": [ts, ts],
                        "infile_step": [
                            steps_per_non_overlap * tile_step,
                            steps_per_non_overlap * tile_step,
                        ],
                        "inregion_begin": [
                            scale * (begin_x + i * tile_step + left - rl),
                            scale * (begin_y + j * tile_step + top - rt),
                        ],
                        "inregion_end": [
                            scale * (end_x + left - rl),
                            scale * (end_y + top - rt),
                        ],
                        "global_begin": [
                            scale * (begin_x + i * tile_step + left),
                            scale * (begin_y + j * tile_step + top),
                        ],
                        "global_end": [
                            scale * (end_x + left),
                            scale * (end_y + top),
                        ],
                        "scale": scale,
                    })

        return plans
