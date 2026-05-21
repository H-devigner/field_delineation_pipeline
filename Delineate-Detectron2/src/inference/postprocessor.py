"""
PostProcessor — Production-grade mask merging and deduplication.

Enhanced from the initial implementation with the full set of techniques
from Delineate-Anything:

  1. **Edge-aware merging** — Tiles mark border pixels with odd IDs (|1).
     `find_edge_mapping()` uses bit-manipulation to detect when the same
     real-world field appears under different IDs across tile boundaries.

  2. **Union-Find ID mapping** — `IncrementalFastMapper` globally tracks
     which IDs should merge. After all tiles are processed, `map()` remaps
     the instance raster in a single vectorized op.

  3. **Multi-criteria merge** — IOU merging + edge merging + relative-area
     merging + asymmetric merging  to handle fields split across 2-4 tiles.

  4. **Two-pass compose** — First pass resolves intra-tile overlaps.
     Second pass re-writes the raster with final IDs in area order.

  5. **Remaining-area filter** — Discards predictions mostly in nodata regions.
"""

import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy import ndimage

from .id_mapper import IncrementalFastMapper

logger = logging.getLogger(__name__)

# Default config values matching the reference project
_DEFAULTS = {
    "pixel_area_threshold": 512,
    "remaining_area_threshold": 0.8,
    "compose_merge_iou": 0.8,
    "merge_iou": 0.8,
    "merge_edge_iou": 0.2,
    "merge_edge_pixels": 32,
    "merge_relative_area_threshold": 0.5,
    "merge_asymetric_pixel_area_threshold": 64,
    "merge_asymetric_relative_area_threshold": 0.5,
    "merging_edge_width": 4,
}


class PostProcessor:
    """Production-grade instance mask composer and merger for a processing region."""

    def __init__(self, region_size: Tuple[int, int], config: dict):
        """
        Parameters
        ----------
        region_size : (width, height)
            Size of the processing region in pixels.
        config : dict
            Merge thresholds (see _DEFAULTS for all keys).
        """
        self.region_w, self.region_h = region_size
        self.config = {**_DEFAULTS, **config}

        # Shared rasters for the region
        self.instances = np.zeros((self.region_h, self.region_w), dtype=np.int32)
        self.weights = np.zeros((self.region_h, self.region_w), dtype=np.float32)

        # Global ID tracking
        self.area_dict: Dict[int, float] = {}
        self.mapping_dict: Dict[int, List[int]] = {}
        self._next_id = 2  # 0=bg, 1=reserved for edge marker
        self._id_increment = 2  # step by 2: even=interior, odd=edge

        # Union-Find for cross-tile ID merging
        self.id_mapper = IncrementalFastMapper(10_000_000)

    def clear(self):
        """Reset for the next processing region."""
        self.instances[:] = 0
        self.weights[:] = 0
        self.area_dict.clear()
        self.mapping_dict.clear()
        self._next_id = 2
        self.id_mapper.reset()

    def apply_background(self, background: np.ndarray):
        """
        Paint LCLU background classes as negative IDs in the instance raster.

        Pixels where `background > 0` AND `instances == 0` (no model prediction)
        are set to `-background[pixel]`. This allows the polygonizer to output
        background polygons with `bg=1` flag and apply different area thresholds.

        Parameters
        ----------
        background : np.ndarray, int32 (height, width)
            LCLU class values for background pixels (0 = not background).
        """
        if background is None:
            return
        mask = (background > 0) & (self.instances == 0)
        self.instances[mask] = -background[mask]

    def process_tile(
        self,
        predictions: Dict,
        nodata_mask: np.ndarray,
        bounds: dict,
    ):
        """
        Process one tile's inference results and write into the region raster.

        This follows the reference project's two-pass approach:
          1. Clean masks → compose on 512×512 local raster with IOU merging
          2. Edge-mark → resize → write into region with weight-based overlap resolution
          3. Run find_edge_mapping for cross-tile merging

        Parameters
        ----------
        predictions : dict
            Keys: masks (N,H,W uint8), scores (N,), boxes (N,4)
        nodata_mask : (H,W) uint8, 1=valid, 0=nodata
        bounds : dict with 'inregion' key
        """
        masks = predictions["masks"]
        scores = predictions["scores"]
        n = len(scores)

        if n == 0:
            return

        MIN_AREA = self.config["pixel_area_threshold"]
        MIN_REL_AREA = self.config["remaining_area_threshold"]
        COMPOSE_MERGE_IOU = self.config["compose_merge_iou"]
        EDGE_WIDTH = self.config["merging_edge_width"]

        tile_h, tile_w = masks.shape[1], masks.shape[2]

        # Prepare nodata masks (clip = hard boundary, filter = for relative area calc)
        nodata_resized = nodata_mask
        if nodata_resized.shape[0] != tile_h or nodata_resized.shape[1] != tile_w:
            nodata_resized = cv2.resize(nodata_mask, (tile_w, tile_h),
                                        interpolation=cv2.INTER_NEAREST)
        clip_mask = nodata_resized
        # Filter mask: eroded version for more conservative area calculation
        k = np.ones((5, 5), dtype=np.uint8)
        filter_mask = cv2.erode(nodata_resized, k)

        # ── Pass 1: Clean individual masks ───────────────────
        fields = []
        bboxes = []
        ids = []
        areas = []
        rel_areas = []

        for i in range(n):
            # Extract bbox with 4px padding
            box = predictions["boxes"][i]
            min_x = max(0, int(box[0]) - 4)
            min_y = max(0, int(box[1]) - 4)
            max_x = min(tile_w, int(np.ceil(box[2])) + 4)
            max_y = min(tile_h, int(np.ceil(box[3])) + 4)

            mask_crop = masks[i, min_y:max_y, min_x:max_x].copy()
            clip_crop = clip_mask[min_y:max_y, min_x:max_x]
            filter_crop = filter_mask[min_y:max_y, min_x:max_x]

            # Largest connected component + clip to valid data
            mask_crop = self._get_biggest_component(mask_crop)
            mask_crop = (mask_crop > 0) & (clip_crop > 0)
            mask_crop = mask_crop.astype(np.uint8)

            initial_area = int(mask_crop.sum())
            valid_area = int((mask_crop * (filter_crop > 0)).sum())

            if initial_area == 0:
                continue

            rel_area = valid_area / initial_area

            fid = self._next_id
            self._next_id += self._id_increment

            fields.append(mask_crop)
            bboxes.append([min_x, max_x, min_y, max_y])
            ids.append(fid)
            areas.append(initial_area)
            rel_areas.append(rel_area)

        if not fields:
            return

        # ── Pass 2: Compose on local tile raster ─────────────
        order = np.argsort(np.array(areas))[::-1]
        instances = np.zeros((tile_h, tile_w), dtype=np.int32)
        weights_local = np.zeros((tile_h, tile_w), dtype=np.float32)

        write_id = list(ids)
        id_area = {ids[i]: areas[i] for i in range(len(fields))}

        for index in order:
            area = areas[index]
            rel_area = rel_areas[index]

            if area < MIN_AREA or rel_area < MIN_REL_AREA:
                write_id[index] = 0
                continue

            index_id = ids[index]
            min_x, max_x, min_y, max_y = bboxes[index]
            field = fields[index]

            # Check intersection with existing composed fields
            existing = instances[min_y:max_y, min_x:max_x][field > 0]
            uniq, counts = np.unique(existing, return_counts=True)
            inter_dict = dict(zip(uniq, counts))

            for key_id in inter_dict:
                if key_id == 0:
                    continue
                area_inter = inter_dict[key_id]
                area_key = id_area.get(key_id, 0)
                if area_key < MIN_AREA:
                    continue
                iou = area_inter / max(1.0, float(area + area_key - area_inter))

                if iou > COMPOSE_MERGE_IOU:
                    # Merge: paint current field with the existing ID
                    id_area[key_id] += area - area_inter
                    id_area[index_id] = 0
                    instances[min_y:max_y, min_x:max_x][field > 0] = key_id
                    write_id[index] = key_id
                    break

            if id_area.get(index_id, 0) >= MIN_AREA:
                instances[min_y:max_y, min_x:max_x][field > 0] = index_id
                for key_id in inter_dict:
                    if key_id > 0:
                        id_area[key_id] = max(0, id_area.get(key_id, 0) - inter_dict[key_id])

        # ── Re-compose in clean raster with final IDs ────────
        instances[:] = 0
        for index in order:
            fid = write_id[index]
            if fid == 0:
                continue
            area = id_area.get(fid, 0)
            if area < MIN_AREA:
                continue

            min_x, max_x, min_y, max_y = bboxes[index]
            field = fields[index]
            instances[min_y:max_y, min_x:max_x][field > 0] = fid
            weights_local[min_y:max_y, min_x:max_x][field > 0] = 1.0 / max(1, area)

            # Track in global area dict (both even and odd variants)
            self.area_dict[fid] = area
            self.area_dict[fid | 1] = area
            self.mapping_dict[fid | 1] = [int(fid)]

        # ── Mark edges (odd bit) for cross-tile merging ──────
        instances[:EDGE_WIDTH, :] |= 1
        instances[-EDGE_WIDTH:, :] |= 1
        instances[:, :EDGE_WIDTH] |= 1
        instances[:, -EDGE_WIDTH:] |= 1

        # ── Resize to region coordinate space ────────────────
        irx, iry, irw, irh = bounds["inregion"]
        instances_resized = cv2.resize(instances, (irw, irh), interpolation=cv2.INTER_NEAREST)
        weights_resized = cv2.resize(weights_local, (irw, irh), interpolation=cv2.INTER_NEAREST)

        # Clamp to region bounds
        px_begin = max(0, irx)
        px_end = min(self.region_w, irx + irw)
        py_begin = max(0, iry)
        py_end = min(self.region_h, iry + irh)

        ipx_begin = px_begin - irx
        ipx_end = px_end - irx
        ipy_begin = py_begin - iry
        ipy_end = py_end - iry

        if px_begin >= px_end or py_begin >= py_end:
            return

        old_instances = self.instances[py_begin:py_end, px_begin:px_end]
        old_weights = self.weights[py_begin:py_end, px_begin:px_end]
        new_instances = instances_resized[ipy_begin:ipy_end, ipx_begin:ipx_end]
        new_weights = weights_resized[ipy_begin:ipy_end, ipx_begin:ipx_end]

        # ── Edge-aware cross-tile merging ─────────────────────
        self._find_edge_mapping(
            old_instances, new_instances,
            self.mapping_dict, self.area_dict,
        )

        # ── Weight-based overlap resolution ──────────────────
        write_mask = new_weights > old_weights
        old_instances[write_mask] = new_instances[write_mask]
        old_weights[write_mask] = new_weights[write_mask]

    def finalize_region(self):
        """
        Finalize the region after all tiles are processed.

        1. Apply union-find ID mapping (resolve cross-tile merges)
        2. Morphological ID opening (clean field boundaries)
        """
        # Apply union-find mapping
        for key, val in self.mapping_dict.items():
            val_plus_key = list(val) + [int(key)]
            self.id_mapper.union(val_plus_key)

        npmap = np.array(self.id_mapper.finalize(), dtype=np.int32)

        # Remap: ensure all referenced IDs are within bounds
        max_id = int(self.instances.max())
        if max_id >= len(npmap):
            extended = np.arange(max_id + 1, dtype=np.int32)
            extended[:len(npmap)] = npmap
            npmap = extended

        self.instances[:] = npmap[self.instances]

        # Morphological cleanup
        self._id_opening(self.instances)

    def _find_edge_mapping(
        self,
        current: np.ndarray,
        new: np.ndarray,
        dst: dict,
        area_dict: dict,
    ):
        """
        Detect fields that should merge across tile boundaries.

        Uses the edge-marking technique from Delineate-Anything:
        IDs are even for interior pixels, odd for edge pixels.
        When two different IDs overlap in edge zones, they are candidates
        for merging based on IOU, edge-overlap, and relative-area criteria.

        Parameters
        ----------
        current : np.ndarray, int32 — existing region raster fragment
        new : np.ndarray, int32 — incoming tile raster fragment
        dst : dict — merge mapping dict to update
        area_dict : dict — global area tracker
        """
        MERGE_IOU = self.config["merge_iou"]
        MERGE_EDGE_IOU = self.config["merge_edge_iou"]
        MERGE_EDGE_PIXELS = self.config["merge_edge_pixels"]
        MERGE_REL_AREA = self.config["merge_relative_area_threshold"]
        MERGE_ASYM_PIXELS = self.config["merge_asymetric_pixel_area_threshold"]
        MERGE_ASYM_REL = self.config["merge_asymetric_relative_area_threshold"]

        # Build intersection counts: encode (current_id, new_id) as uint64
        combined = (current.astype(np.uint64) << np.uint64(32)) | new.astype(np.uint64)
        uniq_all, counts_all = np.unique(combined, return_counts=True)
        inter_dict = dict(zip(uniq_all, counts_all))

        # Per-raster unique counts (for local relative-area)
        uniq_c, cnt_c = np.unique(current, return_counts=True)
        local_current = dict(zip(uniq_c, cnt_c))

        uniq_n, cnt_n = np.unique(new, return_counts=True)
        local_new = dict(zip(uniq_n, cnt_n))

        if not inter_dict:
            return

        for i, key in enumerate(uniq_all):
            key = np.uint64(key)
            key_current = int(key >> np.uint64(32))
            key_new = int(key & np.uint64(0xFFFFFFFF))

            if key_current < 2 or key_new < 2:
                continue

            # Compute base IDs (even) and edge variants (odd)
            kc_0 = np.uint64(2 * (key_current // 2))
            kc_1 = kc_0 | np.uint64(1)
            kn_0 = np.uint64(2 * (key_new // 2))
            kn_1 = kn_0 | np.uint64(1)

            # Sum intersection across all 4 edge/interior combinations
            keys_4 = [
                (kc_0 << np.uint64(32)) | kn_0,
                (kc_0 << np.uint64(32)) | kn_1,
                (kc_1 << np.uint64(32)) | kn_0,
                (kc_1 << np.uint64(32)) | kn_1,
            ]
            area_intersect = sum(inter_dict.get(k, 0) for k in keys_4)

            area_current = area_dict.get(int(key_current), 0)
            area_new = area_dict.get(int(key_new), 0)

            if area_current == 0 or area_new == 0:
                continue

            iou = area_intersect / max(1, area_current + area_new - area_intersect)

            # Case 0: IOU or edge-based merge
            is_edge_merge = (
                (key_current % 2 == 1 or key_new % 2 == 1)
                and (iou > MERGE_EDGE_IOU or area_intersect > MERGE_EDGE_PIXELS)
            )
            is_iou_merge = iou > MERGE_IOU
            case_0 = is_edge_merge or is_iou_merge

            # Case 1: Relative-area or asymmetric merge
            area_c_local = local_current.get(kc_0, 0) + local_current.get(kc_1, 0)
            area_n_local = local_new.get(kn_0, 0) + local_new.get(kn_1, 0)

            if area_c_local == 0 or area_n_local == 0:
                continue

            rel_c = area_intersect / area_c_local
            rel_n = area_intersect / area_n_local

            is_rel_merge = rel_c > MERGE_REL_AREA and rel_n > MERGE_REL_AREA
            is_asym_c = area_intersect > MERGE_ASYM_PIXELS and rel_c > MERGE_ASYM_REL
            is_asym_n = area_intersect > MERGE_ASYM_PIXELS and rel_n > MERGE_ASYM_REL
            case_1 = is_rel_merge or is_asym_c or is_asym_n

            if case_0 and case_1:
                dst[int(key_new)] = dst.get(int(key_new), []) + [int(key_current)]

    @staticmethod
    def _get_biggest_component(mask: np.ndarray) -> np.ndarray:
        """Keep only the largest connected component."""
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return mask

        contours = [c.squeeze(1) if c.ndim == 3 else c for c in contours]
        areas = np.array([cv2.contourArea(c) for c in contours])

        result = np.zeros_like(mask, dtype=np.uint8)
        if len(areas) == 0:
            return result

        biggest = contours[np.argmax(areas)]
        cv2.fillPoly(result, [biggest.astype(np.int32)], 1, cv2.LINE_4)
        return result

    @staticmethod
    def _id_opening(data: np.ndarray):
        """
        Morphological opening on instance IDs to clean field boundaries.

        Processes in 2048×2048 chunks with 2px halo to handle large rasters.
        Steps: nullify borders → fill gaps → remove overgrowth.
        """
        kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
        chunk = 2048
        halo = 2

        for i in range(0, data.shape[0], chunk):
            for j in range(0, data.shape[1], chunk):
                i0 = max(0, i - halo)
                i1 = min(data.shape[0], i + chunk + halo)
                j0 = max(0, j - halo)
                j1 = min(data.shape[1], j + chunk + halo)

                frag = data[i0:i1, j0:j1].copy()

                # 1. Nullify border pixels between different IDs
                mx = ndimage.maximum_filter(frag, footprint=kernel)
                mn = ndimage.minimum_filter(frag, footprint=kernel)
                frag[mx != mn] = 0

                # 2-3. Fill gaps (two passes for wider borders)
                mx = ndimage.maximum_filter(frag, footprint=kernel)
                frag[frag == 0] = mx[frag == 0]
                mx = ndimage.maximum_filter(frag, footprint=kernel)
                frag[frag == 0] = mx[frag == 0]

                # 4. Remove overgrowth
                zero_mask = (frag == 0).astype(np.uint8)
                expanded = cv2.dilate(
                    zero_mask, kernel.astype(np.uint8),
                    borderType=cv2.BORDER_REPLICATE,
                )
                frag[expanded > 0] = 0

                # Write back (skip halo)
                hs = halo if i > 0 else 0
                ws = halo if j > 0 else 0
                ah = min(chunk, data.shape[0] - i)
                aw = min(chunk, data.shape[1] - j)
                data[i:i + ah, j:j + aw] = frag[hs:hs + ah, ws:ws + aw]
