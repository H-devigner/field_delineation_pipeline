"""
infer.py — CLI for field delineation inference with LCLU support.

Features:
  - Multi-GPU parallel inference via ThreadPoolExecutor
  - LCLU (Land Cover Land Use) map integration:
    • Clip/filter masks in data loader
    • Background painting in post-processor
    • Background-aware polygonization
  - Per-stage timing breakdown
  - Async data prefetch

Usage:
    python scripts/infer.py -c configs/inference.yaml -i data/raw/folder -o data/output/fields.gpkg
    python scripts/infer.py -c configs/inference.yaml -i data/raw/folder -o out.gpkg --lclu /path/to/lclu.tif
    python scripts/infer.py -b configs/batch.yaml --verbose
"""

import os
import sys
import time
from argparse import ArgumentParser
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.analyser import DataAnalyser
from src.data.loader import DataLoader, AsyncPrefetchLoader, configure_gdal
from src.data.background_loader import BackgroundLoader, warp_lclu
from src.inference.planner import ExecutionPlanner
from src.inference.predictor import Detectron2Predictor
from src.inference.postprocessor import PostProcessor
from src.inference.polygonizer import Polygonizer
from src.inference.filtering import postdelineation_merge, simplify_geopackage
from src.export.geojson import export_geojson
from src.monitoring.logger import configure_root_logger, get_logger
from src.monitoring.metrics import PipelineMetrics
from src.monitoring.health import HealthChecker


def load_config(config_path: str) -> dict:
    config = yaml.safe_load(Path(config_path).read_text())
    if "base_config" in config and config["base_config"]:
        base = yaml.safe_load(Path(config["base_config"]).read_text())
        return _deep_merge(base, config)
    return config


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


class PipelineTimer:
    """Track time spent in each pipeline stage."""

    def __init__(self):
        self.timers = defaultdict(float)
        self._start = None
        self._name = None

    def start(self, name: str):
        self._name = name
        self._start = time.perf_counter()

    def stop(self):
        if self._start is not None:
            self.timers[self._name] += time.perf_counter() - self._start
            self._start = None

    def summary(self) -> str:
        total = sum(self.timers.values())
        lines = []
        for name, dur in sorted(self.timers.items(), key=lambda x: -x[1]):
            pct = 100 * dur / max(total, 0.001)
            lines.append(f"  {name:25s} {dur:8.1f}s ({pct:5.1f}%)")
        lines.append(f"  {'TOTAL':25s} {total:8.1f}s")
        return "\n".join(lines)


def delineate_folder(config: dict, src_folder: str, output_path: str,
                     lclu_override: str = None, verbose: bool = False):
    """Run the full delineation pipeline on a folder of GeoTIFFs."""
    logger = get_logger("infer")
    perf = config.get("performance", {})
    timer = PipelineTimer()

    # ── Configure GDAL ───────────────────────────────────────
    configure_gdal(cache_mb=perf.get("gdal_cache_mb", 4096))

    # ── Monitoring ───────────────────────────────────────────
    mon_config = config.get("monitoring", {})
    metrics = PipelineMetrics(
        enable=mon_config.get("enable_metrics", True),
        pushgateway_url=mon_config.get("prometheus_pushgateway"),
    )

    # ── Health check ─────────────────────────────────────────
    hc = HealthChecker(output_dir=os.path.dirname(output_path) or ".")
    model_path = config.get("model", {}).get("model_weights")
    health = hc.full_check(model_path)
    logger.info("Health check", extra={"health": health})
    if not health["healthy"]:
        logger.error("Health check failed", extra={"details": health})
        return

    t_start = time.time()

    # ── 1. Discover TIFFs ────────────────────────────────────
    tiffs = sorted([
        os.path.join(src_folder, f)
        for f in os.listdir(src_folder)
        if f.lower().endswith((".tif", ".tiff"))
    ])
    if not tiffs:
        logger.error(f"No TIFF files found in: {src_folder}")
        return
    logger.info(f"Found {len(tiffs)} TIFF(s)")

    # ── 2. Analyse ───────────────────────────────────────────
    dl_config = config.get("data_loader", {})
    bands = dl_config.get("bands", [1, 2, 3])
    analyser = DataAnalyser(tiffs, bands, config.get("super_resolution"))

    if not analyser.is_compatible():
        logger.error("Incompatible TIFFs")
        return

    analyser.calc_normalization_bounds()
    dl_config["min"] = analyser.min_vals
    dl_config["max"] = analyser.max_vals

    # ── 3. LCLU auto-discovery and warp ────────────────────────
    lclu_config = config.get("mask_info")
    bg_config = config.get("background_info", {})
    warped_lclu = None
    bg_loader = None

    # Priority: CLI --lclu > config mask_filepath > mask_root auto-discovery
    lclu_path = lclu_override or config.get("mask_filepath")

    if not lclu_path:
        # Auto-discover: look for <mask_root>/<folder_name>.tif
        mask_root = config.get("mask_root")
        if mask_root:
            folder_name = os.path.basename(os.path.normpath(src_folder))
            candidates = [
                os.path.join(mask_root, folder_name + ".tif"),
                os.path.join(mask_root, folder_name + ".tiff"),
                os.path.join(mask_root, folder_name, folder_name + ".tif"),
                os.path.join(mask_root, folder_name, "lclu.tif"),
            ]
            for cand in candidates:
                if os.path.exists(cand):
                    lclu_path = cand
                    logger.info(f"Auto-discovered LCLU mask: {lclu_path}")
                    break
            if not lclu_path:
                logger.debug(f"No LCLU mask found for '{folder_name}' in {mask_root}")

    if lclu_path and os.path.exists(lclu_path):
        timer.start("lclu_warp")
        temp_dir = os.path.join(os.path.dirname(output_path) or ".", "temp")
        os.makedirs(temp_dir, exist_ok=True)
        folder_name = os.path.basename(os.path.normpath(src_folder))
        dst_lclu = os.path.join(temp_dir, folder_name + ".lclu.tif")

        warped_lclu = warp_lclu(
            lclu_path, dst_lclu, tiffs[0],
            analyser.total_bounds,
            (analyser.pixel_size_x, analyser.pixel_size_y),
        )
        timer.stop()

        if warped_lclu:
            logger.info(f"LCLU warped: {warped_lclu}")
            lclu_range = lclu_config.get("range") if lclu_config else None
            bg_loader = BackgroundLoader(bg_config, warped_lclu, lclu_range)
        else:
            logger.warning("LCLU warp failed — continuing without LCLU")
    elif lclu_path:
        logger.warning(f"LCLU file not found: {lclu_path} — continuing without LCLU")

    # ── 4. Plan ──────────────────────────────────────────────
    planner = ExecutionPlanner(analyser, config.get("execution_planner", {}))
    num_regions = planner.get_num_regions()
    logger.info(f"Regions: {num_regions}, region_size: {planner.region_size}")

    # ── 5. Load model (multi-GPU) ────────────────────────────
    model_config = config["model"]
    model_config["performance"] = perf
    weights = model_config.get("model_weights", "")
    if not os.path.isabs(weights) and not os.path.exists(weights):
        alt = os.path.join(os.path.dirname(src_folder), weights)
        if os.path.exists(alt):
            model_config["model_weights"] = alt

    t_model = time.time()
    predictor = Detectron2Predictor(model_config)
    logger.info(f"Model loaded in {time.time() - t_model:.1f}s ({predictor.num_gpus} GPUs)")

    # ── 6. Prepare output ────────────────────────────────────
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    layer_name = config.get("polygonization", {}).get("layer_name", "fields")

    if os.path.exists(output_path) and config.get("polygonization", {}).get("override_if_exists", True):
        os.remove(output_path)

    polygonizer = Polygonizer(analyser.projection, config.get("filtering", {}))

    # ── 7. Delineate (pipelined with LCLU) ───────────────────
    pass_config = config.get("passes", [{}])[0]
    batch_size = pass_config.get("batch_size", 64)
    tile_step = pass_config.get("tile_step", 0.5)
    tile_size = pass_config.get("tile_size")
    delin_config = pass_config.get("delineation_config", {})

    postproc = PostProcessor(planner.region_size, delin_config)
    prefetcher = AsyncPrefetchLoader(
        dl_config, batch_size,
        max_workers=perf.get("prefetch_regions", 2),
        lclu_path=warped_lclu,
        lclu_config=lclu_config,
    )

    total_polygons = 0
    total_tiles = 0
    total_batches = 0

    with tqdm(total=num_regions, desc="Delineating", unit="region") as pbar:
        while planner.move_to_next_region():
            with metrics.time_region():
                postproc.clear()
                plans = planner.get_plan(tile_size, tile_step)

                if plans:
                    prefetcher.submit(plans[0])

                for plan_idx, plan in enumerate(plans):
                    timer.start("data_load")
                    dataloader = prefetcher.get_or_create(plan)
                    timer.stop()

                    if plan_idx + 1 < len(plans):
                        next_plan = plans[plan_idx + 1]
                        if not dataloader.is_compatible(next_plan):
                            prefetcher.submit(next_plan)

                    while True:
                        timer.start("data_load")
                        images, nodata_masks, bounds = dataloader.get_batch()
                        timer.stop()

                        if images is None:
                            break

                        # GPU inference (parallel across GPUs)
                        timer.start("inference")
                        results = predictor.predict_batch(images)
                        timer.stop()

                        # Post-processing
                        timer.start("postprocess")
                        for i in range(len(images)):
                            if results[i]["masks"].shape[0] > 0:
                                postproc.process_tile(results[i], nodata_masks[i], bounds[i])
                            total_tiles += 1
                        timer.stop()

                        total_batches += 1
                        if metrics.enabled:
                            metrics.tiles_processed.inc(len(images))

                # Apply LCLU background (after all tiles, before polygonization)
                if bg_loader is not None:
                    timer.start("lclu_background")
                    gt = planner.get_geotransform()
                    bg = bg_loader.get_background(
                        gt, planner.region_size[0], planner.region_size[1],
                        analyser.projection,
                    )
                    postproc.apply_background(bg)
                    timer.stop()

                timer.start("finalize")
                postproc.finalize_region()
                timer.stop()

                timer.start("polygonize")
                gt = planner.get_geotransform()
                region_polys = polygonizer.polygonize_region(postproc.instances, gt)
                timer.stop()

                if region_polys:
                    timer.start("write_gpkg")
                    polygonizer.write_to_geopackage(region_polys, output_path, layer_name)
                    timer.stop()
                    total_polygons += len(region_polys)
                    if metrics.enabled:
                        metrics.polygons_created.inc(len(region_polys))

                if metrics.enabled:
                    metrics.regions_completed.inc()
                    metrics.update_gpu_memory()

            pbar.update(1)
            pbar.set_postfix(polys=total_polygons, tiles=total_tiles)

    t_delineate = time.time() - t_start
    tiles_per_sec = total_tiles / max(1, t_delineate)

    logger.info(
        f"Delineation done: {total_polygons} polygons, {total_tiles} tiles "
        f"in {t_delineate:.1f}s ({tiles_per_sec:.1f} tiles/s, {total_batches} batches)"
    )
    logger.info(f"Stage timing breakdown:\n{timer.summary()}")

    # ── 8. Post-delineation merge ────────────────────────────
    t_merge = time.time()
    filter_config = config.get("filtering", {})
    postdelineation_merge(
        output_path, layer_name,
        min_area_m2=filter_config.get("minimum_area_m2", 2500),
        min_hole_area_m2=filter_config.get("minimum_hole_area_m2", 2500),
    )
    logger.info(f"Post-merge: {time.time() - t_merge:.1f}s")

    # ── 9. Simplification ────────────────────────────────────
    simp = config.get("simplification", {})
    if simp.get("simplify", False):
        simplify_geopackage(output_path, layer_name, tolerance=simp.get("tolerance", 1.0))

    # ── 10. Export GeoJSON ───────────────────────────────────
    out_fmt = config.get("output", {}).get("format", "both")
    if out_fmt in ("geojson", "both"):
        geojson_path = export_geojson(output_path, layer_name)
        logger.info(f"GeoJSON exported", extra={"path": geojson_path})

    # ── 11. Summary ──────────────────────────────────────────
    total_time = time.time() - t_start
    logger.info("Pipeline complete", extra={
        "total_seconds": round(total_time, 1),
        "tiles_per_second": round(tiles_per_sec, 1),
        "gpus_used": predictor.num_gpus,
        "lclu_enabled": warped_lclu is not None,
        "output": output_path,
    })
    metrics.push()
    prefetcher.shutdown()
    predictor.shutdown()


def batch_delineate(batch_config_path: str, verbose: bool = False):
    logger = get_logger("infer")
    batch = yaml.safe_load(Path(batch_config_path).read_text())
    base_config = load_config(batch.get("base_config", "configs/inference.yaml"))

    data_root = batch["data_root"]
    output_root = batch["output_root"]
    mask_root = batch.get("mask_root")
    overrides = batch.get("override") or []
    include = batch.get("include")
    exclude = batch.get("exclude")

    # Build per-entry mask override map
    mask_overrides = {}
    for ov in overrides:
        if isinstance(ov, dict) and "entry" in ov and "mask" in ov:
            mask_overrides[ov["entry"]] = ov["mask"]

    os.makedirs(output_root, exist_ok=True)

    folders = sorted([
        f for f in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, f))
    ])

    if include:
        folders = [f for f in folders if f in include]
    if exclude:
        folders = [f for f in folders if f not in exclude]

    logger.info(f"Batch: {len(folders)} folder(s), mask_root={mask_root}")

    for folder in folders:
        config = deepcopy(base_config)
        src = os.path.join(data_root, folder)
        out = os.path.join(output_root, folder + ".gpkg")

        # Set mask_root so auto-discovery works per folder
        if mask_root and "mask_root" not in config:
            config["mask_root"] = mask_root

        # Per-entry mask override
        lclu_override = mask_overrides.get(folder)

        logger.info(f"{'='*60}")
        logger.info(f"Processing: {folder}")

        try:
            delineate_folder(config, src, out, lclu_override=lclu_override, verbose=verbose)
        except Exception as e:
            logger.error(f"Failed: {folder}", extra={"error": str(e)}, exc_info=True)
            continue


def main():
    parser = ArgumentParser(description="Field Delineation — Inference (Multi-GPU + LCLU)")
    parser.add_argument("-c", "--config", default="configs/inference.yaml")
    parser.add_argument("-i", "--input", default=None, help="Input folder of GeoTIFFs")
    parser.add_argument("-o", "--output", default=None, help="Output .gpkg path")
    parser.add_argument("-b", "--batch", default=None, help="Batch config YAML")
    parser.add_argument("--lclu", default=None, help="Path to LCLU raster (overrides config)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    run_id = configure_root_logger(
        level="DEBUG" if args.verbose else "INFO",
        log_format="auto",
    )
    logger = get_logger("infer")
    logger.info(f"Run ID: {run_id}")

    if args.batch:
        batch_delineate(args.batch, verbose=args.verbose)
    elif args.input:
        config = load_config(args.config)
        output = args.output or os.path.join("data/output", "fields.gpkg")
        delineate_folder(config, args.input, output, lclu_override=args.lclu, verbose=args.verbose)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python scripts/infer.py -c configs/inference.yaml -i data/raw/Sample -o out.gpkg")
        print("  python scripts/infer.py -c configs/inference.yaml -i data/raw/Sample -o out.gpkg --lclu lclu.tif")
        print("  python scripts/infer.py -b configs/batch.yaml")


if __name__ == "__main__":
    main()
