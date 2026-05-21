"""
train.py — CLI for Detectron2 Mask R-CNN training with MLflow tracking.

Auto-converts YOLO-format datasets (from Roboflow) to COCO JSON
if the annotation files don't exist yet.

Usage:
    python scripts/train.py -c configs/train.yaml
    python scripts/train.py -c configs/train.yaml --resume
"""

import os
import sys
import json
import time
from argparse import ArgumentParser
from pathlib import Path

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.monitoring.logger import configure_root_logger, get_logger
from src.data.converter import yolo_seg_to_coco


# ── MLflow wrapper (optional dependency) ─────────────────────
class MLflowTracker:
    """Thin wrapper so training works with or without mlflow installed."""

    def __init__(self, tracking_config: dict, logger):
        self.enabled = False
        self.logger = logger

        backend = tracking_config.get("backend", "none")
        if backend not in ("mlflow",):
            logger.info(f"Tracking backend: {backend} (MLflow disabled)")
            return

        try:
            import mlflow
            self.mlflow = mlflow

            uri = tracking_config.get("mlflow_uri", "mlruns")
            mlflow.set_tracking_uri(uri)
            experiment = tracking_config.get("experiment_name", "field-delineation")
            mlflow.set_experiment(experiment)
            self.log_every = tracking_config.get("log_every_n_iter", 50)
            self.save_artifact = tracking_config.get("save_model_artifact", True)
            self.enabled = True
            logger.info(f"MLflow tracking: experiment='{experiment}', uri='{uri}'")
        except ImportError:
            logger.warning("mlflow not installed — tracking disabled. Install with: pip install mlflow")

    def start_run(self, run_name: str = None):
        if self.enabled:
            self.mlflow.start_run(run_name=run_name)

    def log_params(self, params: dict):
        if self.enabled:
            self.mlflow.log_params(params)

    def log_metrics(self, metrics: dict, step: int = None):
        if self.enabled:
            self.mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path: str):
        if self.enabled and self.save_artifact and os.path.exists(path):
            self.mlflow.log_artifact(path)

    def end_run(self):
        if self.enabled:
            self.mlflow.end_run()


def auto_convert_yolo_to_coco(ds_config: dict, logger):
    """Auto-convert YOLO segmentation labels to COCO JSON if missing."""
    yolo_root = ds_config.get("yolo_root")
    if not yolo_root or not os.path.isdir(yolo_root):
        return

    splits = {
        "train": {
            "json": ds_config["train_json"],
            "images": os.path.join(yolo_root, "train", "images"),
            "labels": os.path.join(yolo_root, "train", "labels"),
        },
        "val": {
            "json": ds_config["val_json"],
            "images": os.path.join(yolo_root, "valid", "images"),
            "labels": os.path.join(yolo_root, "valid", "labels"),
        },
    }

    test_dir = os.path.join(yolo_root, "test", "images")
    if os.path.isdir(test_dir):
        splits["test"] = {
            "json": ds_config.get("test_json", "data/training/annotations/test.json"),
            "images": test_dir,
            "labels": os.path.join(yolo_root, "test", "labels"),
        }

    class_name = ds_config.get("class_name", "field")

    for split_name, paths in splits.items():
        json_path = paths["json"]
        if os.path.exists(json_path):
            logger.info(f"COCO JSON exists: {json_path} (skipping)")
            continue
        if not os.path.isdir(paths["images"]) or not os.path.isdir(paths["labels"]):
            logger.warning(f"Missing images/labels for {split_name} — skipping")
            continue

        logger.info(f"Converting YOLO → COCO: {split_name} → {json_path}")
        yolo_seg_to_coco(
            images_dir=paths["images"],
            labels_dir=paths["labels"],
            output_json=json_path,
            class_names=[class_name],
        )


def main():
    parser = ArgumentParser(description="Field Delineation — Training")
    parser.add_argument("-c", "--config", default="configs/train.yaml")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--convert-only", action="store_true", help="Only convert YOLO→COCO")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    run_id = configure_root_logger(level="DEBUG" if args.verbose else "INFO", log_format="text")
    logger = get_logger("train")
    logger.info(f"Run ID: {run_id}")

    # ── Load config ──────────────────────────────────────────
    config = yaml.safe_load(Path(args.config).read_text())
    if "base_config" in config:
        base = yaml.safe_load(Path(config["base_config"]).read_text())
        for k, v in base.items():
            config.setdefault(k, v)

    ds_config = config["dataset"]

    # ── Auto-convert YOLO → COCO ─────────────────────────────
    auto_convert_yolo_to_coco(ds_config, logger)
    if args.convert_only:
        logger.info("Conversion complete (--convert-only). Exiting.")
        return

    # ── Lazy imports ─────────────────────────────────────────
    import torch
    from detectron2 import model_zoo
    from detectron2.config import get_cfg
    from detectron2.data import MetadataCatalog, DatasetCatalog
    from detectron2.data.datasets import register_coco_instances
    from detectron2.engine import DefaultTrainer, HookBase
    from detectron2.evaluation import COCOEvaluator

    logger.info(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            logger.info(f"  GPU {i}: {name} ({mem:.0f} GB)")

    # ── MLflow tracking ──────────────────────────────────────
    tracking_config = config.get("tracking", {})
    tracker = MLflowTracker(tracking_config, logger)

    # ── Verify COCO JSONs ────────────────────────────────────
    for key in ["train_json", "val_json"]:
        if not os.path.exists(ds_config[key]):
            logger.error(f"Missing: {ds_config[key]}. Run with --convert-only first.")
            return

    # ── Register dataset ─────────────────────────────────────
    yolo_root = ds_config.get("yolo_root", ds_config["images_root"])
    for split, json_file, img_subdir in [
        ("train", ds_config["train_json"], os.path.join(yolo_root, "train", "images")),
        ("val", ds_config["val_json"], os.path.join(yolo_root, "valid", "images")),
    ]:
        name = f"{ds_config['name']}_{split}"
        if name not in DatasetCatalog.list():
            register_coco_instances(name, {}, json_file, img_subdir)
            MetadataCatalog.get(name).set(thing_classes=[ds_config.get("class_name", "field")])
        logger.info(f"Registered: {name}")

    # ── Build Detectron2 config ──────────────────────────────
    model_config = config["model"]
    train_config = config["training"]
    output_dir = config.get("output_dir", "output_detectron2")

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_config["config_file"]))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_config["pretrained_weights"])
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = model_config.get("num_classes", 1)
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.DATASETS.TRAIN = (f"{ds_config['name']}_train",)
    cfg.DATASETS.TEST = (f"{ds_config['name']}_val",)
    cfg.SOLVER.MAX_ITER = train_config.get("max_iter", 3000)
    cfg.SOLVER.BASE_LR = train_config.get("base_lr", 0.00025)
    cfg.SOLVER.IMS_PER_BATCH = train_config.get("batch_size", 2)
    cfg.SOLVER.CHECKPOINT_PERIOD = train_config.get("checkpoint_period", 500)
    cfg.SOLVER.STEPS = tuple(train_config.get("lr_decay_steps", [2000, 2500]))
    cfg.SOLVER.GAMMA = train_config.get("lr_decay_factor", 0.1)
    cfg.SOLVER.WARMUP_ITERS = train_config.get("warmup_iters", 100)
    cfg.DATALOADER.NUM_WORKERS = train_config.get("num_workers", 2)
    cfg.TEST.EVAL_PERIOD = train_config.get("eval_period", 500)
    cfg.OUTPUT_DIR = output_dir

    if not torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cpu"
        logger.warning("CUDA not available — training on CPU")

    if args.resume:
        cfg.MODEL.WEIGHTS = os.path.join(output_dir, "last_checkpoint")
        logger.info("Resuming from checkpoint")

    os.makedirs(output_dir, exist_ok=True)

    # ── Save config snapshot ─────────────────────────────────
    snapshot_path = os.path.join(output_dir, "config_snapshot.yaml")
    with open(snapshot_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # ── Start MLflow run ─────────────────────────────────────
    tracker.start_run(run_name=f"maskrcnn-r101-lr{cfg.SOLVER.BASE_LR}-iter{cfg.SOLVER.MAX_ITER}")
    tracker.log_params({
        "model": model_config["config_file"],
        "pretrained": model_config["pretrained_weights"],
        "num_classes": model_config.get("num_classes", 1),
        "lr": cfg.SOLVER.BASE_LR,
        "batch_size": cfg.SOLVER.IMS_PER_BATCH,
        "max_iter": cfg.SOLVER.MAX_ITER,
        "warmup_iters": cfg.SOLVER.WARMUP_ITERS,
        "lr_steps": str(cfg.SOLVER.STEPS),
        "dataset": ds_config["name"],
    })
    tracker.log_artifact(args.config)

    # ── Training hooks ───────────────────────────────────────
    class MLflowLossHook(HookBase):
        """Log training loss to MLflow every N iterations."""
        def __init__(self, tracker, log_every):
            self._tracker = tracker
            self._log_every = log_every

        def after_step(self):
            if (self.trainer.iter + 1) % self._log_every == 0:
                storage = self.trainer.storage
                metrics = {}
                for key in ["total_loss", "loss_cls", "loss_box_reg", "loss_mask", "loss_rpn_cls", "loss_rpn_loc"]:
                    try:
                        val = storage.latest().get(key)
                        if val is not None:
                            metrics[f"train/{key}"] = val[0]
                    except Exception:
                        pass
                try:
                    metrics["train/lr"] = storage.latest().get("lr", (0,))[0]
                except Exception:
                    pass
                if metrics:
                    self._tracker.log_metrics(metrics, step=self.trainer.iter + 1)

    class EvalHook(HookBase):
        """Run COCO evaluation and log to MLflow."""
        def __init__(self, eval_period, cfg, tracker):
            self._period = eval_period
            self._cfg = cfg
            self._tracker = tracker

        def after_step(self):
            if (self.trainer.iter + 1) % self._period == 0:
                evaluator = COCOEvaluator(
                    self._cfg.DATASETS.TEST[0],
                    output_dir=self._cfg.OUTPUT_DIR,
                )
                results = DefaultTrainer.test(self._cfg, self.trainer.model, evaluators=[evaluator])
                logger.info(f"Eval @ iter {self.trainer.iter + 1}", extra={"results": results})

                # Log eval metrics to MLflow
                if "segm" in results:
                    segm = results["segm"]
                    self._tracker.log_metrics({
                        "val/AP": segm.get("AP", 0),
                        "val/AP50": segm.get("AP50", 0),
                        "val/AP75": segm.get("AP75", 0),
                    }, step=self.trainer.iter + 1)
                if "bbox" in results:
                    bbox = results["bbox"]
                    self._tracker.log_metrics({
                        "val/bbox_AP": bbox.get("AP", 0),
                        "val/bbox_AP50": bbox.get("AP50", 0),
                    }, step=self.trainer.iter + 1)

    # ── Train ────────────────────────────────────────────────
    logger.info("Starting training...")
    t_start = time.time()

    trainer = DefaultTrainer(cfg)
    trainer.register_hooks([
        MLflowLossHook(tracker, tracker.log_every if tracker.enabled else 50),
        EvalHook(cfg.TEST.EVAL_PERIOD, cfg, tracker),
    ])
    trainer.resume_or_load(resume=args.resume)
    trainer.train()

    t_elapsed = time.time() - t_start
    logger.info(f"Training complete in {t_elapsed:.0f}s")
    tracker.log_metrics({"train/total_time_s": t_elapsed})

    # ── Final evaluation ─────────────────────────────────────
    evaluator = COCOEvaluator(cfg.DATASETS.TEST[0], output_dir=output_dir)
    results = DefaultTrainer.test(cfg, trainer.model, evaluators=[evaluator])
    logger.info("Final evaluation", extra={"results": results})

    with open(os.path.join(output_dir, "eval_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Log final results + model to MLflow
    if "segm" in results:
        tracker.log_metrics({
            "final/AP": results["segm"].get("AP", 0),
            "final/AP50": results["segm"].get("AP50", 0),
        })
    model_path = os.path.join(output_dir, "model_final.pth")
    tracker.log_artifact(model_path)
    tracker.log_artifact(os.path.join(output_dir, "eval_results.json"))
    tracker.end_run()

    logger.info(f"Model: {model_path}")
    if tracker.enabled:
        logger.info("MLflow run complete — view at your MLflow UI")


if __name__ == "__main__":
    main()
