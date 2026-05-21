"""
evaluate.py — Evaluate a trained model on a validation dataset.

Usage:
    python scripts/evaluate.py -c configs/train.yaml -w output_detectron2/model_final.pth
"""

import os
import sys
import json
from argparse import ArgumentParser
from pathlib import Path

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.monitoring.logger import configure_root_logger, get_logger


def main():
    parser = ArgumentParser(description="Field Delineation — Evaluation")
    parser.add_argument("-c", "--config", default="configs/train.yaml")
    parser.add_argument("-w", "--weights", required=True, help="Model weights path")
    parser.add_argument("-o", "--output", default="eval_results", help="Output directory")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    configure_root_logger(level="DEBUG" if args.verbose else "INFO")
    logger = get_logger("evaluate")

    config = yaml.safe_load(Path(args.config).read_text())

    import torch
    from detectron2 import model_zoo
    from detectron2.config import get_cfg
    from detectron2.data import MetadataCatalog, DatasetCatalog
    from detectron2.data.datasets import register_coco_instances
    from detectron2.engine import DefaultTrainer
    from detectron2.evaluation import COCOEvaluator

    ds_config = config["dataset"]
    images_root = ds_config["images_root"]

    val_name = f"{ds_config['name']}_val"
    if val_name not in DatasetCatalog.list():
        img_dir = os.path.join(images_root, "val", "images")
        register_coco_instances(val_name, {}, ds_config["val_json"], img_dir)
        MetadataCatalog.get(val_name).set(thing_classes=[ds_config.get("class_name", "field")])

    model_config = config["model"]
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_config["config_file"]))
    cfg.MODEL.WEIGHTS = args.weights
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = model_config.get("num_classes", 1)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.DATASETS.TEST = (val_name,)
    cfg.OUTPUT_DIR = args.output

    if not torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cpu"

    os.makedirs(args.output, exist_ok=True)

    evaluator = COCOEvaluator(val_name, output_dir=args.output)
    results = DefaultTrainer.test(cfg, DefaultTrainer.build_model(cfg), evaluators=[evaluator])

    logger.info("Evaluation results", extra={"results": results})

    with open(os.path.join(args.output, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved: {args.output}/results.json")


if __name__ == "__main__":
    main()
