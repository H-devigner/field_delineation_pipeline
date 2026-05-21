"""
Format converter — YOLO to COCO and other dataset format conversions.

Extracted from the original yolo_to_coco.py script for reuse across
training and data preparation workflows.
"""

import json
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def yolo_seg_to_coco(
    images_dir: str,
    labels_dir: str,
    output_json: str,
    class_names: Optional[List[str]] = None,
) -> str:
    """
    Convert YOLOv8 segmentation format to COCO JSON.

    Parameters
    ----------
    images_dir : str
        Directory containing images.
    labels_dir : str
        Directory containing YOLO .txt label files.
    output_json : str
        Output COCO JSON path.
    class_names : list of str or None
        Class names. Defaults to ["field"].

    Returns
    -------
    str
        Path to the output JSON.
    """
    if class_names is None:
        class_names = ["field"]

    categories = [
        {"id": i, "name": name, "supercategory": "none"}
        for i, name in enumerate(class_names)
    ]

    images = []
    annotations = []
    ann_id = 1

    image_files = sorted([
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff"))
    ])

    for img_id, img_file in enumerate(image_files, start=1):
        img_path = os.path.join(images_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            logger.warning(f"Cannot read: {img_path}")
            continue

        h, w = img.shape[:2]
        images.append({
            "id": img_id,
            "file_name": img_file,
            "width": w,
            "height": h,
        })

        # Find matching label file
        label_file = Path(img_file).stem + ".txt"
        label_path = os.path.join(labels_dir, label_file)

        if not os.path.exists(label_path):
            continue

        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 7:  # class + at least 3 polygon points
                    continue

                class_id = int(parts[0])
                coords = list(map(float, parts[1:]))

                # Convert normalized coords to absolute
                poly = []
                for i in range(0, len(coords), 2):
                    px = coords[i] * w
                    py = coords[i + 1] * h
                    poly.extend([px, py])

                # Compute bounding box
                xs = poly[0::2]
                ys = poly[1::2]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                bbox_w = x_max - x_min
                bbox_h = y_max - y_min

                # Compute area from polygon
                area = _polygon_area(xs, ys)

                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": class_id,
                    "segmentation": [poly],
                    "area": area,
                    "bbox": [x_min, y_min, bbox_w, bbox_h],
                    "iscrowd": 0,
                })
                ann_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(coco, f, indent=2)

    logger.info(f"COCO JSON: {len(images)} images, {len(annotations)} annotations → {output_json}")
    return output_json


def _polygon_area(xs: List[float], ys: List[float]) -> float:
    """Compute area using the Shoelace formula."""
    n = len(xs)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += xs[i] * ys[j]
        area -= xs[j] * ys[i]
    return abs(area) / 2.0
