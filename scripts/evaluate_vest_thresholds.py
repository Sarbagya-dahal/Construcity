"""Evaluate a PPE class's confidence thresholds on the labelled validation split."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_dataset import iou, read_labels
from src.processor import FrameProcessor
from src.utils import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate PPE confidence thresholds")
    parser.add_argument("--dataset", default="data/Construction Site.v1i.yolov8")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--output", default="outputs/vest_threshold_report.json")
    parser.add_argument("--class-name", default="vest", choices=["helmet", "vest", "without_helmet", "without_vest"])
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90],
    )
    args = parser.parse_args()

    dataset = (ROOT / args.dataset).resolve()
    config = load_config(str((ROOT / args.config).resolve()))
    config["storage"]["save_violations"] = False
    processor = FrameProcessor(config, str((ROOT / config["model"]["weights_path"]).resolve()))

    records = []
    for image_path in sorted((dataset / "valid" / "images").glob("*.jpg")):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        processor.process_frame(image, save_violation=False, source_name=str(image_path))
        height, width = image.shape[:2]
        truth = [
            item["bbox"]
            for item in read_labels(dataset / "valid" / "labels" / f"{image_path.stem}.txt", width, height)
            if item["class_id"] == config["classes"][args.class_name]
        ]
        raw = [
            item["detection"]
            for item in processor.detector.last_diagnostics.get("ppe", {}).get("raw", {}).get(args.class_name, [])
        ]
        records.append({"image_id": image_path.name, "truth": truth, "raw": raw})

    results = []
    for threshold in sorted(set(args.thresholds)):
        tp = fp = fn = 0
        for record in records:
            matched = [False] * len(record["truth"])
            for prediction in sorted(
                [item for item in record["raw"] if item["confidence"] >= threshold],
                key=lambda item: item["confidence"],
                reverse=True,
            ):
                candidates = [
                    (iou(prediction["bbox"], truth), index)
                    for index, truth in enumerate(record["truth"])
                    if not matched[index]
                ]
                best_iou, best_index = max(candidates, default=(0.0, None))
                if best_index is not None and best_iou >= 0.50:
                    matched[best_index] = True
                    tp += 1
                else:
                    fp += 1
            fn += sum(not value for value in matched)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        results.append({
            "threshold": threshold,
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

    report = {
        "dataset": str(dataset),
        "split": "valid",
        "class": args.class_name,
        "iou_threshold": 0.50,
        "image_count": len(records),
        "raw_model_confidence_floor": config["model"]["confidence_threshold"],
        "results": results,
    }
    output = (ROOT / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
