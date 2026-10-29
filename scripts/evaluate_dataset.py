"""Evaluate raw PPE detections and summarize the end-to-end pipeline.

The dataset labels are polygon annotations. They are converted to enclosing
boxes for class-wise detection metrics. Pipeline violation results are kept as
counts/statuses because person-level violation boxes are not PPE ground truth.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.processor import FrameProcessor
from src.utils import load_config


def iou(box_a, box_b) -> float:
    x1, y1 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    x2, y2 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def read_labels(label_path: Path, width: int, height: int) -> list[dict]:
    labels = []
    if not label_path.exists():
        return labels
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        class_id = int(parts[0])
        values = [float(value) for value in parts[1:]]
        if len(values) == 4:
            cx, cy, box_width, box_height = values
            x1 = (cx - box_width / 2) * width
            y1 = (cy - box_height / 2) * height
            x2 = (cx + box_width / 2) * width
            y2 = (cy + box_height / 2) * height
        else:
            points = list(zip(values[::2], values[1::2]))
            x1 = min(point[0] for point in points) * width
            y1 = min(point[1] for point in points) * height
            x2 = max(point[0] for point in points) * width
            y2 = max(point[1] for point in points) * height
        labels.append({"class_id": class_id, "bbox": [x1, y1, x2, y2]})
    return labels


def average_precision(predictions: list[dict], ground_truth: dict, iou_threshold: float) -> float:
    total_ground_truth = sum(len(items) for items in ground_truth.values())
    if total_ground_truth == 0:
        return float("nan")
    matched = {image_id: [False] * len(items) for image_id, items in ground_truth.items()}
    predictions = sorted(predictions, key=lambda item: item["confidence"], reverse=True)
    true_positives, false_positives = [], []
    for prediction in predictions:
        candidates = ground_truth.get(prediction["image_id"], [])
        best_index, best_iou = None, 0.0
        for index, truth in enumerate(candidates):
            overlap = iou(prediction["bbox"], truth)
            if not matched[prediction["image_id"]][index] and overlap > best_iou:
                best_index, best_iou = index, overlap
        if best_index is not None and best_iou >= iou_threshold:
            matched[prediction["image_id"]][best_index] = True
            true_positives.append(1)
            false_positives.append(0)
        else:
            true_positives.append(0)
            false_positives.append(1)
    tp = np.cumsum(true_positives)
    fp = np.cumsum(false_positives)
    recalls = tp / total_ground_truth
    precisions = tp / np.maximum(tp + fp, 1)
    return float(np.mean([np.max(precisions[recalls >= recall]) if np.any(recalls >= recall) else 0.0
                          for recall in np.linspace(0.0, 1.0, 101)]))


def class_metrics(predictions: list[dict], truths: dict, class_id: int) -> dict:
    class_predictions = [item for item in predictions if item["class_id"] == class_id]
    class_truths = {image_id: boxes for image_id, boxes in truths.items() if boxes}
    result = {"ground_truth": sum(len(items) for items in class_truths.values()), "predictions": len(class_predictions)}
    for threshold_name, threshold in [("0.50", 0.50)]:
        matched = {image_id: [False] * len(items) for image_id, items in class_truths.items()}
        tp = fp = 0
        for prediction in sorted(class_predictions, key=lambda item: item["confidence"], reverse=True):
            candidates = class_truths.get(prediction["image_id"], [])
            best = max((iou(prediction["bbox"], truth) if not matched[prediction["image_id"]][index] else 0.0,
                        index) for index, truth in enumerate(candidates)) if candidates else (0.0, None)
            if best[0] >= threshold:
                matched[prediction["image_id"]][best[1]] = True
                tp += 1
            else:
                fp += 1
        fn = result["ground_truth"] - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        result.update({"precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                      "true_positive": tp, "false_positive": fp, "false_negative": fn})
    aps = [average_precision(class_predictions, class_truths, threshold) for threshold in np.arange(0.50, 1.00, 0.05)]
    result["ap50"] = float(aps[0]) if aps else 0.0
    result["map50_95"] = float(np.nanmean(aps)) if aps else 0.0
    return result


def build_confusion_matrix(records: list[dict], class_names: dict[int, str]) -> dict:
    labels = [class_names[class_id] for class_id in sorted(class_names)] + ["background"]
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    background_index = len(labels) - 1
    for record in records:
        truths = [(class_id, bbox) for class_id, boxes in record["truth"].items() for bbox in boxes]
        predictions = [(item["class_id"], item["bbox"], item.get("confidence", 0.0))
                       for items in record["raw"].values() for item in items]
        matched_truth = set()
        matched_predictions = set()
        for prediction_index, prediction in sorted(enumerate(predictions), key=lambda pair: pair[1][2], reverse=True):
            candidates = [(iou(prediction[1], truth[1]), truth_index) for truth_index, truth in enumerate(truths)
                          if truth_index not in matched_truth]
            if not candidates:
                continue
            overlap, truth_index = max(candidates)
            if overlap >= 0.5:
                true_class, _ = truths[truth_index]
                pred_class = prediction[0]
                matrix[sorted(class_names).index(true_class), sorted(class_names).index(pred_class)] += 1
                matched_truth.add(truth_index)
                matched_predictions.add(prediction_index)
        for truth_index, (true_class, _) in enumerate(truths):
            if truth_index not in matched_truth:
                matrix[sorted(class_names).index(true_class), background_index] += 1
        for prediction_index, (pred_class, _, _) in enumerate(predictions):
            if prediction_index not in matched_predictions:
                matrix[background_index, sorted(class_names).index(pred_class)] += 1
    return {"labels": labels, "values": matrix.tolist(), "iou_threshold": 0.5, "source": "raw_ppe_model"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the labelled PPE test split")
    parser.add_argument("--dataset", default="data/Construction Site.v1i.yolov8")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--output", default="outputs/evaluation_report.json")
    parser.add_argument("--weights", default=None, help="Optional checkpoint override")
    parser.add_argument("--split", default="test", choices=["valid", "test"])
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    dataset = (root / args.dataset).resolve()
    config = load_config(str((root / args.config).resolve()))
    config["storage"]["save_violations"] = False
    weights = args.weights or config["model"]["weights_path"]
    processor = FrameProcessor(config, str((root / weights).resolve()))
    class_names = {int(value): name for name, value in config["classes"].items()}
    records = []
    pipeline_summary = {"images": 0, "persons": 0, "confirmed_violations": 0, "unverified_candidates": 0,
                        "violations_by_type": defaultdict(int), "person_missing_frames": 0}

    image_paths = sorted((dataset / args.split / "images").glob("*.jpg"))
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        result = processor.process_frame(image, save_violation=False, source_name=str(image_path))
        height, width = image.shape[:2]
        truth_by_class = defaultdict(list)
        for label in read_labels(dataset / args.split / "labels" / f"{image_path.stem}.txt", width, height):
            truth_by_class[label["class_id"]].append(label["bbox"])
        raw_by_class = defaultdict(list)
        for class_name, items in processor.detector.last_diagnostics.get("ppe", {}).get("raw", {}).items():
            class_id = config["classes"].get(class_name)
            if class_id is not None:
                raw_by_class[int(class_id)].extend(item["detection"] for item in items)
        filtered_by_class = defaultdict(list)
        for class_name, items in result["detections"].items():
            class_id = config["classes"].get(class_name)
            if class_id is not None:
                filtered_by_class[int(class_id)].extend(items)
        records.append({"image_id": image_path.name, "truth": truth_by_class, "raw": raw_by_class, "filtered": filtered_by_class})
        pipeline_summary["images"] += 1
        pipeline_summary["persons"] += result["violations"]["person_count"]
        pipeline_summary["confirmed_violations"] += result["violations"]["violation_count"]
        pipeline_summary["unverified_candidates"] += len(result["violations"].get("unverified_candidates", []))
        if result["violations"]["status"] == "person_missing":
            pipeline_summary["person_missing_frames"] += 1
        for violation in result["violations"]["violations"]:
            pipeline_summary["violations_by_type"][violation["type"]] += 1

    metric_groups = {}
    for prediction_key in ("raw", "filtered"):
        all_predictions, truths_by_class = [], defaultdict(dict)
        for record in records:
            for class_id, items in record["truth"].items():
                truths_by_class[class_id][record["image_id"]] = items
            for class_id, items in record[prediction_key].items():
                all_predictions.extend({"image_id": record["image_id"], "class_id": class_id,
                                        "bbox": item["bbox"], "confidence": item.get("confidence", 0.0)} for item in items)
        per_class = {class_names[class_id]: class_metrics(all_predictions, truths_by_class.get(class_id, {}), class_id)
                     for class_id in sorted(class_names)}
        valid_maps = [value["map50_95"] for value in per_class.values() if value["ground_truth"]]
        metric_groups[prediction_key] = {"per_class": per_class, "map50_95": float(np.mean(valid_maps)) if valid_maps else 0.0}

    report = {"dataset": str(dataset), "split": args.split, "weights": str((root / weights).resolve()), "image_count": len(records),
              "classes": class_names, "metrics": {"raw_ppe_model": metric_groups["raw"],
              "filtered_pipeline_ppe": metric_groups["filtered"]},
              "confusion_matrix": build_confusion_matrix(records, class_names),
              "end_to_end_pipeline": {**pipeline_summary, "violations_by_type": dict(pipeline_summary["violations_by_type"])} }
    output = (root / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
