"""Generate raw/filter/association diagnostics for the local demo images."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.processor import FrameProcessor
from src.utils import load_config


def main() -> int:
    config = load_config(str(ROOT / "config" / "config.yaml"))
    config["storage"]["save_violations"] = False
    processor = FrameProcessor(config, str((ROOT / config["model"]["weights_path"]).resolve()))
    output_dir = ROOT / "outputs" / "demo_diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "coordinate_system": "original frame pixels; preprocessing preserves frame dimensions",
        "configured_classes": config.get("classes", {}),
        "ppe_model_names": {str(key): value for key, value in processor.detector.ppe_model.names.items()},
        "images": {},
    }

    demo_images_dir = ROOT / "demo_images"
    image_paths = sorted(demo_images_dir.glob("*.jpg"))

    for image_path in image_paths:
        image_name = image_path.name
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Warning: could not read {image_path}, skipping.")
            continue
        result = processor.process_frame(image, save_violation=False, source_name=image_name)
        diagnostics = processor.detector.last_diagnostics
        filtered = result["detections"]
        rejected = {}
        for class_name, items in diagnostics.get("ppe", {}).get("raw", {}).items():
            rejected[class_name] = []
            for item in items:
                detection = item["detection"]
                post_rejection = next(
                    (
                        entry["reasons"]
                        for entry in diagnostics.get("ppe", {}).get("helmet_post_filter_rejections", [])
                        if entry["detection"].get("bbox") == detection.get("bbox")
                    ),
                    [],
                )
                reasons = list(dict.fromkeys(item.get("reasons", []) + post_rejection))
                rejected[class_name].append({
                    "class_id": detection.get("class_id"),
                    "class": class_name,
                    "confidence": detection.get("confidence"),
                    "bbox": detection.get("bbox"),
                    "accepted": not reasons and any(detection == accepted for accepted in filtered.get(class_name, [])),
                    "rejection_reasons": reasons,
                })
        report["images"][image_name] = {
            "shape": list(image.shape),
            "raw_person_detections": diagnostics.get("person", {}).get("raw", []),
            "raw_ppe_detections": rejected,
            "filtered_detections": filtered,
            "head_verifications": result.get("head_verifications", []),
            "violation_result": result["violations"],
            "annotated_output": str(output_dir / image_name),
        }
        cv2.imwrite(str(output_dir / image_name), result["annotated_frame"])

    report_path = output_dir / "diagnostic_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    heuristic_report_path = ROOT / "outputs" / "heuristic_diagnostic_report.json"
    heuristic_report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report saved to: {report_path} and {heuristic_report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
