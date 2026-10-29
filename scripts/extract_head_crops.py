"""Extract head-region crops from full-scene images for the head classifier dataset.

Uses person detection (YOLOv8) to localize workers, then applies HeadRegionLocalizer
(with face-anchor and top-region fallback) to produce localized head crops.
Integrates into data/head_classifier/manifest.csv for human review.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.head_localization import HeadRegionLocalizer
from src.utils import load_config
from ultralytics import YOLO


def file_digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract head crops from full scenes")
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=["demo_images"],
        help="Directories containing source images",
    )
    parser.add_argument(
        "--manifest",
        default="data/head_classifier/manifest.csv",
        help="Path to manifest CSV",
    )
    parser.add_argument(
        "--crops-dir",
        default="data/head_classifier/crops",
        help="Directory to save extracted head crops",
    )
    parser.add_argument(
        "--person-model",
        default="yolov8n.pt",
        help="YOLO model for person detection",
    )
    parser.add_argument(
        "--person-conf",
        type=float,
        default=0.35,
        help="Person detection confidence threshold",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform detection without saving crops or modifying manifest",
    )
    args = parser.parse_args()

    config = load_config(str(ROOT / "config" / "config.yaml"))
    head_cfg = config.get("head_classifier", {})
    localizer = HeadRegionLocalizer(
        scale_factor=float(head_cfg.get("face_scale_factor", 1.1)),
        min_neighbors=int(head_cfg.get("face_min_neighbors", 5)),
        use_top_region_fallback=bool(head_cfg.get("use_top_region_fallback", True)),
        top_region_ratio=float(head_cfg.get("top_region_ratio", 0.28)),
    )

    person_model_path = str(ROOT / args.person_model)
    print(f"Loading person detection model from: {person_model_path}")
    person_model = YOLO(person_model_path)

    manifest_path = ROOT / args.manifest
    crops_dir = ROOT / args.crops_dir
    if not args.dry_run:
        crops_dir.mkdir(parents=True, exist_ok=True)

    # Read existing manifest entries to avoid duplicates
    existing_rows = []
    existing_hashes = set()
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or [
                "crop_path",
                "label",
                "split",
                "review_status",
                "source_image",
                "reviewer",
                "notes",
            ]
            for row in reader:
                existing_rows.append(row)
                crop_p = ROOT / row.get("crop_path", "")
                if crop_p.is_file():
                    existing_hashes.add(file_digest(crop_p.read_bytes()))
    else:
        fieldnames = [
            "crop_path",
            "label",
            "split",
            "review_status",
            "source_image",
            "reviewer",
            "notes",
        ]

    new_rows = []
    total_persons_found = 0
    total_crops_saved = 0

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for input_dir_str in args.inputs:
        input_dir = ROOT / input_dir_str
        if not input_dir.exists():
            print(f"Directory not found: {input_dir}")
            continue

        images = [p for p in input_dir.rglob("*") if p.suffix.lower() in image_extensions]
        print(f"Processing {len(images)} images from {input_dir_str}...")

        for img_path in sorted(images):
            # Read image
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue

            rel_source = str(img_path.relative_to(ROOT)).replace("\\", "/")
            # Run person detection
            results = person_model.predict(
                frame,
                conf=args.person_conf,
                classes=[0],
                verbose=False,
            )

            if not results:
                continue

            person_idx = 0
            for box in results[0].boxes:
                person_idx += 1
                total_persons_found += 1
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                p_bbox = [float(x1), float(y1), float(x2), float(y2)]

                # Localize head
                head_crop_res = localizer.locate(frame, p_bbox)
                if not head_crop_res.reliable or head_crop_res.crop is None:
                    continue

                crop_img = head_crop_res.crop
                crop_bytes = cv2.imencode(".jpg", crop_img, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
                crop_hash = file_digest(crop_bytes)

                if crop_hash in existing_hashes:
                    continue

                existing_hashes.add(crop_hash)
                crop_filename = f"{img_path.stem}_p{person_idx}_{crop_hash[:8]}.jpg"
                crop_rel_path = f"{args.crops_dir}/{crop_filename}".replace("\\", "/")

                if not args.dry_run:
                    crop_full_path = crops_dir / crop_filename
                    crop_full_path.write_bytes(crop_bytes)

                total_crops_saved += 1
                note_text = f"localization={head_crop_res.reason}; person_bbox={[round(v, 1) for v in p_bbox]}"
                new_rows.append({
                    "crop_path": crop_rel_path,
                    "label": "",
                    "split": "pending",
                    "review_status": "manual_review_required",
                    "source_image": rel_source,
                    "reviewer": "",
                    "notes": note_text,
                })

    print(f"\nExtraction summary:")
    print(f"  Persons detected: {total_persons_found}")
    print(f"  New head crops extracted: {total_crops_saved}")

    if not args.dry_run and new_rows:
        all_rows = existing_rows + new_rows
        with open(manifest_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"Updated manifest at: {manifest_path} with {len(new_rows)} new entries.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

