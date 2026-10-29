"""Copy only manually reviewed manifest rows into ImageFolder directories."""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from pathlib import Path

from PIL import Image

CLASSES = {"helmet_worn", "no_helmet"}
SPLITS = {"train", "val", "test"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare reviewed head crops")
    parser.add_argument("--manifest", default="data/head_classifier/manifest.csv")
    parser.add_argument("--output", default="data/head_classifier_reviewed")
    parser.add_argument("--min-train-per-class", type=int, default=20)
    parser.add_argument("--min-eval-per-class", type=int, default=10)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest_path = (root / args.manifest).resolve()
    output = (root / args.output).resolve()
    rows = list(csv.DictReader(manifest_path.read_text(encoding="utf-8").splitlines()))
    reviewed = [row for row in rows if row.get("review_status") == "reviewed"]
    errors = []
    hashes = {}
    counts = {(split, label): 0 for split in SPLITS for label in CLASSES}
    for row in reviewed:
        crop = (root / row.get("crop_path", "")).resolve()
        label, split = row.get("label", ""), row.get("split", "")
        if label not in CLASSES or split not in SPLITS:
            errors.append(f"invalid reviewed row: {row}")
            continue
        if not crop.is_file() or crop.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            errors.append(f"missing or unsupported crop: {crop}")
            continue
        try:
            Image.open(crop).verify()
        except Exception as exc:
            errors.append(f"unreadable crop {crop}: {exc}")
            continue
        digest = hashlib.sha256(crop.read_bytes()).hexdigest()
        if digest in hashes:
            errors.append(f"duplicate crop: {crop} and {hashes[digest]}")
        hashes[digest] = crop
        counts[(split, label)] += 1
    print(f"reviewed_rows={len(reviewed)} counts={counts}")
    for (split, label), count in counts.items():
        minimum = args.min_train_per_class if split == "train" else args.min_eval_per_class
        if count < minimum:
            errors.append(f"{split}/{label}: {count} reviewed crops; need {minimum}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        print("No head-classifier dataset was created")
        return 1
    if output.exists():
        raise SystemExit(f"Refusing to overwrite {output}")
    for split in SPLITS:
        for label in CLASSES:
            (output / split / label).mkdir(parents=True, exist_ok=True)
    for row in reviewed:
        crop = (root / row["crop_path"]).resolve()
        destination = output / row["split"] / row["label"] / crop.name
        shutil.copy2(crop, destination)
    print(f"prepared_dataset={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
