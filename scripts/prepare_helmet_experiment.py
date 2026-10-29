"""Prepare an isolated four-class experiment dataset from reviewed examples."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image


CLASS_NAMES = {0: "helmet", 1: "vest", 2: "without_helmet", 3: "without_vest"}
REQUIRED_CATEGORIES = {
    "bare_head",
    "bald_scalp",
    "cap_or_beanie",
    "headphones",
    "held_or_nearby_helmet",
    "worn_helmet",
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dhash(path: Path) -> int:
    image = Image.open(path).convert("L").resize((9, 8))
    pixels = list(image.getdata())
    value = 0
    for row in range(8):
        for col in range(8):
            value = (value << 1) | int(pixels[row * 9 + col] > pixels[row * 9 + col + 1])
    return value


def hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def image_files(directory: Path) -> list[Path]:
    return sorted(path for path in directory.glob("*") if path.suffix.lower() in IMAGE_SUFFIXES)


def validate_pair(image: Path, label: Path) -> tuple[list[str], Counter]:
    errors: list[str] = []
    counts: Counter = Counter()
    try:
        Image.open(image).verify()
    except Exception as exc:
        return [f"unreadable image {image.name}: {exc}"], counts
    lines = label.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, start=1):
        parts = line.split()
        if len(parts) < 7 or len(parts[1:]) % 2:
            errors.append(f"{label.name}:{line_number}: expected polygon coordinates")
            continue
        try:
            class_id = int(parts[0])
            values = [float(value) for value in parts[1:]]
        except ValueError:
            errors.append(f"{label.name}:{line_number}: non-numeric label")
            continue
        if class_id not in CLASS_NAMES:
            errors.append(f"{label.name}:{line_number}: invalid class id {class_id}")
        if any(value < 0.0 or value > 1.0 for value in values):
            errors.append(f"{label.name}:{line_number}: coordinate outside [0, 1]")
        counts[class_id] += 1
    return errors, counts


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare reviewed helmet data for an isolated experiment")
    parser.add_argument("--dataset", default="data/Construction Site.v1i.yolov8")
    parser.add_argument("--hard-negatives", default="data/hard_negatives")
    parser.add_argument("--output", default="data/helmet_reviewed_experiment")
    parser.add_argument("--manifest", default="data/hard_negatives/review_manifest.json")
    parser.add_argument("--min-reviewed", type=int, default=12)
    parser.add_argument("--near-duplicate-distance", type=int, default=4)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    dataset = (root / args.dataset).resolve()
    hard_root = (root / args.hard_negatives).resolve()
    output = (root / args.output).resolve()
    manifest_path = (root / args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("entries", [])
    if not isinstance(entries, list):
        raise SystemExit("Manifest entries must be a list")

    official_images = [image for split in ("train", "valid", "test") for image in image_files(dataset / split / "images")]
    official_sha = {sha256(image): image for image in official_images}
    official_dhash = [(dhash(image), image) for image in official_images]
    reviewed = []
    errors = []
    category_counts = Counter()
    seen_stems = set()
    for entry in entries:
        image = hard_root / "images" / entry.get("image", "")
        label = hard_root / "labels" / entry.get("label", "")
        if entry.get("status") != "reviewed":
            continue
        if not image.is_file() or not label.is_file():
            errors.append(f"reviewed entry missing image or label: {entry}")
            continue
        if image.stem in seen_stems:
            errors.append(f"duplicate reviewed stem: {image.stem}")
        seen_stems.add(image.stem)
        pair_errors, _ = validate_pair(image, label)
        errors.extend(pair_errors)
        if sha256(image) in official_sha:
            errors.append(f"reviewed image duplicates official split: {image.name}")
        image_hash = dhash(image)
        for existing_hash, existing_image in official_dhash:
            if hamming(image_hash, existing_hash) <= args.near_duplicate_distance:
                errors.append(f"reviewed image is near-duplicate of official image: {image.name} vs {existing_image.name}")
        categories = set(entry.get("categories", []))
        category_counts.update(categories)
        reviewed.append((image, label, entry))

    print(f"reviewed_hard_negative_images={len(reviewed)}")
    print(f"reviewed_categories={dict(sorted(category_counts.items()))}")
    missing_categories = sorted(REQUIRED_CATEGORIES - set(category_counts))
    if len(reviewed) < args.min_reviewed:
        errors.append(f"only {len(reviewed)} reviewed images; need at least {args.min_reviewed}")
    if missing_categories:
        errors.append(f"missing required reviewed categories: {', '.join(missing_categories)}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        print("No experiment dataset was created. Review and add varied labelled images first.")
        return 1

    if output.exists():
        raise SystemExit(f"Refusing to overwrite existing experiment directory: {output}")
    for split in ("train", "valid", "test"):
        for kind in ("images", "labels"):
            (output / split / kind).mkdir(parents=True, exist_ok=True)
            for source in (dataset / split / kind).glob("*"):
                if source.is_file():
                    shutil.copy2(source, output / split / kind / source.name)
    for image, label, _ in reviewed:
        shutil.copy2(image, output / "train" / "images" / image.name)
        shutil.copy2(label, output / "train" / "labels" / label.name)
    (output / "data.yaml").write_text(
        "train: train/images\nval: valid/images\ntest: test/images\n\n"
        "nc: 4\nnames: ['helmet', 'vest', 'without_helmet', 'without_vest']\n",
        encoding="utf-8",
    )

    print(f"experiment_dataset={output}")
    for split in ("train", "valid", "test"):
        image_count = len(image_files(output / split / "images"))
        label_count = len(list((output / split / "labels").glob("*.txt")))
        print(f"{split}_images={image_count} {split}_labels={label_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
