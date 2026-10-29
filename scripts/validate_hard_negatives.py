"""Validate reviewed hard-negative image/label pairs without changing splits."""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from pathlib import Path

from PIL import Image


CLASS_IDS = {0, 1, 2, 3}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def digest(path: Path) -> str:
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate hard-negative annotations")
    parser.add_argument("--root", default="data/hard_negatives")
    parser.add_argument("--dataset", default="data/Construction Site.v1i.yolov8")
    parser.add_argument("--near-duplicate-distance", type=int, default=4)
    args = parser.parse_args()
    root = Path(args.root)
    images = {p.stem: p for p in (root / "images").glob("*") if p.suffix.lower() in IMAGE_SUFFIXES}
    labels = {p.stem: p for p in (root / "labels").glob("*.txt")}
    errors = []
    counts = Counter()
    official_hashes = set()
    official_dhashes = []
    for split in ("train", "valid", "test"):
        for image in ((Path(args.dataset) / split / "images").glob("*")):
            if image.is_file():
                official_hashes.add(digest(image))
                try:
                    official_dhashes.append((dhash(image), image))
                except Exception as exc:
                    errors.append(f"unreadable official image {image}: {exc}")

    for stem in sorted(set(labels) - set(images)):
        errors.append(f"label has no matching image: {labels[stem].name}")

    for stem, image in sorted(images.items()):
        if stem not in labels:
            errors.append(f"missing label file for {image.name}")
            continue
        if digest(image) in official_hashes:
            errors.append(f"duplicate image content appears in official split: {image.name}")
        try:
            image_hash = dhash(image)
            for official_hash, official_image in official_dhashes:
                if hamming(image_hash, official_hash) <= args.near_duplicate_distance:
                    errors.append(f"near-duplicate image appears in official split: {image.name} vs {official_image.name}")
        except Exception as exc:
            errors.append(f"could not fingerprint {image.name}: {exc}")
        try:
            width, height = Image.open(image).size
        except Exception as exc:
            errors.append(f"unreadable image {image.name}: {exc}")
            continue
        for line_number, line in enumerate(labels[stem].read_text(encoding="utf-8").splitlines(), start=1):
            parts = line.split()
            if len(parts) < 7 or len(parts[1:]) % 2:
                errors.append(f"{labels[stem].name}:{line_number}: expected polygon coordinates")
                continue
            try:
                class_id = int(parts[0])
                coordinates = [float(value) for value in parts[1:]]
            except ValueError:
                errors.append(f"{labels[stem].name}:{line_number}: non-numeric label")
                continue
            if class_id not in CLASS_IDS:
                errors.append(f"{labels[stem].name}:{line_number}: invalid class id {class_id}")
            if any(value < 0.0 or value > 1.0 for value in coordinates):
                errors.append(f"{labels[stem].name}:{line_number}: coordinate outside [0, 1]")
            counts[class_id] += 1
        print(f"{image.name}: {width}x{height}, labels={sum(1 for _ in labels[stem].read_text().splitlines())}")

    print(f"class_counts={dict(sorted(counts.items()))}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("Hard-negative annotation validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
