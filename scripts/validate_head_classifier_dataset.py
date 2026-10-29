"""Validate an ImageFolder head-classifier dataset for leakage and completeness."""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from pathlib import Path

from PIL import Image


CLASSES = ("helmet_worn", "no_helmet")
SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dhash(path: Path) -> int:
    pixels = list(Image.open(path).convert("L").resize((9, 8)).getdata())
    value = 0
    for row in range(8):
        for col in range(8):
            value = (value << 1) | int(pixels[row * 9 + col] > pixels[row * 9 + col + 1])
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate head classifier ImageFolder data")
    parser.add_argument("--data", default="data/head_classifier")
    parser.add_argument("--min-per-class", type=int, default=1)
    parser.add_argument("--near-duplicate-distance", type=int, default=4)
    args = parser.parse_args()
    root = Path(args.data)
    errors = []
    hashes = {}
    perceptual = {}
    counts = Counter()
    for split in ("train", "val", "test"):
        for class_name in CLASSES:
            folder = root / split / class_name
            files = [p for p in folder.glob("*") if p.suffix.lower() in SUFFIXES] if folder.exists() else []
            counts[(split, class_name)] = len(files)
            if len(files) < args.min_per_class:
                errors.append(f"{split}/{class_name} has {len(files)} images; need {args.min_per_class}")
            for image in files:
                try:
                    Image.open(image).verify()
                    file_hash = digest(image)
                    image_hash = dhash(image)
                except Exception as exc:
                    errors.append(f"unreadable image {image}: {exc}")
                    continue
                if file_hash in hashes:
                    errors.append(f"exact duplicate across dataset: {image} and {hashes[file_hash]}")
                hashes[file_hash] = image
                for previous_hash, previous in perceptual.items():
                    if (image_hash ^ previous_hash).bit_count() <= args.near_duplicate_distance and image.parent.parent.name != previous.parent.parent.name:
                        errors.append(f"near-duplicate across splits: {image} and {previous}")
                perceptual[image_hash] = image
    print(f"counts={dict((f'{split}/{name}', count) for (split, name), count in counts.items())}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("Head classifier dataset validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
