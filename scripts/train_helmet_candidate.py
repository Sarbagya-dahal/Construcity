"""Fine-tune an isolated PPE candidate without touching production weights."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def print_dataset_counts(data: Path) -> None:
    print(f"Training dataset YAML: {data}")
    for split in ("train", "valid", "test"):
        images = list((data.parent / split / "images").glob("*"))
        labels = list((data.parent / split / "labels").glob("*.txt"))
        print(f"{split}: images={len(images)} labels={len(labels)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train an isolated helmet hard-negative candidate")
    parser.add_argument("--data", default="data/helmet_hard_negative_experiment/data.yaml")
    parser.add_argument("--weights", default="models/trained/best.pt")
    parser.add_argument("--project", default="runs")
    parser.add_argument("--name", default="helmet_hard_negative_experiment")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    data = (root / args.data).resolve()
    weights = (root / args.weights).resolve()
    if not data.exists():
        raise SystemExit(f"Dataset YAML not found: {data}")
    print_dataset_counts(data)
    model = YOLO(str(weights))
    model.train(
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=0,
        cache=False,
        project=str((root / args.project).resolve()),
        name=args.name,
        exist_ok=False,
        pretrained=str(weights),
        patience=10,
        seed=42,
        deterministic=True,
        plots=True,
    )
    print(f"Candidate training complete: {(root / args.project / args.name).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
