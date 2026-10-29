"""Train an isolated two-class head-crop classifier from reviewed folders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.head_classifier import CLASS_NAMES


def main() -> int:
    parser = argparse.ArgumentParser(description="Train helmet_worn/no_helmet head classifier")
    parser.add_argument("--data", default="data/head_classifier")
    parser.add_argument("--output", default="runs/head_classifier")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--min-images-per-class", type=int, default=20)
    args = parser.parse_args()

    root = ROOT
    data = (root / args.data).resolve()
    output = (root / args.output).resolve()
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    train = datasets.ImageFolder(str(data / "train"), transform=transform)
    val = datasets.ImageFolder(str(data / "val"), transform=eval_transform)
    test = datasets.ImageFolder(str(data / "test"), transform=eval_transform)
    expected = list(CLASS_NAMES)
    if train.classes != expected or val.classes != expected or test.classes != expected:
        raise SystemExit(f"Expected class folders {expected}; found train={train.classes}, val={val.classes}, test={test.classes}")
    counts = {name: train.targets.count(index) for index, name in enumerate(train.classes)}
    print(f"train={len(train)} val={len(val)} test={len(test)} train_class_counts={counts}")
    if min(counts.values(), default=0) < args.min_images_per_class:
        raise SystemExit(f"Insufficient reviewed training data; need {args.min_images_per_class} images per class")
    if len(val) == 0 or len(test) == 0:
        raise SystemExit("Validation and test folders must both contain reviewed images")

    device = torch.device(args.device)
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(expected))
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    train_loader = DataLoader(train, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val, batch_size=args.batch, shuffle=False, num_workers=0)
    best_accuracy = -1.0
    output.mkdir(parents=True, exist_ok=False)
    for epoch in range(args.epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                predictions = model(images.to(device)).argmax(dim=1).cpu()
                correct += int((predictions == labels).sum())
                total += len(labels)
        accuracy = correct / total if total else 0.0
        print(f"epoch={epoch + 1} val_accuracy={accuracy:.4f}")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save({"state_dict": model.state_dict(), "classes": expected, "image_size": args.image_size}, output / "best.pt")
    (output / "training_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(f"candidate_checkpoint={output / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
