"""Optional two-class classifier for worn helmets versus no helmet head crops."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms

from src.head_localization import HeadRegionLocalizer


CLASS_NAMES = ("helmet_worn", "no_helmet")


class HeadHelmetClassifier:
    def __init__(self, checkpoint_path: str, config: dict):
        settings = config.get("head_classifier", {})
        self.device = torch.device(settings.get("device", "cpu"))
        self.image_size = int(settings.get("image_size", 224))
        self.min_confidence = float(settings.get("min_confidence", 0.80))
        self.localizer = HeadRegionLocalizer(
            scale_factor=float(settings.get("face_scale_factor", 1.1)),
            min_neighbors=int(settings.get("face_min_neighbors", 5)),
            use_top_region_fallback=bool(settings.get("use_top_region_fallback", True)),
            top_region_ratio=float(settings.get("top_region_ratio", 0.28)),
        )
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model = models.resnet18(weights=None)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, len(CLASS_NAMES))
        state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        self.model.load_state_dict(state_dict)
        self.model.to(self.device).eval()
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def predict_crop(self, crop: np.ndarray) -> dict:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = self.transform(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probabilities = torch.softmax(self.model(tensor), dim=1)[0]
        confidence, class_index = torch.max(probabilities, dim=0)
        confidence_value = float(confidence.cpu())
        label = CLASS_NAMES[int(class_index.cpu())]
        return {
            "label": label,
            "confidence": confidence_value,
            "probabilities": {name: float(probabilities[index].cpu()) for index, name in enumerate(CLASS_NAMES)},
            "status": label if confidence_value >= self.min_confidence else "needs_review",
        }

    def verify_person(self, image: np.ndarray, person_bbox: list[float]) -> dict:
        localized = self.localizer.locate(image, person_bbox)
        if not localized.reliable or localized.crop is None:
            return {"status": "needs_review", "reason": localized.reason, "bbox": localized.bbox}
        prediction = self.predict_crop(localized.crop)
        prediction.update({"bbox": localized.bbox, "localization_reason": localized.reason})
        return prediction

    def verify_persons(self, image: np.ndarray, persons: list[dict]) -> list[dict]:
        return [self.verify_person(image, person["bbox"]) for person in persons]
