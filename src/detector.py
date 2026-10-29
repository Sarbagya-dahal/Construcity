"""Dual-model PPE detector using YOLOv8 person + custom PPE models."""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from ultralytics import YOLO


class PPEDetector:
    """Run person detection and PPE detection with separate YOLO models."""

    def __init__(self, model_path: str, config: dict):
        self.config = config
        self.model_path = model_path
        model_config = config["model"]

        self.conf_threshold = float(model_config["confidence_threshold"])
        self.iou_threshold = float(model_config["iou_threshold"])
        self.device = model_config["device"]

        self.person_model_path = model_config.get("person_model_path", "yolov8n.pt")
        self.person_conf_threshold = float(model_config.get("person_confidence_threshold", 0.35))
        self.person_class_name = model_config.get("person_class_name", "person")

        self.class_conf_thresholds = {
            self._normalize_class_name(name): float(value)
            for name, value in model_config.get("class_confidence_thresholds", {}).items()
        }
        self.min_box_area_ratio = float(model_config.get("min_box_area_ratio", 0.0025))
        self.max_box_area_ratio = float(model_config.get("max_box_area_ratio", 0.08))
        self.min_box_width_ratio = float(model_config.get("min_box_width_ratio", 0.03))
        self.min_box_height_ratio = float(model_config.get("min_box_height_ratio", 0.03))
        self.person_max_box_area_ratio = float(
            model_config.get("person_max_box_area_ratio", 0.90)
        )
        self.class_max_box_area_ratios = {
            self._normalize_class_name(name): float(value)
            for name, value in model_config.get("class_max_box_area_ratios", {}).items()
        }
        self.helmet_upper_region_ratio = float(model_config.get("helmet_upper_region_ratio", 0.65))
        self.helmet_head_region_ratio = float(model_config.get("helmet_head_region_ratio", 0.25))
        self.helmet_min_aspect_ratio = float(model_config.get("helmet_min_aspect_ratio", 0.5))
        self.helmet_max_aspect_ratio = float(model_config.get("helmet_max_aspect_ratio", 2.0))
        self.helmet_min_intensity = float(model_config.get("helmet_min_intensity", 75.0))
        self.helmet_temporal_frames = int(model_config.get("helmet_temporal_frames", 3))
        self.temporal_iou_threshold = float(model_config.get("temporal_iou_threshold", 0.3))
        self._helmet_tracks: List[dict] = []
        self.last_diagnostics: Dict = {}

        print(f"Loading PPE YOLO model from: {model_path}")
        self.ppe_model = YOLO(model_path)
        self._move_model_to_device(self.ppe_model)

        person_model_file = Path(self.person_model_path)
        if not person_model_file.is_absolute():
            person_model_file = Path(model_config.get("person_model_path", "yolov8n.pt"))

        print(f"Loading person YOLO model from: {person_model_file}")
        self.person_model = YOLO(str(person_model_file))
        self._move_model_to_device(self.person_model)

        self.ppe_class_names = self._build_class_map(config.get("classes", {}), self.ppe_model.names)
        print("PPE Detector initialized successfully")

    def _move_model_to_device(self, model: YOLO) -> None:
        """Move YOLO model to the configured device."""
        if self.device == "cpu":
            model.to("cpu")
        elif torch.cuda.is_available():
            model.to(self.device)
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA not available, using CPU")
            model.to("cpu")

    @staticmethod
    def _normalize_class_name(name: str) -> str:
        """Normalize labels into stable pipeline names."""
        normalized = str(name).strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "hardhat": "helmet",
            "safety_helmet": "helmet",
            "helmet_on": "helmet",
            "safety_vest": "vest",
            "reflective_vest": "vest",
            "no_helmet": "without_helmet",
            "missing_helmet": "without_helmet",
            "no_vest": "without_vest",
            "missing_vest": "without_vest",
        }
        return aliases.get(normalized, normalized)

    def _build_class_map(self, configured_classes: dict, model_names: dict) -> Dict[int, str]:
        """Build class-name mapping for the PPE model."""
        class_map: Dict[int, str] = {}
        for name, class_id in configured_classes.items():
            class_map[int(class_id)] = self._normalize_class_name(name)
        for class_id, model_name in model_names.items():
            class_map.setdefault(int(class_id), self._normalize_class_name(model_name))
        return class_map

    @staticmethod
    def _empty_detections() -> Dict[str, List[dict]]:
        """Return a stable detection structure."""
        return {
            "person": [],
            "helmet": [],
            "vest": [],
            "without_helmet": [],
            "without_vest": [],
        }

    def _size_filter_reasons(
        self,
        bbox: List[float],
        image_shape: tuple[int, int, int],
        max_area_ratio: Optional[float] = None,
    ) -> List[str]:
        """Return explicit reasons a box fails the configured size limits."""
        image_height, image_width = image_shape[:2]
        x1, y1, x2, y2 = bbox
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        area = width * height
        image_area = float(image_width * image_height)

        reasons = []
        if width < image_width * self.min_box_width_ratio:
            reasons.append("width_below_minimum")
        if height < image_height * self.min_box_height_ratio:
            reasons.append("height_below_minimum")
        if area < image_area * self.min_box_area_ratio:
            reasons.append("area_below_minimum")
        maximum = self.max_box_area_ratio if max_area_ratio is None else max_area_ratio
        if maximum is not None and area > image_area * maximum:
            reasons.append("area_above_maximum")
        return reasons

    def _passes_size_filters(
        self,
        bbox: List[float],
        image_shape: tuple[int, int, int],
    ) -> bool:
        """Reject boxes outside the PPE size limits."""
        return not self._size_filter_reasons(bbox, image_shape)

    def _passes_person_size_filters(
        self,
        bbox: List[float],
        image_shape: tuple[int, int, int],
    ) -> bool:
        """Apply size limits appropriate for a full person box."""
        return not self._size_filter_reasons(
            bbox, image_shape, max_area_ratio=self.person_max_box_area_ratio
        )

    @staticmethod
    def _bbox_iou(box1: List[float], box2: List[float]) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        inter = (x2 - x1) * (y2 - y1)
        area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
        area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _bbox_center(bbox: List[float]) -> tuple[float, float]:
        return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)

    def _helmet_in_person_head_region(self, helmet_bbox: List[float], person_bbox: List[float]) -> bool:
        """Accept helmets only inside the top head region of a person box."""
        person_x1, person_y1, person_x2, person_y2 = person_bbox
        head_y2 = person_y1 + (person_y2 - person_y1) * self.helmet_head_region_ratio
        center_x, center_y = self._bbox_center(helmet_bbox)
        return person_x1 <= center_x <= person_x2 and person_y1 <= center_y <= head_y2

    def _helmet_has_valid_shape(self, helmet_bbox: List[float]) -> bool:
        width = max(1.0, helmet_bbox[2] - helmet_bbox[0])
        height = max(1.0, helmet_bbox[3] - helmet_bbox[1])
        aspect_ratio = width / height
        return self.helmet_min_aspect_ratio <= aspect_ratio <= self.helmet_max_aspect_ratio

    def _helmet_has_valid_intensity(self, helmet_bbox: List[float], image: np.ndarray) -> bool:
        x1, y1, x2, y2 = [int(round(value)) for value in helmet_bbox]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.shape[1], x2)
        y2 = min(image.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            return False

        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            return False

        mean_intensity = float(roi.mean())
        return mean_intensity >= self.helmet_min_intensity

    def _filter_helmet_detections(
        self,
        helmets: List[dict],
        persons: List[dict],
        image: np.ndarray,
    ) -> List[dict]:
        """Apply helmet-specific false-positive suppression."""
        filtered: List[dict] = []
        image_height = image.shape[0]

        for helmet in helmets:
            bbox = helmet["bbox"]
            confidence = helmet.get("confidence", helmet.get("conf", 0.0))
            reasons = []

            if confidence <= max(0.6, self.class_conf_thresholds.get("helmet", self.conf_threshold)):
                reasons.append("helmet_confidence_not_strictly_above_threshold")
            reasons.extend(
                self._size_filter_reasons(
                    bbox,
                    image.shape,
                    max_area_ratio=self.class_max_box_area_ratios.get("helmet"),
                )
            )
            if not self._helmet_has_valid_shape(bbox):
                reasons.append("invalid_helmet_aspect_ratio")
            if not self._helmet_has_valid_intensity(bbox, image):
                reasons.append("helmet_intensity_below_minimum")

            if persons:
                if not any(self._helmet_in_person_head_region(bbox, person["bbox"]) for person in persons):
                    reasons.append("outside_all_person_head_regions")
            else:
                helmet_center_y = (bbox[1] + bbox[3]) / 2.0
                if helmet_center_y > image_height * self.helmet_upper_region_ratio:
                    reasons.append("below_upper_image_region")

            if reasons:
                self.last_diagnostics.setdefault("ppe", {}).setdefault(
                    "helmet_post_filter_rejections", []
                ).append({"detection": helmet, "reasons": reasons})
                continue

            filtered.append(helmet)

        print("Filtered Helmet Detections:", filtered)
        return filtered

    def _apply_helmet_temporal_smoothing(
        self,
        helmets: List[dict],
        enable_temporal: bool,
    ) -> List[dict]:
        """Confirm helmet detections only after consecutive frames."""
        if not enable_temporal:
            self._helmet_tracks = []
            return helmets

        updated_tracks: List[dict] = []
        confirmed: List[dict] = []

        for helmet in helmets:
            best_track = None
            best_iou = 0.0
            for track in self._helmet_tracks:
                iou = self._bbox_iou(helmet["bbox"], track["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_track = track

            streak = 1
            if best_track is not None and best_iou >= self.temporal_iou_threshold:
                streak = best_track["streak"] + 1

            track = {"bbox": helmet["bbox"], "streak": streak, "detection": helmet}
            updated_tracks.append(track)

            if streak >= self.helmet_temporal_frames:
                confirmed.append(helmet)

        self._helmet_tracks = updated_tracks
        return confirmed

    def _passes_class_filters(
        self,
        class_name: str,
        confidence: float,
        bbox: List[float],
        image_shape: tuple[int, int, int],
    ) -> bool:
        """Apply confidence, size, and helmet-position heuristics."""
        if confidence < self.class_conf_thresholds.get(class_name, self.conf_threshold):
            return False
        if self._class_filter_reasons(class_name, confidence, bbox, image_shape):
            return False

        return True

    def _class_filter_reasons(
        self,
        class_name: str,
        confidence: float,
        bbox: List[float],
        image_shape: tuple[int, int, int],
    ) -> List[str]:
        reasons = []
        if confidence < self.class_conf_thresholds.get(class_name, self.conf_threshold):
            reasons.append("confidence_below_class_threshold")
        reasons.extend(
            self._size_filter_reasons(
                bbox,
                image_shape,
                max_area_ratio=self.class_max_box_area_ratios.get(class_name),
            )
        )
        return reasons

    @staticmethod
    def _build_detection(
        bbox: List[float],
        confidence: float,
        class_id: int,
        class_name: str,
        source: str,
    ) -> dict:
        return {
            "bbox": bbox,
            "confidence": confidence,
            "conf": confidence,
            "class_id": class_id,
            "class_name": class_name,
            "source": source,
        }

    def _run_person_model(self, image: np.ndarray) -> List[dict]:
        """Detect people with the pretrained YOLO model."""
        detections: List[dict] = []
        results = self.person_model.predict(
            image,
            conf=self.person_conf_threshold,
            iou=self.iou_threshold,
            classes=[0],
            verbose=False,
        )

        if not results:
            return detections

        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = float(box.conf[0].cpu().numpy())
            bbox = [float(x1), float(y1), float(x2), float(y2)]

            reasons = self._size_filter_reasons(
                bbox, image.shape, max_area_ratio=self.person_max_box_area_ratio
            )
            self.last_diagnostics.setdefault("person", {}).setdefault("raw", []).append(
                {
                    "detection": self._build_detection(
                        bbox, confidence, 0, self.person_class_name, "person_model"
                    ),
                    "reasons": reasons,
                }
            )
            if reasons:
                continue

            detections.append(
                self._build_detection(
                    bbox=bbox,
                    confidence=confidence,
                    class_id=0,
                    class_name=self.person_class_name,
                    source="person_model",
                )
            )

        return detections

    def _run_ppe_model(self, image: np.ndarray) -> tuple[Dict[str, List[dict]], Dict[str, List[dict]]]:
        """Detect PPE classes and apply post-filters."""
        raw_detections = self._empty_detections()
        filtered_detections = self._empty_detections()

        results = self.ppe_model.predict(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )

        if not results:
            return raw_detections, filtered_detections

        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            class_name = self.ppe_class_names.get(
                cls_id,
                self._normalize_class_name(self.ppe_model.names.get(cls_id, cls_id)),
            )

            if class_name not in filtered_detections:
                continue

            bbox = [float(x1), float(y1), float(x2), float(y2)]
            detection = self._build_detection(
                bbox=bbox,
                confidence=confidence,
                class_id=cls_id,
                class_name=class_name,
                source="ppe_model",
            )
            raw_detections[class_name].append(detection)

            reasons = self._class_filter_reasons(class_name, confidence, bbox, image.shape)
            self.last_diagnostics.setdefault("ppe", {}).setdefault("raw", {}).setdefault(
                class_name, []
            ).append({"detection": detection, "reasons": reasons})
            if not reasons:
                filtered_detections[class_name].append(detection)

        return raw_detections, filtered_detections

    def reset_temporal_state(self) -> None:
        """Reset video/webcam smoothing state."""
        self._helmet_tracks = []

    def detect(self, image: np.ndarray, enable_temporal: bool = False) -> Dict[str, List[dict]]:
        """Run both models and merge detections into one structured output."""
        self.last_diagnostics = {"person": {}, "ppe": {}}
        person_detections = self._run_person_model(image)
        raw_ppe_detections, filtered_ppe_detections = self._run_ppe_model(image)
        filtered_ppe_detections["helmet"] = self._filter_helmet_detections(
            filtered_ppe_detections["helmet"],
            person_detections,
            image,
        )
        filtered_ppe_detections["helmet"] = self._apply_helmet_temporal_smoothing(
            filtered_ppe_detections["helmet"],
            enable_temporal,
        )

        detections = self._empty_detections()
        detections["person"] = person_detections
        for class_name in ("helmet", "vest", "without_helmet", "without_vest"):
            detections[class_name] = filtered_ppe_detections[class_name]

        raw_debug = {"person": person_detections}
        raw_debug.update({name: raw_ppe_detections[name] for name in raw_ppe_detections if name != "person"})

        print("Detections:", raw_debug)
        print("Filtered Detections:", detections)
        self.last_diagnostics["person"]["accepted_count"] = len(person_detections)
        self.last_diagnostics["ppe"]["accepted_counts"] = {
            name: len(items) for name, items in filtered_ppe_detections.items()
        }
        print("Detection Diagnostics:", self.last_diagnostics)
        return detections

    def detect_batch(self, images: List[np.ndarray]) -> List[Dict[str, List[dict]]]:
        """Run detection on a batch of images."""
        return [self.detect(image) for image in images]
