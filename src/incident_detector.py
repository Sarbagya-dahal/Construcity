"""Visual incident and blood detection module.

IMPORTANT OPERATIONAL & SAFETY NOTICE:
1. Reliable blood and injury detection is currently UNAVAILABLE in this deployment.
   No validated computer vision model or verified industrial injury dataset is installed.
2. A simple color-based heuristic is DISABLED in production because red-color thresholds
   produce high false-alarm rates on hazard tape, red tools, safety signage, and red garments.
3. The absence of alerts does NOT establish that an image or workspace is blood-free or
   injury-free. When this capability is unavailable, the system explicitly reports
   status as NOT_EVALUATED_UNAVAILABLE rather than asserting an all-clear.
4. An experimental color/texture heuristic is retained solely for offline research and
   debugging behind enable_experimental_heuristic=True (OFF by default). When active,
   it emits RED_REGION_CANDIDATE anomalies and NEVER asserts a blood probability or
   confirmed medical diagnosis.
"""

from __future__ import annotations

from typing import Dict, List, Optional
import cv2
import numpy as np

LIMITATIONS_NOTICE = (
    "Non-medical notice: Reliable blood/injury detection is currently unavailable. "
    "A simple color heuristic is disabled in production to prevent false alerts on hazard tape, "
    "red garments, and tools. This scan has NOT been evaluated for blood or injuries; "
    "absence of findings does NOT confirm the absence of blood or injury."
)


class IncidentDetector:
    """Incident screening interface with production safety guards."""

    def __init__(self, config: dict):
        self.config = config
        settings = config.get("incident_detection", {})
        # Disabled by default in production.
        self.enabled = bool(settings.get("enabled", False))
        self.enable_experimental_heuristic = bool(
            settings.get("enable_experimental_heuristic", False)
        )
        self.min_region_area_px = int(settings.get("min_region_area_px", 60))
        self.max_region_area_ratio = float(settings.get("max_region_area_ratio", 0.20))
        self.texture_std_min = float(settings.get("texture_std_min", 10.0))
        self.urgent_review_threshold = float(settings.get("urgent_review_threshold", 0.70))

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

    def _is_near_person(self, bbox: List[float], persons: List[dict]) -> bool:
        """Check if candidate is within or adjacent to a person bounding box."""
        if not persons:
            return False
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        for p in persons:
            px1, py1, px2, py2 = p["bbox"]
            margin_x = (px2 - px1) * 0.15
            margin_y = (py2 - py1) * 0.15
            if (px1 - margin_x) <= cx <= (px2 + margin_x) and (py1 - margin_y) <= cy <= (py2 + margin_y):
                return True
            if self._bbox_iou(bbox, p["bbox"]) > 0.05:
                return True
        return False

    def detect_incidents(
        self,
        image: np.ndarray,
        persons: Optional[List[dict]] = None,
    ) -> Dict:
        """Evaluate image for incidents or blood-like regions.

        In production, returns NOT_EVALUATED_UNAVAILABLE with an explicit notice that
        reliable detection is unavailable and this scan has NOT been evaluated for blood.
        """
        # Production default: disabled when neither enabled nor experimental heuristic is true
        if not self.enabled and not self.enable_experimental_heuristic:
            return {
                "available": False,
                "validated": False,
                "has_candidate": False,
                "status": "NOT_EVALUATED_UNAVAILABLE",
                "status_display": "Reliable blood detection unavailable (Not evaluated)",
                "message": (
                    "Reliable blood/injury detection is currently unavailable. No validated model is installed. "
                    "This image has NOT been evaluated for blood or physical injuries; the absence of alerts "
                    "does NOT confirm that no blood or injury is present."
                ),
                "candidates": [],
                "limitations": LIMITATIONS_NOTICE,
            }

        # Heuristic path (when enabled in test or experimental mode)
        persons = persons or []
        img_h, img_w = image.shape[:2]
        total_pixels = float(img_h * img_w)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        b = image[:, :, 0].astype(np.float32)
        g = image[:, :, 1].astype(np.float32)
        r = image[:, :, 2].astype(np.float32)

        # Deep crimson/red hue masks in HSV:
        mask1 = cv2.inRange(hsv, np.array([0, 70, 40]), np.array([8, 255, 210]))
        mask2 = cv2.inRange(hsv, np.array([172, 70, 40]), np.array([180, 255, 210]))
        color_mask = cv2.bitwise_or(mask1, mask2)

        # Ratio filter: red channel strongly dominates green and blue
        ratio_mask = (r > 1.35 * g) & (r > 1.35 * b) & (r > 60)

        # Skin tone suppression
        skin_mask = (r > g) & (g > b) & ((r - g) < 65) & (g > 55)

        combined_mask = color_mask & ratio_mask & (~skin_mask)
        combined_mask = (combined_mask * 255).astype(np.uint8)

        # Morphological opening
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        filtered_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(filtered_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_region_area_px:
                continue

            area_ratio = area / total_pixels
            if area_ratio > self.max_region_area_ratio:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            roi_bgr = image[y:y+h, x:x+w]
            roi_mask = filtered_mask[y:y+h, x:x+w] > 0

            if not np.any(roi_mask):
                continue

            masked_pixels = roi_bgr[roi_mask]
            std_val = float(np.std(masked_pixels))
            if std_val < self.texture_std_min:
                # Flat uniform region (hazard tape, tool, banner, uniform fabric)
                continue

            bbox = [float(x), float(y), float(x + w), float(y + h)]
            near_worker = self._is_near_person(bbox, persons)

            # Raw color/texture anomaly score (NEVER blood probability)
            anomaly_score = min(0.85, 0.45 + (std_val / 80.0) * 0.25)
            if near_worker:
                anomaly_score = min(0.92, anomaly_score + 0.15)
            else:
                anomaly_score = max(0.40, anomaly_score - 0.10)

            candidates.append({
                "label": "RED_REGION_CANDIDATE",
                "bbox": bbox,
                "color_texture_anomaly_score": float(anomaly_score),
                "confidence": float(anomaly_score),
                "area_px": int(area),
                "area_ratio": float(area_ratio),
                "texture_std": float(std_val),
                "near_person": bool(near_worker),
                "review_notice": (
                    "Unverified visual candidate; requires human review to assess."
                ),
            })

        has_candidate = len(candidates) > 0
        if not has_candidate:
            status = "NO_CANDIDATE_DETECTED"
        elif any(c["confidence"] >= self.urgent_review_threshold and c["near_person"] for c in candidates):
            status = "POTENTIAL_INJURY_SUSPECTED"
        else:
            status = "RED_REGION_CANDIDATE"

        return {
            "available": True if self.enabled else False,
            "validated": False,
            "has_candidate": has_candidate,
            "status": status,
            "status_display": "Experimental Heuristic Candidate" if has_candidate else "No Candidate Detected",
            "message": (
                "Color/texture heuristic executed. "
                "These candidates do NOT confirm blood or injury."
            ),
            "candidates": candidates,
            "limitations": LIMITATIONS_NOTICE,
        }
