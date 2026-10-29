"""Head-region localization for person crops.

Face detection is used as an observable anchor and expanded upward to include
the helmet area.  If face detection fails and ``use_top_region_fallback`` is
enabled, the top ``top_region_ratio`` fraction of the person bounding box is
returned as a less-confident crop.  In both cases, if no reliable crop can be
produced, the result is marked uncertain rather than silently assuming the top
of a person box is the head.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Minimum pixel size for a head crop to be considered usable.
_MIN_CROP_PX = 20


@dataclass
class HeadCropResult:
    crop: np.ndarray | None
    bbox: list[float] | None
    reliable: bool
    reason: str


class HeadRegionLocalizer:
    """Localize the head region of a detected person.

    Parameters
    ----------
    scale_factor:
        Haar cascade scaleFactor (default 1.1).
    min_neighbors:
        Haar cascade minNeighbors (default 5).
    use_top_region_fallback:
        When ``True`` (default) and Haar face detection finds no face, fall
        back to the top ``top_region_ratio`` of the person bounding box.  The
        fallback crop is marked with ``reason="top_region_fallback"`` so
        callers can apply a stricter confidence threshold.
    top_region_ratio:
        Fraction of the person bounding-box height to use as the head crop
        when the face-anchor fails (default 0.28).
    """

    def __init__(
        self,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        use_top_region_fallback: bool = True,
        top_region_ratio: float = 0.28,
    ):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_detector = cv2.CascadeClassifier(cascade_path)
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.use_top_region_fallback = use_top_region_fallback
        self.top_region_ratio = top_region_ratio

    @staticmethod
    def _clip_bbox(bbox: list[float], width: int, height: int) -> tuple[int, int, int, int] | None:
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, int(round(x1))), max(0, int(round(y1)))
        x2, y2 = min(width, int(round(x2))), min(height, int(round(y2)))
        return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None

    def locate(self, image: np.ndarray, person_bbox: list[float]) -> HeadCropResult:
        clipped = self._clip_bbox(person_bbox, image.shape[1], image.shape[0])
        if clipped is None:
            return HeadCropResult(None, None, False, "invalid_person_box")
        px1, py1, px2, py2 = clipped
        person_crop = image[py1:py2, px1:px2]
        gray = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(
            gray, scaleFactor=self.scale_factor, minNeighbors=self.min_neighbors
        )
        if len(faces):
            fx, fy, fw, fh = max(faces, key=lambda item: item[2] * item[3])
            # Expand above the face for a worn helmet and slightly below for context.
            hx1 = max(0, int(fx - 0.35 * fw))
            hy1 = max(0, int(fy - 0.70 * fh))
            hx2 = min(person_crop.shape[1], int(fx + 1.35 * fw))
            hy2 = min(person_crop.shape[0], int(fy + 1.25 * fh))
            if hx2 > hx1 and hy2 > hy1:
                absolute = [px1 + hx1, py1 + hy1, px1 + hx2, py1 + hy2]
                return HeadCropResult(person_crop[hy1:hy2, hx1:hx2], absolute, True, "face_anchor")

        # --- Face-anchor failed: optionally use the top-of-person-box fallback ---
        if self.use_top_region_fallback:
            ph, pw = person_crop.shape[:2]
            top_h = int(round(ph * self.top_region_ratio))
            if top_h >= _MIN_CROP_PX and pw >= _MIN_CROP_PX:
                fallback_crop = person_crop[0:top_h, 0:pw]
                absolute = [float(px1), float(py1), float(px2), float(py1 + top_h)]
                return HeadCropResult(fallback_crop, absolute, True, "top_region_fallback")

        return HeadCropResult(None, None, False, "face_not_localized")
