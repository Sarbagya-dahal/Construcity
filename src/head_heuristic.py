"""Geometry-only head-region helmet heuristic.

Determines whether a detected helmet overlaps the localized head-crop region
sufficiently to count as worn.  Applies no colour rules — helmets come in all
colours.  Uses only bounding-box overlap (IoU) between the associated helmet
detection and the head-crop bbox produced by HeadRegionLocalizer.

Two confidence tiers:
  - face_anchor: Haar cascade found a face; medium IoU threshold.
  - top_region_fallback: Top fraction of person box used; stricter IoU required
    for a helmet_worn verdict, otherwise result is needs_review.
"""

from __future__ import annotations

from src.head_localization import HeadCropResult


class HeadHeuristic:
    """Evaluate helmet compliance from a localized head crop and helmet detections.

    Parameters
    ----------
    config:
        Full pipeline config dict.  Reads from ``config["head_heuristic"]``.
    """

    def __init__(self, config: dict):
        settings = config.get("head_heuristic", {})
        # IoU threshold for face-anchor crops (reliable localization).
        self.face_anchor_iou_threshold = float(
            settings.get("face_anchor_iou_threshold", 0.10)
        )
        # Stricter threshold for top-region fallback (less reliable localization).
        self.fallback_iou_threshold = float(
            settings.get("fallback_iou_threshold", 0.30)
        )

    @staticmethod
    def _iou(a: list[float], b: list[float]) -> float:
        """Compute intersection-over-union for two [x1,y1,x2,y2] boxes."""
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        if x2 <= x1 or y2 <= y1:
            return 0.0
        inter = (x2 - x1) * (y2 - y1)
        area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
        area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def evaluate(
        self,
        head_crop: HeadCropResult,
        associated_helmets: list[dict],
    ) -> dict:
        """Return a verdict dict with keys ``status``, ``reason``, ``confidence``.

        Parameters
        ----------
        head_crop:
            Result from HeadRegionLocalizer.locate().
        associated_helmets:
            Helmet detections already associated with this person (after
            ViolationDetector region filtering).  Each dict must have ``bbox``.

        Returns
        -------
        dict with:
          status: "helmet_worn" | "no_helmet" | "needs_review"
          reason: human-readable explanation
          confidence: float (0–1; max IoU when helmet worn, else 0)
        """
        if not head_crop.reliable:
            return {
                "status": "needs_review",
                "reason": head_crop.reason,
                "confidence": 0.0,
            }

        is_fallback = head_crop.reason == "top_region_fallback"
        threshold = (
            self.fallback_iou_threshold if is_fallback else self.face_anchor_iou_threshold
        )
        head_bbox = head_crop.bbox  # [x1, y1, x2, y2] in image coordinates

        best_iou = 0.0
        for helmet in associated_helmets:
            iou = self._iou(head_bbox, helmet["bbox"])
            if iou > best_iou:
                best_iou = iou

        if best_iou >= threshold:
            return {
                "status": "helmet_worn",
                "reason": "helmet_overlaps_head_region",
                "confidence": best_iou,
            }

        # No helmet overlapped the head crop sufficiently.
        if is_fallback:
            # Less certain — do not report a confirmed violation.
            return {
                "status": "needs_review",
                "reason": "fallback_no_helmet_overlap",
                "confidence": 0.0,
            }

        # Face-anchored crop is reliable — no overlap means no helmet detected at head.
        return {
            "status": "no_helmet",
            "reason": "no_helmet_overlaps_head_region",
            "confidence": 0.0,
        }

