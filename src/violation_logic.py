"""Violation logic for person-aware PPE compliance checks."""

from typing import Dict, List

import numpy as np

from src.utils import calculate_iou


class ViolationDetector:
    """Evaluate PPE compliance only when a person detection exists."""

    def __init__(self, config: dict):
        self.config = config
        violation_config = config.get("violation", {})
        self.person_ppe_iou_threshold = float(violation_config.get("person_ppe_iou_threshold", 0.05))
        self.person_center_margin = float(violation_config.get("person_center_margin", 0.05))
        self.helmet_head_region_ratio = float(violation_config.get("helmet_head_region_ratio", 0.35))
        self.helmet_worn_confidence_threshold = float(
            violation_config.get("helmet_worn_confidence_threshold", 0.80)
        )
        self.vest_torso_top_ratio = float(violation_config.get("vest_torso_top_ratio", 0.25))
        self.vest_torso_bottom_ratio = float(violation_config.get("vest_torso_bottom_ratio", 0.85))
        self.vest_min_area_fraction = float(violation_config.get("vest_min_area_fraction", 0.005))
        self.vest_require_center_x_inside = bool(violation_config.get("vest_require_center_x_inside", True))

        print("Violation Detector initialized")

    @staticmethod
    def _bbox_center(bbox: List[float]) -> tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def _center_inside_person(self, person_bbox: List[float], item_bbox: List[float]) -> bool:
        px1, py1, px2, py2 = person_bbox
        cx, cy = self._bbox_center(item_bbox)
        margin_x = (px2 - px1) * self.person_center_margin
        margin_y = (py2 - py1) * self.person_center_margin
        return (px1 - margin_x) <= cx <= (px2 + margin_x) and (py1 - margin_y) <= cy <= (py2 + margin_y)

    def _is_associated(self, person_bbox: List[float], item_bbox: List[float]) -> bool:
        """Associate PPE with a person by center containment or IoU."""
        return self._center_inside_person(person_bbox, item_bbox) or (
            calculate_iou(person_bbox, item_bbox) >= self.person_ppe_iou_threshold
        )

    def _center_in_vertical_region(self, person_bbox: List[float], item_bbox: List[float], top: float, bottom: float) -> bool:
        _, py1, _, py2 = person_bbox
        _, center_y = self._bbox_center(item_bbox)
        height = py2 - py1
        return py1 + height * top <= center_y <= py1 + height * bottom

    def _association_detail(self, person_bbox: List[float], item_bbox: List[float], kind: str) -> dict:
        center_inside = self._center_inside_person(person_bbox, item_bbox)
        overlap = calculate_iou(person_bbox, item_bbox)
        if kind == "helmet":
            region_ok = self._center_in_vertical_region(person_bbox, item_bbox, 0.0, self.helmet_head_region_ratio)
        elif kind == "vest":
            region_ok = self._center_in_vertical_region(
                person_bbox, item_bbox, self.vest_torso_top_ratio, self.vest_torso_bottom_ratio
            )
            if region_ok:
                px1, py1, px2, py2 = person_bbox
                person_area = max((px2 - px1) * (py2 - py1), 1.0)
                ix1, iy1, ix2, iy2 = item_bbox
                item_area = (ix2 - ix1) * (iy2 - iy1)
                area_ok = (item_area / person_area) >= self.vest_min_area_fraction
                if self.vest_require_center_x_inside:
                    cx, _ = self._bbox_center(item_bbox)
                    margin_x = (px2 - px1) * self.person_center_margin
                    cx_ok = (px1 - margin_x) <= cx <= (px2 + margin_x)
                else:
                    cx_ok = True
                region_ok = area_ok and cx_ok
        else:
            region_ok = True
        associated = region_ok and (center_inside or overlap >= self.person_ppe_iou_threshold)
        if associated:
            reason = "center_in_person_region" if center_inside else "iou_above_threshold_in_person_region"
        elif not region_ok:
            if kind == "vest":
                reason = "outside_vest_spatial_bounds"
            else:
                reason = f"outside_{kind}_region"
        else:
            reason = "outside_person_or_iou_below_threshold"
        return {"associated": associated, "reason": reason, "center_inside_person": center_inside,
                "iou": overlap, "region": kind}

    def _assign_items(self, persons: List[dict], items: List[dict], kind: str) -> tuple[dict, dict]:
        """Assign each PPE item to at most one person using spatial evidence."""
        assignments = {index: [] for index in range(len(persons))}
        details = {index: [] for index in range(len(persons))}
        candidates = []
        for item_index, item in enumerate(items):
            for person_index, person in enumerate(persons):
                detail = self._association_detail(person["bbox"], item["bbox"], kind)
                detail.update({"item_index": item_index, "confidence": item.get("confidence", item.get("conf", 0.0)),
                               "bbox": item["bbox"]})
                details[person_index].append(detail)
                if detail["associated"]:
                    score = (1.0 if detail["center_inside_person"] else 0.0) + detail["iou"]
                    candidates.append((score, person_index, item_index))
        assigned_items = set()
        for _, person_index, item_index in sorted(candidates, reverse=True):
            if item_index in assigned_items:
                continue
            assignments[person_index].append(item_index)
            assigned_items.add(item_index)
        return assignments, details

    @staticmethod
    def _total_detections(detections: Dict[str, List[dict]]) -> int:
        return sum(len(items) for items in detections.values())

    def check_violations(self, detections: Dict[str, List[dict]], head_verifications: List[dict] | None = None) -> Dict:
        """Check compliance when person detections are available."""
        persons = detections.get("person", [])
        helmets = detections.get("helmet", [])
        vests = detections.get("vest", [])
        total_detections = self._total_detections(detections)

        if not persons:
            unverified_candidates = []
            for class_name in ("without_helmet", "without_vest"):
                for item in detections.get(class_name, []):
                    unverified_candidates.append(
                        {
                            "class_name": class_name,
                            "bbox": item["bbox"],
                            "confidence": item.get("confidence", item.get("conf", 0.0)),
                            "reason": "no_person_detection_for_association",
                        }
                    )
            result = {
                "violations": [],
                "violation_count": 0,
                "compliance_rate": 0.0,
                "person_count": 0,
                "helmet_count": len(helmets),
                "vest_count": len(vests),
                "total_detections": total_detections,
                "person_results": [],
                "unverified_candidates": unverified_candidates,
                "status": "person_missing",
            }
            print("Violation Result:", result)
            return result

        violations: List[dict] = []
        person_results: List[dict] = []
        compliant_persons = 0
        uncertain_persons = 0
        review_persons = 0

        helmet_assignments, helmet_details = self._assign_items(persons, helmets, "helmet")
        vest_assignments, vest_details = self._assign_items(persons, vests, "vest")
        missing_helmet_items = detections.get("without_helmet", [])
        missing_vest_items = detections.get("without_vest", [])
        missing_helmet_assignments, missing_helmet_details = self._assign_items(
            persons, missing_helmet_items, "without_helmet"
        )
        missing_vest_assignments, missing_vest_details = self._assign_items(
            persons, missing_vest_items, "without_vest"
        )

        for index, person in enumerate(persons, start=1):
            person_bbox = person["bbox"]
            person_index = index - 1
            associated_helmets = [
                item for item in helmet_details[person_index]
                if item["item_index"] in helmet_assignments[person_index]
            ]
            worn_helmet_candidates = [
                item for item in associated_helmets
                if item["confidence"] >= self.helmet_worn_confidence_threshold
            ]
            low_confidence_helmets = [
                item for item in associated_helmets
                if item["confidence"] < self.helmet_worn_confidence_threshold
            ]
            has_helmet = bool(worn_helmet_candidates)
            has_vest = bool(vest_assignments[person_index])
            explicit_without_helmet = bool(missing_helmet_assignments[person_index])
            explicit_without_vest = bool(missing_vest_assignments[person_index])
            head_verification = (head_verifications or [])[person_index] if head_verifications and person_index < len(head_verifications) else None

            person_violations: List[str] = []
            uncertain: List[str] = []
            review_reasons: List[str] = []
            helmet_conflict = bool(associated_helmets and explicit_without_helmet)
            if head_verification and head_verification.get("status") == "helmet_worn":
                has_helmet = True
                if explicit_without_helmet:
                    review_reasons.append("head_classifier_detector_conflict")
                    uncertain.append("helmet_status_uncertain")
            elif head_verification and head_verification.get("status") == "no_helmet":
                if associated_helmets or explicit_without_helmet:
                    review_reasons.append("head_classifier_detector_conflict")
                    uncertain.append("helmet_status_uncertain")
                else:
                    person_violations.append("without_helmet")
            elif head_verification and head_verification.get("status") == "needs_review":
                review_reasons.append(f"head_classifier_{head_verification.get('reason', 'uncertain')}")
                uncertain.append("helmet_status_uncertain")
            elif helmet_conflict:
                review_reasons.append("helmet_positive_negative_conflict")
                uncertain.append("helmet_status_uncertain")
            elif explicit_without_helmet:
                person_violations.append("without_helmet")
            elif low_confidence_helmets:
                review_reasons.append("helmet_below_worn_confidence_threshold")
                uncertain.append("helmet_status_uncertain")
            elif not has_helmet:
                uncertain.append("helmet_status_uncertain")
            if explicit_without_vest:
                person_violations.append("without_vest")
            elif not has_vest:
                uncertain.append("vest_status_uncertain")

            needs_review = bool(review_reasons)
            if needs_review:
                review_persons += 1
            if uncertain:
                uncertain_persons += 1
            elif not person_violations and not needs_review:
                compliant_persons += 1

            person_results.append(
                {
                    "person_id": index,
                    "bbox": person_bbox,
                    "has_helmet": has_helmet,
                    "has_vest": has_vest,
                    "violations": person_violations,
                    "uncertain": uncertain,
                    "needs_review": needs_review,
                    "review_reasons": review_reasons,
                    "head_verification": head_verification,
                    "associations": {
                        "helmet": associated_helmets,
                        "vest": [item for item in vest_details[person_index] if item["item_index"] in vest_assignments[person_index]],
                        "without_helmet": [item for item in missing_helmet_details[person_index] if item["item_index"] in missing_helmet_assignments[person_index]],
                        "without_vest": [item for item in missing_vest_details[person_index] if item["item_index"] in missing_vest_assignments[person_index]],
                    },
                    "source": person.get("source", "unknown"),
                }
            )

            for violation_type in person_violations:
                violations.append(
                    {
                        "type": violation_type,
                        "bbox": person_bbox,
                        "conf": person.get("confidence", person.get("conf", 0.0)),
                        "person_id": index,
                    }
                )

        evaluable_persons = len(persons) - uncertain_persons
        result = {
            "violations": violations,
            "violation_count": len(violations),
            "compliance_rate": (compliant_persons / evaluable_persons) * 100.0 if evaluable_persons else 0.0,
            "person_count": len(persons),
            "evaluable_person_count": evaluable_persons,
            "uncertain_person_count": uncertain_persons,
            "needs_review_person_count": review_persons,
            "helmet_count": len(helmets),
            "vest_count": len(vests),
            "total_detections": total_detections,
            "person_results": person_results,
            "status": (
                "needs_review" if review_persons
                else "evaluated_with_uncertainty" if uncertain_persons
                else "evaluated"
            ),
        }

        print("Violation Result:", result)
        return result

    def get_violation_summary(self, violation_result: Dict) -> str:
        """Generate a readable summary."""
        summary = f"Total Detections: {violation_result.get('total_detections', 0)}\n"
        summary += f"Persons Evaluated: {violation_result.get('person_count', 0)}\n"
        summary += f"Violations: {violation_result.get('violation_count', 0)}\n"
        summary += f"Compliance Rate: {violation_result.get('compliance_rate', 0.0):.1f}%\n"
        return summary

    def create_alert_message(self, violation_result: Dict) -> str:
        """Create a concise alert message."""
        if violation_result.get("status") == "person_missing":
            return "No person detected; compliance not evaluated"
        if violation_result.get("status") == "needs_review":
            return "Helmet evidence needs review for one or more persons"
        if violation_result.get("status") == "evaluated_with_uncertainty":
            return "PPE compliance uncertain for one or more persons"
        if violation_result["violation_count"] == 0:
            return "PPE compliant"
        if violation_result["violation_count"] == 1:
            return "1 PPE violation detected"
        return f"{violation_result['violation_count']} PPE violations detected"


def annotate_violations(
    image: np.ndarray,
    detections: Dict[str, List[dict]],
    violation_result: Dict,
    config: dict,
    diagnostics: Dict | None = None,
) -> np.ndarray:
    """Draw verified detections, unverified candidates, and violations."""
    from src.utils import draw_bbox, format_confidence

    annotated = image.copy()
    colors = config["visualization"]["colors"]
    thickness = config["visualization"]["bbox_thickness"]
    text_size = config["visualization"]["text_size"]

    verified_helmet_indices = set()
    for person in violation_result.get("person_results", []):
        if not person.get("has_helmet"):
            continue
        for association in person.get("associations", {}).get("helmet", []):
            verified_helmet_indices.add(association.get("item_index"))

    CLASS_DISPLAY_NAMES = {
        "helmet": "HELMET",
        "vest": "VEST",
        "without_helmet": "MISSING HELMET",
        "without_vest": "MISSING VEST",
        "person": "PERSON",
    }

    for class_name, items in detections.items():
        for item_index, det in enumerate(items):
            confidence = det.get("confidence", det.get("conf", 0.0))
            display_title = CLASS_DISPLAY_NAMES.get(class_name, class_name.upper())
            label = f"{display_title}: {format_confidence(confidence)}"
            color = tuple(colors.get(class_name, (255, 255, 255)))
            if class_name == "helmet" and item_index not in verified_helmet_indices:
                label = f"UNVERIFIED HELMET CANDIDATE {format_confidence(confidence)}"
                color = (0, 165, 255)
            annotated = draw_bbox(
                annotated,
                det["bbox"],
                label,
                color,
                thickness,
                text_size,
            )

    rejected_helmets = (diagnostics or {}).get("ppe", {}).get("helmet_post_filter_rejections", [])
    displayed_bboxes = {tuple(item["bbox"]) for item in detections.get("helmet", [])}
    for rejected in rejected_helmets:
        det = rejected.get("detection", {})
        bbox = det.get("bbox")
        if not bbox or tuple(bbox) in displayed_bboxes:
            continue
        confidence = det.get("confidence", det.get("conf", 0.0))
        annotated = draw_bbox(
            annotated,
            bbox,
            f"UNVERIFIED HELMET CANDIDATE {format_confidence(confidence)}",
            (0, 165, 255),
            thickness,
            text_size,
        )

    grouped_violations = {}
    for violation in violation_result.get("violations", []):
        key = (violation.get("person_id"), tuple(violation["bbox"]))
        grouped_violations.setdefault(key, []).append(violation["type"])

    for (person_id, bbox), violation_types in grouped_violations.items():
        unique_types = list(dict.fromkeys(violation_types))
        label = "violations: " + ", ".join(item.replace("_", " ") for item in unique_types)
        color_key = unique_types[0] if unique_types else "without_helmet"
        annotated = draw_bbox(
            annotated,
            list(bbox),
            label,
            tuple(colors.get(color_key, (0, 0, 255))),
            thickness + 1,
            text_size,
        )

    return annotated
