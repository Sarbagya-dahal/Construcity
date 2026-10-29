import unittest
import io
import contextlib
from types import SimpleNamespace
from pathlib import Path

import numpy as np

from src.detector import PPEDetector
from src.violation_logic import ViolationDetector


def make_detector():
    detector = object.__new__(PPEDetector)
    detector.min_box_area_ratio = 0.0025
    detector.max_box_area_ratio = 0.08
    detector.min_box_width_ratio = 0.03
    detector.min_box_height_ratio = 0.03
    detector.person_max_box_area_ratio = 0.5
    detector.class_max_box_area_ratios = {"vest": 0.25, "without_vest": 0.8}
    detector.class_conf_thresholds = {"helmet": 0.6, "vest": 0.5}
    detector.conf_threshold = 0.5
    detector.helmet_head_region_ratio = 0.25
    detector.helmet_upper_region_ratio = 0.65
    detector.helmet_min_aspect_ratio = 0.5
    detector.helmet_max_aspect_ratio = 2.0
    detector.helmet_min_intensity = 75.0
    detector.last_diagnostics = {}
    return detector


def detection(bbox, confidence=0.9, class_name="helmet"):
    return {
        "bbox": bbox,
        "confidence": confidence,
        "conf": confidence,
        "class_name": class_name,
    }


class DetectionPipelineTests(unittest.TestCase):
    def test_person_area_limit_is_separate_from_ppe_area_limit(self):
        detector = make_detector()
        image_shape = (416, 416, 3)
        worker = [54.0, 153.0, 165.0, 336.0]

        self.assertFalse(detector._passes_size_filters(worker, image_shape))
        self.assertTrue(detector._passes_person_size_filters(worker, image_shape))
        self.assertEqual(
            detector._size_filter_reasons(worker, image_shape), ["area_above_maximum"]
        )

    def test_small_distant_worker_is_rejected_with_explicit_size_reason(self):
        detector = make_detector()
        image_shape = (1080, 1920, 3)
        small_worker = [10.0, 10.0, 45.0, 70.0]

        self.assertFalse(detector._passes_person_size_filters(small_worker, image_shape))
        reasons = detector._size_filter_reasons(
            small_worker, image_shape, max_area_ratio=detector.person_max_box_area_ratio
        )
        self.assertIn("width_below_minimum", reasons)
        self.assertIn("area_below_minimum", reasons)

    def test_boundary_box_is_clipped_safely_for_helmet_roi(self):
        detector = make_detector()
        image = np.full((100, 100, 3), 200, dtype=np.uint8)

        self.assertTrue(detector._helmet_has_valid_intensity([-5, -5, 10, 10], image))

    def test_dark_helmet_is_rejected_as_false_positive_with_reason(self):
        detector = make_detector()
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        helmet = detection([20, 10, 40, 25], 0.95)

        self.assertEqual(detector._filter_helmet_detections([helmet], [], image), [])
        reasons = detector.last_diagnostics["ppe"]["helmet_post_filter_rejections"][0]["reasons"]
        self.assertIn("helmet_intensity_below_minimum", reasons)

    def test_large_vest_is_accepted_by_class_specific_limit(self):
        detector = make_detector()
        image_shape = (1080, 1920, 3)
        vest = [423.5, 774.2, 1508.3, 1060.5]

        self.assertFalse(detector._passes_size_filters(vest, image_shape))
        self.assertEqual(detector._class_filter_reasons("vest", 0.785, vest, image_shape), [])

        without_vest = [585.6, 246.3, 1192.5, 1020.8]
        self.assertEqual(
            detector._class_filter_reasons("without_vest", 0.852, without_vest, image_shape), []
        )

    def test_rejected_ppe_detection_reports_area_reason(self):
        detector = make_detector()
        image_shape = (1080, 1920, 3)
        vest = [423.5, 774.2, 1508.3, 1060.5]

        reasons = detector._class_filter_reasons("helmet", 0.785, vest, image_shape)

        self.assertIn("area_above_maximum", reasons)

    def test_helmet_is_kept_when_associated_with_retained_person(self):
        detector = make_detector()
        image = np.full((416, 416, 3), 200, dtype=np.uint8)
        person = {"bbox": [235.8, 136.9, 314.9, 415.6]}
        helmet = detection([244.2, 138.3, 271.0, 158.9], 0.873)

        filtered = detector._filter_helmet_detections([helmet], [person], image)

        self.assertEqual(filtered, [helmet])
        self.assertNotIn("helmet_post_filter_rejections", detector.last_diagnostics.get("ppe", {}))

    def test_hand_held_helmet_is_not_counted_as_worn(self):
        detector = ViolationDetector(
            {"violation": {
                "person_ppe_iou_threshold": 0.01,
                "person_center_margin": 0.05,
                "helmet_head_region_ratio": 0.35,
                "vest_torso_top_ratio": 0.15,
                "vest_torso_bottom_ratio": 0.95,
            }}
        )
        result = detector.check_violations({
            "person": [detection([0, 0, 100, 200], class_name="person")],
            "helmet": [detection([20, 140, 70, 190], class_name="helmet")],
            "vest": [],
            "without_helmet": [],
            "without_vest": [],
        })

        self.assertFalse(result["person_results"][0]["has_helmet"])
        self.assertIn("helmet_status_uncertain", result["person_results"][0]["uncertain"])
        self.assertEqual(result["violations"], [])

    def test_helmet_worn_compliance_requires_separate_confidence_gate(self):
        detector = ViolationDetector({"violation": {"helmet_worn_confidence_threshold": 0.80}})
        result = detector.check_violations({
            "person": [detection([0, 0, 100, 200], class_name="person")],
            "helmet": [detection([20, 10, 70, 45], 0.75, "helmet")],
            "vest": [], "without_helmet": [], "without_vest": [],
        })

        self.assertFalse(result["person_results"][0]["has_helmet"])
        self.assertEqual(result["status"], "needs_review")
        self.assertIn("helmet_below_worn_confidence_threshold", result["person_results"][0]["review_reasons"])

    def test_conflicting_helmet_predictions_require_review(self):
        detector = ViolationDetector({"violation": {"helmet_worn_confidence_threshold": 0.80}})
        result = detector.check_violations({
            "person": [detection([0, 0, 100, 200], class_name="person")],
            "helmet": [detection([20, 10, 70, 45], 0.90, "helmet")],
            "vest": [],
            "without_helmet": [detection([20, 10, 70, 45], 0.90, "without_helmet")],
            "without_vest": [],
        })

        self.assertTrue(result["person_results"][0]["has_helmet"])
        self.assertEqual(result["violations"], [])
        self.assertEqual(result["status"], "needs_review")
        self.assertIn("helmet_positive_negative_conflict", result["person_results"][0]["review_reasons"])

    def test_head_classifier_no_helmet_can_confirm_missing_helmet(self):
        detector = ViolationDetector({"violation": {}})
        result = detector.check_violations(
            {"person": [detection([0, 0, 100, 200], class_name="person")],
             "helmet": [], "vest": [], "without_helmet": [], "without_vest": []},
            head_verifications=[{"status": "no_helmet", "confidence": 0.95}],
        )

        self.assertEqual(result["person_results"][0]["violations"], ["without_helmet"])
        self.assertEqual(result["violation_count"], 1)

    def test_uncertain_head_classifier_preserves_needs_review(self):
        detector = ViolationDetector({"violation": {}})
        result = detector.check_violations(
            {"person": [detection([0, 0, 100, 200], class_name="person")],
             "helmet": [], "vest": [], "without_helmet": [], "without_vest": []},
            head_verifications=[{"status": "needs_review", "reason": "face_not_localized"}],
        )

        self.assertEqual(result["violations"], [])
        self.assertEqual(result["status"], "needs_review")

    def test_head_localizer_marks_unanchored_boundary_crop_uncertain(self):
        from src.head_localization import HeadRegionLocalizer

        result = HeadRegionLocalizer().locate(np.zeros((80, 80, 3), dtype=np.uint8), [-20, -10, 35, 45])

        self.assertFalse(result.reliable)
        self.assertIsNone(result.crop)
        self.assertIn(result.reason, {"face_not_localized", "invalid_person_box"})

    def test_below_threshold_helmet_is_drawn_as_unverified_candidate(self):
        import src.violation_logic as violation_logic
        import src.utils as utils

        detector = ViolationDetector({"violation": {"helmet_worn_confidence_threshold": 0.80}})
        detections = {
            "person": [detection([0, 0, 100, 200], class_name="person")],
            "helmet": [detection([20, 10, 70, 45], 0.75, "helmet")],
            "vest": [], "without_helmet": [], "without_vest": [],
        }
        result = detector.check_violations(detections)
        calls = []
        original_draw_bbox = utils.draw_bbox
        utils.draw_bbox = lambda image, bbox, label, color, thickness, text_size: calls.append(label) or image
        try:
            violation_logic.annotate_violations(
                np.zeros((100, 100, 3), dtype=np.uint8),
                detections,
                result,
                {"visualization": {"colors": {}, "bbox_thickness": 2, "text_size": 0.6}},
            )
        finally:
            utils.draw_bbox = original_draw_bbox

        self.assertTrue(any(label.startswith("UNVERIFIED HELMET CANDIDATE") for label in calls))
        self.assertNotIn("helmet 75.0%", calls)

    def test_helmet_rejection_reports_missing_person_association(self):
        detector = make_detector()
        image = np.full((416, 416, 3), 200, dtype=np.uint8)
        helmet = detection([244.2, 138.3, 271.0, 158.9], 0.873)

        self.assertEqual(detector._filter_helmet_detections([helmet], [{"bbox": [349, 208, 371, 236]}], image), [])
        reasons = detector.last_diagnostics["ppe"]["helmet_post_filter_rejections"][0]["reasons"]
        self.assertIn("outside_all_person_head_regions", reasons)

    def test_violation_count_is_per_person_and_per_missing_item(self):
        detector = ViolationDetector(
            {"violation": {"person_ppe_iou_threshold": 0.01, "person_center_margin": 0.05}}
        )
        detections = {
            "person": [
                detection([0, 0, 100, 200], class_name="person"),
                detection([200, 0, 300, 200], class_name="person"),
            ],
            "helmet": [detection([20, 10, 50, 40])],
            "vest": [detection([20, 70, 80, 150], class_name="vest")],
            "without_helmet": [detection([220, 10, 250, 40], class_name="without_helmet")],
            "without_vest": [detection([220, 70, 280, 150], class_name="without_vest")],
        }

        result = detector.check_violations(detections)

        self.assertEqual(result["person_count"], 2)
        self.assertEqual(result["violation_count"], 2)
        self.assertEqual(result["person_results"][0]["violations"], [])
        self.assertEqual(
            result["person_results"][1]["violations"], ["without_helmet", "without_vest"]
        )

    def test_absence_without_explicit_negative_is_uncertain(self):
        detector = ViolationDetector({"violation": {}})
        result = detector.check_violations({
            "person": [detection([0, 0, 100, 200], class_name="person")],
            "helmet": [], "vest": [], "without_helmet": [], "without_vest": [],
        })

        self.assertEqual(result["violation_count"], 0)
        self.assertEqual(result["status"], "evaluated_with_uncertainty")
        self.assertEqual(result["person_results"][0]["uncertain"], ["helmet_status_uncertain", "vest_status_uncertain"])

    def test_one_ppe_detection_is_not_assigned_to_two_people(self):
        detector = ViolationDetector({"violation": {}})
        result = detector.check_violations({
            "person": [
                detection([0, 0, 100, 200], class_name="person"),
                detection([50, 0, 150, 200], class_name="person"),
            ],
            "helmet": [detection([45, 10, 75, 40])],
            "vest": [], "without_helmet": [], "without_vest": [],
        })

        self.assertEqual(sum(item["has_helmet"] for item in result["person_results"]), 1)

    def test_ppe_outside_person_is_not_associated(self):
        detector = ViolationDetector({"violation": {"person_ppe_iou_threshold": 0.01}})
        self.assertFalse(detector._is_associated([0, 0, 100, 100], [200, 200, 250, 250]))

    def test_missing_person_preserves_unverified_violation_candidate(self):
        detector = ViolationDetector({"violation": {}})
        detections = {
            "person": [],
            "helmet": [],
            "vest": [],
            "without_helmet": [],
            "without_vest": [detection([10, 10, 90, 90], 0.852, "without_vest")],
        }

        result = detector.check_violations(detections)

        self.assertEqual(result["status"], "person_missing")
        self.assertEqual(result["violation_count"], 0)
        self.assertEqual(result["unverified_candidates"][0]["class_name"], "without_vest")

    def test_multiple_person_violations_are_drawn_as_one_label(self):
        import src.violation_logic as violation_logic

        calls = []
        def record_draw(image, bbox, label, color, thickness, text_size):
            calls.append(label)
            return image

        # annotate_violations imports draw_bbox inside the function, so patch the utility.
        import src.utils as utils
        original_utils_draw_bbox = utils.draw_bbox
        utils.draw_bbox = record_draw
        try:
            config = {"visualization": {"colors": {}, "bbox_thickness": 2, "text_size": 0.6}}
            violation_logic.annotate_violations(
                np.zeros((100, 100, 3), dtype=np.uint8),
                {"person": [], "helmet": [], "vest": [], "without_helmet": [], "without_vest": []},
                {
                    "violations": [
                        {"person_id": 1, "bbox": [10, 10, 90, 90], "type": "without_helmet"},
                        {"person_id": 1, "bbox": [10, 10, 90, 90], "type": "without_vest"},
                    ]
                },
                config,
            )
        finally:
            utils.draw_bbox = original_utils_draw_bbox

        self.assertEqual(calls, ["violations: without helmet, without vest"])

    def test_cli_summary_is_cp1252_safe(self):
        from src.inference import process_images

        class FakeProcessor:
            def process_image(self, image_path, output_path=None):
                return {
                    "annotated_frame": np.zeros((1, 1, 3), dtype=np.uint8),
                    "violations": {"violation_count": 1},
                }

        args = SimpleNamespace(input=str(Path("demo_images/no_helmet.jpg")), output=None, display=False)
        stream = io.BytesIO()
        stdout = io.TextIOWrapper(stream, encoding="cp1252", errors="strict")
        try:
            with contextlib.redirect_stdout(stdout):
                process_images(FakeProcessor(), args)
            stdout.flush()
        finally:
            stdout.detach()

        self.assertIn(b"Violations detected: 1", stream.getvalue())

    def test_invalid_input_image_is_reported(self):
        from src.processor import FrameProcessor

        processor = object.__new__(FrameProcessor)
        with self.assertRaises(ValueError):
            processor.process_image("does-not-exist.jpg")


class HeadLocalizationTests(unittest.TestCase):
    """Tests for HeadRegionLocalizer, especially the new top-region fallback."""

    def test_head_localizer_top_region_fallback_when_face_not_found(self):
        """Fallback crop returned when Haar finds no face and fallback is enabled."""
        from src.head_localization import HeadRegionLocalizer

        # Black image — Haar will find no face; fallback should trigger.
        image = np.zeros((200, 100, 3), dtype=np.uint8)
        localizer = HeadRegionLocalizer(use_top_region_fallback=True, top_region_ratio=0.30)
        result = localizer.locate(image, [0.0, 0.0, 100.0, 200.0])
        self.assertTrue(result.reliable)
        self.assertIsNotNone(result.crop)
        self.assertEqual(result.reason, "top_region_fallback")
        # Crop height should be approximately 30% of person height (200 px).
        crop_height = result.crop.shape[0]
        self.assertAlmostEqual(crop_height / 200, 0.30, delta=0.05)

    def test_head_localizer_no_fallback_when_disabled(self):
        """When fallback is disabled, face-not-found is still unreliable."""
        from src.head_localization import HeadRegionLocalizer

        image = np.zeros((200, 100, 3), dtype=np.uint8)
        result = HeadRegionLocalizer(use_top_region_fallback=False).locate(
            image, [0.0, 0.0, 100.0, 200.0]
        )
        self.assertFalse(result.reliable)
        self.assertEqual(result.reason, "face_not_localized")

    def test_head_localizer_fallback_reason_is_consistent_with_no_face(self):
        """locate() on a synthetic image produces a known reason set."""
        from src.head_localization import HeadRegionLocalizer

        image = np.zeros((200, 100, 3), dtype=np.uint8)
        result = HeadRegionLocalizer(use_top_region_fallback=True).locate(
            image, [0.0, 0.0, 100.0, 200.0]
        )
        self.assertIn(result.reason, {"face_anchor", "top_region_fallback", "face_not_localized"})

    def test_head_localizer_fallback_bbox_is_image_coordinates(self):
        """Fallback bbox must be expressed in full-image coordinates, not person-crop."""
        from src.head_localization import HeadRegionLocalizer

        image = np.zeros((400, 300, 3), dtype=np.uint8)
        # Person starts 50 px from top, 80 px from left
        person_bbox = [80.0, 50.0, 200.0, 350.0]
        localizer = HeadRegionLocalizer(use_top_region_fallback=True, top_region_ratio=0.28)
        result = localizer.locate(image, person_bbox)
        if result.reason == "top_region_fallback":
            x1, y1, x2, y2 = result.bbox
            # x-range must span the person x-range
            self.assertAlmostEqual(x1, 80.0, delta=1)
            self.assertAlmostEqual(x2, 200.0, delta=1)
            # y1 must be person's top
            self.assertAlmostEqual(y1, 50.0, delta=1)
            # y2 must be person_top + top_region_ratio * person_height
            expected_y2 = 50.0 + 0.28 * (350.0 - 50.0)
            self.assertAlmostEqual(y2, expected_y2, delta=3)


class HeadHeuristicTests(unittest.TestCase):
    """Tests for the geometry-only HeadHeuristic."""

    def _face_anchor_crop(self, bbox: list[float]) -> "HeadCropResult":
        from src.head_localization import HeadCropResult
        return HeadCropResult(
            crop=np.zeros((30, 40, 3), dtype=np.uint8),
            bbox=bbox,
            reliable=True,
            reason="face_anchor",
        )

    def _fallback_crop(self, bbox: list[float]) -> "HeadCropResult":
        from src.head_localization import HeadCropResult
        return HeadCropResult(
            crop=np.zeros((30, 40, 3), dtype=np.uint8),
            bbox=bbox,
            reliable=True,
            reason="top_region_fallback",
        )

    def _unreliable_crop(self) -> "HeadCropResult":
        from src.head_localization import HeadCropResult
        return HeadCropResult(None, None, False, "face_not_localized")

    def test_helmet_worn_when_detection_strongly_overlaps_head_crop(self):
        from src.head_heuristic import HeadHeuristic
        heuristic = HeadHeuristic({"head_heuristic": {"face_anchor_iou_threshold": 0.10}})
        head_crop = self._face_anchor_crop([0.0, 0.0, 50.0, 40.0])
        helmet = {"bbox": [2.0, 1.0, 48.0, 37.0], "confidence": 0.92}
        result = heuristic.evaluate(head_crop, [helmet])
        self.assertEqual(result["status"], "helmet_worn")
        self.assertGreater(result["confidence"], 0.0)

    def test_helmet_worn_when_detection_inside_head_crop_bbox(self):
        from src.head_heuristic import HeadHeuristic
        heuristic = HeadHeuristic({"head_heuristic": {}})
        head_crop = self._face_anchor_crop([10.0, 5.0, 50.0, 35.0])
        helmet = {"bbox": [15.0, 5.0, 45.0, 32.0], "confidence": 0.85}
        result = heuristic.evaluate(head_crop, [helmet])
        self.assertEqual(result["status"], "helmet_worn")

    def test_no_helmet_when_no_helmet_detections_and_face_anchor(self):
        from src.head_heuristic import HeadHeuristic
        heuristic = HeadHeuristic({"head_heuristic": {}})
        head_crop = self._face_anchor_crop([10.0, 5.0, 50.0, 35.0])
        result = heuristic.evaluate(head_crop, [])
        self.assertEqual(result["status"], "no_helmet")
        self.assertEqual(result["reason"], "no_helmet_overlaps_head_region")

    def test_needs_review_when_fallback_crop_and_no_helmet(self):
        from src.head_heuristic import HeadHeuristic
        heuristic = HeadHeuristic({"head_heuristic": {}})
        head_crop = self._fallback_crop([10.0, 5.0, 50.0, 35.0])
        result = heuristic.evaluate(head_crop, [])
        self.assertEqual(result["status"], "needs_review")
        self.assertEqual(result["reason"], "fallback_no_helmet_overlap")

    def test_needs_review_when_head_crop_unreliable(self):
        from src.head_heuristic import HeadHeuristic
        heuristic = HeadHeuristic({"head_heuristic": {}})
        head_crop = self._unreliable_crop()
        result = heuristic.evaluate(head_crop, [{"bbox": [0, 0, 10, 10], "confidence": 0.9}])
        self.assertEqual(result["status"], "needs_review")

    def test_held_helmet_at_waist_does_not_overlap_head_region(self):
        """A helmet detection centered at waist level has near-zero IoU with head crop."""
        from src.head_heuristic import HeadHeuristic
        heuristic = HeadHeuristic({"head_heuristic": {}})
        head_crop = self._face_anchor_crop([10.0, 5.0, 50.0, 35.0])
        # Helmet is far below head crop
        held_helmet = {"bbox": [10.0, 150.0, 50.0, 190.0], "confidence": 0.85}
        result = heuristic.evaluate(head_crop, [held_helmet])
        self.assertEqual(result["status"], "no_helmet")

    def test_fallback_helmet_worn_requires_higher_iou(self):
        """With fallback crop, a weak overlap that would pass face_anchor still needs review."""
        from src.head_heuristic import HeadHeuristic
        heuristic = HeadHeuristic({
            "head_heuristic": {
                "face_anchor_iou_threshold": 0.10,
                "fallback_iou_threshold": 0.50,
            }
        })
        head_crop = self._fallback_crop([0.0, 0.0, 50.0, 40.0])
        # IoU ~12% — passes face_anchor threshold but not fallback threshold
        helmet = {"bbox": [0.0, 0.0, 50.0, 10.0], "confidence": 0.80}
        result = heuristic.evaluate(head_crop, [helmet])
        self.assertEqual(result["status"], "needs_review")

    def test_violation_result_no_helmet_confirmed_by_heuristic(self):
        """ViolationDetector with head_verification no_helmet records a violation."""
        result = ViolationDetector({"violation": {}}).check_violations(
            {
                "person": [detection([0, 0, 100, 200], class_name="person")],
                "helmet": [],
                "vest": [],
                "without_helmet": [],
                "without_vest": [],
            },
            head_verifications=[{
                "status": "no_helmet",
                "reason": "no_helmet_overlaps_head_region",
                "confidence": 0.0,
            }],
        )
        self.assertEqual(result["person_results"][0]["violations"], ["without_helmet"])
        self.assertEqual(result["violation_count"], 1)

    def test_heuristic_iou_symmetry(self):
        """_iou(a, b) == _iou(b, a)."""
        from src.head_heuristic import HeadHeuristic
        heuristic = HeadHeuristic({})
        a = [10.0, 5.0, 50.0, 35.0]
        b = [20.0, 0.0, 60.0, 25.0]
        self.assertAlmostEqual(heuristic._iou(a, b), heuristic._iou(b, a), places=6)

    def test_heuristic_iou_zero_for_non_overlapping(self):
        from src.head_heuristic import HeadHeuristic
        heuristic = HeadHeuristic({})
        self.assertEqual(heuristic._iou([0, 0, 10, 10], [20, 20, 30, 30]), 0.0)

    def test_heuristic_iou_one_for_identical_boxes(self):
        from src.head_heuristic import HeadHeuristic
        heuristic = HeadHeuristic({})
        self.assertAlmostEqual(heuristic._iou([5, 5, 25, 25], [5, 5, 25, 25]), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
