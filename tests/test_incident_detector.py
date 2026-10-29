"""Tests for conservative visual incident / red-region candidate detection."""

import unittest
import numpy as np

from src.incident_detector import IncidentDetector


class IncidentDetectorTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "incident_detection": {
                "enabled": True,
                "min_region_area_px": 50,
                "max_region_area_ratio": 0.15,
                "texture_std_min": 10.0,
                "urgent_review_threshold": 0.70,
            }
        }
        self.detector = IncidentDetector(self.config)

    def test_production_default_disabled_reports_unavailable_not_evaluated(self):
        """When disabled by default in production, detect_incidents reports UNAVAILABLE without declaring blood-free."""
        prod_detector = IncidentDetector({"incident_detection": {"enabled": False}})
        img = np.full((300, 300, 3), 100, dtype=np.uint8)
        result = prod_detector.detect_incidents(img)
        self.assertFalse(result["available"])
        self.assertFalse(result["has_candidate"])
        self.assertEqual(result["status"], "NOT_EVALUATED_UNAVAILABLE")
        self.assertIn("not been evaluated for blood", result["message"].lower())
        self.assertIn("absence of alerts does not confirm", result["message"].lower())

    def test_neutral_image_yields_no_candidate(self):
        """Neutral gray/blue image should detect no candidates."""
        img = np.full((300, 300, 3), (128, 128, 128), dtype=np.uint8)
        result = self.detector.detect_incidents(img, persons=[])
        self.assertFalse(result["has_candidate"])
        self.assertEqual(result["status"], "NO_CANDIDATE_DETECTED")
        self.assertEqual(len(result["candidates"]), 0)
        self.assertIn("non-medical", result["limitations"].lower())

    def test_uniform_red_vest_or_tool_is_rejected(self):
        """Flat uniform bright red (e.g. tool or warning sign) lacks texture variance and is rejected."""
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        # Uniform bright red in BGR: (0, 0, 255)
        img[50:150, 50:150] = (0, 0, 255)
        result = self.detector.detect_incidents(img, persons=[{"bbox": [40, 40, 160, 250]}])
        # Flat red region has standard deviation ~ 0 -> rejected by texture variance check
        self.assertFalse(result["has_candidate"])

    def test_skin_tones_are_not_flagged_as_blood(self):
        """Typical facial/arm skin tones must not be flagged as blood candidates."""
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        # Typical skin tone BGR: (150, 180, 220)
        img[50:100, 50:100] = (150, 180, 220)
        result = self.detector.detect_incidents(img, persons=[{"bbox": [40, 40, 160, 250]}])
        self.assertFalse(result["has_candidate"])

    def test_textured_deep_red_region_detected_as_candidate_never_confirmed(self):
        """Textured deep-red area near person flags as RED_REGION_CANDIDATE, never BLOOD_CONFIRMED."""
        np.random.seed(42)
        img = np.full((400, 400, 3), 100, dtype=np.uint8)

        # Create textured blood-like deep red patch: BGR low B, low G, medium-to-dark R with variance
        patch = np.zeros((60, 60, 3), dtype=np.uint8)
        patch[:, :, 0] = np.random.randint(5, 25, (60, 60))    # B
        patch[:, :, 1] = np.random.randint(5, 25, (60, 60))    # G
        patch[:, :, 2] = np.random.randint(110, 190, (60, 60)) # R
        img[120:180, 120:180] = patch

        person = {"bbox": [80.0, 50.0, 250.0, 380.0]}
        result = self.detector.detect_incidents(img, persons=[person])

        self.assertTrue(result["has_candidate"])
        self.assertEqual(result["status"], "POTENTIAL_INJURY_SUSPECTED")
        for cand in result["candidates"]:
            # MANDATORY: Label must be RED_REGION_CANDIDATE, NEVER BLOOD_CONFIRMED
            self.assertEqual(cand["label"], "RED_REGION_CANDIDATE")
            self.assertNotEqual(cand["label"], "BLOOD_CONFIRMED")
            self.assertIn("requires human review", cand["review_notice"].lower())

    def test_small_noise_red_dots_below_min_area_are_ignored(self):
        """Tiny red specks below min_region_area_px are ignored."""
        img = np.full((200, 200, 3), 80, dtype=np.uint8)
        img[50:53, 50:53] = (10, 10, 160)  # 3x3 = 9 px (< 50 px)
        result = self.detector.detect_incidents(img, persons=[{"bbox": [0, 0, 200, 200]}])
        self.assertFalse(result["has_candidate"])


if __name__ == "__main__":
    unittest.main()
