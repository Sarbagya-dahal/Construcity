"""Tests for vest spatial filtering to prevent false-positive vest associations."""
import unittest
from src.violation_logic import ViolationDetector


BASE_CONFIG = {
    "violation": {
        "person_ppe_iou_threshold": 0.01,
        "person_center_margin": 0.05,
        "helmet_head_region_ratio": 0.35,
        "helmet_worn_confidence_threshold": 0.8,
        "vest_torso_top_ratio": 0.25,
        "vest_torso_bottom_ratio": 0.85,
        "vest_min_area_fraction": 0.005,
        "vest_require_center_x_inside": True,
    }
}


class VestSpatialFilterTests(unittest.TestCase):
    def setUp(self):
        self.detector = ViolationDetector(BASE_CONFIG)

    def _make_person(self, x1, y1, x2, y2):
        return {"bbox": [x1, y1, x2, y2], "confidence": 0.9, "conf": 0.9, "source": "person_model"}

    def _make_vest(self, x1, y1, x2, y2, conf=0.7):
        return {"bbox": [x1, y1, x2, y2], "confidence": conf, "conf": conf, "class_id": 1, "class_name": "vest", "source": "ppe_model"}

    def test_vest_on_torso_is_accepted(self):
        """A vest centered in the torso region is accepted."""
        person = self._make_person(100, 50, 400, 500)
        # Torso region: y from 50 + (500-50)*0.25=162.5 to 50 + (500-50)*0.85=432.5
        vest = self._make_vest(110, 200, 390, 400)  # Large vest, centered in torso
        detections = {"person": [person], "vest": [vest], "helmet": [], "without_helmet": [], "without_vest": []}
        result = self.detector.check_violations(detections)
        person_result = result["person_results"][0]
        self.assertTrue(person_result["has_vest"], "Torso-centered vest should be accepted")

    def test_vest_outside_person_x_range_is_rejected(self):
        """A vest with center-X outside the person bbox is rejected."""
        person = self._make_person(200, 50, 500, 450)
        # Vest center-X = 90, person x range = [200, 500] -> outside
        vest = self._make_vest(50, 150, 130, 350)
        detections = {"person": [person], "vest": [vest], "helmet": [], "without_helmet": [], "without_vest": []}
        result = self.detector.check_violations(detections)
        person_result = result["person_results"][0]
        self.assertFalse(person_result["has_vest"], "Vest outside person X range must be rejected")

    def test_tiny_vest_area_rejected(self):
        """A vest with area fraction below vest_min_area_fraction is rejected."""
        person = self._make_person(100, 50, 600, 550)  # Large person: 500*500 = 250000
        person_area = 500 * 500  # 250000
        # Min area = 0.005 * 250000 = 1250. Small vest = 10*10 = 100 < 1250
        vest = self._make_vest(300, 200, 310, 210, conf=0.7)  # tiny: 10x10
        detections = {"person": [person], "vest": [vest], "helmet": [], "without_helmet": [], "without_vest": []}
        result = self.detector.check_violations(detections)
        person_result = result["person_results"][0]
        self.assertFalse(person_result["has_vest"], "Tiny vest (low area fraction) must be rejected")

    def test_vest_above_torso_top_ratio_rejected(self):
        """A vest centered in the upper head region (above vest_torso_top_ratio) is rejected."""
        person = self._make_person(100, 100, 400, 500)
        # torso top = 100 + (500-100)*0.25 = 200. Vest center-Y = 150 < 200 -> rejected
        vest = self._make_vest(150, 100, 350, 180)  # High up on person
        detections = {"person": [person], "vest": [vest], "helmet": [], "without_helmet": [], "without_vest": []}
        result = self.detector.check_violations(detections)
        person_result = result["person_results"][0]
        self.assertFalse(person_result["has_vest"], "Vest in head region must be rejected as vest")

    def test_blood_jpg_scenario_vest_rejected(self):
        """Simulate blood.jpg: yellow helmet on ground outside person X range is rejected."""
        # blood.jpg person bbox approx: [172, 10, 604, 392]
        # blood.jpg vest (yellow helmet) bbox approx: [33, 101, 204, 273]
        person = self._make_person(172, 10, 604, 392)
        vest = self._make_vest(33, 101, 204, 273, conf=0.61)  # center-X = 118, person x range [172,604]
        detections = {"person": [person], "vest": [vest], "helmet": [], "without_helmet": [], "without_vest": []}
        result = self.detector.check_violations(detections)
        person_result = result["person_results"][0]
        self.assertFalse(person_result["has_vest"], "Off-body object (yellow helmet) must not be counted as vest")


if __name__ == "__main__":
    unittest.main()
