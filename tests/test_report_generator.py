"""Tests for automated PDF report generation and layout integrity."""

import os
import unittest
from datetime import datetime
from pathlib import Path
import numpy as np

from src.report_generator import (
    PDFReportGenerator,
    humanize_flag,
    humanize_flag_list,
    should_generate_report,
)
from src.utils import format_bbox, format_scan_timestamp


class ReportGeneratorTests(unittest.TestCase):
    def setUp(self):
        self.output_dir = Path("outputs/test_reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = {
            "reporting": {
                "enabled": True,
                "output_dir": str(self.output_dir),
                "auto_generate_on_violation": True,
                "auto_generate_on_incident": True,
                "auto_generate_on_review": True,
            }
        }
        self.generator = PDFReportGenerator(self.config)

    def test_should_generate_report_trigger_policy(self):
        """Verify trigger logic correctly activates on actionable findings and stays silent on compliant scans."""
        # 1. Compliant scan -> False
        compliant_res = {
            "violations": {"violation_count": 0, "status": "evaluated"},
            "incidents": {"has_candidate": False},
        }
        self.assertFalse(should_generate_report(compliant_res, self.config))

        # 2. Violation scan -> True
        violation_res = {
            "violations": {"violation_count": 1, "status": "evaluated"},
            "incidents": {"has_candidate": False},
        }
        self.assertTrue(should_generate_report(violation_res, self.config))

        # 3. Needs review -> True
        review_res = {
            "violations": {"violation_count": 0, "status": "needs_review"},
            "incidents": {"has_candidate": False},
        }
        self.assertTrue(should_generate_report(review_res, self.config))

        # 4. Incident candidate -> True
        incident_res = {
            "violations": {"violation_count": 0, "status": "evaluated"},
            "incidents": {"has_candidate": True},
        }
        self.assertTrue(should_generate_report(incident_res, self.config))

    def test_pdf_report_creation_and_header(self):
        """Generate a complete PDF report and verify it is a valid, readable PDF document."""
        dummy_img = np.full((300, 400, 3), 180, dtype=np.uint8)
        result = {
            "scan_timestamp": "2026-10-30T02:45:00+05:45",
            "detections": {
                "person": [{"bbox": [20, 20, 150, 280], "confidence": 0.91, "class_name": "person"}],
                "helmet": [],
                "vest": [{"bbox": [30, 80, 140, 240], "confidence": 0.88, "class_name": "vest"}],
                "without_helmet": [],
                "without_vest": [],
            },
            "violations": {
                "violation_count": 1,
                "compliance_rate": 0.0,
                "person_count": 1,
                "status": "evaluated",
                "violations": [{"type": "without_helmet", "bbox": [20, 20, 150, 280], "conf": 0.91, "person_id": 1}],
                "unverified_candidates": [],
                "person_results": [{
                    "person_id": 1,
                    "bbox": [20, 20, 150, 280],
                    "has_helmet": False,
                    "has_vest": True,
                    "violations": ["without_helmet"],
                    "uncertain": [],
                    "needs_review": False,
                    "review_reasons": [],
                }],
            },
            "incidents": {
                "has_candidate": True,
                "status": "RED_REGION_CANDIDATE",
                "candidates": [{
                    "label": "RED_REGION_CANDIDATE",
                    "bbox": [50.0, 120.0, 90.0, 160.0],
                    "confidence": 0.65,
                    "area_px": 140,
                    "near_person": True,
                    "review_notice": "Unverified visual candidate; requires human review to assess.",
                }],
                "limitations": "Non-medical heuristic: Flags visual red-region candidates for human review only.",
            },
            "original_frame": dummy_img,
            "annotated_frame": dummy_img,
        }

        report_path = self.generator.generate_report(result, "sample_worker.jpg")
        self.assertIsNotNone(report_path)
        self.assertTrue(os.path.exists(report_path), f"Report not found at {report_path}")

        # Check PDF header
        with open(report_path, "rb") as f:
            header = f.read(5)
            self.assertEqual(header, b"%PDF-", "File is not a valid PDF")

        file_size = os.path.getsize(report_path)
        self.assertGreater(file_size, 2000, "PDF size is unexpectedly small")

    def test_timestamp_formatting_and_propagation(self):
        """Verify format_scan_timestamp preserves timezone and handles missing timestamps honestly."""
        # Valid ISO timestamp with offset
        formatted = format_scan_timestamp("2026-10-30T02:45:00+05:45")
        self.assertIn("30 October 2026, 02:45:00", formatted)
        self.assertTrue("UTC+05:45" in formatted or "+0545" in formatted or "Nepal" in formatted)

        # Missing timestamp returns unavailable
        self.assertEqual(format_scan_timestamp(None), "Scan timestamp unavailable")
        self.assertEqual(format_scan_timestamp(""), "Scan timestamp unavailable")
        self.assertEqual(format_scan_timestamp("invalid-date"), "Scan timestamp unavailable")

    def test_bounding_box_formatting(self):
        """Verify format_bbox rounds to clean integers in x1=..., y1=..., x2=..., y2=... format."""
        raw_bbox = [245.1245, 7.3456, 1396.89, 1074.77]
        formatted = format_bbox(raw_bbox)
        self.assertEqual(formatted, "x1=245, y1=7, x2=1397, y2=1075")
        self.assertEqual(format_bbox(None), "N/A")
        self.assertEqual(format_bbox([1, 2]), "N/A")

    def test_humanize_flags(self):
        """Verify internal technical flags convert to human-readable text."""
        self.assertEqual(
            humanize_flag("head_classifier_fallback_no_helmet_overlap"),
            "Needs review — head classifier fallback",
        )
        self.assertEqual(
            humanize_flag("helmet_below_worn_confidence_threshold"),
            "Needs review — low helmet confidence (< 80%)",
        )
        self.assertEqual(
            humanize_flag_list(["without_helmet", "without_vest"]),
            "Missing Helmet; Missing Vest",
        )
        self.assertEqual(humanize_flag_list([]), "Clear")

    def test_multi_person_report_generation(self):
        """Verify multi-person image with long review reasons generates valid multi-page or single-page PDF."""
        dummy_img = np.full((400, 600, 3), 120, dtype=np.uint8)
        person_results = []
        for i in range(1, 6):
            person_results.append({
                "person_id": i,
                "bbox": [50 * i, 30, 50 * i + 80, 350],
                "has_helmet": i % 2 == 0,
                "has_vest": i % 3 == 0,
                "violations": ["without_helmet"] if i % 2 != 0 else [],
                "uncertain": ["helmet_status_uncertain"] if i % 2 != 0 else [],
                "needs_review": True,
                "review_reasons": ["head_classifier_fallback_no_helmet_overlap", "helmet_below_worn_confidence_threshold"],
                "associations": {
                    "helmet": [{"bbox": [50 * i + 10, 35, 50 * i + 60, 90], "confidence": 0.72}],
                    "vest": [],
                    "without_helmet": [],
                    "without_vest": [],
                },
            })

        result = {
            "scan_timestamp": "2026-10-30T02:50:00+05:45",
            "detections": {"person": [], "helmet": [], "vest": [], "without_helmet": [], "without_vest": []},
            "violations": {
                "violation_count": 3,
                "compliance_rate": 20.0,
                "person_count": 5,
                "status": "needs_review",
                "violations": [{"type": "without_helmet", "bbox": [50, 30, 130, 350], "conf": 0.9, "person_id": 1}],
                "unverified_candidates": [],
                "person_results": person_results,
            },
            "incidents": {"has_candidate": False, "available": False},
            "original_frame": dummy_img,
            "annotated_frame": dummy_img,
        }

        report_path = self.generator.generate_report(result, "multi_person_test.jpg")
        self.assertTrue(os.path.exists(report_path))
        self.assertGreater(os.path.getsize(report_path), 5000)


if __name__ == "__main__":
    unittest.main()
