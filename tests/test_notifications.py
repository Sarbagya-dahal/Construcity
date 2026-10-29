"""Tests for manager alerts, email notification delivery, severity levels, and deduplication cooldown."""

import unittest
from unittest.mock import MagicMock, patch
import os

from src.notifications import NotificationManager


class NotificationManagerTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "notifications": {
                "enabled": True,
                "cooldown_seconds": 60,
                "manager_email": "",
                "smtp_server": "",
                "smtp_port": 587,
                "smtp_username": "",
                "smtp_password": "",
                "notify_on_violation": True,
                "notify_on_incident": True,
                "notify_on_review": False,
            }
        }
        self.manager = NotificationManager(self.config)

    def test_missing_credentials_returns_unconfigured_status(self):
        """When SMTP credentials are not configured, return clear status without error."""
        result = {
            "violations": {"violation_count": 1, "status": "evaluated"},
            "incidents": {"has_candidate": False},
        }
        alert = self.manager.send_alert(result, source_name="camera_01.jpg")
        self.assertFalse(alert["sent"])
        self.assertEqual(alert["status"], "Email not configured")
        self.assertEqual(alert["severity"], "VIOLATION_ALERT")
        self.assertFalse(alert["throttled"])

    def test_severity_levels(self):
        """Verify severity levels map accurately to findings."""
        # Incident candidate
        res_incident = {
            "violations": {"violation_count": 0, "status": "evaluated"},
            "incidents": {"has_candidate": True, "status": "POTENTIAL_INJURY_SUSPECTED"},
        }
        sev_inc = self.manager.determine_severity(res_incident)
        self.assertEqual(sev_inc, "URGENT_INJURY_REVIEW")

        # PPE Violation
        res_viol = {
            "violations": {"violation_count": 2, "status": "evaluated"},
            "incidents": {"has_candidate": False},
        }
        sev_viol = self.manager.determine_severity(res_viol)
        self.assertEqual(sev_viol, "VIOLATION_ALERT")

        # Needs Review
        res_rev = {
            "violations": {"violation_count": 0, "status": "needs_review"},
            "incidents": {"has_candidate": False},
        }
        sev_rev = self.manager.determine_severity(res_rev)
        self.assertEqual(sev_rev, "REVIEW_ALERT")

    def test_cooldown_throttling_prevents_duplicate_notifications(self):
        """Same source and incident within cooldown_seconds must be throttled."""
        result = {
            "violations": {"violation_count": 1, "status": "evaluated"},
            "incidents": {"has_candidate": False},
        }
        # First alert records event
        alert1 = self.manager.send_alert(result, source_name="gate_cam_01")
        self.assertFalse(alert1["throttled"])

        # Immediate second alert for same source is throttled
        alert2 = self.manager.send_alert(result, source_name="gate_cam_01")
        self.assertTrue(alert2["throttled"])
        self.assertIn("throttled", alert2["status"].lower())

        # Different source is not throttled
        alert3 = self.manager.send_alert(result, source_name="gate_cam_02")
        self.assertFalse(alert3["throttled"])

    @patch("smtplib.SMTP")
    def test_mock_smtp_successful_transmission(self, mock_smtp_class):
        """When SMTP settings are supplied, mock transport receives email with attachment."""
        smtp_instance = MagicMock()
        mock_smtp_class.return_value.__enter__.return_value = smtp_instance

        configured_config = {
            "notifications": {
                "enabled": True,
                "cooldown_seconds": 60,
                "manager_email": "safety_manager@example.com",
                "smtp_server": "smtp.example.com",
                "smtp_port": 587,
                "smtp_username": "alerts@example.com",
                "smtp_password": "mock_password",
                "notify_on_violation": True,
                "notify_on_incident": True,
                "notify_on_review": True,
            }
        }
        mgr = NotificationManager(configured_config)
        result = {
            "violations": {
                "violation_count": 1,
                "status": "evaluated",
                "violations": [{"type": "without_helmet", "bbox": [0, 0, 10, 10], "conf": 0.9}],
            },
            "incidents": {"has_candidate": False},
        }

        alert = mgr.send_alert(result, source_name="site_inspection.jpg", report_path=None)
        self.assertTrue(alert["sent"])
        self.assertEqual(alert["status"], "Sent")
        self.assertTrue(smtp_instance.send_message.called)


if __name__ == "__main__":
    unittest.main()

