"""Manager alert and notification delivery system with cooldown deduplication."""

from __future__ import annotations

import os
import smtplib
import time
from email.message import EmailMessage
from pathlib import Path
from typing import Dict, Optional


class NotificationManager:
    """Handle alert generation, severity routing, and email dispatch with cooldown."""

    def __init__(self, config: dict):
        self.config = config
        self.settings = config.get("notifications", {})
        self.cooldown_seconds = float(
            os.environ.get("NOTIFICATION_COOLDOWN_SECONDS")
            or self.settings.get("cooldown_seconds", 60)
        )
        self._last_alert_times: Dict[str, float] = {}

    def determine_severity(self, result: dict) -> str:
        """Calculate alert severity level from detection and violation findings."""
        incidents = result.get("incidents", {})
        violations = result.get("violations", {})
        notify_on_incident = bool(self.settings.get("notify_on_incident", True))

        if notify_on_incident and incidents.get("status") == "POTENTIAL_INJURY_SUSPECTED":
            return "URGENT_INJURY_REVIEW"
        if violations.get("violation_count", 0) > 0:
            return "VIOLATION_ALERT"
        if violations.get("status") == "needs_review":
            return "REVIEW_ALERT"
        if notify_on_incident and incidents.get("has_candidate", False):
            return "REVIEW_ALERT"

        return "INFO"

    def _get_smtp_credentials(self) -> dict:
        """Retrieve SMTP settings from environment variables or config with safe defaults."""
        return {
            "server": os.environ.get("SMTP_SERVER") or self.settings.get("smtp_server", ""),
            "port": int(os.environ.get("SMTP_PORT") or self.settings.get("smtp_port", 587)),
            "username": os.environ.get("SMTP_USERNAME") or self.settings.get("smtp_username", ""),
            "password": os.environ.get("SMTP_PASSWORD") or self.settings.get("smtp_password", ""),
            "recipient": os.environ.get("ALERT_RECIPIENT_EMAIL") or self.settings.get("manager_email", ""),
            "sender": os.environ.get("ALERT_SENDER_EMAIL") or self.settings.get("smtp_username", "alerts@safety-monitor.local"),
        }

    def is_throttled(self, source_name: str, severity: str) -> bool:
        """Check if an alert for this source and severity was sent within the cooldown window."""
        key = f"{source_name}:{severity}"
        now = time.time()
        last_time = self._last_alert_times.get(key)
        if last_time is not None and (now - last_time) < self.cooldown_seconds:
            return True
        return False

    def record_alert(self, source_name: str, severity: str) -> None:
        """Record timestamp for cooldown tracking."""
        key = f"{source_name}:{severity}"
        self._last_alert_times[key] = time.time()

    def reset_cooldown(self) -> None:
        """Reset cooldown history (useful for tests or stream restarts)."""
        self._last_alert_times.clear()

    def send_alert(
        self,
        result: dict,
        source_name: str = "unknown",
        report_path: Optional[str] = None,
    ) -> Dict:
        """Format and dispatch notification, respecting cooldown and credential availability.

        Returns
        -------
        dict with:
          sent: bool
          status: str ("Sent", "Email not configured", "Throttled by cooldown", or error)
          severity: str
          throttled: bool
          report_path: str | None
        """
        severity = self.determine_severity(result)

        # Check cooldown throttling
        if self.is_throttled(source_name, severity):
            return {
                "sent": False,
                "status": f"Throttled by cooldown ({self.cooldown_seconds}s window)",
                "severity": severity,
                "throttled": True,
                "report_path": report_path,
            }

        # Check credentials
        creds = self._get_smtp_credentials()
        has_creds = bool(creds["server"] and creds["recipient"])

        if not has_creds:
            # Record local event time to prevent spamming local alerts
            self.record_alert(source_name, severity)
            return {
                "sent": False,
                "status": "Email not configured",
                "severity": severity,
                "throttled": False,
                "report_path": report_path,
            }

        # Build Email message
        msg = EmailMessage()
        msg["Subject"] = f"[PPE Monitor - {severity}] Actionable Finding in {source_name}"
        msg["From"] = creds["sender"]
        msg["To"] = creds["recipient"]

        violations = result.get("violations", {})
        incidents = result.get("incidents", {})
        body_lines = [
            f"PPE Safety Monitor — Notification Alert",
            f"Severity: {severity}",
            f"Source: {source_name}",
            f"Persons Detected: {violations.get('person_count', 0)}",
            f"Violations Flagged: {violations.get('violation_count', 0)}",
            f"Overall Status: {violations.get('status', 'unknown')}",
        ]
        if incidents.get("has_candidate"):
            body_lines.append(f"Visual Incident Status: {incidents.get('status')} (Unverified Candidate)")

        if report_path:
            body_lines.append(f"\nA detailed PDF inspection report is attached: {Path(report_path).name}")

        msg.set_content("\n".join(body_lines))

        # Attach PDF report if present
        if report_path and Path(report_path).is_file():
            pdf_data = Path(report_path).read_bytes()
            msg.add_attachment(
                pdf_data,
                maintype="application",
                subtype="pdf",
                filename=Path(report_path).name,
            )

        # Dispatch via SMTP
        try:
            with smtplib.SMTP(creds["server"], creds["port"], timeout=10) as smtp:
                if creds["port"] == 587:
                    smtp.starttls()
                if creds["username"] and creds["password"]:
                    smtp.login(creds["username"], creds["password"])
                smtp.send_message(msg)

            self.record_alert(source_name, severity)
            return {
                "sent": True,
                "status": "Sent",
                "severity": severity,
                "throttled": False,
                "report_path": report_path,
            }
        except Exception as exc:
            self.record_alert(source_name, severity)
            return {
                "sent": False,
                "status": f"Delivery failed: {exc}",
                "severity": severity,
                "throttled": False,
                "report_path": report_path,
            }

