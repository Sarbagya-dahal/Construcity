"""Persistent camera-session storage, reporting, and packaging."""

from __future__ import annotations

import csv
import json
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image as ReportImage
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from reportlab.lib import colors


def _now() -> datetime:
    return datetime.now().astimezone()


def _iso(value: datetime) -> str:
    return value.astimezone().isoformat(timespec="seconds")


class CameraSession:
    """Own all durable artifacts for one camera monitoring session."""

    def __init__(self, session_dir: Path, interval_seconds: float, config: dict | None = None, record_video: bool = False):
        self.session_dir = Path(session_dir).resolve()
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.original_dir = self.session_dir / "original_frames"
        self.annotated_dir = self.session_dir / "annotated_frames"
        self.reports_dir = self.session_dir / "reports"
        for folder in (self.original_dir, self.annotated_dir, self.reports_dir):
            folder.mkdir(parents=True, exist_ok=True)
        # Compatibility aliases for prior sessions and callers.
        (self.session_dir / "snapshots").mkdir(exist_ok=True)
        (self.session_dir / "annotated").mkdir(exist_ok=True)
        self.session_id = self.session_dir.name
        self.interval_seconds = float(interval_seconds)
        self.config = config or {}
        self.record_video = bool(record_video)
        self.started_at = _now()
        self.ended_at: datetime | None = None
        self.events: list[dict] = []
        self._video_writer = None
        self._video_path = self.session_dir / "recording.mp4"
        self._write_manifest("running")
        self._write_events()

    @property
    def event_count(self) -> int:
        return len(self.events)

    def _manifest(self, status: str) -> dict:
        return {
            "session_id": self.session_id,
            "status": status,
            "started_at": _iso(self.started_at),
            "ended_at": _iso(self.ended_at) if self.ended_at else None,
            "timezone": self.started_at.tzname(),
            "utc_offset": self.started_at.strftime("%z"),
            "interval_seconds": self.interval_seconds,
            "record_video": self.record_video,
            "event_count": len(self.events),
            "successful_frames": sum(e.get("processing_status") == "success" for e in self.events),
            "failed_frames": sum(e.get("processing_status") == "failed" for e in self.events),
            "total_people_detected": sum(e.get("person_count", 0) for e in self.events),
            "confirmed_violations": sum(e.get("violation_count", 0) for e in self.events),
            "uncertain_findings": sum(e.get("uncertain_count", 0) for e in self.events),
            "configuration": self.config,
        }

    def _write_manifest(self, status: str) -> None:
        (self.session_dir / "session.json").write_text(
            json.dumps(self._manifest(status), indent=2, default=str), encoding="utf-8"
        )

    def _write_events(self) -> None:
        (self.session_dir / "events.json").write_text(
            json.dumps(self.events, indent=2, default=str), encoding="utf-8"
        )
        fields = [
            "frame_number", "frame_id", "captured_at", "analyzed_at", "processing_status",
            "original_path", "annotated_path", "person_count", "violation_count",
            "uncertain_count", "status", "error",
        ]
        with (self.session_dir / "events.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for event in self.events:
                writer.writerow({field: event.get(field, "") for field in fields})

    @staticmethod
    def _write_image(path: Path, image: np.ndarray) -> None:
        if image is None:
            return
        if not cv2.imwrite(str(path), image):
            raise OSError(f"Could not save camera evidence: {path}")

    def _filename(self, frame_number: int, captured_at: datetime) -> str:
        return f"frame_{frame_number:06d}_{captured_at.strftime('%Y%m%d_%H%M%S')}.jpg"

    def _ensure_video_writer(self, frame: np.ndarray) -> None:
        if not self.record_video or self._video_writer is not None:
            return
        height, width = frame.shape[:2]
        writer = cv2.VideoWriter(str(self._video_path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (width, height))
        if not writer.isOpened():
            raise OSError("Could not open recording.mp4 writer")
        self._video_writer = writer

    def record_video_frame(self, frame: np.ndarray) -> None:
        """Append one camera-rate frame when continuous recording is enabled."""
        self._ensure_video_writer(frame)
        if self._video_writer is not None:
            self._video_writer.write(frame)

    def _append_event(self, event: dict) -> None:
        self.events.append(event)
        self._write_events()
        self._write_manifest("running")

    def record_frame(
        self,
        frame_number: int,
        frame_id: int,
        original: np.ndarray,
        result: dict,
        captured_at: datetime,
        analyzed_at: datetime,
    ) -> dict:
        """Persist original/annotated evidence and the complete finding record."""
        filename = self._filename(frame_number, captured_at)
        original_path = self.original_dir / filename
        annotated_path = self.annotated_dir / filename
        self._write_image(original_path, original)
        annotated = result.get("annotated_frame")
        if annotated is not None:
            self._write_image(annotated_path, annotated)
        violations = result.get("violations", {})
        event = {
            "frame_number": frame_number,
            "frame_id": frame_id,
            "captured_at": _iso(captured_at),
            "analyzed_at": _iso(analyzed_at),
            "processing_status": "success",
            "original_path": str(original_path.relative_to(self.session_dir)),
            "annotated_path": str(annotated_path.relative_to(self.session_dir)) if annotated is not None else "",
            "person_count": violations.get("person_count", len(result.get("detections", {}).get("person", []))),
            "violation_count": violations.get("violation_count", 0),
            "uncertain_count": len(violations.get("unverified_candidates", [])) + sum(
                len(p.get("uncertain", [])) for p in violations.get("person_results", [])
            ),
            "status": violations.get("status", "unknown"),
            "violations": violations.get("violations", []),
            "unverified_candidates": violations.get("unverified_candidates", []),
            "person_results": violations.get("person_results", []),
            "detections": result.get("detections", {}),
            "incidents": result.get("incidents", {}),
            "alert": result.get("alert", {}),
            "error": "",
        }
        self._append_event(event)
        return event

    def record_failure(self, frame_number: int, frame_id: int, original: np.ndarray, captured_at: datetime, error: str) -> dict:
        filename = self._filename(frame_number, captured_at)
        original_path = self.original_dir / filename
        self._write_image(original_path, original)
        event = {
            "frame_number": frame_number,
            "frame_id": frame_id,
            "captured_at": _iso(captured_at),
            "analyzed_at": _iso(_now()),
            "processing_status": "failed",
            "original_path": str(original_path.relative_to(self.session_dir)),
            "annotated_path": "",
            "person_count": 0,
            "violation_count": 0,
            "uncertain_count": 0,
            "status": "processing_failed",
            "violations": [],
            "unverified_candidates": [],
            "person_results": [],
            "detections": {},
            "incidents": {},
            "alert": {},
            "error": str(error),
        }
        self._append_event(event)
        return event

    def _generate_pdf(self) -> Path:
        report_path = self.reports_dir / "session_report.pdf"
        styles = getSampleStyleSheet()
        story = [Paragraph("PPE Safety Monitor — Camera Session Report", styles["Title"])]
        manifest = self._manifest("completed")
        summary = [
            ["Session ID", self.session_id],
            ["Started", manifest["started_at"]],
            ["Ended", manifest["ended_at"] or "—"],
            ["Timezone / UTC offset", f"{manifest['timezone']} / {manifest['utc_offset']}"],
            ["Capture interval", f"{self.interval_seconds:.2f} seconds"],
            ["Captured frames", str(len(self.events))],
            ["Successful / failed", f"{manifest['successful_frames']} / {manifest['failed_frames']}"],
            ["People detected (not unique)", str(manifest["total_people_detected"])],
            ["Confirmed violations", str(manifest["confirmed_violations"])],
            ["Uncertain findings", str(manifest["uncertain_findings"])],
            ["Recording", "finalized" if self.record_video and self._video_path.exists() else "not enabled"],
        ]
        table = Table(summary, colWidths=[2.1 * inch, 4.8 * inch])
        table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.grey), ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke)]))
        story.extend([table, Spacer(1, 12)])
        violation_counts: dict[str, int] = {}
        incident_statuses: dict[str, int] = {}
        notification_statuses: dict[str, int] = {}
        for event in self.events:
            for violation in event.get("violations", []):
                name = str(violation.get("type", "unknown"))
                violation_counts[name] = violation_counts.get(name, 0) + 1
            incident = event.get("incidents", {})
            incident_name = str(incident.get("status", "not_evaluated"))
            incident_statuses[incident_name] = incident_statuses.get(incident_name, 0) + 1
            alert_name = str(event.get("alert", {}).get("status", "Not triggered"))
            notification_statuses[alert_name] = notification_statuses.get(alert_name, 0) + 1
        distribution = [["Confirmed violation type", "Count"]] + [[name, str(count)] for name, count in sorted(violation_counts.items())]
        if len(distribution) == 1:
            distribution.append(["None", "0"])
        dist_table = Table(distribution, colWidths=[3.6 * inch, 1.2 * inch])
        dist_table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E5E7EB"))]))
        story.extend([Paragraph("Finding distribution", styles["Heading2"]), dist_table])
        model = self.config.get("model", {}) if isinstance(self.config, dict) else {}
        story.extend([
            Spacer(1, 8),
            Paragraph("Model and status information", styles["Heading2"]),
            Paragraph(f"PPE model: {model.get('weights_path', 'configured production model')}", styles["BodyText"]),
            Paragraph(f"Incident statuses: {incident_statuses or {'not_evaluated': 0}}", styles["BodyText"]),
            Paragraph(f"Notification statuses: {notification_statuses or {'Not triggered': 0}}", styles["BodyText"]),
            Spacer(1, 12),
            Paragraph("Chronological evidence", styles["Heading2"]),
        ])
        for index, event in enumerate(self.events):
            story.append(Paragraph(
                f"Frame {event['frame_number']:06d} — captured {event['captured_at']} — "
                f"status: {event['processing_status']} — violations: {event['violation_count']} — "
                f"uncertain: {event['uncertain_count']}", styles["BodyText"]
            ))
            if event.get("error"):
                story.append(Paragraph(f"Failure: {event['error']}", styles["BodyText"]))
            if event.get("violations"):
                story.append(Paragraph(f"Confirmed violations: {event['violations']}", styles["BodyText"]))
            if event.get("unverified_candidates"):
                story.append(Paragraph(f"Unverified candidates: {event['unverified_candidates']}", styles["BodyText"]))
            for key in ("original_path", "annotated_path"):
                path = self.session_dir / event.get(key, "")
                if path.is_file():
                    story.append(Paragraph(f"{key}: {path.name}", styles["BodyText"]))
                    story.append(ReportImage(str(path), width=3.25 * inch, height=2.15 * inch))
            if index < len(self.events) - 1:
                story.append(PageBreak())
        SimpleDocTemplate(str(report_path), pagesize=letter, rightMargin=0.5 * inch, leftMargin=0.5 * inch, topMargin=0.5 * inch, bottomMargin=0.5 * inch).build(story)
        return report_path

    def finalize(self) -> dict:
        if self.ended_at is not None:
            return self.summary()
        self.ended_at = _now()
        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None
        report_path = None
        report_error = ""
        try:
            report_path = self._generate_pdf()
        except Exception as exc:  # evidence remains available even if PDF fails
            report_error = str(exc)
        self._write_manifest("completed")
        bundle_path = self.session_dir / "session_bundle.zip"
        with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in self.session_dir.rglob("*"):
                if path.is_file() and path != bundle_path:
                    archive.write(path, path.relative_to(self.session_dir).as_posix())
        with zipfile.ZipFile(bundle_path) as archive:
            archive.testzip()
        summary = self.summary()
        summary.update({"report_path": str(report_path) if report_path else None, "bundle_path": str(bundle_path), "report_error": report_error})
        (self.session_dir / "session.json").write_text(json.dumps({**self._manifest("completed"), **summary}, indent=2, default=str), encoding="utf-8")
        return summary

    def summary(self) -> dict:
        return {
            "session_id": self.session_id,
            "session_dir": str(self.session_dir),
            "started_at": _iso(self.started_at),
            "ended_at": _iso(self.ended_at) if self.ended_at else None,
            "captured_frames": len(self.events),
            "successful_frames": sum(e.get("processing_status") == "success" for e in self.events),
            "failed_frames": sum(e.get("processing_status") == "failed" for e in self.events),
            "violations": sum(e.get("violation_count", 0) for e in self.events),
            "uncertain_findings": sum(e.get("uncertain_count", 0) for e in self.events),
            "report_path": str(self.reports_dir / "session_report.pdf") if (self.reports_dir / "session_report.pdf").exists() else None,
            "bundle_path": str(self.session_dir / "session_bundle.zip") if (self.session_dir / "session_bundle.zip").exists() else None,
        }
