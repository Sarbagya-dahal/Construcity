"""Automated PDF Incident and PPE Compliance Report Generator using ReportLab."""

from __future__ import annotations

import io
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    HRFlowable,
    Image as RLImage,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.utils import format_bbox, format_confidence, format_scan_timestamp

HUMAN_FLAG_MAP = {
    "head_classifier_fallback_no_helmet_overlap": "Needs review — head classifier fallback",
    "head_classifier_detector_conflict": "Needs review — detector conflict",
    "helmet_positive_negative_conflict": "Needs review — positive/negative prediction conflict",
    "helmet_below_worn_confidence_threshold": "Needs review — low helmet confidence (< 80%)",
    "head_region_has_low_confidence_helmet_candidate": "Needs review — low-confidence candidate at head",
    "fallback_no_helmet_overlap": "Needs review — fallback no helmet overlap",
    "helmet_status_uncertain": "Uncertain — helmet status unconfirmed",
    "vest_status_uncertain": "Uncertain — vest status unconfirmed",
    "without_helmet": "Missing Helmet",
    "without_vest": "Missing Vest",
    "no_person_detection_for_association": "Unverified — no worker associated",
    "outside_vest_spatial_bounds": "Rejected — outside vest torso boundaries",
    "outside_helmet_region": "Rejected — outside head region",
    "outside_person_or_iou_below_threshold": "Unassociated — outside person bounds",
}


def humanize_flag(flag: str) -> str:
    """Convert raw internal diagnostic flags into concise, professional descriptions."""
    if not flag:
        return "Clear"
    if flag in HUMAN_FLAG_MAP:
        return HUMAN_FLAG_MAP[flag]
    cleaned = flag.replace("_", " ").strip()
    return cleaned[0].upper() + cleaned[1:] if cleaned else "Clear"


def humanize_flag_list(flags: list[str]) -> str:
    """Format a list of flags into a deduplicated human-readable string."""
    if not flags:
        return "Clear"
    readable = [humanize_flag(f) for f in flags]
    return "; ".join(dict.fromkeys(readable))


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and draw total page numbers and report headers/footers."""

    report_id = "PPE-REPORT"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_decorations(self, num_pages: int):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#475569"))
        # Divider line
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(36, 32, 576, 32)
        # Left footer: Report ID
        rep_id = getattr(self, "report_id", "PPE-REPORT")
        self.drawString(36, 22, f"Report ID: {rep_id}")
        # Center footer: Document classification
        self.drawCentredString(306, 22, "Confidential — Automated PPE Compliance Record")
        # Right footer: Page X of Y
        self.drawRightString(576, 22, f"Page {self._pageNumber} of {num_pages}")
        self.restoreState()


def make_canvas_class(rep_id: str):
    class CustomNumberedCanvas(NumberedCanvas):
        report_id = rep_id
    return CustomNumberedCanvas


def should_generate_report(result: dict, config: dict) -> bool:
    """Determine whether an automated PDF report should be generated based on findings."""
    reporting_cfg = config.get("reporting", {})
    if not reporting_cfg.get("enabled", True):
        return False

    violations = result.get("violations", {})
    incidents = result.get("incidents", {})

    has_violation = violations.get("violation_count", 0) > 0
    needs_review = violations.get("status") == "needs_review"
    has_incident = incidents.get("has_candidate", False)

    if reporting_cfg.get("auto_generate_on_violation", True) and has_violation:
        return True
    if reporting_cfg.get("auto_generate_on_incident", False) and has_incident:
        return True
    if reporting_cfg.get("auto_generate_on_review", True) and needs_review:
        return True

    return False


class PDFReportGenerator:
    """Generate structured, professional PDF incident and compliance reports."""

    def __init__(self, config: dict):
        self.config = config
        reporting_cfg = config.get("reporting", {})
        self.output_dir = Path(reporting_cfg.get("output_dir", "outputs/reports")).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _determine_overall_status(self, result: dict) -> tuple[str, colors.Color]:
        incidents = result.get("incidents", {})
        violations = result.get("violations", {})

        if incidents.get("validated", False) and incidents.get("status") == "POTENTIAL_INJURY_SUSPECTED":
            return "POTENTIAL INCIDENT DETECTED (URGENT REVIEW REQUIRED)", colors.HexColor("#991B1B")
        if violations.get("violation_count", 0) > 0:
            return "PPE VIOLATION DETECTED", colors.HexColor("#DC2626")
        if violations.get("status") == "needs_review":
            return "NEEDS HUMAN REVIEW", colors.HexColor("#D97706")
        if violations.get("status") == "evaluated_with_uncertainty":
            return "NEEDS REVIEW (UNCERTAIN PPE STATUS)", colors.HexColor("#D97706")
        if violations.get("status") == "person_missing":
            return "NEEDS REVIEW (NO PERSON DETECTED)", colors.HexColor("#4B5563")

        return "PPE COMPLIANT", colors.HexColor("#16A34A")

    @staticmethod
    def _cv2_to_reportlab_image(cv2_image: np.ndarray, max_w_inch: float = 3.4, max_h_inch: float = 2.4) -> RLImage:
        """Convert an in-memory BGR cv2 image into a ReportLab Image Flowable."""
        h, w = cv2_image.shape[:2]
        aspect = w / float(h)

        target_w = max_w_inch * inch
        target_h = target_w / aspect
        if target_h > max_h_inch * inch:
            target_h = max_h_inch * inch
            target_w = target_h * aspect

        success, encoded = cv2.imencode(".jpg", cv2_image, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not success:
            raise ValueError("Failed to encode frame for PDF")

        img_stream = io.BytesIO(encoded.tobytes())
        return RLImage(img_stream, width=target_w, height=target_h)

    def generate_report(
        self,
        result: dict,
        source_name: str,
        original_image: Optional[np.ndarray] = None,
        annotated_image: Optional[np.ndarray] = None,
    ) -> str:
        """Create and write a PDF report file, returning its absolute path."""
        report_id = f"REP-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6].upper()}"

        # Propagate exact scan timestamp from result dictionary
        raw_scan_time = result.get("scan_timestamp") or result.get("timestamp")
        scanned_at_str = format_scan_timestamp(raw_scan_time)

        safe_stem = Path(source_name).stem.replace(" ", "_")
        filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe_stem}_{report_id[-6:]}.pdf"
        report_path = self.output_dir / filename

        doc = SimpleDocTemplate(
            str(report_path),
            pagesize=letter,
            rightMargin=0.5 * inch,
            leftMargin=0.5 * inch,
            topMargin=0.5 * inch,
            bottomMargin=0.6 * inch,
        )

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "DocTitle",
            parent=styles["Heading1"],
            fontSize=16,
            leading=20,
            textColor=colors.HexColor("#1E293B"),
            spaceAfter=3,
        )
        subtitle_style = ParagraphStyle(
            "DocSubTitle",
            parent=styles["Normal"],
            fontSize=9,
            leading=13,
            textColor=colors.HexColor("#475569"),
        )
        heading_style = ParagraphStyle(
            "SectionHeading",
            parent=styles["Heading2"],
            fontSize=11,
            leading=15,
            textColor=colors.HexColor("#0F172A"),
            spaceBefore=8,
            spaceAfter=4,
            keepWithNext=True,
        )
        body_style = ParagraphStyle(
            "Body",
            parent=styles["Normal"],
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#334155"),
        )
        table_header_style = ParagraphStyle(
            "TableHeader",
            parent=styles["Normal"],
            fontSize=8,
            leading=10,
            fontName="Helvetica-Bold",
            textColor=colors.white,
        )
        table_cell_style = ParagraphStyle(
            "TableCell",
            parent=styles["Normal"],
            fontSize=7.5,
            leading=10,
            textColor=colors.HexColor("#1E293B"),
        )
        code_cell_style = ParagraphStyle(
            "CodeCell",
            parent=styles["Normal"],
            fontSize=7,
            leading=9,
            fontName="Courier",
            textColor=colors.HexColor("#0F172A"),
        )
        badge_style = ParagraphStyle(
            "Badge",
            parent=styles["Normal"],
            fontSize=10,
            leading=13,
            fontName="Helvetica-Bold",
            textColor=colors.white,
            alignment=1,
        )

        story = []

        # --- Header ---
        story.append(Paragraph("PPE SAFETY MONITOR — INCIDENT & COMPLIANCE REPORT", title_style))
        story.append(Paragraph(f"Automated Workplace Safety Inspection Log | <b>Scanned at:</b> {scanned_at_str}", subtitle_style))
        story.append(Spacer(1, 4))

        # --- Status Banner ---
        status_text, status_color = self._determine_overall_status(result)
        banner_table = Table(
            [[Paragraph(status_text, badge_style)]],
            colWidths=[7.5 * inch],
        )
        banner_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), status_color),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ]))
        story.append(banner_table)
        story.append(Spacer(1, 6))

        # --- Metadata Summary Table ---
        violations = result.get("violations", {})
        incidents = result.get("incidents", {})
        person_count = violations.get("person_count", 0)
        violation_count = violations.get("violation_count", 0)
        compliance_rate = violations.get("compliance_rate", 0.0)
        blood_status_display = incidents.get("status_display", "Not evaluated (Unavailable)")

        meta_data = [
            [
                Paragraph("<b>Report ID:</b>", body_style), Paragraph(report_id, body_style),
                Paragraph("<b>Source File:</b>", body_style), Paragraph(source_name, body_style),
            ],
            [
                Paragraph("<b>Scanned at:</b>", body_style), Paragraph(scanned_at_str, body_style),
                Paragraph("<b>Persons Detected:</b>", body_style), Paragraph(str(person_count), body_style),
            ],
            [
                Paragraph("<b>Compliance Rate:</b>", body_style), Paragraph(f"{compliance_rate:.1f}%", body_style),
                Paragraph("<b>Violations Flagged:</b>", body_style), Paragraph(str(violation_count), body_style),
            ],
            [
                Paragraph("<b>Blood / Injury Status:</b>", body_style), Paragraph(blood_status_display, body_style),
                Paragraph("<b>Decision Policy:</b>", body_style), Paragraph(status_text, body_style),
            ],
        ]
        meta_table = Table(meta_data, colWidths=[1.4 * inch, 2.35 * inch, 1.4 * inch, 2.35 * inch])
        meta_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(meta_table)
        story.append(Spacer(1, 6))

        # --- Image Comparison Section ---
        orig_frame = original_image if original_image is not None else result.get("original_frame")
        annot_frame = annotated_image if annotated_image is not None else result.get("annotated_frame")

        if orig_frame is not None and annot_frame is not None:
            story.append(Paragraph("Visual Evidence (Original Capture vs. Clean Annotated Scan)", heading_style))
            rl_orig = self._cv2_to_reportlab_image(orig_frame)
            rl_annot = self._cv2_to_reportlab_image(annot_frame)

            img_table = Table(
                [[rl_orig, rl_annot],
                 [Paragraph("Original Image Capture", body_style), Paragraph("Annotated PPE Scan", body_style)]],
                colWidths=[3.75 * inch, 3.75 * inch],
            )
            img_table.setStyle(TableStyle([
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]))
            story.append(img_table)
            story.append(Spacer(1, 6))

        # --- Section 1: Personnel PPE Compliance Details ---
        story.append(Paragraph("1. Personnel PPE Compliance Details", heading_style))
        person_results = violations.get("person_results", [])
        if person_results:
            p_data = [[
                Paragraph("Person ID", table_header_style),
                Paragraph("Helmet Status", table_header_style),
                Paragraph("Vest Status", table_header_style),
                Paragraph("Violations", table_header_style),
                Paragraph("Review Status", table_header_style),
            ]]
            for p in person_results:
                pid = f"Worker #{p.get('person_id', 1)}"
                h_stat = "Worn (Compliant)" if p.get("has_helmet") else "Missing / Unverified"
                v_stat = "Worn (Compliant)" if p.get("has_vest") else "Missing"
                
                raw_viols = p.get("violations", [])
                human_viols = humanize_flag_list(raw_viols)
                
                raw_rev = p.get("review_reasons", [])
                if not raw_rev and p.get("uncertain"):
                    raw_rev = p.get("uncertain", [])
                human_rev = humanize_flag_list(raw_rev)

                p_data.append([
                    Paragraph(pid, table_cell_style),
                    Paragraph(h_stat, table_cell_style),
                    Paragraph(v_stat, table_cell_style),
                    Paragraph(human_viols, table_cell_style),
                    Paragraph(human_rev, table_cell_style),
                ])

            p_table = Table(
                p_data,
                colWidths=[1.0 * inch, 1.3 * inch, 1.1 * inch, 1.6 * inch, 2.5 * inch],
                repeatRows=1,
            )
            p_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ]))
            story.append(p_table)
        else:
            story.append(Paragraph("No individual worker profiles were evaluable.", body_style))
        story.append(Spacer(1, 6))

        # --- Section 2: Visual Incident & Blood / Injury Assessment ---
        story.append(Paragraph("2. Visual Incident & Blood / Injury Assessment", heading_style))
        if not incidents.get("available", False):
            notice_text = (
                "<b>Status: NOT EVALUATED (Reliable blood detection unavailable)</b><br/>"
                "A validated blood or physical injury detector is not currently installed. "
                "The simple color heuristic is disabled in production because it produces false positive alerts "
                "on hazard tape, warning signage, red garments, and tools.<br/>"
                "<b>IMPORTANT SAFETY NOTICE:</b> This image has NOT been evaluated for blood or physical trauma. "
                "The absence of alerts does NOT confirm or imply that the scene or worker is free of blood or injuries."
            )
            notice_table = Table([[Paragraph(notice_text, body_style)]], colWidths=[7.5 * inch])
            notice_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FEF3C7")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#F59E0B")),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ]))
            story.append(notice_table)
        elif incidents.get("validated", False) and incidents.get("candidates"):
            inc_data = [[
                Paragraph("Candidate ID", table_header_style),
                Paragraph("Label", table_header_style),
                Paragraph("Confidence", table_header_style),
                Paragraph("Area", table_header_style),
                Paragraph("Context", table_header_style),
                Paragraph("Review Notice", table_header_style),
            ]]
            for idx, c in enumerate(incidents.get("candidates", []), start=1):
                cid = f"INC-{idx:02d}"
                lbl = c.get("label", "POSSIBLE_BLOOD")
                conf_str = f"{c.get('confidence', 0.0) * 100:.1f}%"
                area_str = f"{c.get('area_px', 0)} px"
                near_w = "Adjacent to Worker" if c.get("near_person") else "General Scene"
                notice = c.get("review_notice", "Requires immediate qualified human review.")
                inc_data.append([
                    Paragraph(cid, table_cell_style),
                    Paragraph(lbl, table_cell_style),
                    Paragraph(conf_str, table_cell_style),
                    Paragraph(area_str, table_cell_style),
                    Paragraph(near_w, table_cell_style),
                    Paragraph(notice, table_cell_style),
                ])

            inc_table = Table(
                inc_data,
                colWidths=[0.9 * inch, 1.5 * inch, 0.8 * inch, 0.8 * inch, 1.4 * inch, 2.1 * inch],
                repeatRows=1,
            )
            inc_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#991B1B")),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ]))
            story.append(inc_table)
        else:
            story.append(Paragraph("No visual incident candidates detected by validated model.", body_style))
        story.append(Spacer(1, 6))

        # --- Section 3: Spatial Detection Details & Coordinates ---
        story.append(Paragraph("3. Spatial Detection Details & Bounding Box Coordinates", heading_style))
        
        det_rows = [[
            Paragraph("Subject / Detection", table_header_style),
            Paragraph("Category", table_header_style),
            Paragraph("Confidence", table_header_style),
            Paragraph("Spatial Coordinates (pixels)", table_header_style),
        ]]

        for p in person_results:
            pid = f"Worker #{p.get('person_id', 1)}"
            det_rows.append([
                Paragraph(f"<b>{pid}</b> (Person Body)", table_cell_style),
                Paragraph("Person", table_cell_style),
                Paragraph("Verified", table_cell_style),
                Paragraph(format_bbox(p.get("bbox")), code_cell_style),
            ])
            for item in p.get("associations", {}).get("helmet", []):
                conf = format_confidence(item.get("confidence", 0.0))
                det_rows.append([
                    Paragraph("&nbsp;&nbsp;↳ Associated Helmet", table_cell_style),
                    Paragraph("Helmet", table_cell_style),
                    Paragraph(conf, table_cell_style),
                    Paragraph(format_bbox(item.get("bbox")), code_cell_style),
                ])
            for item in p.get("associations", {}).get("vest", []):
                conf = format_confidence(item.get("confidence", 0.0))
                det_rows.append([
                    Paragraph("&nbsp;&nbsp;↳ Associated Vest", table_cell_style),
                    Paragraph("Vest", table_cell_style),
                    Paragraph(conf, table_cell_style),
                    Paragraph(format_bbox(item.get("bbox")), code_cell_style),
                ])
            for item in p.get("associations", {}).get("without_helmet", []):
                det_rows.append([
                    Paragraph("&nbsp;&nbsp;↳ Missing Helmet Flag", table_cell_style),
                    Paragraph("Missing Helmet", table_cell_style),
                    Paragraph(format_confidence(item.get("confidence", 0.0)), table_cell_style),
                    Paragraph(format_bbox(item.get("bbox")), code_cell_style),
                ])
            for item in p.get("associations", {}).get("without_vest", []):
                det_rows.append([
                    Paragraph("&nbsp;&nbsp;↳ Missing Vest Flag", table_cell_style),
                    Paragraph("Missing Vest", table_cell_style),
                    Paragraph(format_confidence(item.get("confidence", 0.0)), table_cell_style),
                    Paragraph(format_bbox(item.get("bbox")), code_cell_style),
                ])

        for uv in violations.get("unverified_candidates", []):
            det_rows.append([
                Paragraph(f"Unverified {uv.get('class_name', 'Item')}", table_cell_style),
                Paragraph(uv.get("class_name", "Unverified"), table_cell_style),
                Paragraph(format_confidence(uv.get("confidence", 0.0)), table_cell_style),
                Paragraph(format_bbox(uv.get("bbox")), code_cell_style),
            ])

        for idx, c in enumerate(incidents.get("candidates", []), start=1):
            det_rows.append([
                Paragraph(f"Candidate INC-{idx:02d}", table_cell_style),
                Paragraph("Incident Candidate", table_cell_style),
                Paragraph(format_confidence(c.get("confidence", 0.0)), table_cell_style),
                Paragraph(format_bbox(c.get("bbox")), code_cell_style),
            ])

        if len(det_rows) > 1:
            det_table = Table(
                det_rows,
                colWidths=[1.8 * inch, 1.2 * inch, 1.0 * inch, 3.5 * inch],
                repeatRows=1,
            )
            det_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#334155")),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
                ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ]))
            story.append(det_table)
        else:
            story.append(Paragraph("No spatial bounding boxes recorded.", body_style))
        story.append(Spacer(1, 6))

        # --- Section 4: Operational Configuration & Limitations ---
        story.append(Paragraph("4. Operational Configuration & Limitations", heading_style))
        limitation_text = (
            "<b>Disclaimer & Operational Scope:</b><br/>"
            "• <b>PPE Model:</b> YOLOv8 4-class detector (models/trained/best.pt) + YOLOv8n person detector.<br/>"
            "• <b>Spatial Verification:</b> Helmets and vests are validated using person-relative spatial boundaries. "
            "Helmets below the worn-confidence gate (0.80) or unverified by head geometry escalate to NEEDS REVIEW.<br/>"
            "• <b>Blood & Injury Detection Notice:</b> Reliable automated blood detection is NOT currently available. "
            "Color-threshold heuristics are disabled to prevent false positives on hazard tape and red items. "
            "This scan is NOT a medical screening and must not be used as evidence that an injury is absent.<br/>"
            "• <b>Audit Trail:</b> The timestamp reflects the exact time the scan was processed (Scanned at), preserving timezone context."
        )
        story.append(Paragraph(limitation_text, body_style))

        canvas_class = make_canvas_class(report_id)
        doc.build(story, canvasmaker=canvas_class)
        return str(report_path.resolve())
