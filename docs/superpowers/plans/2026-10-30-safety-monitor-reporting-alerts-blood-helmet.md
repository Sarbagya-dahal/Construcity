# Implementation Plan: PPE Safety Monitor — Reporting, Alerts, Blood/Incident Detection, and Helmet Accuracy

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enhance the PPE detection project with readable high-contrast annotations preserving original image fidelity, automatic PDF incident report generation, manager email/dashboard alerting with cooldown deduplication, conservative candidate blood/injury detection, head-region helmet accuracy evaluation, and unified frame-level interfaces for future camera streaming.

**Architecture:**
- **Annotation & Visualization (`src/utils.py`, `src/violation_logic.py`, `src/processor.py`):** Draw on unmodified original frame copies, scale font/thickness by image dimensions, high-contrast dynamic text color based on background luminance, clear human-readable labels (`HELMET: 85.0%`, `MISSING VEST`, `NEEDS REVIEW`, etc.).
- **Incident PDF Reporting (`src/report_generator.py`):** Automatically create structured PDF reports via `reportlab` on actionable findings (`outputs/reports/`), with download buttons in Streamlit.
- **Alert & Notification System (`src/notifications.py`):** Configurable severity alerts, SMTP email dispatch with attachment, cooldown and incident deduplication, graceful handling of missing credentials, `.env.example`.
- **Blood / Visual Incident Candidate Detector (`src/incident_detector.py`):** Conservative red-region candidate detection (`RED_REGION_CANDIDATE`, never `BLOOD_CONFIRMED`), filtering out red garments/tools/skin tones, with escalation to urgent human review.
- **Helmet Accuracy & Dedicated Head Model (`src/head_classifier.py`, `scripts/`):** Evaluate baseline vs head-verification candidate, train candidate on reviewed splits without overwriting `best.pt`, compare metrics honestly.
- **Stream/Camera Ready Interface & Deduplication (`src/processor.py`, `src/incident_tracker.py`):** Cooldown tracker per incident type/person so persistent video frames do not flood reports/emails.

**Tech Stack:** Python 3.10+, PyTorch, OpenCV, Ultralytics YOLOv8, Streamlit, ReportLab 5.1.0, unittest.

---

## Task 1: Fix Image Brightness Discrepancy & Overhaul Annotation Readability

**Files:**
- Modify: `src/processor.py`
- Modify: `src/utils.py`
- Modify: `src/violation_logic.py`
- Test: `tests/test_visualization.py`

**Interfaces:**
- `draw_bbox(image, bbox, label, color, thickness=None, text_size=None)` dynamically sizes text and border based on image resolution, computes contrasting font color (white or black) based on luminance $L = 0.114B + 0.587G + 0.299R$, and avoids clipping at borders.
- `annotate_violations(image, detections, violation_result, config, diagnostics=None)` annotates directly on a clean copy of the original image (not brightened `enhanced_frame`), uses clear human labels (`HELMET: 94.5%`, `MISSING HELMET`, `MISSING VEST`, `NEEDS REVIEW`, `RED_REGION_CANDIDATE`), and keeps raw rejections off the primary image.

- [ ] **Step 1: Write failing test in `tests/test_visualization.py`**
  - Verify `draw_bbox` text contrast: black text on bright colors (e.g. yellow `(0, 255, 255)`), white text on dark colors (e.g. blue `(255, 0, 0)`).
  - Verify `process_frame` does not alter image brightness outside bounding boxes.
  - Verify labels do not clip outside image boundaries when bbox is near $(0, 0)$ or $(W, H)$.

- [ ] **Step 2: Run test to confirm it fails**
  - Run: `.venv\Scripts\python.exe -m unittest tests.test_visualization -v`

- [ ] **Step 3: Implement dynamic scaling & contrast in `src/utils.py` and original frame annotation in `src/processor.py`**
  - Update `draw_bbox` in `src/utils.py`.
  - In `src/processor.py`, pass `frame.copy()` (not `enhanced_frame`) to `annotate_violations`.
  - In `src/violation_logic.py`, refine labels to clear uppercase indicators (`HELMET: 88.5%`, `UNVERIFIED HELMET`, `MISSING VEST`, `NEEDS REVIEW`).

- [ ] **Step 4: Run tests to verify they pass**
  - Run: `.venv\Scripts\python.exe -m unittest tests.test_visualization -v`

- [ ] **Step 5: Run full test suite**
  - Run: `.venv\Scripts\python.exe -m unittest discover -s tests -v`

---

## Task 2: Implement Blood/Visual Incident Candidate Detector

**Files:**
- Create: `src/incident_detector.py`
- Modify: `src/processor.py`
- Modify: `config/config.yaml`
- Test: `tests/test_incident_detector.py`

**Interfaces:**
- `IncidentDetector.detect_incidents(frame: np.ndarray, persons: list[dict]) -> dict`
- Output dict:
  - `has_candidate: bool`
  - `status: Literal["NO_CANDIDATE_DETECTED", "RED_REGION_CANDIDATE", "POTENTIAL_INJURY_SUSPECTED", "UNCERTAIN"]`
  - `candidates: list[dict]` (each with `bbox`, `confidence`, `label="RED_REGION_CANDIDATE"`, `area_ratio`, `reason`)
  - `limitations: str` (explicit disclaimer: non-medical, candidate heuristic, requires human assessment)

- [ ] **Step 1: Write failing test in `tests/test_incident_detector.py`**
  - Test synthetic neutral image -> `NO_CANDIDATE_DETECTED`.
  - Test localized dark-red textured region near person -> `RED_REGION_CANDIDATE` with explicit non-medical disclaimer.
  - Test synthetic uniform red tool/vest -> filtered out or classified as non-blood candidate due to high uniform saturation/brightness.
  - Test label is NEVER `BLOOD_CONFIRMED`.

- [ ] **Step 2: Run test to confirm it fails**
  - Run: `.venv\Scripts\python.exe -m unittest tests.test_incident_detector -v`

- [ ] **Step 3: Implement `src/incident_detector.py`**
  - Multi-condition red region segmentation in HSV/Lab color space.
  - Texture variance check (blood exhibits gradient/texture variance; uniform plastics/vests exhibit low standard deviation).
  - Person proximity association.
  - Strict classification naming policy.

- [ ] **Step 4: Integrate into `src/processor.py` and `config/config.yaml`**
  - Add `incident_detection` section in `config/config.yaml`.
  - In `FrameProcessor.process_frame`, run `IncidentDetector` and attach `incidents` to result dict.
  - Annotate detected incident candidates with high-visibility badge: `RED_REGION_CANDIDATE (NEEDS REVIEW)`.

- [ ] **Step 5: Run tests to verify they pass**
  - Run: `.venv\Scripts\python.exe -m unittest tests.test_incident_detector -v`

---

## Task 3: Automatic PDF Incident Report Generation

**Files:**
- Create: `src/report_generator.py`
- Modify: `src/processor.py`
- Modify: `config/config.yaml`
- Test: `tests/test_report_generator.py`

**Interfaces:**
- `PDFReportGenerator.generate_report(result: dict, source_name: str, original_image: np.ndarray) -> str` returns path to saved PDF in `outputs/reports/`.
- Report contains: Report ID, timestamp, source filename, embedded original & annotated images, detected persons, PPE detections, confirmed violations, unverified candidates, incident/blood candidates with review notices, overall status (`Compliant`, `Violation detected`, `Needs review`), and system limitations.

- [ ] **Step 1: Write failing test in `tests/test_report_generator.py`**
  - Test generating PDF for compliant scan, violation scan, and incident scan.
  - Verify PDF file is created, non-empty, and valid PDF header `%PDF-`.
  - Verify correct status and metadata are written.

- [ ] **Step 2: Run test to confirm it fails**
  - Run: `.venv\Scripts\python.exe -m unittest tests.test_report_generator -v`

- [ ] **Step 3: Implement `src/report_generator.py` using ReportLab**
  - Clean two-column or stacked layout with header, metadata table, embedded side-by-side images, findings table, and disclaimer.

- [ ] **Step 4: Wire automatic generation into `src/processor.py`**
  - Configurable trigger: generate if `violations > 0` or `status == "needs_review"` or `incidents["has_candidate"]`.
  - Store generated report path in `result["report_path"]`.

- [ ] **Step 5: Run tests to verify they pass**
  - Run: `.venv\Scripts\python.exe -m unittest tests.test_report_generator -v`

---

## Task 4: Manager Alerts, Notifications & Cooldown Deduplication

**Files:**
- Create: `src/notifications.py`
- Create: `.env.example`
- Modify: `config/config.yaml`
- Modify: `src/processor.py`
- Test: `tests/test_notifications.py`

**Interfaces:**
- `NotificationManager.send_alert(result: dict, report_path: str = None) -> dict`
- Returns: `{"sent": bool, "status": str, "severity": str, "channel": "email" | "local_only", "throttled": bool}`.
- Severity levels: `URGENT_INJURY_REVIEW`, `VIOLATION_ALERT`, `REVIEW_ALERT`.
- Throttling: If the same source/incident occurred within `cooldown_seconds` (default 60s), suppress duplicate dispatch.
- Graceful fallback: If SMTP credentials missing, log `Email not configured` and return `sent=False`.

- [ ] **Step 1: Write failing test in `tests/test_notifications.py`**
  - Test missing email config reports `Email not configured` without throwing.
  - Test mock SMTP transport successfully transmits email with PDF attachment.
  - Test cooldown throttling prevents repeated dispatch for same source within cooldown window.

- [ ] **Step 2: Run test to confirm it fails**
  - Run: `.venv\Scripts\python.exe -m unittest tests.test_notifications -v`

- [ ] **Step 3: Implement `src/notifications.py` and `.env.example`**
  - Read from environment / config: `SMTP_SERVER`, `SMTP_PORT`, `SMTP_USERNAME`, `SMTP_PASSWORD`, `ALERT_RECIPIENT_EMAIL`.
  - Build MIME email with report summary and PDF attachment.
  - Timestamp-based incident key deduplication.

- [ ] **Step 4: Wire notification manager into `src/processor.py`**
  - Attach `result["alert"]` with delivery status.

- [ ] **Step 5: Run tests to verify they pass**
  - Run: `.venv\Scripts\python.exe -m unittest tests.test_notifications -v`

---

## Task 5: Frame-Processing Stream/Camera Interface

**Files:**
- Modify: `src/processor.py`
- Test: `tests/test_stream_interface.py`

**Interfaces:**
- `FrameProcessor.process_stream_frame(frame, frame_id, source_id, timestamp=None)`:
  - Standardized interface for both still images and streaming frames.
  - Handles frame-rate skipping, per-stream cooldown tracking, returns structured findings without re-alerting on static scenes.

- [ ] **Step 1: Write failing test in `tests/test_stream_interface.py`**
  - Consecutive identical frames do not generate duplicate reports.
  - Resetting stream state clears tracker.

- [ ] **Step 2: Implement `process_stream_frame` in `src/processor.py`**

- [ ] **Step 3: Run test to verify it passes**
  - Run: `.venv\Scripts\python.exe -m unittest tests.test_stream_interface -v`

---

## Task 6: Streamlit Dashboard Enhancements

**Files:**
- Modify: `app/dashboard.py`
- Test: `tests/test_dashboard_integration.py`

**Features:**
- Side-by-side original and high-contrast annotated image.
- Visual status cards for PPE violations, visual incident candidates, and needs_review.
- Direct PDF report download button.
- Alert delivery status display (e.g. `Email not configured - local alert created`).
- Collapsible diagnostics section for raw detector predictions.

- [ ] **Step 1: Add unit test verifying dashboard component rendering helpers**
- [ ] **Step 2: Update `app/dashboard.py`**
- [ ] **Step 3: Verify dashboard imports and passes headless syntax check**

---

## Task 7: Helmet Accuracy Review & Experiment Evaluation

**Files:**
- Modify: `scripts/evaluate_dataset.py` (or candidate evaluation)
- Review: `data/head_classifier/manifest.csv`
- Test: `tests/test_detection_pipeline.py`

**Work:**
- Benchmark baseline model against head-verification pipeline.
- Verify production checkpoint `best.pt` remains unchanged.
- Document baseline vs candidate metrics and promotion decision.

- [ ] **Step 1: Run quantitative evaluation across demo images and validation set**
- [ ] **Step 2: Verify SHA-256 of `models/trained/best.pt`**
- [ ] **Step 3: Record evaluation metrics in report**

---

## Task 8: Full Verification & Final Deliverable

- [ ] **Step 1: Run full unit & regression suite**
- [ ] **Step 2: Run CLI smoke test on all demo images**
- [ ] **Step 3: Verify generated PDF opens and contains required sections**
- [ ] **Step 4: Prepare final report answering all 10 required deliverable items**

