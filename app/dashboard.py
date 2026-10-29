"""Local Streamlit dashboard backed by the existing FrameProcessor pipeline."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.processor import FrameProcessor
from src.utils import create_detection_summary, format_scan_timestamp, load_config, log_violation

CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
EVALUATION_PATH = PROJECT_ROOT / "outputs" / "evaluation_report.json"
CLASS_ORDER = ("person", "helmet", "vest", "without_helmet", "without_vest")

CONFIG = load_config(str(CONFIG_PATH))
st.set_page_config(page_title=CONFIG.get("dashboard", {}).get("title", "PPE Detection"),
                   page_icon=CONFIG.get("dashboard", {}).get("page_icon", "PPE"),
                   layout=CONFIG.get("dashboard", {}).get("layout", "wide"))
FRONTEND_CSS = Path(__file__).with_name("frontend.css")
if FRONTEND_CSS.exists():
    st.markdown(f"<style>{FRONTEND_CSS.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def ui_title(title: str, kicker: str) -> None:
    """Render presentation-only page hierarchy."""
    st.markdown(f'<p class="kicker">{kicker}</p><h2 class="page-title">{title}</h2>', unsafe_allow_html=True)


def ui_section(title: str) -> None:
    st.markdown(f'<h3 class="section-title">{title}</h3>', unsafe_allow_html=True)


def absolute_config(config: dict) -> dict:
    config = json.loads(json.dumps(config))
    config["model"]["weights_path"] = str((PROJECT_ROOT / config["model"]["weights_path"]).resolve())
    config["model"]["person_model_path"] = str((PROJECT_ROOT / config["model"].get("person_model_path", "yolov8n.pt")).resolve())
    if config.get("head_classifier", {}).get("checkpoint_path"):
        config["head_classifier"]["checkpoint_path"] = str((PROJECT_ROOT / config["head_classifier"]["checkpoint_path"]).resolve())
    for key in ("violations_dir", "log_file"):
        if config.get("storage", {}).get(key):
            config["storage"][key] = str((PROJECT_ROOT / config["storage"][key]).resolve())
    return config


@st.cache_resource
def load_processor(config_mtime: int) -> FrameProcessor:
    config = absolute_config(load_config(str(CONFIG_PATH)))
    model_path = config["model"]["weights_path"]
    if not Path(model_path).exists():
        raise FileNotFoundError(model_path)
    return FrameProcessor(config, model_path)


def read_history(processor: FrameProcessor) -> list[dict]:
    path = Path(processor.config["storage"].get("log_file", ""))
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return data if isinstance(data, list) else []


def append_history(processor: FrameProcessor, result: dict, source: str) -> None:
    log_path = processor.config["storage"].get("log_file")
    if not log_path:
        return
    scan_ts = result.get("scan_timestamp") or result.get("timestamp") or datetime.now().astimezone().isoformat()
    log_violation(log_path, {
        "timestamp": scan_ts,
        "scanned_at": format_scan_timestamp(scan_ts),
        "source": source,
        "source_type": "image",
        "violation_count": result["violations"]["violation_count"],
        "compliance_rate": result["violations"]["compliance_rate"],
        "detection_summary": create_detection_summary(result["detections"]),
        "violations": result["violations"]["violations"],
        "unverified_candidates": result["violations"].get("unverified_candidates", []),
    })


def process_uploaded_image(processor: FrameProcessor, image_bytes: bytes, source_name: str) -> dict:
    """Decode an uploaded image and run the same pipeline used by the CLI."""
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode uploaded image: {source_name}")
    result = processor.process_frame(
        image,
        save_violation=processor.save_violations,
        source_name=source_name,
    )
    append_history(processor, result, source_name)
    return result


def render_result(result: dict, show_confidence: bool = True) -> None:
    detections = result["detections"]
    violations = result["violations"]
    incidents = result.get("incidents", {})
    alert = result.get("alert", {})
    summary = create_detection_summary(detections)

    scan_ts = result.get("scan_timestamp") or result.get("timestamp")
    st.markdown(f'<div class="meta-line">SCAN TIMESTAMP &nbsp; {format_scan_timestamp(scan_ts)}</div>', unsafe_allow_html=True)

    if incidents.get("validated", False) and incidents.get("status") == "POTENTIAL_INJURY_SUSPECTED":
        st.markdown('<div class="status-band alert">URGENT INCIDENT REVIEW REQUIRED<small>Potential injury/incident finding. Human confirmation is required.</small></div>', unsafe_allow_html=True)
    elif violations.get("violation_count"):
        st.markdown(f'<div class="status-band alert">{violations["violation_count"]} CONFIRMED PPE VIOLATION(S)<small>Review the evidence and incident report before taking action.</small></div>', unsafe_allow_html=True)
    elif violations.get("status") == "needs_review":
        st.markdown('<div class="status-band review">REVIEW UNCERTAIN DETECTION<small>Uncertain helmet evidence is not treated as compliance.</small></div>', unsafe_allow_html=True)
    elif violations.get("status") == "evaluated_with_uncertainty":
        st.markdown('<div class="status-band review">PARTIAL EVALUATION<small>Some PPE statuses are uncertain because no positive or explicit negative detection was associated.</small></div>', unsafe_allow_html=True)
    elif violations.get("status") == "person_missing":
        st.markdown('<div class="status-band review">NO WORKER ASSOCIATED<small>PPE-only detections remain unverified and are not counted as confirmed violations.</small></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-band">NO CONFIRMED PPE VIOLATIONS<small>All available evidence was evaluated without a confirmed violation.</small></div>', unsafe_allow_html=True)

    # Alert & Dispatch Status Card
    if alert.get("sent"):
        st.success(f"Safety Alert Dispatched to Manager (Severity: {alert.get('severity', 'NORMAL')})")
    elif alert.get("throttled"):
        st.info(f"Notification Status: {alert.get('status')}")
    elif alert.get("status") == "Email not configured":
        st.info("Notification Status: Email not configured (Local incident logged). Add credentials in .env to enable manager emails.")
    elif alert.get("status") and alert.get("status") not in ("Not triggered", "No actionable finding triggered"):
        st.warning(f"Notification Status: {alert.get('status')}")

    ppe_count = sum(summary.get(name, 0) for name in ("helmet", "vest", "without_helmet", "without_vest"))
    st.markdown(
        '<div class="metric-strip">'
        f'<div><div class="metric-label">Workers</div><div class="metric-value">{summary["person"]}</div></div>'
        f'<div><div class="metric-label">PPE objects</div><div class="metric-value">{ppe_count}</div></div>'
        f'<div><div class="metric-label">Confirmed violations</div><div class="metric-value">{violations["violation_count"]}</div></div>'
        f'<div><div class="metric-label">Compliance</div><div class="metric-value">{violations["compliance_rate"]:.1f}%</div></div>'
        '</div>', unsafe_allow_html=True,
    )

    # Blood / Incident Status Section (Truthful disclosure, Phase 1 & 2)
    if not incidents.get("available", False):
        st.info("ℹ️ **Blood / Injury Detection Status:** Reliable blood detection is currently unavailable. "
                "The simple color heuristic is disabled in production to eliminate false alerts from hazard tape, "
                "red tools, and clothing. *Notice: This scan has not been evaluated for blood or injuries; "
                "absence of alerts does not confirm that no blood or injury is present.*")
    elif incidents.get("validated", False) and incidents.get("candidates"):
        ui_section("Validated incident / injury candidates")
        st.dataframe(pd.DataFrame(incidents["candidates"]), use_container_width=True, hide_index=True)
        st.caption(incidents.get("limitations", ""))
    elif incidents.get("candidates"):
        with st.expander("Experimental Red-Region Debug Diagnostics (NOT Blood Probability)", expanded=False):
            st.dataframe(pd.DataFrame(incidents["candidates"]), use_container_width=True, hide_index=True)
            st.caption(incidents.get("limitations", ""))

    # PPE Detections Table
    rows = []
    verified_helmet_indices = set()
    for person in violations.get("person_results", []):
        if person.get("has_helmet"):
            verified_helmet_indices.update(
                item.get("item_index") for item in person.get("associations", {}).get("helmet", [])
            )
    for class_name in CLASS_ORDER:
        for item_index, item in enumerate(detections.get(class_name, [])):
            display_class = class_name
            if class_name == "helmet" and item_index not in verified_helmet_indices:
                display_class = "UNVERIFIED HELMET CANDIDATE"
            rows.append({"class": display_class, "confidence": item.get("confidence", item.get("conf", 0.0)), "bbox": item["bbox"]})
    if rows:
        table = pd.DataFrame(rows)
        if show_confidence:
            table["confidence"] = table["confidence"].map(lambda value: f"{value:.1%}")
        st.dataframe(table, use_container_width=True, hide_index=True)
    else:
        st.info("No filtered detections for this image.")

    if violations.get("unverified_candidates"):
        ui_section("Unverified PPE candidates")
        st.dataframe(pd.DataFrame(violations["unverified_candidates"]), use_container_width=True, hide_index=True)

    review_rows = [
        {
            "person_id": item.get("person_id"),
            "review_reasons": ", ".join(item.get("review_reasons", [])),
            "uncertain": ", ".join(item.get("uncertain", [])),
        }
        for item in violations.get("person_results", [])
        if item.get("needs_review")
    ]
    if review_rows:
        ui_section("Review queue")
        st.dataframe(pd.DataFrame(review_rows), use_container_width=True, hide_index=True)

    if violations.get("violations"):
        ui_section("Confirmed violation details")
        st.dataframe(pd.DataFrame(violations["violations"]), use_container_width=True, hide_index=True)

    # Collapsible Diagnostics Section (Phase 1 Requirement)
    with st.expander("Diagnostics & Raw Model Predictions", expanded=False):
        st.write("**Head Region Verifications:**")
        st.json(result.get("head_verifications", []))
        st.write("**Raw Detector Diagnostics:**")
        st.json(result.get("diagnostics", {}))


def image_page(processor: FrameProcessor) -> None:
    ui_title("Image analysis", "Evidence workspace")
    uploaded = st.file_uploader("Select a JPG or PNG", type=["jpg", "jpeg", "png"], key="single_image")
    if uploaded is None:
        st.markdown('<div class="notice">Select an image to begin. Files are processed locally.</div>', unsafe_allow_html=True)
        return
    original = Image.open(uploaded).convert("RGB")
    if st.button("Analyze Image", type="primary"):
        with st.spinner("Running the existing inference pipeline..."):
            result = process_uploaded_image(processor, uploaded.getvalue(), uploaded.name)
        st.session_state["image_result"] = result
    col1, col2 = st.columns(2)
    with col1:
        st.image(original, caption="Original evidence", use_column_width=True)
    with col2:
        result = st.session_state.get("image_result")
        if not result:
            st.markdown('<div class="notice">Analyze the image to view the annotated evidence.</div>', unsafe_allow_html=True)
            return
        annotated = cv2.cvtColor(result["annotated_frame"], cv2.COLOR_BGR2RGB)
        st.image(annotated, caption="Annotated evidence", use_column_width=True)
        encoded = cv2.imencode(".jpg", result["annotated_frame"])[1].tobytes()
        btn_cols = st.columns(2)
        btn_cols[0].download_button("Download Annotated Image", encoded, "ppe_result.jpg", "image/jpeg")
        report_path = result.get("report_path")
        if report_path and Path(report_path).is_file():
            pdf_bytes = Path(report_path).read_bytes()
            btn_cols[1].download_button("Download Incident Report", pdf_bytes, Path(report_path).name, "application/pdf")
    if st.session_state.get("image_result"):
        render_result(st.session_state["image_result"])


def batch_page(processor: FrameProcessor) -> None:
    ui_title("Batch testing", "Multi-image review")
    uploads = st.file_uploader("Select JPG/PNG images", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="batch_images")
    if not uploads:
        st.markdown('<div class="notice">Select multiple images to build a local results table.</div>', unsafe_allow_html=True)
        return
    if st.button("Analyze Selected Images", type="primary"):
        rows = []
        for uploaded in uploads:
            image = cv2.imdecode(np.frombuffer(uploaded.getvalue(), np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                rows.append({"image": uploaded.name, "status": "unreadable"})
                continue
            result = process_uploaded_image(processor, uploaded.getvalue(), uploaded.name)
            rows.append({"image": uploaded.name, "status": result["violations"]["status"], "persons": result["violations"]["person_count"],
                         "confirmed_violations": result["violations"]["violation_count"], "unverified_candidates": len(result["violations"].get("unverified_candidates", [])),
                         "compliance_rate": result["violations"]["compliance_rate"]})
        st.session_state["batch_rows"] = rows
    rows = st.session_state.get("batch_rows")
    if rows:
        dataframe = pd.DataFrame(rows)
        st.dataframe(dataframe, use_container_width=True, hide_index=True)
        st.download_button("Download Batch Results CSV", dataframe.to_csv(index=False), "ppe_batch_results.csv", "text/csv")


def overview_page(processor: FrameProcessor) -> None:
    ui_title("Operations overview", "PPE Safety Monitor")
    history = read_history(processor)
    if not history:
        st.markdown('<div class="notice">No saved analysis history. Use Image Analysis or Batch Testing to create the first record.</div>', unsafe_allow_html=True)
        return
    total = sum(int(item.get("violation_count", 0)) for item in history)
    runs_with_violations = sum(item.get("violation_count", 0) > 0 for item in history)
    st.markdown(
        '<div class="metric-strip">'
        f'<div><div class="metric-label">Processed runs</div><div class="metric-value">{len(history)}</div></div>'
        f'<div><div class="metric-label">Confirmed violations</div><div class="metric-value">{total}</div></div>'
        f'<div><div class="metric-label">Runs with violations</div><div class="metric-value">{runs_with_violations}</div></div>'
        f'<div><div class="metric-label">Average violations</div><div class="metric-value">{total / len(history):.2f}</div></div>'
        '</div>', unsafe_allow_html=True,
    )
    table = pd.DataFrame(history)
    ui_section("Saved analysis history")
    st.dataframe(table, use_container_width=True, hide_index=True)


def evaluation_page() -> None:
    ui_title("Evaluation", "Detection performance")
    if not EVALUATION_PATH.exists():
        st.markdown('<div class="notice">No evaluation report exists yet. Run the dataset evaluation command below.</div>', unsafe_allow_html=True)
        st.code("python scripts/evaluate_dataset.py", language="powershell")
        return
    report = json.loads(EVALUATION_PATH.read_text(encoding="utf-8"))
    st.caption(f'{report.get("image_count", 0)} labelled test images from {report.get("dataset", "")}.')
    for key, title in (("raw_ppe_model", "Raw PPE-model metrics"), ("filtered_pipeline_ppe", "Filtered pipeline PPE metrics")):
        group = report.get("metrics", {}).get(key)
        if not group:
            continue
        ui_section(title)
        rows = []
        for class_name, metrics in group.get("per_class", {}).items():
            rows.append({"class": class_name, "ground_truth": metrics.get("ground_truth", 0), "predictions": metrics.get("predictions", 0),
                         "precision": metrics.get("precision", 0.0), "recall": metrics.get("recall", 0.0), "f1": metrics.get("f1", 0.0),
                         "AP50": metrics.get("ap50", 0.0), "mAP50-95": metrics.get("map50_95", 0.0)})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.metric("mAP50-95", f'{group.get("map50_95", 0.0):.3f}')
    ui_section("End-to-end pipeline summary")
    st.json(report.get("end_to_end_pipeline", {}))
    confusion = report.get("confusion_matrix", {})
    if confusion.get("labels") and confusion.get("values"):
        ui_section("Raw PPE-model confusion matrix at IoU 0.50")
        st.dataframe(pd.DataFrame(confusion["values"], index=confusion["labels"], columns=confusion["labels"]), use_container_width=True)
    st.download_button("Download Evaluation Report", EVALUATION_PATH.read_bytes(), EVALUATION_PATH.name, "application/json")


def how_it_works_page(processor: FrameProcessor) -> None:
    ui_title("Detection method", "How it works")
    st.markdown("""
The application runs two YOLO models on the same frame: the trained PPE model
detects helmet, vest, without_helmet, and without_vest; the YOLOv8 person model
provides person boxes for association. `FrameProcessor` preprocesses the frame,
calls `PPEDetector`, then evaluates person-level compliance through
`ViolationDetector`.

PPE detections are filtered by confidence, class-specific size rules, and
helmet-specific checks. PPE is associated with a person by center containment
or IoU. A PPE-only candidate without a person is shown as unverified and is
not counted as a confirmed violation. Video helmet detections may also require
temporal confirmation.
""")
    model = processor.config["model"]
    st.json({"weights": model.get("weights_path"), "person_weights": model.get("person_model_path"),
             "confidence_threshold": model.get("confidence_threshold"), "person_confidence_threshold": model.get("person_confidence_threshold"),
             "iou_threshold": model.get("iou_threshold"), "class_confidence_thresholds": model.get("class_confidence_thresholds"),
             "person_max_box_area_ratio": model.get("person_max_box_area_ratio"), "class_max_box_area_ratios": model.get("class_max_box_area_ratios", {}),
             "association": processor.config.get("violation", {})})


def main() -> None:
    st.markdown('<div class="hero"><h1>PPE Safety Monitor</h1><p>Local evidence review for personal protective equipment compliance.</p></div>', unsafe_allow_html=True)
    try:
        processor = load_processor(CONFIG_PATH.stat().st_mtime_ns)
    except Exception as exc:
        st.error(f"Could not load the local detector: {exc}")
        st.stop()
    with st.sidebar:
        st.markdown('<p class="kicker" style="color:#F2C744">Operations</p>', unsafe_allow_html=True)
        page = st.radio("Workspace", ["Overview", "Image Analysis", "Batch Testing", "Live Camera", "Evaluation", "How It Works"])
        processor.save_violations = st.checkbox("Save violation images", value=processor.config["storage"].get("save_violations", True))
        st.caption(f"PPE model: {Path(processor.detector.model_path).name}")
    if page == "Overview":
        overview_page(processor)
    elif page == "Image Analysis":
        image_page(processor)
    elif page == "Batch Testing":
        batch_page(processor)
    elif page == "Live Camera":
        from app.pages.live_camera import render as render_camera
        render_camera(processor)
    elif page == "Evaluation":
        evaluation_page()
    else:
        how_it_works_page(processor)


if __name__ == "__main__":
    main()
