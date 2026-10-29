"""Live Camera page with persistent, downloadable camera sessions."""
from __future__ import annotations

import json
import sys
import time
import uuid
from pathlib import Path

import cv2
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.camera_worker import CameraWorker


SESSION_BASE = PROJECT_ROOT / "outputs" / "camera_sessions"


def _make_session_dir() -> Path:
    SESSION_BASE.mkdir(parents=True, exist_ok=True)
    session_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    session_dir = SESSION_BASE / session_id
    session_dir.mkdir(parents=True, exist_ok=False)
    return session_dir


def _stop_camera() -> dict | None:
    worker: CameraWorker | None = st.session_state.get("camera_worker")
    summary = None
    if worker is not None:
        worker.stop()
        summary = worker.get_session_summary()
        st.session_state["camera_session_summary"] = summary
        st.session_state["camera_session_dir"] = worker.session_dir
    st.session_state["camera_worker"] = None
    st.session_state["camera_running"] = False
    return summary


def _completed_sessions() -> list[dict]:
    sessions = []
    if not SESSION_BASE.exists():
        return sessions
    for session_dir in sorted(SESSION_BASE.iterdir(), reverse=True):
        manifest = session_dir / "session.json"
        if not session_dir.is_dir() or not manifest.exists():
            continue
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            data["session_dir"] = str(session_dir)
            sessions.append(data)
        except (OSError, json.JSONDecodeError):
            continue
    return sessions


def _download_file(label: str, path: Path, mime: str, key: str) -> None:
    if path.is_file():
        st.download_button(label, path.read_bytes(), path.name, mime, key=key)
    else:
        st.info(f"{label} is not available: {path}")


def _render_session_summary(summary: dict | None) -> None:
    if not summary:
        return
    st.subheader("Completed camera session")
    st.code(summary.get("session_dir", ""), language="text")
    cols = st.columns(6)
    cols[0].metric("Captured", summary.get("captured_frames", 0))
    cols[1].metric("Analyzed", summary.get("successful_frames", 0))
    cols[2].metric("Failed", summary.get("failed_frames", 0))
    cols[3].metric("Violations", summary.get("violations", 0))
    cols[4].metric("Uncertain", summary.get("uncertain_findings", 0))
    cols[5].metric("Status", "Complete" if summary.get("ended_at") else "Partial")
    session_dir = Path(summary["session_dir"])
    a, b = st.columns(2)
    with a:
        _download_file("Download Session PDF", session_dir / "reports" / "session_report.pdf", "application/pdf", "last-session-pdf")
    with b:
        _download_file("Download All Session Files ZIP", session_dir / "session_bundle.zip", "application/zip", "last-session-zip")
    if summary.get("report_error"):
        st.warning(f"PDF generation failed; image evidence remains on disk: {summary['report_error']}")


def _render_history() -> None:
    sessions = _completed_sessions()
    if not sessions:
        st.info("No completed camera sessions are available yet.")
        return
    st.subheader("Session history")
    for item in sessions[:20]:
        session_dir = Path(item["session_dir"])
        with st.expander(f"{item.get('session_id', session_dir.name)} — {item.get('started_at', 'unknown start')}"):
            st.code(str(session_dir), language="text")
            st.write({
                "captured_frames": item.get("event_count", item.get("captured_frames", 0)),
                "successful_frames": item.get("successful_frames", 0),
                "failed_frames": item.get("failed_frames", 0),
                "confirmed_violations": item.get("confirmed_violations", item.get("violations", 0)),
                "ended_at": item.get("ended_at"),
            })
            a, b = st.columns(2)
            with a:
                _download_file("Download PDF", session_dir / "reports" / "session_report.pdf", "application/pdf", f"pdf-{session_dir.name}")
            with b:
                _download_file("Download ZIP", session_dir / "session_bundle.zip", "application/zip", f"zip-{session_dir.name}")


def render(processor) -> None:
    st.header("Live Camera Monitor")
    st.info("The camera runs locally. Each sampled frame is saved with its original, annotated result, findings, and timestamps.")

    col_ctrl, col_cfg = st.columns([1, 2])
    with col_cfg:
        camera_id = st.number_input("Camera device index", min_value=0, max_value=9, value=0, step=1)
        interval = st.slider("Capture and inference interval (seconds)", min_value=1, max_value=30, value=5)
        record_video = st.checkbox("Record continuous video", value=False)
    running = st.session_state.get("camera_running", False)
    with col_ctrl:
        if not running:
            if st.button("Start Camera", type="primary", use_container_width=True):
                session_dir = _make_session_dir()
                worker = CameraWorker(processor, session_dir, interval_seconds=interval, record_video=record_video)
                worker.start(camera_id=int(camera_id))
                st.session_state["camera_worker"] = worker
                st.session_state["camera_session_dir"] = session_dir
                if worker.get_error():
                    st.session_state["camera_session_summary"] = worker.get_session_summary()
                    st.error(f"Camera error: {worker.get_error()}")
                else:
                    st.session_state["camera_running"] = True
                    st.rerun()
        else:
            if st.button("Stop Camera", type="secondary", use_container_width=True):
                _stop_camera()
                st.rerun()

    worker: CameraWorker | None = st.session_state.get("camera_worker")
    if worker is None or not st.session_state.get("camera_running", False):
        _render_session_summary(st.session_state.get("camera_session_summary"))
        _render_history()
        return

    st.caption(f"Session: {worker.session.session_id} | Storage: {worker.session_dir}")
    err = worker.get_error()
    if err:
        st.error(f"Camera error: {err}")
        _stop_camera()
        st.rerun()
        return

    preview_col, result_col = st.columns(2)
    preview_placeholder = preview_col.empty()
    result_placeholder = result_col.empty()
    metric_cols = st.columns(4)
    snap_placeholder = metric_cols[0].empty()
    violation_placeholder = metric_cols[1].empty()
    status_placeholder = metric_cols[2].empty()
    failure_placeholder = metric_cols[3].empty()
    log_placeholder = st.empty()

    for _ in range(10):
        frame = worker.get_preview_frame()
        result = worker.get_latest_result()
        if frame is not None:
            preview_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Live preview", use_column_width=True)
        if result is not None:
            annotated = result.get("annotated_frame")
            if annotated is not None:
                result_placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Latest analysis", use_column_width=True)
            violations = result.get("violations", {})
            snap_placeholder.metric("Frames captured", worker.get_snapshot_count())
            violation_placeholder.metric("Last violations", violations.get("violation_count", 0))
            status_placeholder.metric("Status", violations.get("status", "—"))
            failure_placeholder.metric("Failed frames", sum(e.get("processing_status") == "failed" for e in worker.get_event_log()))
        events = worker.get_event_log()
        if events:
            log_placeholder.dataframe([e for e in reversed(events[-20:])], use_container_width=True, hide_index=True)
        if not worker.is_running():
            st.warning("Camera stopped unexpectedly. The session is being finalized.")
            _stop_camera()
            break
        time.sleep(0.3)
    st.rerun()
