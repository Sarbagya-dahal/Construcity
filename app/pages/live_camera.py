"""Live Camera page for the PPE dashboard."""
from __future__ import annotations

import sys
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.camera_worker import CameraWorker


SESSION_BASE = PROJECT_ROOT / "outputs" / "camera_sessions"


def _make_session_dir() -> Path:
    session_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    session_dir = SESSION_BASE / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def _stop_camera() -> None:
    worker: CameraWorker | None = st.session_state.get("camera_worker")
    if worker is not None:
        worker.stop()
    st.session_state["camera_worker"] = None
    st.session_state["camera_running"] = False


def render(processor) -> None:
    st.header("Live Camera Monitor")
    st.info(
        "Frames are captured from the local webcam at the configured interval and "
        "run through the PPE inference pipeline. All recordings stay local."
    )

    # ── Controls ──────────────────────────────────────────────────────
    col_ctrl, col_cfg = st.columns([1, 2])
    with col_cfg:
        camera_id = st.number_input("Camera device index", min_value=0, max_value=9, value=0, step=1)
        interval = st.slider("Inference interval (seconds)", min_value=1, max_value=30, value=5)

    running = st.session_state.get("camera_running", False)
    with col_ctrl:
        if not running:
            if st.button("▶ Start Camera", type="primary", use_container_width=True):
                session_dir = _make_session_dir()
                worker = CameraWorker(processor, session_dir, interval_seconds=interval)
                worker.start(camera_id=int(camera_id))
                err = worker.get_error()
                if err:
                    st.error(f"Camera error: {err}")
                else:
                    st.session_state["camera_worker"] = worker
                    st.session_state["camera_running"] = True
                    st.session_state["camera_session_dir"] = session_dir
                    st.rerun()
        else:
            if st.button("⏹ Stop Camera", type="secondary", use_container_width=True):
                _stop_camera()
                st.rerun()

    # ── Live Feed ─────────────────────────────────────────────────────
    worker: CameraWorker | None = st.session_state.get("camera_worker")
    if worker is None or not st.session_state.get("camera_running", False):
        st.info("Camera is stopped. Press **▶ Start Camera** to begin.")
        return

    err = worker.get_error()
    if err:
        st.error(f"Camera error: {err}")
        _stop_camera()
        return

    # Two-column live display
    preview_col, result_col = st.columns(2)
    preview_placeholder = preview_col.empty()
    result_placeholder = result_col.empty()

    # Session summary metrics
    metric_cols = st.columns(3)
    snap_placeholder = metric_cols[0].empty()
    violation_placeholder = metric_cols[1].empty()
    status_placeholder = metric_cols[2].empty()

    # Event log
    log_placeholder = st.empty()

    # Refresh loop — Streamlit 1.28 has st.rerun() but not st_autorefresh
    # We do a brief live update loop then rerun
    for _ in range(10):
        frame = worker.get_preview_frame()
        result = worker.get_latest_result()

        if frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            preview_placeholder.image(rgb, caption="Live preview", use_column_width=True)

        if result is not None:
            annotated = result.get("annotated_frame")
            if annotated is not None:
                ann_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                result_placeholder.image(ann_rgb, caption="Latest analysis", use_column_width=True)
            violations = result.get("violations", {})
            snap_placeholder.metric("Snapshots taken", worker.get_snapshot_count())
            violation_placeholder.metric("Last violations", violations.get("violation_count", 0))
            status_placeholder.metric("Status", violations.get("status", "—"))

        events = worker.get_event_log()
        if events:
            log_placeholder.dataframe(
                [e for e in reversed(events[-20:])],
                use_container_width=True,
                hide_index=True,
            )

        if not worker.is_running():
            st.warning("Camera stopped unexpectedly.")
            _stop_camera()
            break

        time.sleep(0.3)

    st.rerun()
