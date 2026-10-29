"""Background camera worker for the Live Camera dashboard page."""
from __future__ import annotations

import json
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional


class CameraWorker:
    """Run OpenCV camera capture in a background thread.

    The worker samples frames at *interval_seconds* and feeds each through the
    FrameProcessor pipeline.  Results and preview frames are stored
    thread-safely so the Streamlit page can read them without blocking.
    """

    def __init__(self, processor, session_dir: Path, interval_seconds: float = 5.0):
        self._processor = processor
        self._session_dir = Path(session_dir)
        self._interval = max(0.5, float(interval_seconds))
        self._cap = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._running = False
        self._latest_frame = None        # numpy array (BGR)
        self._latest_result: Optional[dict] = None
        self._error: Optional[str] = None
        self._event_log: list[dict] = []
        self._snapshot_count = 0

        # Create session subdirectories
        for sub in ("snapshots", "annotated", "reports"):
            (self._session_dir / sub).mkdir(parents=True, exist_ok=True)

        # Write session manifest
        manifest = {
            "session_id": self._session_dir.name,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "interval_seconds": self._interval,
        }
        (self._session_dir / "session.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

    # ------------------------------------------------------------------
    # Public interface (thread-safe)
    # ------------------------------------------------------------------

    def start(self, camera_id: int = 0) -> None:
        """Open the camera and start the background inference thread."""
        with self._lock:
            if self._running:
                return
            import cv2
            cap = cv2.VideoCapture(camera_id)
            if not cap.isOpened():
                self._error = f"Cannot open camera {camera_id}"
                return
            self._cap = cap
            self._running = True
            self._error = None
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the worker thread to stop and release the camera."""
        with self._lock:
            self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10)
        with self._lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None

    def is_running(self) -> bool:
        with self._lock:
            return self._running

    def get_error(self) -> Optional[str]:
        with self._lock:
            return self._error

    def get_preview_frame(self):
        """Return the latest raw camera frame (numpy BGR array) or None."""
        with self._lock:
            return self._latest_frame

    def get_latest_result(self) -> Optional[dict]:
        """Return the latest inference result dict or None."""
        with self._lock:
            return self._latest_result

    def get_event_log(self) -> list[dict]:
        with self._lock:
            return list(self._event_log)

    def get_snapshot_count(self) -> int:
        with self._lock:
            return self._snapshot_count

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        import cv2
        last_inference_time = 0.0
        while True:
            with self._lock:
                if not self._running or self._cap is None:
                    break
                cap = self._cap
            ret, frame = cap.read()
            if not ret:
                with self._lock:
                    self._error = "Failed to read frame from camera"
                    self._running = False
                break
            with self._lock:
                self._latest_frame = frame.copy()

            now = time.time()
            if now - last_inference_time >= self._interval:
                last_inference_time = now
                self._run_inference(frame)

            time.sleep(0.03)  # ~30 fps preview sampling

    def _run_inference(self, frame) -> None:
        import cv2
        ts = datetime.now()
        ts_str = ts.strftime("%Y%m%d_%H%M%S")
        try:
            result = self._processor.process_stream_frame(frame)
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self._event_log.append({"timestamp": ts.isoformat(), "type": "error", "message": str(exc)})
            return

        with self._lock:
            self._latest_result = result
            self._snapshot_count += 1
            count = self._snapshot_count

        # Save snapshot
        snapshot_path = self._session_dir / "snapshots" / f"snapshot_{ts_str}_{count:04d}.jpg"
        cv2.imwrite(str(snapshot_path), frame)

        # Save annotated frame
        annotated_frame = result.get("annotated_frame")
        if annotated_frame is not None:
            annotated_path = self._session_dir / "annotated" / f"annotated_{ts_str}_{count:04d}.jpg"
            cv2.imwrite(str(annotated_path), annotated_frame)

        # Log event
        violations = result.get("violations", {})
        event = {
            "timestamp": ts.isoformat(),
            "type": "inference",
            "snapshot": snapshot_path.name,
            "violation_count": violations.get("violation_count", 0),
            "compliance_rate": violations.get("compliance_rate", 0.0),
            "status": violations.get("status", "unknown"),
        }
        with self._lock:
            self._event_log.append(event)

        # Persist event log
        log_path = self._session_dir / "events.json"
        with self._lock:
            log_snapshot = list(self._event_log)
        try:
            log_path.write_text(json.dumps(log_snapshot, indent=2), encoding="utf-8")
        except OSError:
            pass
