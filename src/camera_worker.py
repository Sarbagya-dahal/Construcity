"""Background camera capture with durable session evidence."""

from __future__ import annotations

import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.camera_session import CameraSession


class CameraWorker:
    """Capture preview frames and analyze one frame at the configured interval."""

    def __init__(self, processor, session_dir: Path, interval_seconds: float = 5.0, record_video: bool = False):
        self._processor = processor
        self._interval = max(0.5, float(interval_seconds))
        config_snapshot = processor.config if hasattr(processor, "config") else {}
        self.session = CameraSession(session_dir, self._interval, config_snapshot, record_video)
        self._cap = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._running = False
        self._finalized = False
        self._latest_frame = None
        self._latest_result: Optional[dict] = None
        self._error: Optional[str] = None
        self._frame_number = 0

    @property
    def session_dir(self) -> Path:
        return self.session.session_dir

    def start(self, camera_id: int = 0) -> None:
        with self._lock:
            if self._running:
                return
            import cv2
            cap = cv2.VideoCapture(camera_id)
            if not cap.isOpened():
                self._error = f"Cannot open camera {camera_id}"
                self._finalize_once()
                return
            self._cap = cap
            self._running = True
            self._error = None
        self._thread = threading.Thread(target=self._run_loop, name="ppe-camera-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        with self._lock:
            self._running = False
        if self._thread and self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join(timeout=10)
        with self._lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None
        self._finalize_once()

    def _finalize_once(self) -> None:
        if self._finalized:
            return
        self._finalized = True
        try:
            self.session.finalize()
        except Exception as exc:
            with self._lock:
                self._error = self._error or f"Session finalization error: {exc}"

    def is_running(self) -> bool:
        with self._lock:
            return self._running

    def get_error(self) -> Optional[str]:
        with self._lock:
            return self._error

    def get_preview_frame(self):
        with self._lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    def get_latest_result(self) -> Optional[dict]:
        with self._lock:
            return self._latest_result

    def get_event_log(self) -> list[dict]:
        return list(self.session.events)

    def get_snapshot_count(self) -> int:
        return self.session.event_count

    def get_session_summary(self) -> dict:
        return self.session.summary()

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
            try:
                self.session.record_video_frame(frame)
            except Exception as exc:
                with self._lock:
                    self._error = self._error or f"Video recording error: {exc}"
            now = time.monotonic()
            if now - last_inference_time >= self._interval:
                last_inference_time = now
                self._run_inference(frame)
            time.sleep(0.03)

    def _run_inference(self, frame) -> None:
        captured_at = datetime.now().astimezone()
        self._frame_number += 1
        frame_number = self._frame_number
        try:
            result = self._processor.process_stream_frame(
                frame,
                frame_id=frame_number,
                source_id=f"camera_{self.session.session_id}",
                capture_timestamp=captured_at,
            )
            analyzed_at = datetime.now().astimezone()
            self.session.record_frame(frame_number, frame_number, frame, result, captured_at, analyzed_at)
            with self._lock:
                self._latest_result = result
        except Exception as exc:
            self.session.record_failure(frame_number, frame_number, frame, captured_at, str(exc))
            with self._lock:
                self._error = f"Frame {frame_number} failed: {exc}"
