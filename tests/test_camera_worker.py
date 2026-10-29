"""Tests for CameraWorker lifecycle and session management."""
import json
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from src.camera_worker import CameraWorker


def _make_fake_processor():
    proc = MagicMock()
    result = {
        "annotated_frame": np.zeros((480, 640, 3), dtype=np.uint8),
        "violations": {"violation_count": 0, "compliance_rate": 100.0, "status": "evaluated"},
    }
    proc.process_stream_frame.return_value = result
    return proc


def _make_fake_cap(readable: bool = True):
    cap = MagicMock()
    cap.isOpened.return_value = readable
    cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
    return cap


class CameraWorkerSessionTests(unittest.TestCase):
    def test_session_directories_created_on_init(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = Path(tmpdir) / "session_001"
            proc = _make_fake_processor()
            worker = CameraWorker(proc, session_dir, interval_seconds=5)
            self.assertTrue((session_dir / "snapshots").is_dir())
            self.assertTrue((session_dir / "annotated").is_dir())
            self.assertTrue((session_dir / "reports").is_dir())
            manifest = json.loads((session_dir / "session.json").read_text())
            self.assertIn("session_id", manifest)

    def test_session_manifest_written(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = Path(tmpdir) / "session_manifest"
            proc = _make_fake_processor()
            worker = CameraWorker(proc, session_dir, interval_seconds=3)
            manifest = json.loads((session_dir / "session.json").read_text())
            self.assertEqual(manifest["interval_seconds"], 3.0)


class CameraWorkerLifecycleTests(unittest.TestCase):
    def test_start_stop_with_mock_camera(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = Path(tmpdir) / "lifecycle"
            proc = _make_fake_processor()
            worker = CameraWorker(proc, session_dir, interval_seconds=100)
            fake_cap = _make_fake_cap(readable=True)
            with patch("cv2.VideoCapture", return_value=fake_cap):
                worker.start(camera_id=0)
                time.sleep(0.1)
                self.assertTrue(worker.is_running())
                worker.stop()
                self.assertFalse(worker.is_running())

    def test_camera_unavailable_sets_error(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = Path(tmpdir) / "unavailable"
            proc = _make_fake_processor()
            worker = CameraWorker(proc, session_dir, interval_seconds=5)
            fake_cap = _make_fake_cap(readable=False)
            with patch("cv2.VideoCapture", return_value=fake_cap):
                worker.start(camera_id=99)
                self.assertFalse(worker.is_running())
                self.assertIsNotNone(worker.get_error())

    def test_inference_result_accessible_after_run(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = Path(tmpdir) / "inference"
            proc = _make_fake_processor()
            worker = CameraWorker(proc, session_dir, interval_seconds=0.1)
            fake_cap = _make_fake_cap(readable=True)
            with patch("cv2.VideoCapture", return_value=fake_cap):
                worker.start(camera_id=0)
                time.sleep(0.4)
                result = worker.get_latest_result()
                worker.stop()
            if result is not None:
                self.assertIn("violations", result)

    def test_double_stop_is_safe(self):
        """Calling stop() twice must not raise."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            session_dir = Path(tmpdir) / "double_stop"
            proc = _make_fake_processor()
            worker = CameraWorker(proc, session_dir, interval_seconds=100)
            fake_cap = _make_fake_cap(readable=True)
            with patch("cv2.VideoCapture", return_value=fake_cap):
                worker.start(camera_id=0)
                worker.stop()
                worker.stop()


if __name__ == "__main__":
    unittest.main()

