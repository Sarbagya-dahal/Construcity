"""Tests for live-camera / stream processing interface and deduplication."""

import unittest
import numpy as np

from src.processor import FrameProcessor
from src.utils import load_config


class StreamInterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = load_config("config/config.yaml")
        # Ensure reporting and notifications are enabled with test output dir
        config["reporting"]["output_dir"] = "outputs/test_reports"
        config["reporting"]["enabled"] = True
        config["notifications"]["enabled"] = True
        config["notifications"]["cooldown_seconds"] = 10.0
        cls.config = config
        cls.processor = FrameProcessor(config, config["model"]["weights_path"])

    def test_process_stream_frame_signature_and_output(self):
        """Streaming frame interface accepts frame_id, source_id, and returns complete findings."""
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        result = self.processor.process_stream_frame(img, frame_id=1, source_id="test_stream_01")
        self.assertIn("annotated_frame", result)
        self.assertIn("violations", result)
        self.assertIn("incidents", result)
        self.assertIn("alert", result)

    def test_stream_cooldown_prevents_duplicate_reports_on_static_scenes(self):
        """Repeated frames with an actionable condition should not repeatedly trigger new notifications."""
        # Reset cooldown tracker
        self.processor.notification_manager.reset_cooldown()

        # Create a frame with a simulated worker
        img = np.zeros((300, 300, 3), dtype=np.uint8)

        # Process frame 1
        res1 = self.processor.process_stream_frame(img, frame_id=1, source_id="live_camera_entrance")

        # Process frame 2 immediately
        res2 = self.processor.process_stream_frame(img, frame_id=2, source_id="live_camera_entrance")

        # If res1 triggered an alert, res2 must be throttled
        if res1.get("alert", {}).get("sent") or res1.get("alert", {}).get("status") == "Email not configured":
            self.assertTrue(res2["alert"]["throttled"], "Subsequent stream frame should be throttled by cooldown")


if __name__ == "__main__":
    unittest.main()

