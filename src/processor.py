"""Frame processor for the PPE detection pipeline."""

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

from src.detector import PPEDetector
from src.head_heuristic import HeadHeuristic
from src.head_localization import HeadRegionLocalizer
from src.incident_detector import IncidentDetector
from src.notifications import NotificationManager
from src.report_generator import PDFReportGenerator, should_generate_report
from src.utils import draw_bbox, ensure_dir, format_confidence, resize_with_aspect_ratio, save_violation_image
from src.violation_logic import ViolationDetector, annotate_violations


class FrameProcessor:
    """Main processing pipeline for PPE detection."""

    def __init__(self, config: dict, model_path: str):
        self.config = config

        print("Initializing PPE Detector...")
        self.detector = PPEDetector(model_path, config)

        print("Initializing Violation Detector...")
        self.violation_detector = ViolationDetector(config)

        # Head localizer and heuristic always instantiated (no checkpoint required).
        head_clf_config = config.get("head_classifier", {})
        self._localizer = HeadRegionLocalizer(
            scale_factor=float(head_clf_config.get("face_scale_factor", 1.1)),
            min_neighbors=int(head_clf_config.get("face_min_neighbors", 5)),
            use_top_region_fallback=bool(head_clf_config.get("use_top_region_fallback", True)),
            top_region_ratio=float(head_clf_config.get("top_region_ratio", 0.28)),
        )
        self._head_heuristic = HeadHeuristic(config)

        self.head_classifier = None
        if head_clf_config.get("enabled", False):
            from src.head_classifier import HeadHelmetClassifier

            checkpoint_path = head_clf_config.get("checkpoint_path")
            if not checkpoint_path or not Path(checkpoint_path).exists():
                raise FileNotFoundError(f"Head classifier checkpoint not found: {checkpoint_path}")
            self.head_classifier = HeadHelmetClassifier(checkpoint_path, config)

        self.incident_detector = IncidentDetector(config)
        self.report_generator = PDFReportGenerator(config)
        self.notification_manager = NotificationManager(config)

        self.violations_dir = config["storage"]["violations_dir"]
        self.save_violations = config["storage"]["save_violations"]

        ensure_dir(self.violations_dir)
        print("Frame Processor initialized")

    @staticmethod
    def preprocess_frame(frame: np.ndarray) -> np.ndarray:
        """Improve low-light frames before running detection."""
        return cv2.convertScaleAbs(frame, alpha=1.3, beta=40)

    def process_frame(
        self,
        frame: np.ndarray,
        save_violation: bool = True,
        source_name: str = "unknown",
        enable_temporal: bool = False,
        timestamp: Optional[datetime] = None,
        capture_timestamp: Optional[datetime] = None,
    ) -> Dict:
        """Process a single frame through preprocessing, detection, and violation logic."""
        # Capture exact timezone-aware processing start timestamp
        if timestamp is None:
            scan_time = datetime.now().astimezone()
        elif isinstance(timestamp, str):
            try:
                scan_time = datetime.fromisoformat(timestamp)
                if scan_time.tzinfo is None:
                    scan_time = scan_time.astimezone()
            except Exception:
                scan_time = datetime.now().astimezone()
        elif isinstance(timestamp, datetime):
            scan_time = timestamp if timestamp.tzinfo is not None else timestamp.astimezone()
        else:
            scan_time = datetime.now().astimezone()

        enhanced_frame = self.preprocess_frame(frame)
        detections = self.detector.detect(enhanced_frame, enable_temporal=enable_temporal)
        head_verifications = None
        if self.head_classifier is not None:
            # Full neural classifier path (requires reviewed dataset + trained checkpoint).
            head_verifications = self.head_classifier.verify_persons(enhanced_frame, detections.get("person", []))
        else:
            # Geometry-only heuristic path — always active when classifier is disabled.
            # The heuristic confirms NO-HELMET (triggering violations) and also confirms
            # HELMET-WORN only for detections that already meet the worn-confidence gate.
            # Low-confidence helmet detections at the head region escalate to needs_review
            # rather than to confirmed compliance — geometry cannot authenticate the object.
            persons = detections.get("person", [])
            helmets = detections.get("helmet", [])
            worn_conf_threshold = float(
                self.config.get("violation", {}).get("helmet_worn_confidence_threshold", 0.80)
            )
            head_verifications = []
            for person in persons:
                head_crop = self._localizer.locate(enhanced_frame, person["bbox"])
                # Find helmet detections whose center is in the top head region of this person.
                px1, py1, px2, py2 = person["bbox"]
                head_bottom = py1 + (py2 - py1) * 0.40  # top 40% of person box
                person_helmets_any = [
                    h for h in helmets
                    if px1 <= (h["bbox"][0] + h["bbox"][2]) / 2 <= px2
                    and py1 <= (h["bbox"][1] + h["bbox"][3]) / 2 <= head_bottom
                ]
                # Only pass high-confidence candidates to the worn-helmet test.
                person_helmets_worn = [
                    h for h in person_helmets_any
                    if h.get("confidence", h.get("conf", 0.0)) >= worn_conf_threshold
                ]
                verdict = self._head_heuristic.evaluate(head_crop, person_helmets_worn)
                # If geometry says no_helmet but low-conf detections exist at the head
                # region, escalate to needs_review instead of reporting no_helmet
                # (low-conf detections are ambiguous; only classifier can resolve them).
                if verdict["status"] == "no_helmet" and person_helmets_any and not person_helmets_worn:
                    verdict = {
                        "status": "needs_review",
                        "reason": "head_region_has_low_confidence_helmet_candidate",
                        "confidence": 0.0,
                    }
                head_verifications.append(verdict)
        violation_result = self.violation_detector.check_violations(
            detections, head_verifications=head_verifications
        )
        annotated_frame = annotate_violations(
            frame,
            detections,
            violation_result,
            self.config,
            diagnostics=self.detector.last_diagnostics,
        )

        if save_violation and self.save_violations and violation_result["violation_count"] > 0:
            self._save_violation_image(annotated_frame, violation_result, source_name)

        # Incident screening (disabled in production; reliable detector unavailable)
        incidents = self.incident_detector.detect_incidents(frame, detections.get("person", []))
        if self.config.get("incident_detection", {}).get("draw_experimental_debug_boxes", False):
            for cand in incidents.get("candidates", []):
                score = cand.get("color_texture_anomaly_score", cand.get("confidence", 0.0))
                annotated_frame = draw_bbox(
                    annotated_frame,
                    cand["bbox"],
                    f"EXP_COLOR_ANOMALY: {format_confidence(score)}",
                    (0, 0, 180),
                )

        total_detections = violation_result.get(
            "total_detections",
            sum(len(items) for items in detections.values()),
        )

        result_dict = {
            "timestamp": scan_time.isoformat(),
            "scan_timestamp": scan_time.isoformat(),
            "capture_timestamp": capture_timestamp.isoformat() if isinstance(capture_timestamp, datetime) else (capture_timestamp or scan_time.isoformat()),
            "original_frame": frame.copy(),
            "annotated_frame": annotated_frame,
            "detections": detections,
            "diagnostics": self.detector.last_diagnostics,
            "head_verifications": head_verifications or [],
            "violations": violation_result,
            "incidents": incidents,
            "summary": {
                "violations": violation_result["violation_count"],
                "compliance_rate": violation_result["compliance_rate"],
                "total_detections": total_detections,
            },
            "report_path": None,
            "alert": {"sent": False, "status": "Not triggered", "throttled": False},
        }

        # Automatic PDF report generation & manager alerting for actionable findings
        if should_generate_report(result_dict, self.config):
            try:
                report_path = self.report_generator.generate_report(
                    result_dict, source_name, frame, annotated_frame
                )
                result_dict["report_path"] = report_path
                alert_res = self.notification_manager.send_alert(result_dict, source_name, report_path)
                result_dict["alert"] = alert_res
            except Exception as report_err:
                print(f"Report generation/alert error: {report_err}")
                result_dict["alert"] = {"sent": False, "status": f"Error: {report_err}", "throttled": False}

        return result_dict

    def process_stream_frame(
        self,
        frame: np.ndarray,
        frame_id: int = 0,
        source_id: str = "stream_cam",
        timestamp: Optional[float | str | datetime] = None,
        capture_timestamp: Optional[datetime | str] = None,
    ) -> Dict:
        """Process an individual streaming video frame with temporal smoothing and deduplication."""
        # Keep one stable notification identity for the whole stream. Frame IDs
        # remain in the result/session event, while cooldowns deduplicate a
        # persistent incident across sampled frames.
        source_name = source_id
        proc_ts = None
        if isinstance(timestamp, (int, float)):
            try:
                proc_ts = datetime.fromtimestamp(timestamp).astimezone()
            except Exception:
                proc_ts = datetime.now().astimezone()
        elif isinstance(timestamp, (str, datetime)):
            proc_ts = timestamp

        return self.process_frame(
            frame,
            save_violation=self.save_violations,
            source_name=source_name,
            enable_temporal=True,
            timestamp=proc_ts,
            capture_timestamp=capture_timestamp,
        )

    def process_image(self, image_path: str, output_path: Optional[str] = None) -> Dict:
        """Process a single image."""
        try:
            frame = cv2.imread(image_path)
        except (FileNotFoundError, OSError) as exc:
            raise ValueError(f"Could not read image: {image_path}") from exc
        if frame is None:
            raise ValueError(f"Could not read image: {image_path}")

        result = self.process_frame(frame, True, image_path)

        if output_path:
            cv2.imwrite(output_path, result["annotated_frame"])

        return result

    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        display: bool = False,
        skip_frames: int = 0,
    ) -> Dict:
        """Process a video file using the same frame pipeline."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        self.detector.reset_temporal_state()

        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if output_path:
            writer = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height),
            )

        stats = {
            "frames_processed": 0,
            "total_violations": 0,
            "frames_with_violations": 0,
            "total_detections": 0,
        }

        frame_id = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if skip_frames > 0 and frame_id % (skip_frames + 1) != 0:
                frame_id += 1
                continue

            result = self.process_frame(
                frame,
                True,
                f"{video_path}_frame_{frame_id}",
                enable_temporal=True,
            )

            stats["frames_processed"] += 1
            stats["total_violations"] += result["violations"]["violation_count"]
            stats["total_detections"] += result["summary"]["total_detections"]

            if result["violations"]["violation_count"] > 0:
                stats["frames_with_violations"] += 1

            if writer:
                writer.write(result["annotated_frame"])

            if display:
                cv2.imshow(
                    "PPE Detection",
                    resize_with_aspect_ratio(result["annotated_frame"]),
                )
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_id += 1

        cap.release()
        if writer:
            writer.release()
        if display:
            cv2.destroyAllWindows()
        self.detector.reset_temporal_state()

        return stats

    def process_webcam(self, camera_id: int = 0):
        """Real-time webcam processing."""
        print(f"Starting webcam (camera {camera_id})")
        print("Press 'q' to quit")

        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise ValueError(f"Could not open camera {camera_id}")
        self.detector.reset_temporal_state()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = self.process_frame(
                frame,
                save_violation=True,
                source_name=f"webcam_{camera_id}",
                enable_temporal=True,
            )

            cv2.imshow(
                "PPE Detection - Webcam",
                resize_with_aspect_ratio(result["annotated_frame"]),
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.detector.reset_temporal_state()
        print("Webcam stopped")

    def _save_violation_image(
        self,
        frame: np.ndarray,
        violation_result: Dict,
        source_name: str,
    ) -> None:
        """Persist annotated violation frames."""
        violation_info = {
            "source": source_name,
            "violation_count": violation_result["violation_count"],
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        }
        save_violation_image(frame, self.violations_dir, violation_info)
