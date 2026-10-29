"""Utility functions for the PPE compliance detection system."""

import json
import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from a YAML file."""
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def ensure_dir(directory: str) -> None:
    """Create a directory if it does not already exist."""
    if directory:
        Path(directory).mkdir(parents=True, exist_ok=True)


def calculate_iou(box1: list, box2: list) -> float:
    """Calculate intersection over union."""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def is_bbox_inside(inner_box: list, outer_box: list, iou_threshold: float = 0.3) -> bool:
    """Check whether one bounding box overlaps enough with another."""
    return calculate_iou(inner_box, outer_box) >= iou_threshold


def get_contrasting_text_color(bg_color: tuple) -> tuple[int, int, int]:
    """Return (0, 0, 0) or (255, 255, 255) to maximize contrast against bg_color (BGR)."""
    b, g, r = bg_color[:3]
    luminance = 0.114 * b + 0.587 * g + 0.299 * r
    return (0, 0, 0) if luminance > 140 else (255, 255, 255)


def draw_bbox(
    image: np.ndarray,
    bbox: list,
    label: str,
    color: tuple,
    thickness: int = 2,
    text_size: float = 0.6,
) -> np.ndarray:
    """Draw a labeled bounding box with dynamic scaling, high contrast, and boundary safety."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]

    # Coordinate boundary clamping
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return image

    scale = max(0.60, min(2.5, max(h, w) / 950.0))
    effective_text_size = max(0.42, text_size * scale)
    effective_thickness = max(2, int(round(thickness * scale)))

    # Draw box outline
    cv2.rectangle(image, (x1, y1), (x2, y2), color, effective_thickness)

    # Compute text size
    (text_width, text_height), baseline = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        effective_text_size,
        max(1, effective_thickness - 1),
    )

    padding_x = max(4, int(round(4 * scale)))
    padding_y = max(4, int(round(4 * scale)))
    total_box_w = text_width + (2 * padding_x)
    total_box_h = text_height + baseline + (2 * padding_y)

    # Vertical position: prefer above box; if clipped at top, place below box; if still clipped, inside
    if y1 - total_box_h >= 0:
        label_top = y1 - total_box_h
        label_bottom = y1
    elif y2 + total_box_h <= h:
        label_top = y2
        label_bottom = y2 + total_box_h
    else:
        label_top = max(0, min(y1, h - total_box_h))
        label_bottom = min(h, label_top + total_box_h)

    # Horizontal position: prevent clipping
    label_left = max(0, min(x1, max(0, w - total_box_w)))
    label_right = min(w, label_left + total_box_w)

    sub = image[label_top:label_bottom, label_left:label_right]
    if sub.shape[0] > 0 and sub.shape[1] > 0:
        # Check luminance to prevent white-patch artifacts on bright/white boxes
        b, g, r = color[:3]
        luminance = 0.114 * b + 0.587 * g + 0.299 * r
        if luminance > 205:
            # Use a dark badge with colored accent border for light colors (prevents whiteout)
            dark_badge = np.full(sub.shape, (35, 35, 35), dtype=np.uint8)
            cv2.addWeighted(dark_badge, 0.85, sub, 0.15, 0, sub)
            text_color = (255, 255, 255)
            border_color = tuple(map(int, color[:3]))
        else:
            badge = np.full(sub.shape, color[:3], dtype=np.uint8)
            cv2.addWeighted(badge, 0.82, sub, 0.18, 0, sub)
            text_color = get_contrasting_text_color(color)
            border_color = (0, 0, 0) if text_color == (255, 255, 255) else (60, 60, 60)

        cv2.rectangle(
            image,
            (label_left, label_top),
            (label_right, label_bottom),
            border_color,
            1,
        )

        text_x = label_left + padding_x
        text_y = label_top + padding_y + text_height
        cv2.putText(
            image,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            effective_text_size,
            text_color,
            max(1, effective_thickness - 1),
            cv2.LINE_AA,
        )

    return image


def save_violation_image(
    image: np.ndarray,
    violations_dir: str,
    violation_info: dict,
) -> str:
    """Save an annotated violation image."""
    ensure_dir(violations_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"violation_{timestamp}.jpg"
    path = os.path.join(violations_dir, filename)

    cv2.imwrite(path, image)
    return path


def log_violation(log_file: str, violation_data: dict) -> None:
    """Append a violation record to a JSON log file safely."""
    ensure_dir(os.path.dirname(log_file))

    if not os.path.exists(log_file) or os.path.getsize(log_file) == 0:
        logs = []
    else:
        try:
            with open(log_file, "r", encoding="utf-8") as file:
                logs = json.load(file)
        except (json.JSONDecodeError, OSError):
            logs = []

    violation_data.setdefault("timestamp", datetime.now().isoformat())
    logs.append(violation_data)

    with open(log_file, "w", encoding="utf-8") as file:
        json.dump(logs, file, indent=4)


def format_confidence(confidence: float) -> str:
    """Format a confidence score."""
    return f"{confidence * 100:.1f}%"


def resize_with_aspect_ratio(
    image: np.ndarray,
    max_width: int = 1280,
    max_height: int = 720,
) -> np.ndarray:
    """Resize while preserving aspect ratio."""
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height)

    if scale < 1:
        new_width = int(width * scale)
        new_height = int(height * scale)
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    return image


def create_detection_summary(detections: dict) -> dict:
    """Create summary counts from normalized detections."""
    total_objects = sum(len(items) for items in detections.values())

    return {
        "total_detections": total_objects,
        "person": len(detections.get("person", [])),
        "helmet": len(detections.get("helmet", [])),
        "vest": len(detections.get("vest", [])),
        "without_helmet": len(detections.get("without_helmet", [])),
        "without_vest": len(detections.get("without_vest", [])),
    }


def format_scan_timestamp(ts) -> str:
    """Format a scan timestamp prominently and unambiguously with timezone.

    Returns: 'DD Month YYYY, HH:MM:SS TZ' (e.g., '30 October 2026, 02:50:55 Nepal Standard Time (UTC+05:45)')
    or 'Scan timestamp unavailable' if ts is None or invalid.
    """
    if ts is None:
        return "Scan timestamp unavailable"
    if isinstance(ts, str):
        if not ts.strip():
            return "Scan timestamp unavailable"
        try:
            ts_dt = datetime.fromisoformat(ts)
        except Exception:
            return "Scan timestamp unavailable"
    elif isinstance(ts, datetime):
        ts_dt = ts
    else:
        return "Scan timestamp unavailable"

    if ts_dt.tzinfo is None:
        ts_dt = ts_dt.astimezone()

    tz_str = ts_dt.tzname()
    off = ts_dt.strftime("%z")
    off_fmt = f"UTC{off[:3]}:{off[3:]}" if len(off) == 5 else (f"UTC{off}" if off else "")
    if tz_str and off_fmt and tz_str != off_fmt:
        tz_disp = f"{tz_str} ({off_fmt})"
    else:
        tz_disp = tz_str or off_fmt or "UTC"

    return f"{ts_dt.strftime('%d %B %Y, %H:%M:%S')} {tz_disp}".strip()


def format_bbox(bbox) -> str:
    """Format bounding box coordinates consistently as integer string: x1=..., y1=..., x2=..., y2=..."""
    if not bbox or len(bbox) != 4:
        return "N/A"
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
    return f"x1={x1}, y1={y1}, x2={x2}, y2={y2}"
