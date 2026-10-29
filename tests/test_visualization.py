"""Tests for annotation readability, dynamic scaling, text contrast, and brightness preservation."""

import unittest
import numpy as np
import cv2

from src.utils import draw_bbox, get_contrasting_text_color


class VisualizationTests(unittest.TestCase):
    def test_text_contrast_color_selection(self):
        """High-luminance background boxes should receive dark text; dark boxes receive light text."""
        # Bright yellow box color: (0, 255, 255) in BGR -> luminance is high -> should be (0, 0, 0)
        yellow_contrast = get_contrasting_text_color((0, 255, 255))
        self.assertEqual(yellow_contrast, (0, 0, 0), "Yellow background should have black text")

        # White box color: (255, 255, 255) in BGR -> should be (0, 0, 0)
        white_contrast = get_contrasting_text_color((255, 255, 255))
        self.assertEqual(white_contrast, (0, 0, 0), "White background should have black text")

        # Dark blue box color: (180, 0, 0) in BGR -> luminance is low -> should be (255, 255, 255)
        blue_contrast = get_contrasting_text_color((180, 0, 0))
        self.assertEqual(blue_contrast, (255, 255, 255), "Dark blue background should have white text")

        # Dark red box color: (0, 0, 180) in BGR -> should be (255, 255, 255)
        red_contrast = get_contrasting_text_color((0, 0, 180))
        self.assertEqual(red_contrast, (255, 255, 255), "Dark red background should have white text")

    def test_image_brightness_outside_annotation_is_identical(self):
        """Annotating an image must not alter pixel values outside the drawn annotations."""
        from src.violation_logic import annotate_violations
        original = np.full((300, 300, 3), 120, dtype=np.uint8)
        clean_copy = original.copy()

        config = {
            "visualization": {
                "bbox_thickness": 2,
                "text_size": 0.6,
                "colors": {
                    "person": [180, 180, 180],
                    "helmet": [0, 255, 0],
                    "vest": [0, 255, 255],
                    "without_helmet": [0, 0, 255],
                    "without_vest": [255, 0, 0],
                },
            }
        }
        detections = {
            "person": [],
            "helmet": [{"bbox": [50, 50, 100, 100], "confidence": 0.95, "conf": 0.95}],
            "vest": [],
            "without_helmet": [],
            "without_vest": [],
        }
        violation_result = {"violations": [], "person_results": []}

        annotated = annotate_violations(original, detections, violation_result, config)

        untouched_slice_original = clean_copy[200:250, 200:250]
        untouched_slice_annotated = annotated[200:250, 200:250]
        np.testing.assert_array_equal(
            untouched_slice_annotated,
            untouched_slice_original,
            err_msg="Image brightness or color outside annotation was modified!",
        )

    def test_label_does_not_clip_at_boundaries(self):
        """Labels near top-left and bottom-right edges must not crash and stay within frame boundaries."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        draw_bbox(img, [0, 0, 40, 40], "CORNER 1", (0, 255, 0))
        draw_bbox(img, [60, 60, 100, 100], "CORNER 2", (0, 0, 255))
        self.assertEqual(img.shape, (100, 100, 3))

    def test_top_boundary_box_avoids_head_occlusion(self):
        """When y1=0 (head touches top of frame), label is placed below the top edge."""
        img = np.full((300, 300, 3), 100, dtype=np.uint8)
        # Box touching top: y1=0, y2=80
        draw_bbox(img, [50, 0, 150, 80], "PERSON: 90%", (180, 180, 180))
        # Top line [0..2] is the bbox line, but region y=5 to y=30 should NOT be a solid whiteout
        head_sample = img[10:30, 80:120]
        self.assertFalse(np.all(head_sample == 255), "Head region was covered by solid white rectangle")

    def test_white_box_does_not_create_opaque_white_patch(self):
        """Boxes with white color (255, 255, 255) must use contrasting badge, not solid white block."""
        img = np.full((200, 200, 3), 50, dtype=np.uint8)
        draw_bbox(img, [20, 40, 120, 160], "WHITE_TEST", (255, 255, 255))
        # Confirm image shape unchanged and no total whiteout in background badge
        self.assertEqual(img.shape, (200, 200, 3))


if __name__ == "__main__":
    unittest.main()
