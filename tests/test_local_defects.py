from __future__ import annotations

import json
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from drag_conveyor.config import profile_from_dict
from drag_conveyor.pipeline.local_defects import (
    LocalDefectFeatures,
    _canonicalize_bar,
    _compute_color_scores_from_reference,
    _compute_shape_scores,
    _make_zone_slices,
    _mask_roi_to_frame_mask,
    _white_lab_anchor,
    build_local_defect_baseline,
)

ROOT = Path(__file__).resolve().parents[1]


def _local_config():
    raw = json.loads((ROOT / "config" / "base_profile.json").read_text(encoding="utf-8"))
    return profile_from_dict(raw).inspection.local_defect


def _full_rect_contour(width: int = 256, height: int = 64) -> np.ndarray:
    return np.array(
        [
            [[0.0, 0.0]],
            [[float(width - 1), 0.0]],
            [[float(width - 1), float(height - 1)]],
            [[0.0, float(height - 1)]],
        ],
        dtype=np.float32,
    )


class LocalDefectsTests(unittest.TestCase):
    def test_mask_roi_to_frame_mask_clips_to_frame(self) -> None:
        mask_roi = np.ones((4, 5), dtype=np.uint8) * 255

        mask_frame = _mask_roi_to_frame_mask(
            mask_roi=mask_roi,
            roi_origin_xy=(8, 7),
            frame_shape_hw=(10, 10),
        )

        self.assertEqual(mask_frame.shape, (10, 10))
        self.assertEqual(int(np.count_nonzero(mask_frame)), 6)
        self.assertTrue(np.all(mask_frame[7:10, 8:10] == 255))

    def test_canonicalize_preserves_image_left_right_for_horizontal_bar(self) -> None:
        config = _local_config()
        frame = np.zeros((80, 180, 3), dtype=np.uint8)
        contour = np.array(
            [
                [[20.0, 30.0]],
                [[160.0, 30.0]],
                [[160.0, 50.0]],
                [[20.0, 50.0]],
            ],
            dtype=np.float32,
        )
        mask_frame = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask_frame, contour[:, 0].astype(np.int32), 255)
        cv2.rectangle(frame, (20, 30), (90, 50), (255, 0, 0), -1)
        cv2.rectangle(frame, (90, 30), (160, 50), (0, 255, 0), -1)

        canonical_crop, _ = _canonicalize_bar(
            frame=frame,
            mask_frame=mask_frame,
            contour_frame=contour,
            config=config,
        )

        self.assertGreater(
            float(canonical_crop[:, :40, 0].mean()),
            float(canonical_crop[:, :40, 1].mean()),
        )
        self.assertGreater(
            float(canonical_crop[:, -40:, 1].mean()),
            float(canonical_crop[:, -40:, 0].mean()),
        )

    def test_canonicalize_preserves_image_left_right_for_rotated_bar(self) -> None:
        config = _local_config()
        frame = np.zeros((140, 180, 3), dtype=np.uint8)
        rect = ((90.0, 70.0), (120.0, 24.0), -22.0)
        box = cv2.boxPoints(rect).astype(np.int32)
        contour = box.reshape(-1, 1, 2).astype(np.float32)
        mask_frame = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask_frame, box, 255)
        cv2.fillConvexPoly(frame, box, (255, 255, 255))

        left_idx = int(np.argmin(box[:, 0]))
        right_idx = int(np.argmax(box[:, 0]))
        left_marker = tuple(int(v) for v in box[left_idx])
        right_marker = tuple(int(v) for v in box[right_idx])
        cv2.circle(frame, left_marker, 8, (255, 0, 0), -1)
        cv2.circle(frame, right_marker, 8, (0, 255, 0), -1)

        canonical_crop, _ = _canonicalize_bar(
            frame=frame,
            mask_frame=mask_frame,
            contour_frame=contour,
            config=config,
        )

        self.assertGreater(
            float(canonical_crop[:, :40, 0].mean()),
            float(canonical_crop[:, :40, 1].mean()),
        )
        self.assertGreater(
            float(canonical_crop[:, -40:, 1].mean()),
            float(canonical_crop[:, -40:, 0].mean()),
        )

    def test_canonicalize_rejects_low_aspect_ratio_contour(self) -> None:
        config = _local_config()
        frame = np.zeros((80, 80, 3), dtype=np.uint8)
        contour = np.array(
            [
                [[20.0, 20.0]],
                [[60.0, 20.0]],
                [[60.0, 60.0]],
                [[20.0, 60.0]],
            ],
            dtype=np.float32,
        )

        with self.assertRaisesRegex(ValueError, "aspect ratio too low"):
            _canonicalize_bar(
                frame=frame,
                mask_frame=np.zeros(frame.shape[:2], dtype=np.uint8),
                contour_frame=contour,
                config=config,
            )

    def test_left_notch_produces_highest_left_score(self) -> None:
        config = _local_config()
        template_mask = np.ones((64, 256), dtype=np.uint8) * 255
        current_mask = template_mask.copy()
        current_mask[:, :32] = 0

        scores = _compute_shape_scores(
            current_mask=current_mask,
            template_mask=template_mask,
            zone_slices=_make_zone_slices(256, config),
            config=config,
        )

        self.assertGreater(scores["left_shape_score"], scores["middle_shape_score"])
        self.assertGreater(scores["left_shape_score"], scores["right_shape_score"])

    def test_right_notch_produces_highest_right_score(self) -> None:
        config = _local_config()
        template_mask = np.ones((64, 256), dtype=np.uint8) * 255
        current_mask = template_mask.copy()
        current_mask[:, -32:] = 0

        scores = _compute_shape_scores(
            current_mask=current_mask,
            template_mask=template_mask,
            zone_slices=_make_zone_slices(256, config),
            config=config,
        )

        self.assertGreater(scores["right_shape_score"], scores["left_shape_score"])
        self.assertGreater(scores["right_shape_score"], scores["middle_shape_score"])

    def test_middle_hole_produces_highest_middle_score(self) -> None:
        config = _local_config()
        template_mask = np.ones((64, 256), dtype=np.uint8) * 255
        current_mask = template_mask.copy()
        current_mask[16:48, 96:160] = 0

        scores = _compute_shape_scores(
            current_mask=current_mask,
            template_mask=template_mask,
            zone_slices=_make_zone_slices(256, config),
            config=config,
        )

        self.assertGreater(scores["middle_shape_score"], scores["left_shape_score"])
        self.assertGreater(scores["middle_shape_score"], scores["right_shape_score"])

    def test_both_sides_notches_produce_left_and_right_scores(self) -> None:
        config = _local_config()
        template_mask = np.ones((64, 256), dtype=np.uint8) * 255
        current_mask = template_mask.copy()
        current_mask[:, :24] = 0
        current_mask[:, -24:] = 0

        scores = _compute_shape_scores(
            current_mask=current_mask,
            template_mask=template_mask,
            zone_slices=_make_zone_slices(256, config),
            config=config,
        )

        self.assertGreater(scores["left_shape_score"], 0.0)
        self.assertGreater(scores["right_shape_score"], 0.0)
        self.assertLess(scores["middle_shape_score"], scores["left_shape_score"])
        self.assertLess(scores["middle_shape_score"], scores["right_shape_score"])

    def test_template_median_resists_small_defect_contamination(self) -> None:
        config = replace(
            _local_config(),
            min_template_samples=3,
            max_template_area_ratio=1.0,
            color_enabled=False,
        )
        frame = np.zeros((64, 256, 3), dtype=np.uint8)
        contour = _full_rect_contour()
        normal_mask = np.ones((64, 256), dtype=np.uint8) * 255
        notched_mask = normal_mask.copy()
        notched_mask[:, :18] = 0

        bars = [
            SimpleNamespace(
                source_frame=frame.copy(),
                contour_frame=contour,
                mask_roi=normal_mask.copy(),
                roi_origin_xy=(0, 0),
            ),
            SimpleNamespace(
                source_frame=frame.copy(),
                contour_frame=contour,
                mask_roi=normal_mask.copy(),
                roi_origin_xy=(0, 0),
            ),
            SimpleNamespace(
                source_frame=frame.copy(),
                contour_frame=contour,
                mask_roi=notched_mask.copy(),
                roi_origin_xy=(0, 0),
            ),
        ]

        baseline = build_local_defect_baseline(bars=bars, config=config)

        self.assertEqual(int(baseline.template_mask[32, 8]), 255)

    def test_baseline_color_anchor_is_fixed_white(self) -> None:
        config = replace(
            _local_config(),
            min_template_samples=3,
            max_template_area_ratio=1.0,
        )
        frame = np.full((64, 256, 3), 255, dtype=np.uint8)
        contour = _full_rect_contour()
        mask = np.ones((64, 256), dtype=np.uint8) * 255
        bars = [
            SimpleNamespace(
                source_frame=frame.copy(),
                contour_frame=contour,
                mask_roi=mask.copy(),
                roi_origin_xy=(0, 0),
            )
            for _ in range(3)
        ]

        baseline = build_local_defect_baseline(bars=bars, config=config)

        self.assertTrue(np.array_equal(baseline.lab_median, _white_lab_anchor()))
        self.assertEqual(baseline.color_abnormal_ratio_p95, 0.0)
        self.assertEqual(baseline.dark_ratio_p95, 0.0)

    def test_lab_color_black_patch_increases_abnormal_ratio(self) -> None:
        white_crop = np.full((64, 256, 3), 255, dtype=np.uint8)
        black_patch_crop = white_crop.copy()
        black_patch_crop[20:44, 100:156] = 0
        full_mask = np.ones((64, 256), dtype=np.uint8) * 255
        lab_median = np.median(
            cv2.cvtColor(white_crop, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32),
            axis=0,
        ).astype(np.float32)

        scores = _compute_color_scores_from_reference(
            canonical_crop_bgr=black_patch_crop,
            current_mask=full_mask,
            template_mask=full_mask,
            lab_median=lab_median,
            color_delta_threshold=18.0,
            dark_l_threshold=90.0,
            erode_mask_iterations=0,
            min_color_pixels=50,
        )

        self.assertGreater(scores["color_abnormal_ratio"], 0.0)
        self.assertGreater(scores["color_delta_p95"], 0.0)

    def test_dark_patch_increases_dark_pixel_ratio(self) -> None:
        white_crop = np.full((64, 256, 3), 255, dtype=np.uint8)
        dark_patch_crop = white_crop.copy()
        dark_patch_crop[20:44, 100:156] = 20
        full_mask = np.ones((64, 256), dtype=np.uint8) * 255
        lab_median = np.median(
            cv2.cvtColor(white_crop, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32),
            axis=0,
        ).astype(np.float32)

        scores = _compute_color_scores_from_reference(
            canonical_crop_bgr=dark_patch_crop,
            current_mask=full_mask,
            template_mask=full_mask,
            lab_median=lab_median,
            color_delta_threshold=18.0,
            dark_l_threshold=90.0,
            erode_mask_iterations=0,
            min_color_pixels=50,
        )

        self.assertGreater(scores["dark_pixel_ratio"], 0.0)

    def test_zero_failed_contains_failure_flags(self) -> None:
        failed = LocalDefectFeatures.zero_failed()

        self.assertEqual(failed.local_analysis_success, 0.0)
        self.assertEqual(failed.local_canonicalize_failed, 1.0)
        self.assertEqual(failed.local_alignment_low, 1.0)
        self.assertEqual(failed.local_color_pixels_insufficient, 1.0)


if __name__ == "__main__":
    unittest.main()
