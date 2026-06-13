from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from drag_conveyor.app import batch
from drag_conveyor.config import CalibrationResult, FeatureStats, profile_from_dict
from drag_conveyor.inspection_modes import AVERAGE_RATIO_INSPECTION_MODE
from drag_conveyor.pipeline.local_defects import LocalDefectBaseline, LocalDefectFeatures

ROOT = Path(__file__).resolve().parents[1]


def _profile():
    raw = json.loads((ROOT / "config" / "base_profile.json").read_text(encoding="utf-8"))
    return profile_from_dict(raw)


def _calibration_result() -> CalibrationResult:
    return CalibrationResult(
        created_at="2026-06-09T00:00:00",
        rules_updated_at="2026-06-09T00:00:00",
        rules_version="length_width_auto_baseline_v1",
        sample_count=80,
        valid_records=80,
        inlier_count=80,
        outlier_count=0,
        inlier_ratio=1.0,
        thresholds_source="test",
        features={
            "length": FeatureStats(
                median=100.0,
                mad=1.0,
                p1=90.0,
                p2=91.0,
                p3=92.0,
                p4=93.0,
                p5=95.0,
                p95=105.0,
                p96=106.0,
                p97=107.0,
                p98=109.0,
                p99=110.0,
            ),
            "width": FeatureStats(
                median=20.0,
                mad=1.0,
                p1=16.0,
                p2=17.0,
                p3=17.5,
                p4=17.8,
                p5=18.0,
                p95=22.0,
                p96=22.4,
                p97=22.7,
                p98=23.0,
                p99=24.0,
            ),
        },
    )


def _baseline() -> LocalDefectBaseline:
    template_mask = np.ones((64, 256), dtype=np.uint8) * 255
    return LocalDefectBaseline(
        template_mask=template_mask,
        template_prob=template_mask.astype(np.float32) / 255.0,
        template_area_ratio=1.0,
        lab_median=np.array([200.0, 128.0, 128.0], dtype=np.float32),
        lab_mad=np.array([2.0, 2.0, 2.0], dtype=np.float32),
        color_abnormal_ratio_p95=0.05,
        dark_ratio_p95=0.04,
        baseline_alignment_iou_p50=0.9,
        baseline_alignment_iou_p10=0.8,
        canonicalize_failure_ratio=0.0,
        samples_used=40,
        zone_slices={
            "left": slice(0, 84),
            "middle": slice(84, 169),
            "right": slice(169, 256),
        },
    )


def _collected_bar() -> batch.CollectedBar:
    frame = np.zeros((64, 256, 3), dtype=np.uint8)
    contour = np.array(
        [
            [[0.0, 0.0]],
            [[255.0, 0.0]],
            [[255.0, 63.0]],
            [[0.0, 63.0]],
        ],
        dtype=np.float32,
    )
    return batch.CollectedBar(
        frame_id=1,
        track_id=7,
        measurements={"length": 100.0, "width": 20.0},
        bbox_frame_xyxy=(0.0, 0.0, 255.0, 63.0),
        overlap_ratio=1.0,
        contour_frame=contour,
        mask_roi=np.ones((64, 256), dtype=np.uint8) * 255,
        roi_origin_xy=(0, 0),
        source_frame=frame,
        latency_ms=3.0,
    )


class BatchLocalDefectsTests(unittest.TestCase):
    def test_auto_baseline_enabled_calls_local_defect_helpers(self) -> None:
        profile = _profile()
        calibration_result = _calibration_result()
        local_features = LocalDefectFeatures(
            left_shape_score=0.2,
            middle_shape_score=0.0,
            right_shape_score=0.0,
            max_shape_score=0.2,
            left_defect_weighted_pixels=40.0,
            middle_defect_weighted_pixels=0.0,
            right_defect_weighted_pixels=0.0,
            shape_alignment_iou=0.8,
            mask_area_ratio=1.0,
            local_alignment_low=0.0,
            color_delta_mean=0.0,
            color_delta_p95=0.0,
            color_abnormal_ratio=0.0,
            dark_pixel_ratio=0.0,
            local_color_pixels_insufficient=0.0,
            local_analysis_success=1.0,
            local_canonicalize_failed=0.0,
        )

        with (
            mock.patch.object(
                batch.CalibrationEngine,
                "calibrate",
                return_value=mock.Mock(
                    success=True,
                    reason="ok",
                    calibration_result=calibration_result,
                    updated_profile=profile,
                ),
            ),
            mock.patch.object(batch, "build_local_defect_baseline", return_value=_baseline()) as build_mock,
            mock.patch.object(batch, "analyze_local_defects", return_value=local_features) as analyze_mock,
        ):
            outcome = batch._classify_with_auto_baseline([_collected_bar()], profile)

        build_mock.assert_called_once()
        analyze_mock.assert_called_once()
        self.assertEqual(outcome.bars[0].result, "suspected_defect")
        self.assertEqual(outcome.bars[0].reasons, ["deform_left"])
        self.assertIn("left_shape_score", outcome.bars[0].measurements)

    def test_local_baseline_failure_propagates_specific_reason(self) -> None:
        profile = _profile()
        calibration_result = _calibration_result()

        with (
            mock.patch.object(
                batch.CalibrationEngine,
                "calibrate",
                return_value=mock.Mock(
                    success=True,
                    reason="ok",
                    calibration_result=calibration_result,
                    updated_profile=profile,
                ),
            ),
            mock.patch.object(
                batch,
                "build_local_defect_baseline",
                side_effect=ValueError("local_baseline_not_stable: not enough template samples"),
            ),
        ):
            with self.assertRaisesRegex(
                ValueError,
                "local_baseline_not_stable: not enough template samples",
            ):
                batch._classify_with_auto_baseline([_collected_bar()], profile)

    def test_average_ratio_mode_does_not_call_local_defect_analysis(self) -> None:
        profile = _profile()

        with (
            mock.patch.object(batch, "build_local_defect_baseline") as build_mock,
            mock.patch.object(batch, "analyze_local_defects") as analyze_mock,
        ):
            outcome = batch._classify_collected_bars(
                collected=[_collected_bar()],
                profile=profile,
                inspection_mode=AVERAGE_RATIO_INSPECTION_MODE,
            )

        build_mock.assert_not_called()
        analyze_mock.assert_not_called()
        self.assertEqual(len(outcome.bars), 1)

    def test_disabled_local_defect_does_not_call_local_helpers(self) -> None:
        profile = _profile()
        profile.inspection.local_defect.enabled = False
        calibration_result = _calibration_result()

        with (
            mock.patch.object(
                batch.CalibrationEngine,
                "calibrate",
                return_value=mock.Mock(
                    success=True,
                    reason="ok",
                    calibration_result=calibration_result,
                    updated_profile=profile,
                ),
            ),
            mock.patch.object(batch, "build_local_defect_baseline") as build_mock,
            mock.patch.object(batch, "analyze_local_defects") as analyze_mock,
        ):
            outcome = batch._classify_with_auto_baseline([_collected_bar()], profile)

        build_mock.assert_not_called()
        analyze_mock.assert_not_called()
        self.assertEqual(outcome.bars[0].result, "normal")


if __name__ == "__main__":
    unittest.main()
