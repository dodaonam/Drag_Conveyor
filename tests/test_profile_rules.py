from __future__ import annotations

import json
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from drag_conveyor.calibration import CalibrationEngine
from drag_conveyor.config import (
    CalibrationResult,
    FeatureStats,
    LocalDefectConfig,
    ProfileError,
    profile_from_dict,
    profile_to_dict,
)
from drag_conveyor.pipeline.local_defects import LocalDefectBaseline
from drag_conveyor.pipeline.rules import RuleEngine
from drag_conveyor.pipeline.trigger import build_trigger_band

ROOT = Path(__file__).resolve().parents[1]


def _base_profile():
    raw = json.loads((ROOT / "config" / "base_profile.json").read_text(encoding="utf-8"))
    return profile_from_dict(raw)


def _base_profile_dict() -> dict:
    return json.loads((ROOT / "config" / "base_profile.json").read_text(encoding="utf-8"))


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


def _local_defect_baseline(
    *,
    color_abnormal_ratio_p95: float = 0.05,
    dark_ratio_p95: float = 0.04,
) -> LocalDefectBaseline:
    template_mask = np.ones((64, 256), dtype=np.uint8) * 255
    return LocalDefectBaseline(
        template_mask=template_mask,
        template_prob=template_mask.astype(np.float32) / 255.0,
        template_area_ratio=1.0,
        lab_median=np.array([200.0, 128.0, 128.0], dtype=np.float32),
        lab_mad=np.array([2.0, 2.0, 2.0], dtype=np.float32),
        color_abnormal_ratio_p95=color_abnormal_ratio_p95,
        dark_ratio_p95=dark_ratio_p95,
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


class ProfileRulesTests(unittest.TestCase):
    def test_with_roi_validates_and_uses_collection_trigger_band(self) -> None:
        profile = _base_profile()
        updated = profile.with_roi(
            {
                "x": 10,
                "y": 20,
                "w": 100,
                "h": 120,
                "frame_width": 320,
                "frame_height": 240,
            }
        )

        self.assertEqual(updated.region.roi.x, 10)
        self.assertEqual(updated.region.roi.y, 20)
        self.assertEqual(updated.collection.trigger_band.position_ratio, 0.5)
        self.assertEqual(updated.collection.trigger_band.thickness_ratio, 0.25)
        band = build_trigger_band(updated.region, updated.collection.trigger_band)
        self.assertEqual((band.x1, band.y1, band.x2, band.y2), (10, 65, 110, 95))
        self.assertEqual(band.centerline, 80.0)

        with self.assertRaises(ProfileError):
            profile.with_roi(
                {
                    "x": -1,
                    "y": 0,
                    "w": 50,
                    "h": 50,
                    "frame_width": 320,
                    "frame_height": 240,
                }
            )
        with self.assertRaisesRegex(ProfileError, "Unsupported ROI keys"):
            profile.with_roi(
                {
                    "x": 10,
                    "y": 20,
                    "w": 100,
                    "h": 120,
                    "frame_width": 320,
                    "frame_height": 240,
                    "position_ratio": 0.5,
                }
            )
        with self.assertRaisesRegex(ProfileError, "Missing ROI keys"):
            profile.with_roi(
                {
                    "x": 10,
                    "y": 20,
                    "w": 100,
                    "h": 120,
                    "frame_width": 320,
                }
            )

    def test_profile_rejects_old_top_level_schema_and_legacy_knobs(self) -> None:
        raw = _base_profile_dict()

        raw_with_old_region = json.loads(json.dumps(raw))
        raw_with_old_region["inspection_region"] = {
            "frame_width": 1280,
            "frame_height": 720,
            "x": 0,
            "y": 0,
            "w": 1280,
            "h": 720,
        }
        with self.assertRaisesRegex(ProfileError, "Unsupported profile keys"):
            profile_from_dict(raw_with_old_region)

        raw_with_legacy_rules = json.loads(json.dumps(raw))
        raw_with_legacy_rules["rules"] = {"score_threshold": 0.5}
        with self.assertRaisesRegex(ProfileError, "Unsupported profile keys"):
            profile_from_dict(raw_with_legacy_rules)

        raw_with_bad_lower = json.loads(json.dumps(raw))
        raw_with_bad_lower["inspection"]["auto_baseline"]["lower_percentile"] = "p10"
        with self.assertRaisesRegex(ProfileError, "inspection.auto_baseline.lower_percentile"):
            profile_from_dict(raw_with_bad_lower)

        raw_with_bad_upper = json.loads(json.dumps(raw))
        raw_with_bad_upper["inspection"]["auto_baseline"]["upper_percentile"] = "p90"
        with self.assertRaisesRegex(ProfileError, "inspection.auto_baseline.upper_percentile"):
            profile_from_dict(raw_with_bad_upper)

        raw_with_dead_trigger_mode = json.loads(json.dumps(raw))
        raw_with_dead_trigger_mode["collection"]["trigger_band"]["trigger_mode"] = (
            "centroid_crossing_and_mask_overlap"
        )
        with self.assertRaisesRegex(ProfileError, "Unsupported collection.trigger_band keys"):
            profile_from_dict(raw_with_dead_trigger_mode)

    def test_profile_requires_explicit_base_config_fields(self) -> None:
        raw_without_postprocess = _base_profile_dict()
        raw_without_postprocess["model"].pop("postprocess")
        with self.assertRaisesRegex(ProfileError, "model.postprocess"):
            profile_from_dict(raw_without_postprocess)

        raw_without_padding = _base_profile_dict()
        del raw_without_padding["model"]["preprocess"]["padding_value"]
        with self.assertRaisesRegex(ProfileError, "model.preprocess.padding_value"):
            profile_from_dict(raw_without_padding)

    def test_profile_rejects_non_list_config_fields(self) -> None:
        raw_with_string_providers = _base_profile_dict()
        raw_with_string_providers["model"]["providers"] = "CPUExecutionProvider"
        with self.assertRaisesRegex(ProfileError, "model.providers must be a list"):
            profile_from_dict(raw_with_string_providers)

        raw_with_string_class_ids = _base_profile_dict()
        raw_with_string_class_ids["model"]["postprocess"]["target_class_ids"] = "0"
        with self.assertRaisesRegex(ProfileError, "model.postprocess.target_class_ids must be a list"):
            profile_from_dict(raw_with_string_class_ids)

    def test_profile_loads_local_defect_config_and_exports_class(self) -> None:
        profile = _base_profile()

        self.assertIsInstance(profile.inspection.local_defect, LocalDefectConfig)
        self.assertTrue(profile.inspection.local_defect.enabled)
        self.assertEqual(profile.inspection.local_defect.canonical_width, 256)

    def test_profile_migrates_v1_0_0_to_v1_1_0_with_local_defect_disabled(self) -> None:
        raw = _base_profile_dict()
        raw["profile_version"] = "1.0.0"
        raw["inspection"].pop("local_defect")

        profile = profile_from_dict(raw)

        self.assertEqual(profile.profile_version, "1.1.0")
        self.assertFalse(profile.inspection.local_defect.enabled)

    def test_profile_rejects_invalid_local_defect_config(self) -> None:
        raw_with_unknown_local_key = _base_profile_dict()
        raw_with_unknown_local_key["inspection"]["local_defect"]["unknown"] = True
        with self.assertRaisesRegex(ProfileError, "Unsupported inspection.local_defect keys"):
            profile_from_dict(raw_with_unknown_local_key)

        raw_with_short_zone = _base_profile_dict()
        raw_with_short_zone["inspection"]["local_defect"]["zone_left"] = [0.0]
        with self.assertRaisesRegex(ProfileError, "inspection.local_defect.zone_left"):
            profile_from_dict(raw_with_short_zone)

        raw_with_string_zone = _base_profile_dict()
        raw_with_string_zone["inspection"]["local_defect"]["zone_left"] = ["a", "b"]
        with self.assertRaisesRegex(ProfileError, "inspection.local_defect.zone_left"):
            profile_from_dict(raw_with_string_zone)

        raw_with_bool_zone = _base_profile_dict()
        raw_with_bool_zone["inspection"]["local_defect"]["zone_left"] = [False, 0.3]
        with self.assertRaisesRegex(ProfileError, "inspection.local_defect.zone_left"):
            profile_from_dict(raw_with_bool_zone)

        raw_with_reversed_zone = _base_profile_dict()
        raw_with_reversed_zone["inspection"]["local_defect"]["zone_left"] = [0.4, 0.3]
        with self.assertRaisesRegex(ProfileError, "inspection.local_defect.zone_left"):
            profile_from_dict(raw_with_reversed_zone)

        raw_with_out_of_range_zone = _base_profile_dict()
        raw_with_out_of_range_zone["inspection"]["local_defect"]["zone_left"] = [-0.1, 0.3]
        with self.assertRaisesRegex(ProfileError, "inspection.local_defect.zone_left"):
            profile_from_dict(raw_with_out_of_range_zone)

        raw_with_even_kernel = _base_profile_dict()
        raw_with_even_kernel["inspection"]["local_defect"]["morph_kernel_size"] = 4
        with self.assertRaisesRegex(ProfileError, "inspection.local_defect.morph_kernel_size"):
            profile_from_dict(raw_with_even_kernel)

        raw_with_bad_template_area = _base_profile_dict()
        raw_with_bad_template_area["inspection"]["local_defect"]["min_template_area_ratio"] = 0.5
        raw_with_bad_template_area["inspection"]["local_defect"]["max_template_area_ratio"] = 0.5
        with self.assertRaisesRegex(ProfileError, "min_template_area_ratio"):
            profile_from_dict(raw_with_bad_template_area)

    def test_frontend_trigger_preview_matches_runtime_contract(self) -> None:
        html = (ROOT / "server" / "static" / "index.html").read_text(encoding="utf-8")
        raw = _base_profile_dict()

        self.assertEqual(raw["collection"]["trigger_band"]["position_ratio"], 0.5)
        self.assertEqual(raw["collection"]["trigger_band"]["thickness_ratio"], 0.25)
        self.assertIn("/api/runtime-config", html)
        self.assertIn("collection?.trigger_band", html)
        self.assertIn("bandConfig.position_ratio", html)
        self.assertIn("bandConfig.thickness_ratio", html)
        self.assertNotIn("TRIGGER_POSITION_RATIO", html)
        self.assertNotIn("TRIGGER_THICKNESS_RATIO", html)
        self.assertNotIn("inspection_region?.trigger_band", html)
        self.assertNotIn("sl-pos", html)
        self.assertNotIn("sl-thick", html)
        self.assertIn("local_defect", raw["inspection"])

    def test_rule_engine_uses_length_width_auto_baseline_contract(self) -> None:
        engine = RuleEngine()
        profile = _base_profile()
        rules = profile.inspection.auto_baseline
        policy = profile.inspection.defect_policy
        calibration = _calibration_result()

        normal = engine.evaluate({"length": 100.0, "width": 20.0}, rules, policy, calibration)
        self.assertEqual(normal.result, "normal")
        self.assertEqual(normal.score, 0.0)
        self.assertFalse(hasattr(normal, "violated_soft_rules"))

        length_short = engine.evaluate({"length": 80.0, "width": 20.0}, rules, policy, calibration)
        self.assertEqual(length_short.result, "suspected_defect")
        self.assertEqual(length_short.reasons, ["length_too_short"])
        self.assertEqual(length_short.score, 0.5)

        both = engine.evaluate({"length": 120.0, "width": 25.0}, rules, policy, calibration)
        self.assertCountEqual(both.reasons, ["length_too_long", "width_too_large"])
        self.assertEqual(both.score, 1.0)

        require_two_dimensions = replace(policy, min_violated_dimensions=2)
        one_violation = engine.evaluate(
            {"length": 80.0, "width": 20.0},
            rules,
            require_two_dimensions,
            calibration,
        )
        self.assertEqual(one_violation.result, "normal")
        self.assertEqual(one_violation.score, 0.5)

    def test_rule_engine_uses_configured_percentile_range(self) -> None:
        engine = RuleEngine()
        profile = _base_profile()
        calibration = _calibration_result()
        policy = profile.inspection.defect_policy
        rules = replace(
            profile.inspection.auto_baseline,
            lower_percentile="p2",
            upper_percentile="p98",
        )

        normal = engine.evaluate({"length": 92.0, "width": 18.0}, rules, policy, calibration)
        short = engine.evaluate({"length": 90.5, "width": 18.0}, rules, policy, calibration)
        long = engine.evaluate({"length": 109.5, "width": 18.0}, rules, policy, calibration)

        self.assertEqual(normal.result, "normal")
        self.assertEqual(short.reasons, ["length_too_short"])
        self.assertEqual(long.reasons, ["length_too_long"])
        self.assertEqual(normal.thresholds["length_min"], 91.0)
        self.assertEqual(normal.thresholds["length_max"], 109.0)
        with self.assertRaisesRegex(ValueError, "missing percentile: p10"):
            engine.evaluate(
                {"length": 100.0, "width": 20.0},
                replace(rules, lower_percentile="p10"),
                policy,
                calibration,
            )

    def test_rule_engine_local_defect_marks_result_independently(self) -> None:
        engine = RuleEngine()
        profile = _base_profile()
        rules = profile.inspection.auto_baseline
        policy = profile.inspection.defect_policy
        local_config = profile.inspection.local_defect
        calibration = _calibration_result()

        evaluation = engine.evaluate(
            {
                "length": 100.0,
                "width": 20.0,
                "left_shape_score": 0.2,
                "middle_shape_score": 0.0,
                "right_shape_score": 0.0,
                "left_defect_weighted_pixels": 40.0,
                "middle_defect_weighted_pixels": 0.0,
                "right_defect_weighted_pixels": 0.0,
                "local_alignment_low": 0.0,
            },
            rules,
            policy,
            calibration,
            local_defect_config=local_config,
            local_defect_baseline=_local_defect_baseline(),
        )

        self.assertEqual(evaluation.result, "suspected_defect")
        self.assertEqual(evaluation.reasons, ["deform_left"])

    def test_rule_engine_supports_multi_label_local_reasons(self) -> None:
        engine = RuleEngine()
        profile = _base_profile()
        rules = profile.inspection.auto_baseline
        policy = profile.inspection.defect_policy
        local_config = profile.inspection.local_defect
        calibration = _calibration_result()

        evaluation = engine.evaluate(
            {
                "length": 100.0,
                "width": 20.0,
                "left_shape_score": 0.11,
                "middle_shape_score": 0.2,
                "right_shape_score": 0.11,
                "left_defect_weighted_pixels": 40.0,
                "middle_defect_weighted_pixels": 50.0,
                "right_defect_weighted_pixels": 45.0,
                "local_alignment_low": 0.0,
                "color_abnormal_ratio": 0.3,
                "color_delta_p95": 30.0,
                "dark_pixel_ratio": 0.2,
            },
            rules,
            policy,
            calibration,
            local_defect_config=local_config,
            local_defect_baseline=_local_defect_baseline(
                color_abnormal_ratio_p95=0.2,
                dark_ratio_p95=0.1,
            ),
        )

        self.assertCountEqual(
            evaluation.reasons,
            ["deform_both_sides", "deform_middle", "color_defect"],
        )
        self.assertEqual(evaluation.result, "suspected_defect")

    def test_rule_engine_uses_low_alignment_severe_fallback(self) -> None:
        engine = RuleEngine()
        profile = _base_profile()
        rules = profile.inspection.auto_baseline
        policy = profile.inspection.defect_policy
        local_config = profile.inspection.local_defect
        calibration = _calibration_result()

        mild = engine.evaluate(
            {
                "length": 100.0,
                "width": 20.0,
                "left_shape_score": 0.2,
                "middle_shape_score": 0.0,
                "right_shape_score": 0.0,
                "left_defect_weighted_pixels": 0.0,
                "middle_defect_weighted_pixels": 0.0,
                "right_defect_weighted_pixels": 0.0,
                "local_alignment_low": 1.0,
            },
            rules,
            policy,
            calibration,
            local_defect_config=local_config,
            local_defect_baseline=_local_defect_baseline(),
        )
        severe = engine.evaluate(
            {
                "length": 100.0,
                "width": 20.0,
                "left_shape_score": 0.35,
                "middle_shape_score": 0.0,
                "right_shape_score": 0.0,
                "left_defect_weighted_pixels": 0.0,
                "middle_defect_weighted_pixels": 0.0,
                "right_defect_weighted_pixels": 0.0,
                "local_alignment_low": 1.0,
            },
            rules,
            policy,
            calibration,
            local_defect_config=local_config,
            local_defect_baseline=_local_defect_baseline(),
        )

        self.assertEqual(mild.reasons, [])
        self.assertEqual(severe.reasons, ["deform_left"])

    def test_rule_engine_uses_static_white_anchor_color_thresholds(self) -> None:
        engine = RuleEngine()
        profile = _base_profile()
        rules = profile.inspection.auto_baseline
        policy = profile.inspection.defect_policy
        local_config = profile.inspection.local_defect
        calibration = _calibration_result()

        below = engine.evaluate(
            {
                "length": 100.0,
                "width": 20.0,
                "color_abnormal_ratio": 0.14,
                "color_delta_p95": 114.0,
                "dark_pixel_ratio": 0.05,
            },
            rules,
            policy,
            calibration,
            local_defect_config=local_config,
            local_defect_baseline=_local_defect_baseline(),
        )
        above = engine.evaluate(
            {
                "length": 100.0,
                "width": 20.0,
                "color_abnormal_ratio": 0.16,
                "color_delta_p95": 116.0,
                "dark_pixel_ratio": 0.05,
            },
            rules,
            policy,
            calibration,
            local_defect_config=local_config,
            local_defect_baseline=_local_defect_baseline(),
        )

        self.assertEqual(below.reasons, [])
        self.assertEqual(above.reasons, ["color_defect"])
        self.assertAlmostEqual(
            above.thresholds["color_abnormal_ratio_threshold"],
            0.15,
            places=6,
        )
        self.assertAlmostEqual(above.thresholds["dark_pixel_ratio_threshold"], 0.12, places=6)

    def test_rule_engine_does_not_require_dynamic_color_baseline(self) -> None:
        engine = RuleEngine()
        profile = _base_profile()
        evaluation = engine.evaluate(
            {
                "length": 100.0,
                "width": 20.0,
                "color_abnormal_ratio": 0.16,
                "color_delta_p95": 116.0,
                "dark_pixel_ratio": 0.0,
            },
            profile.inspection.auto_baseline,
            profile.inspection.defect_policy,
            _calibration_result(),
            local_defect_config=profile.inspection.local_defect,
            local_defect_baseline=None,
        )

        self.assertEqual(evaluation.result, "suspected_defect")
        self.assertEqual(evaluation.reasons, ["color_defect"])

    def test_rule_engine_ignores_local_metrics_when_local_defect_disabled(self) -> None:
        engine = RuleEngine()
        profile = _base_profile()
        disabled_local = replace(profile.inspection.local_defect, enabled=False)

        evaluation = engine.evaluate(
            {
                "length": 100.0,
                "width": 20.0,
                "left_shape_score": 0.5,
                "left_defect_weighted_pixels": 99.0,
                "local_alignment_low": 0.0,
            },
            profile.inspection.auto_baseline,
            profile.inspection.defect_policy,
            _calibration_result(),
            local_defect_config=disabled_local,
            local_defect_baseline=_local_defect_baseline(),
        )

        self.assertEqual(evaluation.result, "normal")
        self.assertEqual(evaluation.reasons, [])

    def test_rule_engine_does_not_inflate_score_for_non_triggered_local_metrics(self) -> None:
        engine = RuleEngine()
        profile = _base_profile()

        evaluation = engine.evaluate(
            {
                "length": 100.0,
                "width": 20.0,
                "middle_shape_score": 0.13,
                "middle_defect_weighted_pixels": 50.0,
                "local_alignment_low": 0.0,
            },
            profile.inspection.auto_baseline,
            profile.inspection.defect_policy,
            _calibration_result(),
            local_defect_config=profile.inspection.local_defect,
            local_defect_baseline=_local_defect_baseline(),
        )

        self.assertEqual(evaluation.result, "normal")
        self.assertEqual(evaluation.reasons, [])
        self.assertEqual(evaluation.score, 0.0)

    def test_rule_engine_color_delta_p95_defect_contributes_to_score(self) -> None:
        engine = RuleEngine()
        profile = _base_profile()

        evaluation = engine.evaluate(
            {
                "length": 100.0,
                "width": 20.0,
                "color_abnormal_ratio": 0.0,
                "color_delta_p95": 130.0,
                "dark_pixel_ratio": 0.0,
            },
            profile.inspection.auto_baseline,
            profile.inspection.defect_policy,
            _calibration_result(),
            local_defect_config=profile.inspection.local_defect,
            local_defect_baseline=_local_defect_baseline(),
        )

        self.assertEqual(evaluation.result, "suspected_defect")
        self.assertEqual(evaluation.reasons, ["color_defect"])
        self.assertEqual(evaluation.score, 1.0)

    def test_calibration_updates_profile_without_mutating_rule_schema(self) -> None:
        profile = _base_profile()
        auto = profile.inspection.auto_baseline
        auto.calibration.min_valid_records = 3
        auto.lower_percentile = "p2"
        auto.upper_percentile = "p98"
        records = [
            {"length": 100.0, "width": 20.0},
            {"length": 101.0, "width": 20.5},
            {"length": 99.5, "width": 19.8},
        ]

        outcome = CalibrationEngine().calibrate(records, profile)

        self.assertTrue(outcome.success)
        self.assertIsNotNone(outcome.updated_profile)
        assert outcome.updated_profile is not None
        assert outcome.calibration_result is not None
        self.assertEqual(outcome.calibration_result.thresholds_source, "auto_baseline_median_mad_p2_p98")
        self.assertIs(outcome.updated_profile.inspection.auto_baseline.calibration_result, outcome.calibration_result)
        auto_payload = profile_to_dict(outcome.updated_profile)["inspection"]["auto_baseline"]
        self.assertEqual(auto_payload["lower_percentile"], "p2")
        self.assertEqual(auto_payload["upper_percentile"], "p98")
        self.assertEqual(auto_payload["rules_version"], "geometry_v1")
        self.assertIn("calibration", auto_payload)
        self.assertIn("calibration_result", auto_payload)


if __name__ == "__main__":
    unittest.main()
