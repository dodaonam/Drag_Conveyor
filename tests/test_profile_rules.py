from __future__ import annotations

import json
import unittest
from dataclasses import replace
from pathlib import Path

from drag_conveyor.calibration import CalibrationEngine
from drag_conveyor.config import (
    CalibrationResult,
    FeatureStats,
    ProfileError,
    profile_from_dict,
    profile_to_dict,
)
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
