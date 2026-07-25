from __future__ import annotations

import copy
import json
import unittest
import tempfile
from pathlib import Path

from drag_conveyor.geometry_v2.config import algorithm_config_hash, canonical_algorithm_projection, resolved_geometry_values
from drag_conveyor.geometry_v2.config_loader import load_geometry_config

ROOT = Path(__file__).resolve().parents[2]


def _config():
    return {
        "model": {"artifact_path": "weights/a.onnx", "expected_sha256": "abc"},
        "geometry": {"default_chain_band_width_ratio": 0.05, "chain_band_width_ratio_min": 0.02, "chain_band_width_ratio_max": 0.20},
        "components": {"boundary_margin_ratio": 0.003},
        "side_integrity": {
            "valid_maximum_internal_gap_ratio": 0.08,
            "broken_minimum_internal_gap_ratio": 0.15,
            "exclusion_margin_roi_width_ratio": 0.005,
            "minimum_projected_span_px": 20,
            "minimum_projected_span_roi_width_ratio": 0.04,
        },
        "center_topology": {"minimum_plausible_side_thickness_roi_width_ratio": 0.002, "maximum_plausible_side_thickness_roi_width_ratio": 0.08},
        "angle": {
            "decision_guard_deg": 0.5,
            "side_threshold_deg": 8.0,
            "global_tilt_threshold_deg": 5.0,
            "center_kink_threshold_deg": 10.0,
            "minimum_outer_endpoint_separation_px": 2,
            "minimum_outer_endpoint_separation_roi_width_ratio": 0.02,
        },
    }


class ConfigTests(unittest.TestCase):
    def test_local_artifact_path_does_not_change_algorithm_hash(self) -> None:
        first = _config()
        second = _config()
        second["model"]["artifact_path"] = "/other/machine/model.onnx"

        self.assertNotIn("artifact_path", canonical_algorithm_projection(first)["model"])
        self.assertEqual(algorithm_config_hash(first), algorithm_config_hash(second))

    def test_resolved_values_use_spec_pixel_formulas(self) -> None:
        values = resolved_geometry_values(_config(), roi_width=1000, roi_height=500, chain_span_px=510.0)

        self.assertEqual(values["chain_band_width_px"], 50.0)
        self.assertEqual(values["chain_band_half_width_px"], 25.0)
        self.assertEqual(values["boundary_margin_px"], 3)
        self.assertEqual(values["minimum_side_projected_span_px"], 40.0)

    def test_loader_requires_v2_schema_and_returns_identity_hash(self) -> None:
        config = json.loads((ROOT / "config" / "geometry_v2.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "geometry_v2.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            loaded, identity = load_geometry_config(path)

        self.assertEqual(loaded, config)
        self.assertEqual(identity, algorithm_config_hash(config))

    def test_bootstrap_config_loads(self) -> None:
        config, identity = load_geometry_config(ROOT / "config" / "geometry_v2.json")

        self.assertEqual(config["rule_version"], "geometry_v2_rules/2.0.0")
        self.assertEqual(len(identity), 64)

    def test_loader_rejects_unknown_or_missing_config_keys(self) -> None:
        config = json.loads((ROOT / "config" / "geometry_v2.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "geometry_v2.json"
            unknown = copy.deepcopy(config)
            unknown["geometry"]["surprise"] = 1
            path.write_text(json.dumps(unknown), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unknown geometry config"):
                load_geometry_config(path)

            missing = copy.deepcopy(config)
            del missing["decision"]["vlm_policy"]
            path.write_text(json.dumps(missing), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Missing geometry config"):
                load_geometry_config(path)


if __name__ == "__main__":
    unittest.main()
