from __future__ import annotations

import unittest
from pathlib import Path

from drag_conveyor.config import load_profile
from drag_conveyor.geometry_v2.config_loader import load_geometry_config
from drag_conveyor.geometry_v2.pipeline import _pinned_geometry_model


ROOT = Path(__file__).resolve().parents[2]


class GeometryModelPinningTests(unittest.TestCase):
    def test_roi_model_zoo_selection_never_changes_geometry_model(self) -> None:
        config, _ = load_geometry_config(ROOT / "config" / "geometry_v2.json")
        base = load_profile(ROOT / "config" / "base_profile.json")
        for long_side in (300, 400, 500):
            profile = base.with_roi({"x": 0, "y": 0, "w": long_side, "h": 200, "frame_width": 1280, "frame_height": 720})
            model = _pinned_geometry_model(profile.model, config)
            self.assertEqual(model.path, config["model"]["artifact_path"])
            self.assertEqual(model.input_size, config["model"]["expected_input_size"])
            self.assertIsNone(model.model_zoo)


if __name__ == "__main__":
    unittest.main()
