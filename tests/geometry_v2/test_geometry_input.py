from __future__ import annotations

import unittest

from drag_conveyor.geometry_v2.pipeline import GeometryInput


class GeometryInputTests(unittest.TestCase):
    def test_motion_direction_is_explicit_and_validated(self) -> None:
        value = {"schema_version": "geometry_input/2.0", "chain_centerline": {"top": {"x": 50, "y": 0}, "bottom": {"x": 50, "y": 100}}, "motion_direction": "negative_s"}
        parsed = GeometryInput.from_mapping(value, default_ratio=.05)
        self.assertEqual(parsed.motion_direction, "negative_s")
        value["motion_direction"] = "sideways"
        with self.assertRaisesRegex(ValueError, "motion_direction"):
            GeometryInput.from_mapping(value, default_ratio=.05)


if __name__ == "__main__":
    unittest.main()
