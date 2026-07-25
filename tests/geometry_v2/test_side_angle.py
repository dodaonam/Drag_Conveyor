from __future__ import annotations

import math
import unittest

from drag_conveyor.geometry_v2.decision import FinalStatus
from drag_conveyor.geometry_v2.side_angle import SideAxis, classify_angles, exceeds_transverse_angle


def _axis(angle: float) -> SideAxis:
    return SideAxis(math.tan(math.radians(angle)), 0.0, 100.0, 1.0, 100)


class AngleTests(unittest.TestCase):
    def test_side_and_both_bend_are_classified_without_summing_global_tilt(self) -> None:
        normal = classify_angles(left=_axis(0), right=_axis(0), side_threshold_deg=8, global_tilt_threshold_deg=5, center_kink_threshold_deg=10, decision_guard_deg=.5)
        tilted = classify_angles(left=_axis(-6), right=_axis(6), side_threshold_deg=8, global_tilt_threshold_deg=5, center_kink_threshold_deg=20, decision_guard_deg=.5)
        left = classify_angles(left=_axis(10), right=_axis(2), side_threshold_deg=8, global_tilt_threshold_deg=30, center_kink_threshold_deg=30, decision_guard_deg=.5)
        both = classify_angles(left=_axis(10), right=_axis(-12), side_threshold_deg=8, global_tilt_threshold_deg=30, center_kink_threshold_deg=30, decision_guard_deg=.5)
        self.assertEqual(normal.status, FinalStatus.NORMAL)
        self.assertEqual(tilted.status, FinalStatus.BENT_BOTH)
        self.assertEqual(left.status, FinalStatus.BENT_LEFT)
        self.assertEqual(both.status, FinalStatus.BENT_BOTH)

    def test_guard_band_abstains(self) -> None:
        summary = classify_angles(left=_axis(8), right=_axis(0), side_threshold_deg=8, global_tilt_threshold_deg=30, center_kink_threshold_deg=30, decision_guard_deg=.5)
        self.assertIsNone(summary.status)

    def test_scale_free_15_degree_rule_is_strictly_greater(self) -> None:
        self.assertFalse(exceeds_transverse_angle(_axis(15)))
        self.assertTrue(exceeds_transverse_angle(_axis(15.01)))


if __name__ == "__main__":
    unittest.main()
