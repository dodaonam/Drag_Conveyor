from __future__ import annotations

import unittest

from drag_conveyor.inspection_modes import AverageRatioInspector, AverageRatioThresholds


def _thresholds() -> AverageRatioThresholds:
    return AverageRatioThresholds(
        width_min_ratio=0.84,
        width_max_ratio=1.06,
        length_min_ratio=0.94,
        length_max_ratio=1.1,
    )


class AverageRatioInspectorTests(unittest.TestCase):
    def test_compute_averages(self) -> None:
        inspector = AverageRatioInspector(
            _thresholds(),
            min_violated_dimensions=1,
            score_dimension_count=2,
        )

        averages = inspector.compute_averages(
            [
                {"length": 100.0, "width": 20.0},
                {"length": 102.0, "width": 22.0},
            ]
        )

        self.assertEqual(averages, {"length": 101.0, "width": 21.0})

    def test_evaluate_uses_fixed_ratio_thresholds(self) -> None:
        inspector = AverageRatioInspector(
            _thresholds(),
            min_violated_dimensions=1,
            score_dimension_count=2,
        )
        averages = {"length": 100.0, "width": 20.0}

        normal = inspector.evaluate({"length": 100.5, "width": 21.0}, averages)
        defect = inspector.evaluate({"length": 93.9, "width": 25.0}, averages)

        self.assertEqual(normal.result, "normal")
        self.assertEqual(normal.reasons, [])
        self.assertEqual(defect.result, "suspected_defect")
        self.assertIn("length_too_short", defect.reasons)
        self.assertIn("width_too_large", defect.reasons)


if __name__ == "__main__":
    unittest.main()
