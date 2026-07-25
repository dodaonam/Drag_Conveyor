from __future__ import annotations

import unittest

from drag_conveyor.geometry_v2.statistics import clopper_pearson_one_sided


class StatisticsTests(unittest.TestCase):
    def test_exact_bounds_handle_extremes_and_are_conservative(self) -> None:
        self.assertEqual(clopper_pearson_one_sided(0, 10)[0], 0.0)
        self.assertEqual(clopper_pearson_one_sided(10, 10)[1], 1.0)
        lower, upper = clopper_pearson_one_sided(99, 100)
        self.assertLess(lower, .99)
        self.assertGreater(upper, .99)
        self.assertLess(lower, upper)


if __name__ == "__main__":
    unittest.main()
