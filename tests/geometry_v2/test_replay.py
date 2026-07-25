from __future__ import annotations

import unittest

from drag_conveyor.geometry_v2.replay import canonical_decision_hash, compare_replays


class ReplayTests(unittest.TestCase):
    def test_runtime_only_fields_do_not_change_decision_hash(self) -> None:
        first = {"inspection_mode": "geometry_v2", "total_bars": 1, "defects": [{"vision_status": "uncertain", "latency_ms": 2.1, "snapshot_url": "a"}]}
        second = {"inspection_mode": "geometry_v2", "total_bars": 1, "defects": [{"vision_status": "uncertain", "latency_ms": 3.2, "snapshot_url": "b"}]}
        comparison = compare_replays(first, second)

        self.assertTrue(comparison.identical)
        self.assertEqual(canonical_decision_hash(first), canonical_decision_hash(second))

    def test_decision_difference_reports_path_and_non_finite_fails(self) -> None:
        comparison = compare_replays({"total_bars": 1}, {"total_bars": 2})
        self.assertFalse(comparison.identical)
        self.assertEqual(comparison.differing_paths, ("$.total_bars",))
        with self.assertRaises(ValueError):
            canonical_decision_hash({"total_bars": float("nan")})


if __name__ == "__main__":
    unittest.main()
