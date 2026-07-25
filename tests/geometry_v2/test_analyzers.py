from __future__ import annotations

import unittest

import numpy as np

from drag_conveyor.geometry_v2 import ChainCoordinates, Point, Roi
from drag_conveyor.geometry_v2.analyzers import CenterTopology, analyze_center_bridge


class CenterTopologyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.coordinates = ChainCoordinates.create(Roi(0, 0, 100, 100), Point(50, 0), Point(50, 100))

    def test_connected_mask_has_present_center_bridge(self) -> None:
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[45:55, 20:80] = 1
        self.assertEqual(analyze_center_bridge(mask, self.coordinates, anchor_s=50, chain_band_half_width=5, q_bins=10), CenterTopology.PRESENT)

    def test_disconnected_sides_have_absent_center_bridge(self) -> None:
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[45:55, 20:45] = 1
        mask[45:55, 56:80] = 1
        self.assertEqual(analyze_center_bridge(mask, self.coordinates, anchor_s=50, chain_band_half_width=5, q_bins=10), CenterTopology.ABSENT)


if __name__ == "__main__":
    unittest.main()
