from __future__ import annotations

import unittest
import numpy as np

from drag_conveyor.geometry_v2.coordinates import ChainCoordinates
from drag_conveyor.geometry_v2.side_integrity import SideIntegrity, analyze_side_integrity
from drag_conveyor.geometry_v2.types import Point, Roi


class SideIntegrityTests(unittest.TestCase):
    def test_internal_gap_is_positive_localized_break_evidence(self) -> None:
        coordinates = ChainCoordinates.create(Roi(0, 0, 100, 100), Point(50, 0), Point(50, 100))
        mask = np.zeros((100, 100), dtype=bool)
        mask[45:55, 5:35] = True
        mask[45:55, 41:49] = True
        result = analyze_side_integrity(mask, coordinates, side="left", chain_band_half_width=2, bins=20, valid_minimum_coverage_ratio=.85, broken_minimum_internal_gap_ratio=.10)
        self.assertEqual(result.state, SideIntegrity.BROKEN_LOCALIZED)
