from __future__ import annotations

import unittest

import numpy as np

from drag_conveyor.geometry_v2 import ChainCoordinates, Point, Roi
from drag_conveyor.geometry_v2.observations import ComponentExtractionConfig, extract_components
from drag_conveyor.geometry_v2.pairing import PairingConfig, pair_left_right_components


def _coordinates() -> ChainCoordinates:
    return ChainCoordinates.create(Roi(0, 0, 100, 100), Point(50, 0), Point(50, 100))


def _components(mask: np.ndarray):
    return extract_components(
        mask,
        source_frame_id=1,
        source_detection_id="d01",
        source_detection_score=0.9,
        source_model_output_row_index=1,
        class_id=0,
        coordinates=_coordinates(),
        config=ComponentExtractionConfig(chain_band_half_width=4.0, boundary_margin_px=2),
        model_bbox_roi_xyxy=(0.0, 0.0, 100.0, 100.0),
    )


class PairingTests(unittest.TestCase):
    def test_two_paddles_are_matched_without_crossing_order(self) -> None:
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:25, 10:30] = 1
        mask[10:25, 70:90] = 1
        mask[60:75, 10:30] = 1
        mask[60:75, 70:90] = 1

        result = pair_left_right_components(_components(mask), coordinates=_coordinates(), config=PairingConfig())

        self.assertEqual(len(result.matched_pairs), 2)
        self.assertEqual(result.ambiguous_component_ids, frozenset())

    def test_equal_competing_matches_mark_only_symmetric_difference_ambiguous(self) -> None:
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[40:60, 10:30] = 1
        mask[40:60, 70:80] = 1
        mask[40:60, 85:95] = 1

        components = _components(mask)
        result = pair_left_right_components(components, coordinates=_coordinates(), config=PairingConfig())

        self.assertEqual(len(result.ambiguous_component_ids), 3)
        self.assertEqual(result.matched_pairs, ())

    def test_far_components_never_form_match(self) -> None:
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:30, 10:30] = 1
        mask[70:90, 70:90] = 1

        result = pair_left_right_components(_components(mask), coordinates=_coordinates(), config=PairingConfig())

        self.assertEqual(result.matched_pairs, ())


if __name__ == "__main__":
    unittest.main()
