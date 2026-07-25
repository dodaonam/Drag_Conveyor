from __future__ import annotations

import unittest

import numpy as np

from drag_conveyor.geometry_v2.coordinates import ChainCoordinates
from drag_conveyor.geometry_v2.observation_builder import ObservationType, build_frame_observations
from drag_conveyor.geometry_v2.observations import ComponentExtractionConfig, extract_components
from drag_conveyor.geometry_v2.pairing import PairingConfig
from drag_conveyor.geometry_v2.types import Point, Roi


def _coordinates() -> ChainCoordinates:
    return ChainCoordinates.create(Roi(0, 0, 100, 100), Point(50, 0), Point(50, 100))


def _components(mask: np.ndarray):
    coordinates = _coordinates()
    return extract_components(
        mask,
        source_frame_id=1,
        source_detection_id="d01",
        source_detection_score=0.9,
        source_model_output_row_index=1,
        class_id=0,
        coordinates=coordinates,
        config=ComponentExtractionConfig(chain_band_half_width=5, boundary_margin_px=2),
        model_bbox_roi_xyxy=(0, 0, 100, 100),
    )


class ObservationBuilderTests(unittest.TestCase):
    def test_left_right_pair_is_disconnected_without_mask_union(self) -> None:
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[40:60, 10:40] = 1
        mask[40:60, 60:90] = 1
        observations = build_frame_observations(
            _components(mask), source_timestamp_sec=0.0, coordinates=_coordinates(), chain_band_half_width=5,
            pairing_config=PairingConfig(), q_bins=10, minimum_q_coverage=0.90,
        )
        self.assertEqual([item.kind for item in observations], [ObservationType.DISCONNECTED_BOTH])
        self.assertFalse(observations[0].mask_roi.any())
        self.assertIsNotNone(observations[0].left_mask_roi)
        self.assertIsNotNone(observations[0].right_mask_roi)
        self.assertFalse(np.shares_memory(observations[0].left_mask_roi, observations[0].right_mask_roi))

    def test_connected_spanning_component_requires_raw_bridge(self) -> None:
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[40:60, 10:90] = 1
        observations = build_frame_observations(
            _components(mask), source_timestamp_sec=0.0, coordinates=_coordinates(), chain_band_half_width=5,
            pairing_config=PairingConfig(), q_bins=10, minimum_q_coverage=0.90,
        )
        self.assertEqual([item.kind for item in observations], [ObservationType.CONNECTED_WHOLE])


if __name__ == "__main__":
    unittest.main()
