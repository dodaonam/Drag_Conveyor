from __future__ import annotations

import unittest

import numpy as np

from drag_conveyor.geometry_v2.coordinates import ChainCoordinates
from drag_conveyor.geometry_v2.decision import FinalStatus
from drag_conveyor.geometry_v2.fusion import PhysicalEvent
from drag_conveyor.geometry_v2.observation_builder import ObservationType, PaddleObservation
from drag_conveyor.geometry_v2.pipeline import _classify_placeholder_event
from drag_conveyor.geometry_v2.tracklets import TrackLifecycleConfig, TrackMeasurement, Tracklet
from drag_conveyor.geometry_v2.tracking import KalmanConfig
from drag_conveyor.geometry_v2.types import Point, Roi


class PipelineDecisionTests(unittest.TestCase):
    def test_connected_horizontal_observation_is_normal_under_15_degree_rule(self) -> None:
        coordinates = ChainCoordinates.create(Roi(0, 0, 100, 100), Point(50, 0), Point(50, 100))
        mask = np.zeros((100, 100), dtype=bool)
        mask[45:55, 5:95] = True
        observation = PaddleObservation("f000000001-o01", 1, 0.0, ObservationType.CONNECTED_WHOLE, (), 50.0, 1.0, mask)
        track = Tracklet.seed(1, TrackMeasurement(observation.observation_id, 1, 0.0, 50.0, 1.0)).append_measurement(
            TrackMeasurement("f000000002-o01", 2, .1, 55.0, 1.0), chain_span_px=100.0,
            kalman_config=KalmanConfig(chain_span_px=100.0), lifecycle_config=TrackLifecycleConfig(),
        ).append_measurement(
            TrackMeasurement("f000000003-o01", 3, .2, 60.0, 1.0), chain_span_px=100.0,
            kalman_config=KalmanConfig(chain_span_px=100.0), lifecycle_config=TrackLifecycleConfig(),
        )
        event = PhysicalEvent(1, (1,), .0)
        second = PaddleObservation("f000000002-o01", 2, .1, ObservationType.CONNECTED_WHOLE, (), 55.0, 1.0, mask)
        third = PaddleObservation("f000000003-o01", 3, .2, ObservationType.CONNECTED_WHOLE, (), 60.0, 1.0, mask)
        config = _decision_config(minimum_spacing_frames=1, minimum_spacing_sec=.01)
        result = _classify_placeholder_event(event, (track,), {observation.observation_id: observation, second.observation_id: second, third.observation_id: third}, coordinates, 2.5, config)
        self.assertEqual(result.status, FinalStatus.NORMAL)
        self.assertEqual(result.diagnostics["center_state"], "intact")
        self.assertEqual(result.diagnostics["observability_grade"], "GRADE_A")
        self.assertEqual(result.diagnostics["independent_angle_sample_count"], 3)
        self.assertAlmostEqual(result.diagnostics["left_angle_deg"], 0.0)

    def test_independent_disconnected_observations_become_broken_center(self) -> None:
        coordinates = ChainCoordinates.create(Roi(0, 0, 100, 100), Point(50, 0), Point(50, 100))
        empty = np.zeros((100, 100), dtype=bool)
        left = np.zeros((100, 100), dtype=bool)
        right = np.zeros((100, 100), dtype=bool)
        left[45:55, 5:49] = True
        right[45:55, 51:95] = True
        first = PaddleObservation("f000000001-o01", 1, 0.0, ObservationType.DISCONNECTED_BOTH, (), 50.0, 1.0, empty, left, right)
        second = PaddleObservation("f000000003-o01", 3, .2, ObservationType.DISCONNECTED_BOTH, (), 55.0, 1.0, empty, left, right)
        track = Tracklet.seed(1, TrackMeasurement(first.observation_id, 1, 0.0, 50.0, 1.0)).append_measurement(
            TrackMeasurement(second.observation_id, 3, .2, 55.0, 1.0), chain_span_px=100.0,
            kalman_config=KalmanConfig(chain_span_px=100.0), lifecycle_config=TrackLifecycleConfig(),
        )
        config = _decision_config(minimum_spacing_frames=2, minimum_spacing_sec=.05)
        result = _classify_placeholder_event(PhysicalEvent(1, (1,), .1), (track,), {first.observation_id: first, second.observation_id: second}, coordinates, 2.5, config)
        self.assertEqual(result.status, FinalStatus.BROKEN_CENTER)

    def test_repeated_left_internal_gap_with_valid_right_is_broken_left(self) -> None:
        coordinates = ChainCoordinates.create(Roi(0, 0, 100, 100), Point(50, 0), Point(50, 100))
        mask = np.zeros((100, 100), dtype=bool)
        mask[45:55, 5:35] = True
        mask[45:55, 42:96] = True
        first = PaddleObservation("f000000001-o01", 1, 0.0, ObservationType.CONNECTED_WHOLE, (), 50.0, 1.0, mask)
        second = PaddleObservation("f000000002-o01", 2, .1, ObservationType.CONNECTED_WHOLE, (), 55.0, 1.0, mask)
        third = PaddleObservation("f000000003-o01", 3, .2, ObservationType.CONNECTED_WHOLE, (), 60.0, 1.0, mask)
        track = Tracklet.seed(1, TrackMeasurement(first.observation_id, 1, 0.0, 50.0, 1.0)).append_measurement(
            TrackMeasurement(second.observation_id, 2, .1, 55.0, 1.0), chain_span_px=100.0,
            kalman_config=KalmanConfig(chain_span_px=100.0), lifecycle_config=TrackLifecycleConfig(),
        ).append_measurement(
            TrackMeasurement(third.observation_id, 3, .2, 60.0, 1.0), chain_span_px=100.0,
            kalman_config=KalmanConfig(chain_span_px=100.0), lifecycle_config=TrackLifecycleConfig(),
        )
        result = _classify_placeholder_event(
            PhysicalEvent(1, (1,), .1), (track,),
            {item.observation_id: item for item in (first, second, third)}, coordinates, 2.5,
            _decision_config(minimum_spacing_frames=1, minimum_spacing_sec=.01),
        )
        self.assertEqual(result.status, FinalStatus.BROKEN_LEFT)


def _decision_config(*, minimum_spacing_frames: int, minimum_spacing_sec: float):
    return {
        "angle": {"side_threshold_deg": 15.0, "minimum_frames": 3, "maximum_mad_deg": 1.5, "decision_guard_deg": 0.0},
        "evidence": {"minimum_spacing_frames": minimum_spacing_frames, "minimum_spacing_sec": minimum_spacing_sec, "minimum_left_presence_bins": 2, "minimum_right_presence_bins": 2},
        "center_topology": {"minimum_connected_bins": 2, "minimum_disconnected_same_frame_bins": 2},
        "side_integrity": {"coverage_bins": 20, "valid_minimum_coverage_ratio": .85, "broken_minimum_internal_gap_ratio": .10, "minimum_evidence_bins": 2},
    }


if __name__ == "__main__":
    unittest.main()
