from __future__ import annotations

import unittest

import numpy as np

from drag_conveyor.geometry_v2.observation_builder import ObservationType, PaddleObservation
from drag_conveyor.geometry_v2.online_tracking import OnlineTrackManager, OnlineTrackingConfig
from drag_conveyor.geometry_v2.tracking import KalmanConfig
from drag_conveyor.geometry_v2.tracklets import TrackLifecycleConfig, TrackState


def _observation(frame_id: int, timestamp: float, anchor: float, kind: ObservationType, index: int = 1) -> PaddleObservation:
    return PaddleObservation(f"f{frame_id:09d}-o{index:02d}", frame_id, timestamp, kind, (), anchor, 1.0, np.zeros((1, 1), dtype=bool))


class OnlineTrackingTests(unittest.TestCase):
    def test_complementary_left_right_observations_stay_on_one_track(self) -> None:
        manager = OnlineTrackManager(
            config=OnlineTrackingConfig(chain_span_px=100.0),
            kalman_config=KalmanConfig(chain_span_px=100.0),
            lifecycle_config=TrackLifecycleConfig(),
        )
        first = manager.update((_observation(1, 0.0, 20.0, ObservationType.LEFT_ONLY),), timestamp_sec=0.0)
        second = manager.update((_observation(2, 0.1, 25.0, ObservationType.RIGHT_ONLY),), timestamp_sec=0.1)

        self.assertEqual(first.observation_track_ids["f000000001-o01"], second.observation_track_ids["f000000002-o01"])
        self.assertEqual(second.active_tracklets[0].state, TrackState.CONFIRMED)

    def test_order_preserving_association_does_not_swap_neighboring_tracks(self) -> None:
        manager = OnlineTrackManager(
            config=OnlineTrackingConfig(chain_span_px=100.0),
            kalman_config=KalmanConfig(chain_span_px=100.0),
            lifecycle_config=TrackLifecycleConfig(),
        )
        manager.update((_observation(1, 0.0, 20.0, ObservationType.LEFT_ONLY, 1), _observation(1, 0.0, 50.0, ObservationType.LEFT_ONLY, 2)), timestamp_sec=0.0)
        result = manager.update((_observation(2, 0.1, 24.0, ObservationType.RIGHT_ONLY, 1), _observation(2, 0.1, 54.0, ObservationType.RIGHT_ONLY, 2)), timestamp_sec=0.1)

        self.assertEqual(tuple(sorted(result.observation_track_ids.values())), (1, 2))

    def test_equal_cost_dp_paths_have_a_stable_tie_break(self) -> None:
        manager = OnlineTrackManager(
            config=OnlineTrackingConfig(chain_span_px=100.0),
            kalman_config=KalmanConfig(chain_span_px=100.0),
            lifecycle_config=TrackLifecycleConfig(),
        )
        manager.update((_observation(1, 0.0, 20.0, ObservationType.LEFT_ONLY),), timestamp_sec=0.0)
        result = manager.update((), timestamp_sec=0.1)
        self.assertEqual(result.active_tracklets[0].track_id, 1)


if __name__ == "__main__":
    unittest.main()
