from __future__ import annotations

import unittest

from drag_conveyor.geometry_v2.tracklets import TrackLifecycleConfig, TrackMeasurement, TrackState, Tracklet
from drag_conveyor.geometry_v2.tracking import KalmanConfig


def _measurement(frame: int, time: float, s: float) -> TrackMeasurement:
    return TrackMeasurement(f"f{frame}-o1", frame, time, s, 1.0)


class TrackletTests(unittest.TestCase):
    def test_seed_becomes_confirmed_on_second_independent_frame(self) -> None:
        track = Tracklet.seed(1, _measurement(1, 0.0, 10.0))
        track = track.append_measurement(
            _measurement(2, 0.1, 15.0),
            chain_span_px=100.0,
            kalman_config=KalmanConfig(chain_span_px=100.0),
            lifecycle_config=TrackLifecycleConfig(),
        )

        self.assertEqual(track.state, TrackState.CONFIRMED)
        self.assertIsNotNone(track.kalman)

    def test_tentative_gap_rejects_but_confirmed_gap_finalizes(self) -> None:
        config = TrackLifecycleConfig(maximum_track_gap_sec=0.2)
        kalman = KalmanConfig(chain_span_px=100.0)
        tentative = Tracklet.seed(1, _measurement(1, 0.0, 10.0)).miss(0.3, kalman_config=kalman, lifecycle_config=config)
        confirmed = Tracklet.seed(2, _measurement(1, 0.0, 10.0)).append_measurement(
            _measurement(2, 0.1, 15.0), chain_span_px=100.0, kalman_config=kalman, lifecycle_config=config
        ).miss(0.4, kalman_config=kalman, lifecycle_config=config)

        self.assertEqual(tentative.state, TrackState.REJECTED)
        self.assertEqual(confirmed.state, TrackState.FINALIZABLE)

    def test_lost_single_measurement_track_is_rejected_after_timeout(self) -> None:
        config = TrackLifecycleConfig(maximum_track_gap_sec=0.2)
        kalman = KalmanConfig(chain_span_px=100.0)
        track = Tracklet.seed(1, _measurement(1, 0.0, 10.0))
        lost = track.miss(0.1, kalman_config=kalman, lifecycle_config=config)
        timed_out = lost.miss(0.3, kalman_config=kalman, lifecycle_config=config)

        self.assertEqual(lost.state, TrackState.LOST)
        self.assertEqual(timed_out.state, TrackState.REJECTED)


if __name__ == "__main__":
    unittest.main()
