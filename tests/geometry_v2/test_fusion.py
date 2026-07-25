from __future__ import annotations

import unittest

from drag_conveyor.geometry_v2.fusion import FusionConfig, fuse_tracklets
from drag_conveyor.geometry_v2.tracklets import TrackMeasurement, Tracklet
from drag_conveyor.geometry_v2.tracking import KalmanConfig
from drag_conveyor.geometry_v2.tracklets import TrackLifecycleConfig


def _track(track_id: int, first_time: float, first_s: float, second_time: float, second_s: float) -> Tracklet:
    track = Tracklet.seed(track_id, TrackMeasurement(f"{track_id}-a", 1, first_time, first_s, 1.0))
    return track.append_measurement(TrackMeasurement(f"{track_id}-b", 2, second_time, second_s, 1.0), chain_span_px=100.0, kalman_config=KalmanConfig(chain_span_px=100.0), lifecycle_config=TrackLifecycleConfig())


class FusionTests(unittest.TestCase):
    def test_complementary_tracklets_with_same_crossing_fuse(self) -> None:
        events = fuse_tracklets((_track(2, 0.0, 46, 0.1, 54), _track(1, 0.01, 48, 0.11, 56)), config=FusionConfig(trigger_center_s=50))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].track_ids, (1, 2))
        self.assertEqual(events[0].paddle_id, 1)

    def test_neighboring_crossings_do_not_cross_merge(self) -> None:
        events = fuse_tracklets((_track(1, 0.0, 46, 0.1, 54), _track(2, 0.5, 46, 0.6, 54)), config=FusionConfig(trigger_center_s=50))
        self.assertEqual([event.track_ids for event in events], [(1,), (2,)])

    def test_invalid_tracklet_does_not_discard_valid_event(self) -> None:
        invalid = Tracklet.seed(99, TrackMeasurement("99-a", 1, 0.0, 10.0, 1.0))
        events = fuse_tracklets((_track(1, 0.0, 46, 0.1, 54), invalid), config=FusionConfig(trigger_center_s=50))

        self.assertEqual([event.track_ids for event in events], [(1,)])


if __name__ == "__main__":
    unittest.main()
