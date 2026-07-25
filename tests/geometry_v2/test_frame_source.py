from __future__ import annotations

from itertools import islice
from pathlib import Path
import unittest

from drag_conveyor.geometry_v2.frame_source import iter_original_frames


ROOT = Path(__file__).resolve().parents[2]


class FrameSourceTests(unittest.TestCase):
    def test_raw_video_has_monotonic_pts_and_one_based_frame_ids(self) -> None:
        frames = list(islice(iter_original_frames(ROOT / "data" / "raw_data" / "vid_1.mp4"), 3))

        self.assertEqual([frame.source_frame_id for frame in frames], [1, 2, 3])
        self.assertTrue(all(frame.is_original for frame in frames))
        self.assertTrue(all(frame.timestamp_source == "decoder_pts" for frame in frames))
        self.assertLess(frames[0].source_timestamp_sec, frames[1].source_timestamp_sec)
        self.assertEqual(frames[0].image_bgr.shape[2], 3)


if __name__ == "__main__":
    unittest.main()
