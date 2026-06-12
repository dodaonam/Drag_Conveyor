from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from drag_conveyor.app import batch


class SnapshotOutputTests(unittest.TestCase):
    def test_defect_snapshot_contains_only_box_and_contour_overlays(self) -> None:
        frame = np.zeros((80, 100, 3), dtype=np.uint8)
        bar = batch.BarResult(
            frame_id=7,
            track_id=3,
            result="suspected_defect",
            score=1.0,
            reasons=["length_too_short"],
            measurements={"length": 10.0, "width": 2.0},
            thresholds={},
            margins={},
            bbox_frame_xyxy=(40.0, 30.0, 70.0, 50.0),
            contour_frame=np.array(
                [[[45.0, 35.0]], [[65.0, 35.0]], [[65.0, 45.0]], [[45.0, 45.0]]],
                dtype=np.float32,
            ),
            latency_ms=5.0,
            source_frame=frame.copy(),
        )
        captured: dict[str, object] = {}

        def fake_imwrite(path: str, image: np.ndarray) -> bool:
            captured["path"] = path
            captured["image"] = image.copy()
            return True

        with mock.patch.object(batch.cv2, "imwrite", side_effect=fake_imwrite):
            batch._save_box_contour_snapshot(frame, bar, Path("/tmp/snapshots"))

        image = captured["image"]
        assert isinstance(image, np.ndarray)
        self.assertTrue(np.any(np.all(image == [0, 0, 255], axis=2)), "bbox overlay missing")
        self.assertTrue(np.any(np.all(image == [0, 255, 0], axis=2)), "contour overlay missing")
        self.assertFalse(np.any(np.all(image == [255, 180, 0], axis=2)), "ROI overlay should not be drawn")
        self.assertFalse(np.any(np.all(image == [0, 255, 255], axis=2)), "trigger band overlay should not be drawn")

        red = np.all(image == [0, 0, 255], axis=2)
        expected_box_area = np.zeros(red.shape, dtype=bool)
        expected_box_area[27:54, 37:74] = True
        self.assertFalse(np.any(red & ~expected_box_area), "text overlay should not be drawn")

    def test_write_defect_snapshots_uses_captured_source_frame(self) -> None:
        frame = np.zeros((40, 40, 3), dtype=np.uint8)
        bar = batch.BarResult(
            frame_id=3,
            track_id=11,
            result="suspected_defect",
            score=1.0,
            reasons=["width_too_small"],
            measurements={"length": 5.0, "width": 1.0},
            thresholds={},
            margins={},
            bbox_frame_xyxy=(10.0, 10.0, 20.0, 20.0),
            contour_frame=np.array(
                [[[12.0, 12.0]], [[18.0, 12.0]], [[18.0, 18.0]], [[12.0, 18.0]]],
                dtype=np.float32,
            ),
            latency_ms=1.0,
            source_frame=frame.copy(),
        )

        with mock.patch.object(batch, "_save_box_contour_snapshot") as save_mock:
            batch._write_defect_snapshots([bar], Path("/tmp/snapshots"))

        save_mock.assert_called_once()
        saved_frame, saved_bar, saved_dir = save_mock.call_args.args
        self.assertTrue(np.array_equal(saved_frame, frame))
        self.assertIs(saved_bar, bar)
        self.assertEqual(saved_dir, Path("/tmp/snapshots"))


if __name__ == "__main__":
    unittest.main()
