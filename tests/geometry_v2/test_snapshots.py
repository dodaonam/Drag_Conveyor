from __future__ import annotations

import tempfile
import unittest
from unittest import mock
from pathlib import Path

import cv2
import numpy as np

from drag_conveyor.geometry_v2.coordinates import ChainCoordinates
from drag_conveyor.geometry_v2.decision import FinalStatus
from drag_conveyor.geometry_v2.frame_source import SourceFrame
from drag_conveyor.geometry_v2.fusion import PhysicalEvent
from drag_conveyor.geometry_v2.pipeline import GeometryEventResult, _write_event_snapshots
from drag_conveyor.geometry_v2.snapshots import geometry_snapshot_filename, render_geometry_snapshot
from drag_conveyor.geometry_v2.types import Point, Roi


class SnapshotTests(unittest.TestCase):
    def test_snapshot_filename_and_geometry_overlay_are_written(self) -> None:
        coordinates = ChainCoordinates.create(Roi(0, 0, 100, 100), Point(50, 0), Point(50, 100))
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / geometry_snapshot_filename(7, 13)
            render_geometry_snapshot(np.zeros((100, 100, 3), dtype=np.uint8), roi_xywh=(0, 0, 100, 100), coordinates=coordinates, trigger_strip=coordinates.trigger_strip(.5, .2), status="uncertain", output_path=target)
            rendered = cv2.imread(str(target))
        self.assertEqual(target.name, "track_000007_frame_000000013.jpg")
        self.assertIsNotNone(rendered)
        self.assertGreater(int(rendered.sum()), 0)


class EventSnapshotTests(unittest.TestCase):
    def test_selected_original_frame_writes_one_snapshot_per_event(self) -> None:
        coordinates = ChainCoordinates.create(Roi(0, 0, 100, 100), Point(50, 0), Point(50, 100))
        events = (
            GeometryEventResult(PhysicalEvent(1, (7,), 0.1), FinalStatus.NORMAL, (), (2,), 2),
            GeometryEventResult(PhysicalEvent(2, (8,), 0.1), FinalStatus.UNCERTAIN, (), (2,), 2),
        )
        frames = (SourceFrame(2, 0.1, np.zeros((100, 100, 3), dtype=np.uint8)),)
        with tempfile.TemporaryDirectory() as directory, mock.patch(
            "drag_conveyor.geometry_v2.pipeline.iter_original_frames", return_value=iter(frames)
        ):
            defects, normals = _write_event_snapshots(
                source="ignored.mp4", results=events, snapshots_root=directory, run_id="job", roi_xywh=(0, 0, 100, 100),
                coordinates=coordinates, trigger=coordinates.trigger_strip(.5, .2), jpeg_quality=92, strict=True,
                timestamp_epsilon_sec=1e-9,
            )
            self.assertTrue((normals / geometry_snapshot_filename(7, 2)).is_file())
            self.assertTrue((defects / geometry_snapshot_filename(8, 2)).is_file())


if __name__ == "__main__":
    unittest.main()
