from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from drag_conveyor.geometry_v2.annotations import ANNOTATION_SCHEMA_VERSION, load_annotation_dataset
from drag_conveyor.geometry_v2.decision import FinalStatus


def _event(**changes):
    value = {
        "video_id": "xt015",
        "physical_paddle_id": 7,
        "entry_frame": 1,
        "entry_timestamp_sec": 0.0,
        "trigger_crossing_frame": 2,
        "trigger_crossing_timestamp_sec": .1,
        "exit_frame": 3,
        "exit_timestamp_sec": .2,
        "status": "broken_left",
        "visible_left": True,
        "visible_right": True,
        "center_visible": True,
        "annotator_ids": ["a1", "a2"],
        "adjudicated": True,
        "partial_boundary": False,
    }
    value.update(changes)
    return value


class AnnotationTests(unittest.TestCase):
    def test_loader_validates_event_level_schema_and_converts_to_ground_truth(self) -> None:
        raw = {"schema_version": ANNOTATION_SCHEMA_VERSION, "dataset_version": "xt015-v1", "frame_pts_table_sha256": "a" * 64, "events": [_event()]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "annotations.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            dataset = load_annotation_dataset(path)

        self.assertEqual(dataset.events[0].as_ground_truth().status, FinalStatus.BROKEN_LEFT)
        self.assertEqual(dataset.events[0].as_ground_truth().event_id, "xt015:7")

    def test_loader_rejects_uncertain_and_unadjudicated_defect(self) -> None:
        for event in (_event(status="uncertain"), _event(annotator_ids=["a1"], adjudicated=False)):
            raw = {"schema_version": ANNOTATION_SCHEMA_VERSION, "dataset_version": "xt015-v1", "frame_pts_table_sha256": "a" * 64, "events": [event]}
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "annotations.json"
                path.write_text(json.dumps(raw), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_annotation_dataset(path)


if __name__ == "__main__":
    unittest.main()
