from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from drag_conveyor.app.batch import run_batch_inspection
from drag_conveyor.config import load_profile
from drag_conveyor.geometry_v2.decision import FinalStatus
from drag_conveyor.geometry_v2.fusion import PhysicalEvent
from drag_conveyor.geometry_v2.pipeline import GeometryEventResult, GeometryPipelineResult


ROOT = Path(__file__).resolve().parents[1]


class GeometryV2BatchSeamTests(unittest.TestCase):
    def test_geometry_mode_dispatches_before_legacy_runtime(self) -> None:
        profile = load_profile(ROOT / "config" / "base_profile.json")
        outcome = GeometryPipelineResult(
            success=True, failure_reason="", frames_scanned=3,
            events=(GeometryEventResult(PhysicalEvent(1, (7,), 0.5), FinalStatus.UNCERTAIN, ("model_capability_not_validated",), (1, 2, 3)),),
            model_hash="a" * 64, algorithm_config_hash="b" * 64, timestamp_source="decoder_pts",
        )
        with tempfile.NamedTemporaryFile(suffix=".mp4") as source:
            with mock.patch("drag_conveyor.geometry_v2.pipeline.run_geometry_v2_pipeline", return_value=outcome) as geometry, \
                 mock.patch("drag_conveyor.app.batch.CentroidTracker", side_effect=AssertionError("legacy tracker called")), \
                 mock.patch("drag_conveyor.app.batch.RuleEngine", side_effect=AssertionError("legacy rules called")), \
                 mock.patch("drag_conveyor.app.batch.VlmInspector", side_effect=AssertionError("VLM called")):
                result = run_batch_inspection(
                    profile=profile, source=source.name, run_id="geometry-seam", inspection_mode="geometry_v2",
                    geometry_input={"schema_version": "geometry_input/2.0", "chain_centerline": {"top": {"x": 640, "y": 0}, "bottom": {"x": 640, "y": 720}}},
                )
        geometry.assert_called_once()
        self.assertEqual(result.inspection_mode, "geometry_v2")
        self.assertEqual(result.vlm_request_count, 0)
        self.assertEqual(result.bars[0].vision_status, "uncertain")
        self.assertTrue(result.bars[0].review_required)

    def test_geometry_normal_stays_in_legacy_normal_bucket(self) -> None:
        profile = load_profile(ROOT / "config" / "base_profile.json")
        outcome = GeometryPipelineResult(True, "", 3, (GeometryEventResult(PhysicalEvent(1, (7,), .5), FinalStatus.NORMAL, ("angle_within_15_deg",), (1, 2)),), "a" * 64, "b" * 64, "decoder_pts")
        with tempfile.NamedTemporaryFile(suffix=".mp4") as source:
            with mock.patch("drag_conveyor.geometry_v2.pipeline.run_geometry_v2_pipeline", return_value=outcome):
                result = run_batch_inspection(profile=profile, source=source.name, run_id="geometry-normal", inspection_mode="geometry_v2", geometry_input={"schema_version": "geometry_input/2.0", "chain_centerline": {"top": {"x": 640, "y": 0}, "bottom": {"x": 640, "y": 720}}})
        self.assertEqual(result.normal_bars, 1)
        self.assertEqual(result.defect_bars, 0)
        self.assertEqual(result.bars[0].result, "normal")


if __name__ == "__main__":
    unittest.main()
