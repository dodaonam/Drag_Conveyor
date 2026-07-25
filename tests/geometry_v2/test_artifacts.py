from __future__ import annotations

import unittest
from pathlib import Path

from drag_conveyor.geometry_v2.artifacts import canonical_record_hash, read_onnx_artifact_manifest


ROOT = Path(__file__).resolve().parents[2]


class ArtifactManifestTests(unittest.TestCase):
    def test_current_onnx_manifest_is_content_addressed_and_describes_io(self) -> None:
        manifest = read_onnx_artifact_manifest(ROOT / "weights/model_imgsz_640/best.onnx", class_names={"0": "white_bar"})

        self.assertEqual(manifest["sha256"], "a3a8d3cad91ce8105f94864107318bc09924375fb924b5a3e790099a409c49e8")
        self.assertEqual(manifest["artifact_manifest_id"], canonical_record_hash(manifest, omit="artifact_manifest_id"))
        self.assertEqual(manifest["input"]["shape"], [1, 3, 640, 640])
        self.assertEqual(len(manifest["outputs"]), 2)


if __name__ == "__main__":
    unittest.main()
