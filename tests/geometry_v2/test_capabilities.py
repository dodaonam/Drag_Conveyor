from __future__ import annotations

import unittest
from types import SimpleNamespace

from drag_conveyor.geometry_v2.capabilities import build_candidate_capability_record


class CapabilityRecordTests(unittest.TestCase):
    def test_candidate_record_is_content_addressed_and_explicitly_unvalidated(self) -> None:
        model = SimpleNamespace(
            providers=["CPUExecutionProvider"],
            preprocess=SimpleNamespace(normalize="0_1", color_format="bgr"),
            postprocess=SimpleNamespace(confidence=0.5, iou=0.5),
        )
        manifest = {"artifact_manifest_id": "a" * 64}
        first = build_candidate_capability_record(artifact_manifest=manifest, model=model, algorithm_config_hash="b" * 64, rule_version="geometry_v2_rules/2.0.0")
        second = build_candidate_capability_record(artifact_manifest=manifest, model=model, algorithm_config_hash="b" * 64, rule_version="geometry_v2_rules/2.0.0")

        self.assertEqual(first["capability_record_hash"], second["capability_record_hash"])
        self.assertEqual(first["binding_state"], "unvalidated")
        self.assertFalse(any(first["production_enabled"].values()))
        self.assertEqual(first["runtime"]["fingerprint"], second["runtime"]["fingerprint"])


if __name__ == "__main__":
    unittest.main()
