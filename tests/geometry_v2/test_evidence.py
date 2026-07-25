from __future__ import annotations

import unittest

import numpy as np

from drag_conveyor.geometry_v2.evidence import BoundedEvidenceStore, EvidenceConfig, EvidenceSample, EvidenceType
from drag_conveyor.geometry_v2.observation_builder import ObservationType, PaddleObservation
from drag_conveyor.geometry_v2.triggering import TriggerState, advance_trigger
from drag_conveyor.geometry_v2.coordinates import TriggerStrip


class EvidenceTests(unittest.TestCase):
    def test_evidence_is_decorrelated_and_synthetic_frames_never_vote(self) -> None:
        store = BoundedEvidenceStore(EvidenceConfig(minimum_spacing_frames=2, minimum_spacing_sec=0.05))
        store.add(EvidenceSample(EvidenceType.CONNECTED, 10, 1.0, "a", 1.0))
        store.add(EvidenceSample(EvidenceType.CONNECTED, 11, 1.1, "b", 1.0))
        store.add(EvidenceSample(EvidenceType.CONNECTED, 12, 1.2, "c", 1.0, is_original=False))
        store.add(EvidenceSample(EvidenceType.CONNECTED, 12, 1.2, "d", 1.0))
        self.assertEqual([sample.observation_id for sample in store.samples(EvidenceType.CONNECTED)], ["a", "d"])

    def test_trigger_is_not_evaluation_and_crossing_time_interpolates(self) -> None:
        update = advance_trigger(TriggerState.NOT_REACHED, previous_s=40, previous_timestamp_sec=1.0, current_s=60, current_timestamp_sec=1.2, strip=TriggerStrip(50, 70))
        self.assertEqual(update.state, TriggerState.TRIGGERED)
        self.assertTrue(update.triggered_now)
        self.assertAlmostEqual(update.crossing_timestamp_sec or 0, 1.1)


if __name__ == "__main__":
    unittest.main()
