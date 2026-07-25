from __future__ import annotations

import unittest

from drag_conveyor.geometry_v2.acceptance import GroundTruthEvent, PredictedEvent, evaluate_classification, evaluate_events
from drag_conveyor.geometry_v2.acceptance_gates import evaluate_bootstrap_acceptance
from drag_conveyor.geometry_v2.decision import FinalStatus


class AcceptanceGateTests(unittest.TestCase):
    def test_empty_denominator_is_a_closed_gate_not_a_pass(self) -> None:
        events = evaluate_events((), (), maximum_crossing_delta_sec=.12)
        classifications = evaluate_classification((), (), events)
        report = evaluate_bootstrap_acceptance((), (), events, classifications)

        self.assertFalse(report.passed)
        self.assertTrue(all(not gate.passed for gate in report.gates))

    def test_exact_bound_rejects_tiny_perfect_sample(self) -> None:
        predictions = (PredictedEvent(1, 1.0, FinalStatus.NORMAL),)
        truth = (GroundTruthEvent("normal", 1.0, FinalStatus.NORMAL),)
        events = evaluate_events(predictions, truth, maximum_crossing_delta_sec=.12)
        classifications = evaluate_classification(predictions, truth, events)
        report = evaluate_bootstrap_acceptance(predictions, truth, events, classifications)
        precision = next(gate for gate in report.gates if gate.name == "event_precision")

        self.assertEqual(precision.observed, 1.0)
        self.assertFalse(precision.passed)
        self.assertLess(precision.bound, .99)


if __name__ == "__main__":
    unittest.main()
