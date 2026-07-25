from __future__ import annotations

import unittest

from drag_conveyor.geometry_v2.acceptance import GroundTruthEvent, PredictedEvent, evaluate_classification, evaluate_events
from drag_conveyor.geometry_v2.decision import FinalStatus


class AcceptanceTests(unittest.TestCase):
    def test_matching_is_one_to_one_and_reports_miss_extra_and_status_pairs(self) -> None:
        metrics = evaluate_events(
            (
                PredictedEvent(1, 1.02, FinalStatus.NORMAL),
                PredictedEvent(2, 2.01, FinalStatus.BENT_LEFT),
                PredictedEvent(3, 5.0, FinalStatus.UNCERTAIN),
            ),
            (
                GroundTruthEvent("a", 1.0, FinalStatus.NORMAL),
                GroundTruthEvent("b", 2.0, FinalStatus.BENT_RIGHT),
                GroundTruthEvent("c", 3.0, FinalStatus.NORMAL),
            ),
            maximum_crossing_delta_sec=.05,
        )
        self.assertEqual((metrics.matched_events, metrics.missed_events, metrics.extra_events), (2, 1, 1))
        self.assertAlmostEqual(metrics.precision, 2 / 3)
        self.assertAlmostEqual(metrics.recall, 2 / 3)
        self.assertEqual(metrics.status_pairs, (("normal", "normal"), ("bent_right", "bent_left")))

    def test_local_ground_truth_interval_tightens_match_gate(self) -> None:
        metrics = evaluate_events(
            (PredictedEvent(1, .05, FinalStatus.NORMAL),),
            (GroundTruthEvent("a", 0.0, FinalStatus.NORMAL), GroundTruthEvent("b", .10, FinalStatus.NORMAL)),
            maximum_crossing_delta_sec=.12,
        )

        self.assertEqual(metrics.matched_events, 0)
        self.assertEqual(metrics.missed_events, 2)

    def test_partial_boundary_truth_is_excluded_and_extra_in_gate_is_duplicate(self) -> None:
        metrics = evaluate_events(
            (PredictedEvent(1, 1.0, FinalStatus.NORMAL), PredictedEvent(2, 1.01, FinalStatus.NORMAL)),
            (GroundTruthEvent("partial", .2, FinalStatus.NORMAL, partial_boundary=True), GroundTruthEvent("a", 1.0, FinalStatus.NORMAL)),
            maximum_crossing_delta_sec=.12,
        )

        self.assertEqual((metrics.matched_events, metrics.missed_events, metrics.extra_events, metrics.duplicate_events), (1, 0, 1, 1))

    def test_classification_keeps_uncertain_as_abstention_but_counts_safety_alert(self) -> None:
        predictions = (
            PredictedEvent(1, 1.0, FinalStatus.UNCERTAIN, suspected_breakage=True),
            PredictedEvent(2, 2.0, FinalStatus.NORMAL),
        )
        truth = (
            GroundTruthEvent("broken", 1.0, FinalStatus.BROKEN_LEFT),
            GroundTruthEvent("normal", 2.0, FinalStatus.NORMAL),
        )
        events = evaluate_events(predictions, truth, maximum_crossing_delta_sec=.12)
        metrics = evaluate_classification(predictions, truth, events)

        self.assertEqual((metrics.definitive_events, metrics.uncertain_events), (1, 1))
        self.assertEqual(metrics.coverage, .5)
        self.assertEqual(metrics.selective_exact_accuracy, 1.0)
        self.assertEqual(metrics.broken_safety_alert_recall, 1.0)
        self.assertEqual(metrics.dangerous_false_normal_rate, 0.0)


if __name__ == "__main__":
    unittest.main()
