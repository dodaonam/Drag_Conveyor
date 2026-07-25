from __future__ import annotations

import unittest

from drag_conveyor.geometry_v2.decision import CenterState, EventEvidence, FinalStatus, SideState, classify_event


class DecisionTests(unittest.TestCase):
    def test_single_side_always_remains_uncertain(self) -> None:
        decision = classify_event(
            EventEvidence(
                is_single_side_only=True,
                possible_breakage_statuses=(FinalStatus.BROKEN_CENTER, FinalStatus.BROKEN_RIGHT),
            )
        )

        self.assertEqual(decision.status, FinalStatus.UNCERTAIN)
        self.assertTrue(decision.suspected_breakage)
        self.assertEqual(decision.primary_reason, "single_side_only_location_unidentifiable")

    def test_center_break_requires_both_valid_sides(self) -> None:
        decision = classify_event(
            EventEvidence(center=CenterState.BROKEN_TOPOLOGICAL, left=SideState.VALID, right=SideState.VALID, definitive_support_bins=2)
        )

        self.assertEqual(decision.status, FinalStatus.BROKEN_CENTER)
        self.assertEqual(decision.evidence_support_score, 2 / 3)

    def test_both_localized_side_breaks_have_no_invented_status(self) -> None:
        decision = classify_event(
            EventEvidence(center=CenterState.INTACT, left=SideState.BROKEN_LOCALIZED, right=SideState.BROKEN_LOCALIZED)
        )

        self.assertEqual(decision.status, FinalStatus.UNCERTAIN)
        self.assertEqual(decision.primary_reason, "both_sides_broken_no_canonical_label")

    def test_angle_is_not_used_until_breakage_is_excluded_and_capability_enabled(self) -> None:
        blocked = classify_event(
            EventEvidence(center=CenterState.INTACT, left=SideState.VALID, right=SideState.VALID, angle_status=FinalStatus.BENT_LEFT)
        )
        allowed = classify_event(
            EventEvidence(
                center=CenterState.INTACT,
                left=SideState.VALID,
                right=SideState.VALID,
                angle_enabled=True,
                angle_status=FinalStatus.BENT_LEFT,
                definitive_support_bins=3,
            )
        )

        self.assertEqual(blocked.primary_reason, "model_capability_not_validated")
        self.assertEqual(allowed.status, FinalStatus.BENT_LEFT)
        self.assertEqual(allowed.evidence_support_score, 1.0)

    def test_unknown_center_allows_only_explicit_localized_break(self) -> None:
        decision = classify_event(EventEvidence(definitive_localized_right_break_with_both_sides_observed=True, definitive_support_bins=1))

        self.assertEqual(decision.status, FinalStatus.BROKEN_RIGHT)


if __name__ == "__main__":
    unittest.main()
