from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class FinalStatus(StrEnum):
    NORMAL = "normal"
    BENT_LEFT = "bent_left"
    BENT_RIGHT = "bent_right"
    BENT_BOTH = "bent_both"
    BROKEN_LEFT = "broken_left"
    BROKEN_RIGHT = "broken_right"
    BROKEN_CENTER = "broken_center"
    UNCERTAIN = "uncertain"


class CenterState(StrEnum):
    INTACT = "intact"
    BROKEN_TOPOLOGICAL = "broken_topological"
    BROKEN_TEMPORAL = "broken_temporal"
    UNKNOWN = "unknown"
    CONFLICT = "conflict"


class SideState(StrEnum):
    VALID = "valid"
    BROKEN_LOCALIZED = "broken_localized"
    UNKNOWN = "unknown"
    CONFLICT = "conflict"


@dataclass(frozen=True, slots=True)
class EventEvidence:
    has_hard_identity_or_geometry_conflict: bool = False
    primary_conflict_reason: str = "conflicting_geometry_evidence"
    is_single_side_only: bool = False
    possible_breakage_statuses: tuple[FinalStatus, ...] = ()
    center: CenterState = CenterState.UNKNOWN
    left: SideState = SideState.UNKNOWN
    right: SideState = SideState.UNKNOWN
    has_positive_break_evidence: bool = False
    has_positive_or_temporal_break_suspicion: bool = False
    definitive_localized_left_break_with_both_sides_observed: bool = False
    definitive_localized_right_break_with_both_sides_observed: bool = False
    angle_enabled: bool = False
    angle_status: FinalStatus | None = None
    definitive_support_bins: int = 0


@dataclass(frozen=True, slots=True)
class Decision:
    status: FinalStatus
    primary_reason: str
    reason_codes: tuple[str, ...]
    suspected_breakage: bool
    possible_breakage_statuses: tuple[FinalStatus, ...]
    evidence_support_score: float
    confidence_semantics: str = "unavailable_until_calibrated"


def classify_event(evidence: EventEvidence) -> Decision:
    """Apply the V2 final-decision precedence without probabilistic fallback."""
    if evidence.has_hard_identity_or_geometry_conflict:
        return _uncertain(evidence.primary_conflict_reason, evidence.has_positive_break_evidence)
    if evidence.is_single_side_only:
        return _uncertain(
            "single_side_only_location_unidentifiable",
            True,
            evidence.possible_breakage_statuses,
        )
    if evidence.center == CenterState.CONFLICT or evidence.left == SideState.CONFLICT or evidence.right == SideState.CONFLICT:
        return _uncertain("conflicting_geometry_evidence", evidence.has_positive_break_evidence)
    if evidence.center in {CenterState.BROKEN_TOPOLOGICAL, CenterState.BROKEN_TEMPORAL}:
        if evidence.left == SideState.VALID and evidence.right == SideState.VALID:
            reason = (
                "center_disconnected_same_frame_multi_bin"
                if evidence.center == CenterState.BROKEN_TOPOLOGICAL
                else "center_disconnected_temporal_multi_bin"
            )
            return _definitive(FinalStatus.BROKEN_CENTER, reason, evidence.definitive_support_bins)
        return _uncertain("center_break_with_side_state_unresolved", True)
    if evidence.center == CenterState.INTACT:
        if evidence.left == SideState.BROKEN_LOCALIZED and evidence.right == SideState.VALID:
            return _definitive(FinalStatus.BROKEN_LEFT, "left_localized_internal_gap", evidence.definitive_support_bins)
        if evidence.left == SideState.VALID and evidence.right == SideState.BROKEN_LOCALIZED:
            return _definitive(FinalStatus.BROKEN_RIGHT, "right_localized_internal_gap", evidence.definitive_support_bins)
        if evidence.left == SideState.BROKEN_LOCALIZED and evidence.right == SideState.BROKEN_LOCALIZED:
            return _uncertain("both_sides_broken_no_canonical_label", True)
        if evidence.left != SideState.VALID or evidence.right != SideState.VALID:
            return _uncertain("side_integrity_unresolved", evidence.has_positive_break_evidence)
        if not evidence.angle_enabled:
            return _uncertain("model_capability_not_validated", False)
        if evidence.angle_status not in {FinalStatus.NORMAL, FinalStatus.BENT_LEFT, FinalStatus.BENT_RIGHT, FinalStatus.BENT_BOTH}:
            return _uncertain("insufficient_angle_frames", False)
        reasons = {
            FinalStatus.NORMAL: "normal_geometry_within_thresholds",
            FinalStatus.BENT_LEFT: "bent_left_side_angle",
            FinalStatus.BENT_RIGHT: "bent_right_side_angle",
            FinalStatus.BENT_BOTH: "bent_both_side_angles",
        }
        return _definitive(evidence.angle_status, reasons[evidence.angle_status], evidence.definitive_support_bins)
    if evidence.definitive_localized_left_break_with_both_sides_observed:
        return _definitive(FinalStatus.BROKEN_LEFT, "left_localized_internal_gap", evidence.definitive_support_bins)
    if evidence.definitive_localized_right_break_with_both_sides_observed:
        return _definitive(FinalStatus.BROKEN_RIGHT, "right_localized_internal_gap", evidence.definitive_support_bins)
    return _uncertain("center_topology_unresolved", evidence.has_positive_or_temporal_break_suspicion)


def _definitive(status: FinalStatus, reason: str, support_bins: int) -> Decision:
    return Decision(
        status=status,
        primary_reason=reason,
        reason_codes=(reason,),
        suspected_breakage=status in {FinalStatus.BROKEN_LEFT, FinalStatus.BROKEN_RIGHT, FinalStatus.BROKEN_CENTER},
        possible_breakage_statuses=(),
        evidence_support_score=min(1.0, support_bins / 3.0),
    )


def _uncertain(
    reason: str,
    suspected_breakage: bool,
    possibilities: tuple[FinalStatus, ...] = (),
) -> Decision:
    return Decision(
        status=FinalStatus.UNCERTAIN,
        primary_reason=reason,
        reason_codes=(reason,),
        suspected_breakage=suspected_breakage,
        possible_breakage_statuses=possibilities,
        evidence_support_score=0.0,
    )
