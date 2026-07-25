from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from .analyzers import CenterTopology, analyze_center_bridge
from .coordinates import ChainCoordinates
from .observations import Component, SideHint
from .pairing import PairingConfig, pair_left_right_components


class ObservationType(StrEnum):
    CONNECTED_WHOLE = "connected_whole"
    DISCONNECTED_BOTH = "disconnected_both"
    LEFT_ONLY = "left_only"
    RIGHT_ONLY = "right_only"
    CENTER_ONLY = "center_only"
    MULTI_PADDLE_MERGED = "multi_paddle_merged"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True, slots=True)
class PaddleObservation:
    """A deterministic one-frame hypothesis, prior to temporal association."""

    observation_id: str
    source_frame_id: int
    source_timestamp_sec: float
    kind: ObservationType
    component_ids: tuple[str, ...]
    s_anchor: float
    s_anchor_sigma: float
    mask_roi: np.ndarray
    left_mask_roi: np.ndarray | None = None
    right_mask_roi: np.ndarray | None = None

    @property
    def has_left(self) -> bool:
        return self.kind in {ObservationType.CONNECTED_WHOLE, ObservationType.DISCONNECTED_BOTH, ObservationType.LEFT_ONLY}

    @property
    def has_right(self) -> bool:
        return self.kind in {ObservationType.CONNECTED_WHOLE, ObservationType.DISCONNECTED_BOTH, ObservationType.RIGHT_ONLY}


def build_frame_observations(
    components: tuple[Component, ...],
    *,
    source_timestamp_sec: float,
    coordinates: ChainCoordinates,
    chain_band_half_width: float,
    pairing_config: PairingConfig,
    q_bins: int,
    minimum_q_coverage: float,
) -> tuple[PaddleObservation, ...]:
    """Build safe one-frame observations without joining independent masks.

    A matched left/right pair is deliberately ``DISCONNECTED_BOTH``: a connection
    cannot be inferred by OR-ing two instance masks.  Only a single component
    spanning both sides can produce ``CONNECTED_WHOLE`` after raw bridge analysis.
    """
    if not components:
        return ()
    if len({component.source_frame_id for component in components}) != 1:
        raise ValueError("A frame observation set must contain exactly one source frame")
    pairing = pair_left_right_components(components, coordinates=coordinates, config=pairing_config)
    by_id = {component.component_id: component for component in components}
    consumed: set[str] = set()
    pending: list[tuple[ObservationType, tuple[Component, ...]]] = []
    ambiguous = pairing.ambiguous_component_ids

    for component_id in sorted(ambiguous):
        if component_id in consumed:
            continue
        consumed.add(component_id)
        pending.append((ObservationType.AMBIGUOUS, (by_id[component_id],)))

    for left_id, right_id in pairing.matched_pairs:
        if left_id in consumed or right_id in consumed:
            continue
        consumed.update((left_id, right_id))
        pending.append((ObservationType.DISCONNECTED_BOTH, (by_id[left_id], by_id[right_id])))

    for component in sorted(components, key=_component_key):
        if component.component_id in consumed:
            continue
        consumed.add(component.component_id)
        if component.side_hint == SideHint.LEFT:
            kind = ObservationType.LEFT_ONLY
        elif component.side_hint == SideHint.RIGHT:
            kind = ObservationType.RIGHT_ONLY
        elif component.side_hint == SideHint.SPANS_BOTH:
            bridge = analyze_center_bridge(
                component.mask_roi,
                coordinates,
                anchor_s=component.s_anchor,
                chain_band_half_width=chain_band_half_width,
                q_bins=q_bins,
                minimum_q_coverage=minimum_q_coverage,
            )
            kind = ObservationType.CONNECTED_WHOLE if bridge == CenterTopology.PRESENT else ObservationType.AMBIGUOUS
        else:
            kind = ObservationType.CENTER_ONLY
        pending.append((kind, (component,)))

    pending.sort(key=lambda item: (_anchor(item[1]), tuple(component.component_id for component in item[1])))
    frame_id = components[0].source_frame_id
    return tuple(
        PaddleObservation(
            observation_id=f"f{frame_id:09d}-o{index:02d}",
            source_frame_id=frame_id,
            source_timestamp_sec=source_timestamp_sec,
            kind=kind,
            component_ids=tuple(component.component_id for component in members),
            s_anchor=_anchor(members),
            s_anchor_sigma=max(component.s_anchor_sigma for component in members),
            mask_roi=_single_mask_or_empty(members),
            left_mask_roi=_side_mask(members, "left"),
            right_mask_roi=_side_mask(members, "right"),
        )
        for index, (kind, members) in enumerate(pending, start=1)
    )


def _component_key(component: Component) -> tuple[float, float, str]:
    return component.s_anchor, component.q_median, component.component_id


def _anchor(members: tuple[Component, ...]) -> float:
    return sum(component.s_anchor for component in members) / len(members)


def _single_mask_or_empty(members: tuple[Component, ...]) -> np.ndarray:
    if len(members) == 1:
        return members[0].mask_roi
    # Independent instances must never be unioned to fabricate center evidence.
    return np.zeros_like(members[0].mask_roi, dtype=bool)


def _side_mask(members: tuple[Component, ...], side: str) -> np.ndarray | None:
    """Keep source masks separate; paired instances must never be unioned."""
    if side not in {"left", "right"}:
        raise ValueError("side must be left or right")
    if len(members) == 1:
        component = members[0]
        if component.side_hint == SideHint.SPANS_BOTH:
            return component.mask_roi
        if (side == "left" and component.side_hint == SideHint.LEFT) or (side == "right" and component.side_hint == SideHint.RIGHT):
            return component.mask_roi
        return None
    matching = [component for component in members if (side == "left" and component.side_hint == SideHint.LEFT) or (side == "right" and component.side_hint == SideHint.RIGHT)]
    return matching[0].mask_roi if len(matching) == 1 else None
