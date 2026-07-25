from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .coordinates import TriggerStrip


class TriggerState(StrEnum):
    NOT_REACHED = "not_reached"
    IN_STRIP = "in_strip"
    TRIGGERED = "triggered"
    EVALUATED = "evaluated"


@dataclass(frozen=True, slots=True)
class TriggerUpdate:
    state: TriggerState
    crossing_timestamp_sec: float | None
    triggered_now: bool


def advance_trigger(
    state: TriggerState,
    *,
    previous_s: float | None,
    previous_timestamp_sec: float | None,
    current_s: float,
    current_timestamp_sec: float,
    strip: TriggerStrip,
) -> TriggerUpdate:
    """Detect a forward crossing once; evaluation is intentionally a later phase."""
    if state in {TriggerState.TRIGGERED, TriggerState.EVALUATED}:
        return TriggerUpdate(state, None, False)
    inside = strip.top_s <= current_s <= strip.bottom_s
    crossed = previous_s is not None and previous_s < strip.top_s <= current_s
    if not (inside or crossed):
        return TriggerUpdate(TriggerState.NOT_REACHED, None, False)
    crossing = current_timestamp_sec
    if crossed and previous_timestamp_sec is not None and current_s != previous_s:
        fraction = (strip.top_s - previous_s) / (current_s - previous_s)
        crossing = previous_timestamp_sec + fraction * (current_timestamp_sec - previous_timestamp_sec)
    return TriggerUpdate(TriggerState.TRIGGERED, crossing, True)
