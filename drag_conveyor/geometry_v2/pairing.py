from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import math

from .coordinates import ChainCoordinates
from .observations import AnchorQuality, Component, SideHint


class PairingOperationKind(StrEnum):
    MATCH = "match"
    LEFT_UNMATCHED = "left_unmatched"
    RIGHT_UNMATCHED = "right_unmatched"


@dataclass(frozen=True, slots=True)
class PairingConfig:
    same_frame_anchor_gate_ratio: float = 0.03
    pairing_uncertainty_weight: float = 0.25
    unmatched_cost: float = 0.05
    pairing_ambiguity_margin: float = 0.01
    cost_quantization: float = 0.000000001


@dataclass(frozen=True, slots=True)
class PairingOperation:
    kind: PairingOperationKind
    left_component_id: str | None
    right_component_id: str | None

    @property
    def lexical_key(self) -> tuple[str, str, str]:
        return (self.kind.value, self.left_component_id or "", self.right_component_id or "")


@dataclass(frozen=True, slots=True)
class PairingPath:
    operations: tuple[PairingOperation, ...]
    cost_int: int

    @property
    def match_count(self) -> int:
        return sum(operation.kind == PairingOperationKind.MATCH for operation in self.operations)

    @property
    def unmatched_count(self) -> int:
        return len(self.operations) - self.match_count

    @property
    def rank_key(self) -> tuple[int, int, int, tuple[tuple[str, str, str], ...]]:
        return (
            self.cost_int,
            -self.match_count,
            self.unmatched_count,
            tuple(operation.lexical_key for operation in self.operations),
        )


@dataclass(frozen=True, slots=True)
class PairingResult:
    best_path: PairingPath
    second_path: PairingPath | None
    ambiguous_component_ids: frozenset[str]

    @property
    def matched_pairs(self) -> tuple[tuple[str, str], ...]:
        ambiguous = self.ambiguous_component_ids
        return tuple(
            (operation.left_component_id, operation.right_component_id)
            for operation in self.best_path.operations
            if operation.kind == PairingOperationKind.MATCH
            and operation.left_component_id not in ambiguous
            and operation.right_component_id not in ambiguous
        )  # type: ignore[arg-type]


def pair_left_right_components(
    components: tuple[Component, ...],
    *,
    coordinates: ChainCoordinates,
    config: PairingConfig,
) -> PairingResult:
    """Find deterministic, order-preserving left/right matches in one frame."""
    if config.cost_quantization <= 0.0:
        raise ValueError("cost_quantization must be positive")
    left = tuple(sorted((component for component in components if component.side_hint == SideHint.LEFT), key=_component_order))
    right = tuple(sorted((component for component in components if component.side_hint == SideHint.RIGHT), key=_component_order))
    grid: list[list[tuple[PairingPath, ...]]] = [[() for _ in range(len(right) + 1)] for _ in range(len(left) + 1)]
    grid[0][0] = (PairingPath((), 0),)

    unmatched_int = _quantize(config.unmatched_cost, config.cost_quantization)
    for left_index in range(len(left) + 1):
        for right_index in range(len(right) + 1):
            if left_index == 0 and right_index == 0:
                continue
            candidates: list[PairingPath] = []
            if left_index:
                operation = PairingOperation(PairingOperationKind.LEFT_UNMATCHED, left[left_index - 1].component_id, None)
                candidates.extend(_append(path, operation, unmatched_int) for path in grid[left_index - 1][right_index])
            if right_index:
                operation = PairingOperation(PairingOperationKind.RIGHT_UNMATCHED, None, right[right_index - 1].component_id)
                candidates.extend(_append(path, operation, unmatched_int) for path in grid[left_index][right_index - 1])
            if left_index and right_index and _eligible(left[left_index - 1], right[right_index - 1], coordinates, config):
                operation = PairingOperation(
                    PairingOperationKind.MATCH,
                    left[left_index - 1].component_id,
                    right[right_index - 1].component_id,
                )
                cost_int = _quantize(_pair_cost(left[left_index - 1], right[right_index - 1], coordinates, config), config.cost_quantization)
                candidates.extend(_append(path, operation, cost_int) for path in grid[left_index - 1][right_index - 1])
            grid[left_index][right_index] = _best_two_distinct(candidates)

    best, *rest = grid[-1][-1]
    second = rest[0] if rest else None
    ambiguous_ids: frozenset[str] = frozenset()
    if second is not None and (second.cost_int - best.cost_int) * config.cost_quantization < config.pairing_ambiguity_margin:
        ambiguous_ids = frozenset(_symmetric_difference_component_ids(best, second))
    return PairingResult(best_path=best, second_path=second, ambiguous_component_ids=ambiguous_ids)


def _component_order(component: Component) -> tuple[float, float, str]:
    return (component.s_anchor, component.q_median, component.component_id)


def _eligible(left: Component, right: Component, coordinates: ChainCoordinates, config: PairingConfig) -> bool:
    return (
        left.anchor_quality == AnchorQuality.OK
        and right.anchor_quality == AnchorQuality.OK
        and abs(left.s_anchor - right.s_anchor) <= config.same_frame_anchor_gate_ratio * coordinates.span
    )


def _pair_cost(left: Component, right: Component, coordinates: ChainCoordinates, config: PairingConfig) -> float:
    return (
        abs(left.s_anchor - right.s_anchor) / coordinates.span
        + config.pairing_uncertainty_weight * math.hypot(left.s_anchor_sigma, right.s_anchor_sigma) / coordinates.span
    )


def _quantize(cost: float, quantization: float) -> int:
    return round(cost / quantization)


def _append(path: PairingPath, operation: PairingOperation, increment: int) -> PairingPath:
    return PairingPath(operations=(*path.operations, operation), cost_int=path.cost_int + increment)


def _best_two_distinct(paths: list[PairingPath]) -> tuple[PairingPath, ...]:
    unique = {path.operations: path for path in paths}
    return tuple(sorted(unique.values(), key=lambda path: path.rank_key)[:2])


def _symmetric_difference_component_ids(first: PairingPath, second: PairingPath) -> set[str]:
    first_operations = set(first.operations)
    second_operations = set(second.operations)
    return {
        component_id
        for operation in first_operations.symmetric_difference(second_operations)
        for component_id in (operation.left_component_id, operation.right_component_id)
        if component_id is not None
    }
