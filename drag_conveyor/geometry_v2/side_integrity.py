from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from .coordinates import ChainCoordinates


class SideIntegrity(StrEnum):
    VALID = "valid"
    BROKEN_LOCALIZED = "broken_localized"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class SideIntegrityResult:
    state: SideIntegrity
    coverage_ratio: float | None
    largest_internal_gap_ratio: float | None


def analyze_side_integrity(mask_roi: np.ndarray, coordinates: ChainCoordinates, *, side: str, chain_band_half_width: float, bins: int, valid_minimum_coverage_ratio: float, broken_minimum_internal_gap_ratio: float) -> SideIntegrityResult:
    if side not in {"left", "right"} or bins < 3:
        raise ValueError("Invalid side integrity arguments")
    ys, xs = np.nonzero(mask_roi)
    if not len(xs):
        return SideIntegrityResult(SideIntegrity.UNKNOWN, None, None)
    points = np.array([coordinates.project_pixel_center(int(x), int(y)) for x, y in zip(xs, ys)], dtype=np.float64)
    q = points[:, 1]
    outward = -q[q < -chain_band_half_width] if side == "left" else q[q > chain_band_half_width]
    if len(outward) < 2:
        return SideIntegrityResult(SideIntegrity.UNKNOWN, None, None)
    low, high = float(outward.min()), float(outward.max())
    if high <= low:
        return SideIntegrityResult(SideIntegrity.UNKNOWN, None, None)
    occupied = np.histogram(outward, bins=np.linspace(low, high, bins + 1))[0] > 0
    coverage = float(occupied.mean())
    runs = _largest_false_run(occupied)
    gap = runs / bins
    if gap >= broken_minimum_internal_gap_ratio and np.any(occupied[:1]) and np.any(occupied[-1:]):
        return SideIntegrityResult(SideIntegrity.BROKEN_LOCALIZED, coverage, gap)
    if coverage >= valid_minimum_coverage_ratio:
        return SideIntegrityResult(SideIntegrity.VALID, coverage, gap)
    return SideIntegrityResult(SideIntegrity.UNKNOWN, coverage, gap)


def _largest_false_run(values: np.ndarray) -> int:
    best = current = 0
    for value in values:
        current = 0 if value else current + 1
        best = max(best, current)
    return best
