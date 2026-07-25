from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class Point:
    """A continuous ROI-local point."""

    x: float
    y: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.x) or not math.isfinite(self.y):
            raise ValueError("Point coordinates must be finite")


@dataclass(frozen=True, slots=True)
class Roi:
    """An inspection ROI in full-frame coordinates."""

    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("ROI width and height must be positive")


@dataclass(frozen=True, slots=True)
class Centerline:
    """Canonical ROI-local chain centerline with its top endpoint first."""

    top: Point
    bottom: Point

    @classmethod
    def canonical(cls, first: Point, second: Point) -> "Centerline":
        if (first.y, first.x) <= (second.y, second.x):
            return cls(top=first, bottom=second)
        return cls(top=second, bottom=first)
