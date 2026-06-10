from __future__ import annotations

from dataclasses import dataclass

import numpy as np

AUTO_BASELINE_INSPECTION_MODE = "auto_baseline"
AVERAGE_RATIO_INSPECTION_MODE = "average_ratio"
DEFAULT_INSPECTION_MODE = AUTO_BASELINE_INSPECTION_MODE
SUPPORTED_INSPECTION_MODES = {
    AUTO_BASELINE_INSPECTION_MODE,
    AVERAGE_RATIO_INSPECTION_MODE,
}


def is_supported_inspection_mode(mode: str) -> bool:
    return mode in SUPPORTED_INSPECTION_MODES


@dataclass(frozen=True, slots=True)
class AverageRatioThresholds:
    width_min_ratio: float = 0.85
    width_max_ratio: float = 1.1
    length_min_ratio: float = 0.95
    length_max_ratio: float = 1.05


@dataclass(frozen=True, slots=True)
class AverageRatioDecision:
    result: str
    score: float
    reasons: list[str]
    thresholds: dict[str, float]
    margins: dict[str, float]
    average_length: float
    average_width: float


class AverageRatioInspector:
    """Standalone classifier for average-based defect checks."""

    def __init__(self, thresholds: AverageRatioThresholds | None = None) -> None:
        self.thresholds = thresholds or AverageRatioThresholds()

    def evaluate(self, measurements: dict[str, float], averages: dict[str, float]) -> AverageRatioDecision:
        length = float(measurements["length"])
        width = float(measurements["width"])
        average_length = float(averages["length"])
        average_width = float(averages["width"])

        if average_length <= 0 or average_width <= 0:
            raise ValueError("average length/width must be > 0")

        length_min = average_length * self.thresholds.length_min_ratio
        length_max = average_length * self.thresholds.length_max_ratio
        width_min = average_width * self.thresholds.width_min_ratio
        width_max = average_width * self.thresholds.width_max_ratio

        reasons: list[str] = []
        if length < length_min:
            reasons.append("length_too_short")
        elif length > length_max:
            reasons.append("length_too_long")

        if width < width_min:
            reasons.append("width_too_small")
        elif width > width_max:
            reasons.append("width_too_large")

        violated_dimensions = len(reasons)
        return AverageRatioDecision(
            result="suspected_defect" if violated_dimensions >= 1 else "normal",
            score=violated_dimensions / 2.0,
            reasons=reasons,
            thresholds={
                "length_avg": average_length,
                "length_min": length_min,
                "length_max": length_max,
                "width_avg": average_width,
                "width_min": width_min,
                "width_max": width_max,
            },
            margins={
                "length_margin": min(length - length_min, length_max - length),
                "width_margin": min(width - width_min, width_max - width),
            },
            average_length=average_length,
            average_width=average_width,
        )

    def compute_averages(self, measurements: list[dict[str, float]]) -> dict[str, float]:
        if not measurements:
            raise ValueError("measurements must not be empty")

        lengths = np.array([float(item["length"]) for item in measurements], dtype=np.float64)
        widths = np.array([float(item["width"]) for item in measurements], dtype=np.float64)

        if not np.isfinite(lengths).all() or not np.isfinite(widths).all():
            raise ValueError("measurements must be finite")

        average_length = float(lengths.mean())
        average_width = float(widths.mean())
        if average_length <= 0 or average_width <= 0:
            raise ValueError("average length/width must be > 0")

        return {
            "length": average_length,
            "width": average_width,
        }


__all__ = [
    "AUTO_BASELINE_INSPECTION_MODE",
    "AVERAGE_RATIO_INSPECTION_MODE",
    "DEFAULT_INSPECTION_MODE",
    "SUPPORTED_INSPECTION_MODES",
    "AverageRatioDecision",
    "AverageRatioInspector",
    "AverageRatioThresholds",
    "is_supported_inspection_mode",
]
