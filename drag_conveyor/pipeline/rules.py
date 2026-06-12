from __future__ import annotations

from dataclasses import dataclass

from ..config import CalibrationResult, RulesConfig


@dataclass(frozen=True, slots=True)
class RuleEvaluation:
    result: str
    score: float
    reasons: list[str]
    measurements: dict[str, float]
    thresholds: dict[str, float]
    margins: dict[str, float]
    hard_fail: bool


class RuleEngine:
    def evaluate(
        self,
        measurements: dict[str, float],
        rules: RulesConfig,
        calibration_result: CalibrationResult,
    ) -> RuleEvaluation:
        if rules.mode != "length_width_auto_baseline":
            raise ValueError(f"Unsupported rules.mode: {rules.mode}")

        reasons: list[str] = []

        length = float(measurements["length"])
        width = float(measurements["width"])

        length_min = _percentile(calibration_result.features["length"], rules.lower_percentile)
        length_max = _percentile(calibration_result.features["length"], rules.upper_percentile)
        width_min = _percentile(calibration_result.features["width"], rules.lower_percentile)
        width_max = _percentile(calibration_result.features["width"], rules.upper_percentile)

        thresholds: dict[str, float] = {
            "length_min": float(length_min),
            "length_max": float(length_max),
            "width_min": float(width_min),
            "width_max": float(width_max),
        }
        margins: dict[str, float] = {
            "length_margin": float(min(length - length_min, length_max - length)),
            "width_margin": float(min(width - width_min, width_max - width)),
        }
        violated_dimensions = 0

        if length < length_min:
            reasons.append("length_too_short")
            violated_dimensions += 1
        elif length > length_max:
            reasons.append("length_too_long")
            violated_dimensions += 1

        if width < width_min:
            reasons.append("width_too_small")
            violated_dimensions += 1
        elif width > width_max:
            reasons.append("width_too_large")
            violated_dimensions += 1

        score = violated_dimensions / 2.0
        result = "suspected_defect" if violated_dimensions >= 1 else "normal"
        return RuleEvaluation(
            result=result,
            score=score,
            reasons=reasons,
            measurements=measurements,
            thresholds=thresholds,
            margins=margins,
            hard_fail=result == "suspected_defect",
        )


def _percentile(stats, name: str) -> float:
    value = getattr(stats, name, None)
    if value is None:
        raise ValueError(f"Calibration feature stats missing percentile: {name}")
    return float(value)


__all__ = ["RuleEngine", "RuleEvaluation"]
