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
    violated_soft_rules: list[str]


class RuleEngine:
    def evaluate(
        self,
        measurements: dict[str, float],
        rules: RulesConfig,
        calibration_result: CalibrationResult,
    ) -> RuleEvaluation:
        reasons: list[str] = []

        area = float(measurements["area"])
        length = float(measurements["length"])
        width = float(measurements["width"])
        aspect_ratio = float(measurements["aspect_ratio"])

        area_min = _lower(calibration_result.features["area"])
        area_max = _upper(calibration_result.features["area"])
        length_min = _lower(calibration_result.features["length"])
        length_max = _upper(calibration_result.features["length"])
        width_min = _lower(calibration_result.features["width"])
        width_max = _upper(calibration_result.features["width"])
        aspect_ratio_min = _lower(calibration_result.features["aspect_ratio"])
        aspect_ratio_max = _upper(calibration_result.features["aspect_ratio"])
        thresholds: dict[str, float] = {
            "area_min": float(area_min),
            "area_max": float(area_max),
            "length_min": float(length_min),
            "length_max": float(length_max),
            "width_min": float(width_min),
            "width_max": float(width_max),
            "aspect_ratio_min": float(aspect_ratio_min),
            "aspect_ratio_max": float(aspect_ratio_max),
        }
        margins: dict[str, float] = {
            "area_margin": float(min(area - area_min, area_max - area)),
            "length_margin": float(min(length - length_min, length_max - length)),
            "width_margin": float(min(width - width_min, width_max - width)),
            "aspect_ratio_margin": float(min(aspect_ratio - aspect_ratio_min, aspect_ratio_max - aspect_ratio)),
        }
        violated_soft_rules: list[str] = []

        if area < area_min:
            reasons.append("area_too_small")
            violated_soft_rules.append("area")
        elif area > area_max:
            reasons.append("area_too_large")
            violated_soft_rules.append("area")

        if length < length_min:
            reasons.append("length_too_short")
            violated_soft_rules.append("length")
        elif length > length_max:
            reasons.append("length_too_long")
            violated_soft_rules.append("length")

        if width < width_min:
            reasons.append("width_too_small")
            violated_soft_rules.append("width")
        elif width > width_max:
            reasons.append("width_too_large")
            violated_soft_rules.append("width")

        if aspect_ratio < aspect_ratio_min or aspect_ratio > aspect_ratio_max:
            reasons.append("aspect_ratio_out_of_range")
            violated_soft_rules.append("aspect_ratio")

        total_features = 4.0
        score = len(violated_soft_rules) / total_features
        result = "suspected_defect" if score >= rules.score_threshold else "normal"
        return RuleEvaluation(
            result=result,
            score=score,
            reasons=reasons,
            measurements=measurements,
            thresholds=thresholds,
            margins=margins,
            hard_fail=result == "suspected_defect",
            violated_soft_rules=violated_soft_rules,
        )


def _lower(stats) -> float:
    if getattr(stats, "p1", None) is not None:
        return float(stats.p1)
    return float(stats.p2 if stats.p2 is not None else stats.p5)


def _upper(stats) -> float:
    if getattr(stats, "p99", None) is not None:
        return float(stats.p99)
    return float(stats.p98 if stats.p98 is not None else stats.p95)


__all__ = ["RuleEngine", "RuleEvaluation"]
