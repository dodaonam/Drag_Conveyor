from __future__ import annotations

from dataclasses import dataclass

from ..config import AutoBaselineConfig, CalibrationResult, DefectPolicyConfig


@dataclass(frozen=True, slots=True)
class RuleEvaluation:
    result: str
    score: float
    reasons: list[str]
    thresholds: dict[str, float]
    margins: dict[str, float]


class RuleEngine:
    def evaluate(
        self,
        measurements: dict[str, float],
        rules: AutoBaselineConfig,
        defect_policy: DefectPolicyConfig,
        calibration_result: CalibrationResult,
    ) -> RuleEvaluation:
        reasons: list[str] = []

        length = float(measurements["length"])
        width = float(measurements["width"])

        length_min = _percentile(calibration_result.features["length"], rules.length_lower_percentile) * (1.0 - rules.length_lower_margin)
        length_max = _percentile(calibration_result.features["length"], rules.length_upper_percentile) * (1.0 + rules.length_upper_margin)
        width_min = _percentile(calibration_result.features["width"], rules.width_lower_percentile) * (1.0 - rules.width_lower_margin)
        width_max = _percentile(calibration_result.features["width"], rules.width_upper_percentile) * (1.0 + rules.width_upper_margin)

        # Deadband chống nhiễu đo: chỉ coi là vi phạm khi vượt biên một khoảng tối thiểu
        # tolerance = max(floor, |bound| * ratio). Loại dao động nhỏ do segmentation,
        # vẫn giữ lỗi hình học rõ. Áp dụng cho cả bốn biên.
        length_min_tol = _tolerance(length_min, defect_policy)
        length_max_tol = _tolerance(length_max, defect_policy)
        width_min_tol = _tolerance(width_min, defect_policy)
        width_max_tol = _tolerance(width_max, defect_policy)

        thresholds: dict[str, float] = {
            "length_min": float(length_min),
            "length_max": float(length_max),
            "width_min": float(width_min),
            "width_max": float(width_max),
            "length_min_tol": float(length_min_tol),
            "length_max_tol": float(length_max_tol),
            "width_min_tol": float(width_min_tol),
            "width_max_tol": float(width_max_tol),
        }
        margins: dict[str, float] = {
            "length_margin": float(min(length - length_min, length_max - length)),
            "width_margin": float(min(width - width_min, width_max - width)),
        }
        violated_dimensions = 0

        if length < length_min - length_min_tol:
            reasons.append("length_too_short")
            violated_dimensions += 1
        elif length > length_max + length_max_tol:
            reasons.append("length_too_long")
            violated_dimensions += 1

        if width < width_min - width_min_tol:
            reasons.append("width_too_small")
            violated_dimensions += 1
        elif width > width_max + width_max_tol:
            reasons.append("width_too_large")
            violated_dimensions += 1

        score = violated_dimensions / float(defect_policy.score_dimension_count)
        result = (
            "suspected_defect"
            if violated_dimensions >= defect_policy.min_violated_dimensions
            else "normal"
        )
        return RuleEvaluation(
            result=result,
            score=score,
            reasons=reasons,
            thresholds=thresholds,
            margins=margins,
        )


def _percentile(stats, name: str) -> float:
    value = getattr(stats, name, None)
    if value is None:
        raise ValueError(f"Calibration feature stats missing percentile: {name}")
    return float(value)


def _tolerance(bound: float, defect_policy: DefectPolicyConfig) -> float:
    """Deadband cho một biên: max(floor, |bound| * ratio)."""
    return max(defect_policy.deadband_floor, abs(float(bound)) * defect_policy.deadband_ratio)


__all__ = ["RuleEngine", "RuleEvaluation"]
