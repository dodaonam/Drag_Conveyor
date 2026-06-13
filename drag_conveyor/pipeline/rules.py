from __future__ import annotations

from dataclasses import dataclass

from ..config import AutoBaselineConfig, CalibrationResult, DefectPolicyConfig, LocalDefectConfig
from .local_defects import LocalDefectBaseline


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
        rules: AutoBaselineConfig,
        defect_policy: DefectPolicyConfig,
        calibration_result: CalibrationResult,
        *,
        local_defect_config: LocalDefectConfig | None = None,
        local_defect_baseline: LocalDefectBaseline | None = None,
    ) -> RuleEvaluation:
        geometry_reasons: list[str] = []
        local_reasons: list[str] = []

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
            geometry_reasons.append("length_too_short")
            violated_dimensions += 1
        elif length > length_max:
            geometry_reasons.append("length_too_long")
            violated_dimensions += 1

        if width < width_min:
            geometry_reasons.append("width_too_small")
            violated_dimensions += 1
        elif width > width_max:
            geometry_reasons.append("width_too_large")
            violated_dimensions += 1

        geometry_defect = violated_dimensions >= defect_policy.min_violated_dimensions
        geometry_score = violated_dimensions / float(defect_policy.score_dimension_count)

        use_local_defects = local_defect_config is not None and local_defect_config.enabled
        left_score = 0.0
        middle_score = 0.0
        right_score = 0.0
        color_abnormal_ratio = 0.0
        dark_pixel_ratio = 0.0
        color_delta_p95 = 0.0
        shape_score_norm = 0.0
        color_score_norm = 0.0

        if use_local_defects and local_defect_config is not None:
            left_score = float(measurements.get("left_shape_score", 0.0))
            middle_score = float(measurements.get("middle_shape_score", 0.0))
            right_score = float(measurements.get("right_shape_score", 0.0))
            left_weighted_pixels = float(measurements.get("left_defect_weighted_pixels", 0.0))
            middle_weighted_pixels = float(measurements.get("middle_defect_weighted_pixels", 0.0))
            right_weighted_pixels = float(measurements.get("right_defect_weighted_pixels", 0.0))
            alignment_low = float(measurements.get("local_alignment_low", 0.0)) >= 0.5

            left_bad = (
                left_score >= local_defect_config.shape_threshold
                and left_weighted_pixels >= local_defect_config.min_zone_defect_weighted_pixels
            )
            right_bad = (
                right_score >= local_defect_config.shape_threshold
                and right_weighted_pixels >= local_defect_config.min_zone_defect_weighted_pixels
            )
            middle_bad = (
                middle_score >= local_defect_config.middle_shape_threshold
                and middle_weighted_pixels >= local_defect_config.min_zone_defect_weighted_pixels
            )
            both_sides_bad = (
                left_score >= local_defect_config.both_sides_threshold
                and right_score >= local_defect_config.both_sides_threshold
                and left_weighted_pixels >= local_defect_config.min_zone_defect_weighted_pixels
                and right_weighted_pixels >= local_defect_config.min_zone_defect_weighted_pixels
            )

            if alignment_low:
                left_bad = left_score >= local_defect_config.severe_shape_threshold
                right_bad = right_score >= local_defect_config.severe_shape_threshold
                middle_bad = middle_score >= local_defect_config.severe_shape_threshold
                both_sides_bad = left_bad and right_bad

            if both_sides_bad:
                local_reasons.append("deform_both_sides")
                shape_score_norm = max(
                    shape_score_norm,
                    max(
                        left_score / max(local_defect_config.both_sides_threshold, 1e-6),
                        right_score / max(local_defect_config.both_sides_threshold, 1e-6),
                    ),
                )
            elif left_bad:
                local_reasons.append("deform_left")
                threshold = (
                    local_defect_config.severe_shape_threshold
                    if alignment_low
                    else local_defect_config.shape_threshold
                )
                shape_score_norm = max(shape_score_norm, left_score / max(threshold, 1e-6))
            elif right_bad:
                local_reasons.append("deform_right")
                threshold = (
                    local_defect_config.severe_shape_threshold
                    if alignment_low
                    else local_defect_config.shape_threshold
                )
                shape_score_norm = max(shape_score_norm, right_score / max(threshold, 1e-6))

            if middle_bad:
                local_reasons.append("deform_middle")
                threshold = (
                    local_defect_config.severe_shape_threshold
                    if alignment_low
                    else local_defect_config.middle_shape_threshold
                )
                shape_score_norm = max(shape_score_norm, middle_score / max(threshold, 1e-6))

            thresholds.update(
                {
                    "shape_threshold": local_defect_config.shape_threshold,
                    "middle_shape_threshold": local_defect_config.middle_shape_threshold,
                    "both_sides_threshold": local_defect_config.both_sides_threshold,
                    "severe_shape_threshold": local_defect_config.severe_shape_threshold,
                }
            )
            margins.update(
                {
                    "left_shape_margin": local_defect_config.shape_threshold - left_score,
                    "middle_shape_margin": local_defect_config.middle_shape_threshold - middle_score,
                    "right_shape_margin": local_defect_config.shape_threshold - right_score,
                }
            )

            if local_defect_config.color_enabled:
                color_abnormal_ratio = float(measurements.get("color_abnormal_ratio", 0.0))
                color_delta_p95 = float(measurements.get("color_delta_p95", 0.0))
                dark_pixel_ratio = float(measurements.get("dark_pixel_ratio", 0.0))

                effective_abnormal_ratio_threshold = local_defect_config.color_abnormal_ratio_threshold
                effective_dark_ratio_threshold = local_defect_config.dark_pixel_ratio_threshold
                abnormal_ratio_bad = color_abnormal_ratio >= effective_abnormal_ratio_threshold
                color_delta_bad = color_delta_p95 >= local_defect_config.color_delta_p95_threshold
                dark_ratio_bad = local_defect_config.dark_pixel_enabled and (
                    dark_pixel_ratio >= effective_dark_ratio_threshold
                )

                thresholds.update(
                    {
                        "color_abnormal_ratio_threshold": effective_abnormal_ratio_threshold,
                        "color_delta_p95_threshold": local_defect_config.color_delta_p95_threshold,
                        "dark_pixel_ratio_threshold": effective_dark_ratio_threshold,
                    }
                )
                margins.update(
                    {
                        "color_abnormal_ratio_margin": (
                            effective_abnormal_ratio_threshold - color_abnormal_ratio
                        ),
                        "dark_pixel_ratio_margin": effective_dark_ratio_threshold - dark_pixel_ratio,
                    }
                )

                if abnormal_ratio_bad:
                    color_score_norm = max(
                        color_score_norm,
                        color_abnormal_ratio / max(effective_abnormal_ratio_threshold, 1e-6),
                    )
                if color_delta_bad:
                    color_score_norm = max(
                        color_score_norm,
                        color_delta_p95 / max(local_defect_config.color_delta_p95_threshold, 1e-6),
                    )
                if dark_ratio_bad:
                    color_score_norm = max(
                        color_score_norm,
                        dark_pixel_ratio / max(effective_dark_ratio_threshold, 1e-6),
                    )

                color_bad = abnormal_ratio_bad or color_delta_bad or dark_ratio_bad
                if color_bad:
                    local_reasons.append("color_defect")

        reasons = geometry_reasons + local_reasons
        local_defect = bool(local_reasons)
        result = "suspected_defect" if geometry_defect or local_defect else "normal"

        score = min(1.0, max(geometry_score, shape_score_norm, color_score_norm))
        return RuleEvaluation(
            result=result,
            score=score,
            reasons=reasons,
            measurements=dict(measurements),
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
