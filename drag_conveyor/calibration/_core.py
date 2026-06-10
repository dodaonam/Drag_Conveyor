from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from ..config import CalibrationResult, FeatureStats, Profile

CRITICAL_FEATURES = ["length", "width"]
ALL_FEATURES = ["length", "width"]


@dataclass(frozen=True, slots=True)
class CalibrationOutcome:
    success: bool
    reason: str
    calibration_result: CalibrationResult | None
    updated_profile: Profile | None


class CalibrationEngine:
    def calibrate(self, records: list[dict[str, float]], profile: Profile) -> CalibrationOutcome:
        cfg = profile.calibration
        valid = [r for r in records if _record_has_all_features(r)]

        if len(valid) < cfg.min_valid_records:
            return CalibrationOutcome(
                success=False,
                reason=(
                    f"valid_records={len(valid)} below min_valid_records={cfg.min_valid_records}; "
                    "baseline not stable"
                ),
                calibration_result=None,
                updated_profile=None,
            )

        outlier_mask = _compute_outlier_mask(valid)
        outliers = [r for r, is_out in zip(valid, outlier_mask) if is_out]
        inliers = [r for r, is_out in zip(valid, outlier_mask) if not is_out]

        valid_records = len(valid)
        inlier_count = len(inliers)
        outlier_count = len(outliers)
        inlier_ratio = inlier_count / valid_records if valid_records else 0.0
        outlier_ratio = outlier_count / valid_records if valid_records else 1.0

        if inlier_count < cfg.min_valid_records:
            return CalibrationOutcome(
                success=False,
                reason=(
                    f"inlier_count={inlier_count} below min_valid_records={cfg.min_valid_records}; "
                    "baseline not stable"
                ),
                calibration_result=None,
                updated_profile=None,
            )

        if inlier_ratio < cfg.min_inlier_ratio:
            return CalibrationOutcome(
                success=False,
                reason=(
                    f"inlier_ratio={inlier_ratio:.3f} below min_inlier_ratio={cfg.min_inlier_ratio:.3f}; "
                    "baseline not stable"
                ),
                calibration_result=None,
                updated_profile=None,
            )

        if outlier_ratio > cfg.max_outlier_ratio:
            return CalibrationOutcome(
                success=False,
                reason=(
                    f"outlier_ratio={outlier_ratio:.3f} above max_outlier_ratio={cfg.max_outlier_ratio:.3f}; "
                    "baseline not stable"
                ),
                calibration_result=None,
                updated_profile=None,
            )

        feature_stats = {name: _feature_stats([row[name] for row in inliers]) for name in ALL_FEATURES}

        now = datetime.now().isoformat(timespec="seconds")
        result = CalibrationResult(
            created_at=now,
            rules_updated_at=now,
            rules_version="geometry_v1",
            sample_count=len(records),
            valid_records=valid_records,
            inlier_count=inlier_count,
            outlier_count=outlier_count,
            inlier_ratio=inlier_ratio,
            thresholds_source="auto_baseline_median_mad_p1_p99",
            features=feature_stats,
        )

        updated = _apply_calibration_to_profile(profile, result)
        return CalibrationOutcome(success=True, reason="ok", calibration_result=result, updated_profile=updated)


def _record_has_all_features(record: dict[str, float]) -> bool:
    try:
        for name in ALL_FEATURES:
            val = float(record[name])
            if not np.isfinite(val):
                return False
    except (KeyError, TypeError, ValueError):
        return False
    return True


def _compute_outlier_mask(records: list[dict[str, float]]) -> list[bool]:
    mask = [False] * len(records)

    for name in CRITICAL_FEATURES:
        values = np.array([float(r[name]) for r in records], dtype=np.float64)
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))

        if mad > 1e-12:
            z = 0.6745 * (values - median) / mad
            feature_out = np.abs(z) > 3.5
        else:
            q1 = float(np.percentile(values, 25))
            q3 = float(np.percentile(values, 75))
            iqr = q3 - q1
            if iqr <= 1e-12:
                feature_out = np.zeros_like(values, dtype=bool)
            else:
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                feature_out = (values < lower) | (values > upper)

        for idx, out in enumerate(feature_out.tolist()):
            if out:
                mask[idx] = True

    return mask


def _feature_stats(values: list[float]) -> FeatureStats:
    arr = np.array(values, dtype=np.float64)
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    p1 = float(np.percentile(arr, 1))
    p2 = float(np.percentile(arr, 2))
    p5 = float(np.percentile(arr, 5))
    p95 = float(np.percentile(arr, 95))
    p98 = float(np.percentile(arr, 98))
    p99 = float(np.percentile(arr, 99))
    return FeatureStats(median=median, mad=mad, p5=p5, p95=p95, p1=p1, p99=p99, p2=p2, p98=p98)


def _apply_calibration_to_profile(profile: Profile, result: CalibrationResult) -> Profile:
    updated = copy.deepcopy(profile)
    updated.calibration_result = result
    return updated


def _lower(stats: FeatureStats) -> float:
    if stats.p1 is not None:
        return stats.p1
    return stats.p2 if stats.p2 is not None else stats.p5


def _upper(stats: FeatureStats) -> float:
    if stats.p99 is not None:
        return stats.p99
    return stats.p98 if stats.p98 is not None else stats.p95
