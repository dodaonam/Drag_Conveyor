from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .coordinates import ChainCoordinates, mad, quantile_type7
from .decision import FinalStatus


@dataclass(frozen=True, slots=True)
class SideAxis:
    slope_q_per_s: float
    intercept_q: float
    projected_span_px: float
    median_residual_px: float
    sample_count: int

    @property
    def angle_deg(self) -> float:
        return math.degrees(math.atan(self.slope_q_per_s))


@dataclass(frozen=True, slots=True)
class AngleSummary:
    left_deg: float | None
    right_deg: float | None
    global_tilt_deg: float | None
    center_kink_deg: float | None
    status: FinalStatus | None


def fit_side_axis(mask_roi: np.ndarray, coordinates: ChainCoordinates, *, side: str, chain_band_half_width: float, minimum_pixels: int = 2) -> SideAxis | None:
    if side not in {"left", "right"}:
        raise ValueError("side must be left or right")
    ys, xs = np.nonzero(mask_roi)
    if not len(xs):
        return None
    points = np.array([coordinates.project_pixel_center(int(x), int(y)) for x, y in zip(xs, ys)], dtype=np.float64)
    keep = points[:, 1] < -chain_band_half_width if side == "left" else points[:, 1] > chain_band_half_width
    selected = points[keep]
    if len(selected) < minimum_pixels:
        return None
    s, q = selected[:, 0], selected[:, 1]
    design = np.column_stack((s, np.ones_like(s)))
    slope, intercept = np.linalg.lstsq(design, q, rcond=None)[0]
    residual = q - (slope * s + intercept)
    scale = max(1.0, 1.4826 * mad(residual))
    inlier = np.abs(residual) <= 3.0 * scale
    if int(inlier.sum()) < minimum_pixels:
        return None
    s, q = s[inlier], q[inlier]
    slope, intercept = np.linalg.lstsq(np.column_stack((s, np.ones_like(s))), q, rcond=None)[0]
    residual = q - (slope * s + intercept)
    return SideAxis(float(slope), float(intercept), quantile_type7(s, .95) - quantile_type7(s, .05), quantile_type7(np.abs(residual), .5), int(len(s)))


def classify_angles(*, left: SideAxis, right: SideAxis, side_threshold_deg: float, global_tilt_threshold_deg: float, center_kink_threshold_deg: float, decision_guard_deg: float) -> AngleSummary:
    left_deg, right_deg = left.angle_deg, right.angle_deg
    # Symmetric outward axes: average is the common global tilt; their difference is kink.
    global_tilt = (left_deg - right_deg) / 2.0
    center_kink = abs(left_deg + right_deg)
    values = (abs(left_deg), abs(right_deg), abs(global_tilt), center_kink)
    thresholds = (side_threshold_deg, side_threshold_deg, global_tilt_threshold_deg, center_kink_threshold_deg)
    if any(threshold - decision_guard_deg <= value < threshold + decision_guard_deg for value, threshold in zip(values, thresholds)):
        return AngleSummary(left_deg, right_deg, global_tilt, center_kink, None)
    left_bad = abs(left_deg) > side_threshold_deg
    right_bad = abs(right_deg) > side_threshold_deg
    if abs(global_tilt) > global_tilt_threshold_deg or center_kink > center_kink_threshold_deg or (left_bad and right_bad):
        status = FinalStatus.BENT_BOTH
    elif left_bad:
        status = FinalStatus.BENT_LEFT
    elif right_bad:
        status = FinalStatus.BENT_RIGHT
    else:
        status = FinalStatus.NORMAL
    return AngleSummary(left_deg, right_deg, global_tilt, center_kink, status)


def exceeds_transverse_angle(axis: SideAxis, *, threshold_deg: float = 15.0) -> bool:
    """Scale-free provisional rule: a visible wing is faulty only above 15°."""
    return abs(axis.angle_deg) > threshold_deg
