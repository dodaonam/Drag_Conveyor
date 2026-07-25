from __future__ import annotations

from enum import StrEnum

import numpy as np

from .coordinates import ChainCoordinates


class CenterTopology(StrEnum):
    PRESENT = "present"
    ABSENT = "absent"
    UNKNOWN = "unknown"


def analyze_center_bridge(
    mask_roi: np.ndarray,
    coordinates: ChainCoordinates,
    *,
    anchor_s: float,
    chain_band_half_width: float,
    q_bins: int = 20,
    minimum_q_coverage: float = 0.90,
) -> CenterTopology:
    """Classify same-frame center connectivity without morphology or mask union."""
    if mask_roi.shape != (coordinates.roi.height, coordinates.roi.width):
        raise ValueError("Mask dimensions must match the inspection ROI")
    if q_bins < 2 or not 0.0 < minimum_q_coverage <= 1.0:
        raise ValueError("Invalid center topology parameters")
    ys, xs = np.nonzero(mask_roi)
    if not len(xs):
        return CenterTopology.UNKNOWN
    points_s, points_q = _project_pixels(coordinates, xs, ys)
    window = max(1.0, 0.02 * coordinates.span)
    keep = np.abs(points_s - anchor_s) <= window / 2.0
    q = points_q[keep]
    if not len(q):
        return CenterTopology.UNKNOWN
    has_left = bool(np.any(q < -chain_band_half_width))
    has_right = bool(np.any(q > chain_band_half_width))
    if not has_left or not has_right:
        return CenterTopology.UNKNOWN
    edges = np.linspace(-chain_band_half_width, chain_band_half_width, q_bins + 1)
    center_q = q[(q >= -chain_band_half_width) & (q <= chain_band_half_width)]
    covered = np.histogram(center_q, bins=edges)[0] > 0
    return CenterTopology.PRESENT if covered.mean() >= minimum_q_coverage else CenterTopology.ABSENT


def _project_pixels(coordinates: ChainCoordinates, xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    px = xs.astype(np.float64) + 0.5 - coordinates.centerline.top.x
    py = ys.astype(np.float64) + 0.5 - coordinates.centerline.top.y
    return (
        px * coordinates.direction[0] + py * coordinates.direction[1],
        px * coordinates.horizontal[0] + py * coordinates.horizontal[1],
    )
