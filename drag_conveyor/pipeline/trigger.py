from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import InspectionRegionConfig

TRIGGER_POSITION_RATIO = 0.5
TRIGGER_THICKNESS_RATIO = 0.25


@dataclass(frozen=True, slots=True)
class BandRect:
    x1: int
    y1: int
    x2: int
    y2: int
    centerline: float


def build_trigger_band(region: InspectionRegionConfig) -> BandRect:
    x1 = int(region.x)
    y1 = int(region.y)
    x2 = int(region.x + region.w)
    y2 = int(region.y + region.h)

    thickness = max(1, int(round(region.h * TRIGGER_THICKNESS_RATIO)))
    center = region.y + region.h * TRIGGER_POSITION_RATIO
    half = thickness / 2.0
    by1 = max(y1, int(round(center - half)))
    by2 = min(y2, int(round(center + half)))
    return BandRect(x1=x1, y1=by1, x2=x2, y2=by2, centerline=center)


def centroid_crossed(
    prev_xy: tuple[float, float] | None,
    curr_xy: tuple[float, float],
    centerline: float,
) -> bool:
    if prev_xy is None:
        return False

    prev_y = prev_xy[1]
    curr_y = curr_xy[1]
    return prev_y < centerline <= curr_y


def centroid_inside_band(curr_xy: tuple[float, float], band: BandRect) -> bool:
    x, y = curr_xy
    return band.x1 <= x <= band.x2 and band.y1 <= y <= band.y2


def mask_overlap_ratio_with_band(
    mask_roi: np.ndarray,
    roi_origin_xy: tuple[int, int],
    band: BandRect,
) -> float:
    mask_area = int(mask_roi.sum())
    if mask_area <= 0:
        return 0.0

    roi_x, roi_y = roi_origin_xy
    roi_h, roi_w = mask_roi.shape[:2]
    roi_x2 = roi_x + roi_w
    roi_y2 = roi_y + roi_h

    ix1 = max(roi_x, band.x1)
    iy1 = max(roi_y, band.y1)
    ix2 = min(roi_x2, band.x2)
    iy2 = min(roi_y2, band.y2)
    if ix1 >= ix2 or iy1 >= iy2:
        return 0.0

    mx1 = ix1 - roi_x
    my1 = iy1 - roi_y
    mx2 = ix2 - roi_x
    my2 = iy2 - roi_y

    overlap = int(mask_roi[my1:my2, mx1:mx2].sum())
    return float(overlap / mask_area)


class TriggerEngine:
    def __init__(self, pending_ttl_frames: int = 3, allow_inside_band_trigger: bool = True) -> None:
        self.pending_ttl_frames = max(0, int(pending_ttl_frames))
        self.allow_inside_band_trigger = bool(allow_inside_band_trigger)
        self._pending: dict[int, int] = {}
        self._processed: set[int] = set()

    def begin_frame(self) -> None:
        expired: list[int] = []
        for track_id, ttl in self._pending.items():
            next_ttl = ttl - 1
            if next_ttl <= 0:
                expired.append(track_id)
            else:
                self._pending[track_id] = next_ttl
        for track_id in expired:
            self._pending.pop(track_id, None)

    def is_processed(self, track_id: int) -> bool:
        return track_id in self._processed

    def mark_processed(self, track_id: int) -> None:
        self._processed.add(track_id)
        self._pending.pop(track_id, None)

    def should_trigger(
        self,
        *,
        track_id: int,
        prev_xy: tuple[float, float] | None,
        curr_xy: tuple[float, float],
        centerline: float,
        band: BandRect,
        overlap_ratio: float,
        min_overlap_ratio: float,
    ) -> bool:
        if track_id in self._processed:
            return False

        crossed = centroid_crossed(
            prev_xy=prev_xy,
            curr_xy=curr_xy,
            centerline=centerline,
        )
        inside_band = self.allow_inside_band_trigger and centroid_inside_band(curr_xy, band)
        overlap_ready = overlap_ratio >= min_overlap_ratio

        if overlap_ready and (crossed or inside_band or track_id in self._pending):
            return True

        if self.pending_ttl_frames > 0 and (crossed or inside_band) and not overlap_ready:
            self._pending[track_id] = self.pending_ttl_frames

        return False


__all__ = [
    "BandRect",
    "TRIGGER_POSITION_RATIO",
    "TRIGGER_THICKNESS_RATIO",
    "TriggerEngine",
    "build_trigger_band",
    "centroid_crossed",
    "centroid_inside_band",
    "mask_overlap_ratio_with_band",
]
