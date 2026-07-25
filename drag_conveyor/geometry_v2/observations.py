from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum
import math

import cv2
import numpy as np

from .coordinates import ChainCoordinates, quantile_type7


class SideHint(StrEnum):
    LEFT = "left"
    RIGHT = "right"
    CENTER = "center"
    SPANS_BOTH = "spans_both"


class AnchorQuality(StrEnum):
    OK = "ok"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True, slots=True)
class ComponentExtractionConfig:
    chain_band_half_width: float
    boundary_margin_px: int
    maximum_anchor_spread_ratio: float = 0.05
    secondary_anchor_peak_ratio: float = 0.80
    duplicate_anchor_gate_ratio: float = 0.03
    duplicate_iou_threshold: float = 0.70
    duplicate_ios_threshold: float = 0.85


@dataclass(frozen=True, slots=True)
class DuplicateAlias:
    component_id: str
    iou: float
    ios: float


@dataclass(frozen=True, slots=True)
class Component:
    """One accepted 8-connected foreground component from a detection mask."""

    component_id: str
    source_frame_id: int
    source_detection_id: str
    source_detection_score: float
    source_model_output_row_index: int | None
    class_id: int
    area_px: int
    bbox_roi_xyxy: tuple[int, int, int, int]
    centroid_sq: tuple[float, float]
    s_anchor: float
    s_anchor_sigma: float
    q_median: float
    side_hint: SideHint
    anchor_quality: AnchorQuality
    touches_roi_boundary: bool
    touches_model_bbox_boundary: bool
    mask_roi: np.ndarray = field(repr=False, compare=False)
    duplicate_aliases: tuple[DuplicateAlias, ...] = ()


def extract_components(
    mask_roi: np.ndarray,
    *,
    source_frame_id: int,
    source_detection_id: str,
    source_detection_score: float,
    source_model_output_row_index: int | None,
    class_id: int,
    coordinates: ChainCoordinates,
    config: ComponentExtractionConfig,
    model_bbox_roi_xyxy: tuple[float, float, float, float] | None,
) -> tuple[Component, ...]:
    """Extract deterministic topology components without morphology.

    ``mask_roi`` remains the source of truth. This function never fills, opens,
    closes, or joins foreground pixels.
    """
    binary_mask = _as_binary_mask(mask_roi)
    roi_h, roi_w = binary_mask.shape
    if (roi_w, roi_h) != (coordinates.roi.width, coordinates.roi.height):
        raise ValueError("Mask dimensions must match the inspection ROI")

    instance_area = int(binary_mask.sum())
    minimum_area = max(16.0, 0.00002 * roi_w * roi_h, 0.005 * instance_area)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

    pending: list[Component] = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < minimum_area:
            continue
        component_mask = labels == label
        ys, xs = np.nonzero(component_mask)
        component = _build_component(
            component_mask=component_mask,
            xs=xs,
            ys=ys,
            source_frame_id=source_frame_id,
            source_detection_id=source_detection_id,
            source_detection_score=source_detection_score,
            source_model_output_row_index=source_model_output_row_index,
            class_id=class_id,
            coordinates=coordinates,
            config=config,
            model_bbox_roi_xyxy=model_bbox_roi_xyxy,
        )
        pending.append(component)

    pending.sort(key=lambda component: (component.s_anchor, component.q_median, -component.area_px))
    return tuple(
        replace(component, component_id=f"f{source_frame_id:09d}-{source_detection_id}-c{index:02d}")
        for index, component in enumerate(pending, start=1)
    )


def deduplicate_components(
    components: tuple[Component, ...],
    *,
    coordinates: ChainCoordinates,
    config: ComponentExtractionConfig,
) -> tuple[Component, ...]:
    """Suppress direct same-frame duplicates without transitive closure or mask union."""
    ordered = sorted(
        components,
        key=lambda component: (
            component.touches_roi_boundary or component.touches_model_bbox_boundary,
            -component.area_px,
            -component.source_detection_score,
            component.s_anchor_sigma,
            component.component_id,
        ),
    )
    representatives: list[Component] = []
    for candidate in ordered:
        matches: list[tuple[float, float, int]] = []
        for index, representative in enumerate(representatives):
            if not _can_be_duplicate(candidate, representative, coordinates, config):
                continue
            iou, ios = component_overlap(candidate, representative)
            if ios >= config.duplicate_ios_threshold or iou >= config.duplicate_iou_threshold:
                matches.append((ios, iou, index))
        if not matches:
            representatives.append(candidate)
            continue
        _, _, representative_index = max(matches, key=lambda item: (item[0], item[1], -item[2]))
        representative = representatives[representative_index]
        iou, ios = component_overlap(candidate, representative)
        aliases = (*representative.duplicate_aliases, DuplicateAlias(candidate.component_id, iou, ios))
        representatives[representative_index] = replace(representative, duplicate_aliases=aliases)
    return tuple(representatives)


def component_overlap(first: Component, second: Component) -> tuple[float, float]:
    if first.mask_roi.shape != second.mask_roi.shape:
        raise ValueError("Components must share ROI dimensions to compare overlap")
    intersection = int(np.logical_and(first.mask_roi, second.mask_roi).sum())
    if intersection == 0:
        return (0.0, 0.0)
    first_area = int(first.mask_roi.sum())
    second_area = int(second.mask_roi.sum())
    union = first_area + second_area - intersection
    return (intersection / union, intersection / min(first_area, second_area))


def _build_component(
    *,
    component_mask: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    source_frame_id: int,
    source_detection_id: str,
    source_detection_score: float,
    source_model_output_row_index: int | None,
    class_id: int,
    coordinates: ChainCoordinates,
    config: ComponentExtractionConfig,
    model_bbox_roi_xyxy: tuple[float, float, float, float] | None,
) -> Component:
    s_values, q_values = _project_pixels(coordinates, xs, ys)
    s_anchor, s_sigma, quality = _fragment_anchor(s_values, q_values, coordinates, config)
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    bbox = (x1, y1, x2, y2)
    return Component(
        component_id="",
        source_frame_id=source_frame_id,
        source_detection_id=source_detection_id,
        source_detection_score=source_detection_score,
        source_model_output_row_index=source_model_output_row_index,
        class_id=class_id,
        area_px=int(component_mask.sum()),
        bbox_roi_xyxy=bbox,
        centroid_sq=(float(np.mean(s_values)), float(np.mean(q_values))),
        s_anchor=s_anchor,
        s_anchor_sigma=s_sigma,
        q_median=quantile_type7(q_values, 0.5),
        side_hint=_side_hint(q_values, config.chain_band_half_width),
        anchor_quality=quality,
        touches_roi_boundary=x1 == 0 or y1 == 0 or x2 == coordinates.roi.width or y2 == coordinates.roi.height,
        touches_model_bbox_boundary=_touches_model_bbox(bbox, model_bbox_roi_xyxy, config.boundary_margin_px),
        mask_roi=component_mask,
    )


def _as_binary_mask(mask_roi: np.ndarray) -> np.ndarray:
    if mask_roi.ndim != 2:
        raise ValueError("mask_roi must be a 2D array")
    return np.ascontiguousarray(mask_roi.astype(bool).astype(np.uint8))


def _project_pixels(coordinates: ChainCoordinates, xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    px = xs.astype(np.float64) + 0.5 - coordinates.centerline.top.x
    py = ys.astype(np.float64) + 0.5 - coordinates.centerline.top.y
    s_values = px * coordinates.direction[0] + py * coordinates.direction[1]
    q_values = px * coordinates.horizontal[0] + py * coordinates.horizontal[1]
    return s_values, q_values


def _fragment_anchor(
    s_values: np.ndarray,
    q_values: np.ndarray,
    coordinates: ChainCoordinates,
    config: ComponentExtractionConfig,
) -> tuple[float, float, AnchorQuality]:
    distances = np.maximum(0.0, np.abs(q_values) - config.chain_band_half_width)
    k = max(5, math.ceil(0.10 * len(s_values)))
    kth_distance = np.partition(distances, k - 1)[k - 1]
    near_chain = s_values[distances <= kth_distance]
    bin_width = max(1.0, 0.002 * coordinates.span)
    bin_count = max(1, math.ceil(coordinates.span / bin_width))
    indices = np.floor((near_chain - coordinates.s_min) / bin_width).astype(np.int64)
    indices = np.clip(indices, 0, bin_count - 1)
    histogram = np.bincount(indices, minlength=bin_count)
    window_sums = np.array(
        [histogram[max(0, index - 1) : min(bin_count, index + 2)].sum() for index in range(bin_count)],
        dtype=np.int64,
    )
    median_s = quantile_type7(s_values, 0.5)
    peak_indices = _local_peak_indices(window_sums)
    best = min(
        peak_indices,
        key=lambda index: (-int(window_sums[index]), abs((coordinates.s_min + (index + 0.5) * bin_width) - median_s), index),
    )
    in_window = np.abs(indices - best) <= 1
    chosen = near_chain[in_window]
    anchor = quantile_type7(chosen, 0.5)
    center = coordinates.s_min + (best + 0.5) * bin_width
    competing = [
        index
        for index in peak_indices
        if abs(index - best) > 2
        and window_sums[index] >= config.secondary_anchor_peak_ratio * window_sums[best]
        and abs((coordinates.s_min + (index + 0.5) * bin_width) - center)
        > config.maximum_anchor_spread_ratio * coordinates.span
    ]
    spread = quantile_type7(chosen, 0.95) - quantile_type7(chosen, 0.05)
    quality = AnchorQuality.AMBIGUOUS if competing or spread > config.maximum_anchor_spread_ratio * coordinates.span else AnchorQuality.OK
    sigma = max(1.0, 1.4826 * quantile_type7(np.abs(chosen - anchor), 0.5))
    return (anchor, sigma, quality)


def _local_peak_indices(window_sums: np.ndarray) -> tuple[int, ...]:
    """Return one deterministic representative for every local-maximum plateau."""
    peaks: list[int] = []
    start = 0
    while start < len(window_sums):
        end = start
        while end + 1 < len(window_sums) and window_sums[end + 1] == window_sums[start]:
            end += 1
        before = window_sums[start - 1] if start else -1
        after = window_sums[end + 1] if end + 1 < len(window_sums) else -1
        if window_sums[start] > before and window_sums[start] > after:
            peaks.append((start + end) // 2)
        start = end + 1
    return tuple(peaks) or (int(np.argmax(window_sums)),)


def _side_hint(q_values: np.ndarray, band_half_width: float) -> SideHint:
    q05 = quantile_type7(q_values, 0.05)
    q95 = quantile_type7(q_values, 0.95)
    if q95 < -band_half_width:
        return SideHint.LEFT
    if q05 > band_half_width:
        return SideHint.RIGHT
    if q05 <= -band_half_width and q95 >= band_half_width:
        return SideHint.SPANS_BOTH
    return SideHint.CENTER


def _touches_model_bbox(
    bbox: tuple[int, int, int, int],
    model_bbox: tuple[float, float, float, float] | None,
    margin: int,
) -> bool:
    if model_bbox is None:
        return False
    x1, y1, x2, y2 = bbox
    bx1, by1, bx2, by2 = model_bbox
    return x1 <= bx1 + margin or y1 <= by1 + margin or x2 >= bx2 - margin or y2 >= by2 - margin


def _can_be_duplicate(
    first: Component,
    second: Component,
    coordinates: ChainCoordinates,
    config: ComponentExtractionConfig,
) -> bool:
    return (
        first.source_frame_id == second.source_frame_id
        and first.class_id == second.class_id
        and first.side_hint == second.side_hint
        and abs(first.s_anchor - second.s_anchor) <= config.duplicate_anchor_gate_ratio * coordinates.span
    )
