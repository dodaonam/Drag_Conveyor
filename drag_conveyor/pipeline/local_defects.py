from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from ..config import LocalDefectConfig


@dataclass(frozen=True, slots=True)
class LocalDefectFeatures:
    left_shape_score: float
    middle_shape_score: float
    right_shape_score: float
    max_shape_score: float
    left_defect_weighted_pixels: float
    middle_defect_weighted_pixels: float
    right_defect_weighted_pixels: float
    shape_alignment_iou: float
    mask_area_ratio: float
    local_alignment_low: float
    color_delta_mean: float
    color_delta_p95: float
    color_abnormal_ratio: float
    dark_pixel_ratio: float
    local_color_pixels_insufficient: float
    local_analysis_success: float
    local_canonicalize_failed: float

    def to_dict(self) -> dict[str, float]:
        return {
            "left_shape_score": self.left_shape_score,
            "middle_shape_score": self.middle_shape_score,
            "right_shape_score": self.right_shape_score,
            "max_shape_score": self.max_shape_score,
            "left_defect_weighted_pixels": self.left_defect_weighted_pixels,
            "middle_defect_weighted_pixels": self.middle_defect_weighted_pixels,
            "right_defect_weighted_pixels": self.right_defect_weighted_pixels,
            "shape_alignment_iou": self.shape_alignment_iou,
            "mask_area_ratio": self.mask_area_ratio,
            "local_alignment_low": self.local_alignment_low,
            "color_delta_mean": self.color_delta_mean,
            "color_delta_p95": self.color_delta_p95,
            "color_abnormal_ratio": self.color_abnormal_ratio,
            "dark_pixel_ratio": self.dark_pixel_ratio,
            "local_color_pixels_insufficient": self.local_color_pixels_insufficient,
            "local_analysis_success": self.local_analysis_success,
            "local_canonicalize_failed": self.local_canonicalize_failed,
        }

    @classmethod
    def zero_failed(cls) -> "LocalDefectFeatures":
        return cls(
            left_shape_score=0.0,
            middle_shape_score=0.0,
            right_shape_score=0.0,
            max_shape_score=0.0,
            left_defect_weighted_pixels=0.0,
            middle_defect_weighted_pixels=0.0,
            right_defect_weighted_pixels=0.0,
            shape_alignment_iou=0.0,
            mask_area_ratio=0.0,
            local_alignment_low=1.0,
            color_delta_mean=0.0,
            color_delta_p95=0.0,
            color_abnormal_ratio=0.0,
            dark_pixel_ratio=0.0,
            local_color_pixels_insufficient=1.0,
            local_analysis_success=0.0,
            local_canonicalize_failed=1.0,
        )


@dataclass(frozen=True, slots=True)
class LocalDefectBaseline:
    template_mask: np.ndarray
    template_prob: np.ndarray
    template_area_ratio: float
    lab_median: np.ndarray
    lab_mad: np.ndarray
    color_abnormal_ratio_p95: float
    dark_ratio_p95: float
    baseline_alignment_iou_p50: float
    baseline_alignment_iou_p10: float
    canonicalize_failure_ratio: float
    samples_used: int
    zone_slices: dict[str, slice]


def _white_lab_anchor() -> np.ndarray:
    white_bgr = np.full((1, 1, 3), 255, dtype=np.uint8)
    return cv2.cvtColor(white_bgr, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)


def build_local_defect_baseline(
    *,
    bars: list[Any],
    config: LocalDefectConfig,
) -> LocalDefectBaseline:
    if len(bars) < config.min_template_samples:
        raise ValueError("local_baseline_not_stable: not enough geometry-normal candidates")

    canonical_samples: list[tuple[np.ndarray, np.ndarray]] = []
    canonicalize_failures = 0

    for bar in bars:
        mask_frame = _mask_roi_to_frame_mask(
            mask_roi=bar.mask_roi,
            roi_origin_xy=bar.roi_origin_xy,
            frame_shape_hw=bar.source_frame.shape[:2],
        )
        try:
            canonical_crop, canonical_mask = _canonicalize_bar(
                frame=bar.source_frame,
                mask_frame=mask_frame,
                contour_frame=bar.contour_frame,
                config=config,
            )
        except ValueError:
            canonicalize_failures += 1
            continue

        canonical_samples.append(
            (
                canonical_crop,
                _cleanup_mask(
                    canonical_mask,
                    kernel_size=config.morph_kernel_size,
                ),
            )
        )

    if len(canonical_samples) < config.min_template_samples:
        raise ValueError("local_baseline_not_stable: not enough template samples")

    canonicalize_failure_ratio = canonicalize_failures / float(len(bars))
    if canonicalize_failure_ratio > config.max_canonicalize_failure_ratio:
        raise ValueError("local_baseline_not_stable: canonicalize failure ratio too high")

    mask_stack = np.stack(
        [sample_mask.astype(np.float32) / 255.0 for _, sample_mask in canonical_samples],
        axis=0,
    )
    template_prob = np.median(mask_stack, axis=0).astype(np.float32)
    template_mask = (template_prob >= config.template_mask_threshold).astype(np.uint8) * 255
    template_area_ratio = (
        float(np.count_nonzero(template_mask) / template_mask.size) if template_mask.size else 0.0
    )
    if not config.min_template_area_ratio <= template_area_ratio <= config.max_template_area_ratio:
        raise ValueError("local_baseline_not_stable: template area ratio out of range")

    alignment_ious: list[float] = []
    template_bool = template_mask > 0
    for _, candidate_mask in canonical_samples:
        current_bool = candidate_mask > 0
        intersection = np.count_nonzero(np.logical_and(current_bool, template_bool))
        union = np.count_nonzero(np.logical_or(current_bool, template_bool))
        alignment_ious.append(float(intersection / union) if union else 0.0)

    baseline_alignment_iou_p50 = float(np.percentile(alignment_ious, 50))
    baseline_alignment_iou_p10 = float(np.percentile(alignment_ious, 10))
    if baseline_alignment_iou_p50 < config.min_baseline_alignment_iou_p50:
        raise ValueError("local_baseline_not_stable: template IoU p50 below threshold")
    if baseline_alignment_iou_p10 < config.min_baseline_alignment_iou_p10:
        raise ValueError("local_baseline_not_stable: template IoU p10 below threshold")

    zone_slices = _make_zone_slices(template_mask.shape[1], config)

    if config.color_enabled:
        lab_median = _white_lab_anchor()
        lab_mad = np.zeros(3, dtype=np.float32)
    else:
        lab_median = np.zeros(3, dtype=np.float32)
        lab_mad = np.zeros(3, dtype=np.float32)
    color_abnormal_ratio_p95 = 0.0
    dark_ratio_p95 = 0.0

    return LocalDefectBaseline(
        template_mask=template_mask,
        template_prob=template_prob,
        template_area_ratio=template_area_ratio,
        lab_median=lab_median,
        lab_mad=lab_mad,
        color_abnormal_ratio_p95=color_abnormal_ratio_p95,
        dark_ratio_p95=dark_ratio_p95,
        baseline_alignment_iou_p50=baseline_alignment_iou_p50,
        baseline_alignment_iou_p10=baseline_alignment_iou_p10,
        canonicalize_failure_ratio=canonicalize_failure_ratio,
        samples_used=len(canonical_samples),
        zone_slices=zone_slices,
    )


def analyze_local_defects(
    *,
    frame: np.ndarray,
    contour_frame: np.ndarray,
    mask_roi: np.ndarray,
    roi_origin_xy: tuple[int, int],
    baseline: LocalDefectBaseline,
    config: LocalDefectConfig,
) -> LocalDefectFeatures:
    try:
        mask_frame = _mask_roi_to_frame_mask(
            mask_roi=mask_roi,
            roi_origin_xy=roi_origin_xy,
            frame_shape_hw=frame.shape[:2],
        )
        canonical_crop, canonical_mask = _canonicalize_bar(
            frame=frame,
            mask_frame=mask_frame,
            contour_frame=contour_frame,
            config=config,
        )
        canonical_mask = _cleanup_mask(
            canonical_mask,
            kernel_size=config.morph_kernel_size,
        )
    except ValueError:
        return LocalDefectFeatures.zero_failed()

    shape_scores = _compute_shape_scores(
        current_mask=canonical_mask,
        template_mask=baseline.template_mask,
        zone_slices=baseline.zone_slices,
        config=config,
    )

    if config.color_enabled:
        color_scores = _compute_color_scores_from_reference(
            canonical_crop_bgr=canonical_crop,
            current_mask=canonical_mask,
            template_mask=baseline.template_mask,
            lab_median=baseline.lab_median,
            color_delta_threshold=config.color_delta_threshold,
            dark_l_threshold=config.dark_l_threshold,
            erode_mask_iterations=config.erode_mask_iterations,
            min_color_pixels=config.min_color_pixels_per_sample,
        )
    else:
        color_scores = {
            "color_delta_mean": 0.0,
            "color_delta_p95": 0.0,
            "color_abnormal_ratio": 0.0,
            "dark_pixel_ratio": 0.0,
            "local_color_pixels_insufficient": 0.0,
        }

    return LocalDefectFeatures(
        left_shape_score=shape_scores["left_shape_score"],
        middle_shape_score=shape_scores["middle_shape_score"],
        right_shape_score=shape_scores["right_shape_score"],
        max_shape_score=max(
            shape_scores["left_shape_score"],
            shape_scores["middle_shape_score"],
            shape_scores["right_shape_score"],
        ),
        left_defect_weighted_pixels=shape_scores["left_defect_weighted_pixels"],
        middle_defect_weighted_pixels=shape_scores["middle_defect_weighted_pixels"],
        right_defect_weighted_pixels=shape_scores["right_defect_weighted_pixels"],
        shape_alignment_iou=shape_scores["shape_alignment_iou"],
        mask_area_ratio=shape_scores["mask_area_ratio"],
        local_alignment_low=1.0 if shape_scores["shape_alignment_iou"] < config.min_alignment_iou else 0.0,
        color_delta_mean=color_scores["color_delta_mean"],
        color_delta_p95=color_scores["color_delta_p95"],
        color_abnormal_ratio=color_scores["color_abnormal_ratio"],
        dark_pixel_ratio=color_scores["dark_pixel_ratio"],
        local_color_pixels_insufficient=color_scores["local_color_pixels_insufficient"],
        local_analysis_success=1.0,
        local_canonicalize_failed=0.0,
    )


def _mask_roi_to_frame_mask(
    *,
    mask_roi: np.ndarray,
    roi_origin_xy: tuple[int, int],
    frame_shape_hw: tuple[int, int],
) -> np.ndarray:
    frame_h, frame_w = frame_shape_hw
    roi_x, roi_y = roi_origin_xy

    mask_frame = np.zeros((frame_h, frame_w), dtype=np.uint8)
    if mask_roi.size == 0:
        return mask_frame

    mask_h, mask_w = mask_roi.shape[:2]
    x1 = max(0, int(roi_x))
    y1 = max(0, int(roi_y))
    x2 = min(frame_w, x1 + mask_w)
    y2 = min(frame_h, y1 + mask_h)

    if x2 <= x1 or y2 <= y1:
        return mask_frame

    mask_src = mask_roi[: y2 - y1, : x2 - x1]
    mask_frame[y1:y2, x1:x2] = (mask_src > 0).astype(np.uint8) * 255
    return mask_frame


def _canonicalize_bar(
    *,
    frame: np.ndarray,
    mask_frame: np.ndarray,
    contour_frame: np.ndarray,
    config: LocalDefectConfig,
) -> tuple[np.ndarray, np.ndarray]:
    if contour_frame is None or len(contour_frame) < 3:
        raise ValueError("local_canonicalize_failed: contour has fewer than 3 points")

    rect = cv2.minAreaRect(contour_frame.astype(np.float32))
    (_, _), (rect_w, rect_h), _ = rect
    if rect_w <= 1.0 or rect_h <= 1.0:
        raise ValueError("local_canonicalize_failed: rectangle too small")

    length = max(rect_w, rect_h)
    width = min(rect_w, rect_h)
    if width <= 1e-6 or length / width < config.min_bar_aspect_ratio:
        raise ValueError("local_canonicalize_failed: aspect ratio too low")

    src = _order_box_points_preserve_image_left_right(
        box=cv2.boxPoints(rect),
        min_endpoint_x_separation_ratio=config.min_endpoint_x_separation_ratio,
    )
    dst = np.array(
        [
            [0, 0],
            [config.canonical_width - 1, 0],
            [config.canonical_width - 1, config.canonical_height - 1],
            [0, config.canonical_height - 1],
        ],
        dtype=np.float32,
    )

    transform = cv2.getPerspectiveTransform(src, dst)
    canonical_crop = cv2.warpPerspective(
        frame,
        transform,
        (config.canonical_width, config.canonical_height),
        flags=cv2.INTER_LINEAR,
    )
    canonical_mask = cv2.warpPerspective(
        mask_frame,
        transform,
        (config.canonical_width, config.canonical_height),
        flags=cv2.INTER_NEAREST,
    )
    canonical_mask = (canonical_mask > 0).astype(np.uint8) * 255

    if config.orientation_flip_x:
        canonical_crop = cv2.flip(canonical_crop, 1)
        canonical_mask = cv2.flip(canonical_mask, 1)

    return canonical_crop, canonical_mask


def _order_box_points_preserve_image_left_right(
    *,
    box: np.ndarray,
    min_endpoint_x_separation_ratio: float,
) -> np.ndarray:
    pts = box.astype(np.float32)
    edges: list[tuple[float, int, int]] = []
    for idx in range(4):
        next_idx = (idx + 1) % 4
        length = float(np.linalg.norm(pts[next_idx] - pts[idx]))
        edges.append((length, idx, next_idx))

    end_edge_a = sorted(edges, key=lambda edge: edge[0])[0]
    used = {end_edge_a[1], end_edge_a[2]}
    remaining = [idx for idx in range(4) if idx not in used]
    end_edge_b = (
        float(np.linalg.norm(pts[remaining[1]] - pts[remaining[0]])),
        remaining[0],
        remaining[1],
    )

    a_pts = pts[[end_edge_a[1], end_edge_a[2]]]
    b_pts = pts[[end_edge_b[1], end_edge_b[2]]]
    a_center = a_pts.mean(axis=0)
    b_center = b_pts.mean(axis=0)

    endpoint_x_separation = abs(float(a_center[0]) - float(b_center[0]))
    long_edge_length = max(edge[0] for edge in edges)
    if endpoint_x_separation < min_endpoint_x_separation_ratio * long_edge_length:
        raise ValueError("local_canonicalize_failed: endpoint x separation too low")

    if a_center[0] <= b_center[0]:
        left_end = a_pts
        right_end = b_pts
    else:
        left_end = b_pts
        right_end = a_pts

    left_sorted = left_end[np.argsort(left_end[:, 1])]
    right_sorted = right_end[np.argsort(right_end[:, 1])]

    return np.array(
        [
            left_sorted[0],
            right_sorted[0],
            right_sorted[1],
            left_sorted[1],
        ],
        dtype=np.float32,
    )


def _cleanup_mask(mask: np.ndarray, *, kernel_size: int) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    return (mask_u8 > 0).astype(np.uint8) * 255


def _make_zone_slices(width: int, config: LocalDefectConfig) -> dict[str, slice]:
    def make(pair: list[float]) -> slice:
        start = max(0, min(width, int(round(pair[0] * width))))
        end = max(0, min(width, int(round(pair[1] * width))))
        return slice(start, end)

    return {
        "left": make(config.zone_left),
        "middle": make(config.zone_middle),
        "right": make(config.zone_right),
    }


def _compute_shape_scores(
    *,
    current_mask: np.ndarray,
    template_mask: np.ndarray,
    zone_slices: dict[str, slice],
    config: LocalDefectConfig,
) -> dict[str, float]:
    current = current_mask > 0
    template = template_mask > 0
    missing = np.logical_and(template, np.logical_not(current))
    extra = np.logical_and(current, np.logical_not(template))
    weighted_defect = (
        missing.astype(np.float32) * config.missing_weight
        + extra.astype(np.float32) * config.extra_weight
    )

    template_pixels = float(np.count_nonzero(template))
    current_pixels = float(np.count_nonzero(current))
    intersection = np.count_nonzero(np.logical_and(current, template))
    union = np.count_nonzero(np.logical_or(current, template))
    shape_alignment_iou = float(intersection / union) if union else 0.0
    mask_area_ratio = float(current_pixels / template_pixels) if template_pixels else 0.0

    output: dict[str, float] = {
        "shape_alignment_iou": shape_alignment_iou,
        "mask_area_ratio": mask_area_ratio,
    }
    for zone_name, x_slice in zone_slices.items():
        zone_weighted_pixels = float(weighted_defect[:, x_slice].sum())
        zone_expected_pixels = float(np.count_nonzero(template[:, x_slice]))
        zone_score = (
            zone_weighted_pixels / zone_expected_pixels if zone_expected_pixels > 0 else 0.0
        )
        output[f"{zone_name}_shape_score"] = zone_score
        output[f"{zone_name}_defect_weighted_pixels"] = zone_weighted_pixels

    return output


def _collect_lab_pixels(
    *,
    canonical_crop_bgr: np.ndarray,
    current_mask: np.ndarray,
    template_mask: np.ndarray,
    erode_mask_iterations: int,
    min_color_pixels: int,
) -> np.ndarray | None:
    lab = cv2.cvtColor(canonical_crop_bgr, cv2.COLOR_BGR2LAB)
    analysis_mask = current_mask > 0
    if erode_mask_iterations > 0:
        analysis_mask = cv2.erode(
            analysis_mask.astype(np.uint8),
            np.ones((3, 3), np.uint8),
            iterations=erode_mask_iterations,
        ).astype(bool)

    if np.count_nonzero(analysis_mask) < min_color_pixels:
        return None
    return lab[analysis_mask].astype(np.float32)


def _compute_color_scores_from_reference(
    *,
    canonical_crop_bgr: np.ndarray,
    current_mask: np.ndarray,
    template_mask: np.ndarray,
    lab_median: np.ndarray,
    color_delta_threshold: float,
    dark_l_threshold: float,
    erode_mask_iterations: int,
    min_color_pixels: int,
) -> dict[str, float]:
    pixels = _collect_lab_pixels(
        canonical_crop_bgr=canonical_crop_bgr,
        current_mask=current_mask,
        template_mask=template_mask,
        erode_mask_iterations=erode_mask_iterations,
        min_color_pixels=min_color_pixels,
    )
    if pixels is None:
        return {
            "color_delta_mean": 0.0,
            "color_delta_p95": 0.0,
            "color_abnormal_ratio": 0.0,
            "dark_pixel_ratio": 0.0,
            "local_color_pixels_insufficient": 1.0,
        }

    diff = pixels - lab_median.astype(np.float32)
    delta = np.sqrt(
        0.25 * diff[:, 0] ** 2
        + diff[:, 1] ** 2
        + diff[:, 2] ** 2
    )

    return {
        "color_delta_mean": float(np.mean(delta)),
        "color_delta_p95": float(np.percentile(delta, 95)),
        "color_abnormal_ratio": float(np.mean(delta > color_delta_threshold)),
        "dark_pixel_ratio": float(np.mean(pixels[:, 0] < dark_l_threshold)),
        "local_color_pixels_insufficient": 0.0,
    }


__all__ = [
    "LocalDefectBaseline",
    "LocalDefectFeatures",
    "analyze_local_defects",
    "build_local_defect_baseline",
]
