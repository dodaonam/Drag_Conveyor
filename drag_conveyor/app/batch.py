from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..calibration import CalibrationEngine
from ..config import CalibrationResult, Profile
from ..inference import OnnxRuntimeEngine, postprocess_segmentation, preprocess_roi
from ..inspection_modes import (
    AVERAGE_RATIO_INSPECTION_MODE,
    AverageRatioInspector,
    AverageRatioThresholds,
    is_supported_inspection_mode,
)
from ..pipeline.measure import measure_contour
from ..pipeline.rules import RuleEngine
from ..pipeline.tracking import CentroidTracker
from ..pipeline.trigger import (
    BandRect,
    TriggerEngine,
    build_trigger_band,
    mask_overlap_ratio_with_band,
)
from ..video_io import open_video_source
from .ids import generate_run_id

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CollectedBar:
    frame_id: int
    track_id: int
    measurements: dict[str, float]
    bbox_frame_xyxy: tuple[float, float, float, float]
    overlap_ratio: float
    contour_frame: np.ndarray
    source_frame: np.ndarray
    latency_ms: float


@dataclass(frozen=True, slots=True)
class BarResult:
    frame_id: int
    track_id: int
    result: str
    score: float
    reasons: list[str]
    measurements: dict[str, float]
    thresholds: dict[str, float]
    margins: dict[str, float]
    bbox_frame_xyxy: tuple[float, float, float, float]
    contour_frame: np.ndarray
    latency_ms: float
    source_frame: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class BatchInspectionResult:
    run_id: str
    success: bool
    failure_reason: str
    calibration_result: CalibrationResult | None
    bars: list[BarResult]
    total_bars: int
    normal_bars: int
    defect_bars: int
    frames_scanned: int
    inlier_count: int
    outlier_count: int
    inlier_ratio: float
    defect_snapshots_dir: Path | None


@dataclass(frozen=True, slots=True)
class _ClassificationOutcome:
    calibration_result: CalibrationResult | None
    bars: list[BarResult]
    inlier_count: int
    outlier_count: int
    inlier_ratio: float


def run_batch_inspection(
    *,
    profile: Profile,
    source: str,
    run_id: str | None = None,
    defect_snapshots_root: Path | None = None,
    inspection_mode: str | None = None,
) -> BatchInspectionResult:
    """Single-pass collect-all: infer full video, calibrate on all data, classify all bars."""
    if not Path(source).exists():
        raise FileNotFoundError(f"Video source not found: {source}")
    if run_id is not None and ((".." in run_id) or ("/" in run_id) or ("\\" in run_id)):
        raise ValueError(f"run_id must not contain path separators: {run_id!r}")
    inspection_mode = inspection_mode or profile.inspection.mode
    if not is_supported_inspection_mode(inspection_mode):
        raise ValueError(f"Unsupported inspection_mode: {inspection_mode}")
    run_id = run_id or generate_run_id()
    region = profile.region
    roi_config = region.roi
    trigger_band = profile.collection.trigger_band
    tracker_config = profile.collection.tracker
    postprocess_config = profile.model.postprocess

    engine = OnnxRuntimeEngine(providers=profile.model.providers)
    engine.load(str(_resolve_model_path(profile.model.path)), profile.model)

    tracker = CentroidTracker(
        max_jump_px=tracker_config.max_jump_px,
        ttl_frames=tracker_config.ttl_frames,
        min_hits=tracker_config.min_hits,
        max_reverse_px=tracker_config.max_reverse_px,
        max_area_ratio_change=tracker_config.max_area_ratio_change,
    )
    band: BandRect = build_trigger_band(region, trigger_band)
    trigger_engine = TriggerEngine(
        pending_ttl_frames=trigger_band.pending_ttl_frames,
        allow_inside_band_trigger=trigger_band.allow_inside_band_trigger,
    )

    collected: list[CollectedBar] = []
    frame_count = 0

    # --- Phase 1: Thu thập toàn bộ thanh từ video ---
    cap, _ = open_video_source(source)
    try:
        while True:
            t0 = time.perf_counter()
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frame_count += 1

            if (
                frame.shape[1] != region.frame_width
                or frame.shape[0] != region.frame_height
            ):
                LOGGER.warning(
                    "Frame size mismatch: expected %dx%d got %dx%d — skipping frame %d",
                    region.frame_width,
                    region.frame_height,
                    frame.shape[1],
                    frame.shape[0],
                    frame_count,
                )
                continue

            roi = frame[
                roi_config.y : roi_config.y + roi_config.h,
                roi_config.x : roi_config.x + roi_config.w,
            ]
            prep = preprocess_roi(
                roi,
                roi_origin_xy=(roi_config.x, roi_config.y),
                input_size=profile.model.input_size,
                normalize=profile.model.preprocess.normalize,
                color_format=profile.model.preprocess.color_format,
                padding_value=profile.model.preprocess.padding_value,
            )
            det_out, proto_out = engine.infer(prep.tensor)
            detections = postprocess_segmentation(
                det_out,
                proto_out,
                preprocess=prep,
                model_spec=profile.model,
                postprocess_config=postprocess_config,
            )
            latency_ms = (time.perf_counter() - t0) * 1000.0

            tracks = tracker.update(detections)
            trigger_engine.begin_frame()

            for track in tracks:
                if not track.confirmed or track.missed_frames != 0:
                    continue
                if trigger_engine.is_processed(track.track_id):
                    continue

                overlap = mask_overlap_ratio_with_band(
                    mask_roi=track.detection.mask_roi,
                    roi_origin_xy=(roi_config.x, roi_config.y),
                    band=band,
                )
                if not trigger_engine.should_trigger(
                    track_id=track.track_id,
                    prev_xy=track.prev_centroid_xy,
                    curr_xy=track.centroid_xy,
                    centerline=band.centerline,
                    band=band,
                    overlap_ratio=overlap,
                    min_overlap_ratio=trigger_band.min_overlap_ratio,
                ):
                    continue

                trigger_engine.mark_processed(track.track_id)
                measurements = measure_contour(track.detection.contour_frame).to_dict()
                collected.append(
                    CollectedBar(
                        frame_id=frame_count,
                        track_id=track.track_id,
                        measurements=measurements,
                        bbox_frame_xyxy=track.detection.bbox_frame_xyxy,
                        overlap_ratio=float(overlap),
                        contour_frame=track.detection.contour_frame,
                        source_frame=frame.copy(),
                        latency_ms=latency_ms,
                    )
                )
    finally:
        cap.release()
        engine.close()

    LOGGER.info("Collected %d bars from %d frames", len(collected), frame_count)

    if not collected:
        LOGGER.error("No bars detected. Check model, confidence threshold, and trigger band config.")
        return BatchInspectionResult(
            run_id=run_id,
            success=False,
            failure_reason="no_bars_detected",
            calibration_result=None,
            bars=[],
            total_bars=0,
            normal_bars=0,
            defect_bars=0,
            frames_scanned=frame_count,
            inlier_count=0,
            outlier_count=0,
            inlier_ratio=0.0,
            defect_snapshots_dir=None,
        )

    try:
        classified = _classify_collected_bars(
            collected=collected,
            profile=profile,
            inspection_mode=inspection_mode,
        )
    except ValueError as exc:
        LOGGER.warning("Classification failed: %s", exc)
        return BatchInspectionResult(
            run_id=run_id,
            success=False,
            failure_reason=str(exc),
            calibration_result=None,
            bars=[],
            total_bars=len(collected),
            normal_bars=0,
            defect_bars=0,
            frames_scanned=frame_count,
            inlier_count=0,
            outlier_count=0,
            inlier_ratio=0.0,
            defect_snapshots_dir=None,
        )

    normal_bars = sum(1 for r in classified.bars if r.result == "normal")
    defect_bars = len(classified.bars) - normal_bars
    LOGGER.info("Classification: %d normal, %d defect", normal_bars, defect_bars)

    # --- Phase 4: Ghi defect snapshot (seek-based, không buffer frame) ---
    defect_snapshots_dir: Path | None = None
    if defect_snapshots_root is not None:
        defects = [r for r in classified.bars if r.result == "suspected_defect"]
        if defects:
            defect_snapshots_dir = Path(defect_snapshots_root) / run_id
            _write_defect_snapshots(defects, defect_snapshots_dir)

    return BatchInspectionResult(
        run_id=run_id,
        success=True,
        failure_reason="",
        calibration_result=classified.calibration_result,
        bars=classified.bars,
        total_bars=len(classified.bars),
        normal_bars=normal_bars,
        defect_bars=defect_bars,
        frames_scanned=frame_count,
        inlier_count=classified.inlier_count,
        outlier_count=classified.outlier_count,
        inlier_ratio=classified.inlier_ratio,
        defect_snapshots_dir=defect_snapshots_dir,
    )


def _classify_collected_bars(
    *,
    collected: list[CollectedBar],
    profile: Profile,
    inspection_mode: str,
) -> _ClassificationOutcome:
    if inspection_mode == AVERAGE_RATIO_INSPECTION_MODE:
        return _classify_with_average_ratio(collected, profile)
    return _classify_with_auto_baseline(collected, profile)


def _classify_with_auto_baseline(
    collected: list[CollectedBar],
    profile: Profile,
) -> _ClassificationOutcome:
    records = [bar.measurements for bar in collected]
    outcome = CalibrationEngine().calibrate(records, profile)

    if not outcome.success or outcome.calibration_result is None or outcome.updated_profile is None:
        raise ValueError(outcome.reason)

    rule_engine = RuleEngine()
    calibration_result = outcome.calibration_result
    calibrated_rules = outcome.updated_profile.inspection.auto_baseline
    defect_policy = outcome.updated_profile.inspection.defect_policy
    bar_results: list[BarResult] = []

    for bar in collected:
        evaluation = rule_engine.evaluate(
            measurements=bar.measurements,
            rules=calibrated_rules,
            defect_policy=defect_policy,
            calibration_result=calibration_result,
        )
        bar_results.append(
            BarResult(
                frame_id=bar.frame_id,
                track_id=bar.track_id,
                result=evaluation.result,
                score=evaluation.score,
                reasons=list(evaluation.reasons),
                measurements=dict(bar.measurements),
                thresholds=dict(evaluation.thresholds),
                margins=dict(evaluation.margins),
                bbox_frame_xyxy=bar.bbox_frame_xyxy,
                contour_frame=bar.contour_frame,
                latency_ms=bar.latency_ms,
                source_frame=bar.source_frame,
            )
        )

    return _ClassificationOutcome(
        calibration_result=calibration_result,
        bars=bar_results,
        inlier_count=calibration_result.inlier_count,
        outlier_count=calibration_result.outlier_count,
        inlier_ratio=calibration_result.inlier_ratio,
    )


def _classify_with_average_ratio(collected: list[CollectedBar], profile: Profile) -> _ClassificationOutcome:
    average_ratio = profile.inspection.average_ratio
    defect_policy = profile.inspection.defect_policy
    thresholds = AverageRatioThresholds(
        width_min_ratio=average_ratio.width_min_ratio,
        width_max_ratio=average_ratio.width_max_ratio,
        length_min_ratio=average_ratio.length_min_ratio,
        length_max_ratio=average_ratio.length_max_ratio,
    )
    inspector = AverageRatioInspector(
        thresholds=thresholds,
        min_violated_dimensions=defect_policy.min_violated_dimensions,
        score_dimension_count=defect_policy.score_dimension_count,
    )
    averages = inspector.compute_averages([bar.measurements for bar in collected])
    bar_results: list[BarResult] = []

    for bar in collected:
        evaluation = inspector.evaluate(bar.measurements, averages)
        bar_results.append(
            BarResult(
                frame_id=bar.frame_id,
                track_id=bar.track_id,
                result=evaluation.result,
                score=evaluation.score,
                reasons=list(evaluation.reasons),
                measurements=dict(bar.measurements),
                thresholds=dict(evaluation.thresholds),
                margins=dict(evaluation.margins),
                bbox_frame_xyxy=bar.bbox_frame_xyxy,
                contour_frame=bar.contour_frame,
                latency_ms=bar.latency_ms,
                source_frame=bar.source_frame,
            )
        )

    return _ClassificationOutcome(
        calibration_result=None,
        bars=bar_results,
        inlier_count=len(collected),
        outlier_count=0,
        inlier_ratio=1.0,
    )


def _write_defect_snapshots(defects: list[BarResult], snapshots_dir: Path) -> None:
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    for defect in defects:
        if defect.source_frame is None:
            continue
        _save_box_contour_snapshot(defect.source_frame, defect, snapshots_dir)


def _save_box_contour_snapshot(
    frame: np.ndarray,
    bar: BarResult,
    snapshots_dir: Path,
) -> None:
    image = frame.copy()
    bx1, by1, bx2, by2 = [int(v) for v in bar.bbox_frame_xyxy]
    cv2.rectangle(image, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
    contour = bar.contour_frame.astype(np.int32)
    cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)

    output = snapshots_dir / f"track_{bar.track_id:06d}_frame_{bar.frame_id:09d}.jpg"
    try:
        cv2.imwrite(str(output), image)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Snapshot write failure for track_id=%s: %s", bar.track_id, exc)


def _resolve_model_path(model_path: str) -> Path:
    path = Path(model_path)
    if path.is_absolute():
        return path
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / path


__all__ = [
    "BatchInspectionResult",
    "BarResult",
    "CollectedBar",
    "run_batch_inspection",
]
