from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from ..calibration import CalibrationEngine
from ..config import CalibrationResult, Profile
from ..inference import OnnxRuntimeEngine, postprocess_segmentation, preprocess_roi
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
    csv_path: Path | None
    defect_snapshots_dir: Path | None


def run_batch_inspection(
    *,
    profile: Profile,
    source: str,
    model_path: str,
    run_id: str | None = None,
    logs_dir: Path | None = None,
    defect_snapshots_root: Path | None = None,
    save_defect_snapshot: bool = True,
) -> BatchInspectionResult:
    """Single-pass collect-all: infer full video, calibrate on all data, classify all bars."""
    if not Path(source).exists():
        raise FileNotFoundError(f"Video source not found: {source}")
    if run_id is not None and ((".." in run_id) or ("/" in run_id) or ("\\" in run_id)):
        raise ValueError(f"run_id must not contain path separators: {run_id!r}")
    run_id = run_id or generate_run_id()
    region = profile.inspection_region

    engine = OnnxRuntimeEngine()
    engine.load(model_path, profile.model)

    tracker = CentroidTracker(
        max_jump_px=profile.tracker.max_jump_px,
        ttl_frames=profile.tracker.ttl_frames,
        min_hits=profile.tracker.min_hits,
    )
    band: BandRect = build_trigger_band(region)
    trigger_engine = TriggerEngine(
        pending_ttl_frames=region.trigger_band.pending_ttl_frames,
        allow_inside_band_trigger=region.trigger_band.allow_inside_band_trigger,
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

            roi = frame[region.y : region.y + region.h, region.x : region.x + region.w]
            prep = preprocess_roi(
                roi,
                roi_origin_xy=(region.x, region.y),
                input_size=profile.model.input_size,
                normalize=profile.model.preprocess.normalize,
                color_format=profile.model.preprocess.color_format,
            )
            det_out, proto_out = engine.infer(prep.tensor)
            detections = postprocess_segmentation(
                det_out,
                proto_out,
                preprocess=prep,
                model_spec=profile.model,
                conf_threshold=profile.model.conf_threshold,
                iou_threshold=profile.model.iou_threshold,
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
                    roi_origin_xy=(region.x, region.y),
                    band=band,
                )
                if not trigger_engine.should_trigger(
                    track_id=track.track_id,
                    prev_xy=track.prev_centroid_xy,
                    curr_xy=track.centroid_xy,
                    centerline=band.centerline,
                    band=band,
                    overlap_ratio=overlap,
                    min_overlap_ratio=region.trigger_band.min_overlap_ratio,
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
            csv_path=None,
            defect_snapshots_dir=None,
        )

    # --- Phase 2: Calibrate trên toàn bộ dữ liệu thu thập ---
    records = [bar.measurements for bar in collected]
    outcome = CalibrationEngine().calibrate(records, profile)

    if not outcome.success or outcome.calibration_result is None or outcome.updated_profile is None:
        LOGGER.warning("Calibration failed: %s", outcome.reason)
        return BatchInspectionResult(
            run_id=run_id,
            success=False,
            failure_reason=outcome.reason,
            calibration_result=None,
            bars=[],
            total_bars=len(collected),
            normal_bars=0,
            defect_bars=0,
            frames_scanned=frame_count,
            inlier_count=0,
            outlier_count=0,
            inlier_ratio=0.0,
            csv_path=None,
            defect_snapshots_dir=None,
        )

    # --- Phase 3: Classify toàn bộ thanh ---
    rule_engine = RuleEngine()
    calibration_result = outcome.calibration_result
    calibrated_rules = outcome.updated_profile.rules

    bar_results: list[BarResult] = []
    for bar in collected:
        evaluation = rule_engine.evaluate(
            measurements=bar.measurements,
            rules=calibrated_rules,
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
            )
        )

    normal_bars = sum(1 for r in bar_results if r.result == "normal")
    defect_bars = len(bar_results) - normal_bars
    LOGGER.info("Classification: %d normal, %d defect", normal_bars, defect_bars)

    # --- Phase 4: Ghi CSV ---
    csv_path: Path | None = None
    if logs_dir is not None:
        logs_dir = Path(logs_dir)
        logs_dir.mkdir(parents=True, exist_ok=True)
        csv_path = logs_dir / f"{run_id}_inspection.csv"
        _write_results_csv(csv_path, run_id, bar_results)

    # --- Phase 5: Ghi defect snapshot (seek-based, không buffer frame) ---
    defect_snapshots_dir: Path | None = None
    if save_defect_snapshot and defect_snapshots_root is not None:
        defects = [r for r in bar_results if r.result == "suspected_defect"]
        if defects:
            defect_snapshots_dir = Path(defect_snapshots_root) / run_id
            _write_defect_snapshots(source, defects, defect_snapshots_dir)

    return BatchInspectionResult(
        run_id=run_id,
        success=True,
        failure_reason="",
        calibration_result=calibration_result,
        bars=bar_results,
        total_bars=len(bar_results),
        normal_bars=normal_bars,
        defect_bars=defect_bars,
        frames_scanned=frame_count,
        inlier_count=outcome.calibration_result.inlier_count,
        outlier_count=outcome.calibration_result.outlier_count,
        inlier_ratio=outcome.calibration_result.inlier_ratio,
        csv_path=csv_path,
        defect_snapshots_dir=defect_snapshots_dir,
    )


def _write_results_csv(path: Path, run_id: str, bars: list[BarResult]) -> None:
    columns = [
        "run_id", "timestamp", "frame_id", "track_id",
        "result", "score", "reasons", "length", "width",
        "inference_fps_estimate", "latency_ms",
    ]
    ts = datetime.now().isoformat(timespec="milliseconds")
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=columns)
        writer.writeheader()
        for bar in bars:
            fps_est = 1000.0 / bar.latency_ms if bar.latency_ms > 0 else 0.0
            writer.writerow({
                "run_id": run_id,
                "timestamp": ts,
                "frame_id": bar.frame_id,
                "track_id": bar.track_id,
                "result": bar.result,
                "score": f"{bar.score:.4f}",
                "reasons": json.dumps(bar.reasons, ensure_ascii=False),
                "length": f"{bar.measurements.get('length', 0.0):.4f}",
                "width": f"{bar.measurements.get('width', 0.0):.4f}",
                "inference_fps_estimate": f"{fps_est:.2f}",
                "latency_ms": f"{bar.latency_ms:.2f}",
            })


def _write_defect_snapshots(
    source: str,
    defects: list[BarResult],
    snapshots_dir: Path,
) -> None:
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    by_frame: dict[int, list[BarResult]] = {}
    for d in defects:
        by_frame.setdefault(d.frame_id, []).append(d)

    cap, _ = open_video_source(source)
    try:
        for frame_id, frame_defects in sorted(by_frame.items()):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id - 1)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            for d in frame_defects:
                _save_box_contour_snapshot(frame, d, snapshots_dir)
    finally:
        cap.release()


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


__all__ = [
    "BatchInspectionResult",
    "BarResult",
    "CollectedBar",
    "run_batch_inspection",
]
