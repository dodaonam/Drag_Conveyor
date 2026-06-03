from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import cv2

from .camera_io import is_camera_index_source, open_capture_source
from .calibration import CalibrationEngine, CalibrationOutcome
from .config import Profile
from .inference import OnnxRuntimeEngine, postprocess_segmentation, preprocess_roi
from .measure import measure_contour
from .tracking import CentroidTracker
from .trigger import TriggerEngine, build_trigger_band, mask_overlap_ratio_with_band


@dataclass(frozen=True, slots=True)
class CalibrationRunResult:
    outcome: CalibrationOutcome
    records_collected: int
    frames_scanned: int
    video_loops_completed: int
    artifacts_dir: Path | None
    records_csv_path: Path | None
    report_json_path: Path | None


def build_calibration_artifacts_dir(runtime_dir: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return runtime_dir / "calibration" / ts


def run_auto_calibration(
    *,
    profile: Profile,
    source: str,
    model_path: str,
    max_frames: int,
    show_preview: bool = False,
    artifacts_dir: Path | None = None,
    loop_video: bool = False,
) -> CalibrationRunResult:
    engine = OnnxRuntimeEngine()
    engine.load(model_path, profile.model)

    tracker = CentroidTracker(
        max_jump_px=profile.tracker.max_jump_px,
        ttl_frames=profile.tracker.ttl_frames,
        min_hits=profile.tracker.min_hits,
        direction=profile.inspection_region.direction,
    )
    band = build_trigger_band(profile.inspection_region)
    trigger_engine = TriggerEngine(
        pending_ttl_frames=profile.inspection_region.trigger_band.pending_ttl_frames,
        allow_inside_band_trigger=profile.inspection_region.trigger_band.allow_inside_band_trigger,
    )

    cap = _open_capture_simple(source, profile=profile)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    records: list[dict[str, float]] = []
    records_with_meta: list[dict[str, float | int]] = []
    frame_count = 0
    video_loops_completed = 0

    region = profile.inspection_region
    should_loop_video = loop_video and not is_camera_index_source(source)

    try:
        while frame_count < max_frames and len(records) < profile.calibration.target_valid_records:
            ok, frame = cap.read()
            if not ok or frame is None:
                if should_loop_video:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    tracker.reset()
                    trigger_engine.reset()
                    video_loops_completed += 1
                    continue
                break
            frame_count += 1

            if frame.shape[1] != region.frame_width or frame.shape[0] != region.frame_height:
                raise RuntimeError(
                    "Frame size mismatch with profile setup size during calibration. "
                    "Remap inspection region before calibrating."
                )

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
                should_trigger = trigger_engine.should_trigger(
                    track_id=track.track_id,
                    prev_xy=track.prev_centroid_xy,
                    curr_xy=track.centroid_xy,
                    direction=region.direction,
                    centerline=band.centerline,
                    band=band,
                    overlap_ratio=overlap,
                    min_overlap_ratio=region.trigger_band.min_overlap_ratio,
                )
                if not should_trigger:
                    continue

                trigger_engine.mark_processed(track.track_id)
                measurement = measure_contour(track.detection.contour_frame).to_dict()
                records.append(measurement)
                records_with_meta.append(
                    {
                        "frame_id": frame_count,
                        "track_id": track.track_id,
                        "overlap_ratio": float(overlap),
                        "area": float(measurement["area"]),
                        "length": float(measurement["length"]),
                        "width": float(measurement["width"]),
                        "aspect_ratio": float(measurement["aspect_ratio"]),
                    }
                )

            if show_preview:
                preview = frame.copy()
                cv2.rectangle(
                    preview,
                    (region.x, region.y),
                    (region.x + region.w, region.y + region.h),
                    (255, 180, 0),
                    2,
                )
                cv2.rectangle(preview, (band.x1, band.y1), (band.x2, band.y2), (0, 255, 255), 2)
                cv2.putText(
                    preview,
                    f"calibration records={len(records)}/{profile.calibration.target_valid_records}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )
                cv2.imshow("Calibration", preview)
                if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                    break

    finally:
        cap.release()
        engine.close()
        if show_preview:
            cv2.destroyAllWindows()

    outcome = CalibrationEngine().calibrate(records, profile)
    records_csv_path: Path | None = None
    report_json_path: Path | None = None
    out_dir: Path | None = artifacts_dir.resolve() if artifacts_dir is not None else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        records_csv_path = out_dir / "calibration_records.csv"
        report_json_path = out_dir / "calibration_report.json"
        _write_calibration_records_csv(records_csv_path, records_with_meta)
        _write_calibration_report_json(
            path=report_json_path,
            outcome=outcome,
            profile=profile,
            source=source,
            model_path=model_path,
            frames_scanned=frame_count,
            records_collected=len(records),
            records_csv_path=records_csv_path,
            loop_video=bool(should_loop_video),
            video_loops_completed=video_loops_completed,
        )

    return CalibrationRunResult(
        outcome=outcome,
        records_collected=len(records),
        frames_scanned=frame_count,
        video_loops_completed=video_loops_completed,
        artifacts_dir=out_dir,
        records_csv_path=records_csv_path,
        report_json_path=report_json_path,
    )


def _open_capture_simple(source: str, profile: Profile) -> cv2.VideoCapture:
    cap, _backend_name = open_capture_source(
        source,
        width=profile.camera.width,
        height=profile.camera.height,
        fps=profile.camera.fps,
        buffersize=1,
    )
    return cap


def _write_calibration_records_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    columns = [
        "frame_id",
        "track_id",
        "overlap_ratio",
        "area",
        "length",
        "width",
        "aspect_ratio",
    ]
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_calibration_report_json(
    *,
    path: Path,
    outcome: CalibrationOutcome,
    profile: Profile,
    source: str,
    model_path: str,
    frames_scanned: int,
    records_collected: int,
    records_csv_path: Path,
    loop_video: bool = False,
    video_loops_completed: int = 0,
) -> None:
    payload: dict[str, object] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "success": bool(outcome.success),
        "reason": str(outcome.reason),
        "source": source,
        "model_path": model_path,
        "frames_scanned": int(frames_scanned),
        "records_collected": int(records_collected),
        "loop_video": bool(loop_video),
        "video_loops_completed": int(video_loops_completed),
        "target_valid_records": int(profile.calibration.target_valid_records),
        "min_valid_records": int(profile.calibration.min_valid_records),
        "min_inlier_ratio": float(profile.calibration.min_inlier_ratio),
        "max_outlier_ratio": float(profile.calibration.max_outlier_ratio),
        "baseline_updated": bool(outcome.success and outcome.updated_profile is not None),
        "kept_previous_baseline": bool((not outcome.success) and (profile.calibration_result is not None)),
        "records_csv_path": str(records_csv_path),
    }
    if outcome.calibration_result is not None:
        payload["calibration_result"] = asdict(outcome.calibration_result)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
