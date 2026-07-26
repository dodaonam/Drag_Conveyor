from __future__ import annotations

import json
import logging
import math
import shutil
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2

from path_bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

import db
import preprocess
import r2
import settings

LOGGER = logging.getLogger(__name__)


_worker_wakeup = threading.Event()
_start_lock = threading.Lock()
_started = False


def wake() -> None:
    _worker_wakeup.set()


# ── Summary builder ───────────────────────────────────────────────────────────

def _build_summary(result, defect_keys: list[str], normal_keys: list[str]) -> dict:
    def snapshot_name(bar) -> str:
        """Match Geometry V2's snapshot frame, not its first observation frame."""
        frame_id = bar.snapshot_metadata.get("primary_source_frame_id")
        if not isinstance(frame_id, int):
            frame_id = bar.frame_id
        return f"track_{bar.track_id:06d}_frame_{frame_id:09d}.jpg"

    defects = []
    for bar in result.bars:
        if bar.result != "suspected_defect":
            continue
        snap_name = snapshot_name(bar)
        snap_key = next((k for k in defect_keys if k.endswith(snap_name)), None)
        defects.append({
            "bar_id": bar.bar_id,
            "track_id": bar.track_id,
            "frame_id": bar.frame_id,
            "reasons": bar.reasons,
            "rule_result": bar.rule_result or bar.result,
            "defect_type": bar.defect_type,
            "score": bar.score,
            "vlm_called": bar.vlm_called,
            "length": bar.measurements.get("length", 0.0),
            "width": bar.measurements.get("width", 0.0),
            "thresholds": bar.thresholds,
            "margins": bar.margins,
            "snapshot_key": snap_key,
            "paddle_id": bar.paddle_id,
            "track_ids": list(bar.track_ids),
            "vision_status": bar.vision_status,
            "final_reviewed_status": bar.final_reviewed_status,
            "classification_source": bar.classification_source,
            "decision_confidence": bar.decision_confidence,
            "evidence_support_score": bar.evidence_support_score,
            "suspected_breakage": bar.suspected_breakage,
            "possible_breakage_statuses": list(bar.possible_breakage_statuses),
            "review_required": bar.review_required,
            "geometry_analysis": bar.geometry_analysis,
            "snapshot_metadata": bar.snapshot_metadata,
            "legacy_measurements_available": bar.legacy_measurements_available,
        })
    normals = []
    for bar in result.bars:
        if bar.result != "normal":
            continue
        snap_name = snapshot_name(bar)
        snap_key = next((k for k in normal_keys if k.endswith(snap_name)), None)
        normals.append({
            "bar_id": bar.bar_id,
            "track_id": bar.track_id,
            "frame_id": bar.frame_id,
            "reasons": bar.reasons,
            "rule_result": bar.rule_result or bar.result,
            "defect_type": bar.defect_type,
            "score": bar.score,
            "vlm_called": bar.vlm_called,
            "length": bar.measurements.get("length", 0.0),
            "width": bar.measurements.get("width", 0.0),
            "thresholds": bar.thresholds,
            "margins": bar.margins,
            "snapshot_key": snap_key,
            "paddle_id": bar.paddle_id,
            "track_ids": list(bar.track_ids),
            "vision_status": bar.vision_status,
            "final_reviewed_status": bar.final_reviewed_status,
            "classification_source": bar.classification_source,
            "decision_confidence": bar.decision_confidence,
            "evidence_support_score": bar.evidence_support_score,
            "suspected_breakage": bar.suspected_breakage,
            "possible_breakage_statuses": list(bar.possible_breakage_statuses),
            "review_required": bar.review_required,
            "geometry_analysis": bar.geometry_analysis,
            "snapshot_metadata": bar.snapshot_metadata,
            "legacy_measurements_available": bar.legacy_measurements_available,
        })
    summary = {
        "total_bars": result.total_bars,
        "normal_bars": result.normal_bars,
        "defect_bars": result.defect_bars,
        "frames_scanned": result.frames_scanned,
        "inlier_count": result.inlier_count,
        "outlier_count": result.outlier_count,
        "inlier_ratio": result.inlier_ratio,
        "vlm_request_count": result.vlm_request_count,
        "inspection_mode": result.inspection_mode,
        "paddle_schema_version": result.paddle_schema_version,
        "summary_schema_version": result.summary_schema_version,
        "rule_version": result.rule_version,
        "confirmed_defect_bars": result.confirmed_defect_bars,
        "uncertain_bars": result.uncertain_bars,
        "review_required_bars": result.review_required_bars,
        "status_counts": result.status_counts,
        "geometry_diagnostics": result.geometry_diagnostics,
        "model_metadata": result.model_metadata,
        "geometry_metadata": result.geometry_metadata,
        "capability_metadata": result.capability_metadata,
        "timestamp_source": result.timestamp_source,
        "count_certified": result.success,
        "possible_event_count_min": result.total_bars if result.success else None,
        "possible_event_count_max": result.total_bars if result.success else None,
        "report_export_allowed": result.success,
        "failure_reason": result.failure_reason,
        "defects": defects,
        "normals": normals,
    }
    _assert_plain_finite_json(summary)
    return summary


def _assert_plain_finite_json(value: object, path: str = "$") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"result_serialization_failed: non-finite number at {path}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_plain_finite_json(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"result_serialization_failed: non-string key at {path}")
            _assert_plain_finite_json(item, f"{path}.{key}")
        return
    raise ValueError(f"result_serialization_failed: unsupported type at {path}")


# ── Job processor ─────────────────────────────────────────────────────────────

def _process_job(job_id: str) -> None:
    # Import here to avoid circular import at module load time
    from drag_conveyor.app.batch import run_batch_inspection
    from drag_conveyor.config import load_profile

    row = db.get_job(job_id)
    if row is None:
        LOGGER.error("[%s] Job not found in DB", job_id)
        return

    roi_config: dict = json.loads(row["roi_config_json"])
    temp_job_dir = settings.TEMP_DIR / job_id

    try:
        # 1. Source video: direct local path for Local Runner, R2 otherwise.
        is_local_source = str(row["object_key"]).startswith("local:")
        if is_local_source:
            video_path = Path(str(row["object_key"])[len("local:"):])
            try:
                video_path.resolve().relative_to(settings.LOCAL_MEDIA_ROOT)
            except ValueError as exc:
                raise RuntimeError("Local video is outside LOCAL_MEDIA_ROOT") from exc
            if not video_path.is_file():
                raise RuntimeError("Local video no longer exists")
            LOGGER.info("[%s] Using local video %s", job_id, video_path)
        else:
            ext = Path(row["object_key"]).suffix  # .mp4 / .webm / .mov
            video_path = temp_job_dir / f"input{ext}"
            LOGGER.info("[%s] Downloading %s", job_id, row["object_key"])
            r2.download_file(row["object_key"], video_path)

        # 2. Validate video is readable
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError("Cannot open video file — unsupported format or corrupt")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        LOGGER.info("[%s] Video OK: %d frames", job_id, total_frames)

        # 2.5 Slow the conveyor down before inference (interpolated frames) so each
        #     bar moves less per frame → more stable tracking. Falls back to the
        #     original video if preprocessing fails for any reason.
        db.update_status(job_id, "processing", db.now())
        inference_source = video_path
        inspection_mode = str(row["inspection_mode"] or "auto_baseline")
        if inspection_mode != "geometry_v2" and preprocess.SLOWDOWN_FACTOR < 1.0:
            try:
                slow_path = temp_job_dir / "slowmo.mp4"
                preprocess.slow_down_video(video_path, slow_path)
                inference_source = slow_path
                LOGGER.info("[%s] Slow-motion applied (%.2fx)", job_id, preprocess.SLOWDOWN_FACTOR)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("[%s] Slow-motion preprocessing failed, using original: %s", job_id, exc)
                inference_source = video_path

        # 3. Load base profile and apply client ROI
        profile = load_profile(settings.BASE_PROFILE_PATH)
        inspection_mode = str(row["inspection_mode"] or profile.inspection.mode)
        geometry_input = roi_config.pop("geometry", None)
        profile = profile.with_roi(roi_config)

        # 4. Run batch inspection
        LOGGER.info("[%s] Starting inspection", job_id)
        result = run_batch_inspection(
            profile=profile,
            source=str(inference_source),
            run_id=job_id,
            snapshots_root=temp_job_dir / "snapshots",
            inspection_mode=inspection_mode,
            geometry_input=geometry_input,
        )
        LOGGER.info(
            "[%s] Inspection done: success=%s, total=%d, defects=%d",
            job_id, result.success, result.total_bars, result.defect_bars,
        )

        # 5. Upload results to R2 — MUST happen before any cleanup
        defect_keys: list[str] = []
        if result.defect_snapshots_dir and result.defect_snapshots_dir.exists():
            for img in sorted(result.defect_snapshots_dir.glob("*.jpg")):
                key = f"local:{img.resolve()}" if is_local_source else f"results/{job_id}/snapshots/defects/{img.name}"
                if not is_local_source:
                    r2.upload_file(img, key, "image/jpeg")
                defect_keys.append(key)
            LOGGER.info("[%s] Defect snapshots ready: %d", job_id, len(defect_keys))
        normal_keys: list[str] = []
        if result.normal_snapshots_dir and result.normal_snapshots_dir.exists():
            for img in sorted(result.normal_snapshots_dir.glob("*.jpg")):
                key = f"local:{img.resolve()}" if is_local_source else f"results/{job_id}/snapshots/normals/{img.name}"
                if not is_local_source:
                    r2.upload_file(img, key, "image/jpeg")
                normal_keys.append(key)
            LOGGER.info("[%s] Normal snapshots ready: %d", job_id, len(normal_keys))

        # 6. Save summary + mark completed (or failed if inspection itself failed)
        summary = _build_summary(result, defect_keys, normal_keys)
        db.save_result(
            job_id=job_id,
            summary=summary,
            now=db.now(),
            success=result.success,
        )

        # 7. Cleanup — only after SQLite write confirmed
        if settings.DELETE_VIDEO_AFTER_SUCCESS and result.success and not is_local_source:
            try:
                r2.delete_object(row["object_key"])
                LOGGER.info("[%s] Video deleted from R2", job_id)
            except Exception as exc:
                LOGGER.warning("[%s] Could not delete video from R2: %s", job_id, exc)

        if not is_local_source:
            shutil.rmtree(temp_job_dir, ignore_errors=True)
        LOGGER.info("[%s] Done", job_id)

    except Exception as exc:
        LOGGER.exception("[%s] Job failed: %s", job_id, exc)
        db.update_status(job_id, "failed", db.now(), error_message=str(exc))
        # Keep temp files on failure — useful for post-mortem


# ── Worker thread ─────────────────────────────────────────────────────────────

def _claim_and_process_next_job() -> bool:
    job_id = db.claim_next_uploaded_job(db.now())
    if job_id is None:
        return False

    LOGGER.info("Starting job: %s", job_id)
    _process_job(job_id)
    return True


def _worker_loop() -> None:
    LOGGER.info("Inspection worker ready")
    while True:
        try:
            if _claim_and_process_next_job():
                continue
            _worker_wakeup.wait(timeout=1.0)
        except Exception as exc:
            LOGGER.exception("Unhandled exception in worker loop: %s", exc)
        finally:
            _worker_wakeup.clear()


# ── Cleanup thread ────────────────────────────────────────────────────────────

def _run_cleanup() -> None:
    now_dt = datetime.now(timezone.utc)
    now_iso = now_dt.isoformat()

    upload_cutoff = (now_dt - timedelta(seconds=settings.UPLOAD_EXPIRE_SECONDS)).isoformat()
    for jid in db.expire_stale_uploads(upload_cutoff, now_iso):
        LOGGER.info("[%s] upload_expired (no upload after %ds)", jid, settings.UPLOAD_EXPIRE_SECONDS)

    proc_cutoff = (now_dt - timedelta(seconds=settings.MAX_JOB_DURATION_SECONDS)).isoformat()
    for jid in db.timeout_processing(proc_cutoff, now_iso):
        LOGGER.warning(
            "[%s] failed: processing timeout (>%ds)", jid, settings.MAX_JOB_DURATION_SECONDS
        )


def _cleanup_loop() -> None:
    LOGGER.info("Job cleanup loop ready (upload_expire=%ds, proc_timeout=%ds)",
                settings.UPLOAD_EXPIRE_SECONDS, settings.MAX_JOB_DURATION_SECONDS)
    while True:
        time.sleep(60)
        try:
            _run_cleanup()
        except Exception as exc:
            LOGGER.exception("Cleanup error: %s", exc)


def start() -> None:
    global _started
    with _start_lock:
        if _started:
            wake()
            return

        t = threading.Thread(target=_worker_loop, daemon=True, name="inspection-worker")
        t.start()
        tc = threading.Thread(target=_cleanup_loop, daemon=True, name="job-cleanup")
        tc.start()
        _started = True

    wake()
