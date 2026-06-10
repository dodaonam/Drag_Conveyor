from __future__ import annotations

import json
import logging
import shutil
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2

import db
import r2
import settings
from drag_conveyor.inspection_modes import DEFAULT_INSPECTION_MODE

LOGGER = logging.getLogger(__name__)

_UPLOAD_EXPIRE_SECONDS = 1800   # waiting_upload → upload_expired after 30 min

_worker_wakeup = threading.Event()
_start_lock = threading.Lock()
_started = False


def wake() -> None:
    _worker_wakeup.set()


# ── Summary builder ───────────────────────────────────────────────────────────

def _build_summary(result, snapshot_keys: list[str]) -> dict:
    defects = []
    for bar in result.bars:
        if bar.result != "suspected_defect":
            continue
        snap_name = f"track_{bar.track_id:06d}_frame_{bar.frame_id:09d}.jpg"
        snap_key = next((k for k in snapshot_keys if k.endswith(snap_name)), None)
        defects.append({
            "track_id": bar.track_id,
            "frame_id": bar.frame_id,
            "reasons": bar.reasons,
            "length": bar.measurements.get("length", 0.0),
            "width": bar.measurements.get("width", 0.0),
            "snapshot_key": snap_key,
        })
    return {
        "total_bars": result.total_bars,
        "normal_bars": result.normal_bars,
        "defect_bars": result.defect_bars,
        "frames_scanned": result.frames_scanned,
        "inlier_count": result.inlier_count,
        "outlier_count": result.outlier_count,
        "inlier_ratio": result.inlier_ratio,
        "failure_reason": result.failure_reason,
        "defects": defects,
    }


# ── Job processor ─────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _process_job(job_id: str) -> None:
    # Import here to avoid circular import at module load time
    from drag_conveyor.app.batch import run_batch_inspection
    from drag_conveyor.config import load_profile

    row = db.get_job(job_id)
    if row is None:
        LOGGER.error("[%s] Job not found in DB", job_id)
        return

    roi_config: dict = json.loads(row["roi_config_json"])
    inspection_mode = str(row["inspection_mode"] or DEFAULT_INSPECTION_MODE)
    temp_job_dir = settings.TEMP_DIR / job_id

    try:
        # 1. Download video from R2
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

        # 3. Load base profile and apply client ROI
        db.update_status(job_id, "processing", _now())
        profile = load_profile(settings.BASE_PROFILE_PATH)
        profile = profile.with_roi(roi_config)

        # 4. Run batch inspection
        temp_snapshots = temp_job_dir / "snapshots"
        LOGGER.info("[%s] Starting inspection", job_id)
        result = run_batch_inspection(
            profile=profile,
            source=str(video_path),
            model_path=str(settings.MODEL_PATH),
            run_id=job_id,
            defect_snapshots_root=temp_snapshots,
            inspection_mode=inspection_mode,
        )
        LOGGER.info(
            "[%s] Inspection done: success=%s, total=%d, defects=%d",
            job_id, result.success, result.total_bars, result.defect_bars,
        )

        # 5. Upload results to R2 — MUST happen before any cleanup
        snapshot_keys: list[str] = []
        if result.defect_snapshots_dir and result.defect_snapshots_dir.exists():
            for img in sorted(result.defect_snapshots_dir.glob("*.jpg")):
                key = f"results/{job_id}/snapshots/{img.name}"
                r2.upload_file(img, key, "image/jpeg")
                snapshot_keys.append(key)
            LOGGER.info("[%s] Snapshots uploaded: %d", job_id, len(snapshot_keys))

        # 6. Save summary + mark completed (or failed if inspection itself failed)
        summary = _build_summary(result, snapshot_keys)
        db.save_result(
            job_id=job_id,
            summary=summary,
            now=_now(),
            success=result.success,
        )

        # 7. Cleanup — only after SQLite write confirmed
        if settings.DELETE_VIDEO_AFTER_SUCCESS:
            try:
                r2.delete_object(row["object_key"])
                LOGGER.info("[%s] Video deleted from R2", job_id)
            except Exception as exc:
                LOGGER.warning("[%s] Could not delete video from R2: %s", job_id, exc)

        shutil.rmtree(temp_job_dir, ignore_errors=True)
        LOGGER.info("[%s] Done", job_id)

    except Exception as exc:
        LOGGER.exception("[%s] Job failed: %s", job_id, exc)
        db.update_status(job_id, "failed", _now(), error_message=str(exc))
        # Keep temp files on failure — useful for post-mortem


# ── Worker thread ─────────────────────────────────────────────────────────────

def _claim_and_process_next_job() -> bool:
    job_id = db.claim_next_uploaded_job(_now())
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

    upload_cutoff = (now_dt - timedelta(seconds=_UPLOAD_EXPIRE_SECONDS)).isoformat()
    for jid in db.expire_stale_uploads(upload_cutoff, now_iso):
        LOGGER.info("[%s] upload_expired (no upload after %ds)", jid, _UPLOAD_EXPIRE_SECONDS)

    proc_cutoff = (now_dt - timedelta(seconds=settings.MAX_JOB_DURATION_SECONDS)).isoformat()
    for jid in db.timeout_processing(proc_cutoff, now_iso):
        LOGGER.warning(
            "[%s] failed: processing timeout (>%ds)", jid, settings.MAX_JOB_DURATION_SECONDS
        )


def _cleanup_loop() -> None:
    LOGGER.info("Job cleanup loop ready (upload_expire=%ds, proc_timeout=%ds)",
                _UPLOAD_EXPIRE_SECONDS, settings.MAX_JOB_DURATION_SECONDS)
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
