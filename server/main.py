from __future__ import annotations

import json
import logging
import math
import re
import secrets
import sys
import unicodedata
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

_TZ_ICT = timezone(timedelta(hours=7))
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlencode

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from path_bootstrap import ensure_repo_root_on_path, RUNTIME_DIR

ensure_repo_root_on_path()

import db
import excel_log
import r2
import report
import settings
import worker
from drag_conveyor.config import Profile, load_profile
from drag_conveyor.inspection_modes import is_supported_inspection_mode
from drag_conveyor.geometry_v2.config_loader import load_geometry_config

# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    import os as _os
    import time as _time
    _gui_log = _os.environ.get("GUI_LOG_PATH", "")
    if _gui_log:
        _log_path = Path(_gui_log)
    else:
        _log_dir = RUNTIME_DIR / "logs"
        _log_dir.mkdir(parents=True, exist_ok=True)
        _log_path = _log_dir / _time.strftime("%d-%m-%Y_%H%M%S.log")

    _fmt = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    _fh = logging.FileHandler(_log_path, encoding="utf-8")
    _fh.setFormatter(_fmt)
    _root = logging.getLogger()
    _root.setLevel(logging.INFO)
    _root.addHandler(_fh)
    if not getattr(sys, "frozen", False):
        _sh = logging.StreamHandler()
        _sh.setFormatter(_fmt)
        _root.addHandler(_sh)

    _logger = logging.getLogger(__name__)
    _logger.info("Server starting")
    db.init_db()
    settings.TEMP_DIR.mkdir(parents=True, exist_ok=True)
    worker.start()
    _logger.info("Server ready")
    yield
    _logger.info("Server shutting down")
    _root.removeHandler(_fh)
    _fh.close()


app = FastAPI(title="Drag Conveyor Inspection Server", lifespan=lifespan)

# ── Static assets (frontend CSS/JS) ─────────────────────────────────────────────
_STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

# ── Auth ──────────────────────────────────────────────────────────────────────

def require_auth(authorization: str | None = Header(default=None)) -> None:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    if authorization.removeprefix("Bearer ").strip() != settings.API_AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")


# ── Helpers ───────────────────────────────────────────────────────────────────

_ALLOWED_CONTENT_TYPES = {"video/mp4", "video/webm", "video/quicktime"}

_EXT_MAP = {"video/mp4": "mp4", "video/webm": "webm", "video/quicktime": "mov"}
_LOCAL_VIDEO_EXTENSIONS = {".mp4": "video/mp4", ".webm": "video/webm", ".mov": "video/quicktime"}

_STATUS_MESSAGES = {
    "waiting_upload": "Đang chờ upload video...",
    "upload_expired": "Presigned URL đã hết hạn, vui lòng tạo job mới.",
    "uploaded": "Upload hoàn tất, đang chờ xử lý...",
    "downloading": "Đang tải video về server...",
    "processing": "Đang phân tích video...",
    "completed": "Hoàn tất.",
    "failed": "Xử lý thất bại.",
}


def _normalize_name(name: str) -> str:
    s = name.strip().lower().replace('đ', 'd')
    s = unicodedata.normalize('NFD', s)
    s = ''.join(c for c in s if unicodedata.category(c) != 'Mn')
    s = re.sub(r'[^a-z0-9]', '', s)
    return s or 'x'


def _load_base_profile() -> Profile:
    try:
        return load_profile(settings.BASE_PROFILE_PATH)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Invalid base profile: {exc}") from exc


def _local_media_path(relative_path: str) -> Path:
    candidate = (settings.LOCAL_MEDIA_ROOT / relative_path).resolve()
    try:
        candidate.relative_to(settings.LOCAL_MEDIA_ROOT)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="Local path is outside LOCAL_MEDIA_ROOT") from exc
    return candidate

# ── Request / Response models ─────────────────────────────────────────────────

class RoiIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x: int
    y: int
    w: int
    h: int
    frame_width: int
    frame_height: int

    @field_validator("x", "y")
    @classmethod
    def _check_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("must be >= 0")
        return v

    @field_validator("w", "h", "frame_width", "frame_height")
    @classmethod
    def _check_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("must be > 0")
        return v

    @model_validator(mode="after")
    def _check_inside_frame(self) -> "RoiIn":
        if self.x + self.w > self.frame_width or self.y + self.h > self.frame_height:
            raise ValueError("ROI extends outside frame bounds")
        return self


class GeometryPointIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    x: float
    y: float

    @field_validator("x", "y", mode="before")
    @classmethod
    def _finite_number(cls, value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("must be a finite number")
        return float(value)


class GeometryIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["geometry_input/2.0"]
    chain_centerline: dict[str, GeometryPointIn]
    chain_band_width_ratio: float | None = None
    motion_direction: Literal["positive_s", "negative_s"] | None = None

    @field_validator("chain_band_width_ratio", mode="before")
    @classmethod
    def _finite_ratio(cls, value: object) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("must be a finite number")
        return float(value)

    @model_validator(mode="after")
    def _check_shape(self) -> "GeometryIn":
        if set(self.chain_centerline) != {"top", "bottom"}:
            raise ValueError("invalid geometry_v2 input")
        if self.chain_band_width_ratio is not None and not 0.02 <= self.chain_band_width_ratio <= 0.20:
            raise ValueError("chain_band_width_ratio must be in [0.02, 0.20]")
        return self


class CreateJobIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content_type: str
    size_bytes: int
    roi: RoiIn
    inspector_name: str
    conveyor_name: str
    inspection_mode: str | None = None
    geometry: GeometryIn | None = None

    @field_validator("inspection_mode")
    @classmethod
    def _check_inspection_mode(cls, v: str | None) -> str | None:
        if v is None:
            return None
        if not is_supported_inspection_mode(v):
            raise ValueError(f"unsupported inspection_mode: {v}")
        return v

    @model_validator(mode="after")
    def _geometry_matches_mode(self) -> "CreateJobIn":
        if self.inspection_mode == "geometry_v2" and self.geometry is None:
            raise ValueError("geometry is required for geometry_v2")
        if self.inspection_mode != "geometry_v2" and self.geometry is not None:
            raise ValueError("geometry is only valid for geometry_v2")
        if self.geometry is not None:
            top = self.geometry.chain_centerline["top"]
            bottom = self.geometry.chain_centerline["bottom"]
            if not (0.0 <= top.x <= self.roi.w and 0.0 <= bottom.x <= self.roi.w and 0.0 <= top.y <= self.roi.h and 0.0 <= bottom.y <= self.roi.h):
                raise ValueError("geometry centerline must be ROI-local and inside ROI bounds")
            vertical_span = abs(bottom.y - top.y)
            if vertical_span < 0.70 * self.roi.h:
                raise ValueError("centerline span must be at least 70% of ROI height")
            if math.degrees(math.atan2(abs(bottom.x - top.x), vertical_span)) > 15.0:
                raise ValueError("centerline roll must not exceed 15 degrees")
        return self


class CreateJobOut(BaseModel):
    job_id: str
    presigned_put_url: str
    expires_in: int


class LocalJobIn(CreateJobIn):
    path: str


class StatusOut(BaseModel):
    job_id: str
    status: str
    message: str
    updated_at: str


class CorrectionIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    track_id: int
    defect_type: str


class ReportIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    inspector_name: str
    conveyor_name: str
    corrections: list[CorrectionIn]
    expected_review_revision: int = 0


# ── GET / (frontend) ──────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def serve_frontend() -> HTMLResponse:
    html = (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(content=html)


# ── GET /api/health ────────────────────────────────────────────────────────────

@app.get("/api/health", dependencies=[Depends(require_auth)])
def health() -> dict[str, str]:
    return {"status": "ok"}


# ── GET /api/runtime-config ─────────────────────────────────────────────────

@app.get("/api/runtime-config", dependencies=[Depends(require_auth)])
def runtime_config() -> dict[str, Any]:
    profile = _load_base_profile()
    band = profile.collection.trigger_band
    config, _ = load_geometry_config(Path(__file__).resolve().parents[1] / "config" / "geometry_v2.json")
    return {
        "inspection": {
            "mode": profile.inspection.mode,
            "supported_modes": ["auto_baseline", "average_ratio", "geometry_v2"],
        },
        "collection": {
            "trigger_band": {
                "position_ratio": band.position_ratio,
                "thickness_ratio": band.thickness_ratio,
                "min_overlap_ratio": band.min_overlap_ratio,
                "pending_ttl_frames": band.pending_ttl_frames,
                "allow_inside_band_trigger": band.allow_inside_band_trigger,
            }
        },
        "geometry_v2": {
            "available": True,
            "unavailable_reason": None,
            "input_schema_version": "geometry_input/2.0",
            "defaults": {"chain_band_width_ratio": config["geometry"]["default_chain_band_width_ratio"], "motion_direction": "positive_s"},
            "validation": {"minimum_centerline_span_ratio": config["geometry"]["minimum_centerline_span_ratio"], "maximum_allowed_roll_deg": config["geometry"]["maximum_allowed_roll_deg"], "minimum_chain_band_width_ratio": config["geometry"]["chain_band_width_ratio_min"], "maximum_chain_band_width_ratio": config["geometry"]["chain_band_width_ratio_max"]},
            "trigger": {"center_ratio": config["trigger"]["center_ratio"], "height_ratio": config["trigger"]["height_ratio"]},
        },
        "local_media": {"root_label": str(settings.LOCAL_MEDIA_ROOT), "enabled": settings.LOCAL_MEDIA_ROOT.exists()},
        "local_mode": settings.LOCAL_MODE,
    }


@app.get("/api/local/videos", dependencies=[Depends(require_auth)])
def list_local_videos(path: str = "") -> dict[str, Any]:
    directory = _local_media_path(path)
    if not directory.is_dir():
        raise HTTPException(status_code=404, detail="Local directory not found")
    entries = []
    for child in sorted(directory.iterdir(), key=lambda item: (not item.is_dir(), item.name.casefold())):
        if child.is_dir() or child.suffix.lower() in _LOCAL_VIDEO_EXTENSIONS:
            relative = str(child.relative_to(settings.LOCAL_MEDIA_ROOT))
            entries.append({"name": child.name, "path": relative, "is_dir": child.is_dir(), "size_bytes": None if child.is_dir() else child.stat().st_size, "content_type": None if child.is_dir() else _LOCAL_VIDEO_EXTENSIONS[child.suffix.lower()]})
    return {"root_label": str(settings.LOCAL_MEDIA_ROOT), "path": path, "parent": str(Path(path).parent) if path else None, "entries": entries}


@app.get("/api/local/video", include_in_schema=False)
def local_video(path: str, token: str) -> FileResponse:
    if token != settings.API_AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")
    source = _local_media_path(path)
    content_type = _LOCAL_VIDEO_EXTENSIONS.get(source.suffix.lower())
    if content_type is None or not source.is_file():
        raise HTTPException(status_code=404, detail="Local video not found")
    return FileResponse(source, media_type=content_type)


@app.get("/api/local/snapshot", include_in_schema=False)
def local_snapshot(path: str, token: str) -> FileResponse:
    if token != settings.API_AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")
    source = Path(path).resolve()
    try:
        source.relative_to(settings.TEMP_DIR)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="Snapshot path is outside local runtime") from exc
    if source.suffix.lower() != ".jpg" or not source.is_file():
        raise HTTPException(status_code=404, detail="Local snapshot not found")
    return FileResponse(source, media_type="image/jpeg")


# ── POST /api/jobs ─────────────────────────────────────────────────────────────

@app.post("/api/jobs", response_model=CreateJobOut, dependencies=[Depends(require_auth)])
def create_job(body: CreateJobIn) -> CreateJobOut:
    if settings.LOCAL_MODE:
        raise HTTPException(status_code=409, detail="Local Mode chỉ nhận video qua Chọn từ thư mục máy")
    profile = _load_base_profile()
    inspection_mode = body.inspection_mode or profile.inspection.mode
    if body.content_type not in _ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=422, detail=f"Unsupported content_type: {body.content_type}")
    if body.size_bytes <= 0:
        raise HTTPException(status_code=422, detail="size_bytes must be > 0")
    if body.size_bytes > settings.MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=422,
            detail=f"File too large: {body.size_bytes} bytes, max {settings.MAX_UPLOAD_BYTES}",
        )
    roi = body.roi

    now_dt = datetime.now(_TZ_ICT)
    job_id = "{}_{}_{}_{}_{}".format(
        now_dt.strftime('%d%m%Y'),
        _normalize_name(body.inspector_name),
        _normalize_name(body.conveyor_name),
        now_dt.strftime('%H%M%S'),
        secrets.token_hex(3),
    )
    ext = _EXT_MAP[body.content_type]
    object_key = f"uploads/{job_id}/input.{ext}"
    now = db.now()

    put_url = r2.presigned_put_url(
        object_key=object_key,
        content_type=body.content_type,
        expires=settings.PRESIGNED_PUT_EXPIRES,
    )

    db.create_job(
        job_id=job_id,
        status="waiting_upload",
        object_key=object_key,
        content_type=body.content_type,
        size_bytes=body.size_bytes,
        inspection_mode=inspection_mode,
        roi_config={**roi.model_dump(), **({"geometry": body.geometry.model_dump()} if body.geometry is not None else {})},
        now=now,
    )

    return CreateJobOut(job_id=job_id, presigned_put_url=put_url, expires_in=settings.PRESIGNED_PUT_EXPIRES)


@app.post("/api/local/jobs", response_model=CreateJobOut, dependencies=[Depends(require_auth)])
def create_local_job(body: LocalJobIn) -> CreateJobOut:
    source = _local_media_path(body.path)
    content_type = _LOCAL_VIDEO_EXTENSIONS.get(source.suffix.lower())
    if content_type is None or not source.is_file():
        raise HTTPException(status_code=422, detail="Unsupported local video")
    if source.stat().st_size <= 0 or source.stat().st_size > settings.MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=422, detail="Local video size is invalid")
    profile = _load_base_profile()
    inspection_mode = body.inspection_mode or profile.inspection.mode
    now_dt = datetime.now(_TZ_ICT)
    job_id = "{}_{}_{}_{}_{}".format(now_dt.strftime('%d%m%Y'), _normalize_name(body.inspector_name), _normalize_name(body.conveyor_name), now_dt.strftime('%H%M%S'), secrets.token_hex(3))
    now = db.now()
    db.create_job(job_id=job_id, status="uploaded", object_key=f"local:{source}", content_type=content_type, size_bytes=source.stat().st_size, inspection_mode=inspection_mode, roi_config={**body.roi.model_dump(), **({"geometry": body.geometry.model_dump()} if body.geometry is not None else {})}, now=now)
    db.mark_uploaded(job_id, now)
    worker.wake()
    return CreateJobOut(job_id=job_id, presigned_put_url="", expires_in=0)


# ── POST /api/jobs/{job_id}/upload-complete ────────────────────────────────────

@app.post("/api/jobs/{job_id}/upload-complete", dependencies=[Depends(require_auth)])
def upload_complete(job_id: str) -> dict[str, str]:
    row = db.get_job(job_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if row["status"] != "waiting_upload":
        raise HTTPException(status_code=409, detail=f"Job already in state: {row['status']}")
    if not r2.object_exists(row["object_key"]):
        raise HTTPException(status_code=422, detail="Video not found on R2 — upload may have failed")

    db.mark_uploaded(job_id, db.now())
    worker.wake()

    return {"status": "uploaded"}


# ── GET /api/jobs/{job_id}/status ─────────────────────────────────────────────

@app.get("/api/jobs/{job_id}/status", response_model=StatusOut, dependencies=[Depends(require_auth)])
def get_status(job_id: str) -> StatusOut:
    row = db.get_job(job_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Job not found")
    status = row["status"]
    msg = _STATUS_MESSAGES.get(status, status)
    if status == "failed" and row["error_message"]:
        msg = row["error_message"]
    return StatusOut(job_id=job_id, status=status, message=msg, updated_at=row["updated_at"])


# ── GET /api/jobs/{job_id}/result ─────────────────────────────────────────────

@app.get("/api/jobs/{job_id}/result", dependencies=[Depends(require_auth)])
def get_result(job_id: str) -> dict[str, Any]:
    row = db.get_job(job_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if row["status"] != "completed":
        raise HTTPException(status_code=409, detail=f"Job not completed: {row['status']}")

    summary: dict = json.loads(row["result_summary_json"])

    # Generate presigned GET URLs for each asset
    for defect in summary.get("defects", []):
        key = defect.get("snapshot_key")
        if key:
            defect["snapshot_url"] = f"/api/local/snapshot?{urlencode({'path': key[len('local:'):], 'token': settings.API_AUTH_TOKEN})}" if key.startswith("local:") else r2.presigned_get_url(key, settings.PRESIGNED_GET_EXPIRES)
    for normal in summary.get("normals", []):
        key = normal.get("snapshot_key")
        if key:
            normal["snapshot_url"] = f"/api/local/snapshot?{urlencode({'path': key[len('local:'):], 'token': settings.API_AUTH_TOKEN})}" if key.startswith("local:") else r2.presigned_get_url(key, settings.PRESIGNED_GET_EXPIRES)

    summary["job_id"] = job_id
    return summary


# ── POST /api/jobs/{job_id}/report ────────────────────────────────────────────

@app.post("/api/jobs/{job_id}/report", dependencies=[Depends(require_auth)])
def save_report(job_id: str, body: ReportIn) -> dict[str, Any]:
    row = db.get_job(job_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if row["status"] != "completed":
        raise HTTPException(status_code=409, detail=f"Job not completed: {row['status']}")

    summary = json.loads(row["result_summary_json"])
    if summary.get("inspection_mode") == "geometry_v2" and not summary.get("report_export_allowed", True):
        raise HTTPException(status_code=409, detail="Geometry V2 result is not eligible for report export")
    key_prefix = f"results/{job_id}/"

    def fetch_image(snapshot_key: str) -> bytes | None:
        if snapshot_key.startswith("local:"):
            local_snapshot_path = Path(snapshot_key[len("local:"):]).resolve()
            try:
                local_snapshot_path.relative_to(settings.TEMP_DIR)
                return local_snapshot_path.read_bytes()
            except (OSError, ValueError):
                return None
        if not snapshot_key or ".." in snapshot_key or not snapshot_key.startswith(key_prefix):
            return None
        try:
            return r2.download_bytes(snapshot_key)
        except Exception as exc:  # noqa: BLE001
            logging.getLogger(__name__).warning("report image fetch failed for %s: %s", snapshot_key, exc)
            return None

    meta = {
        "inspector_name": body.inspector_name,
        "conveyor_name": body.conveyor_name,
        "datetime_str": report.format_ict(row["created_at"]),
    }
    try:
        corrections = [c.model_dump() for c in body.corrections]
        filename = report.save_report(
            summary=summary,
            corrections=corrections,
            meta=meta,
            reports_dir=settings.REPORTS_DIR,
            fetch_image=fetch_image,
        )
        report_data = report.build_report_data(
            summary,
            corrections,
        )
        excel_filename = excel_log.save_inspection_log(
            report_data=report_data,
            job_id=job_id,
            inspected_at_iso=row["created_at"],
            inspector_name=body.inspector_name,
            conveyor_name=body.conveyor_name,
            pdf_filename=filename,
            reports_dir=settings.REPORTS_DIR,
        )
        if summary.get("inspection_mode") == "geometry_v2":
            saved = db.save_review_revision(
                job_id=job_id,
                expected_revision=body.expected_review_revision,
                reviewed_statuses=report_data["resolved_statuses"],
                reviewer=body.inspector_name,
                pdf_filename=filename,
                excel_filename=excel_filename,
                now=db.now(),
            )
            if not saved:
                try:
                    (settings.REPORTS_DIR / filename).unlink(missing_ok=True)
                except OSError:
                    logging.getLogger(__name__).exception("Unable to remove report after review CAS conflict")
                raise HTTPException(status_code=409, detail="Review revision conflict; refresh the job result and retry")
    except report.ReportError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except excel_log.ExcelLogError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"PDF đã được lưu nhưng không thể cập nhật Excel: {exc}",
        ) from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Cannot write report: {exc}") from exc

    return {"filename": filename, "excel_filename": excel_filename, "saved": True}
