from __future__ import annotations

import json
import logging
import re
import secrets
import unicodedata
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

_TZ_ICT = timezone(timedelta(hours=7))
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from path_bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

import db
import r2
import report
import settings
import worker
from drag_conveyor.config import Profile, load_profile
from drag_conveyor.inspection_modes import is_supported_inspection_mode

# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    db.init_db()
    settings.TEMP_DIR.mkdir(parents=True, exist_ok=True)
    worker.start()
    yield


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


class CreateJobIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content_type: str
    size_bytes: int
    roi: RoiIn
    inspector_name: str
    conveyor_name: str
    inspection_mode: str | None = None

    @field_validator("inspection_mode")
    @classmethod
    def _check_inspection_mode(cls, v: str | None) -> str | None:
        if v is None:
            return None
        if not is_supported_inspection_mode(v):
            raise ValueError(f"unsupported inspection_mode: {v}")
        return v


class CreateJobOut(BaseModel):
    job_id: str
    presigned_put_url: str
    expires_in: int


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
    return {
        "inspection": {
            "mode": profile.inspection.mode,
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
    }


# ── POST /api/jobs ─────────────────────────────────────────────────────────────

@app.post("/api/jobs", response_model=CreateJobOut, dependencies=[Depends(require_auth)])
def create_job(body: CreateJobIn) -> CreateJobOut:
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
        roi_config=roi.model_dump(),
        now=now,
    )

    return CreateJobOut(job_id=job_id, presigned_put_url=put_url, expires_in=settings.PRESIGNED_PUT_EXPIRES)


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
            defect["snapshot_url"] = r2.presigned_get_url(key, settings.PRESIGNED_GET_EXPIRES)
    for normal in summary.get("normals", []):
        key = normal.get("snapshot_key")
        if key:
            normal["snapshot_url"] = r2.presigned_get_url(key, settings.PRESIGNED_GET_EXPIRES)

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
    key_prefix = f"results/{job_id}/"

    def fetch_image(snapshot_key: str) -> bytes | None:
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
        filename = report.save_report(
            summary=summary,
            corrections=[c.model_dump() for c in body.corrections],
            meta=meta,
            job_id=job_id,
            created_at_iso=row["created_at"],
            reports_dir=settings.REPORTS_DIR,
            fetch_image=fetch_image,
        )
    except report.ReportError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Cannot write report: {exc}") from exc

    return {"filename": filename, "saved": True}
