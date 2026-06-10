from __future__ import annotations

import json
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

import db
import r2
import settings
import worker

# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    db.init_db()
    settings.TEMP_DIR.mkdir(parents=True, exist_ok=True)
    worker.start()
    yield


app = FastAPI(title="Drag Conveyor Inspection Server", lifespan=lifespan)

# ── Auth ──────────────────────────────────────────────────────────────────────

def require_auth(authorization: str | None = Header(default=None)) -> None:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    if authorization.removeprefix("Bearer ").strip() != settings.API_AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    filename: str
    roi: RoiIn


class CreateJobOut(BaseModel):
    job_id: str
    presigned_put_url: str
    expires_in: int


class StatusOut(BaseModel):
    job_id: str
    status: str
    message: str
    updated_at: str


# ── GET / (frontend) ──────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def serve_frontend() -> HTMLResponse:
    html = (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(content=html)


# ── GET /api/health ────────────────────────────────────────────────────────────

@app.get("/api/health", dependencies=[Depends(require_auth)])
def health() -> dict[str, str]:
    return {"status": "ok"}


# ── POST /api/jobs ─────────────────────────────────────────────────────────────

@app.post("/api/jobs", response_model=CreateJobOut, dependencies=[Depends(require_auth)])
def create_job(body: CreateJobIn) -> CreateJobOut:
    if body.content_type not in _ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=422, detail=f"Unsupported content_type: {body.content_type}")
    if body.size_bytes > settings.MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=422,
            detail=f"File too large: {body.size_bytes} bytes, max {settings.MAX_UPLOAD_BYTES}",
        )
    roi = body.roi

    job_id = uuid.uuid4().hex
    ext = _EXT_MAP[body.content_type]
    object_key = f"uploads/{job_id}/input.{ext}"
    now = _now()

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

    db.mark_uploaded(job_id, _now())
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
    if row["result_csv_key"]:
        summary["csv_url"] = r2.presigned_get_url(row["result_csv_key"], settings.PRESIGNED_GET_EXPIRES)

    for defect in summary.get("defects", []):
        key = defect.get("snapshot_key")
        if key:
            defect["snapshot_url"] = r2.presigned_get_url(key, settings.PRESIGNED_GET_EXPIRES)

    summary["job_id"] = job_id
    return summary
