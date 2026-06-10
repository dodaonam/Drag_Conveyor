from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")


def _req(key: str) -> str:
    val = os.environ.get(key, "").strip()
    if not val:
        raise RuntimeError(f"Missing required env var: {key}")
    return val


R2_ENDPOINT_URL = _req("R2_ENDPOINT_URL")
R2_ACCESS_KEY_ID = _req("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = _req("R2_SECRET_ACCESS_KEY")
R2_BUCKET_NAME = _req("R2_BUCKET_NAME")

API_AUTH_TOKEN = _req("API_AUTH_TOKEN")

_SERVER_DIR = Path(__file__).parent

BASE_PROFILE_PATH = (_SERVER_DIR / os.environ.get("BASE_PROFILE_PATH", "../config/base_profile.json")).resolve()
MODEL_PATH = (_SERVER_DIR / os.environ.get("MODEL_PATH", "../weights/best.onnx")).resolve()
TEMP_DIR = (_SERVER_DIR / os.environ.get("TEMP_DIR", "runtime/temp")).resolve()

MAX_UPLOAD_BYTES: int = int(os.environ.get("MAX_UPLOAD_BYTES", 209_715_200))
MAX_JOB_DURATION_SECONDS: int = int(os.environ.get("MAX_JOB_DURATION_SECONDS", 900))
PRESIGNED_PUT_EXPIRES: int = int(os.environ.get("PRESIGNED_PUT_EXPIRES_SECONDS", 1800))
PRESIGNED_GET_EXPIRES: int = int(os.environ.get("PRESIGNED_GET_EXPIRES_SECONDS", 3600))
DELETE_VIDEO_AFTER_SUCCESS: bool = (
    os.environ.get("DELETE_VIDEO_AFTER_SUCCESS", "true").strip().lower() == "true"
)
