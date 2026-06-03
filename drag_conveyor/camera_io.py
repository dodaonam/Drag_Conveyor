from __future__ import annotations

from dataclasses import dataclass
import sys

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class CameraProbe:
    index: int
    backend: str
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class VideoSourceInfo:
    path: str
    width: int
    height: int
    fps: float
    first_frame_bgr: np.ndarray


def is_camera_index_source(source: str) -> bool:
    return source.strip().isdigit()


def open_capture_source(
    source: str,
    *,
    width: int | None = None,
    height: int | None = None,
    fps: int | None = None,
    buffersize: int | None = 1,
) -> tuple[cv2.VideoCapture, str]:
    source = source.strip()

    if is_camera_index_source(source):
        _ensure_windows_camera_runtime()
        cap, backend_name = _open_camera_index_windows_first(int(source))
        if cap.isOpened():
            _apply_camera_props(cap, width=width, height=height, fps=fps, buffersize=buffersize)
        return cap, backend_name

    return cv2.VideoCapture(source), "FILE"


def probe_camera_indices(
    max_index: int,
    *,
    width: int | None = None,
    height: int | None = None,
    fps: int | None = None,
) -> list[CameraProbe]:
    found: list[CameraProbe] = []
    for index in range(max_index):
        cap, backend_name = open_capture_source(
            str(index),
            width=width,
            height=height,
            fps=fps,
            buffersize=1,
        )
        if not cap.isOpened():
            cap.release()
            continue

        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            continue

        h, w = frame.shape[:2]
        found.append(CameraProbe(index=index, backend=backend_name, width=w, height=h))
        cap.release()

    return found


def probe_video_source(source: str) -> VideoSourceInfo:
    source = source.strip()
    if is_camera_index_source(source):
        raise RuntimeError("Expected a video file path, got camera index source")

    cap = cv2.VideoCapture(source)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Khong mo duoc video: {source}")

        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Khong doc duoc frame dau tien tu video: {source}")

        h, w = frame.shape[:2]
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0:
            fps = 30.0
        return VideoSourceInfo(
            path=source,
            width=int(w),
            height=int(h),
            fps=fps,
            first_frame_bgr=frame.copy(),
        )
    finally:
        cap.release()


def _open_camera_index_windows_first(index: int) -> tuple[cv2.VideoCapture, str]:
    dshow = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if dshow.isOpened():
        return dshow, "DSHOW"
    dshow.release()

    msmf = cv2.VideoCapture(index, cv2.CAP_MSMF)
    if msmf.isOpened():
        return msmf, "MSMF"
    # Keep runtime strict to Windows design spec: DSHOW primary, MSMF fallback.
    # Do not introduce additional backend fallback in production path.
    return msmf, "MSMF"


def _ensure_windows_camera_runtime() -> None:
    if not sys.platform.startswith("win"):
        raise RuntimeError(
            "Camera index source chỉ hỗ trợ Windows theo thiết kế (CAP_DSHOW -> CAP_MSMF). "
            "Nếu cần test tạm trên Linux, hãy dùng video file thay vì camera index."
        )


def _apply_camera_props(
    cap: cv2.VideoCapture,
    *,
    width: int | None,
    height: int | None,
    fps: int | None,
    buffersize: int | None,
) -> None:
    if width is not None and width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    if height is not None and height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    if fps is not None and fps > 0:
        cap.set(cv2.CAP_PROP_FPS, int(fps))
    if buffersize is not None and buffersize > 0:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, int(buffersize))
