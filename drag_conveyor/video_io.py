from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class VideoSourceInfo:
    path: str
    width: int
    height: int
    fps: float
    first_frame_bgr: np.ndarray


def open_video_source(source: str) -> tuple[cv2.VideoCapture, str]:
    return cv2.VideoCapture(source.strip()), "FILE"


def probe_video_source(source: str) -> VideoSourceInfo:
    source = source.strip()
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
