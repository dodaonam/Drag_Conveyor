from __future__ import annotations

import cv2


def open_video_source(source: str) -> tuple[cv2.VideoCapture, str]:
    return cv2.VideoCapture(source.strip()), "FILE"
