from __future__ import annotations

import cv2


def open_video_source(source: str) -> cv2.VideoCapture:
    return cv2.VideoCapture(source.strip())
