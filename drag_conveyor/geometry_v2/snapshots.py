from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .coordinates import ChainCoordinates, TriggerStrip


def geometry_snapshot_filename(track_id: int, source_frame_id: int) -> str:
    return f"track_{track_id:06d}_frame_{source_frame_id:09d}.jpg"


def render_geometry_snapshot(
    frame_bgr: np.ndarray,
    *,
    roi_xywh: tuple[int, int, int, int],
    coordinates: ChainCoordinates,
    trigger_strip: TriggerStrip,
    status: str,
    output_path: str | Path,
    jpeg_quality: int = 92,
) -> Path:
    """Write a single-frame V2 overlay without legacy contour semantics."""
    if not 0 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be in [0, 100]")
    x, y, width, height = roi_xywh
    if frame_bgr.shape[0] < y + height or frame_bgr.shape[1] < x + width:
        raise ValueError("ROI is outside the supplied source frame")
    image = frame_bgr.copy()
    cv2.rectangle(image, (x, y), (x + width, y + height), (255, 180, 0), 2)
    top, bottom = coordinates.centerline.top, coordinates.centerline.bottom
    cv2.line(image, (round(x + top.x), round(y + top.y)), (round(x + bottom.x), round(y + bottom.y)), (0, 255, 255), 2)
    polygon = coordinates.trigger_strip_polygon(trigger_strip)
    if polygon:
        points = np.array([[round(x + point.x), round(y + point.y)] for point in polygon], dtype=np.int32)
        cv2.polylines(image, [points], True, (255, 0, 255), 2)
    cv2.putText(image, status, (x + 8, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target), image, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]):
        raise OSError(f"Unable to write geometry snapshot: {target}")
    return target
