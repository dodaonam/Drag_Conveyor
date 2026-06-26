"""Video pre-processing applied after upload, before inference.

Slows the conveyor down by inserting optical-flow interpolated frames so each
bar moves a smaller distance between consecutive frames. This keeps the
centroid tracker's per-frame jump below ``max_jump_px`` (see
``drag_conveyor/pipeline/tracking.py``), which makes detection / counting more
stable for fast-moving conveyors.

``factor`` is the target playback speed: 0.75 means "play at 75% speed", which
produces ~1/0.75 = 1.33x as many frames (≈4 output frames for every 3 input
frames). Output frames that land between two real frames are interpolated.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)

# Target playback speed before inference. 0.75 = 75% speed (~1.33x frames).
# Set to >= 1.0 to disable slow-motion preprocessing.
SLOWDOWN_FACTOR = 0.75

# Optical flow is computed on a downscaled copy for speed, then scaled back up.
_FLOW_SCALE = 0.5


def _make_dis() -> "cv2.DISOpticalFlow":
    return cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)


def _flow(prev_gray: np.ndarray, next_gray: np.ndarray, dis) -> np.ndarray:
    if _FLOW_SCALE == 1.0:
        return dis.calc(prev_gray, next_gray, None)
    small_prev = cv2.resize(prev_gray, None, fx=_FLOW_SCALE, fy=_FLOW_SCALE, interpolation=cv2.INTER_AREA)
    small_next = cv2.resize(next_gray, None, fx=_FLOW_SCALE, fy=_FLOW_SCALE, interpolation=cv2.INTER_AREA)
    flow = dis.calc(small_prev, small_next, None)
    flow = cv2.resize(flow, (prev_gray.shape[1], prev_gray.shape[0]), interpolation=cv2.INTER_LINEAR)
    flow *= (1.0 / _FLOW_SCALE)
    return flow


def _warp(img: np.ndarray, flow: np.ndarray, grid_x: np.ndarray, grid_y: np.ndarray, scale: float) -> np.ndarray:
    map_x = grid_x - scale * flow[..., 0]
    map_y = grid_y - scale * flow[..., 1]
    return cv2.remap(
        img, map_x, map_y,
        interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )


def slow_down_video(src_path, dst_path, factor: float = SLOWDOWN_FACTOR) -> str:
    """Write a slowed copy of ``src_path`` to ``dst_path`` and return its path.

    Resamples the video onto a denser timeline: output frame ``k`` is sampled at
    input position ``k * factor``. When that position falls between two real
    frames, an optical-flow interpolated frame is generated; when it lands on a
    real frame, that frame is copied verbatim. Returns ``str(src_path)`` unchanged
    when slowdown is disabled (factor >= 1) or the video has fewer than 2 frames.
    """
    src_path, dst_path = str(src_path), str(dst_path)
    if not (0.0 < factor < 1.0):
        return src_path

    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video for slow-motion preprocessing")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ok, prev = cap.read()
    if not ok:
        cap.release()
        return src_path  # nothing to slow down

    writer = cv2.VideoWriter(dst_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Cannot open VideoWriter for slow-motion preprocessing")

    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    dis = _make_dis()
    eps = 1e-6

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    prev_index = 0
    out_k = 0
    frames_in, frames_out = 1, 0

    while True:
        ok, curr = cap.read()
        if not ok:
            break
        frames_in += 1
        curr_index = prev_index + 1
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        flows = None  # lazily computed only when an interpolated frame is needed

        # Emit every output frame whose source position lands in [prev_index, curr_index).
        while out_k * factor < curr_index - eps:
            t = (out_k * factor) - prev_index
            if t <= eps:
                writer.write(prev)                       # lands on the real frame
            else:
                if flows is None:
                    flow_fwd = _flow(prev_gray, curr_gray, dis)   # prev -> curr
                    flow_bwd = _flow(curr_gray, prev_gray, dis)   # curr -> prev
                    flows = (flow_fwd, flow_bwd)
                flow_fwd, flow_bwd = flows
                warp_prev = _warp(prev, flow_fwd, grid_x, grid_y, t)
                warp_curr = _warp(curr, flow_bwd, grid_x, grid_y, 1.0 - t)
                writer.write(cv2.addWeighted(warp_prev, 1.0 - t, warp_curr, t, 0.0))
            out_k += 1
            frames_out += 1

        prev, prev_gray, prev_index = curr, curr_gray, curr_index

    # Tail: any output that maps exactly onto the final frame.
    while out_k * factor <= prev_index + eps:
        writer.write(prev)
        out_k += 1
        frames_out += 1

    cap.release()
    writer.release()
    LOGGER.info(
        "Slow-motion preprocess: %d -> %d frames (factor=%.2f)",
        frames_in, frames_out, factor,
    )
    return dst_path
