from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LetterboxMapping:
    frame_width: int
    frame_height: int
    widget_width: int
    widget_height: int
    scale: float
    content_width: float
    content_height: float
    offset_x: float
    offset_y: float


@dataclass(frozen=True, slots=True)
class Rect:
    x: float
    y: float
    w: float
    h: float


def compute_letterbox_mapping(
    frame_width: int, frame_height: int, widget_width: int, widget_height: int
) -> LetterboxMapping:
    if frame_width <= 0 or frame_height <= 0 or widget_width <= 0 or widget_height <= 0:
        raise ValueError("All dimensions must be positive")

    scale = min(widget_width / frame_width, widget_height / frame_height)
    content_width = frame_width * scale
    content_height = frame_height * scale
    offset_x = (widget_width - content_width) / 2.0
    offset_y = (widget_height - content_height) / 2.0

    return LetterboxMapping(
        frame_width=frame_width,
        frame_height=frame_height,
        widget_width=widget_width,
        widget_height=widget_height,
        scale=scale,
        content_width=content_width,
        content_height=content_height,
        offset_x=offset_x,
        offset_y=offset_y,
    )


def frame_to_widget(mapping: LetterboxMapping, x: float, y: float) -> tuple[float, float]:
    wx = x * mapping.scale + mapping.offset_x
    wy = y * mapping.scale + mapping.offset_y
    return wx, wy


def widget_to_frame(mapping: LetterboxMapping, x: float, y: float, clamp: bool = False) -> tuple[float, float] | None:
    inside_x = mapping.offset_x <= x <= (mapping.offset_x + mapping.content_width)
    inside_y = mapping.offset_y <= y <= (mapping.offset_y + mapping.content_height)
    if not (inside_x and inside_y):
        if not clamp:
            return None
        x = min(max(x, mapping.offset_x), mapping.offset_x + mapping.content_width)
        y = min(max(y, mapping.offset_y), mapping.offset_y + mapping.content_height)

    fx = (x - mapping.offset_x) / mapping.scale
    fy = (y - mapping.offset_y) / mapping.scale
    return fx, fy


def frame_rect_to_widget(mapping: LetterboxMapping, rect: Rect) -> Rect:
    x1, y1 = frame_to_widget(mapping, rect.x, rect.y)
    x2, y2 = frame_to_widget(mapping, rect.x + rect.w, rect.y + rect.h)
    return Rect(x=x1, y=y1, w=(x2 - x1), h=(y2 - y1))


def widget_rect_to_frame(mapping: LetterboxMapping, rect: Rect, clamp: bool = True) -> Rect | None:
    p1 = widget_to_frame(mapping, rect.x, rect.y, clamp=clamp)
    p2 = widget_to_frame(mapping, rect.x + rect.w, rect.y + rect.h, clamp=clamp)
    if p1 is None or p2 is None:
        return None
    x1, y1 = p1
    x2, y2 = p2
    return Rect(x=x1, y=y1, w=(x2 - x1), h=(y2 - y1))
