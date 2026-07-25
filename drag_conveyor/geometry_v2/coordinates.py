from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from .types import Centerline, Point, Roi


class GeometryValidationError(ValueError):
    """Raised when continuous geometry cannot satisfy the V2 input contract."""


@dataclass(frozen=True, slots=True)
class TriggerStrip:
    top_s: float
    bottom_s: float


@dataclass(frozen=True, slots=True)
class ChainCoordinates:
    """The canonical longitudinal/transverse coordinate system from spec section 6."""

    roi: Roi
    centerline: Centerline
    direction: tuple[float, float]
    horizontal: tuple[float, float]
    s_min: float
    s_max: float

    @classmethod
    def create(
        cls,
        roi: Roi,
        first: Point,
        second: Point,
        *,
        minimum_span_ratio: float = 0.70,
        maximum_roll_deg: float = 15.0,
    ) -> "ChainCoordinates":
        centerline = Centerline.canonical(first, second)
        _validate_point_in_closed_roi(centerline.top, roi)
        _validate_point_in_closed_roi(centerline.bottom, roi)

        dx = centerline.bottom.x - centerline.top.x
        dy = centerline.bottom.y - centerline.top.y
        length = math.hypot(dx, dy)
        if length == 0.0:
            raise GeometryValidationError("Centerline endpoints must be distinct")
        if length < minimum_span_ratio * roi.height:
            raise GeometryValidationError("Centerline span is shorter than the configured minimum")

        direction = (dx / length, dy / length)
        roll_deg = math.degrees(math.atan2(abs(direction[0]), abs(direction[1])))
        if roll_deg > maximum_roll_deg:
            raise GeometryValidationError("Centerline roll exceeds the configured maximum")

        interval = _line_rectangle_interval(centerline.top, direction, roi.width, roi.height)
        if interval is None:
            raise GeometryValidationError("Centerline does not intersect ROI")
        s_min, s_max = interval
        if not s_max > s_min:
            raise GeometryValidationError("Centerline has no positive span inside ROI")

        return cls(
            roi=roi,
            centerline=centerline,
            direction=direction,
            horizontal=(direction[1], -direction[0]),
            s_min=s_min,
            s_max=s_max,
        )

    @property
    def span(self) -> float:
        return self.s_max - self.s_min

    @property
    def roll_deg(self) -> float:
        return math.degrees(math.atan2(abs(self.direction[0]), abs(self.direction[1])))

    def project(self, point: Point) -> tuple[float, float]:
        dx = point.x - self.centerline.top.x
        dy = point.y - self.centerline.top.y
        return (
            dx * self.direction[0] + dy * self.direction[1],
            dx * self.horizontal[0] + dy * self.horizontal[1],
        )

    def project_pixel_center(self, x: int, y: int) -> tuple[float, float]:
        return self.project(Point(x + 0.5, y + 0.5))

    def point_at(self, s: float, q: float = 0.0) -> Point:
        return Point(
            self.centerline.top.x + s * self.direction[0] + q * self.horizontal[0],
            self.centerline.top.y + s * self.direction[1] + q * self.horizontal[1],
        )

    def available_side_extents(self, s: float) -> tuple[float, float]:
        """Return left then right ray extents to the closed ROI boundary at ``s``."""
        point = self.point_at(s)
        return (
            _ray_extent_to_rectangle(point, (-self.horizontal[0], -self.horizontal[1]), self.roi),
            _ray_extent_to_rectangle(point, self.horizontal, self.roi),
        )

    def trigger_strip(self, center_ratio: float, height_ratio: float) -> TriggerStrip:
        if not 0.0 <= center_ratio <= 1.0:
            raise GeometryValidationError("Trigger center ratio must be in [0, 1]")
        if not 0.0 < height_ratio <= 1.0:
            raise GeometryValidationError("Trigger height ratio must be in (0, 1]")
        center_s = self.s_min + center_ratio * self.span
        half_height = 0.5 * height_ratio * self.span
        return TriggerStrip(top_s=center_s - half_height, bottom_s=center_s + half_height)

    def trigger_strip_polygon(self, strip: TriggerStrip) -> tuple[Point, ...]:
        """Clip the infinite transverse strip to the closed ROI rectangle."""
        extent = math.hypot(self.roi.width, self.roi.height) + 1.0
        polygon = [
            self.point_at(strip.top_s, -extent),
            self.point_at(strip.top_s, extent),
            self.point_at(strip.bottom_s, extent),
            self.point_at(strip.bottom_s, -extent),
        ]
        return tuple(_clip_polygon_to_roi(polygon, self.roi))


def quantile_type7(values: Iterable[float], quantile: float) -> float:
    """Hyndman-Fan type-7 quantile, fixed by the V2 numeric contract."""
    data = np.asarray(tuple(values), dtype=np.float64)
    if data.size == 0:
        raise ValueError("Cannot calculate a quantile of an empty sequence")
    if not np.isfinite(data).all():
        raise ValueError("Quantile input must contain only finite values")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("Quantile must be in [0, 1]")
    return float(np.quantile(data, quantile, method="linear"))


def mad(values: Iterable[float]) -> float:
    data = np.asarray(tuple(values), dtype=np.float64)
    median = quantile_type7(data, 0.5)
    return quantile_type7(np.abs(data - median), 0.5)


def rasterized_crop_bbox(
    bbox_xyxy: tuple[float, float, float, float],
    *,
    roi_width: int,
    roi_height: int,
) -> tuple[int, int, int, int]:
    """Clip then floor a model bbox into its exact half-open mask slice."""
    if roi_width <= 0 or roi_height <= 0:
        raise ValueError("ROI dimensions must be positive")
    x1, y1, x2, y2 = bbox_xyxy
    if not all(math.isfinite(value) for value in bbox_xyxy):
        raise ValueError("Bounding box coordinates must be finite")
    x1 = min(max(x1, 0.0), float(roi_width))
    y1 = min(max(y1, 0.0), float(roi_height))
    x2 = min(max(x2, 0.0), float(roi_width))
    y2 = min(max(y2, 0.0), float(roi_height))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return (math.floor(x1), math.floor(y1), math.floor(x2), math.floor(y2))


def _validate_point_in_closed_roi(point: Point, roi: Roi) -> None:
    if not (0.0 <= point.x <= roi.width and 0.0 <= point.y <= roi.height):
        raise GeometryValidationError("Centerline endpoints must lie in the closed ROI bounds")


def _line_rectangle_interval(
    origin: Point,
    direction: tuple[float, float],
    width: int,
    height: int,
) -> tuple[float, float] | None:
    lower, upper = -math.inf, math.inf
    for coordinate, velocity, boundary in ((origin.x, direction[0], width), (origin.y, direction[1], height)):
        if velocity == 0.0:
            if not 0.0 <= coordinate <= boundary:
                return None
            continue
        first = (0.0 - coordinate) / velocity
        second = (boundary - coordinate) / velocity
        lower = max(lower, min(first, second))
        upper = min(upper, max(first, second))
    return (lower, upper) if lower <= upper else None


def _ray_extent_to_rectangle(point: Point, direction: tuple[float, float], roi: Roi) -> float:
    candidates: list[float] = []
    for coordinate, velocity, boundary in ((point.x, direction[0], roi.width), (point.y, direction[1], roi.height)):
        if velocity > 0.0:
            candidates.append((boundary - coordinate) / velocity)
        elif velocity < 0.0:
            candidates.append((0.0 - coordinate) / velocity)
    nonnegative = [candidate for candidate in candidates if candidate >= 0.0]
    if not nonnegative:
        raise GeometryValidationError("Ray origin lies outside the ROI")
    return min(nonnegative)


def _clip_polygon_to_roi(polygon: list[Point], roi: Roi) -> list[Point]:
    edges = (
        (lambda point: point.x >= 0.0, lambda start, end: _intersect_x(start, end, 0.0)),
        (lambda point: point.x <= roi.width, lambda start, end: _intersect_x(start, end, float(roi.width))),
        (lambda point: point.y >= 0.0, lambda start, end: _intersect_y(start, end, 0.0)),
        (lambda point: point.y <= roi.height, lambda start, end: _intersect_y(start, end, float(roi.height))),
    )
    clipped = polygon
    for inside, intersection in edges:
        if not clipped:
            break
        output: list[Point] = []
        start = clipped[-1]
        start_inside = inside(start)
        for end in clipped:
            end_inside = inside(end)
            if end_inside != start_inside:
                output.append(intersection(start, end))
            if end_inside:
                output.append(end)
            start, start_inside = end, end_inside
        clipped = output
    return clipped


def _intersect_x(start: Point, end: Point, x: float) -> Point:
    if end.x == start.x:
        return Point(x, start.y)
    ratio = (x - start.x) / (end.x - start.x)
    return Point(x, start.y + ratio * (end.y - start.y))


def _intersect_y(start: Point, end: Point, y: float) -> Point:
    if end.y == start.y:
        return Point(start.x, y)
    ratio = (y - start.y) / (end.y - start.y)
    return Point(start.x + ratio * (end.x - start.x), y)
