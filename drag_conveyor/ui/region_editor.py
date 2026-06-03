from __future__ import annotations

from dataclasses import dataclass

from ..config import Profile


@dataclass(slots=True)
class _RegionSnapshot:
    x: int
    y: int
    w: int
    h: int


class RegionEditor:
    def __init__(self, edge_tolerance_px: int = 10) -> None:
        self.edge_tolerance_px = edge_tolerance_px
        self._mode = "idle"
        self._anchor_xy: tuple[float, float] | None = None
        self._snapshot: _RegionSnapshot | None = None

    def begin(self, profile: Profile, frame_xy: tuple[float, float]) -> None:
        self._anchor_xy = frame_xy
        region = profile.inspection_region
        self._snapshot = _RegionSnapshot(x=region.x, y=region.y, w=region.w, h=region.h)
        self._mode = self._detect_mode(profile, frame_xy)

    def update(
        self,
        profile: Profile,
        frame_xy: tuple[float, float],
        frame_size: tuple[int, int],
    ) -> bool:
        if self._mode == "idle" or self._anchor_xy is None or self._snapshot is None:
            return False

        frame_w, frame_h = frame_size
        ax, ay = self._anchor_xy
        cx, cy = frame_xy
        region = profile.inspection_region
        changed = False

        if self._mode == "draw":
            x1 = int(round(min(ax, cx)))
            y1 = int(round(min(ay, cy)))
            x2 = int(round(max(ax, cx)))
            y2 = int(round(max(ay, cy)))
            x1 = max(0, min(frame_w - 1, x1))
            y1 = max(0, min(frame_h - 1, y1))
            x2 = max(1, min(frame_w, x2))
            y2 = max(1, min(frame_h, y2))
            if x2 - x1 > 2 and y2 - y1 > 2:
                region.x = x1
                region.y = y1
                region.w = x2 - x1
                region.h = y2 - y1
                changed = True
        elif self._mode == "move":
            dx = int(round(cx - ax))
            dy = int(round(cy - ay))
            new_x = self._snapshot.x + dx
            new_y = self._snapshot.y + dy
            new_x = max(0, min(frame_w - self._snapshot.w, new_x))
            new_y = max(0, min(frame_h - self._snapshot.h, new_y))
            region.x = int(new_x)
            region.y = int(new_y)
            changed = True
        elif self._mode in {
            "resize_left",
            "resize_right",
            "resize_top",
            "resize_bottom",
            "resize_top_left",
            "resize_top_right",
            "resize_bottom_left",
            "resize_bottom_right",
        }:
            changed = self._resize_region(profile, frame_xy, frame_size)
        elif self._mode == "adjust_band":
            changed = self._adjust_trigger_band(profile, frame_xy)
        elif self._mode == "adjust_band_thickness_start":
            changed = self._adjust_trigger_band_thickness(profile, frame_xy)
        elif self._mode == "adjust_band_thickness_end":
            changed = self._adjust_trigger_band_thickness(profile, frame_xy)

        if changed:
            region.frame_width = frame_w
            region.frame_height = frame_h
        return changed

    def end(self) -> None:
        self._mode = "idle"
        self._anchor_xy = None
        self._snapshot = None

    def _detect_mode(self, profile: Profile, frame_xy: tuple[float, float]) -> str:
        x, y = frame_xy
        r = profile.inspection_region
        inside = (r.x <= x <= r.x + r.w) and (r.y <= y <= r.y + r.h)
        if not inside:
            return "draw"

        if r.direction in {"top_to_bottom", "bottom_to_top"}:
            center = r.y + r.h * r.trigger_band.position_ratio
            thickness = max(1.0, r.h * r.trigger_band.thickness_ratio)
            edge1 = center - thickness / 2.0
            edge2 = center + thickness / 2.0
            if abs(y - edge1) <= self.edge_tolerance_px:
                return "adjust_band_thickness_start"
            if abs(y - edge2) <= self.edge_tolerance_px:
                return "adjust_band_thickness_end"
            if abs(y - center) <= self.edge_tolerance_px:
                return "adjust_band"
        else:
            center = r.x + r.w * r.trigger_band.position_ratio
            thickness = max(1.0, r.w * r.trigger_band.thickness_ratio)
            edge1 = center - thickness / 2.0
            edge2 = center + thickness / 2.0
            if abs(x - edge1) <= self.edge_tolerance_px:
                return "adjust_band_thickness_start"
            if abs(x - edge2) <= self.edge_tolerance_px:
                return "adjust_band_thickness_end"
            if abs(x - center) <= self.edge_tolerance_px:
                return "adjust_band"

        near_left = abs(x - r.x) <= self.edge_tolerance_px
        near_right = abs(x - (r.x + r.w)) <= self.edge_tolerance_px
        near_top = abs(y - r.y) <= self.edge_tolerance_px
        near_bottom = abs(y - (r.y + r.h)) <= self.edge_tolerance_px

        if near_left and near_top:
            return "resize_top_left"
        if near_right and near_top:
            return "resize_top_right"
        if near_left and near_bottom:
            return "resize_bottom_left"
        if near_right and near_bottom:
            return "resize_bottom_right"
        if near_left:
            return "resize_left"
        if near_right:
            return "resize_right"
        if near_top:
            return "resize_top"
        if near_bottom:
            return "resize_bottom"
        return "move"

    def _resize_region(
        self,
        profile: Profile,
        frame_xy: tuple[float, float],
        frame_size: tuple[int, int],
    ) -> bool:
        if self._snapshot is None:
            return False

        frame_w, frame_h = frame_size
        cx, cy = frame_xy
        snap = self._snapshot
        min_size = 8

        x1 = snap.x
        y1 = snap.y
        x2 = snap.x + snap.w
        y2 = snap.y + snap.h

        if self._mode == "resize_left":
            x1 = int(round(cx))
        elif self._mode == "resize_right":
            x2 = int(round(cx))
        elif self._mode == "resize_top":
            y1 = int(round(cy))
        elif self._mode == "resize_bottom":
            y2 = int(round(cy))
        elif self._mode == "resize_top_left":
            x1 = int(round(cx))
            y1 = int(round(cy))
        elif self._mode == "resize_top_right":
            x2 = int(round(cx))
            y1 = int(round(cy))
        elif self._mode == "resize_bottom_left":
            x1 = int(round(cx))
            y2 = int(round(cy))
        elif self._mode == "resize_bottom_right":
            x2 = int(round(cx))
            y2 = int(round(cy))

        x1 = max(0, min(frame_w - min_size, x1))
        y1 = max(0, min(frame_h - min_size, y1))
        x2 = max(min_size, min(frame_w, x2))
        y2 = max(min_size, min(frame_h, y2))

        if x2 - x1 < min_size:
            if self._mode == "resize_left":
                x1 = x2 - min_size
            else:
                x2 = x1 + min_size
        if y2 - y1 < min_size:
            if self._mode == "resize_top":
                y1 = y2 - min_size
            else:
                y2 = y1 + min_size

        profile.inspection_region.x = int(max(0, x1))
        profile.inspection_region.y = int(max(0, y1))
        profile.inspection_region.w = int(min(frame_w - profile.inspection_region.x, x2 - x1))
        profile.inspection_region.h = int(min(frame_h - profile.inspection_region.y, y2 - y1))
        return True

    def _adjust_trigger_band(self, profile: Profile, frame_xy: tuple[float, float]) -> bool:
        x, y = frame_xy
        region = profile.inspection_region
        if region.direction in {"top_to_bottom", "bottom_to_top"}:
            ratio = (y - region.y) / max(region.h, 1)
        else:
            ratio = (x - region.x) / max(region.w, 1)
        ratio = max(0.0, min(1.0, float(ratio)))
        region.trigger_band.position_ratio = ratio
        return True

    def _adjust_trigger_band_thickness(self, profile: Profile, frame_xy: tuple[float, float]) -> bool:
        x, y = frame_xy
        region = profile.inspection_region
        min_px = 2.0

        if region.direction in {"top_to_bottom", "bottom_to_top"}:
            center = region.y + region.h * region.trigger_band.position_ratio
            edge = max(float(region.y), min(float(region.y + region.h), float(y)))
            thickness_px = max(min_px, abs(edge - center) * 2.0)
            ratio = thickness_px / max(float(region.h), 1.0)
        else:
            center = region.x + region.w * region.trigger_band.position_ratio
            edge = max(float(region.x), min(float(region.x + region.w), float(x)))
            thickness_px = max(min_px, abs(edge - center) * 2.0)
            ratio = thickness_px / max(float(region.w), 1.0)

        region.trigger_band.thickness_ratio = max(0.01, min(1.0, float(ratio)))
        return True
