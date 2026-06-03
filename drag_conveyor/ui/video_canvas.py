from __future__ import annotations

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from ..config import Profile
from ..mapping import compute_letterbox_mapping, frame_to_widget, widget_to_frame
from ..pipeline_worker import DisplayPacket
from .region_editor import RegionEditor


class VideoCanvas(QtWidgets.QWidget):
    regionChanged = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(640, 360)
        self.setMouseTracking(True)

        self._profile: Profile | None = None
        self._frame_bgr: np.ndarray | None = None
        self._display_packet: DisplayPacket | None = None
        self._freeze_enabled = False
        self._frozen_frame: np.ndarray | None = None
        self._mapping = None
        self._editor = RegionEditor()

    def set_profile(self, profile: Profile) -> None:
        self._profile = profile
        self.update()

    def set_frame(self, frame_bgr: np.ndarray) -> None:
        self._frame_bgr = frame_bgr.copy()
        self._display_packet = None
        if self._freeze_enabled and self._frozen_frame is None:
            self._frozen_frame = self._frame_bgr.copy()
        self.update()

    def set_display_packet(self, packet: DisplayPacket) -> None:
        self._display_packet = packet
        self._frame_bgr = packet.frame_bgr.copy()
        if self._freeze_enabled and self._frozen_frame is None:
            self._frozen_frame = self._frame_bgr.copy()
        self.update()

    def set_freeze_enabled(self, enabled: bool) -> None:
        self._freeze_enabled = bool(enabled)
        if self._freeze_enabled:
            if self._frame_bgr is not None:
                self._frozen_frame = self._frame_bgr.copy()
        else:
            self._frozen_frame = None
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802
        del event
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor(30, 30, 30))

        frame = self._display_frame()
        if frame is None:
            return

        h, w = frame.shape[:2]
        self._mapping = compute_letterbox_mapping(w, h, self.width(), self.height())
        out_w = max(1, int(round(w * self._mapping.scale)))
        out_h = max(1, int(round(h * self._mapping.scale)))
        resized = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        image = QtGui.QImage(rgb.data, out_w, out_h, out_w * 3, QtGui.QImage.Format.Format_RGB888).copy()

        px = int(round(self._mapping.offset_x))
        py = int(round(self._mapping.offset_y))
        painter.drawImage(QtCore.QPoint(px, py), image)

        self._draw_overlay(painter)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        if not self._freeze_enabled or self._profile is None:
            return
        frame_xy = self._widget_to_frame(event.position().x(), event.position().y())
        if frame_xy is None:
            return
        self._editor.begin(self._profile, frame_xy)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        if not self._freeze_enabled or self._profile is None:
            return
        frame_xy = self._widget_to_frame(event.position().x(), event.position().y())
        if frame_xy is None:
            return
        frame_size = self._frame_size()
        if frame_size is None:
            return
        changed = self._editor.update(self._profile, frame_xy, frame_size)
        if changed:
            self.regionChanged.emit()
            self.update()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        del event
        if not self._freeze_enabled:
            return
        self._editor.end()

    def _display_frame(self) -> np.ndarray | None:
        if self._freeze_enabled and self._frozen_frame is not None:
            return self._frozen_frame
        return self._frame_bgr

    def _frame_size(self) -> tuple[int, int] | None:
        frame = self._display_frame()
        if frame is None:
            return None
        h, w = frame.shape[:2]
        return (w, h)

    def _widget_to_frame(self, x: float, y: float) -> tuple[float, float] | None:
        frame = self._display_frame()
        if frame is None:
            return None
        h, w = frame.shape[:2]
        mapping = compute_letterbox_mapping(w, h, self.width(), self.height())
        mapped = widget_to_frame(mapping, x, y, clamp=True)
        return mapped

    def _draw_overlay(self, painter: QtGui.QPainter) -> None:
        frame = self._display_frame()
        if frame is None or self._profile is None or self._mapping is None:
            return

        r = self._profile.inspection_region
        if self._display_packet is not None and not self._freeze_enabled:
            self._draw_detection_overlay(painter, self._display_packet)

        x1, y1 = frame_to_widget(self._mapping, r.x, r.y)
        x2, y2 = frame_to_widget(self._mapping, r.x + r.w, r.y + r.h)

        roi_pen = QtGui.QPen(QtGui.QColor(255, 180, 0))
        roi_pen.setWidth(2)
        painter.setPen(roi_pen)
        painter.drawRect(QtCore.QRectF(x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1)))

        band_pen = QtGui.QPen(QtGui.QColor(0, 255, 255))
        band_pen.setWidth(2)
        painter.setPen(band_pen)
        band_edges: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None
        if r.direction in {"top_to_bottom", "bottom_to_top"}:
            center = r.y + r.h * r.trigger_band.position_ratio
            thickness = max(1, int(round(r.h * r.trigger_band.thickness_ratio)))
            by1 = max(r.y, int(round(center - thickness / 2.0)))
            by2 = min(r.y + r.h, int(round(center + thickness / 2.0)))
            bx1w, by1w = frame_to_widget(self._mapping, r.x, by1)
            bx2w, by2w = frame_to_widget(self._mapping, r.x + r.w, by2)
            painter.drawRect(QtCore.QRectF(bx1w, by1w, max(1.0, bx2w - bx1w), max(1.0, by2w - by1w)))
            cxw, cyw = frame_to_widget(self._mapping, r.x + r.w / 2.0, center)
            _, ey1w = frame_to_widget(self._mapping, r.x + r.w / 2.0, by1)
            _, ey2w = frame_to_widget(self._mapping, r.x + r.w / 2.0, by2)
            band_edges = ((cxw, cyw), (cxw, ey1w), (cxw, ey2w))
        else:
            center = r.x + r.w * r.trigger_band.position_ratio
            thickness = max(1, int(round(r.w * r.trigger_band.thickness_ratio)))
            bx1 = max(r.x, int(round(center - thickness / 2.0)))
            bx2 = min(r.x + r.w, int(round(center + thickness / 2.0)))
            bx1w, by1w = frame_to_widget(self._mapping, bx1, r.y)
            bx2w, by2w = frame_to_widget(self._mapping, bx2, r.y + r.h)
            painter.drawRect(QtCore.QRectF(bx1w, by1w, max(1.0, bx2w - bx1w), max(1.0, by2w - by1w)))
            cxw, cyw = frame_to_widget(self._mapping, center, r.y + r.h / 2.0)
            ex1w, _ = frame_to_widget(self._mapping, bx1, r.y + r.h / 2.0)
            ex2w, _ = frame_to_widget(self._mapping, bx2, r.y + r.h / 2.0)
            band_edges = ((cxw, cyw), (ex1w, cyw), (ex2w, cyw))

        if self._freeze_enabled:
            self._draw_edit_handles(
                painter=painter,
                roi_rect=(x1, y1, x2, y2),
                band_edges=band_edges,
            )

    def _draw_detection_overlay(self, painter: QtGui.QPainter, packet: DisplayPacket) -> None:
        event_by_track = {event.track_id: event for event in packet.events}

        for track in packet.tracks:
            event = event_by_track.get(track.track_id)
            if event is not None and event.result == "suspected_defect":
                color = QtGui.QColor(255, 70, 70, 180)
            elif event is not None and event.result == "normal":
                color = QtGui.QColor(60, 210, 90, 180)
            else:
                color = QtGui.QColor(60, 180, 255, 170)

            contour = track.contour_frame.astype(np.float32)
            if contour.size >= 6:
                path = QtGui.QPainterPath()
                px0, py0 = frame_to_widget(self._mapping, float(contour[0, 0, 0]), float(contour[0, 0, 1]))
                path.moveTo(px0, py0)
                for point in contour[1:]:
                    px, py = frame_to_widget(self._mapping, float(point[0, 0]), float(point[0, 1]))
                    path.lineTo(px, py)
                path.closeSubpath()
                painter.fillPath(path, QtGui.QColor(color.red(), color.green(), color.blue(), 50))
                contour_pen = QtGui.QPen(QtGui.QColor(color.red(), color.green(), color.blue()))
                contour_pen.setWidth(2)
                painter.setPen(contour_pen)
                painter.drawPath(path)

            x1, y1, x2, y2 = track.bbox_frame_xyxy
            bx1, by1 = frame_to_widget(self._mapping, x1, y1)
            bx2, by2 = frame_to_widget(self._mapping, x2, y2)
            rect_pen = QtGui.QPen(QtGui.QColor(color.red(), color.green(), color.blue()))
            rect_pen.setWidth(2)
            painter.setPen(rect_pen)
            painter.drawRect(QtCore.QRectF(bx1, by1, max(1.0, bx2 - bx1), max(1.0, by2 - by1)))

            label_parts = [f"id={track.track_id}"]
            if event is not None:
                label_parts.append(event.result)
            label = " ".join(label_parts)
            text_rect = QtCore.QRectF(bx1, max(0.0, by1 - 18.0), 220.0, 18.0)
            painter.fillRect(text_rect, QtGui.QColor(0, 0, 0, 120))
            painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255)))
            painter.drawText(text_rect.adjusted(3.0, 0.0, -3.0, 0.0), QtCore.Qt.AlignmentFlag.AlignVCenter, label)

        if packet.events:
            latest = packet.events[-1]
            reason = latest.reasons[0] if latest.reasons else "none"
            info = f"Last event: id={latest.track_id} {latest.result} reason={reason}"
            banner = QtCore.QRectF(12.0, 12.0, min(float(self.width()) - 24.0, 620.0), 24.0)
            painter.fillRect(banner, QtGui.QColor(0, 0, 0, 140))
            painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255)))
            painter.drawText(
                banner.adjusted(6.0, 0.0, -6.0, 0.0),
                QtCore.Qt.AlignmentFlag.AlignVCenter,
                info,
            )

    def _draw_edit_handles(
        self,
        *,
        painter: QtGui.QPainter,
        roi_rect: tuple[float, float, float, float],
        band_edges: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None,
    ) -> None:
        x1, y1, x2, y2 = roi_rect
        handles = [
            (x1, y1),
            (x2, y1),
            (x1, y2),
            (x2, y2),
        ]
        if band_edges is not None:
            handles.extend(list(band_edges))

        painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255)))
        painter.setBrush(QtGui.QBrush(QtGui.QColor(20, 20, 20, 210)))
        for hx, hy in handles:
            painter.drawEllipse(QtCore.QPointF(hx, hy), 4.0, 4.0)
