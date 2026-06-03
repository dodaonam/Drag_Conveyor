from __future__ import annotations

import csv
import json
import logging
import queue
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from ..worker_status import WorkerStatus

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LogPayload:
    run_id: str
    frame_id: int
    track_id: int
    result: str
    score: float
    reasons: list[str]
    measurements: dict[str, float]
    inference_fps_estimate: float
    latency_ms: float
    bbox_frame_xyxy: tuple[float, float, float, float]
    roi_rect_xyxy: tuple[int, int, int, int]
    trigger_band_xyxy: tuple[int, int, int, int]
    thresholds: dict[str, float]
    margins: dict[str, float]
    hard_fail: bool
    violated_soft_rules: list[str]
    contour_frame: np.ndarray
    snapshot_frame: np.ndarray | None
    timestamp: str = field(
        default_factory=lambda: datetime.now().isoformat(timespec="milliseconds")
    )


class LoggerWorker:
    def __init__(
        self,
        logs_dir: Path,
        defect_snapshots_root: Path,
        run_id: str,
        save_defect_snapshot: bool,
        debug_enabled: bool = False,
    ) -> None:
        self.logs_dir = logs_dir
        self.defect_snapshots_root = defect_snapshots_root
        self.run_id = run_id
        self.save_defect_snapshot = save_defect_snapshot
        self.debug_enabled = debug_enabled

        self._queue: queue.Queue[LogPayload | None] = queue.Queue(maxsize=1024)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._status_lock = threading.Lock()
        self._status = WorkerStatus(name="LoggerWorker", state="IDLE", message="Chưa khởi động")

        self._csv_path = self.logs_dir / f"{run_id}_inspection.csv"
        self._snapshot_dir = self.defect_snapshots_root / run_id

    def start(self) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self._snapshot_dir.mkdir(parents=True, exist_ok=True)
        self._set_status("STARTING", "Đang khởi tạo logger")
        self._thread = threading.Thread(target=self._run, name="LoggerWorker", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        if self.get_status().state != "ERROR":
            self._set_status("STOPPED", "Đã dừng")

    def enqueue(self, payload: LogPayload) -> None:
        # Immutable payload guarantee: deep-copy mutable frame/contour before queueing.
        contour_copy = payload.contour_frame.copy()
        frame_copy = payload.snapshot_frame.copy() if payload.snapshot_frame is not None else None
        copied = LogPayload(
            run_id=payload.run_id,
            frame_id=payload.frame_id,
            track_id=payload.track_id,
            result=payload.result,
            score=payload.score,
            reasons=list(payload.reasons),
            measurements=dict(payload.measurements),
            inference_fps_estimate=payload.inference_fps_estimate,
            latency_ms=payload.latency_ms,
            bbox_frame_xyxy=payload.bbox_frame_xyxy,
            roi_rect_xyxy=payload.roi_rect_xyxy,
            trigger_band_xyxy=payload.trigger_band_xyxy,
            thresholds=dict(payload.thresholds),
            margins=dict(payload.margins),
            hard_fail=payload.hard_fail,
            violated_soft_rules=list(payload.violated_soft_rules),
            contour_frame=contour_copy,
            snapshot_frame=frame_copy,
            timestamp=payload.timestamp,
        )
        try:
            self._queue.put_nowait(copied)
        except queue.Full:
            LOGGER.warning("Logger queue is full; dropping log payload for track_id=%s", payload.track_id)

    def get_status(self) -> WorkerStatus:
        with self._status_lock:
            return self._status

    def _run(self) -> None:
        header = [
            "run_id",
            "timestamp",
            "frame_id",
            "track_id",
            "result",
            "score",
            "reasons",
            "area",
            "length",
            "width",
            "aspect_ratio",
            "inference_fps_estimate",
            "latency_ms",
        ]
        if self.debug_enabled:
            header.extend(
                [
                    "bbox_frame_xyxy",
                    "roi_rect_xyxy",
                    "trigger_band_xyxy",
                    "area_min",
                    "area_margin",
                    "length_min",
                    "length_margin",
                    "width_min",
                    "width_margin",
                    "aspect_ratio_min",
                    "aspect_ratio_margin",
                    "hard_fail",
                    "violated_soft_rules",
                ]
            )

        self._set_status("RUNNING", "Đang ghi log")
        try:
            with self._csv_path.open("w", newline="", encoding="utf-8") as fp:
                writer = csv.DictWriter(fp, fieldnames=header)
                writer.writeheader()

                while not self._stop.is_set() or not self._queue.empty():
                    try:
                        payload = self._queue.get(timeout=0.2)
                    except queue.Empty:
                        continue
                    if payload is None:
                        continue

                    row = {
                        "run_id": payload.run_id,
                        "timestamp": payload.timestamp,
                        "frame_id": payload.frame_id,
                        "track_id": payload.track_id,
                        "result": payload.result,
                        "score": f"{payload.score:.4f}",
                        "reasons": json.dumps(payload.reasons, ensure_ascii=False),
                        "area": f"{payload.measurements.get('area', 0.0):.4f}",
                        "length": f"{payload.measurements.get('length', 0.0):.4f}",
                        "width": f"{payload.measurements.get('width', 0.0):.4f}",
                        "aspect_ratio": f"{payload.measurements.get('aspect_ratio', 0.0):.4f}",
                        "inference_fps_estimate": f"{payload.inference_fps_estimate:.2f}",
                        "latency_ms": f"{payload.latency_ms:.2f}",
                    }
                    if self.debug_enabled:
                        row["bbox_frame_xyxy"] = json.dumps(payload.bbox_frame_xyxy)
                        row["roi_rect_xyxy"] = json.dumps(payload.roi_rect_xyxy)
                        row["trigger_band_xyxy"] = json.dumps(payload.trigger_band_xyxy)
                        row["area_min"] = f"{payload.thresholds.get('area_min', 0.0):.4f}"
                        row["area_margin"] = f"{payload.margins.get('area_margin', 0.0):.4f}"
                        row["length_min"] = f"{payload.thresholds.get('length_min', 0.0):.4f}"
                        row["length_margin"] = f"{payload.margins.get('length_margin', 0.0):.4f}"
                        row["width_min"] = f"{payload.thresholds.get('width_min', 0.0):.4f}"
                        row["width_margin"] = f"{payload.margins.get('width_margin', 0.0):.4f}"
                        row["aspect_ratio_min"] = f"{payload.thresholds.get('aspect_ratio_min', 0.0):.4f}"
                        row["aspect_ratio_margin"] = f"{payload.margins.get('aspect_ratio_margin', 0.0):.4f}"
                        row["hard_fail"] = "1" if payload.hard_fail else "0"
                        row["violated_soft_rules"] = json.dumps(payload.violated_soft_rules, ensure_ascii=False)

                    try:
                        writer.writerow(row)
                    except Exception as exc:  # noqa: BLE001
                        LOGGER.warning("CSV write failure for track_id=%s: %s", payload.track_id, exc)

                    if self.save_defect_snapshot and payload.result == "suspected_defect":
                        self._save_snapshot(payload)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Logger worker crashed")
            self._set_status("ERROR", "Logger worker lỗi", last_error=str(exc))
            return

        if self.get_status().state != "ERROR":
            self._set_status("STOPPED", "Đã dừng")

    def _save_snapshot(self, payload: LogPayload) -> None:
        if payload.snapshot_frame is None:
            return

        image = payload.snapshot_frame.copy()

        x1, y1, x2, y2 = payload.roi_rect_xyxy
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 180, 0), 2)

        tx1, ty1, tx2, ty2 = payload.trigger_band_xyxy
        cv2.rectangle(image, (tx1, ty1), (tx2, ty2), (0, 255, 255), 2)

        bx1, by1, bx2, by2 = [int(v) for v in payload.bbox_frame_xyxy]
        cv2.rectangle(image, (bx1, by1), (bx2, by2), (0, 0, 255), 2)

        contour = payload.contour_frame.astype(np.int32)
        cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)

        lines = _wrap_overlay_text(
            [
                f"track={payload.track_id} {payload.result}",
                f"score={payload.score:.2f}",
                f"reasons={','.join(payload.reasons) if payload.reasons else 'none'}",
            ],
            max_width_px=max(80, image.shape[1] - 20),
            font_scale=0.6,
            thickness=2,
        )
        y = 30
        for line in lines:
            cv2.putText(image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
            y += 24

        for key in ["area", "length", "width", "aspect_ratio"]:
            value = payload.measurements.get(key, 0.0)
            cv2.putText(
                image,
                f"{key}={value:.2f}",
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            y += 24

        output = self._snapshot_dir / f"track_{payload.track_id:06d}_frame_{payload.frame_id:09d}.jpg"
        try:
            cv2.imwrite(str(output), image)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Snapshot write failure for track_id=%s: %s", payload.track_id, exc)

    def _set_status(self, state: str, message: str, last_error: str | None = None) -> None:
        with self._status_lock:
            self._status = WorkerStatus(
                name="LoggerWorker",
                state=state,
                message=message,
                last_error=last_error,
            )


def _wrap_overlay_text(
    parts: list[str],
    *,
    max_width_px: int,
    font_scale: float,
    thickness: int,
) -> list[str]:
    lines: list[str] = []
    for part in parts:
        tokens = part.split(",") if part.startswith("reasons=") else part.split()
        current = ""
        separator = "," if part.startswith("reasons=") else " "
        for token in tokens:
            candidate = token if not current else f"{current}{separator}{token}"
            width = cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0][0]
            if current and width > max_width_px:
                lines.append(current)
                current = token
            else:
                current = candidate
        if current:
            lines.append(current)
    return lines
