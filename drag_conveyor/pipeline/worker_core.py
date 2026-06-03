from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

import numpy as np

from ..inference.worker import InferenceWorker
from ..logging.core import LoggerWorker
from ..worker_status import WorkerStatus
from .core import CounterSnapshot, PipelineCore, TrackOverlay, TriggerEventOverlay

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DisplayPacket:
    frame_id: int
    frame_bgr: np.ndarray
    roi_rect_xyxy: tuple[int, int, int, int]
    trigger_band_xyxy: tuple[int, int, int, int]
    tracks: list[TrackOverlay]
    events: list[TriggerEventOverlay]


class PipelineWorker:
    """Consumes latest inference results and runs pipeline core in background thread."""

    def __init__(
        self,
        inference_worker: InferenceWorker,
        pipeline_core: PipelineCore,
        run_id: str,
        logger_worker: LoggerWorker | None,
        poll_interval_sec: float = 0.001,
    ) -> None:
        self.inference_worker = inference_worker
        self.pipeline_core = pipeline_core
        self.run_id = run_id
        self.logger_worker = logger_worker
        self.poll_interval_sec = poll_interval_sec

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._paused = threading.Event()

        self._status_lock = threading.Lock()
        self._status = WorkerStatus(name="PipelineWorker", state="IDLE", message="Chưa khởi động")

        self._counter_lock = threading.Lock()
        self._latest_counters = CounterSnapshot(
            total_processed_bars=0,
            normal_bars=0,
            suspected_defect_bars=0,
            current_tracked_bars=0,
        )
        self._display_lock = threading.Lock()
        self._latest_display_packet: DisplayPacket | None = None

        self._last_processed_frame_id = -1

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._paused.clear()
        self._set_status("STARTING", "Đang khởi tạo pipeline")
        self._thread = threading.Thread(target=self._run, name="PipelineWorker", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        if self.get_status().state != "ERROR":
            self._set_status("STOPPED", "Đã dừng")

    def pause(self) -> None:
        self._paused.set()

    def resume(self) -> None:
        self._paused.clear()

    def get_status(self) -> WorkerStatus:
        with self._status_lock:
            return self._status

    def get_latest_counters(self) -> CounterSnapshot:
        with self._counter_lock:
            return self._latest_counters

    def get_latest_display_packet(self) -> DisplayPacket | None:
        with self._display_lock:
            packet = self._latest_display_packet
            if packet is None:
                return None
            return DisplayPacket(
                frame_id=packet.frame_id,
                frame_bgr=packet.frame_bgr.copy(),
                roi_rect_xyxy=packet.roi_rect_xyxy,
                trigger_band_xyxy=packet.trigger_band_xyxy,
                tracks=[
                    TrackOverlay(
                        track_id=t.track_id,
                        confirmed=t.confirmed,
                        missed_frames=t.missed_frames,
                        bbox_frame_xyxy=t.bbox_frame_xyxy,
                        centroid_frame_xy=t.centroid_frame_xy,
                        mask_roi=t.mask_roi.copy(),
                        contour_frame=t.contour_frame.copy(),
                    )
                    for t in packet.tracks
                ],
                events=[
                    TriggerEventOverlay(
                        frame_id=e.frame_id,
                        track_id=e.track_id,
                        result=e.result,
                        score=e.score,
                        reasons=list(e.reasons),
                        bbox_frame_xyxy=e.bbox_frame_xyxy,
                    )
                    for e in packet.events
                ],
            )

    def _run(self) -> None:
        self._set_status("RUNNING", "Đang xử lý pipeline")
        try:
            while not self._stop.is_set():
                if self._paused.is_set():
                    time.sleep(self.poll_interval_sec)
                    continue

                infer_packet = self.inference_worker.get_latest_result()
                if infer_packet is None:
                    time.sleep(self.poll_interval_sec)
                    continue
                if infer_packet.frame_id == self._last_processed_frame_id:
                    time.sleep(self.poll_interval_sec)
                    continue

                self._last_processed_frame_id = infer_packet.frame_id
                counters = self.pipeline_core.process(
                    run_id=self.run_id,
                    frame_id=infer_packet.frame_id,
                    frame=infer_packet.frame_bgr,
                    detections=infer_packet.detections,
                    latency_ms=infer_packet.latency_ms,
                    logger=self.logger_worker,
                )
                with self._counter_lock:
                    self._latest_counters = counters
                self._publish_display_packet(infer_packet.frame_id, infer_packet.frame_bgr)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Pipeline worker crashed")
            self._set_status("ERROR", "Pipeline worker lỗi", last_error=str(exc))
            self._stop.set()
        finally:
            if self.get_status().state != "ERROR":
                self._set_status("STOPPED", "Đã dừng")

    def _set_status(self, state: str, message: str, last_error: str | None = None) -> None:
        with self._status_lock:
            self._status = WorkerStatus(
                name="PipelineWorker",
                state=state,
                message=message,
                last_error=last_error,
            )

    def _publish_display_packet(self, frame_id: int, frame_bgr: np.ndarray) -> None:
        if not hasattr(self.pipeline_core, "profile") or not hasattr(self.pipeline_core, "band"):
            return
        if not hasattr(self.pipeline_core, "get_latest_tracks") or not hasattr(self.pipeline_core, "get_latest_events"):
            return

        region = self.pipeline_core.profile.inspection_region
        band = self.pipeline_core.band
        roi_rect = (region.x, region.y, region.x + region.w, region.y + region.h)
        trigger_band = (band.x1, band.y1, band.x2, band.y2)
        tracks = self.pipeline_core.get_latest_tracks()
        events = self.pipeline_core.get_latest_events()

        packet = DisplayPacket(
            frame_id=frame_id,
            frame_bgr=frame_bgr.copy(),
            roi_rect_xyxy=roi_rect,
            trigger_band_xyxy=trigger_band,
            tracks=tracks,
            events=events,
        )
        with self._display_lock:
            self._latest_display_packet = packet
