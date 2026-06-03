from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..camera.core import CameraWorker
from ..config import Profile
from ..paths import resolve_model_path
from ..worker_status import WorkerStatus
from . import (
    Detection,
    ModelDiagnostics,
    OnnxRuntimeEngine,
    postprocess_segmentation,
    preprocess_roi,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class InferencePacket:
    frame_id: int
    timestamp: float
    frame_bgr: np.ndarray
    detections: list[Detection]
    latency_ms: float


class InferenceWorker:
    """Consumes latest frame and publishes latest inference result."""

    def __init__(
        self,
        camera_worker: CameraWorker,
        profile: Profile,
        app_root: Path,
        poll_interval_sec: float = 0.001,
    ) -> None:
        self.camera_worker = camera_worker
        self.profile = profile
        self.app_root = app_root
        self.poll_interval_sec = poll_interval_sec

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

        self._latest_lock = threading.Lock()
        self._latest_result: InferencePacket | None = None

        self._status_lock = threading.Lock()
        self._status = WorkerStatus(name="InferenceWorker", state="IDLE", message="Chưa khởi động")

        self._last_frame_id = -1
        self._engine = OnnxRuntimeEngine()
        self._model_loaded = False
        self._diagnostics: ModelDiagnostics | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._set_status("STARTING", "Đang khởi tạo")
        self._thread = threading.Thread(target=self._run, name="InferenceWorker", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        self._engine.close()
        self._model_loaded = False
        if self.get_status().state != "ERROR":
            self._set_status("STOPPED", "Đã dừng")

    def get_latest_result(self) -> InferencePacket | None:
        with self._latest_lock:
            return self._latest_result

    def get_status(self) -> WorkerStatus:
        with self._status_lock:
            return self._status

    def get_diagnostics(self) -> ModelDiagnostics | None:
        return self._diagnostics

    def _load_model_if_needed(self) -> None:
        if self._model_loaded:
            return
        model_path = resolve_model_path(self.app_root, self.profile.model.path)
        self._diagnostics = self._engine.load(str(model_path), self.profile.model)
        self._model_loaded = True

    def _run(self) -> None:
        try:
            self._set_status("STARTING", "Đang tải model")
            self._load_model_if_needed()
            self._set_status("RUNNING", "Đang suy luận")

            while not self._stop.is_set():
                frame_packet = self.camera_worker.get_latest_frame()
                if frame_packet is None:
                    time.sleep(self.poll_interval_sec)
                    continue
                if frame_packet.frame_id == self._last_frame_id:
                    time.sleep(self.poll_interval_sec)
                    continue

                self._last_frame_id = frame_packet.frame_id
                start = time.perf_counter()
                frame_copy = frame_packet.frame.copy()

                status = self.camera_worker.get_status()
                if not status.frame_size_match:
                    result = InferencePacket(
                        frame_id=frame_packet.frame_id,
                        timestamp=frame_packet.timestamp,
                        frame_bgr=frame_copy,
                        detections=[],
                        latency_ms=(time.perf_counter() - start) * 1000.0,
                    )
                    with self._latest_lock:
                        self._latest_result = result
                    continue

                detections = self._infer_frame(frame_copy)
                latency_ms = (time.perf_counter() - start) * 1000.0
                result = InferencePacket(
                    frame_id=frame_packet.frame_id,
                    timestamp=frame_packet.timestamp,
                    frame_bgr=frame_copy,
                    detections=detections,
                    latency_ms=latency_ms,
                )
                with self._latest_lock:
                    self._latest_result = result
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Inference worker crashed")
            self._set_status("ERROR", "Inference worker lỗi", last_error=str(exc))
            self._stop.set()
        finally:
            if self.get_status().state != "ERROR":
                self._set_status("STOPPED", "Đã dừng")

    def _infer_frame(self, frame: np.ndarray) -> list[Detection]:
        region = self.profile.inspection_region
        roi = frame[region.y : region.y + region.h, region.x : region.x + region.w]
        if roi.size == 0:
            return []

        prep = preprocess_roi(
            roi,
            roi_origin_xy=(region.x, region.y),
            input_size=self.profile.model.input_size,
            normalize=self.profile.model.preprocess.normalize,
            color_format=self.profile.model.preprocess.color_format,
        )
        det_out, proto_out = self._engine.infer(prep.tensor)
        detections = postprocess_segmentation(
            det_out,
            proto_out,
            preprocess=prep,
            model_spec=self.profile.model,
            conf_threshold=self.profile.model.conf_threshold,
            iou_threshold=self.profile.model.iou_threshold,
        )
        return detections

    def _set_status(self, state: str, message: str, last_error: str | None = None) -> None:
        with self._status_lock:
            self._status = WorkerStatus(
                name="InferenceWorker",
                state=state,
                message=message,
                last_error=last_error,
            )
