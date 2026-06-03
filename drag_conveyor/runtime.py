from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .camera_worker import CameraStatus, CameraWorker
from .config import Profile, validate_profile
from .inference import ModelDiagnostics, OnnxRuntimeEngine
from .inference_worker import InferenceWorker
from .logger_worker import LoggerWorker
from .paths import DeploymentPaths
from .pipeline_core import CounterSnapshot, PipelineCore
from .pipeline_worker import DisplayPacket, PipelineWorker
from .runtime_ids import generate_run_id
from .state_machine import AppState, AppStateMachine, InvalidStateTransition
from .worker_status import WorkerStatus

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RunSummary:
    run_id: str
    total_frames: int
    frames_inferred: int
    total_processed_bars: int
    normal_bars: int
    suspected_defect_bars: int
    current_tracked_bars: int
    avg_fps: float
    avg_latency_ms: float
    p95_latency_ms: float
    inference_fps_estimate: float
    elapsed_sec: float


@dataclass(frozen=True, slots=True)
class RuntimeStatus:
    app_state: AppState
    camera_status: CameraStatus
    inference_status: WorkerStatus
    pipeline_status: WorkerStatus
    logger_status: WorkerStatus
    counters: CounterSnapshot
    frame_size_match: bool
    last_error: str | None


class InspectionRuntime:
    """Threaded runtime using CameraWorker + InferenceWorker + PipelineCore."""

    def __init__(
        self,
        profile: Profile,
        deployment_paths: DeploymentPaths,
        model_path: Path,
    ) -> None:
        self.profile = profile
        self.paths = deployment_paths
        self.model_path = model_path

        self.state = AppStateMachine()
        self.pipeline = PipelineCore(profile)

        self._camera_worker: CameraWorker | None = None
        self._inference_worker: InferenceWorker | None = None
        self._pipeline_worker: PipelineWorker | None = None
        self._logger: LoggerWorker | None = None
        self._run_id: str | None = None
        self._paused = False

        self._model_diagnostics: ModelDiagnostics | None = None
        self._last_processed_frame_id = -1
        self._total_frames = 0
        self._inferred_frames = 0
        self._latency_sum_ms = 0.0
        self._last_latency_ms = 0.0
        self._latency_samples_ms: list[float] = []
        self._run_started_monotonic: float | None = None
        self._last_counters = CounterSnapshot(
            total_processed_bars=0,
            normal_bars=0,
            suspected_defect_bars=0,
            current_tracked_bars=0,
        )
        self._runtime_error_message: str | None = None

    @property
    def run_id(self) -> str | None:
        return self._run_id

    @property
    def model_diagnostics(self) -> ModelDiagnostics | None:
        return self._model_diagnostics

    def load_model(self) -> None:
        engine = OnnxRuntimeEngine()
        self._model_diagnostics = engine.load(str(self.model_path), self.profile.model)
        engine.close()
        LOGGER.info("Model diagnostics: %s", self._model_diagnostics)

    def start(
        self,
        source: str,
        enable_logging: bool = True,
        run_id: str | None = None,
        loop_file: bool = False,
        camera_ready_timeout_sec: float = 5.0,
        inference_ready_timeout_sec: float = 10.0,
        pipeline_ready_timeout_sec: float = 5.0,
    ) -> str:
        if self._camera_worker is not None:
            raise RuntimeError("Runtime is already started")

        validate_profile(self.profile)
        if self.profile.calibration_result is None:
            raise RuntimeError("Chưa có calibration_result. Hãy chạy calibration trước khi Start Detection.")
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        self.paths.ensure_runtime_dirs()
        self.pipeline.reset()
        self._last_processed_frame_id = -1
        self._total_frames = 0
        self._inferred_frames = 0
        self._latency_sum_ms = 0.0
        self._last_latency_ms = 0.0
        self._latency_samples_ms = []
        self._run_started_monotonic = time.perf_counter()
        self._last_counters = CounterSnapshot(
            total_processed_bars=0,
            normal_bars=0,
            suspected_defect_bars=0,
            current_tracked_bars=0,
        )
        self._paused = False
        self._runtime_error_message = None
        self.state = AppStateMachine()

        self._run_id = run_id or generate_run_id()

        try:
            self._camera_worker = CameraWorker(source=source, profile=self.profile, loop_file=loop_file)
            self._camera_worker.start()
            self._wait_camera_ready(timeout_sec=camera_ready_timeout_sec)

            self._inference_worker = InferenceWorker(
                camera_worker=self._camera_worker,
                profile=self.profile,
                app_root=self.paths.root,
            )
            self._inference_worker.start()
            self._wait_inference_ready(timeout_sec=inference_ready_timeout_sec)
            self._model_diagnostics = self._inference_worker.get_diagnostics()

            if enable_logging:
                self._logger = LoggerWorker(
                    logs_dir=self.paths.logs_dir,
                    defect_snapshots_root=self.paths.defect_snapshots_dir,
                    run_id=self._run_id,
                    save_defect_snapshot=self.profile.logging.save_defect_snapshot,
                    debug_enabled=self.profile.logging.save_debug_frames,
                )
                self._logger.start()
            else:
                self._logger = None

            self._pipeline_worker = PipelineWorker(
                inference_worker=self._inference_worker,
                pipeline_core=self.pipeline,
                run_id=self._run_id,
                logger_worker=self._logger,
            )
            self._pipeline_worker.start()
            self._wait_pipeline_ready(timeout_sec=pipeline_ready_timeout_sec)

            self.state.transition("connect_camera_success")
            self.state.transition("enter_setup")
            self.state.transition("save_valid_profile")
            self.state.transition("start_detection")
            return self._run_id
        except Exception as exc:  # noqa: BLE001
            self._runtime_error_message = str(exc)
            self._transition_fatal_error()
            self.stop()
            raise

    def stop(self) -> None:
        if self._pipeline_worker is not None:
            self._pipeline_worker.stop()
            self._pipeline_worker = None

        if self._inference_worker is not None:
            self._inference_worker.stop()
            self._inference_worker = None

        if self._camera_worker is not None:
            self._camera_worker.stop()
            self._camera_worker = None

        if self._logger is not None:
            self._logger.stop()
            self._logger = None

        self._paused = False
        if self.state.state in {AppState.RUNNING, AppState.PAUSED, AppState.CAMERA_LOST}:
            try:
                self.state.transition("user_stop")
            except InvalidStateTransition:
                pass

    def pause(self) -> None:
        if self.state.state == AppState.RUNNING:
            self.state.transition("user_pause")
            self._paused = True
            if self._pipeline_worker is not None:
                self._pipeline_worker.pause()

    def resume(self) -> None:
        if self.state.state == AppState.PAUSED:
            self.state.transition("user_resume")
            self._paused = False
            if self._pipeline_worker is not None:
                self._pipeline_worker.resume()

    def poll(self) -> RuntimeStatus:
        if self._camera_worker is None or self._inference_worker is None or self._pipeline_worker is None:
            raise RuntimeError("Runtime is not started")

        camera_status = self._camera_worker.get_status()
        self._update_state_from_camera(camera_status)

        infer_packet = self._inference_worker.get_latest_result()
        inference_status = self._inference_worker.get_status()
        pipeline_status = self._pipeline_worker.get_status()
        logger_status = (
            self._logger.get_status()
            if self._logger is not None
            else WorkerStatus(name="LoggerWorker", state="IDLE", message="Logger disabled")
        )

        self._update_state_from_workers(camera_status, inference_status, pipeline_status, logger_status)

        if infer_packet is not None:
            self._total_frames = max(self._total_frames, infer_packet.frame_id)
            self._last_latency_ms = infer_packet.latency_ms
            if infer_packet.frame_id != self._last_processed_frame_id:
                self._last_processed_frame_id = infer_packet.frame_id
                self._inferred_frames += 1
                self._latency_sum_ms += infer_packet.latency_ms
                self._latency_samples_ms.append(float(infer_packet.latency_ms))

        counters = self._pipeline_worker.get_latest_counters()
        self._last_counters = counters
        return RuntimeStatus(
            app_state=self.state.state,
            camera_status=camera_status,
            inference_status=inference_status,
            pipeline_status=pipeline_status,
            logger_status=logger_status,
            counters=counters,
            frame_size_match=camera_status.frame_size_match,
            last_error=self._runtime_error_message,
        )

    def get_latest_frame_for_display(self) -> np.ndarray | None:
        if self._pipeline_worker is not None:
            packet = self._pipeline_worker.get_latest_display_packet()
            if packet is not None:
                return packet.frame_bgr.copy()

        if self._inference_worker is not None:
            infer_packet = self._inference_worker.get_latest_result()
            if infer_packet is not None:
                return infer_packet.frame_bgr.copy()

        if self._camera_worker is None:
            return None
        cam_packet = self._camera_worker.get_latest_frame()
        if cam_packet is None:
            return None
        return cam_packet.frame

    def get_latest_display_packet(self) -> DisplayPacket | None:
        if self._pipeline_worker is None:
            return None
        return self._pipeline_worker.get_latest_display_packet()

    def get_summary(self) -> RunSummary:
        avg_latency_ms = self._latency_sum_ms / self._inferred_frames if self._inferred_frames else 0.0
        elapsed_sec = 0.0
        if self._run_started_monotonic is not None:
            elapsed_sec = max(0.0, time.perf_counter() - self._run_started_monotonic)
        throughput_fps = (self._inferred_frames / elapsed_sec) if elapsed_sec > 0 else 0.0
        p95_latency_ms = (
            float(np.percentile(np.array(self._latency_samples_ms, dtype=np.float32), 95))
            if self._latency_samples_ms
            else 0.0
        )
        inference_fps_estimate = 1000.0 / avg_latency_ms if avg_latency_ms > 0 else 0.0
        return RunSummary(
            run_id=self._run_id or "",
            total_frames=self._total_frames,
            frames_inferred=self._inferred_frames,
            total_processed_bars=self.pipeline.total_processed_bars,
            normal_bars=self.pipeline.normal_bars,
            suspected_defect_bars=self.pipeline.suspected_defect_bars,
            current_tracked_bars=self._last_counters.current_tracked_bars,
            avg_fps=throughput_fps,
            avg_latency_ms=avg_latency_ms,
            p95_latency_ms=p95_latency_ms,
            inference_fps_estimate=inference_fps_estimate,
            elapsed_sec=elapsed_sec,
        )

    def run(
        self,
        source: str,
        max_frames: int | None = None,
        show_preview: bool = False,
        loop_file: bool = False,
    ) -> RunSummary:
        if self.profile.calibration_result is None:
            raise RuntimeError(
                "Calibration thresholds are missing. Run calibration first or load a profile with calibration_result."
            )

        self.start(source=source, enable_logging=True, loop_file=loop_file)

        try:
            while True:
                status = self.poll()

                frame = self.get_latest_frame_for_display()
                if frame is not None and show_preview:
                    preview = self._draw_preview(frame)
                    cv2.imshow("White Bar Inspection", preview)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break

                if max_frames is not None and self._inferred_frames >= max_frames:
                    break

                camera_status = status.camera_status
                if camera_status.state == "STOPPED":
                    break
                if status.app_state == AppState.ERROR:
                    raise RuntimeError(status.last_error or "Runtime entered ERROR state")

                time.sleep(0.001)
        finally:
            self.stop()
            if show_preview:
                cv2.destroyAllWindows()

        return self.get_summary()

    def _update_state_from_camera(self, camera_status: CameraStatus) -> None:
        state = self.state.state

        if state == AppState.RUNNING and camera_status.state == "RECONNECTING":
            self.state.transition("camera_lost")
            return

        if state == AppState.CAMERA_LOST and camera_status.state == "CONNECTED":
            if camera_status.frame_size_match:
                self.state.transition("reconnect_success_frame_match")
            else:
                self.state.transition("reconnect_success_frame_mismatch")
            return

    def _update_state_from_workers(
        self,
        camera_status: CameraStatus,
        inference_status: WorkerStatus,
        pipeline_status: WorkerStatus,
        logger_status: WorkerStatus,
    ) -> None:
        if camera_status.state == "ERROR":
            self._runtime_error_message = camera_status.message
            self._transition_fatal_error()
            return

        if inference_status.state == "ERROR":
            self._runtime_error_message = inference_status.last_error or inference_status.message
            self._transition_fatal_error()
            return

        if pipeline_status.state == "ERROR":
            self._runtime_error_message = pipeline_status.last_error or pipeline_status.message
            self._transition_fatal_error()
            return

        if logger_status.state == "ERROR":
            self._runtime_error_message = logger_status.last_error or logger_status.message
            self._transition_fatal_error()
            return

    def _wait_camera_ready(self, timeout_sec: float) -> None:
        if self._camera_worker is None:
            raise RuntimeError("Camera worker is not initialized")

        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            status = self._camera_worker.get_status()
            frame = self._camera_worker.get_latest_frame()
            if status.state == "ERROR":
                raise RuntimeError(status.message)
            if frame is not None:
                if not status.frame_size_match:
                    raise RuntimeError(status.message)
                return
            time.sleep(0.02)

        raise TimeoutError("Camera readiness timeout: không nhận được frame đầu tiên")

    def _wait_inference_ready(self, timeout_sec: float) -> None:
        if self._inference_worker is None:
            raise RuntimeError("Inference worker is not initialized")

        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            status = self._inference_worker.get_status()
            if status.state == "ERROR":
                raise RuntimeError(status.last_error or status.message)
            if status.state == "RUNNING":
                if self._inference_worker.get_diagnostics() is None:
                    raise RuntimeError("Model diagnostics is missing")
                return
            time.sleep(0.02)

        raise TimeoutError("Inference readiness timeout")

    def _wait_pipeline_ready(self, timeout_sec: float) -> None:
        if self._pipeline_worker is None:
            raise RuntimeError("Pipeline worker is not initialized")

        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            status = self._pipeline_worker.get_status()
            if status.state == "ERROR":
                raise RuntimeError(status.last_error or status.message)
            if status.state == "RUNNING":
                return
            time.sleep(0.02)

        raise TimeoutError("Pipeline readiness timeout")

    def _transition_fatal_error(self) -> None:
        if self.state.state == AppState.ERROR:
            return
        try:
            self.state.transition("fatal_error")
        except InvalidStateTransition:
            self.state.force_error()

    def _draw_preview(self, frame: np.ndarray) -> np.ndarray:
        out = frame.copy()
        region = self.profile.inspection_region
        band = self.pipeline.band

        cv2.rectangle(out, (region.x, region.y), (region.x + region.w, region.y + region.h), (255, 180, 0), 2)
        cv2.rectangle(out, (band.x1, band.y1), (band.x2, band.y2), (0, 255, 255), 2)

        lines = [
            f"state={self.state.state.value}",
            f"processed={self.pipeline.total_processed_bars}",
            f"normal={self.pipeline.normal_bars}",
            f"suspected_defect={self.pipeline.suspected_defect_bars}",
            f"latency_ms={self._last_latency_ms:.2f}",
        ]
        y = 24
        for line in lines:
            cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y += 28

        return out
