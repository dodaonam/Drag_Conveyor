from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from ..app.runtime_controller import RuntimeController
from ..calibration_runner import build_calibration_artifacts_dir, run_auto_calibration
from ..camera_io import probe_camera_indices
from ..camera_worker import CameraWorker
from ..config import Profile, ProfileError, load_profile, save_profile
from ..demo_video import is_video_file_source, probe_and_sync_profile_to_video
from ..paths import resolve_model_path, resolve_paths
from ..state_machine import AppState
from .result_panel import ResultPanel
from .settings_panel import SettingsPanel
from .video_canvas import VideoCanvas

LOGGER = logging.getLogger(__name__)


class MainWindow(QtWidgets.QMainWindow):
    calibrationFinished = QtCore.Signal(bool, str, object)

    def __init__(self, app_root: Path, profile_path: Path) -> None:
        super().__init__()
        self.setWindowTitle("White Bar Inspection V1")
        self.resize(1400, 860)

        self.app_root = app_root
        self.paths = resolve_paths(app_root)
        self.profile_path = profile_path
        self.profile = self._load_profile_or_default(profile_path)

        self._preview_worker: CameraWorker | None = None
        self._runtime: RuntimeController | None = None
        self._runtime_started = False
        self._calibration_thread: threading.Thread | None = None

        self.video_canvas = VideoCanvas(self)
        self.settings_panel = SettingsPanel(self)
        self.result_panel = ResultPanel(self)
        self._build_layout()
        self._wire_signals()
        self._refresh_controls_from_profile()
        self._refresh_model_status()
        self._set_worker_idle()

        self._ui_timer = QtCore.QTimer(self)
        self._ui_timer.setInterval(33)
        self._ui_timer.timeout.connect(self._ui_loop)
        self._ui_timer.start()

    def _build_layout(self) -> None:
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)

        root_layout = QtWidgets.QVBoxLayout(central)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(8)

        top_layout = QtWidgets.QHBoxLayout()
        top_layout.setSpacing(8)
        top_layout.addWidget(self.video_canvas, 3)
        top_layout.addWidget(self.settings_panel, 1)
        root_layout.addLayout(top_layout, 1)
        root_layout.addWidget(self.result_panel, 0)

    def _wire_signals(self) -> None:
        self.settings_panel.selectVideoRequested.connect(self.select_video)
        self.settings_panel.scanCameraRequested.connect(self.scan_cameras)
        self.settings_panel.connectCameraRequested.connect(self.connect_camera)
        self.settings_panel.saveProfileRequested.connect(self.save_profile_action)
        self.settings_panel.loadProfileRequested.connect(self.load_profile_action)
        self.settings_panel.calibrateRequested.connect(self.start_calibration)
        self.settings_panel.startDetectionRequested.connect(self.start_detection)
        self.settings_panel.pauseDetectionRequested.connect(self.pause_detection)
        self.settings_panel.resumeDetectionRequested.connect(self.resume_detection)
        self.settings_panel.stopDetectionRequested.connect(self.stop_detection)
        self.settings_panel.freezeToggled.connect(self.video_canvas.set_freeze_enabled)
        self.settings_panel.controlsChanged.connect(self._on_controls_changed)
        self.video_canvas.regionChanged.connect(self._on_region_changed)
        self.calibrationFinished.connect(self._on_calibration_finished)

    def _load_profile_or_default(self, profile_path: Path) -> Profile:
        if not profile_path.exists():
            return Profile()
        try:
            return load_profile(profile_path)
        except ProfileError as exc:
            LOGGER.warning("Failed to load profile %s: %s", profile_path, exc)
            QtWidgets.QMessageBox.warning(
                self,
                "Profile",
                f"Khong nap duoc profile: {exc}\nSu dung profile mac dinh.",
            )
            return Profile()

    def _refresh_controls_from_profile(self, *, preserve_source: bool = True) -> None:
        current_source = self.settings_panel.source() if preserve_source else ""
        self.settings_panel.set_profile(self.profile)
        if current_source:
            self.settings_panel.set_source(current_source)
        self.video_canvas.set_profile(self.profile)

    def _start_preview(self, source: str) -> None:
        self._preview_worker = CameraWorker(
            source=source,
            profile=self.profile,
            loop_file=is_video_file_source(source),
        )
        self._preview_worker.start()

    def _on_controls_changed(self) -> None:
        self._apply_controls_to_profile()
        self.video_canvas.update()

    def _on_region_changed(self) -> None:
        self.result_panel.set_message(
            (
                "ROI cap nhat: "
                f"x={self.profile.inspection_region.x}, y={self.profile.inspection_region.y}, "
                f"w={self.profile.inspection_region.w}, h={self.profile.inspection_region.h}"
            )
        )

    def _apply_controls_to_profile(self) -> None:
        region = self.profile.inspection_region
        region.direction = "top_to_bottom"

    def _refresh_model_status(self) -> None:
        model_path = resolve_model_path(self.paths.root, self.profile.model.path)
        if model_path.exists():
            status = f"San sang ({model_path.name})"
        else:
            status = f"Thieu model: {model_path}"
        self.settings_panel.set_model_status(status)
        self.result_panel.set_model(status)

    def scan_cameras(self) -> None:
        self._apply_controls_to_profile()
        probes = probe_camera_indices(
            max_index=10,
            width=self.profile.camera.width,
            height=self.profile.camera.height,
            fps=self.profile.camera.fps,
        )
        if not probes:
            self.result_panel.set_message("Khong tim thay camera o index 0..9.")
            return

        selected = next((probe for probe in probes if probe.index == self.profile.camera.index), probes[0])
        self.settings_panel.set_source(str(selected.index))
        listing = ", ".join(
            f"{probe.index}:{probe.backend}:{probe.width}x{probe.height}"
            for probe in probes[:4]
        )
        self.result_panel.set_message(f"Da tim thay camera: {listing}. Da chon source={selected.index}.")

    def select_video(self) -> None:
        if self._calibration_thread is not None and self._calibration_thread.is_alive():
            self.result_panel.set_message("Dang lay chuan. Hay cho hoan tat truoc khi doi video.")
            return

        self._stop_runtime_if_needed(update_message=False)
        self._stop_preview_if_needed()

        selected, _filter = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Chon video demo",
            str(self.paths.root / "data" / "raw_data"),
            "Video files (*.mp4 *.avi *.mov *.mkv);;All files (*.*)",
        )
        if not selected:
            return

        try:
            info, sync_result = probe_and_sync_profile_to_video(self.profile, selected)
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(self, "Video demo", f"Khong mo duoc video: {exc}")
            return

        self.profile = sync_result.profile
        self._refresh_controls_from_profile(preserve_source=False)
        self.settings_panel.set_source(selected)
        self.video_canvas.set_frame(info.first_frame_bgr)
        self._refresh_model_status()

        self._start_preview(selected)
        self.result_panel.set_state(AppState.CAMERA_CONNECTED.value)

        notes = [f"Da chon video demo: {Path(selected).name} ({info.width}x{info.height})"]
        if sync_result.calibration_cleared:
            notes.append("Can lay chuan lai do video/resolution da doi.")
        if sync_result.roi_reset:
            notes.append("ROI da duoc reset theo kich thuoc video.")
        self.result_panel.set_message(" ".join(notes))

    def connect_camera(self) -> None:
        self._stop_runtime_if_needed(update_message=False)
        self._stop_preview_if_needed()
        self._apply_controls_to_profile()

        source = self.settings_panel.source()
        if not source:
            self.result_panel.set_message("Source trong. Hay nhap camera index.")
            return

        self._start_preview(source)
        self.result_panel.set_state(AppState.CAMERA_CONNECTED.value)
        if is_video_file_source(source):
            self.result_panel.set_message("Dang phat demo video...")
        else:
            self.result_panel.set_message("Dang ket noi source...")

    def save_profile_action(self) -> None:
        self._apply_controls_to_profile()
        try:
            save_profile(self.profile, self.profile_path)
        except Exception as exc:  # noqa: BLE001
            self.result_panel.set_message(f"Luu profile that bai: {exc}")
            return
        self.result_panel.set_message(f"Da luu profile: {self.profile_path}")

    def load_profile_action(self) -> None:
        self._stop_runtime_if_needed(update_message=False)
        self._stop_preview_if_needed()

        if not self.profile_path.exists():
            QtWidgets.QMessageBox.warning(self, "Profile", f"Khong tim thay: {self.profile_path}")
            return
        try:
            self.profile = load_profile(self.profile_path)
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(self, "Profile", f"Nap profile that bai: {exc}")
            return

        self._refresh_controls_from_profile(preserve_source=False)
        self._refresh_model_status()
        self.result_panel.set_message(f"Da nap profile: {self.profile_path}")

    def start_calibration(self) -> None:
        if self._runtime_started:
            self.result_panel.set_message("Dang chay detection. Hay Stop truoc khi calibrate.")
            return
        if self._calibration_thread is not None and self._calibration_thread.is_alive():
            return

        self._apply_controls_to_profile()
        source = self.settings_panel.source()
        if not source:
            self.result_panel.set_message("Source trong. Khong the calibrate.")
            return

        model_path = resolve_model_path(self.paths.root, self.profile.model.path)
        if not model_path.exists():
            self.result_panel.set_message(f"Khong tim thay model: {model_path}")
            return

        self.settings_panel.btn_calibrate.setEnabled(False)
        self.result_panel.set_state(AppState.CALIBRATING.value)
        self.result_panel.set_message("Dang chay lay chuan tu dong...")
        artifacts_dir = build_calibration_artifacts_dir(self.paths.runtime_dir)

        def work() -> None:
            try:
                result = run_auto_calibration(
                    profile=self.profile.clone(),
                    source=source,
                    model_path=str(model_path),
                    max_frames=6000,
                    show_preview=False,
                    artifacts_dir=artifacts_dir,
                    loop_video=is_video_file_source(source),
                )
                outcome = result.outcome
                if outcome.success and outcome.updated_profile is not None:
                    self.calibrationFinished.emit(
                        True,
                        (
                            "Lay chuan thanh cong: "
                            f"records={result.records_collected}, "
                            f"inlier_ratio={outcome.calibration_result.inlier_ratio:.3f}, "
                            f"report={result.report_json_path}"
                        ),
                        outcome.updated_profile,
                    )
                else:
                    self.calibrationFinished.emit(
                        False,
                        (
                            f"Lay chuan that bai: {outcome.reason} "
                            f"(records={result.records_collected}, report={result.report_json_path})"
                        ),
                        None,
                    )
            except Exception as exc:  # noqa: BLE001
                self.calibrationFinished.emit(False, f"Calibration loi: {exc}", None)

        self._calibration_thread = threading.Thread(target=work, name="CalibrationThread", daemon=True)
        self._calibration_thread.start()

    @QtCore.Slot(bool, str, object)
    def _on_calibration_finished(self, success: bool, message: str, updated_profile: object) -> None:
        self.settings_panel.btn_calibrate.setEnabled(True)
        if success and isinstance(updated_profile, Profile):
            self.profile = updated_profile
            try:
                save_profile(self.profile, self.profile_path)
            except Exception as exc:  # noqa: BLE001
                self.result_panel.set_message(f"Calibration xong nhung luu profile that bai: {exc}")
                return
            self._refresh_controls_from_profile()
            self.result_panel.set_state(AppState.READY.value)
        else:
            self.result_panel.set_state(AppState.SETUP.value)

        self.result_panel.set_message(message)

    def start_detection(self) -> None:
        if self._runtime_started:
            return
        self._apply_controls_to_profile()

        if self.profile.calibration_result is None:
            QtWidgets.QMessageBox.warning(
                self,
                "Calibration",
                "Chua co calibration_result. Hay chay 'Lay chuan tu dong' truoc.",
            )
            return

        source = self.settings_panel.source()
        if not source:
            self.result_panel.set_message("Source trong. Khong the start detection.")
            return

        model_path = resolve_model_path(self.paths.root, self.profile.model.path)
        if not model_path.exists():
            QtWidgets.QMessageBox.warning(self, "Model", f"Khong tim thay model: {model_path}")
            self._refresh_model_status()
            return

        self._stop_preview_if_needed()
        runtime = RuntimeController(
            profile=self.profile,
            deployment_paths=self.paths,
            model_path=model_path,
        )
        try:
            runtime.load_model()
            diagnostics = runtime.model_diagnostics
            if diagnostics is not None:
                providers = ",".join(diagnostics.providers)
                model_text = f"Da nap ({providers})"
            else:
                model_text = "Da nap"
            self.settings_panel.set_model_status(model_text)
            self.result_panel.set_model(model_text)

            runtime.start(
                source=source,
                enable_logging=True,
                loop_file=is_video_file_source(source),
            )
        except Exception as exc:  # noqa: BLE001
            runtime.stop()
            if is_video_file_source(source):
                self._start_preview(source)
                self.result_panel.set_message(f"Start detection loi: {exc}. Da khoi phuc preview video.")
                self.result_panel.set_state(AppState.CAMERA_CONNECTED.value)
            else:
                self.result_panel.set_message(f"Start detection loi: {exc}")
                self.result_panel.set_state(AppState.ERROR.value)
            return

        self._runtime = runtime
        self._runtime_started = True
        self.result_panel.set_state(AppState.RUNNING.value)
        self.result_panel.set_message("Dang chay detection.")

    def pause_detection(self) -> None:
        if not self._runtime_started or self._runtime is None:
            return
        self._runtime.pause()
        self.result_panel.set_state(AppState.PAUSED.value)

    def resume_detection(self) -> None:
        if not self._runtime_started or self._runtime is None:
            return
        self._runtime.resume()
        self.result_panel.set_state(AppState.RUNNING.value)

    def stop_detection(self) -> None:
        self._stop_runtime_if_needed(update_message=True)

    def _stop_runtime_if_needed(self, update_message: bool) -> None:
        if self._runtime is None:
            self._runtime_started = False
            return
        summary = self._runtime.get_summary()
        self._runtime.stop()
        self._runtime = None
        self._runtime_started = False
        self._set_worker_idle()
        self.result_panel.set_state(AppState.READY.value)
        self.result_panel.set_perf(
            summary.avg_fps,
            summary.inference_fps_estimate,
            summary.avg_latency_ms,
            summary.p95_latency_ms,
        )
        if update_message:
            self.result_panel.set_message(
                (
                    f"Run xong: {summary.run_id} "
                    f"processed={summary.total_processed_bars} "
                    f"throughput_fps={summary.avg_fps:.2f}"
                )
            )

    def _stop_preview_if_needed(self) -> None:
        if self._preview_worker is None:
            return
        self._preview_worker.stop()
        self._preview_worker = None

    def _ui_loop(self) -> None:
        if self._runtime is not None and self._runtime_started:
            self._update_from_runtime()
            return
        if self._preview_worker is not None:
            self._update_from_preview()
            return
        self.result_panel.set_camera("Chua ket noi")

    def _update_from_runtime(self) -> None:
        if self._runtime is None:
            return
        snapshot = self._runtime.snapshot()
        status = snapshot.status
        summary = snapshot.summary
        if status is None:
            if snapshot.last_error:
                self.result_panel.set_state(AppState.ERROR.value)
                self.result_panel.set_message(snapshot.last_error)
                self._stop_runtime_if_needed(update_message=False)
            if summary is not None:
                self.result_panel.set_perf(
                    summary.avg_fps,
                    summary.inference_fps_estimate,
                    summary.avg_latency_ms,
                    summary.p95_latency_ms,
                )
            return

        self.result_panel.set_state(status.app_state.value)
        self.result_panel.set_camera(status.camera_status.message)
        self.result_panel.set_worker_states(
            status.inference_status.state,
            status.pipeline_status.state,
            status.logger_status.state,
        )
        self.result_panel.set_counters(
            status.counters.total_processed_bars,
            status.counters.normal_bars,
            status.counters.suspected_defect_bars,
        )

        if summary is None:
            summary = self._runtime.get_summary()
        self.result_panel.set_perf(
            summary.avg_fps,
            summary.inference_fps_estimate,
            summary.avg_latency_ms,
            summary.p95_latency_ms,
        )

        display_packet = snapshot.display_packet or self._runtime.get_latest_display_packet()
        if display_packet is not None:
            self.video_canvas.set_display_packet(display_packet)
        else:
            frame = self._runtime.get_latest_frame_for_display()
            if frame is not None:
                self.video_canvas.set_frame(frame)

        if snapshot.last_error:
            self.result_panel.set_message(snapshot.last_error)
        elif status.last_error:
            self.result_panel.set_message(status.last_error)

    def _update_from_preview(self) -> None:
        if self._preview_worker is None:
            return
        status = self._preview_worker.get_status()
        self.result_panel.set_camera(status.message)
        self.result_panel.set_worker_states("IDLE", "IDLE", "IDLE")

        if status.state == "ERROR":
            self.result_panel.set_state(AppState.ERROR.value)
        elif status.state == "RECONNECTING":
            self.result_panel.set_state(AppState.CAMERA_LOST.value)
        elif status.state == "CONNECTED":
            self.result_panel.set_state(AppState.CAMERA_CONNECTED.value)
        elif status.state == "STOPPED":
            self.result_panel.set_state(AppState.IDLE.value)

        packet = self._preview_worker.get_latest_frame()
        if packet is not None:
            self.video_canvas.set_frame(packet.frame)

    def _set_worker_idle(self) -> None:
        self.result_panel.set_worker_states("IDLE", "IDLE", "IDLE")

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self._ui_timer.stop()
        self._stop_runtime_if_needed(update_message=False)
        self._stop_preview_if_needed()
        super().closeEvent(event)


def run_gui(app_root: Path, profile_path: Path, smoke_seconds: float | None = None) -> None:
    app = QtWidgets.QApplication.instance()
    owns_app = app is None
    if app is None:
        app = QtWidgets.QApplication(sys.argv)

    window = MainWindow(app_root=app_root, profile_path=profile_path)
    window.showMaximized()
    if smoke_seconds is not None and smoke_seconds > 0:
        QtCore.QTimer.singleShot(int(smoke_seconds * 1000), window.close)

    if owns_app:
        app.exec()
