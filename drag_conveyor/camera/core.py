from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from ..camera_io import is_camera_index_source, open_capture_source
from ..config import Profile

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FramePacket:
    frame_id: int
    frame: np.ndarray
    timestamp: float
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class CameraStatus:
    state: str
    frame_size_match: bool
    message: str


class CameraWorker:
    """Reads camera frames continuously with latest-frame-only buffer."""

    def __init__(
        self,
        source: str,
        profile: Profile,
        max_consecutive_read_failures: int = 10,
        reconnect_interval_sec: float = 2.0,
        loop_file: bool = False,
    ) -> None:
        self.source = source
        self.profile = profile
        self.max_consecutive_read_failures = max_consecutive_read_failures
        self.reconnect_interval_sec = reconnect_interval_sec
        self.loop_file = loop_file

        self._capture: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

        self._latest_lock = threading.Lock()
        self._latest_frame: FramePacket | None = None

        self._status_lock = threading.Lock()
        self._status = CameraStatus(state="IDLE", frame_size_match=False, message="Chua ket noi")

        self._frame_id = 0
        self._is_camera_source = is_camera_index_source(source)
        self._connected_message = "Da ket noi"
        self._file_frame_interval_sec = 0.0

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="CameraWorker", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        self._release_capture()
        self._set_status("STOPPED", False, "Da dung")

    def get_latest_frame(self) -> FramePacket | None:
        with self._latest_lock:
            if self._latest_frame is None:
                return None
            packet = self._latest_frame
            return FramePacket(
                frame_id=packet.frame_id,
                frame=packet.frame.copy(),
                timestamp=packet.timestamp,
                width=packet.width,
                height=packet.height,
            )

    def get_status(self) -> CameraStatus:
        with self._status_lock:
            return self._status

    def _run(self) -> None:
        fail_count = 0
        eof_reached = False
        try:
            while not self._stop.is_set():
                if self._capture is None or not self._capture.isOpened():
                    if not self._open_capture():
                        time.sleep(self.reconnect_interval_sec)
                        continue

                ok, frame = self._capture.read()
                if not ok or frame is None:
                    if not self._is_camera_source:
                        if self.loop_file and self._capture is not None:
                            self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            continue
                        eof_reached = True
                        break

                    fail_count += 1
                    if fail_count >= self.max_consecutive_read_failures:
                        self._set_status("RECONNECTING", False, "Dang ket noi lai")
                        self._release_capture()
                        fail_count = 0
                        time.sleep(self.reconnect_interval_sec)
                    continue

                fail_count = 0
                h, w = frame.shape[:2]
                frame_size_match = (
                    w == self.profile.inspection_region.frame_width
                    and h == self.profile.inspection_region.frame_height
                )
                if frame_size_match:
                    self._set_status("CONNECTED", True, self._connected_message)
                else:
                    self._set_status(
                        "CONNECTED",
                        False,
                        (
                            f"{self._connected_message}. Sai do phan giai: {w}x{h} != "
                            f"{self.profile.inspection_region.frame_width}x{self.profile.inspection_region.frame_height}"
                        ),
                    )

                self._frame_id += 1
                packet = FramePacket(
                    frame_id=self._frame_id,
                    frame=frame,
                    timestamp=time.time(),
                    width=w,
                    height=h,
                )
                with self._latest_lock:
                    self._latest_frame = packet

                if not self._is_camera_source and self._file_frame_interval_sec > 0:
                    time.sleep(self._file_frame_interval_sec)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Camera worker crashed")
            self._set_status("ERROR", False, f"Camera worker loi: {exc}")
        finally:
            self._release_capture()
            if eof_reached and self.get_status().state != "ERROR":
                self._set_status("STOPPED", True, "Ket thuc video")

    def _open_capture(self) -> bool:
        self._release_capture()
        backend_name = "UNKNOWN"

        if self._is_camera_source:
            capture, backend_name = open_capture_source(
                self.source,
                width=self.profile.camera.width,
                height=self.profile.camera.height,
                fps=self.profile.camera.fps,
                buffersize=1,
            )
            self._file_frame_interval_sec = 0.0
        else:
            capture, backend_name = open_capture_source(self.source)
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            self._file_frame_interval_sec = (1.0 / fps) if fps > 0 else 0.0

        self._capture = capture
        opened = capture.isOpened()
        self._connected_message = f"Da ket noi ({backend_name})"
        if opened:
            self._set_status("CONNECTED", False, self._connected_message)
        else:
            self._set_status("RECONNECTING", False, f"Khong mo duoc nguon: {self.source}. Dang ket noi lai")
        return opened

    def _release_capture(self) -> None:
        if self._capture is not None:
            try:
                self._capture.release()
            except Exception:  # noqa: BLE001
                LOGGER.exception("Failed to release capture")
            finally:
                self._capture = None

    def _set_status(self, state: str, frame_size_match: bool, message: str) -> None:
        with self._status_lock:
            self._status = CameraStatus(state=state, frame_size_match=frame_size_match, message=message)
