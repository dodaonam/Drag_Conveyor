from __future__ import annotations

import threading
import time
from pathlib import Path

from ..config import Profile
from ..paths import DeploymentPaths
from ..pipeline_worker import DisplayPacket
from ..runtime import InspectionRuntime, RunSummary
from ..state_machine import AppState
from .runtime_store import RuntimeSnapshot, RuntimeStore


class RuntimeController:
    """
    Application-layer runtime supervisor.

    - Owns InspectionRuntime workers lifecycle.
    - Runs background monitor loop to keep RuntimeStore updated.
    - UI reads snapshots from RuntimeStore instead of driving runtime.poll().
    """

    def __init__(
        self,
        *,
        profile: Profile,
        deployment_paths: DeploymentPaths,
        model_path: Path,
        monitor_interval_sec: float = 0.01,
    ) -> None:
        self._runtime = InspectionRuntime(
            profile=profile,
            deployment_paths=deployment_paths,
            model_path=model_path,
        )
        self._store = RuntimeStore()
        self._monitor_interval_sec = monitor_interval_sec
        self._monitor_thread: threading.Thread | None = None
        self._monitor_stop = threading.Event()

    @property
    def store(self) -> RuntimeStore:
        return self._store

    @property
    def state(self):
        return self._runtime.state

    @property
    def model_diagnostics(self):
        return self._runtime.model_diagnostics

    def load_model(self) -> None:
        self._runtime.load_model()

    def start(
        self,
        *,
        source: str,
        enable_logging: bool = True,
        run_id: str | None = None,
        loop_file: bool = False,
    ) -> str:
        run = self._runtime.start(
            source=source,
            enable_logging=enable_logging,
            run_id=run_id,
            loop_file=loop_file,
        )
        self._start_monitor()
        return run

    def stop(self) -> None:
        self._stop_monitor()
        self._runtime.stop()
        self._publish_snapshot(last_error=None)

    def pause(self) -> None:
        self._runtime.pause()

    def resume(self) -> None:
        self._runtime.resume()

    def get_summary(self) -> RunSummary:
        return self._runtime.get_summary()

    def get_latest_display_packet(self) -> DisplayPacket | None:
        return self._runtime.get_latest_display_packet()

    def get_latest_frame_for_display(self):
        return self._runtime.get_latest_frame_for_display()

    def snapshot(self) -> RuntimeSnapshot:
        return self._store.snapshot()

    def _start_monitor(self) -> None:
        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            return
        self._monitor_stop.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name="RuntimeMonitor",
            daemon=True,
        )
        self._monitor_thread.start()

    def _stop_monitor(self) -> None:
        self._monitor_stop.set()
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=2.0)
            self._monitor_thread = None

    def _monitor_loop(self) -> None:
        last_error: str | None = None
        while not self._monitor_stop.is_set():
            try:
                status = self._runtime.poll()
                summary = self._runtime.get_summary()
                display = self._runtime.get_latest_display_packet()
                last_error = status.last_error
                self._store.update(
                    status=status,
                    summary=summary,
                    display_packet=display,
                    last_error=last_error,
                )
                if status.app_state == AppState.ERROR:
                    break
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                self._store.update(
                    status=None,
                    summary=self._runtime.get_summary(),
                    display_packet=self._runtime.get_latest_display_packet(),
                    last_error=last_error,
                )
                break
            time.sleep(self._monitor_interval_sec)

    def _publish_snapshot(self, *, last_error: str | None) -> None:
        self._store.update(
            status=None,
            summary=self._runtime.get_summary(),
            display_packet=self._runtime.get_latest_display_packet(),
            last_error=last_error,
        )


__all__ = ["RuntimeController"]
