from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AppState(str, Enum):
    IDLE = "IDLE"
    CAMERA_CONNECTED = "CAMERA_CONNECTED"
    SETUP = "SETUP"
    CALIBRATING = "CALIBRATING"
    READY = "READY"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    CAMERA_LOST = "CAMERA_LOST"
    ERROR = "ERROR"


@dataclass(slots=True)
class ReadinessFlags:
    camera_ready: bool = False
    model_ready: bool = False
    profile_ready: bool = False
    frame_size_match: bool = False
    inspection_region_ready: bool = False
    calibration_ready: bool = False

    def detection_start_enabled(self) -> bool:
        return all(
            [
                self.camera_ready,
                self.model_ready,
                self.profile_ready,
                self.frame_size_match,
                self.inspection_region_ready,
                self.calibration_ready,
            ]
        )

    def calibration_start_enabled(self) -> bool:
        return all(
            [
                self.camera_ready,
                self.model_ready,
                self.profile_ready,
                self.frame_size_match,
                self.inspection_region_ready,
            ]
        )


TRANSITIONS: dict[AppState, dict[str, AppState]] = {
    AppState.IDLE: {"connect_camera_success": AppState.CAMERA_CONNECTED, "fatal_error": AppState.ERROR},
    AppState.CAMERA_CONNECTED: {"enter_setup": AppState.SETUP, "fatal_error": AppState.ERROR},
    AppState.SETUP: {"save_valid_profile": AppState.READY, "fatal_error": AppState.ERROR},
    AppState.READY: {
        "start_calibration": AppState.CALIBRATING,
        "start_detection": AppState.RUNNING,
        "edit_region_or_remap": AppState.SETUP,
        "fatal_error": AppState.ERROR,
    },
    AppState.CALIBRATING: {
        "calibration_success": AppState.READY,
        "calibration_failed_keep_previous": AppState.READY,
        "calibration_failed_invalid_roi_trigger": AppState.SETUP,
        "fatal_error": AppState.ERROR,
    },
    AppState.RUNNING: {
        "camera_lost": AppState.CAMERA_LOST,
        "user_pause": AppState.PAUSED,
        "user_stop": AppState.READY,
        "fatal_error": AppState.ERROR,
    },
    AppState.PAUSED: {
        "user_resume": AppState.RUNNING,
        "user_stop": AppState.READY,
        "fatal_error": AppState.ERROR,
    },
    AppState.CAMERA_LOST: {
        "reconnect_success_frame_match": AppState.RUNNING,
        "reconnect_success_frame_mismatch": AppState.READY,
        "fatal_error": AppState.ERROR,
    },
    AppState.ERROR: {"user_reset_session": AppState.IDLE},
}


class InvalidStateTransition(RuntimeError):
    pass


class AppStateMachine:
    def __init__(self) -> None:
        self._state = AppState.IDLE

    @property
    def state(self) -> AppState:
        return self._state

    def transition(self, event: str) -> AppState:
        next_state = TRANSITIONS.get(self._state, {}).get(event)
        if next_state is None:
            raise InvalidStateTransition(f"Invalid transition from {self._state} via event '{event}'")
        self._state = next_state
        return self._state

    def force_error(self) -> AppState:
        self._state = AppState.ERROR
        return self._state
