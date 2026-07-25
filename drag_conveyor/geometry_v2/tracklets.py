from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum

from .tracking import KalmanConfig, KalmanState, initialize_from_two_measurements, predict, update


class TrackState(StrEnum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    LOST = "lost"
    FINALIZABLE = "finalizable"
    FINALIZED = "finalized"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class TrackMeasurement:
    observation_id: str
    source_frame_id: int
    timestamp_sec: float
    s_anchor: float
    s_anchor_sigma: float


@dataclass(frozen=True, slots=True)
class TrackLifecycleConfig:
    maximum_track_gap_sec: float = 0.35
    minimum_track_hits: int = 2
    minimum_track_duration_sec: float = 0.04
    maximum_seed_displacement_ratio: float = 0.08
    maximum_reverse_ratio: float = 0.015


@dataclass(frozen=True, slots=True)
class Tracklet:
    track_id: int
    state: TrackState
    measurements: tuple[TrackMeasurement, ...]
    kalman: KalmanState | None = None
    identity_ambiguity_tainted: bool = False

    @property
    def first_measurement(self) -> TrackMeasurement:
        return self.measurements[0]

    @property
    def last_measurement(self) -> TrackMeasurement:
        return self.measurements[-1]

    @classmethod
    def seed(cls, track_id: int, measurement: TrackMeasurement) -> "Tracklet":
        return cls(track_id=track_id, state=TrackState.TENTATIVE, measurements=(measurement,))

    def append_measurement(
        self,
        measurement: TrackMeasurement,
        *,
        chain_span_px: float,
        kalman_config: KalmanConfig,
        lifecycle_config: TrackLifecycleConfig,
        association_ambiguous: bool = False,
    ) -> "Tracklet":
        if self.state not in {TrackState.TENTATIVE, TrackState.CONFIRMED, TrackState.LOST}:
            raise ValueError("Only active tracklets can receive a measurement")
        previous = self.last_measurement
        if measurement.source_frame_id <= previous.source_frame_id or measurement.timestamp_sec <= previous.timestamp_sec:
            raise ValueError("Track measurements require strictly increasing source frame and timestamp")
        if measurement.timestamp_sec - previous.timestamp_sec > lifecycle_config.maximum_track_gap_sec:
            raise ValueError("Measurement arrives after the maximum track gap")
        if self.kalman is None:
            displacement = measurement.s_anchor - previous.s_anchor
            if abs(displacement) > lifecycle_config.maximum_seed_displacement_ratio * chain_span_px:
                raise ValueError("Seed displacement exceeds the configured gate")
            if max(0.0, previous.s_anchor - measurement.s_anchor) > lifecycle_config.maximum_reverse_ratio * chain_span_px:
                raise ValueError("Seed reverse displacement exceeds the configured gate")
            kalman = initialize_from_two_measurements(
                previous.s_anchor,
                previous.s_anchor_sigma,
                previous.timestamp_sec,
                measurement.s_anchor,
                measurement.s_anchor_sigma,
                measurement.timestamp_sec,
                kalman_config,
            )
        else:
            kalman = update(predict(self.kalman, measurement.timestamp_sec, kalman_config), measurement.s_anchor, measurement.s_anchor_sigma, kalman_config).state
        measurements = (*self.measurements, measurement)
        duration = measurements[-1].timestamp_sec - measurements[0].timestamp_sec
        state = TrackState.CONFIRMED if len(measurements) >= lifecycle_config.minimum_track_hits and duration >= lifecycle_config.minimum_track_duration_sec else TrackState.TENTATIVE
        return replace(self, state=state, measurements=measurements, kalman=kalman, identity_ambiguity_tainted=self.identity_ambiguity_tainted or association_ambiguous)

    def miss(self, timestamp_sec: float, *, kalman_config: KalmanConfig, lifecycle_config: TrackLifecycleConfig) -> "Tracklet":
        if self.state not in {TrackState.TENTATIVE, TrackState.CONFIRMED, TrackState.LOST}:
            return self
        elapsed = timestamp_sec - self.last_measurement.timestamp_sec
        if elapsed < 0.0:
            raise ValueError("Track clock cannot run backwards")
        if elapsed > lifecycle_config.maximum_track_gap_sec:
            duration = self.last_measurement.timestamp_sec - self.first_measurement.timestamp_sec
            is_confirmed = (
                len(self.measurements) >= lifecycle_config.minimum_track_hits
                and duration >= lifecycle_config.minimum_track_duration_sec
            )
            state = TrackState.FINALIZABLE if is_confirmed else TrackState.REJECTED
            return replace(self, state=state)
        kalman = predict(self.kalman, timestamp_sec, kalman_config).state if self.kalman is not None else None
        return replace(self, state=TrackState.LOST, kalman=kalman)
