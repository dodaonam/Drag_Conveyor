from __future__ import annotations

from dataclasses import dataclass
import math

from .observation_builder import PaddleObservation
from .tracking import KalmanConfig, predict
from .tracklets import TrackLifecycleConfig, TrackMeasurement, TrackState, Tracklet


@dataclass(frozen=True, slots=True)
class OnlineTrackingConfig:
    chain_span_px: float
    maximum_absolute_innovation_ratio: float = 0.08
    miss_track_cost: float = 0.65
    new_track_cost: float = 0.65
    cost_quantization: float = 0.000000001


@dataclass(frozen=True, slots=True)
class OnlineTrackingResult:
    active_tracklets: tuple[Tracklet, ...]
    finalizable_tracklets: tuple[Tracklet, ...]
    observation_track_ids: dict[str, int]


class OnlineTrackManager:
    """Bounded, deterministic sequence association for original-frame observations."""

    def __init__(
        self,
        *,
        config: OnlineTrackingConfig,
        kalman_config: KalmanConfig,
        lifecycle_config: TrackLifecycleConfig,
    ) -> None:
        if config.chain_span_px <= 0 or config.cost_quantization <= 0:
            raise ValueError("Tracking geometry and quantization must be positive")
        self._config = config
        self._kalman_config = kalman_config
        self._lifecycle_config = lifecycle_config
        self._active: tuple[Tracklet, ...] = ()
        self._finalizable: tuple[Tracklet, ...] = ()
        self._next_track_id = 1

    def update(self, observations: tuple[PaddleObservation, ...], *, timestamp_sec: float) -> OnlineTrackingResult:
        ordered_tracks = tuple(sorted(self._active, key=lambda track: (_expected_anchor(track, timestamp_sec, self._kalman_config), track.track_id)))
        ordered_observations = tuple(sorted(observations, key=lambda observation: (observation.s_anchor, observation.observation_id)))
        operations = _associate(ordered_tracks, ordered_observations, timestamp_sec, self._config, self._kalman_config)
        remaining: list[Tracklet] = []
        finalizable = list(self._finalizable)
        assignments: dict[str, int] = {}
        matched_observations: set[int] = set()

        for track_index, observation_index in operations:
            if track_index is None:
                continue
            track = ordered_tracks[track_index]
            if observation_index is None:
                missed = track.miss(timestamp_sec, kalman_config=self._kalman_config, lifecycle_config=self._lifecycle_config)
                if missed.state == TrackState.FINALIZABLE:
                    finalizable.append(missed)
                elif missed.state != TrackState.REJECTED:
                    remaining.append(missed)
                continue
            observation = ordered_observations[observation_index]
            measurement = TrackMeasurement(observation.observation_id, observation.source_frame_id, observation.source_timestamp_sec, observation.s_anchor, observation.s_anchor_sigma)
            try:
                updated = track.append_measurement(
                    measurement,
                    chain_span_px=self._config.chain_span_px,
                    kalman_config=self._kalman_config,
                    lifecycle_config=self._lifecycle_config,
                )
            except ValueError:
                # A gated association must never silently force an identity merge.
                missed = track.miss(timestamp_sec, kalman_config=self._kalman_config, lifecycle_config=self._lifecycle_config)
                if missed.state == TrackState.FINALIZABLE:
                    finalizable.append(missed)
                elif missed.state != TrackState.REJECTED:
                    remaining.append(missed)
                continue
            remaining.append(updated)
            assignments[observation.observation_id] = track.track_id
            matched_observations.add(observation_index)

        for observation_index, observation in enumerate(ordered_observations):
            if observation_index in matched_observations:
                continue
            track = Tracklet.seed(self._next_track_id, TrackMeasurement(observation.observation_id, observation.source_frame_id, observation.source_timestamp_sec, observation.s_anchor, observation.s_anchor_sigma))
            self._next_track_id += 1
            remaining.append(track)
            assignments[observation.observation_id] = track.track_id

        self._active = tuple(sorted(remaining, key=lambda track: track.track_id))
        self._finalizable = tuple(sorted(finalizable, key=lambda track: track.track_id))
        return OnlineTrackingResult(self._active, self._finalizable, assignments)

    def finish(self, *, timestamp_sec: float) -> tuple[Tracklet, ...]:
        finalized = list(self._finalizable)
        for track in self._active:
            state = track.miss(timestamp_sec + self._lifecycle_config.maximum_track_gap_sec + 1e-12, kalman_config=self._kalman_config, lifecycle_config=self._lifecycle_config)
            if state.state == TrackState.FINALIZABLE:
                finalized.append(state)
        self._active = ()
        self._finalizable = tuple(sorted(finalized, key=lambda track: track.track_id))
        return self._finalizable


def _associate(
    tracks: tuple[Tracklet, ...],
    observations: tuple[PaddleObservation, ...],
    timestamp_sec: float,
    config: OnlineTrackingConfig,
    kalman_config: KalmanConfig,
) -> tuple[tuple[int | None, int | None], ...]:
    """Order-preserving dynamic programming; each track/observation is used once."""
    rows, cols = len(tracks), len(observations)
    grid: list[list[tuple[int, tuple[tuple[int | None, int | None], ...]] | None]] = [[None] * (cols + 1) for _ in range(rows + 1)]
    grid[0][0] = (0, ())
    for row in range(rows + 1):
        for col in range(cols + 1):
            current = grid[row][col]
            if current is None:
                continue
            cost, path = current
            if row < rows:
                _offer(grid, row + 1, col, cost + _quantize(config.miss_track_cost, config), (*path, (row, None)))
            if col < cols:
                _offer(grid, row, col + 1, cost + _quantize(config.new_track_cost, config), (*path, (None, col)))
            if row < rows and col < cols:
                match_cost = _match_cost(tracks[row], observations[col], timestamp_sec, config, kalman_config)
                if match_cost is not None:
                    _offer(grid, row + 1, col + 1, cost + _quantize(match_cost, config), (*path, (row, col)))
    result = grid[rows][cols]
    return result[1] if result is not None else ()


def _offer(grid, row: int, col: int, cost: int, path: tuple[tuple[int | None, int | None], ...]) -> None:
    candidate = (cost, path)
    current = grid[row][col]
    if current is None or _path_rank(candidate) < _path_rank(current):
        grid[row][col] = candidate


def _path_rank(candidate: tuple[int, tuple[tuple[int | None, int | None], ...]]) -> tuple[int, tuple[tuple[int, int], ...]]:
    """Never compare ``None`` to an index when deterministic DP costs tie."""
    cost, path = candidate
    return cost, tuple((track if track is not None else -1, observation if observation is not None else -1) for track, observation in path)


def _match_cost(track: Tracklet, observation: PaddleObservation, timestamp_sec: float, config: OnlineTrackingConfig, kalman_config: KalmanConfig) -> float | None:
    expected = _expected_anchor(track, timestamp_sec, kalman_config)
    innovation = abs(observation.s_anchor - expected)
    if innovation > config.maximum_absolute_innovation_ratio * config.chain_span_px:
        return None
    return innovation / config.chain_span_px


def _expected_anchor(track: Tracklet, timestamp_sec: float, kalman_config: KalmanConfig) -> float:
    if track.kalman is None:
        return track.last_measurement.s_anchor
    state = predict(track.kalman, timestamp_sec, kalman_config).state
    value = float(state.mean[0])
    if not math.isfinite(value):
        raise ValueError("Track prediction is not finite")
    return value


def _quantize(cost: float, config: OnlineTrackingConfig) -> int:
    return round(cost / config.cost_quantization)
