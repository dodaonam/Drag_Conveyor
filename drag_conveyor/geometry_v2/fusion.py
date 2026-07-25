from __future__ import annotations

from dataclasses import dataclass

from .tracklets import Tracklet


@dataclass(frozen=True, slots=True)
class FusionConfig:
    trigger_center_s: float
    maximum_crossing_delta_sec: float = 0.12
    maximum_relative_velocity_delta: float = 0.25


@dataclass(frozen=True, slots=True)
class PhysicalEvent:
    paddle_id: int
    track_ids: tuple[int, ...]
    crossing_timestamp_sec: float
    identity_ambiguous: bool = False


def fuse_tracklets(tracklets: tuple[Tracklet, ...], *, config: FusionConfig) -> tuple[PhysicalEvent, ...]:
    """Merge only uniquely compatible tracklets, then issue deterministic paddle IDs."""
    candidates = []
    for track in tracklets:
        try:
            candidates.append(_candidate(track, config.trigger_center_s))
        except ValueError:
            # A rejected/noisy track must not suppress otherwise valid paddles.
            continue
    candidates.sort(key=lambda item: (item.crossing_timestamp_sec, item.track.track_id))
    groups: list[list[_TrackCandidate]] = []
    for candidate in candidates:
        compatible = [index for index, group in enumerate(groups) if _compatible(candidate, group[0], config)]
        if len(compatible) == 1:
            groups[compatible[0]].append(candidate)
        else:
            groups.append([candidate])
    ordered = sorted(groups, key=lambda group: (min(item.crossing_timestamp_sec for item in group), min(item.track.track_id for item in group)))
    return tuple(
        PhysicalEvent(
            paddle_id=index,
            track_ids=tuple(sorted(item.track.track_id for item in group)),
            crossing_timestamp_sec=sum(item.crossing_timestamp_sec for item in group) / len(group),
            identity_ambiguous=len([other for other in groups if other is not group and _compatible(group[0], other[0], config)]) > 0,
        )
        for index, group in enumerate(ordered, start=1)
    )


@dataclass(frozen=True, slots=True)
class _TrackCandidate:
    track: Tracklet
    crossing_timestamp_sec: float
    velocity: float


def _candidate(track: Tracklet, trigger_center_s: float) -> _TrackCandidate:
    measurements = track.measurements
    if len(measurements) < 2:
        raise ValueError("Fusion requires confirmed tracklets")
    first, last = measurements[0], measurements[-1]
    velocity = (last.s_anchor - first.s_anchor) / (last.timestamp_sec - first.timestamp_sec)
    if velocity <= 0:
        raise ValueError("Fusion requires positive longitudinal motion")
    for previous, current in zip(measurements, measurements[1:]):
        if previous.s_anchor <= trigger_center_s <= current.s_anchor:
            ratio = (trigger_center_s - previous.s_anchor) / (current.s_anchor - previous.s_anchor)
            crossing = previous.timestamp_sec + ratio * (current.timestamp_sec - previous.timestamp_sec)
            return _TrackCandidate(track, crossing, velocity)
    return _TrackCandidate(track, first.timestamp_sec + (trigger_center_s - first.s_anchor) / velocity, velocity)


def _compatible(first: _TrackCandidate, second: _TrackCandidate, config: FusionConfig) -> bool:
    if abs(first.crossing_timestamp_sec - second.crossing_timestamp_sec) > config.maximum_crossing_delta_sec:
        return False
    relative = abs(first.velocity - second.velocity) / max(abs(first.velocity), abs(second.velocity))
    return relative <= config.maximum_relative_velocity_delta
