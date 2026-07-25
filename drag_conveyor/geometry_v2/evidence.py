from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .observation_builder import PaddleObservation


class EvidenceType(StrEnum):
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    LEFT_PRESENT = "left_present"
    RIGHT_PRESENT = "right_present"
    ANGLE = "angle"


@dataclass(frozen=True, slots=True)
class EvidenceConfig:
    minimum_spacing_frames: int = 2
    minimum_spacing_sec: float = 0.05
    top_k_per_type: int = 8


@dataclass(frozen=True, slots=True)
class EvidenceSample:
    evidence_type: EvidenceType
    source_frame_id: int
    source_timestamp_sec: float
    observation_id: str
    support_score: float
    is_original: bool = True


@dataclass(frozen=True, slots=True)
class EventEvidenceSummary:
    connected_bins: int
    disconnected_bins: int
    left_present_bins: int
    right_present_bins: int
    source_frame_ids: tuple[int, ...]


class BoundedEvidenceStore:
    """Per-event bounded evidence with one original-frame vote per evidence type."""

    def __init__(self, config: EvidenceConfig) -> None:
        if config.minimum_spacing_frames < 1 or config.minimum_spacing_sec < 0 or config.top_k_per_type < 1:
            raise ValueError("Invalid evidence configuration")
        self._config = config
        self._samples: dict[EvidenceType, tuple[EvidenceSample, ...]] = {}

    def add_observation(self, observation: PaddleObservation, *, is_original: bool = True) -> None:
        if not is_original:
            return
        if observation.kind.value == "connected_whole":
            self.add(EvidenceSample(EvidenceType.CONNECTED, observation.source_frame_id, observation.source_timestamp_sec, observation.observation_id, 1.0))
        elif observation.kind.value == "disconnected_both":
            self.add(EvidenceSample(EvidenceType.DISCONNECTED, observation.source_frame_id, observation.source_timestamp_sec, observation.observation_id, 1.0))
        if observation.has_left:
            self.add(EvidenceSample(EvidenceType.LEFT_PRESENT, observation.source_frame_id, observation.source_timestamp_sec, observation.observation_id, 1.0))
        if observation.has_right:
            self.add(EvidenceSample(EvidenceType.RIGHT_PRESENT, observation.source_frame_id, observation.source_timestamp_sec, observation.observation_id, 1.0))

    def add(self, sample: EvidenceSample) -> None:
        if not sample.is_original:
            return
        existing = self._samples.get(sample.evidence_type, ())
        if any(item.source_frame_id == sample.source_frame_id for item in existing):
            return
        if any(abs(item.source_frame_id - sample.source_frame_id) < self._config.minimum_spacing_frames or abs(item.source_timestamp_sec - sample.source_timestamp_sec) < self._config.minimum_spacing_sec for item in existing):
            return
        ranked = tuple(sorted((*existing, sample), key=lambda item: (-item.support_score, item.source_frame_id, item.observation_id))[: self._config.top_k_per_type])
        self._samples[sample.evidence_type] = ranked

    def samples(self, evidence_type: EvidenceType) -> tuple[EvidenceSample, ...]:
        return self._samples.get(evidence_type, ())

    def count(self, evidence_type: EvidenceType) -> int:
        return len(self.samples(evidence_type))

    def all_samples(self) -> tuple[EvidenceSample, ...]:
        return tuple(sample for kind in EvidenceType for sample in self.samples(kind))


def summarize_event_observations(
    observations: tuple[PaddleObservation, ...], *, config: EvidenceConfig
) -> EventEvidenceSummary:
    """Build the bounded, original-frame-only evidence view for one event."""
    store = BoundedEvidenceStore(config)
    for observation in observations:
        store.add_observation(observation)
    return EventEvidenceSummary(
        connected_bins=store.count(EvidenceType.CONNECTED),
        disconnected_bins=store.count(EvidenceType.DISCONNECTED),
        left_present_bins=store.count(EvidenceType.LEFT_PRESENT),
        right_present_bins=store.count(EvidenceType.RIGHT_PRESENT),
        source_frame_ids=tuple(sorted({sample.source_frame_id for sample in store.all_samples()})),
    )
