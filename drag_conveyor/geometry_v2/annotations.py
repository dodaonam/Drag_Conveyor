from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .acceptance import GroundTruthEvent
from .decision import FinalStatus


ANNOTATION_SCHEMA_VERSION = "geometry_v2_ground_truth/1.0"
_PHYSICAL_STATUSES = frozenset(FinalStatus) - {FinalStatus.UNCERTAIN}


@dataclass(frozen=True, slots=True)
class AnnotationEvent:
    video_id: str
    physical_paddle_id: int
    entry_frame: int
    entry_timestamp_sec: float
    trigger_crossing_frame: int
    trigger_crossing_timestamp_sec: float
    exit_frame: int
    exit_timestamp_sec: float
    status: FinalStatus
    visible_left: bool
    visible_right: bool
    center_visible: bool
    annotator_ids: tuple[str, ...]
    adjudicated: bool
    partial_boundary: bool

    def as_ground_truth(self) -> GroundTruthEvent:
        return GroundTruthEvent(
            event_id=f"{self.video_id}:{self.physical_paddle_id}",
            crossing_timestamp_sec=self.trigger_crossing_timestamp_sec,
            status=self.status,
            partial_boundary=self.partial_boundary,
        )


@dataclass(frozen=True, slots=True)
class AnnotationDataset:
    dataset_version: str
    frame_pts_table_sha256: str
    events: tuple[AnnotationEvent, ...]


def load_annotation_dataset(path: str | Path) -> AnnotationDataset:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Cannot read a valid geometry V2 annotation dataset") from exc
    if not isinstance(raw, Mapping) or raw.get("schema_version") != ANNOTATION_SCHEMA_VERSION:
        raise ValueError("Unsupported geometry V2 annotation schema")
    dataset_version = _nonempty_string(raw.get("dataset_version"), "dataset_version")
    pts_hash = _sha256(raw.get("frame_pts_table_sha256"), "frame_pts_table_sha256")
    raw_events = raw.get("events")
    if not isinstance(raw_events, list) or not raw_events:
        raise ValueError("Annotation dataset requires a nonempty events list")
    events = tuple(_parse_event(item) for item in raw_events)
    ids = [(event.video_id, event.physical_paddle_id) for event in events]
    if len(ids) != len(set(ids)):
        raise ValueError("Annotation events must have unique (video_id, physical_paddle_id)")
    return AnnotationDataset(dataset_version, pts_hash, events)


def _parse_event(raw: Any) -> AnnotationEvent:
    if not isinstance(raw, Mapping):
        raise ValueError("Each annotation event must be an object")
    status = _status(raw.get("status"))
    annotators = raw.get("annotator_ids")
    if not isinstance(annotators, list) or not all(isinstance(item, str) and item for item in annotators):
        raise ValueError("annotator_ids must be a nonempty string list")
    adjudicated = _bool(raw.get("adjudicated"), "adjudicated")
    if status != FinalStatus.NORMAL and (len(set(annotators)) < 2 or not adjudicated):
        raise ValueError("Non-normal physical labels require two annotators and adjudication")
    entry_frame = _positive_int(raw.get("entry_frame"), "entry_frame")
    crossing_frame = _positive_int(raw.get("trigger_crossing_frame"), "trigger_crossing_frame")
    exit_frame = _positive_int(raw.get("exit_frame"), "exit_frame")
    entry_time = _finite(raw.get("entry_timestamp_sec"), "entry_timestamp_sec")
    crossing_time = _finite(raw.get("trigger_crossing_timestamp_sec"), "trigger_crossing_timestamp_sec")
    exit_time = _finite(raw.get("exit_timestamp_sec"), "exit_timestamp_sec")
    if not (entry_frame <= crossing_frame <= exit_frame and entry_time <= crossing_time <= exit_time):
        raise ValueError("Annotation event entry/crossing/exit order is invalid")
    return AnnotationEvent(
        video_id=_nonempty_string(raw.get("video_id"), "video_id"),
        physical_paddle_id=_positive_int(raw.get("physical_paddle_id"), "physical_paddle_id"),
        entry_frame=entry_frame,
        entry_timestamp_sec=entry_time,
        trigger_crossing_frame=crossing_frame,
        trigger_crossing_timestamp_sec=crossing_time,
        exit_frame=exit_frame,
        exit_timestamp_sec=exit_time,
        status=status,
        visible_left=_bool(raw.get("visible_left"), "visible_left"),
        visible_right=_bool(raw.get("visible_right"), "visible_right"),
        center_visible=_bool(raw.get("center_visible"), "center_visible"),
        annotator_ids=tuple(annotators),
        adjudicated=adjudicated,
        partial_boundary=_bool(raw.get("partial_boundary", False), "partial_boundary"),
    )


def _status(value: Any) -> FinalStatus:
    try:
        status = FinalStatus(value)
    except ValueError as exc:
        raise ValueError("Annotation status is not a canonical final status") from exc
    if status not in _PHYSICAL_STATUSES:
        raise ValueError("Ground truth cannot use uncertain")
    return status


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value.lower()):
        raise ValueError(f"{name} must be a SHA-256 hex string")
    return value.lower()


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be boolean")
    return value
