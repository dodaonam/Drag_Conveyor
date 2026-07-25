from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class ReplayComparison:
    identical: bool
    first_hash: str
    second_hash: str
    differing_paths: tuple[str, ...]


def canonical_decision_hash(summary: Mapping[str, Any]) -> str:
    """Hash stable decision data only; runtime timing and local paths are excluded."""
    canonical = _decision_projection(summary)
    encoded = json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def compare_replays(first: Mapping[str, Any], second: Mapping[str, Any]) -> ReplayComparison:
    first_projection = _decision_projection(first)
    second_projection = _decision_projection(second)
    return ReplayComparison(
        identical=first_projection == second_projection,
        first_hash=canonical_decision_hash(first),
        second_hash=canonical_decision_hash(second),
        differing_paths=tuple(_differing_paths(first_projection, second_projection)),
    )


def _decision_projection(summary: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {
        "inspection_mode", "paddle_schema_version", "summary_schema_version", "rule_version",
        "model_metadata", "geometry_metadata", "capability_metadata", "timestamp_source",
        "count_certified", "possible_event_count_min", "possible_event_count_max", "failure_reason",
        "total_bars", "normal_bars", "defect_bars", "status_counts", "defects", "normals",
    }
    return _normalize({key: summary[key] for key in sorted(allowed & set(summary))})


def _normalize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _normalize(nested) for key, nested in sorted(value.items(), key=lambda item: str(item[0])) if key not in {"latency_ms", "snapshot_url", "snapshot_key", "filename"}}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Replay payload contains non-finite number")
        return value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise ValueError(f"Replay payload contains unsupported type: {type(value).__name__}")


def _differing_paths(first: Any, second: Any, path: str = "$"):
    if type(first) is not type(second):
        yield path
    elif isinstance(first, dict):
        for key in sorted(set(first) | set(second)):
            if key not in first or key not in second:
                yield f"{path}.{key}"
            else:
                yield from _differing_paths(first[key], second[key], f"{path}.{key}")
    elif isinstance(first, list):
        if len(first) != len(second):
            yield path
        for index, (left, right) in enumerate(zip(first, second)):
            yield from _differing_paths(left, right, f"{path}[{index}]")
    elif first != second:
        yield path
