from __future__ import annotations

from dataclasses import asdict, is_dataclass
import platform
import sys
from typing import Any, Mapping

import cv2
import numpy as np

from .artifacts import canonical_record_hash


ADAPTER_VERSION = "current_yolo_seg_adapter/2.0"


def build_candidate_capability_record(
    *,
    artifact_manifest: Mapping[str, Any],
    model: Any,
    algorithm_config_hash: str,
    rule_version: str,
) -> dict[str, Any]:
    """Produce an immutable, explicitly unvalidated system capability record.

    This records the exact runtime candidate without asserting accuracy or enabling
    production capability flags. A validated deployment may replace this record
    only after acceptance-gate evidence exists.
    """
    preprocess = _as_plain(getattr(model, "preprocess"))
    postprocess = _as_plain(getattr(model, "postprocess"))
    system_signature = {
        "artifact_manifest_id": artifact_manifest["artifact_manifest_id"],
        "preprocess_fingerprint": canonical_record_hash({"value": preprocess}, omit="unused"),
        "postprocess_fingerprint": canonical_record_hash({"value": postprocess}, omit="unused"),
        "adapter_version": ADAPTER_VERSION,
        "geometry_rule_version": rule_version,
        "algorithm_config_hash": algorithm_config_hash,
    }
    runtime = _runtime_record(getattr(model, "providers", ()))
    record: dict[str, Any] = {
        "schema_version": "geometry_capabilities/1.0",
        "system_signature_hash": canonical_record_hash({"value": system_signature}, omit="unused"),
        "system_signature": system_signature,
        "runtime": runtime,
        "binding_state": "unvalidated",
        "validation": {
            "same_frame_center_topology": "provisional",
            "temporal_complementary_emission": "provisional",
            "side_geometry_validity": "provisional",
            "localized_side_break": "provisional",
            "angle_classification": "provisional",
            "single_side_localization": "unvalidated",
            "absence_as_negative_evidence": "unvalidated",
        },
        "production_enabled": {
            "same_frame_center_topology": False,
            "temporal_center_break": False,
            "side_geometry_validity": False,
            "localized_side_break": False,
            "angle_classification": False,
            "single_side_localization": False,
            "absence_as_negative_evidence": False,
        },
    }
    record["capability_record_hash"] = canonical_record_hash(record, omit="capability_record_hash")
    return record


def _runtime_record(providers: Any) -> dict[str, Any]:
    try:
        import onnxruntime

        onnxruntime_version = onnxruntime.__version__
    except ImportError:  # pragma: no cover - runtime dependency is required by pipeline
        onnxruntime_version = "unavailable"
    runtime = {
        "python": platform.python_version(),
        "onnxruntime": onnxruntime_version,
        "execution_provider_order": list(providers),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "os": platform.platform(),
        "architecture": platform.machine(),
    }
    runtime["fingerprint"] = canonical_record_hash(runtime, omit="fingerprint")
    return runtime


def _as_plain(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return {str(key): _as_plain(nested) for key, nested in value.items()}
    if isinstance(value, (tuple, list)):
        return [_as_plain(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
