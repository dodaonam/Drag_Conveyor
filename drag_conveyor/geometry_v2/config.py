from __future__ import annotations

import copy
import hashlib
import json
import math
from typing import Any, Mapping


class GeometryConfigError(ValueError):
    pass


def canonical_algorithm_projection(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the algorithm identity projection defined by spec section 27.3."""
    projection = copy.deepcopy(dict(config))
    model = projection.get("model")
    if isinstance(model, dict):
        model.pop("artifact_path", None)
    _validate_json_value(projection, path="$")
    return projection


def algorithm_config_hash(config: Mapping[str, Any]) -> str:
    projection = canonical_algorithm_projection(config)
    encoded = json.dumps(projection, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def resolved_geometry_values(
    config: Mapping[str, Any],
    *,
    roi_width: int,
    roi_height: int,
    chain_span_px: float,
) -> dict[str, float | int]:
    """Resolve pixel values that must be persisted with each V2 job."""
    if roi_width <= 0 or roi_height <= 0 or not math.isfinite(chain_span_px) or chain_span_px <= 0.0:
        raise GeometryConfigError("ROI and chain span must be positive")
    try:
        geometry = config["geometry"]
        components = config["components"]
        side = config["side_integrity"]
        center = config["center_topology"]
        angle = config["angle"]
    except KeyError as exc:
        raise GeometryConfigError(f"Missing required config section: {exc.args[0]}") from exc
    ratio = _number(geometry, "default_chain_band_width_ratio")
    ratio_min = _number(geometry, "chain_band_width_ratio_min")
    ratio_max = _number(geometry, "chain_band_width_ratio_max")
    if not ratio_min <= ratio <= ratio_max:
        raise GeometryConfigError("default chain band ratio is outside its configured range")
    if not 0.0 < _number(side, "valid_maximum_internal_gap_ratio") < _number(side, "broken_minimum_internal_gap_ratio") < 1.0:
        raise GeometryConfigError("side gap thresholds are invalid")
    if not 0.0 <= _number(angle, "decision_guard_deg") < min(_number(angle, "side_threshold_deg"), _number(angle, "global_tilt_threshold_deg"), _number(angle, "center_kink_threshold_deg")):
        raise GeometryConfigError("angle decision guard is invalid")
    band_width = ratio * roi_width
    return {
        "chain_span_px": chain_span_px,
        "chain_band_width_px": band_width,
        "chain_band_half_width_px": band_width / 2.0,
        "boundary_margin_px": max(2, round(_number(components, "boundary_margin_ratio") * max(roi_width, roi_height))),
        "side_exclusion_margin_px": _number(side, "exclusion_margin_roi_width_ratio") * roi_width,
        "minimum_side_projected_span_px": max(_number(side, "minimum_projected_span_px"), _number(side, "minimum_projected_span_roi_width_ratio") * roi_width),
        "minimum_plausible_side_thickness_px": _number(center, "minimum_plausible_side_thickness_roi_width_ratio") * roi_width,
        "maximum_plausible_side_thickness_px": _number(center, "maximum_plausible_side_thickness_roi_width_ratio") * roi_width,
        "minimum_outer_endpoint_separation_px": max(_number(angle, "minimum_outer_endpoint_separation_px"), _number(angle, "minimum_outer_endpoint_separation_roi_width_ratio") * roi_width),
        "angle_decision_guard_deg": _number(angle, "decision_guard_deg"),
    }


def _number(section: Any, name: str) -> float:
    try:
        value = section[name]
    except (KeyError, TypeError) as exc:
        raise GeometryConfigError(f"Missing numeric config value: {name}") from exc
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise GeometryConfigError(f"Config value {name} must be a finite number")
    return float(value)


def _validate_json_value(value: Any, *, path: str) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise GeometryConfigError(f"Config key at {path} must be a string")
            _validate_json_value(nested, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _validate_json_value(nested, path=f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise GeometryConfigError(f"Config number at {path} must be finite")
    elif value is not None and not isinstance(value, (str, int, float, bool)):
        raise GeometryConfigError(f"Config value at {path} is not JSON-compatible")
