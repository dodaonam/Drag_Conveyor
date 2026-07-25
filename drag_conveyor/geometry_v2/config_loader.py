from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .config import GeometryConfigError, algorithm_config_hash, canonical_algorithm_projection


GEOMETRY_CONFIG_SCHEMA_VERSION = "geometry_v2_config/2.0"


_REQUIRED_KEYS: dict[str, set[str]] = {
    "model": {"artifact_path", "expected_sha256", "expected_input_size", "fail_on_hash_mismatch", "artifact_manifest_version"},
    "video": {"decoder_backend", "timestamp_epsilon_sec", "allow_cfr_index_fallback", "require_cfr_confirmation_for_fallback"},
    "deployment": {"capability_record_version", "outside_domain_policy"},
    "geometry": {"minimum_roi_width_px", "minimum_roi_height_px", "minimum_centerline_span_ratio", "maximum_allowed_roll_deg", "minimum_side_field_of_view_ratio", "chain_band_width_ratio_min", "chain_band_width_ratio_max", "default_chain_band_width_ratio", "motion_direction"},
    "components": {"connectivity", "minimum_absolute_area_px", "minimum_roi_area_ratio", "minimum_instance_area_ratio", "anchor_nearest_pixel_ratio", "minimum_anchor_pixels", "anchor_histogram_bin_height_ratio", "anchor_histogram_window_bins", "maximum_anchor_spread_ratio", "secondary_anchor_peak_ratio", "boundary_margin_ratio", "topology_morphology", "geometry_fill_small_holes"},
    "deduplication": {"minimum_overlap_over_smaller", "minimum_iou", "anchor_gate_ratio"},
    "observations": {"same_frame_anchor_gate_ratio", "multi_anchor_separation_ratio", "pairing_ambiguity_margin", "pairing_uncertainty_weight", "unmatched_cost"},
    "tracking": {"minimum_track_hits", "minimum_track_duration_sec", "maximum_track_gap_sec", "maximum_nis", "maximum_absolute_innovation_ratio", "maximum_reverse_ratio", "sigma_acceleration_ratio_per_sec2", "minimum_measurement_sigma_px", "minimum_velocity_sigma_ratio_per_sec", "miss_track_cost", "new_track_cost", "association_ambiguity_margin", "cost_weights", "seed_cost_weights"},
    "trigger": {"center_ratio", "height_ratio", "preferred_evidence_window_half_height_ratio", "minimum_velocity_ratio_per_sec", "maximum_crossing_extrapolation_sec", "maximum_crossing_sigma_sec"},
    "evidence": {"minimum_spacing_frames", "minimum_spacing_sec", "top_k_per_type", "maximum_metadata_observations_per_track", "minimum_left_presence_bins", "minimum_right_presence_bins", "minimum_joint_two_side_opportunity_bins"},
    "fusion": {"maximum_crossing_delta_sec", "maximum_crossing_interval_ratio", "minimum_uncertainty_gate_sec", "uncertainty_sigma_multiplier", "maximum_relative_velocity_delta", "velocity_epsilon_ratio_per_sec", "maximum_trajectory_residual_ratio", "maximum_fusion_extrapolation_sec", "ambiguity_margin", "maximum_identity_conflict_tracklets", "maximum_identity_hypotheses_per_group", "minimum_unambiguous_events_for_interval", "expected_paddle_interval_sec", "expected_paddle_interval_tolerance_ratio"},
    "center_topology": {"q_bins", "minimum_q_coverage", "minimum_cross_section_thickness_ratio", "corridor_half_thickness_multiplier", "corridor_minimum_half_height_ratio", "minimum_plausible_side_thickness_roi_width_ratio", "maximum_plausible_side_thickness_roi_width_ratio", "inner_extent_chain_band_multiplier", "inner_extent_available_side_ratio", "minimum_connected_bins", "minimum_disconnected_same_frame_bins"},
    "side_integrity": {"exclusion_margin_roi_width_ratio", "minimum_side_pixels", "minimum_projected_span_px", "minimum_projected_span_roi_width_ratio", "side_outlier_residual_thickness_ratio", "maximum_side_fit_iterations", "minimum_linearity_ratio", "maximum_median_residual_thickness_ratio", "coverage_bins", "minimum_coverage_pixels_per_bin", "minimum_coverage_thickness_ratio", "minimum_intrinsic_profile_bins", "valid_minimum_coverage_ratio", "valid_maximum_internal_gap_ratio", "broken_minimum_internal_gap_ratio", "broken_maximum_length_ratio", "minimum_evidence_bins", "minimum_support_ratio", "minimum_reference_other_paddles", "reference_percentile", "reference_anchor_gate_ratio", "minimum_fov_margin_ratio", "expected_left_extent_px_at_trigger", "expected_right_extent_px_at_trigger", "expected_extent_tolerance_ratio"},
    "angle": {"window_half_height_ratio", "minimum_frames", "maximum_mad_deg", "side_threshold_deg", "global_tilt_threshold_deg", "center_kink_threshold_deg", "sign_deadband_deg", "decision_guard_deg", "axis_orientation_epsilon", "minimum_outer_endpoint_separation_px", "minimum_outer_endpoint_separation_roi_width_ratio"},
    "decision": {"minimum_reportable_events", "single_side_only_policy", "both_sides_broken_policy", "vlm_policy"},
    "snapshots": {"strict_write", "save_debug_artifacts", "jpeg_quality"},
    "limits": {"maximum_roi_pixels", "maximum_instances_for_mask_reconstruction_per_frame", "maximum_active_tracks", "maximum_components_per_frame", "maximum_events_per_job", "maximum_transient_mask_bytes", "maximum_in_memory_evidence_bytes", "maximum_spooled_evidence_bytes", "maximum_spooled_evidence_bytes_per_event", "binary_mask_encoding"},
    "determinism": {"cost_quantization", "single_thread_geometry_reductions"},
    "tracking.cost_weights": {"nis", "anchor", "type"},
    "tracking.seed_cost_weights": {"anchor", "type"},
}

_ROOT_KEYS = {"schema_version", "rule_version", *[key for key in _REQUIRED_KEYS if "." not in key]}


def load_geometry_config(path: str | Path) -> tuple[dict[str, Any], str]:
    """Load a V2 config file without silently falling back to legacy settings."""
    config_path = Path(path)
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise GeometryConfigError(f"Cannot read geometry config: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise GeometryConfigError("Geometry config is not valid JSON") from exc
    if not isinstance(raw, dict):
        raise GeometryConfigError("Geometry config root must be an object")
    if raw.get("schema_version") != GEOMETRY_CONFIG_SCHEMA_VERSION:
        raise GeometryConfigError("Unsupported geometry config schema version")
    if not isinstance(raw.get("rule_version"), str) or not raw["rule_version"]:
        raise GeometryConfigError("Geometry config requires a nonempty rule_version")
    _validate_shape(raw)
    canonical_algorithm_projection(raw)
    return raw, algorithm_config_hash(raw)


def _validate_shape(raw: dict[str, Any]) -> None:
    _validate_exact_keys(raw, _ROOT_KEYS, "$")
    for section, keys in _REQUIRED_KEYS.items():
        current: Any = raw
        for part in section.split("."):
            if not isinstance(current, dict) or part not in current:
                raise GeometryConfigError(f"Geometry config requires section: {section}")
            current = current[part]
        if not isinstance(current, dict):
            raise GeometryConfigError(f"Geometry config section must be an object: {section}")
        _validate_exact_keys(current, keys, section)
    _validate_semantics(raw)


def _validate_exact_keys(value: dict[str, Any], expected: set[str], path: str) -> None:
    unknown = sorted(set(value) - expected)
    missing = sorted(expected - set(value))
    if unknown:
        raise GeometryConfigError(f"Unknown geometry config key(s) at {path}: {', '.join(unknown)}")
    if missing:
        raise GeometryConfigError(f"Missing geometry config key(s) at {path}: {', '.join(missing)}")


def _validate_semantics(raw: dict[str, Any]) -> None:
    model = raw["model"]
    video = raw["video"]
    geometry = raw["geometry"]
    components = raw["components"]
    tracking = raw["tracking"]
    center = raw["center_topology"]
    side = raw["side_integrity"]
    angle = raw["angle"]
    decision = raw["decision"]
    for path, value in _numbers(raw):
        if not math.isfinite(float(value)):
            raise GeometryConfigError(f"Geometry config value {path} must be finite")
    if not isinstance(model["artifact_path"], str) or not model["artifact_path"] or not isinstance(model["expected_sha256"], str) or len(model["expected_sha256"]) != 64:
        raise GeometryConfigError("Geometry model artifact path and SHA-256 are required")
    if isinstance(model["expected_input_size"], bool) or not isinstance(model["expected_input_size"], int) or model["expected_input_size"] <= 0:
        raise GeometryConfigError("Geometry model input size must be a positive integer")
    if video["decoder_backend"] != "pyav" or video["timestamp_epsilon_sec"] <= 0:
        raise GeometryConfigError("Geometry V2 requires PyAV and a positive timestamp epsilon")
    if components["connectivity"] != 8 or components["topology_morphology"] != "none" or components["geometry_fill_small_holes"] is not False:
        raise GeometryConfigError("Geometry V2 topology must use raw 8-connected masks without morphology")
    if not 0.02 <= geometry["chain_band_width_ratio_min"] <= geometry["default_chain_band_width_ratio"] <= geometry["chain_band_width_ratio_max"] <= 0.20:
        raise GeometryConfigError("Geometry chain-band ratios are invalid")
    if geometry["motion_direction"] != "positive_s" or decision["single_side_only_policy"] != "uncertain" or decision["both_sides_broken_policy"] != "uncertain" or decision["vlm_policy"] != "disabled":
        raise GeometryConfigError("Geometry V2 safety policies are invalid")
    if not 0 < side["valid_maximum_internal_gap_ratio"] < side["broken_minimum_internal_gap_ratio"] < 1:
        raise GeometryConfigError("Geometry side-gap thresholds are invalid")
    if not 0 < side["broken_maximum_length_ratio"] < 1 - side["expected_extent_tolerance_ratio"] <= 1:
        raise GeometryConfigError("Geometry side-length thresholds are invalid")
    if not 0 < center["minimum_plausible_side_thickness_roi_width_ratio"] < center["maximum_plausible_side_thickness_roi_width_ratio"]:
        raise GeometryConfigError("Geometry center thickness thresholds are invalid")
    if not 0 <= angle["decision_guard_deg"] < min(angle["side_threshold_deg"], angle["global_tilt_threshold_deg"], angle["center_kink_threshold_deg"]):
        raise GeometryConfigError("Geometry angle decision guard is invalid")
    if tracking["maximum_track_gap_sec"] <= 0 or abs(sum(tracking["cost_weights"].values()) - 1.0) > 1e-9 or abs(sum(tracking["seed_cost_weights"].values()) - 1.0) > 1e-9:
        raise GeometryConfigError("Geometry tracking configuration is invalid")


def _numbers(value: Any, path: str = "$"):
    if isinstance(value, dict):
        for key, nested in value.items():
            yield from _numbers(nested, f"{path}.{key}")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield path, value
