from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..inference import OnnxRuntimeEngine, postprocess_segmentation, preprocess_roi
from .artifacts import read_onnx_artifact_manifest
from .capabilities import build_candidate_capability_record
from .config_loader import load_geometry_config
from .coordinates import ChainCoordinates, mad, quantile_type7
from .decision import CenterState, EventEvidence, FinalStatus, SideState, classify_event
from .evidence import EvidenceConfig, summarize_event_observations
from .frame_source import FrameSourceError, iter_original_frames
from .fusion import FusionConfig, PhysicalEvent, fuse_tracklets
from .observation_builder import PaddleObservation, build_frame_observations
from .observations import ComponentExtractionConfig, deduplicate_components, extract_components
from .online_tracking import OnlineTrackManager, OnlineTrackingConfig
from .pairing import PairingConfig
from .tracking import KalmanConfig
from .tracklets import TrackLifecycleConfig, TrackState
from .types import Point, Roi
from .side_angle import SideAxis, classify_angles, fit_side_axis
from .side_integrity import SideIntegrity, analyze_side_integrity
from .snapshots import geometry_snapshot_filename, render_geometry_snapshot


@dataclass(frozen=True, slots=True)
class GeometryInput:
    chain_top: Point
    chain_bottom: Point
    chain_band_width_ratio: float
    motion_direction: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, default_ratio: float) -> "GeometryInput":
        if value.get("schema_version") != "geometry_input/2.0":
            raise ValueError("geometry_input schema_version must be geometry_input/2.0")
        line = value.get("chain_centerline")
        if not isinstance(line, Mapping):
            raise ValueError("geometry_input requires chain_centerline")
        top, bottom = line.get("top"), line.get("bottom")
        if not isinstance(top, Mapping) or not isinstance(bottom, Mapping):
            raise ValueError("geometry_input chain_centerline requires top and bottom points")
        ratio = value.get("chain_band_width_ratio", default_ratio)
        if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
            raise ValueError("geometry_input chain_band_width_ratio must be numeric")
        direction = value.get("motion_direction", "positive_s")
        if direction not in {"positive_s", "negative_s"}:
            raise ValueError("geometry_input motion_direction must be positive_s or negative_s")
        return cls(Point(float(top["x"]), float(top["y"])), Point(float(bottom["x"]), float(bottom["y"])), float(ratio), direction)


@dataclass(frozen=True, slots=True)
class GeometryEventResult:
    event: PhysicalEvent
    status: FinalStatus
    reason_codes: tuple[str, ...]
    source_frame_ids: tuple[int, ...]
    snapshot_source_frame_id: int | None = None
    evidence_summary: dict[str, int] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GeometryPipelineResult:
    success: bool
    failure_reason: str
    frames_scanned: int
    events: tuple[GeometryEventResult, ...]
    model_hash: str
    algorithm_config_hash: str
    timestamp_source: str
    defect_snapshots_dir: Path | None = None
    normal_snapshots_dir: Path | None = None
    artifact_manifest: dict[str, Any] = field(default_factory=dict)
    capability_metadata: dict[str, Any] = field(default_factory=dict)


def run_geometry_v2_pipeline(
    *,
    profile: Any,
    source: str,
    geometry_input: Mapping[str, Any],
    geometry_config_path: str | Path,
    snapshots_root: str | Path | None = None,
    run_id: str | None = None,
) -> GeometryPipelineResult:
    config, config_hash = load_geometry_config(geometry_config_path)
    region = profile.region
    roi_config = region.roi
    if roi_config.w * roi_config.h > int(config["limits"]["maximum_roi_pixels"]):
        return GeometryPipelineResult(False, "geometry_resource_limit_exceeded", 0, (), "", config_hash, "decoder_pts")
    input_value = GeometryInput.from_mapping(geometry_input, default_ratio=float(config["geometry"]["default_chain_band_width_ratio"]))
    if not float(config["geometry"]["chain_band_width_ratio_min"]) <= input_value.chain_band_width_ratio <= float(config["geometry"]["chain_band_width_ratio_max"]):
        raise ValueError("geometry_invalid_chain_band")
    line_first, line_second = (input_value.chain_top, input_value.chain_bottom) if input_value.motion_direction == "positive_s" else (input_value.chain_bottom, input_value.chain_top)
    coordinates = ChainCoordinates.create(
        Roi(roi_config.x, roi_config.y, roi_config.w, roi_config.h), line_first, line_second,
        minimum_span_ratio=float(config["geometry"]["minimum_centerline_span_ratio"]), maximum_roll_deg=float(config["geometry"]["maximum_allowed_roll_deg"]),
    )
    half_band = input_value.chain_band_width_ratio * roi_config.w / 2.0
    trigger = coordinates.trigger_strip(float(config["trigger"]["center_ratio"]), float(config["trigger"]["height_ratio"]))
    geometry_model = _pinned_geometry_model(profile.model, config)
    model_path = _resolve_model_path(geometry_model.path)
    engine = OnnxRuntimeEngine(providers=geometry_model.providers)
    diagnostics = engine.load(str(model_path), geometry_model)
    if config["model"]["fail_on_hash_mismatch"] and diagnostics.model_hash != config["model"]["expected_sha256"]:
        engine.close()
        return GeometryPipelineResult(False, "model_hash_mismatch", 0, (), diagnostics.model_hash, config_hash, "decoder_pts")
    artifact_manifest = read_onnx_artifact_manifest(model_path)
    if artifact_manifest["sha256"] != diagnostics.model_hash:
        engine.close()
        return GeometryPipelineResult(False, "model_manifest_hash_mismatch", 0, (), diagnostics.model_hash, config_hash, "decoder_pts")
    capability_metadata = build_candidate_capability_record(
        artifact_manifest=artifact_manifest,
        model=geometry_model,
        algorithm_config_hash=config_hash,
        rule_version=str(config["rule_version"]),
    )
    manager = OnlineTrackManager(
        config=OnlineTrackingConfig(chain_span_px=coordinates.span, maximum_absolute_innovation_ratio=float(config["tracking"]["maximum_absolute_innovation_ratio"]), miss_track_cost=float(config["tracking"]["miss_track_cost"]), new_track_cost=float(config["tracking"]["new_track_cost"]), cost_quantization=float(config["determinism"]["cost_quantization"])),
        kalman_config=KalmanConfig(chain_span_px=coordinates.span, sigma_acceleration_ratio_per_sec2=float(config["tracking"]["sigma_acceleration_ratio_per_sec2"]), minimum_measurement_sigma_px=float(config["tracking"]["minimum_measurement_sigma_px"]), minimum_velocity_sigma_ratio_per_sec=float(config["tracking"]["minimum_velocity_sigma_ratio_per_sec"])),
        lifecycle_config=TrackLifecycleConfig(maximum_track_gap_sec=float(config["tracking"]["maximum_track_gap_sec"]), minimum_track_hits=int(config["tracking"]["minimum_track_hits"]), minimum_track_duration_sec=float(config["tracking"]["minimum_track_duration_sec"]), maximum_reverse_ratio=float(config["tracking"]["maximum_reverse_ratio"])),
    )
    observations: dict[str, PaddleObservation] = {}
    frames = 0
    last_time = 0.0
    try:
        for frame in iter_original_frames(source, timestamp_epsilon_sec=float(config["video"]["timestamp_epsilon_sec"])):
            frames += 1
            last_time = frame.source_timestamp_sec
            if frame.image_bgr.shape[:2] != (region.frame_height, region.frame_width):
                return GeometryPipelineResult(False, "video_geometry_changed", frames, (), diagnostics.model_hash, config_hash, frame.timestamp_source, artifact_manifest=artifact_manifest, capability_metadata=capability_metadata)
            roi = frame.image_bgr[roi_config.y : roi_config.y + roi_config.h, roi_config.x : roi_config.x + roi_config.w]
            prep = preprocess_roi(roi, (roi_config.x, roi_config.y), geometry_model.input_size, geometry_model.preprocess.normalize, geometry_model.preprocess.color_format, geometry_model.preprocess.padding_value)
            det_out, proto_out = engine.infer(prep.tensor)
            detections = postprocess_segmentation(det_out, proto_out, prep, geometry_model, geometry_model.postprocess)
            if len(detections) > int(config["limits"]["maximum_instances_for_mask_reconstruction_per_frame"]):
                return GeometryPipelineResult(False, "geometry_resource_limit_exceeded", frames, (), diagnostics.model_hash, config_hash, frame.timestamp_source, artifact_manifest=artifact_manifest, capability_metadata=capability_metadata)
            components = []
            for index, detection in enumerate(detections, start=1):
                if detection.model_bbox_roi_xyxy is None or detection.model_bbox_crop_roi_xyxy is None or detection.model_output_row_index is None:
                    return GeometryPipelineResult(False, "model_contract_mismatch", frames, (), diagnostics.model_hash, config_hash, frame.timestamp_source, artifact_manifest=artifact_manifest, capability_metadata=capability_metadata)
                components.extend(extract_components(detection.mask_roi, source_frame_id=frame.source_frame_id, source_detection_id=f"d{index:02d}", source_detection_score=detection.score, source_model_output_row_index=detection.model_output_row_index, class_id=detection.class_id, coordinates=coordinates, config=ComponentExtractionConfig(chain_band_half_width=half_band, boundary_margin_px=max(2, round(float(config["components"]["boundary_margin_ratio"]) * max(roi_config.w, roi_config.h))), maximum_anchor_spread_ratio=float(config["components"]["maximum_anchor_spread_ratio"]), secondary_anchor_peak_ratio=float(config["components"]["secondary_anchor_peak_ratio"]), duplicate_anchor_gate_ratio=float(config["deduplication"]["anchor_gate_ratio"]), duplicate_iou_threshold=float(config["deduplication"]["minimum_iou"]), duplicate_ios_threshold=float(config["deduplication"]["minimum_overlap_over_smaller"])), model_bbox_roi_xyxy=detection.model_bbox_roi_xyxy))
            if len(components) > int(config["limits"]["maximum_components_per_frame"]):
                return GeometryPipelineResult(False, "geometry_resource_limit_exceeded", frames, (), diagnostics.model_hash, config_hash, frame.timestamp_source, artifact_manifest=artifact_manifest, capability_metadata=capability_metadata)
            canonical = deduplicate_components(tuple(components), coordinates=coordinates, config=ComponentExtractionConfig(chain_band_half_width=half_band, boundary_margin_px=2, duplicate_anchor_gate_ratio=float(config["deduplication"]["anchor_gate_ratio"]), duplicate_iou_threshold=float(config["deduplication"]["minimum_iou"]), duplicate_ios_threshold=float(config["deduplication"]["minimum_overlap_over_smaller"])))
            frame_observations = build_frame_observations(canonical, source_timestamp_sec=frame.source_timestamp_sec, coordinates=coordinates, chain_band_half_width=half_band, pairing_config=PairingConfig(same_frame_anchor_gate_ratio=float(config["observations"]["same_frame_anchor_gate_ratio"]), pairing_uncertainty_weight=float(config["observations"]["pairing_uncertainty_weight"]), unmatched_cost=float(config["observations"]["unmatched_cost"]), pairing_ambiguity_margin=float(config["observations"]["pairing_ambiguity_margin"]), cost_quantization=float(config["determinism"]["cost_quantization"])), q_bins=int(config["center_topology"]["q_bins"]), minimum_q_coverage=float(config["center_topology"]["minimum_q_coverage"]))
            observations.update({item.observation_id: item for item in frame_observations})
            tracking_result = manager.update(frame_observations, timestamp_sec=frame.source_timestamp_sec)
            if len(tracking_result.active_tracklets) > int(config["limits"]["maximum_active_tracks"]):
                return GeometryPipelineResult(False, "geometry_resource_limit_exceeded", frames, (), diagnostics.model_hash, config_hash, frame.timestamp_source, artifact_manifest=artifact_manifest, capability_metadata=capability_metadata)
    except FrameSourceError as exc:
        failure = "invalid_video_timestamps" if "PTS" in str(exc) else "video_open_failed"
        return GeometryPipelineResult(False, failure, frames, (), diagnostics.model_hash, config_hash, "decoder_pts", artifact_manifest=artifact_manifest, capability_metadata=capability_metadata)
    finally:
        engine.close()
    tracklets = tuple(track for track in manager.finish(timestamp_sec=last_time) if track.state == TrackState.FINALIZABLE)
    try:
        events = fuse_tracklets(tracklets, config=FusionConfig(trigger_center_s=(trigger.top_s + trigger.bottom_s) / 2.0, maximum_crossing_delta_sec=float(config["fusion"]["maximum_crossing_delta_sec"]), maximum_relative_velocity_delta=float(config["fusion"]["maximum_relative_velocity_delta"])))
    except ValueError:
        events = ()
    if len(events) > int(config["limits"]["maximum_events_per_job"]):
        return GeometryPipelineResult(False, "geometry_resource_limit_exceeded", frames, (), diagnostics.model_hash, config_hash, "decoder_pts", artifact_manifest=artifact_manifest, capability_metadata=capability_metadata)
    results = tuple(_classify_placeholder_event(event, tracklets, observations, coordinates, half_band, config) for event in events)
    if any(event.identity_ambiguous for event in events):
        return GeometryPipelineResult(False, "event_cardinality_unresolved", frames, results, diagnostics.model_hash, config_hash, "decoder_pts", artifact_manifest=artifact_manifest, capability_metadata=capability_metadata)
    if not results:
        return GeometryPipelineResult(False, "no_reportable_paddles", frames, (), diagnostics.model_hash, config_hash, "decoder_pts", artifact_manifest=artifact_manifest, capability_metadata=capability_metadata)
    defect_snapshots_dir, normal_snapshots_dir = _write_event_snapshots(
        source=source,
        results=results,
        snapshots_root=snapshots_root,
        run_id=run_id,
        roi_xywh=(roi_config.x, roi_config.y, roi_config.w, roi_config.h),
        coordinates=coordinates,
        trigger=trigger,
        jpeg_quality=int(config["snapshots"]["jpeg_quality"]),
        strict=bool(config["snapshots"]["strict_write"]),
        timestamp_epsilon_sec=float(config["video"]["timestamp_epsilon_sec"]),
    )
    return GeometryPipelineResult(True, "", frames, results, diagnostics.model_hash, config_hash, "decoder_pts", defect_snapshots_dir, normal_snapshots_dir, artifact_manifest, capability_metadata)


def _classify_placeholder_event(event: PhysicalEvent, tracklets, observations: Mapping[str, PaddleObservation], coordinates: ChainCoordinates, half_band: float, config: Mapping[str, Any]) -> GeometryEventResult:
    members = [track for track in tracklets if track.track_id in event.track_ids]
    event_observations = [observations[measurement.observation_id] for track in members for measurement in track.measurements]
    threshold_deg = float(config["angle"]["side_threshold_deg"])
    source_frame_ids = tuple(sorted({item.source_frame_id for item in event_observations}))
    snapshot_frame_id = _snapshot_frame_id(event, event_observations)
    center_minimum = int(config["center_topology"]["minimum_connected_bins"])
    disconnected_minimum = int(config["center_topology"]["minimum_disconnected_same_frame_bins"])
    spacing_frames = int(config["evidence"]["minimum_spacing_frames"])
    spacing_seconds = float(config["evidence"]["minimum_spacing_sec"])
    evidence = summarize_event_observations(tuple(event_observations), config=EvidenceConfig(minimum_spacing_frames=spacing_frames, minimum_spacing_sec=spacing_seconds, top_k_per_type=int(config["evidence"].get("top_k_per_type", 8))))
    evidence_data = {"connected_bins": evidence.connected_bins, "disconnected_bins": evidence.disconnected_bins, "left_present_bins": evidence.left_present_bins, "right_present_bins": evidence.right_present_bins}
    if evidence.connected_bins >= center_minimum and evidence.disconnected_bins >= disconnected_minimum:
        center = CenterState.CONFLICT
    elif evidence.disconnected_bins >= disconnected_minimum:
        center = CenterState.BROKEN_TOPOLOGICAL
    elif evidence.connected_bins >= center_minimum:
        center = CenterState.INTACT
    else:
        center = CenterState.UNKNOWN
    side_samples = []
    side_config = config["side_integrity"]
    for observation in event_observations:
        left_mask = observation.left_mask_roi
        right_mask = observation.right_mask_roi
        if left_mask is None and observation.kind.value == "connected_whole":
            left_mask = observation.mask_roi
        if right_mask is None and observation.kind.value == "connected_whole":
            right_mask = observation.mask_roi
        if left_mask is None or right_mask is None:
            continue
        left_integrity = analyze_side_integrity(
            left_mask,
            coordinates,
            side="left",
            chain_band_half_width=half_band,
            bins=int(side_config["coverage_bins"]),
            valid_minimum_coverage_ratio=float(side_config["valid_minimum_coverage_ratio"]),
            broken_minimum_internal_gap_ratio=float(side_config["broken_minimum_internal_gap_ratio"]),
        )
        right_integrity = analyze_side_integrity(
            right_mask,
            coordinates,
            side="right",
            chain_band_half_width=half_band,
            bins=int(side_config["coverage_bins"]),
            valid_minimum_coverage_ratio=float(side_config["valid_minimum_coverage_ratio"]),
            broken_minimum_internal_gap_ratio=float(side_config["broken_minimum_internal_gap_ratio"]),
        )
        side_samples.append((observation.source_frame_id, observation.source_timestamp_sec, left_integrity, right_integrity))
    independent_sides = _independent_angle_samples(
        side_samples,
        minimum_frame_delta=spacing_frames,
        minimum_timestamp_delta=spacing_seconds,
    )
    left_states = [item[2].state for item in independent_sides]
    right_states = [item[3].state for item in independent_sides]
    evidence_data.update({
        "left_valid_side_bins": left_states.count(SideIntegrity.VALID),
        "left_broken_side_bins": left_states.count(SideIntegrity.BROKEN_LOCALIZED),
        "right_valid_side_bins": right_states.count(SideIntegrity.VALID),
        "right_broken_side_bins": right_states.count(SideIntegrity.BROKEN_LOCALIZED),
    })
    side_minimum = int(side_config["minimum_evidence_bins"])
    left_state = _side_state(evidence_data["left_valid_side_bins"], evidence_data["left_broken_side_bins"], side_minimum)
    right_state = _side_state(evidence_data["right_valid_side_bins"], evidence_data["right_broken_side_bins"], side_minimum)
    angle_samples = []
    independent = []
    angle_reason = "insufficient_angle_frames"
    angle_status = None
    angle_diagnostics: dict[str, float | None] = {
        "left_angle_deg": None,
        "right_angle_deg": None,
        "global_tilt_deg": None,
        "center_kink_deg": None,
    }
    if center == CenterState.INTACT and left_state == SideState.VALID and right_state == SideState.VALID:
        for observation in event_observations:
            if not (observation.kind.value == "connected_whole" and observation.mask_roi.any()):
                continue
            left = fit_side_axis(observation.mask_roi, coordinates, side="left", chain_band_half_width=half_band)
            right = fit_side_axis(observation.mask_roi, coordinates, side="right", chain_band_half_width=half_band)
            if left is not None and right is not None:
                angle_samples.append((observation.source_frame_id, observation.source_timestamp_sec, left, right))
        independent = _independent_angle_samples(angle_samples, minimum_frame_delta=spacing_frames, minimum_timestamp_delta=spacing_seconds)
        if len(independent) >= int(config["angle"]["minimum_frames"]):
            left_angles = [item[2].angle_deg for item in independent]
            right_angles = [item[3].angle_deg for item in independent]
            if max(mad(left_angles), mad(right_angles)) <= float(config["angle"]["maximum_mad_deg"]):
                left = SideAxis(float(np.tan(np.radians(quantile_type7(left_angles, .5)))), 0.0, 0.0, 0.0, len(independent))
                right = SideAxis(float(np.tan(np.radians(quantile_type7(right_angles, .5)))), 0.0, 0.0, 0.0, len(independent))
                angles = classify_angles(left=left, right=right, side_threshold_deg=threshold_deg, global_tilt_threshold_deg=threshold_deg, center_kink_threshold_deg=threshold_deg, decision_guard_deg=float(config["angle"]["decision_guard_deg"]))
                angle_status = angles.status
                angle_diagnostics = {
                    "left_angle_deg": angles.left_deg,
                    "right_angle_deg": angles.right_deg,
                    "global_tilt_deg": angles.global_tilt_deg,
                    "center_kink_deg": angles.center_kink_deg,
                }
                angle_reason = "angle_threshold_guard_band" if angles.status is None else ""
            else:
                angle_reason = "unstable_angle_measurement"
    decision = classify_event(EventEvidence(
        has_hard_identity_or_geometry_conflict=event.identity_ambiguous,
        primary_conflict_reason="identity_ambiguous",
        is_single_side_only=not any(item.has_left and item.has_right for item in event_observations),
        center=center,
        left=left_state,
        right=right_state,
        has_positive_break_evidence=center == CenterState.BROKEN_TOPOLOGICAL or left_state == SideState.BROKEN_LOCALIZED or right_state == SideState.BROKEN_LOCALIZED,
        has_positive_or_temporal_break_suspicion=evidence.disconnected_bins > 0,
        definitive_localized_left_break_with_both_sides_observed=left_state == SideState.BROKEN_LOCALIZED and evidence.right_present_bins >= side_minimum,
        definitive_localized_right_break_with_both_sides_observed=right_state == SideState.BROKEN_LOCALIZED and evidence.left_present_bins >= side_minimum,
        angle_enabled=True,
        angle_status=angle_status,
        definitive_support_bins=max(evidence.connected_bins, evidence.disconnected_bins, evidence_data["left_broken_side_bins"], evidence_data["right_broken_side_bins"]),
    ))
    reasons = decision.reason_codes
    if decision.primary_reason == "insufficient_angle_frames" and angle_reason != "insufficient_angle_frames":
        reasons = (angle_reason,)
    diagnostics: dict[str, Any] = {
        "center_state": center.value,
        "left_side_state": left_state.value,
        "right_side_state": right_state.value,
        "observability_grade": _observability_grade(
            identity_ambiguous=event.identity_ambiguous,
            center=center,
            left=left_state,
            right=right_state,
            evidence=evidence,
            config=config,
        ),
        "angle_sample_count": len(angle_samples),
        "independent_angle_sample_count": len(independent),
        **angle_diagnostics,
    }
    return GeometryEventResult(event, decision.status, reasons, source_frame_ids, snapshot_frame_id, evidence_data, diagnostics)


def _side_state(valid_bins: int, broken_bins: int, minimum_evidence_bins: int) -> SideState:
    if valid_bins >= minimum_evidence_bins and broken_bins >= minimum_evidence_bins:
        return SideState.CONFLICT
    if broken_bins >= minimum_evidence_bins:
        return SideState.BROKEN_LOCALIZED
    if valid_bins >= minimum_evidence_bins:
        return SideState.VALID
    return SideState.UNKNOWN


def _observability_grade(*, identity_ambiguous: bool, center: CenterState, left: SideState, right: SideState, evidence, config: Mapping[str, Any]) -> str:
    if identity_ambiguous or center == CenterState.CONFLICT or left == SideState.CONFLICT or right == SideState.CONFLICT:
        return "GRADE_D"
    strong_topology = max(evidence.connected_bins, evidence.disconnected_bins) >= 2
    if strong_topology:
        return "GRADE_A"
    if evidence.left_present_bins >= int(config["evidence"]["minimum_left_presence_bins"]) and evidence.right_present_bins >= int(config["evidence"]["minimum_right_presence_bins"]):
        return "GRADE_B"
    return "GRADE_C"


def _independent_angle_samples(samples, *, minimum_frame_delta: int, minimum_timestamp_delta: float):
    selected = []
    for sample in sorted(samples, key=lambda item: (item[0], item[1])):
        if all(abs(sample[0] - prior[0]) >= minimum_frame_delta and abs(sample[1] - prior[1]) >= minimum_timestamp_delta for prior in selected):
            selected.append(sample)
    return selected


def _snapshot_frame_id(event: PhysicalEvent, observations: list[PaddleObservation]) -> int | None:
    if not observations:
        return None
    return min(
        observations,
        key=lambda item: (abs(item.source_timestamp_sec - event.crossing_timestamp_sec), item.source_frame_id),
    ).source_frame_id


def _write_event_snapshots(
    *,
    source: str,
    results: tuple[GeometryEventResult, ...],
    snapshots_root: str | Path | None,
    run_id: str | None,
    roi_xywh: tuple[int, int, int, int],
    coordinates: ChainCoordinates,
    trigger,
    jpeg_quality: int,
    strict: bool,
    timestamp_epsilon_sec: float,
) -> tuple[Path | None, Path | None]:
    if snapshots_root is None:
        return None, None
    root = Path(snapshots_root) / (run_id or "geometry_v2")
    requested: dict[int, list[GeometryEventResult]] = {}
    for result in results:
        if result.snapshot_source_frame_id is not None:
            requested.setdefault(result.snapshot_source_frame_id, []).append(result)
    if not requested:
        return None, None
    written: dict[str, Path] = {}
    remaining = sum(len(items) for items in requested.values())
    try:
        for frame in iter_original_frames(source, timestamp_epsilon_sec=timestamp_epsilon_sec):
            frame_results = requested.get(frame.source_frame_id)
            if frame_results is None:
                continue
            for result in frame_results:
                kind = "normals" if result.status == FinalStatus.NORMAL else "defects"
                target_dir = root / kind
                output = target_dir / geometry_snapshot_filename(result.event.track_ids[0], frame.source_frame_id)
                render_geometry_snapshot(
                    frame.image_bgr,
                    roi_xywh=roi_xywh,
                    coordinates=coordinates,
                    trigger_strip=trigger,
                    status=result.status.value,
                    output_path=output,
                    jpeg_quality=jpeg_quality,
                )
                written[kind] = target_dir
                remaining -= 1
            if remaining == 0:
                break
    except Exception:
        if strict:
            raise
    return written.get("defects"), written.get("normals")


def _resolve_model_path(model_path: str) -> Path:
    path = Path(model_path)
    return path if path.is_absolute() else Path(__file__).resolve().parents[2] / path


def _pinned_geometry_model(profile_model: Any, config: Mapping[str, Any]):
    """Detach V2 from legacy ROI-driven model-zoo selection."""
    return replace(
        profile_model,
        path=str(config["model"]["artifact_path"]),
        input_size=int(config["model"]["expected_input_size"]),
        model_zoo=None,
    )
