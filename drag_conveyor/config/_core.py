from __future__ import annotations

import copy
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
import tempfile
from typing import Any

PROFILE_VERSION = "1.0.0"


class ProfileError(Exception):
    """Raised when a profile cannot be loaded or validated."""


@dataclass(slots=True)
class PreprocessConfig:
    type: str
    normalize: bool
    color_format: str
    padding_value: int


@dataclass(slots=True)
class OutputFormatConfig:
    type: str
    box_format: str
    has_objectness: bool
    class_encoding: str
    num_classes: int
    num_mask_coeffs: int


@dataclass(slots=True)
class PostprocessConfig:
    conf_threshold: float
    iou_threshold: float
    target_class_ids: list[int]
    mask_threshold: float
    crop_mask_to_bbox: bool
    min_contour_area: float
    contour_mode: str


@dataclass(slots=True)
class ModelConfig:
    path: str
    backend: str
    providers: list[str]
    task: str
    input_size: int
    preprocess: PreprocessConfig
    output_format: OutputFormatConfig
    postprocess: PostprocessConfig


@dataclass(slots=True)
class RoiConfig:
    x: int
    y: int
    w: int
    h: int


@dataclass(slots=True)
class RegionConfig:
    frame_width: int
    frame_height: int
    roi: RoiConfig


@dataclass(slots=True)
class TriggerBandConfig:
    position_ratio: float
    thickness_ratio: float
    min_overlap_ratio: float
    pending_ttl_frames: int
    allow_inside_band_trigger: bool


@dataclass(slots=True)
class TrackerConfig:
    type: str
    max_jump_px: float
    ttl_frames: int
    min_hits: int
    max_reverse_px: float
    max_area_ratio_change: float


@dataclass(slots=True)
class CollectionConfig:
    trigger_band: TriggerBandConfig
    tracker: TrackerConfig


@dataclass(slots=True)
class DefectPolicyConfig:
    min_violated_dimensions: int
    score_dimension_count: int


@dataclass(slots=True)
class CalibrationOutlierConfig:
    modified_z_score_threshold: float
    iqr_multiplier: float


@dataclass(slots=True)
class CalibrationConfig:
    min_valid_records: int
    min_inlier_ratio: float
    max_outlier_ratio: float
    outlier: CalibrationOutlierConfig


@dataclass(slots=True)
class FeatureStats:
    median: float
    mad: float
    p1: float | None = None
    p2: float | None = None
    p3: float | None = None
    p4: float | None = None
    p5: float | None = None
    p95: float | None = None
    p96: float | None = None
    p97: float | None = None
    p98: float | None = None
    p99: float | None = None


@dataclass(slots=True)
class CalibrationResult:
    created_at: str
    rules_updated_at: str
    rules_version: str
    sample_count: int
    valid_records: int
    inlier_count: int
    outlier_count: int
    inlier_ratio: float
    thresholds_source: str
    features: dict[str, FeatureStats]


@dataclass(slots=True)
class AutoBaselineConfig:
    lower_percentile: str
    upper_percentile: str
    rules_version: str
    calibration: CalibrationConfig
    calibration_result: CalibrationResult | None = None


@dataclass(slots=True)
class AverageRatioConfig:
    width_min_ratio: float
    width_max_ratio: float
    length_min_ratio: float
    length_max_ratio: float


@dataclass(slots=True)
class InspectionConfig:
    mode: str
    defect_policy: DefectPolicyConfig
    auto_baseline: AutoBaselineConfig
    average_ratio: AverageRatioConfig


@dataclass(slots=True)
class Profile:
    profile_version: str
    model: ModelConfig
    region: RegionConfig
    collection: CollectionConfig
    inspection: InspectionConfig

    def clone(self) -> "Profile":
        return copy.deepcopy(self)

    def with_roi(self, roi_config: dict) -> "Profile":
        """Return a deep copy of this profile with region.roi overridden by roi_config."""
        missing_keys = ROI_CONFIG_KEYS - set(roi_config)
        if missing_keys:
            raise ProfileError(f"Missing ROI keys: {', '.join(sorted(missing_keys))}")
        unknown_keys = set(roi_config) - ROI_CONFIG_KEYS
        if unknown_keys:
            raise ProfileError(f"Unsupported ROI keys: {', '.join(sorted(unknown_keys))}")

        cloned = self.clone()
        cloned.region.frame_width = int(roi_config["frame_width"])
        cloned.region.frame_height = int(roi_config["frame_height"])
        roi = cloned.region.roi
        roi.x = int(roi_config["x"])
        roi.y = int(roi_config["y"])
        roi.w = int(roi_config["w"])
        roi.h = int(roi_config["h"])
        validate_profile(cloned)
        return cloned


REQUIRED_CALIBRATION_FEATURES = {
    "length",
    "width",
}

ROI_CONFIG_KEYS = {
    "x",
    "y",
    "w",
    "h",
    "frame_width",
    "frame_height",
}

LOWER_RULE_PERCENTILES = ("p1", "p2", "p3", "p4", "p5")
UPPER_RULE_PERCENTILES = ("p95", "p96", "p97", "p98", "p99")


def _parse_version(version: str) -> tuple[int, int, int]:
    parts = version.split(".")
    if len(parts) != 3:
        raise ProfileError(f"Invalid profile_version format: {version}")
    try:
        return tuple(int(x) for x in parts)  # type: ignore[return-value]
    except ValueError as exc:
        raise ProfileError(f"Invalid profile_version format: {version}") from exc


def _feature_stats_from_dict(data: dict[str, Any]) -> FeatureStats:
    return FeatureStats(
        median=_required_float(data, "median"),
        mad=_required_float(data, "mad"),
        p1=_required_float(data, "p1"),
        p2=_required_float(data, "p2"),
        p3=_required_float(data, "p3"),
        p4=_required_float(data, "p4"),
        p5=_required_float(data, "p5"),
        p95=_required_float(data, "p95"),
        p96=_required_float(data, "p96"),
        p97=_required_float(data, "p97"),
        p98=_required_float(data, "p98"),
        p99=_required_float(data, "p99"),
    )


def _calibration_result_from_dict(data: dict[str, Any]) -> CalibrationResult:
    features_raw = _required_section(data, "features", "calibration_result.features")
    return CalibrationResult(
        created_at=_required_str(data, "created_at", "calibration_result.created_at"),
        rules_updated_at=_required_str(
            data,
            "rules_updated_at",
            "calibration_result.rules_updated_at",
        ),
        rules_version=_required_str(data, "rules_version", "calibration_result.rules_version"),
        sample_count=_required_int(data, "sample_count", "calibration_result.sample_count"),
        valid_records=_required_int(data, "valid_records", "calibration_result.valid_records"),
        inlier_count=_required_int(data, "inlier_count", "calibration_result.inlier_count"),
        outlier_count=_required_int(data, "outlier_count", "calibration_result.outlier_count"),
        inlier_ratio=_required_float(data, "inlier_ratio", "calibration_result.inlier_ratio"),
        thresholds_source=_required_str(
            data,
            "thresholds_source",
            "calibration_result.thresholds_source",
        ),
        features={
            str(name): _feature_stats_from_dict(stats)
            for name, stats in features_raw.items()
            if isinstance(stats, dict) and str(name) in REQUIRED_CALIBRATION_FEATURES
        },
    )


def _required_value(data: dict[str, Any], key: str, label: str | None = None) -> Any:
    name = label or key
    if key not in data:
        raise ProfileError(f"Missing required config field: {name}")
    return data[key]


def _required_section(data: dict[str, Any], key: str, label: str | None = None) -> dict[str, Any]:
    name = label or key
    value = data.get(key)
    if not isinstance(value, dict):
        raise ProfileError(f"Missing or invalid '{name}' section")
    return value


def _required_str(data: dict[str, Any], key: str, label: str | None = None) -> str:
    name = label or key
    value = _required_value(data, key, name)
    if not isinstance(value, str):
        raise ProfileError(f"{name} must be a string")
    return value


def _required_bool(data: dict[str, Any], key: str, label: str | None = None) -> bool:
    name = label or key
    value = _required_value(data, key, name)
    if not isinstance(value, bool):
        raise ProfileError(f"{name} must be a boolean")
    return value


def _required_int(data: dict[str, Any], key: str, label: str | None = None) -> int:
    name = label or key
    value = _required_value(data, key, name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProfileError(f"{name} must be an integer")
    return value


def _required_float(data: dict[str, Any], key: str, label: str | None = None) -> float:
    name = label or key
    value = _required_value(data, key, name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProfileError(f"{name} must be a number")
    return float(value)


def _required_list(data: dict[str, Any], key: str, label: str | None = None) -> list[Any]:
    name = label or key
    value = _required_value(data, key, name)
    if not isinstance(value, list):
        raise ProfileError(f"{name} must be a list")
    return value


def _required_str_list(data: dict[str, Any], key: str, label: str | None = None) -> list[str]:
    name = label or key
    values = _required_list(data, key, name)
    if any(not isinstance(value, str) for value in values):
        raise ProfileError(f"{name} must contain only strings")
    return list(values)


def _required_int_list(data: dict[str, Any], key: str, label: str | None = None) -> list[int]:
    name = label or key
    values = _required_list(data, key, name)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ProfileError(f"{name} must contain only integers")
    return list(values)


def _reject_unknown_keys(data: dict[str, Any], allowed: set[str], label: str) -> None:
    unknown = set(data) - allowed
    if unknown:
        raise ProfileError(f"Unsupported {label} keys: {', '.join(sorted(unknown))}")


def migrate_profile_dict(raw: dict[str, Any]) -> dict[str, Any]:
    version = raw.get("profile_version")
    if not isinstance(version, str):
        raise ProfileError("profile_version is required")

    current = _parse_version(PROFILE_VERSION)
    requested = _parse_version(version)
    if requested == current:
        return raw
    if requested > current:
        raise ProfileError(
            f"Profile version {version} is newer than supported {PROFILE_VERSION}. Please upgrade the app."
        )
    raise ProfileError(f"No migration path from profile_version {version} to {PROFILE_VERSION}")


def profile_from_dict(raw: dict[str, Any]) -> Profile:
    data = migrate_profile_dict(raw)
    _reject_unknown_keys(
        data,
        {"profile_version", "model", "region", "collection", "inspection"},
        "profile",
    )

    model_raw = _required_section(data, "model")
    preprocess_raw = _required_section(model_raw, "preprocess", "model.preprocess")
    output_format_raw = _required_section(model_raw, "output_format", "model.output_format")
    postprocess_raw = _required_section(model_raw, "postprocess", "model.postprocess")
    region_raw = _required_section(data, "region")
    roi_raw = _required_section(region_raw, "roi", "region.roi")
    collection_raw = _required_section(data, "collection")
    trigger_raw = _required_section(collection_raw, "trigger_band", "collection.trigger_band")
    tracker_raw = _required_section(collection_raw, "tracker", "collection.tracker")
    inspection_raw = _required_section(data, "inspection")
    defect_policy_raw = _required_section(
        inspection_raw,
        "defect_policy",
        "inspection.defect_policy",
    )
    auto_raw = _required_section(inspection_raw, "auto_baseline", "inspection.auto_baseline")
    average_ratio_raw = _required_section(
        inspection_raw,
        "average_ratio",
        "inspection.average_ratio",
    )
    calibration_raw = _required_section(
        auto_raw,
        "calibration",
        "inspection.auto_baseline.calibration",
    )
    outlier_raw = _required_section(
        calibration_raw,
        "outlier",
        "inspection.auto_baseline.calibration.outlier",
    )

    _reject_unknown_keys(
        model_raw,
        {"path", "backend", "providers", "task", "input_size", "preprocess", "output_format", "postprocess"},
        "model",
    )
    _reject_unknown_keys(
        preprocess_raw,
        {"type", "normalize", "color_format", "padding_value"},
        "model.preprocess",
    )
    _reject_unknown_keys(
        output_format_raw,
        {"type", "box_format", "has_objectness", "class_encoding", "num_classes", "num_mask_coeffs"},
        "model.output_format",
    )
    _reject_unknown_keys(
        postprocess_raw,
        {
            "conf_threshold",
            "iou_threshold",
            "target_class_ids",
            "mask_threshold",
            "crop_mask_to_bbox",
            "min_contour_area",
            "contour_mode",
        },
        "model.postprocess",
    )
    _reject_unknown_keys(region_raw, {"frame_width", "frame_height", "roi"}, "region")
    _reject_unknown_keys(roi_raw, {"x", "y", "w", "h"}, "region.roi")
    _reject_unknown_keys(collection_raw, {"trigger_band", "tracker"}, "collection")
    _reject_unknown_keys(
        trigger_raw,
        {
            "position_ratio",
            "thickness_ratio",
            "min_overlap_ratio",
            "pending_ttl_frames",
            "allow_inside_band_trigger",
        },
        "collection.trigger_band",
    )
    _reject_unknown_keys(
        tracker_raw,
        {
            "type",
            "max_jump_px",
            "ttl_frames",
            "min_hits",
            "max_reverse_px",
            "max_area_ratio_change",
        },
        "collection.tracker",
    )
    _reject_unknown_keys(
        inspection_raw,
        {"mode", "defect_policy", "auto_baseline", "average_ratio"},
        "inspection",
    )
    _reject_unknown_keys(
        defect_policy_raw,
        {"min_violated_dimensions", "score_dimension_count"},
        "inspection.defect_policy",
    )
    _reject_unknown_keys(
        auto_raw,
        {"lower_percentile", "upper_percentile", "rules_version", "calibration", "calibration_result"},
        "inspection.auto_baseline",
    )
    _reject_unknown_keys(
        average_ratio_raw,
        {"width_min_ratio", "width_max_ratio", "length_min_ratio", "length_max_ratio"},
        "inspection.average_ratio",
    )
    _reject_unknown_keys(
        calibration_raw,
        {"min_valid_records", "min_inlier_ratio", "max_outlier_ratio", "outlier"},
        "inspection.auto_baseline.calibration",
    )
    _reject_unknown_keys(
        outlier_raw,
        {"modified_z_score_threshold", "iqr_multiplier"},
        "inspection.auto_baseline.calibration.outlier",
    )

    cal_result_raw = auto_raw.get("calibration_result")
    if isinstance(cal_result_raw, dict):
        calibration_result = _calibration_result_from_dict(cal_result_raw)
    elif cal_result_raw is None:
        calibration_result = None
    else:
        raise ProfileError("inspection.auto_baseline.calibration_result must be null or an object")

    try:
        profile = Profile(
            profile_version=_required_str(data, "profile_version"),
            model=ModelConfig(
                path=_required_str(model_raw, "path", "model.path"),
                backend=_required_str(model_raw, "backend", "model.backend"),
                providers=_required_str_list(model_raw, "providers", "model.providers"),
                task=_required_str(model_raw, "task", "model.task"),
                input_size=_required_int(model_raw, "input_size", "model.input_size"),
                preprocess=PreprocessConfig(
                    type=_required_str(preprocess_raw, "type", "model.preprocess.type"),
                    normalize=_required_bool(preprocess_raw, "normalize", "model.preprocess.normalize"),
                    color_format=_required_str(
                        preprocess_raw,
                        "color_format",
                        "model.preprocess.color_format",
                    ),
                    padding_value=_required_int(
                        preprocess_raw,
                        "padding_value",
                        "model.preprocess.padding_value",
                    ),
                ),
                output_format=OutputFormatConfig(
                    type=_required_str(output_format_raw, "type", "model.output_format.type"),
                    box_format=_required_str(
                        output_format_raw,
                        "box_format",
                        "model.output_format.box_format",
                    ),
                    has_objectness=_required_bool(
                        output_format_raw,
                        "has_objectness",
                        "model.output_format.has_objectness",
                    ),
                    class_encoding=_required_str(
                        output_format_raw,
                        "class_encoding",
                        "model.output_format.class_encoding",
                    ),
                    num_classes=_required_int(
                        output_format_raw,
                        "num_classes",
                        "model.output_format.num_classes",
                    ),
                    num_mask_coeffs=_required_int(
                        output_format_raw,
                        "num_mask_coeffs",
                        "model.output_format.num_mask_coeffs",
                    ),
                ),
                postprocess=PostprocessConfig(
                    conf_threshold=_required_float(
                        postprocess_raw,
                        "conf_threshold",
                        "model.postprocess.conf_threshold",
                    ),
                    iou_threshold=_required_float(
                        postprocess_raw,
                        "iou_threshold",
                        "model.postprocess.iou_threshold",
                    ),
                    target_class_ids=_required_int_list(
                        postprocess_raw,
                        "target_class_ids",
                        "model.postprocess.target_class_ids",
                    ),
                    mask_threshold=_required_float(
                        postprocess_raw,
                        "mask_threshold",
                        "model.postprocess.mask_threshold",
                    ),
                    crop_mask_to_bbox=_required_bool(
                        postprocess_raw,
                        "crop_mask_to_bbox",
                        "model.postprocess.crop_mask_to_bbox",
                    ),
                    min_contour_area=_required_float(
                        postprocess_raw,
                        "min_contour_area",
                        "model.postprocess.min_contour_area",
                    ),
                    contour_mode=_required_str(
                        postprocess_raw,
                        "contour_mode",
                        "model.postprocess.contour_mode",
                    ),
                ),
            ),
            region=RegionConfig(
                frame_width=_required_int(region_raw, "frame_width", "region.frame_width"),
                frame_height=_required_int(region_raw, "frame_height", "region.frame_height"),
                roi=RoiConfig(
                    x=_required_int(roi_raw, "x", "region.roi.x"),
                    y=_required_int(roi_raw, "y", "region.roi.y"),
                    w=_required_int(roi_raw, "w", "region.roi.w"),
                    h=_required_int(roi_raw, "h", "region.roi.h"),
                ),
            ),
            collection=CollectionConfig(
                trigger_band=TriggerBandConfig(
                    position_ratio=_required_float(
                        trigger_raw,
                        "position_ratio",
                        "collection.trigger_band.position_ratio",
                    ),
                    thickness_ratio=_required_float(
                        trigger_raw,
                        "thickness_ratio",
                        "collection.trigger_band.thickness_ratio",
                    ),
                    min_overlap_ratio=_required_float(
                        trigger_raw,
                        "min_overlap_ratio",
                        "collection.trigger_band.min_overlap_ratio",
                    ),
                    pending_ttl_frames=_required_int(
                        trigger_raw,
                        "pending_ttl_frames",
                        "collection.trigger_band.pending_ttl_frames",
                    ),
                    allow_inside_band_trigger=_required_bool(
                        trigger_raw,
                        "allow_inside_band_trigger",
                        "collection.trigger_band.allow_inside_band_trigger",
                    ),
                ),
                tracker=TrackerConfig(
                    type=_required_str(tracker_raw, "type", "collection.tracker.type"),
                    max_jump_px=_required_float(
                        tracker_raw,
                        "max_jump_px",
                        "collection.tracker.max_jump_px",
                    ),
                    ttl_frames=_required_int(
                        tracker_raw,
                        "ttl_frames",
                        "collection.tracker.ttl_frames",
                    ),
                    min_hits=_required_int(
                        tracker_raw,
                        "min_hits",
                        "collection.tracker.min_hits",
                    ),
                    max_reverse_px=_required_float(
                        tracker_raw,
                        "max_reverse_px",
                        "collection.tracker.max_reverse_px",
                    ),
                    max_area_ratio_change=_required_float(
                        tracker_raw,
                        "max_area_ratio_change",
                        "collection.tracker.max_area_ratio_change",
                    ),
                ),
            ),
            inspection=InspectionConfig(
                mode=_required_str(inspection_raw, "mode", "inspection.mode"),
                defect_policy=DefectPolicyConfig(
                    min_violated_dimensions=_required_int(
                        defect_policy_raw,
                        "min_violated_dimensions",
                        "inspection.defect_policy.min_violated_dimensions",
                    ),
                    score_dimension_count=_required_int(
                        defect_policy_raw,
                        "score_dimension_count",
                        "inspection.defect_policy.score_dimension_count",
                    ),
                ),
                auto_baseline=AutoBaselineConfig(
                    lower_percentile=_required_str(
                        auto_raw,
                        "lower_percentile",
                        "inspection.auto_baseline.lower_percentile",
                    ),
                    upper_percentile=_required_str(
                        auto_raw,
                        "upper_percentile",
                        "inspection.auto_baseline.upper_percentile",
                    ),
                    rules_version=_required_str(
                        auto_raw,
                        "rules_version",
                        "inspection.auto_baseline.rules_version",
                    ),
                    calibration=CalibrationConfig(
                        min_valid_records=_required_int(
                            calibration_raw,
                            "min_valid_records",
                            "inspection.auto_baseline.calibration.min_valid_records",
                        ),
                        min_inlier_ratio=_required_float(
                            calibration_raw,
                            "min_inlier_ratio",
                            "inspection.auto_baseline.calibration.min_inlier_ratio",
                        ),
                        max_outlier_ratio=_required_float(
                            calibration_raw,
                            "max_outlier_ratio",
                            "inspection.auto_baseline.calibration.max_outlier_ratio",
                        ),
                        outlier=CalibrationOutlierConfig(
                            modified_z_score_threshold=_required_float(
                                outlier_raw,
                                "modified_z_score_threshold",
                                "inspection.auto_baseline.calibration.outlier.modified_z_score_threshold",
                            ),
                            iqr_multiplier=_required_float(
                                outlier_raw,
                                "iqr_multiplier",
                                "inspection.auto_baseline.calibration.outlier.iqr_multiplier",
                            ),
                        ),
                    ),
                    calibration_result=calibration_result,
                ),
                average_ratio=AverageRatioConfig(
                    width_min_ratio=_required_float(
                        average_ratio_raw,
                        "width_min_ratio",
                        "inspection.average_ratio.width_min_ratio",
                    ),
                    width_max_ratio=_required_float(
                        average_ratio_raw,
                        "width_max_ratio",
                        "inspection.average_ratio.width_max_ratio",
                    ),
                    length_min_ratio=_required_float(
                        average_ratio_raw,
                        "length_min_ratio",
                        "inspection.average_ratio.length_min_ratio",
                    ),
                    length_max_ratio=_required_float(
                        average_ratio_raw,
                        "length_max_ratio",
                        "inspection.average_ratio.length_max_ratio",
                    ),
                ),
            ),
        )
    except (TypeError, ValueError, KeyError) as exc:
        raise ProfileError(f"Invalid profile content: {exc}") from exc

    validate_profile(profile)
    return profile


def validate_profile(profile: Profile) -> None:
    inspection = profile.inspection
    if inspection.mode not in {"auto_baseline", "average_ratio"}:
        raise ProfileError("inspection.mode must be auto_baseline or average_ratio")

    model = profile.model
    if not model.path.strip():
        raise ProfileError("model.path must not be empty")
    if not model.providers or any(not provider.strip() for provider in model.providers):
        raise ProfileError("model.providers must contain at least one provider")
    if model.backend.lower() != "onnxruntime":
        raise ProfileError("model.backend must be onnxruntime")
    if model.task != "segmentation":
        raise ProfileError("model.task must be segmentation")
    if model.input_size <= 0:
        raise ProfileError("model.input_size must be positive")
    if model.preprocess.type not in {"letterbox"}:
        raise ProfileError("model.preprocess.type must be 'letterbox'")
    if model.preprocess.color_format.upper() not in {"RGB", "BGR"}:
        raise ProfileError("model.preprocess.color_format must be RGB or BGR")
    if not 0 <= model.preprocess.padding_value <= 255:
        raise ProfileError("model.preprocess.padding_value must be in [0, 255]")
    if model.output_format.type != "yolo_seg_proto":
        raise ProfileError("model.output_format.type must be yolo_seg_proto")
    if model.output_format.box_format not in {"xywh", "xyxy"}:
        raise ProfileError("model.output_format.box_format must be xywh or xyxy")
    if model.output_format.class_encoding not in {"id", "scores"}:
        raise ProfileError("model.output_format.class_encoding must be id or scores")
    if model.output_format.num_classes <= 0:
        raise ProfileError("model.output_format.num_classes must be > 0")
    if model.output_format.num_mask_coeffs <= 0:
        raise ProfileError("model.output_format.num_mask_coeffs must be > 0")

    postprocess = model.postprocess
    if not 0 <= postprocess.conf_threshold <= 1:
        raise ProfileError("model.postprocess.conf_threshold must be in [0, 1]")
    if not 0 <= postprocess.iou_threshold <= 1:
        raise ProfileError("model.postprocess.iou_threshold must be in [0, 1]")
    if not postprocess.target_class_ids:
        raise ProfileError("model.postprocess.target_class_ids must not be empty")
    if any(class_id < 0 for class_id in postprocess.target_class_ids):
        raise ProfileError("model.postprocess.target_class_ids must be >= 0")
    if not 0 <= postprocess.mask_threshold <= 1:
        raise ProfileError("model.postprocess.mask_threshold must be in [0, 1]")
    if postprocess.min_contour_area < 0:
        raise ProfileError("model.postprocess.min_contour_area must be >= 0")
    if postprocess.contour_mode not in {"largest", "union"}:
        raise ProfileError("model.postprocess.contour_mode must be largest or union")

    region = profile.region
    roi = region.roi
    if region.frame_width <= 0 or region.frame_height <= 0:
        raise ProfileError("region frame size must be positive")
    if roi.x < 0 or roi.y < 0:
        raise ProfileError("region.roi x/y must be >= 0")
    if roi.w <= 0 or roi.h <= 0:
        raise ProfileError("region.roi dimensions must be positive")
    if roi.x + roi.w > region.frame_width or roi.y + roi.h > region.frame_height:
        raise ProfileError("region.roi must be inside frame bounds")

    band = profile.collection.trigger_band
    if not 0 <= band.position_ratio <= 1:
        raise ProfileError("collection.trigger_band.position_ratio must be in [0, 1]")
    if not 0 < band.thickness_ratio <= 1:
        raise ProfileError("collection.trigger_band.thickness_ratio must be in (0, 1]")
    if not 0 <= band.min_overlap_ratio <= 1:
        raise ProfileError("collection.trigger_band.min_overlap_ratio must be in [0, 1]")
    if band.pending_ttl_frames < 0:
        raise ProfileError("collection.trigger_band.pending_ttl_frames must be >= 0")

    tracker = profile.collection.tracker
    if tracker.max_jump_px <= 0:
        raise ProfileError("collection.tracker.max_jump_px must be positive")
    if tracker.ttl_frames < 0:
        raise ProfileError("collection.tracker.ttl_frames must be >= 0")
    if tracker.min_hits < 1:
        raise ProfileError("collection.tracker.min_hits must be >= 1")
    if tracker.max_reverse_px < 0:
        raise ProfileError("collection.tracker.max_reverse_px must be >= 0")
    if tracker.max_area_ratio_change < 1:
        raise ProfileError("collection.tracker.max_area_ratio_change must be >= 1")

    policy = inspection.defect_policy
    if policy.min_violated_dimensions < 1:
        raise ProfileError("inspection.defect_policy.min_violated_dimensions must be >= 1")
    if policy.score_dimension_count < 1:
        raise ProfileError("inspection.defect_policy.score_dimension_count must be >= 1")
    if policy.min_violated_dimensions > policy.score_dimension_count:
        raise ProfileError(
            "inspection.defect_policy.min_violated_dimensions must be <= score_dimension_count"
        )

    auto = inspection.auto_baseline
    if auto.lower_percentile not in LOWER_RULE_PERCENTILES:
        allowed = ", ".join(LOWER_RULE_PERCENTILES)
        raise ProfileError(f"inspection.auto_baseline.lower_percentile must be one of: {allowed}")
    if auto.upper_percentile not in UPPER_RULE_PERCENTILES:
        allowed = ", ".join(UPPER_RULE_PERCENTILES)
        raise ProfileError(f"inspection.auto_baseline.upper_percentile must be one of: {allowed}")
    if not auto.rules_version.strip():
        raise ProfileError("inspection.auto_baseline.rules_version must not be empty")

    average_ratio = inspection.average_ratio
    ratio_fields = {
        "inspection.average_ratio.width_min_ratio": average_ratio.width_min_ratio,
        "inspection.average_ratio.width_max_ratio": average_ratio.width_max_ratio,
        "inspection.average_ratio.length_min_ratio": average_ratio.length_min_ratio,
        "inspection.average_ratio.length_max_ratio": average_ratio.length_max_ratio,
    }
    for name, value in ratio_fields.items():
        if value <= 0:
            raise ProfileError(f"{name} must be > 0")
    if average_ratio.width_min_ratio >= average_ratio.width_max_ratio:
        raise ProfileError("inspection.average_ratio.width_min_ratio must be < width_max_ratio")
    if average_ratio.length_min_ratio >= average_ratio.length_max_ratio:
        raise ProfileError("inspection.average_ratio.length_min_ratio must be < length_max_ratio")

    calibration = auto.calibration
    if calibration.min_valid_records <= 0:
        raise ProfileError("inspection.auto_baseline.calibration.min_valid_records must be > 0")
    if not 0 <= calibration.min_inlier_ratio <= 1:
        raise ProfileError("inspection.auto_baseline.calibration.min_inlier_ratio must be in [0, 1]")
    if not 0 <= calibration.max_outlier_ratio <= 1:
        raise ProfileError("inspection.auto_baseline.calibration.max_outlier_ratio must be in [0, 1]")
    if calibration.outlier.modified_z_score_threshold <= 0:
        raise ProfileError(
            "inspection.auto_baseline.calibration.outlier.modified_z_score_threshold must be > 0"
        )
    if calibration.outlier.iqr_multiplier <= 0:
        raise ProfileError("inspection.auto_baseline.calibration.outlier.iqr_multiplier must be > 0")

    if auto.calibration_result is not None:
        missing_features = REQUIRED_CALIBRATION_FEATURES - set(auto.calibration_result.features.keys())
        if missing_features:
            missing = ", ".join(sorted(missing_features))
            raise ProfileError(f"calibration_result missing required features: {missing}")


def profile_to_dict(profile: Profile) -> dict[str, Any]:
    return asdict(profile)


def load_profile(path: str | Path) -> Profile:
    p = Path(path)
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ProfileError(f"Profile not found: {p}") from exc
    except json.JSONDecodeError as exc:
        raise ProfileError(f"Invalid JSON in profile: {p}") from exc

    if not isinstance(raw, dict):
        raise ProfileError("Profile root must be a JSON object")
    return profile_from_dict(raw)


def save_profile(profile: Profile, path: str | Path) -> None:
    validate_profile(profile)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(profile_to_dict(profile), indent=2, ensure_ascii=False)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(p.parent),
        prefix=f"{p.name}.",
        suffix=".tmp",
        delete=False,
    ) as fp:
        fp.write(payload)
        fp.flush()
        os.fsync(fp.fileno())
        tmp_name = fp.name
    Path(tmp_name).replace(p)
