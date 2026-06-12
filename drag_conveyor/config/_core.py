from __future__ import annotations

import copy
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
import tempfile
from typing import Any

PROFILE_VERSION = "1.0.0"


class ProfileError(Exception):
    """Raised when a profile cannot be loaded or validated."""


@dataclass(slots=True)
class PreprocessConfig:
    type: str = "letterbox"
    normalize: bool = True
    color_format: str = "RGB"


@dataclass(slots=True)
class OutputFormatConfig:
    type: str = "yolo_seg_proto"
    box_format: str = "xyxy"
    has_objectness: bool = True
    class_encoding: str = "id"
    num_classes: int = 1
    num_mask_coeffs: int = 32


@dataclass(slots=True)
class ModelConfig:
    path: str = "weights/best.onnx"
    backend: str = "onnxruntime"
    task: str = "segmentation"
    model_family: str = "yolo-seg"
    input_size: int = 640
    class_names: list[str] = field(default_factory=lambda: ["white_bar"])
    conf_threshold: float = 0.4
    iou_threshold: float = 0.5
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    output_format: OutputFormatConfig = field(default_factory=OutputFormatConfig)


@dataclass(slots=True)
class TriggerBandConfig:
    min_overlap_ratio: float = 0.10
    pending_ttl_frames: int = 3
    allow_inside_band_trigger: bool = True


@dataclass(slots=True)
class InspectionRegionConfig:
    frame_width: int = 1280
    frame_height: int = 720
    x: int = 120
    y: int = 80
    w: int = 960
    h: int = 520
    trigger_band: TriggerBandConfig = field(default_factory=TriggerBandConfig)


@dataclass(slots=True)
class TrackerConfig:
    type: str = "centroid"
    max_jump_px: float = 80.0
    ttl_frames: int = 10
    min_hits: int = 2


@dataclass(slots=True)
class RulesConfig:
    mode: str = "length_width_auto_baseline"
    lower_percentile: str = "p1"
    upper_percentile: str = "p99"


@dataclass(slots=True)
class CalibrationConfig:
    min_valid_records: int = 50
    min_inlier_ratio: float = 0.70
    max_outlier_ratio: float = 0.30


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
class Profile:
    profile_version: str = PROFILE_VERSION
    profile_name: str = "line_01"
    model: ModelConfig = field(default_factory=ModelConfig)
    inspection_region: InspectionRegionConfig = field(default_factory=InspectionRegionConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    rules: RulesConfig = field(default_factory=RulesConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    calibration_result: CalibrationResult | None = None

    def clone(self) -> "Profile":
        return copy.deepcopy(self)

    def with_roi(self, roi_config: dict) -> "Profile":
        """Return a deep copy of this profile with inspection_region overridden by roi_config.

        roi_config keys: x, y, w, h, frame_width, frame_height.
        Trigger band placement is fixed by the runtime contract.
        """
        missing_keys = ROI_CONFIG_KEYS - set(roi_config)
        if missing_keys:
            raise ProfileError(f"Missing ROI keys: {', '.join(sorted(missing_keys))}")
        unknown_keys = set(roi_config) - ROI_CONFIG_KEYS
        if unknown_keys:
            raise ProfileError(f"Unsupported ROI keys: {', '.join(sorted(unknown_keys))}")

        cloned = self.clone()
        region = cloned.inspection_region
        region.x = int(roi_config["x"])
        region.y = int(roi_config["y"])
        region.w = int(roi_config["w"])
        region.h = int(roi_config["h"])
        region.frame_width = int(roi_config["frame_width"])
        region.frame_height = int(roi_config["frame_height"])
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
    p5 = float(data["p5"])
    p95 = float(data["p95"])
    return FeatureStats(
        median=float(data["median"]),
        mad=float(data["mad"]),
        p1=float(data.get("p1", p5)),
        p2=float(data.get("p2", p5)),
        p3=float(data.get("p3", p5)),
        p4=float(data.get("p4", p5)),
        p5=p5,
        p95=p95,
        p96=float(data.get("p96", p95)),
        p97=float(data.get("p97", p95)),
        p98=float(data.get("p98", p95)),
        p99=float(data.get("p99", p95)),
    )


def _required_section(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ProfileError(f"Missing or invalid '{key}' section")
    return value


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

    # Migration stub for future extension.
    if version.startswith("0."):
        upgraded = copy.deepcopy(raw)
        upgraded["profile_version"] = PROFILE_VERSION
        return upgraded

    raise ProfileError(f"No migration path from profile_version {version} to {PROFILE_VERSION}")


def profile_from_dict(raw: dict[str, Any]) -> Profile:
    data = migrate_profile_dict(raw)

    model_raw = _required_section(data, "model")
    region_raw = _required_section(data, "inspection_region")
    tracker_raw = _required_section(data, "tracker")
    rules_raw = _required_section(data, "rules")
    cal_raw = _required_section(data, "calibration")
    trigger_raw = region_raw.get("trigger_band", {})
    if not isinstance(trigger_raw, dict):
        raise ProfileError("Invalid 'inspection_region.trigger_band' section")

    if "direction" in region_raw:
        raise ProfileError("inspection_region.direction is no longer supported")
    unknown_rule_keys = set(rules_raw) - {"mode", "lower_percentile", "upper_percentile"}
    if unknown_rule_keys:
        raise ProfileError(f"Unsupported rules keys: {', '.join(sorted(unknown_rule_keys))}")
    supported_trigger_keys = {
        "min_overlap_ratio",
        "pending_ttl_frames",
        "allow_inside_band_trigger",
    }
    unknown_trigger_keys = set(trigger_raw) - supported_trigger_keys
    if unknown_trigger_keys:
        raise ProfileError(f"Unsupported trigger_band keys: {', '.join(sorted(unknown_trigger_keys))}")

    try:
        profile = Profile(
            profile_version=str(data["profile_version"]),
            profile_name=str(data.get("profile_name", "line_01")),
            model=ModelConfig(
                path=str(model_raw.get("path", "weights/best.onnx")),
                backend=str(model_raw.get("backend", "onnxruntime")),
                task=str(model_raw.get("task", "segmentation")),
                model_family=str(model_raw.get("model_family", "yolo-seg")),
                input_size=int(model_raw.get("input_size", 640)),
                class_names=[str(x) for x in model_raw.get("class_names", ["white_bar"])],
                conf_threshold=float(model_raw.get("conf_threshold", 0.4)),
                iou_threshold=float(model_raw.get("iou_threshold", 0.5)),
                preprocess=PreprocessConfig(
                    type=str(model_raw.get("preprocess", {}).get("type", "letterbox")),
                    normalize=bool(model_raw.get("preprocess", {}).get("normalize", True)),
                    color_format=str(model_raw.get("preprocess", {}).get("color_format", "RGB")),
                ),
                output_format=OutputFormatConfig(
                    type=str(model_raw.get("output_format", {}).get("type", "yolo_seg_proto")),
                    box_format=str(model_raw.get("output_format", {}).get("box_format", "xyxy")),
                    has_objectness=bool(model_raw.get("output_format", {}).get("has_objectness", True)),
                    class_encoding=str(model_raw.get("output_format", {}).get("class_encoding", "id")),
                    num_classes=int(model_raw.get("output_format", {}).get("num_classes", 1)),
                    num_mask_coeffs=int(model_raw.get("output_format", {}).get("num_mask_coeffs", 32)),
                ),
            ),
            inspection_region=InspectionRegionConfig(
                frame_width=int(region_raw.get("frame_width", 1280)),
                frame_height=int(region_raw.get("frame_height", 720)),
                x=int(region_raw.get("x", 0)),
                y=int(region_raw.get("y", 0)),
                w=int(region_raw.get("w", 1280)),
                h=int(region_raw.get("h", 720)),
                trigger_band=TriggerBandConfig(
                    min_overlap_ratio=float(trigger_raw.get("min_overlap_ratio", 0.10)),
                    pending_ttl_frames=int(trigger_raw.get("pending_ttl_frames", 3)),
                    allow_inside_band_trigger=bool(trigger_raw.get("allow_inside_band_trigger", True)),
                ),
            ),
            tracker=TrackerConfig(
                type=str(tracker_raw.get("type", "centroid")),
                max_jump_px=float(tracker_raw.get("max_jump_px", 80)),
                ttl_frames=int(tracker_raw.get("ttl_frames", 10)),
                min_hits=int(tracker_raw.get("min_hits", 2)),
            ),
            rules=RulesConfig(
                mode=str(rules_raw.get("mode", "length_width_auto_baseline")),
                lower_percentile=str(rules_raw.get("lower_percentile", "p1")),
                upper_percentile=str(rules_raw.get("upper_percentile", "p99")),
            ),
            calibration=CalibrationConfig(
                min_valid_records=int(cal_raw.get("min_valid_records", 50)),
                min_inlier_ratio=float(cal_raw.get("min_inlier_ratio", 0.70)),
                max_outlier_ratio=float(cal_raw.get("max_outlier_ratio", 0.30)),
            ),
        )
    except (TypeError, ValueError, KeyError) as exc:
        raise ProfileError(f"Invalid profile content: {exc}") from exc

    cal_result_raw = data.get("calibration_result")
    if isinstance(cal_result_raw, dict):
        features_raw = cal_result_raw.get("features", {})
        if not isinstance(features_raw, dict):
            raise ProfileError("calibration_result.features must be an object")
        try:
            profile.calibration_result = CalibrationResult(
                created_at=str(cal_result_raw.get("created_at", "")),
                rules_updated_at=str(cal_result_raw.get("rules_updated_at", "")),
                rules_version=str(cal_result_raw.get("rules_version", "geometry_v1")),
                sample_count=int(cal_result_raw.get("sample_count", 0)),
                valid_records=int(cal_result_raw.get("valid_records", 0)),
                inlier_count=int(cal_result_raw.get("inlier_count", 0)),
                outlier_count=int(cal_result_raw.get("outlier_count", 0)),
                inlier_ratio=float(cal_result_raw.get("inlier_ratio", 0.0)),
                thresholds_source=str(
                    cal_result_raw.get("thresholds_source", "auto_baseline_median_mad_p1_p99")
                ),
                features={
                    str(name): _feature_stats_from_dict(stats)
                    for name, stats in features_raw.items()
                    if isinstance(stats, dict) and str(name) in REQUIRED_CALIBRATION_FEATURES
                },
            )
        except (TypeError, ValueError, KeyError) as exc:
            raise ProfileError(f"Invalid calibration_result: {exc}") from exc

    validate_profile(profile)
    return profile


def validate_profile(profile: Profile) -> None:
    model = profile.model
    if not model.path.strip():
        raise ProfileError("model.path must not be empty")
    if model.input_size <= 0:
        raise ProfileError("model.input_size must be positive")
    if not 0 <= model.conf_threshold <= 1:
        raise ProfileError("model.conf_threshold must be in [0, 1]")
    if not 0 <= model.iou_threshold <= 1:
        raise ProfileError("model.iou_threshold must be in [0, 1]")
    if not model.class_names:
        raise ProfileError("model.class_names must not be empty")
    if model.preprocess.type not in {"letterbox"}:
        raise ProfileError("model.preprocess.type must be 'letterbox'")
    if model.preprocess.color_format.upper() not in {"RGB", "BGR"}:
        raise ProfileError("model.preprocess.color_format must be RGB or BGR")
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

    region = profile.inspection_region
    if region.frame_width <= 0 or region.frame_height <= 0:
        raise ProfileError("inspection_region frame size must be positive")
    if region.x < 0 or region.y < 0:
        raise ProfileError("inspection_region x/y must be >= 0")
    if region.w <= 0 or region.h <= 0:
        raise ProfileError("inspection_region dimensions must be positive")
    if region.x + region.w > region.frame_width or region.y + region.h > region.frame_height:
        raise ProfileError("inspection_region must be inside frame bounds")

    band = region.trigger_band
    if not 0 <= band.min_overlap_ratio <= 1:
        raise ProfileError("trigger_band.min_overlap_ratio must be in [0, 1]")
    if band.pending_ttl_frames < 0:
        raise ProfileError("trigger_band.pending_ttl_frames must be >= 0")

    tracker = profile.tracker
    if tracker.max_jump_px <= 0:
        raise ProfileError("tracker.max_jump_px must be positive")
    if tracker.ttl_frames < 0:
        raise ProfileError("tracker.ttl_frames must be >= 0")
    if tracker.min_hits < 1:
        raise ProfileError("tracker.min_hits must be >= 1")

    rules = profile.rules
    if rules.mode != "length_width_auto_baseline":
        raise ProfileError("rules.mode must be 'length_width_auto_baseline'")
    if rules.lower_percentile not in LOWER_RULE_PERCENTILES:
        allowed = ", ".join(LOWER_RULE_PERCENTILES)
        raise ProfileError(f"rules.lower_percentile must be one of: {allowed}")
    if rules.upper_percentile not in UPPER_RULE_PERCENTILES:
        allowed = ", ".join(UPPER_RULE_PERCENTILES)
        raise ProfileError(f"rules.upper_percentile must be one of: {allowed}")

    calibration = profile.calibration
    if calibration.min_valid_records <= 0:
        raise ProfileError("calibration.min_valid_records must be > 0")
    if not 0 <= calibration.min_inlier_ratio <= 1:
        raise ProfileError("calibration.min_inlier_ratio must be in [0, 1]")
    if not 0 <= calibration.max_outlier_ratio <= 1:
        raise ProfileError("calibration.max_outlier_ratio must be in [0, 1]")

    if profile.calibration_result is not None:
        missing_features = REQUIRED_CALIBRATION_FEATURES - set(profile.calibration_result.features.keys())
        if missing_features:
            missing = ", ".join(sorted(missing_features))
            raise ProfileError(f"calibration_result missing required features: {missing}")


def profile_to_dict(profile: Profile) -> dict[str, Any]:
    data = asdict(profile)
    return data


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


def default_profile(model_path: str = "weights/best.onnx") -> Profile:
    profile = Profile()
    profile.model.path = model_path
    return profile
