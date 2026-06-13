# Feature Specification v5: Local Conveyor Bar Defect Analysis Using Segmentation Masks and LAB Color

## 0. Document status

This is the corrected implementation specification for adding local defect classification to the `Drag_Conveyor` project.

This version supersedes:

- `local_defects_feature_spec.md`
- `local_defects_feature_spec_v2.md`
- `local_defects_feature_spec_v3.md`
- `local_defects_feature_spec_v4.md`

This v5 document includes the technical corrections identified during v2, v3, and v4 review:

1. Preserve the existing `DefectPolicyConfig.min_violated_dimensions` behavior for geometry defects.
2. Preserve backward compatibility for existing positional calls to `RuleEngine.evaluate(...)`.
3. Use a safer `max_template_area_ratio` for tight canonical masks.
4. Add strict config parsing helpers for zone float pairs.
5. Add aspect-ratio, contour, and endpoint x-separation guards in canonicalization.
6. Document the limitation of using the current contour's `minAreaRect` for full-end loss localization.
7. Add the repository-specific GitNexus workflow required by `AGENTS.md`.
8. Define `LocalDefectFeatures.zero_failed()` explicitly.
9. Avoid circular baseline-color helper design.
10. Do not allow low alignment IoU to suppress severe local shape defects silently.
11. Make migration from profile `1.0.0` backward-compatible by disabling local defect for migrated profiles.
12. Explicitly add `local_defect` to `inspection` allowed keys.
13. Make `both_sides_threshold` meaningful instead of redundant.
14. Guard color-rule threshold variables when color analysis is disabled or when the local baseline is missing.
15. Make the left/right box-ordering helper receive the endpoint x-separation threshold explicitly.
16. Correct the documented current `RuleEngine.evaluate(...)` signature to match the source.
17. Preserve `BarResult.source_frame` behavior for both normal and defect bars in phase 1.

The goal is that a developer can read this document and implement the feature with low ambiguity.

---

## 1. Business and product requirements

The following decisions are confirmed and must be implemented as stated:

1. **Left/right are defined from the camera-viewer's perspective.**
   - `deform_left` means the left side as seen in the image.
   - `deform_right` means the right side as seen in the image.
   - Do not define left/right by belt movement direction or physical conveyor orientation.

2. **A single bar may have multiple defects at the same time.**
   - Backend output must remain multi-label via `reasons: list[str]`.
   - Example:

```json
{
  "track_id": 17,
  "reasons": ["deform_both_sides", "deform_middle", "color_defect"]
}
```

3. **`color_defect` is based on LAB pixel comparison inside the detected mask.**
   - Use the normal bars in the same video to build the color baseline.
   - Compare each triggered bar's masked pixels against that baseline.
   - If the abnormal color ratio exceeds the threshold, add `color_defect`.

4. **If local baseline cannot be built or is not stable, the job fails.**
   - This applies only when `inspection.mode = auto_baseline` and `inspection.local_defect.enabled = true`.
   - The returned `failure_reason` must be explicit, for example:
     - `local_baseline_not_stable: not enough template samples`
     - `local_baseline_not_stable: template IoU p50 below threshold`
     - `local_baseline_not_stable: color baseline empty`

5. **Normal bars are primarily white; color defects are primarily black or unusually dark regions.**
   - LAB comparison is the primary mechanism.
   - A dark-pixel ratio based on the LAB `L` channel should be included as a secondary signal.

6. **Reports are out of scope.**
   - The user currently creates reports manually.
   - Do not implement a report generator or modify report templates in this phase.

---

## 2. Feature scope

### 2.1 In scope for phase 1

Implement local defect analysis as a post-processing layer after the existing YOLO segmentation, tracking, trigger, and geometry calibration pipeline.

Phase 1 must:

- Keep the existing YOLO segmentation model unchanged.
- Keep `weights/best.onnx` unchanged.
- Keep existing segmentation post-processing unchanged.
- Keep existing tracking and trigger-band behavior unchanged.
- Keep existing `measure_contour()` for `length` and `width` unchanged.
- Add a new module: `drag_conveyor/pipeline/local_defects.py`.
- Add local mask-shape analysis by normalized left/middle/right zones.
- Add color analysis using LAB and dark-pixel ratio.
- Add new reasons:
  - `deform_left`
  - `deform_right`
  - `deform_middle`
  - `deform_both_sides`
  - `color_defect`
- Support local defect analysis only for `auto_baseline` in phase 1.
- Fail the job when local baseline is required but cannot be built or is unstable.
- Add tests using `unittest`, consistent with the current repository.

### 2.2 Out of scope for phase 1

Do not implement:

- YOLO retraining.
- Additional YOLO classes such as `deform_left` or `color_defect`.
- A deep-learning classifier.
- Missing-bar detection based on pitch/distance.
- Conveyor full-loop detection.
- Report generation.
- UI configuration for local defect thresholds.
- Exposure of local defect config through `/api/runtime-config`.
- Uploading canonical debug masks or defect maps to R2 by default.
- Text overlays on the existing default defect snapshots, unless tests are updated explicitly.

### 2.3 Mode behavior

Phase 1 behavior must be explicit:

| Inspection mode | `local_defect.enabled` | Behavior |
|---|---:|---|
| `auto_baseline` | `true` | Run geometry calibration, build local baseline, run local shape/color analysis, fail if local baseline is not stable. |
| `auto_baseline` | `false` | Existing behavior only. |
| `average_ratio` | any | Existing behavior only. Do not run local defect analysis in phase 1. |

The local-baseline failure rule applies only to:

```text
auto_baseline + local_defect.enabled = true
```

---

## 3. Current code references

The relevant current implementation is spread across these files.

### 3.1 Batch pipeline

File:

```text
drag_conveyor/app/batch.py
```

Relevant symbols:

- `CollectedBar`
- `BarResult`
- `BatchInspectionResult`
- `_ClassificationOutcome`
- `run_batch_inspection(...)`
- `_classify_collected_bars(...)`
- `_classify_with_auto_baseline(...)`
- `_classify_with_average_ratio(...)`
- `_save_defect_snapshots(...)`
- `_save_box_contour_snapshot(...)`

Current high-level flow:

```text
Video source
  -> cv2/open_video_source
  -> ROI crop
  -> preprocess_roi()
  -> ONNX YOLO segmentation
  -> postprocess_segmentation()
  -> CentroidTracker
  -> TriggerEngine + trigger band
  -> measure_contour()
  -> collect CollectedBar
  -> CalibrationEngine or AverageRatioInspector
  -> RuleEngine / AverageRatioInspector evaluation
  -> BarResult
  -> snapshots and BatchInspectionResult
```

The new local defect analysis must be inserted in `_classify_with_auto_baseline(...)` after geometry calibration and before final `RuleEngine.evaluate(...)` for each collected bar.

### 3.2 Detection outputs already available

File:

```text
drag_conveyor/inference/_core.py
```

Relevant symbol:

```python
@dataclass(slots=True)
class Detection:
    class_id: int
    score: float
    bbox_roi_xyxy: tuple[float, float, float, float]
    bbox_frame_xyxy: tuple[float, float, float, float]
    centroid_frame_xy: tuple[float, float]
    mask_roi: np.ndarray
    contour_frame: np.ndarray
```

`mask_roi` is the binary segmentation mask in ROI coordinates.

`contour_frame` and `bbox_frame_xyxy` are already offset to full-frame coordinates.

Phase 1 does not require changing `Detection` or `postprocess_segmentation(...)`.

### 3.3 Segmentation postprocess

File:

```text
drag_conveyor/inference/yolo_seg_postprocess.py
```

Relevant behavior:

- Decode boxes, classes, scores, and mask coefficients.
- Filter by confidence threshold.
- Filter by `target_class_ids`.
- Apply NMS.
- Build masks from YOLO segmentation prototypes.
- Resize masks to ROI coordinates.
- Threshold masks.
- Optionally crop masks to bbox.
- Extract contour.
- Return `Detection` objects.

Do not modify this module in phase 1 unless a test reveals a direct bug unrelated to local defect analysis.

### 3.4 Geometry measurement

File:

```text
drag_conveyor/pipeline/measure.py
```

Relevant symbol:

```python
def measure_contour(contour: np.ndarray) -> Measurements:
    rect = cv2.minAreaRect(contour)
    w, h = rect[1]
    length = max(w, h)
    width = min(w, h)
```

Do not modify this function in phase 1. Local analysis must be implemented separately to keep blast radius small.

### 3.5 Existing geometry rule engine

File:

```text
drag_conveyor/pipeline/rules.py
```

Relevant symbols:

- `RuleEvaluation`
- `RuleEngine.evaluate(...)`
- `_percentile(...)`

Current reasons:

- `length_too_short`
- `length_too_long`
- `width_too_small`
- `width_too_large`

Important existing behavior that must be preserved:

```python
result = (
    "suspected_defect"
    if violated_dimensions >= defect_policy.min_violated_dimensions
    else "normal"
)
```

Do not replace this with `result = "suspected_defect" if reasons else "normal"`, because that would break existing `min_violated_dimensions` semantics and existing tests.

The correct phase-1 behavior is:

```text
geometry_defect = violated_dimensions >= defect_policy.min_violated_dimensions
local_defect = bool(local_reasons)
result = suspected_defect if geometry_defect or local_defect else normal
```

### 3.6 Config parser and validation

File:

```text
drag_conveyor/config/_core.py
```

Relevant symbols:

- `PROFILE_VERSION`
- `ProfileError`
- `InspectionConfig`
- `profile_from_dict(...)`
- `validate_profile(...)`
- `migrate_profile_dict(...)`
- `profile_to_dict(...)`
- strict unknown-key rejection helpers

Important: this project rejects unknown config keys. Adding `inspection.local_defect` requires changing both parsing and validation. Updating `config/base_profile.json` alone is not enough.

### 3.7 Config exports

File:

```text
drag_conveyor/config/__init__.py
```

When `LocalDefectConfig` is added in `_core.py`, it must also be exported from `config/__init__.py` and included in `__all__`, otherwise imports such as this will fail:

```python
from ..config import LocalDefectConfig
```

### 3.8 Server summary

File:

```text
server/worker.py
```

Relevant symbol:

- `_build_summary(...)`

Current defect summary includes geometry fields such as `length` and `width`. It can be extended with local metrics without changing the reporting system.

Do not implement report generation in phase 1.

### 3.9 Runtime config endpoint

File:

```text
server/main.py
```

Relevant endpoint:

```text
/api/runtime-config
```

Do not expose local-defect config through this endpoint in phase 1. That would increase scope and require frontend work.

---

## 4. New architecture

### 4.1 Final phase-1 pipeline

For `auto_baseline + local_defect.enabled = true`, the pipeline becomes:

```text
Frame collection
  -> YOLO segmentation
  -> tracking
  -> trigger band
  -> collect CollectedBar with length/width + mask + source frame
  -> CalibrationEngine calibrates length/width
  -> select geometry-normal candidate bars
  -> build LocalDefectBaseline from candidates
  -> analyze every collected bar with local_defects.py
  -> merge local metrics into measurements
  -> RuleEngine evaluates geometry + local reasons
  -> BarResult
```

For `average_ratio`, keep existing behavior:

```text
Frame collection
  -> AverageRatioInspector
  -> no local_defects.py call
```

### 4.2 New module

Add:

```text
drag_conveyor/pipeline/local_defects.py
```

Responsibilities:

1. Convert `mask_roi` to full-frame mask.
2. Canonicalize each bar crop and mask into a fixed coordinate system.
3. Build a median template mask from geometry-normal bars.
4. Build LAB color baseline from geometry-normal bars.
5. Compute left/middle/right shape defect scores.
6. Compute LAB color-delta metrics.
7. Compute dark-pixel ratio.
8. Return feature metrics only; do not decide final business reasons here.

`local_defects.py` should not import server modules.

---

## 5. Data model changes

### 5.1 Extend `CollectedBar`

File:

```text
drag_conveyor/app/batch.py
```

Current:

```python
@dataclass(frozen=True, slots=True)
class CollectedBar:
    frame_id: int
    track_id: int
    measurements: dict[str, float]
    bbox_frame_xyxy: tuple[float, float, float, float]
    overlap_ratio: float
    contour_frame: np.ndarray
    source_frame: np.ndarray
    latency_ms: float
```

Change to:

```python
@dataclass(frozen=True, slots=True)
class CollectedBar:
    frame_id: int
    track_id: int
    measurements: dict[str, float]
    bbox_frame_xyxy: tuple[float, float, float, float]
    overlap_ratio: float
    contour_frame: np.ndarray
    mask_roi: np.ndarray
    roi_origin_xy: tuple[int, int]
    source_frame: np.ndarray
    latency_ms: float
```

When collecting:

```python
collected.append(
    CollectedBar(
        frame_id=frame_count,
        track_id=track.track_id,
        measurements=measurements,
        bbox_frame_xyxy=track.detection.bbox_frame_xyxy,
        overlap_ratio=float(overlap),
        contour_frame=track.detection.contour_frame,
        mask_roi=track.detection.mask_roi.copy(),
        roi_origin_xy=(roi_config.x, roi_config.y),
        source_frame=frame.copy(),
        latency_ms=latency_ms,
    )
)
```

Use `.copy()` for `mask_roi` because the detection object may not be retained after tracker updates.

### 5.2 `LocalDefectFeatures`

Add in:

```text
drag_conveyor/pipeline/local_defects.py
```

Recommended shape:

```python
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class LocalDefectFeatures:
    left_shape_score: float
    middle_shape_score: float
    right_shape_score: float
    max_shape_score: float

    left_defect_weighted_pixels: float
    middle_defect_weighted_pixels: float
    right_defect_weighted_pixels: float

    shape_alignment_iou: float
    mask_area_ratio: float
    local_alignment_low: float

    color_delta_mean: float
    color_delta_p95: float
    color_abnormal_ratio: float
    dark_pixel_ratio: float
    local_color_pixels_insufficient: float

    local_analysis_success: float
    local_canonicalize_failed: float

    def to_dict(self) -> dict[str, float]:
        return {
            "left_shape_score": self.left_shape_score,
            "middle_shape_score": self.middle_shape_score,
            "right_shape_score": self.right_shape_score,
            "max_shape_score": self.max_shape_score,
            "left_defect_weighted_pixels": self.left_defect_weighted_pixels,
            "middle_defect_weighted_pixels": self.middle_defect_weighted_pixels,
            "right_defect_weighted_pixels": self.right_defect_weighted_pixels,
            "shape_alignment_iou": self.shape_alignment_iou,
            "mask_area_ratio": self.mask_area_ratio,
            "local_alignment_low": self.local_alignment_low,
            "color_delta_mean": self.color_delta_mean,
            "color_delta_p95": self.color_delta_p95,
            "color_abnormal_ratio": self.color_abnormal_ratio,
            "dark_pixel_ratio": self.dark_pixel_ratio,
            "local_color_pixels_insufficient": self.local_color_pixels_insufficient,
            "local_analysis_success": self.local_analysis_success,
            "local_canonicalize_failed": self.local_canonicalize_failed,
        }

    @classmethod
    def zero_failed(cls) -> "LocalDefectFeatures":
        return cls(
            left_shape_score=0.0,
            middle_shape_score=0.0,
            right_shape_score=0.0,
            max_shape_score=0.0,
            left_defect_weighted_pixels=0.0,
            middle_defect_weighted_pixels=0.0,
            right_defect_weighted_pixels=0.0,
            shape_alignment_iou=0.0,
            mask_area_ratio=0.0,
            local_alignment_low=1.0,
            color_delta_mean=0.0,
            color_delta_p95=0.0,
            color_abnormal_ratio=0.0,
            dark_pixel_ratio=0.0,
            local_color_pixels_insufficient=1.0,
            local_analysis_success=0.0,
            local_canonicalize_failed=1.0,
        )
```

Reason for `float` flags rather than booleans: `measurements` is currently `dict[str, float]`.

### 5.3 `LocalDefectBaseline`

Add in:

```text
drag_conveyor/pipeline/local_defects.py
```

Recommended shape:

```python
@dataclass(frozen=True, slots=True)
class LocalDefectBaseline:
    template_mask: np.ndarray
    template_prob: np.ndarray
    template_area_ratio: float

    lab_median: np.ndarray
    lab_mad: np.ndarray
    color_abnormal_ratio_p95: float
    dark_ratio_p95: float

    baseline_alignment_iou_p50: float
    baseline_alignment_iou_p10: float
    canonicalize_failure_ratio: float
    samples_used: int

    zone_slices: dict[str, slice]
```

---

## 6. Config changes

### 6.1 Add `LocalDefectConfig`

File:

```text
drag_conveyor/config/_core.py
```

Add:

```python
@dataclass(slots=True)
class LocalDefectConfig:
    enabled: bool

    canonical_width: int
    canonical_height: int
    min_bar_aspect_ratio: float
    min_endpoint_x_separation_ratio: float

    zone_left: list[float]
    zone_middle: list[float]
    zone_right: list[float]

    shape_threshold: float
    middle_shape_threshold: float
    both_sides_threshold: float
    severe_shape_threshold: float

    missing_weight: float
    extra_weight: float
    min_zone_defect_weighted_pixels: float

    min_template_samples: int
    template_mask_threshold: float
    min_template_area_ratio: float
    max_template_area_ratio: float
    min_baseline_alignment_iou_p50: float
    min_baseline_alignment_iou_p10: float
    max_canonicalize_failure_ratio: float
    min_color_pixels_per_sample: int

    color_enabled: bool
    color_delta_threshold: float
    color_abnormal_ratio_threshold: float
    color_abnormal_ratio_margin: float
    color_delta_p95_threshold: float

    dark_pixel_enabled: bool
    dark_l_threshold: float
    dark_pixel_ratio_threshold: float
    dark_pixel_ratio_margin: float

    erode_mask_iterations: int
    morph_kernel_size: int
    min_alignment_iou: float
    orientation_flip_x: bool
    debug_save_canonical: bool
```

### 6.2 Add to `InspectionConfig`

Current:

```python
@dataclass(slots=True)
class InspectionConfig:
    mode: str
    defect_policy: DefectPolicyConfig
    auto_baseline: AutoBaselineConfig
    average_ratio: AverageRatioConfig
```

Change to:

```python
@dataclass(slots=True)
class InspectionConfig:
    mode: str
    defect_policy: DefectPolicyConfig
    auto_baseline: AutoBaselineConfig
    average_ratio: AverageRatioConfig
    local_defect: LocalDefectConfig
```

### 6.3 Default config values

Add under `inspection` in `config/base_profile.json`:

```json
"local_defect": {
  "enabled": true,

  "canonical_width": 256,
  "canonical_height": 64,
  "min_bar_aspect_ratio": 2.0,
  "min_endpoint_x_separation_ratio": 0.25,

  "zone_left": [0.0, 0.33],
  "zone_middle": [0.33, 0.66],
  "zone_right": [0.66, 1.0],

  "shape_threshold": 0.12,
  "middle_shape_threshold": 0.14,
  "both_sides_threshold": 0.10,
  "severe_shape_threshold": 0.30,

  "missing_weight": 1.0,
  "extra_weight": 0.6,
  "min_zone_defect_weighted_pixels": 30.0,

  "min_template_samples": 30,
  "template_mask_threshold": 0.5,
  "min_template_area_ratio": 0.10,
  "max_template_area_ratio": 0.98,
  "min_baseline_alignment_iou_p50": 0.65,
  "min_baseline_alignment_iou_p10": 0.45,
  "max_canonicalize_failure_ratio": 0.20,
  "min_color_pixels_per_sample": 200,

  "color_enabled": true,
  "color_delta_threshold": 18.0,
  "color_abnormal_ratio_threshold": 0.15,
  "color_abnormal_ratio_margin": 0.05,
  "color_delta_p95_threshold": 28.0,

  "dark_pixel_enabled": true,
  "dark_l_threshold": 90.0,
  "dark_pixel_ratio_threshold": 0.12,
  "dark_pixel_ratio_margin": 0.05,

  "erode_mask_iterations": 1,
  "morph_kernel_size": 3,
  "min_alignment_iou": 0.35,
  "orientation_flip_x": false,
  "debug_save_canonical": false
}
```

Important note about `max_template_area_ratio`:

Because canonicalization warps the bar into a tight `minAreaRect`-based rectangle, a normal solid bar may occupy most of the canonical canvas. Therefore `max_template_area_ratio` must not be set too low. A value around `0.98` is safer than `0.90`.

### 6.4 Config parsing helper for zone pairs

Add a helper in `drag_conveyor/config/_core.py`:

```python
def _required_float_pair(data: dict[str, Any], key: str, label: str) -> list[float]:
    values = _required_list(data, key, label)
    if len(values) != 2:
        raise ProfileError(f"{label} must contain exactly 2 numbers")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in values):
        raise ProfileError(f"{label} must contain only numbers")
    return [float(values[0]), float(values[1])]
```

Use it for:

```python
zone_left=_required_float_pair(local_defect_raw, "zone_left", "inspection.local_defect.zone_left")
zone_middle=_required_float_pair(local_defect_raw, "zone_middle", "inspection.local_defect.zone_middle")
zone_right=_required_float_pair(local_defect_raw, "zone_right", "inspection.local_defect.zone_right")
```

This prevents invalid JSON such as this from producing a raw `TypeError` later:

```json
"zone_left": ["a", "b"]
```

Invalid config must fail with `ProfileError`.

### 6.5 Parsing in `profile_from_dict(...)`

In `profile_from_dict(...)`, add `local_defect` parsing under `inspection`.

First, update the existing unknown-key rejection for the `inspection` section itself. This is mandatory because the project rejects unknown config keys:

```python
_reject_unknown_keys(
    inspection_raw,
    {
        "mode",
        "defect_policy",
        "auto_baseline",
        "average_ratio",
        "local_defect",
    },
    "inspection",
)
```

If this is not updated, profiles containing `inspection.local_defect` will fail before `local_defect_raw` is parsed.

Then parse the nested section:

```python
local_defect_raw = _required_section(
    inspection_raw,
    "local_defect",
    "inspection.local_defect",
)
_reject_unknown_keys(
    local_defect_raw,
    {
        "enabled",
        "canonical_width",
        "canonical_height",
        "min_bar_aspect_ratio",
        "min_endpoint_x_separation_ratio",
        "zone_left",
        "zone_middle",
        "zone_right",
        "shape_threshold",
        "middle_shape_threshold",
        "both_sides_threshold",
        "severe_shape_threshold",
        "missing_weight",
        "extra_weight",
        "min_zone_defect_weighted_pixels",
        "min_template_samples",
        "template_mask_threshold",
        "min_template_area_ratio",
        "max_template_area_ratio",
        "min_baseline_alignment_iou_p50",
        "min_baseline_alignment_iou_p10",
        "max_canonicalize_failure_ratio",
        "min_color_pixels_per_sample",
        "color_enabled",
        "color_delta_threshold",
        "color_abnormal_ratio_threshold",
        "color_abnormal_ratio_margin",
        "color_delta_p95_threshold",
        "dark_pixel_enabled",
        "dark_l_threshold",
        "dark_pixel_ratio_threshold",
        "dark_pixel_ratio_margin",
        "erode_mask_iterations",
        "morph_kernel_size",
        "min_alignment_iou",
        "orientation_flip_x",
        "debug_save_canonical",
    },
    "inspection.local_defect",
)
```

Then instantiate `LocalDefectConfig`.

### 6.6 Validation rules

Add validation in `validate_profile(...)`.

Rules:

```text
canonical_width > 0
canonical_height > 0
min_bar_aspect_ratio >= 1.0
0.0 <= min_endpoint_x_separation_ratio <= 1.0
all zones contain exactly two floats
0.0 <= zone start < zone end <= 1.0
zone_left[0] == 0.0 is recommended, not required
zone_right[1] == 1.0 is recommended, not required
zones should be non-overlapping and ordered: left.end <= middle.start, middle.end <= right.start
shape thresholds > 0
severe_shape_threshold >= shape_threshold
missing_weight >= 0
extra_weight >= 0
min_template_samples > 0
0.0 < template_mask_threshold < 1.0
0.0 < min_template_area_ratio < max_template_area_ratio <= 1.0
0.0 <= min_baseline_alignment_iou_p10 <= min_baseline_alignment_iou_p50 <= 1.0
0.0 <= max_canonicalize_failure_ratio < 1.0
min_color_pixels_per_sample > 0
color_delta_threshold > 0
color thresholds/margins >= 0
0.0 <= min_alignment_iou <= 1.0
morph_kernel_size >= 1 and odd
erode_mask_iterations >= 0
dark_l_threshold within [0, 255]
```

### 6.7 Profile version and migration

Current `PROFILE_VERSION` is:

```python
PROFILE_VERSION = "1.0.0"
```

This feature changes profile schema. Recommended implementation:

```python
PROFILE_VERSION = "1.1.0"
```

Add two defaults with different purposes:

1. `DEFAULT_LOCAL_DEFECT_DICT_FOR_BASE_PROFILE` may have `enabled: True` and is used only as a reference when updating `config/base_profile.json`.
2. `DEFAULT_LOCAL_DEFECT_DICT_FOR_MIGRATION` must have `enabled: False` to preserve old behavior for existing `1.0.0` profiles.

Recommended migration default:

```python
DEFAULT_LOCAL_DEFECT_DICT_FOR_MIGRATION: dict[str, Any] = {
    "enabled": False,
    "canonical_width": 256,
    "canonical_height": 64,
    "min_bar_aspect_ratio": 2.0,
    "min_endpoint_x_separation_ratio": 0.25,
    "zone_left": [0.0, 0.33],
    "zone_middle": [0.33, 0.66],
    "zone_right": [0.66, 1.0],
    "shape_threshold": 0.12,
    "middle_shape_threshold": 0.14,
    "both_sides_threshold": 0.10,
    "severe_shape_threshold": 0.30,
    "missing_weight": 1.0,
    "extra_weight": 0.6,
    "min_zone_defect_weighted_pixels": 30.0,
    "min_template_samples": 30,
    "template_mask_threshold": 0.5,
    "min_template_area_ratio": 0.10,
    "max_template_area_ratio": 0.98,
    "min_baseline_alignment_iou_p50": 0.65,
    "min_baseline_alignment_iou_p10": 0.45,
    "max_canonicalize_failure_ratio": 0.20,
    "min_color_pixels_per_sample": 200,
    "color_enabled": True,
    "color_delta_threshold": 18.0,
    "color_abnormal_ratio_threshold": 0.15,
    "color_abnormal_ratio_margin": 0.05,
    "color_delta_p95_threshold": 28.0,
    "dark_pixel_enabled": True,
    "dark_l_threshold": 90.0,
    "dark_pixel_ratio_threshold": 0.12,
    "dark_pixel_ratio_margin": 0.05,
    "erode_mask_iterations": 1,
    "morph_kernel_size": 3,
    "min_alignment_iou": 0.35,
    "orientation_flip_x": False,
    "debug_save_canonical": False,
}
```

In `migrate_profile_dict(...)`, add migration from `1.0.0` to `1.1.0`:

```python
if requested == (1, 0, 0) and current == (1, 1, 0):
    migrated = copy.deepcopy(raw)
    migrated["profile_version"] = "1.1.0"
    inspection = migrated.setdefault("inspection", {})
    inspection.setdefault(
        "local_defect",
        copy.deepcopy(DEFAULT_LOCAL_DEFECT_DICT_FOR_MIGRATION),
    )
    return migrated
```

This prevents existing profiles from changing behavior unexpectedly. Old profiles will still load and behave as before because `local_defect.enabled` is `false` after migration.

`config/base_profile.json` can explicitly set `inspection.local_defect.enabled` to `true` for new deployments that should use this feature.

If migration is not implemented, this feature becomes a breaking config change. That is not recommended.

### 6.8 Export config class

Update:

```text
drag_conveyor/config/__init__.py
```

Add import:

```python
from ._core import LocalDefectConfig
```

Add to `__all__`:

```python
"LocalDefectConfig",
```

---

## 7. Local defect module design

Create:

```text
drag_conveyor/pipeline/local_defects.py
```

Recommended public API:

```python
__all__ = [
    "LocalDefectBaseline",
    "LocalDefectFeatures",
    "build_local_defect_baseline",
    "analyze_local_defects",
]
```

Recommended functions:

```python
def build_local_defect_baseline(
    *,
    bars: list,
    config: LocalDefectConfig,
) -> LocalDefectBaseline:
    ...
```

```python
def analyze_local_defects(
    *,
    frame: np.ndarray,
    contour_frame: np.ndarray,
    mask_roi: np.ndarray,
    roi_origin_xy: tuple[int, int],
    baseline: LocalDefectBaseline,
    config: LocalDefectConfig,
) -> LocalDefectFeatures:
    ...
```

Private helper functions:

```python
def _mask_roi_to_frame_mask(...): ...
def _canonicalize_bar(...): ...
def _cleanup_mask(...): ...
def _make_zone_slices(...): ...
def _compute_shape_scores(...): ...
def _collect_lab_pixels(...): ...
def _compute_color_scores_from_reference(...): ...
def _validate_local_baseline(...): ...
```

---

## 8. Mask conversion

### 8.1 Convert ROI mask to frame mask

Function:

```python
def _mask_roi_to_frame_mask(
    *,
    mask_roi: np.ndarray,
    roi_origin_xy: tuple[int, int],
    frame_shape_hw: tuple[int, int],
) -> np.ndarray:
    ...
```

Implementation must be defensive and clipped:

```python
def _mask_roi_to_frame_mask(*, mask_roi, roi_origin_xy, frame_shape_hw):
    frame_h, frame_w = frame_shape_hw
    roi_x, roi_y = roi_origin_xy

    mask_frame = np.zeros((frame_h, frame_w), dtype=np.uint8)
    if mask_roi.size == 0:
        return mask_frame

    mask_h, mask_w = mask_roi.shape[:2]
    x1 = max(0, int(roi_x))
    y1 = max(0, int(roi_y))
    x2 = min(frame_w, x1 + mask_w)
    y2 = min(frame_h, y1 + mask_h)

    if x2 <= x1 or y2 <= y1:
        return mask_frame

    mask_src = mask_roi[: y2 - y1, : x2 - x1]
    mask_frame[y1:y2, x1:x2] = (mask_src > 0).astype(np.uint8) * 255
    return mask_frame
```

Even though ROI validation should guarantee valid coordinates, clipping prevents hard-to-debug crashes.

---

## 9. Canonicalization

Canonicalization is the most important part of this feature because left/right must match the image viewer's perspective.

### 9.1 Goal

Convert each bar into a fixed-size coordinate system:

```text
canonical_width = 256
canonical_height = 64
```

After canonicalization:

```text
canonical x=0       -> left side as seen in the source image
canonical x=W-1     -> right side as seen in the source image
canonical y=0       -> top side of the bar in the canonical view
canonical y=H-1     -> bottom side of the bar in the canonical view
```

### 9.2 Required guards

Before computing perspective transform:

```python
if contour_frame is None or len(contour_frame) < 3:
    raise ValueError("local_canonicalize_failed: contour has fewer than 3 points")

rect = cv2.minAreaRect(contour_frame)
(cx, cy), (rw, rh), angle = rect

if rw <= 1.0 or rh <= 1.0:
    raise ValueError("local_canonicalize_failed: rectangle too small")

length = max(rw, rh)
width = min(rw, rh)

if width <= 1e-6 or length / width < config.min_bar_aspect_ratio:
    raise ValueError("local_canonicalize_failed: aspect ratio too low")
```

Reason: if the contour is nearly square or corrupted, left/right localization is unreliable.

Because this project defines left/right by the camera view, also reject bars whose two endpoints do not have enough horizontal separation in the image. This prevents a long but nearly vertical corrupted contour from producing meaningless left/right labels:

```python
endpoint_x_separation = abs(float(right_center[0]) - float(left_center[0]))
if endpoint_x_separation < config.min_endpoint_x_separation_ratio * length:
    raise ValueError("local_canonicalize_failed: endpoint x separation too low")
```

This check is evaluated after the two endpoint centers have been computed as described below.

### 9.3 Preserve left/right by image x-coordinate

Do not rely only on the common `sum/diff` point-ordering trick. It can flip the bar direction in edge cases.

Required approach:

1. Get `box = cv2.boxPoints(rect)`.
2. Determine which two opposite edges are the short edges of the rectangle.
3. The centers of the two short edges represent the two ends of the bar.
4. The end whose center has smaller `x` in the original image is the image-left end.
5. The end whose center has larger `x` is the image-right end.
6. Map image-left end to canonical left.
7. Map image-right end to canonical right.

Conceptual implementation:

```python
def _order_box_points_preserve_image_left_right(
    *,
    box: np.ndarray,
    min_endpoint_x_separation_ratio: float,
) -> np.ndarray:
    # box shape: (4, 2), float32
    # cv2.boxPoints returns points in cyclic order for the rotated rectangle.
    pts = box.astype(np.float32)

    # Compute edge lengths between cyclic neighbors.
    edges = []
    for i in range(4):
        j = (i + 1) % 4
        length = float(np.linalg.norm(pts[j] - pts[i]))
        edges.append((length, i, j))

    # The two shorter opposite edges are the bar ends.
    sorted_edges = sorted(edges, key=lambda e: e[0])
    end_edge_a = sorted_edges[0]

    # Find the opposite edge. For edge (i, j), the opposite edge connects the two remaining points.
    used = {end_edge_a[1], end_edge_a[2]}
    remaining = [idx for idx in range(4) if idx not in used]
    end_edge_b = (float(np.linalg.norm(pts[remaining[1]] - pts[remaining[0]])), remaining[0], remaining[1])

    a_pts = pts[[end_edge_a[1], end_edge_a[2]]]
    b_pts = pts[[end_edge_b[1], end_edge_b[2]]]

    a_center = a_pts.mean(axis=0)
    b_center = b_pts.mean(axis=0)

    endpoint_x_separation = abs(float(a_center[0]) - float(b_center[0]))
    long_edge_length = max(edge[0] for edge in edges)
    if endpoint_x_separation < min_endpoint_x_separation_ratio * long_edge_length:
        raise ValueError("local_canonicalize_failed: endpoint x separation too low")

    if a_center[0] <= b_center[0]:
        left_end = a_pts
        right_end = b_pts
    else:
        left_end = b_pts
        right_end = a_pts

    # Sort points within each end by y to get top/bottom.
    left_sorted = left_end[np.argsort(left_end[:, 1])]
    right_sorted = right_end[np.argsort(right_end[:, 1])]

    left_top, left_bottom = left_sorted[0], left_sorted[1]
    right_top, right_bottom = right_sorted[0], right_sorted[1]

    return np.array(
        [left_top, right_top, right_bottom, left_bottom],
        dtype=np.float32,
    )
```

Destination points:

```python
dst = np.array(
    [
        [0, 0],
        [config.canonical_width - 1, 0],
        [config.canonical_width - 1, config.canonical_height - 1],
        [0, config.canonical_height - 1],
    ],
    dtype=np.float32,
)
```

Then:

```python
M = cv2.getPerspectiveTransform(src, dst)
canonical_crop = cv2.warpPerspective(frame, M, (W, H), flags=cv2.INTER_LINEAR)
canonical_mask = cv2.warpPerspective(mask_frame, M, (W, H), flags=cv2.INTER_NEAREST)
canonical_mask = (canonical_mask > 0).astype(np.uint8) * 255
```

### 9.4 `orientation_flip_x`

Default:

```json
"orientation_flip_x": false
```

Because the requirement is left/right by image view, this should normally remain `false`.

Keep this option as a safety switch if camera mounting or canonicalization output is found to be inverted during field validation:

```python
if config.orientation_flip_x:
    canonical_crop = cv2.flip(canonical_crop, 1)
    canonical_mask = cv2.flip(canonical_mask, 1)
```

### 9.5 Known limitation: full-end loss

Phase 1 canonicalizes a bar using `cv2.minAreaRect()` from the current bar contour.

If an entire end of the bar is missing, the current contour's rectangle can become shorter. When warped back to `256 x 64`, the shortened object may be stretched to full canonical width. This means:

- `length_too_short` should still catch the geometry defect.
- `deform_left` or `deform_right` may not always localize full-end loss correctly.
- Local shape analysis is strongest for local chips, dents, holes, middle deformation, partial side deformation, and color defects.

If the product later requires reliable localization of a fully missing left or right end, phase 2 should align bars against expected geometry or baseline endpoints rather than only the current contour's `minAreaRect`.

---

## 10. Mask cleanup

Function:

```python
def _cleanup_mask(mask: np.ndarray, *, kernel_size: int) -> np.ndarray:
    ...
```

Recommended behavior:

```python
kernel = np.ones((kernel_size, kernel_size), np.uint8)
mask_u8 = (mask > 0).astype(np.uint8) * 255
mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
return (mask_u8 > 0).astype(np.uint8) * 255
```

Do not use an overly large kernel. Default `3` is safer.

---

## 11. Build local baseline

### 11.1 Candidate selection

After `CalibrationEngine().calibrate(records, profile)` succeeds in `_classify_with_auto_baseline(...)`, select geometry-normal candidate bars.

Important terminology: these are **geometry-normal candidates**, not necessarily exact `CalibrationEngine` internal inliers. The current `CalibrationEngine` does not expose per-record inlier indices.

Select candidates using the calibrated percentile thresholds from `calibration_result`.

Do not import the private `_percentile(...)` helper from `pipeline/rules.py` into `batch.py`. Instead, add a local helper in `batch.py` or a small public helper if needed:

```python
def _feature_percentile(
    calibration_result: CalibrationResult,
    feature_name: str,
    percentile_name: str,
) -> float:
    stats = calibration_result.features[feature_name]
    value = getattr(stats, percentile_name, None)
    if value is None:
        raise ValueError(
            f"Calibration feature stats missing percentile: {feature_name}.{percentile_name}"
        )
    return float(value)
```

Then select candidates:

```python
length_min = _feature_percentile(calibration_result, "length", rules.lower_percentile)
length_max = _feature_percentile(calibration_result, "length", rules.upper_percentile)
width_min = _feature_percentile(calibration_result, "width", rules.lower_percentile)
width_max = _feature_percentile(calibration_result, "width", rules.upper_percentile)

candidates = [
    bar for bar in collected
    if length_min <= bar.measurements["length"] <= length_max
    and width_min <= bar.measurements["width"] <= width_max
]
```

Do not modify `CalibrationEngine` just to expose inlier indices in phase 1 unless needed later.

### 11.2 Build template mask

For each candidate:

1. Convert `mask_roi` to `mask_frame`.
2. Canonicalize `source_frame` and `mask_frame`.
3. Cleanup canonical mask.
4. Store normalized mask: `mask.astype(np.float32) / 255.0`.

Then:

```python
template_prob = np.median(np.stack(masks, axis=0), axis=0)
template_mask = (template_prob >= config.template_mask_threshold).astype(np.uint8) * 255
```

Use median, not mean, to reduce contamination from a small number of defective bars that pass geometry checks.

### 11.3 Build LAB baseline

Source frames from `cv2.VideoCapture` are BGR. Always use:

```python
lab = cv2.cvtColor(canonical_crop_bgr, cv2.COLOR_BGR2LAB)
```

Do not use `cv2.COLOR_RGB2LAB`.

Collect LAB pixels from stable mask regions:

```python
stable_mask = (template_mask > 0) & (current_mask > 0)
```

Erode the stable mask to reduce boundary contamination:

```python
if config.erode_mask_iterations > 0:
    stable_mask = cv2.erode(
        stable_mask.astype(np.uint8),
        np.ones((3, 3), np.uint8),
        iterations=config.erode_mask_iterations,
    ).astype(bool)
```

Skip a candidate's color pixels if:

```python
np.count_nonzero(stable_mask) < config.min_color_pixels_per_sample
```

For the full baseline:

```python
all_pixels = np.concatenate(pixel_arrays, axis=0)
lab_median = np.median(all_pixels, axis=0)
lab_mad = np.median(np.abs(all_pixels - lab_median), axis=0)
```

### 11.4 Avoid circular color helper design

Do not implement baseline color statistics by calling a helper that requires a fully constructed `LocalDefectBaseline`.

Instead, use a low-level helper:

```python
def _compute_color_scores_from_reference(
    *,
    canonical_crop_bgr: np.ndarray,
    current_mask: np.ndarray,
    template_mask: np.ndarray,
    lab_median: np.ndarray,
    color_delta_threshold: float,
    dark_l_threshold: float,
    erode_mask_iterations: int,
    min_color_pixels: int,
) -> dict[str, float]:
    ...
```

This helper can be used both while building baseline and while analyzing final bars.

For final per-bar analysis, insufficient stable color pixels must not be reported as `local_canonicalize_failed`. The helper should return zero color metrics plus an explicit flag instead of raising:

```python
if np.count_nonzero(stable_mask) < min_color_pixels:
    return {
        "color_delta_mean": 0.0,
        "color_delta_p95": 0.0,
        "color_abnormal_ratio": 0.0,
        "dark_pixel_ratio": 0.0,
        "local_color_pixels_insufficient": 1.0,
    }
```

For baseline building, insufficient color samples still contribute to baseline stability failure if too few samples have enough color pixels.

### 11.5 Dynamic baseline p95 values

After computing `template_mask` and `lab_median`, compute normal-sample color ratios for each candidate:

```python
candidate_color_abnormal_ratios = []
candidate_dark_ratios = []
```

For each successfully canonicalized candidate, call `_compute_color_scores_from_reference(...)` using the new `template_mask` and `lab_median`.

Then:

```python
color_abnormal_ratio_p95 = float(np.percentile(candidate_color_abnormal_ratios, 95))
dark_ratio_p95 = float(np.percentile(candidate_dark_ratios, 95))
```

These values let rules avoid flagging small normal dark areas such as bolt holes, shadows, or consistent black regions.

### 11.6 Baseline stability checks

Baseline must fail if any required condition fails.

Checks:

1. Enough samples:

```python
samples_used >= config.min_template_samples
```

2. Canonicalization failure ratio:

```python
canonicalize_failure_ratio <= config.max_canonicalize_failure_ratio
```

3. Template area ratio:

```python
config.min_template_area_ratio <= template_area_ratio <= config.max_template_area_ratio
```

4. Baseline alignment quality:

For each candidate mask, compute IoU with `template_mask`:

```python
intersection = np.count_nonzero(candidate_mask & template_mask)
union = np.count_nonzero(candidate_mask | template_mask)
iou = intersection / union if union else 0.0
```

Then:

```python
baseline_alignment_iou_p50 >= config.min_baseline_alignment_iou_p50
baseline_alignment_iou_p10 >= config.min_baseline_alignment_iou_p10
```

5. Color baseline not empty:

```python
total_lab_pixels > 0
```

6. Enough color pixels across baseline:

```python
number_of_candidates_with_enough_color_pixels >= config.min_template_samples
```

If any check fails, raise:

```python
ValueError("local_baseline_not_stable: <specific reason>")
```

Do not silently disable local defect analysis, because the confirmed requirement is that the job should fail if local baseline is insufficient.

---

## 12. Shape scoring

### 12.1 Zones

Create slices from config:

```python
def _make_zone_slices(width: int, config: LocalDefectConfig) -> dict[str, slice]:
    def make(pair: list[float]) -> slice:
        return slice(int(round(pair[0] * width)), int(round(pair[1] * width)))

    return {
        "left": make(config.zone_left),
        "middle": make(config.zone_middle),
        "right": make(config.zone_right),
    }
```

Default zones:

```text
left   = x 0%  -> 33%
middle = x 33% -> 66%
right  = x 66% -> 100%
```

### 12.2 Difference maps

Given:

```python
current = current_mask > 0
template = template_mask > 0
```

Compute:

```python
missing = template & ~current
extra = current & ~template
```

Weighted defect map:

```python
weighted_defect = (
    missing.astype(np.float32) * config.missing_weight
    + extra.astype(np.float32) * config.extra_weight
)
```

Missing regions are usually more important than extra regions, hence default:

```json
"missing_weight": 1.0,
"extra_weight": 0.6
```

### 12.3 Zone scores

For each zone:

```python
zone_defect_weighted_pixels = float(weighted_defect[:, x_slice].sum())
zone_expected_pixels = float(np.count_nonzero(template[:, x_slice]))
zone_score = zone_defect_weighted_pixels / zone_expected_pixels if zone_expected_pixels > 0 else 0.0
```

Store weighted-pixel metrics with explicit names:

- `left_defect_weighted_pixels`
- `middle_defect_weighted_pixels`
- `right_defect_weighted_pixels`

Do not name them `left_defect_pixels`, because the values are weighted floats, not raw pixel counts.

### 12.4 Alignment IoU

Compute:

```python
intersection = np.count_nonzero(current & template)
union = np.count_nonzero(current | template)
shape_alignment_iou = float(intersection / union) if union > 0 else 0.0
```

Also compute:

```python
mask_area_ratio = float(np.count_nonzero(current) / np.count_nonzero(template))
```

Handle division by zero defensively.

### 12.5 Low alignment must not hide severe defects

If `shape_alignment_iou < config.min_alignment_iou`, set:

```python
local_alignment_low = 1.0
```

However, low alignment must not always suppress local reasons. A severely deformed bar may naturally have low IoU.

Rule design:

```text
If alignment is acceptable:
  use normal local thresholds.

If alignment is low:
  do not add normal left/right/middle reasons unless the corresponding score is severe.
  allow severe local defect if max_shape_score >= severe_shape_threshold.
```

This prevents two failures:

1. False positives from bad canonical alignment.
2. False negatives where severe deformation is ignored because IoU is low.

---

## 13. Color scoring

### 13.1 LAB distance

Use LAB on BGR input:

```python
lab = cv2.cvtColor(canonical_crop_bgr, cv2.COLOR_BGR2LAB)
```

Use stable mask:

```python
stable_mask = (template_mask > 0) & (current_mask > 0)
```

Optionally erode:

```python
stable_mask = cv2.erode(...)
```

Get pixels:

```python
pixels = lab[stable_mask].astype(np.float32)
```

Compute weighted LAB distance:

```python
diff = pixels - lab_median.astype(np.float32)
delta = np.sqrt(
    0.25 * diff[:, 0] ** 2
    + diff[:, 1] ** 2
    + diff[:, 2] ** 2
)
```

The `L` channel is down-weighted because it is more sensitive to illumination.

Metrics:

```python
color_delta_mean = float(np.mean(delta))
color_delta_p95 = float(np.percentile(delta, 95))
color_abnormal_ratio = float(np.mean(delta > config.color_delta_threshold))
```

### 13.2 Dark-pixel ratio

Because normal bars are mostly white and color defects are often black/dark, compute:

```python
L = lab[:, :, 0]
dark_pixel_ratio = float(np.mean(L[stable_mask] < config.dark_l_threshold))
```

OpenCV LAB `L` is in `[0, 255]`, so `dark_l_threshold = 90.0` is a reasonable starting point.

### 13.3 Effective thresholds

Use both static config thresholds and baseline dynamic p95 margins. These effective thresholds are valid only when color analysis is enabled and a valid `LocalDefectBaseline` exists:

```python
effective_abnormal_ratio_threshold = max(
    config.color_abnormal_ratio_threshold,
    baseline.color_abnormal_ratio_p95 + config.color_abnormal_ratio_margin,
)

effective_dark_ratio_threshold = max(
    config.dark_pixel_ratio_threshold,
    baseline.dark_ratio_p95 + config.dark_pixel_ratio_margin,
)
```

Then `color_defect` may be added if:

```python
color_abnormal_ratio >= effective_abnormal_ratio_threshold
or color_delta_p95 >= config.color_delta_p95_threshold
or (
    config.dark_pixel_enabled
    and dark_pixel_ratio >= effective_dark_ratio_threshold
)
```

If `config.color_enabled = false`, do not compute or output effective color thresholds in `RuleEngine` thresholds/margins. If `config.color_enabled = true` but the caller does not provide `local_defect_baseline`, `RuleEngine.evaluate(...)` should raise `ValueError("local_defect_baseline is required when local color defect analysis is enabled")` rather than silently using incomplete thresholds.

---

## 14. `analyze_local_defects(...)`

Recommended flow:

```python
def analyze_local_defects(
    *,
    frame,
    contour_frame,
    mask_roi,
    roi_origin_xy,
    baseline,
    config,
):
    try:
        mask_frame = _mask_roi_to_frame_mask(
            mask_roi=mask_roi,
            roi_origin_xy=roi_origin_xy,
            frame_shape_hw=frame.shape[:2],
        )

        canonical_crop, canonical_mask = _canonicalize_bar(
            frame=frame,
            mask_frame=mask_frame,
            contour_frame=contour_frame,
            config=config,
        )

        canonical_mask = _cleanup_mask(
            canonical_mask,
            kernel_size=config.morph_kernel_size,
        )

        shape = _compute_shape_scores(
            current_mask=canonical_mask,
            template_mask=baseline.template_mask,
            zone_slices=baseline.zone_slices,
            config=config,
        )

        color = _compute_color_scores_from_reference(
            canonical_crop_bgr=canonical_crop,
            current_mask=canonical_mask,
            template_mask=baseline.template_mask,
            lab_median=baseline.lab_median,
            color_delta_threshold=config.color_delta_threshold,
            dark_l_threshold=config.dark_l_threshold,
            erode_mask_iterations=config.erode_mask_iterations,
            min_color_pixels=config.min_color_pixels_per_sample,
        )

        return LocalDefectFeatures(
            left_shape_score=shape["left_shape_score"],
            middle_shape_score=shape["middle_shape_score"],
            right_shape_score=shape["right_shape_score"],
            max_shape_score=max(
                shape["left_shape_score"],
                shape["middle_shape_score"],
                shape["right_shape_score"],
            ),
            left_defect_weighted_pixels=shape["left_defect_weighted_pixels"],
            middle_defect_weighted_pixels=shape["middle_defect_weighted_pixels"],
            right_defect_weighted_pixels=shape["right_defect_weighted_pixels"],
            shape_alignment_iou=shape["shape_alignment_iou"],
            mask_area_ratio=shape["mask_area_ratio"],
            local_alignment_low=1.0 if shape["shape_alignment_iou"] < config.min_alignment_iou else 0.0,
            color_delta_mean=color["color_delta_mean"],
            color_delta_p95=color["color_delta_p95"],
            color_abnormal_ratio=color["color_abnormal_ratio"],
            dark_pixel_ratio=color["dark_pixel_ratio"],
            local_color_pixels_insufficient=color.get("local_color_pixels_insufficient", 0.0),
            local_analysis_success=1.0,
            local_canonicalize_failed=0.0,
        )
    except ValueError:
        # This path is reserved for mask conversion or canonicalization failures.
        # Color insufficiency should be handled inside _compute_color_scores_from_reference(...)
        # and should not be mislabeled as canonicalization failure.
        return LocalDefectFeatures.zero_failed()
```

For baseline building, failed canonicalization should contribute to `canonicalize_failure_ratio`. For per-bar final analysis, failed canonicalization should not crash the entire job, but must be visible in metrics.

Do not raise `ValueError` for final per-bar color insufficiency. Use `local_color_pixels_insufficient = 1.0` instead.

---

## 15. Batch integration

### 15.1 Extend imports

In `drag_conveyor/app/batch.py`:

```python
from ..pipeline.local_defects import (
    analyze_local_defects,
    build_local_defect_baseline,
)
```

### 15.2 Collect mask and ROI origin

Update `CollectedBar` construction as described in section 5.1.

### 15.3 Update `_classify_with_auto_baseline(...)`

Recommended high-level structure:

```python
def _classify_with_auto_baseline(
    *,
    collected: list[CollectedBar],
    profile: Profile,
) -> _ClassificationOutcome:
    records = [bar.measurements for bar in collected]

    outcome = CalibrationEngine().calibrate(records, profile)
    if not outcome.success:
        raise ValueError(outcome.reason)

    updated_profile = outcome.updated_profile
    calibration_result = outcome.calibration_result
    if calibration_result is None:
        raise ValueError("missing calibration_result")

    rules = updated_profile.inspection.auto_baseline
    defect_policy = updated_profile.inspection.defect_policy
    local_config = updated_profile.inspection.local_defect

    local_baseline = None
    if local_config.enabled:
        candidates = _select_geometry_normal_candidates(
            collected=collected,
            calibration_result=calibration_result,
            rules=rules,
        )
        local_baseline = build_local_defect_baseline(
            bars=candidates,
            config=local_config,
        )

    rule_engine = RuleEngine()
    bars: list[BarResult] = []

    for bar in collected:
        measurements = dict(bar.measurements)

        if local_config.enabled and local_baseline is not None:
            local_features = analyze_local_defects(
                frame=bar.source_frame,
                contour_frame=bar.contour_frame,
                mask_roi=bar.mask_roi,
                roi_origin_xy=bar.roi_origin_xy,
                baseline=local_baseline,
                config=local_config,
            )
            measurements.update(local_features.to_dict())

        evaluation = rule_engine.evaluate(
            measurements=measurements,
            rules=rules,
            defect_policy=defect_policy,
            calibration_result=calibration_result,
            local_defect_config=local_config if local_config.enabled else None,
            local_defect_baseline=local_baseline,
        )

        bars.append(
            BarResult(
                frame_id=bar.frame_id,
                track_id=bar.track_id,
                result=evaluation.result,
                score=evaluation.score,
                reasons=evaluation.reasons,
                measurements=evaluation.measurements,
                thresholds=evaluation.thresholds,
                margins=evaluation.margins,
                bbox_frame_xyxy=bar.bbox_frame_xyxy,
                contour_frame=bar.contour_frame,
                latency_ms=bar.latency_ms,
                source_frame=bar.source_frame,
            )
        )

    return _ClassificationOutcome(...)
```

### 15.4 Helper `_select_geometry_normal_candidates(...)`

Add a private helper in `batch.py` or `local_defects.py`. Keeping it in `batch.py` is acceptable because it knows about `CollectedBar` and calibration profile rules.

Do not import `CollectedBar` into `local_defects.py` if that creates avoidable coupling. A generic baseline builder can accept objects with required attributes, but explicit typing can be relaxed.

---

## 16. Rule engine changes

File:

```text
drag_conveyor/pipeline/rules.py
```

### 16.1 Signature

Current:

```python
def evaluate(
    self,
    measurements: dict[str, float],
    rules: AutoBaselineConfig,
    defect_policy: DefectPolicyConfig,
    calibration_result: CalibrationResult,
) -> RuleEvaluation:
```

Change to:

```python
def evaluate(
    self,
    measurements: dict[str, float],
    rules: AutoBaselineConfig,
    defect_policy: DefectPolicyConfig,
    calibration_result: CalibrationResult,
    *,
    local_defect_config: LocalDefectConfig | None = None,
    local_defect_baseline: object | None = None,
) -> RuleEvaluation:
```

The first four parameters must remain positional-compatible because current tests and callers use positional calls such as:

```python
engine.evaluate({"length": 100.0, "width": 20.0}, rules, policy, calibration)
```

Only the new local-defect parameters should be keyword-only. Use a specific `LocalDefectBaseline | None` type if importing it does not create cyclic imports.

### 16.2 Preserve geometry policy

Correct logic:

```python
geometry_reasons: list[str] = []
local_reasons: list[str] = []
violated_dimensions = 0

if length < length_min:
    geometry_reasons.append("length_too_short")
    violated_dimensions += 1
elif length > length_max:
    geometry_reasons.append("length_too_long")
    violated_dimensions += 1

if width < width_min:
    geometry_reasons.append("width_too_small")
    violated_dimensions += 1
elif width > width_max:
    geometry_reasons.append("width_too_large")
    violated_dimensions += 1

geometry_defect = violated_dimensions >= defect_policy.min_violated_dimensions
```

Do not mark geometry-only defects as `suspected_defect` merely because `geometry_reasons` is non-empty. Existing tests depend on `min_violated_dimensions`.

### 16.3 Local shape reasons

Read metrics:

```python
left_score = float(measurements.get("left_shape_score", 0.0))
middle_score = float(measurements.get("middle_shape_score", 0.0))
right_score = float(measurements.get("right_shape_score", 0.0))

left_weighted_pixels = float(measurements.get("left_defect_weighted_pixels", 0.0))
middle_weighted_pixels = float(measurements.get("middle_defect_weighted_pixels", 0.0))
right_weighted_pixels = float(measurements.get("right_defect_weighted_pixels", 0.0))

alignment_low = float(measurements.get("local_alignment_low", 0.0)) >= 0.5
```

Normal local rules when alignment is acceptable:

```python
left_bad = (
    left_score >= local_defect_config.shape_threshold
    and left_weighted_pixels >= local_defect_config.min_zone_defect_weighted_pixels
)
right_bad = (
    right_score >= local_defect_config.shape_threshold
    and right_weighted_pixels >= local_defect_config.min_zone_defect_weighted_pixels
)
middle_bad = (
    middle_score >= local_defect_config.middle_shape_threshold
    and middle_weighted_pixels >= local_defect_config.min_zone_defect_weighted_pixels
)
```

Low-alignment severe fallback:

```python
if alignment_low:
    left_bad = left_score >= local_defect_config.severe_shape_threshold
    right_bad = right_score >= local_defect_config.severe_shape_threshold
    middle_bad = middle_score >= local_defect_config.severe_shape_threshold
```

Both-sides detection must be computed independently so that `both_sides_threshold` is meaningful. With the default values, `both_sides_threshold = 0.10` is intentionally lower than `shape_threshold = 0.12`, allowing mild but simultaneous defects on both sides to be reported as `deform_both_sides`.

```python
both_sides_bad = (
    left_score >= local_defect_config.both_sides_threshold
    and right_score >= local_defect_config.both_sides_threshold
    and left_weighted_pixels >= local_defect_config.min_zone_defect_weighted_pixels
    and right_weighted_pixels >= local_defect_config.min_zone_defect_weighted_pixels
)
```

For low-alignment severe fallback, use severe thresholds for all side decisions:

```python
if alignment_low:
    left_bad = left_score >= local_defect_config.severe_shape_threshold
    right_bad = right_score >= local_defect_config.severe_shape_threshold
    middle_bad = middle_score >= local_defect_config.severe_shape_threshold
    both_sides_bad = left_bad and right_bad
```

Reason generation:

```python
if both_sides_bad:
    local_reasons.append("deform_both_sides")
elif left_bad:
    local_reasons.append("deform_left")
elif right_bad:
    local_reasons.append("deform_right")

if middle_bad:
    local_reasons.append("deform_middle")
```

Because multi-label is required, `deform_middle` is independent from side reasons.

### 16.4 Color reason

Color reasoning requires both `local_defect_config` and `local_defect_baseline`. Do not use baseline-derived thresholds when the baseline is missing.

```python
color_abnormal_ratio = float(measurements.get("color_abnormal_ratio", 0.0))
color_delta_p95 = float(measurements.get("color_delta_p95", 0.0))
dark_pixel_ratio = float(measurements.get("dark_pixel_ratio", 0.0))

effective_abnormal_ratio_threshold: float | None = None
effective_dark_ratio_threshold: float | None = None

if local_defect_config.color_enabled:
    if local_defect_baseline is None:
        raise ValueError(
            "local_defect_baseline is required when local color defect analysis is enabled"
        )

    effective_abnormal_ratio_threshold = max(
        local_defect_config.color_abnormal_ratio_threshold,
        local_defect_baseline.color_abnormal_ratio_p95
        + local_defect_config.color_abnormal_ratio_margin,
    )
    effective_dark_ratio_threshold = max(
        local_defect_config.dark_pixel_ratio_threshold,
        local_defect_baseline.dark_ratio_p95
        + local_defect_config.dark_pixel_ratio_margin,
    )

    color_bad = (
        color_abnormal_ratio >= effective_abnormal_ratio_threshold
        or color_delta_p95 >= local_defect_config.color_delta_p95_threshold
        or (
            local_defect_config.dark_pixel_enabled
            and dark_pixel_ratio >= effective_dark_ratio_threshold
        )
    )
    if color_bad:
        local_reasons.append("color_defect")
```

When `color_enabled = false`, keep `effective_abnormal_ratio_threshold` and `effective_dark_ratio_threshold` as `None`; do not add color thresholds/margins to the `RuleEvaluation` output.

### 16.5 Final result

Correct final decision:

```python
local_defect = bool(local_reasons)
reasons = geometry_reasons + local_reasons

result = "suspected_defect" if geometry_defect or local_defect else "normal"
```

This preserves old geometry policy and allows local defects to independently mark a bar defective.

### 16.6 Score

Existing score is geometry-only:

```python
geometry_score = violated_dimensions / float(defect_policy.score_dimension_count)
```

New score should reflect the strongest evidence while remaining in `[0, 1]`:

```python
geometry_score = violated_dimensions / float(defect_policy.score_dimension_count)
shape_score_norm = 0.0
color_score_norm = 0.0

if local_defect_config is not None:
    shape_threshold = max(local_defect_config.shape_threshold, 1e-6)
    shape_score_norm = max(left_score, middle_score, right_score) / shape_threshold

    if (
        local_defect_config.color_enabled
        and effective_abnormal_ratio_threshold is not None
    ):
        color_threshold = max(effective_abnormal_ratio_threshold, 1e-6)
        color_score_norm = color_abnormal_ratio / color_threshold

score = min(1.0, max(geometry_score, shape_score_norm, color_score_norm))
```

If local config is disabled, score should remain effectively current geometry score behavior.

### 16.7 Thresholds and margins output

Add local thresholds to `thresholds` only when local defect config is provided. Add color thresholds only when color analysis is enabled and the effective thresholds have been computed:

```python
thresholds.update({
    "shape_threshold": local_defect_config.shape_threshold,
    "middle_shape_threshold": local_defect_config.middle_shape_threshold,
    "both_sides_threshold": local_defect_config.both_sides_threshold,
    "severe_shape_threshold": local_defect_config.severe_shape_threshold,
})

if (
    local_defect_config.color_enabled
    and effective_abnormal_ratio_threshold is not None
    and effective_dark_ratio_threshold is not None
):
    thresholds.update({
        "effective_color_abnormal_ratio_threshold": effective_abnormal_ratio_threshold,
        "color_delta_p95_threshold": local_defect_config.color_delta_p95_threshold,
        "effective_dark_pixel_ratio_threshold": effective_dark_ratio_threshold,
    })
```

Add local margins for debugging if useful. As with thresholds, do not write color margins when color effective thresholds are unavailable:

```python
margins.update({
    "left_shape_margin": local_defect_config.shape_threshold - left_score,
    "middle_shape_margin": local_defect_config.middle_shape_threshold - middle_score,
    "right_shape_margin": local_defect_config.shape_threshold - right_score,
})

if (
    local_defect_config.color_enabled
    and effective_abnormal_ratio_threshold is not None
    and effective_dark_ratio_threshold is not None
):
    margins.update({
        "color_abnormal_ratio_margin": effective_abnormal_ratio_threshold - color_abnormal_ratio,
        "dark_pixel_ratio_margin": effective_dark_ratio_threshold - dark_pixel_ratio,
    })
```

---

## 17. Server summary changes

File:

```text
server/worker.py
```

In `_build_summary(...)`, extend defect entries with local metrics:

```python
{
    "track_id": bar.track_id,
    "frame_id": bar.frame_id,
    "result": bar.result,
    "score": bar.score,
    "reasons": bar.reasons,
    "length": bar.measurements.get("length", 0.0),
    "width": bar.measurements.get("width", 0.0),
    "left_shape_score": bar.measurements.get("left_shape_score", 0.0),
    "middle_shape_score": bar.measurements.get("middle_shape_score", 0.0),
    "right_shape_score": bar.measurements.get("right_shape_score", 0.0),
    "max_shape_score": bar.measurements.get("max_shape_score", 0.0),
    "shape_alignment_iou": bar.measurements.get("shape_alignment_iou", 0.0),
    "color_delta_mean": bar.measurements.get("color_delta_mean", 0.0),
    "color_delta_p95": bar.measurements.get("color_delta_p95", 0.0),
    "color_abnormal_ratio": bar.measurements.get("color_abnormal_ratio", 0.0),
    "dark_pixel_ratio": bar.measurements.get("dark_pixel_ratio", 0.0),
    "local_analysis_success": bar.measurements.get("local_analysis_success", 0.0),
    "local_canonicalize_failed": bar.measurements.get("local_canonicalize_failed", 0.0),
    "snapshot_key": snapshot_key,
}
```

This is backward compatible because existing keys remain unchanged.

---

## 18. Debug artifacts

Phase 1 default behavior:

- Existing defect snapshots remain unchanged unless tests are updated.
- Do not write canonical debug artifacts by default.
- Do not upload canonical debug artifacts to R2 by default.

If `debug_save_canonical = true`, optional local files can be written under a run-local debug directory, for example:

```text
defect_snapshots/<run_id>/debug/
  track_000017_frame_000123_canonical_mask.png
  track_000017_frame_000123_template_mask.png
  track_000017_frame_000123_defect_map.png
```

Keep this disabled by default.

---

## 19. Failure behavior

### 19.1 Job-level failures

For `auto_baseline + local_defect.enabled = true`, these should fail classification and therefore return `BatchInspectionResult(success=False, failure_reason=...)`:

```text
local_baseline_not_stable: not enough geometry-normal candidates
local_baseline_not_stable: not enough template samples
local_baseline_not_stable: canonicalize failure ratio too high
local_baseline_not_stable: template area ratio out of range
local_baseline_not_stable: template IoU p50 below threshold
local_baseline_not_stable: template IoU p10 below threshold
local_baseline_not_stable: color baseline empty
local_baseline_not_stable: not enough color samples
```

### 19.2 Per-bar local analysis failure

A single bar failing local canonicalization should not crash the entire job after baseline has been built.

It should produce:

```python
LocalDefectFeatures.zero_failed()
```

This ensures the output records that local analysis failed:

```json
{
  "local_analysis_success": 0.0,
  "local_canonicalize_failed": 1.0
}
```

Do not automatically add a defect reason for per-bar canonicalization failure in phase 1 unless later required by production policy.

---

## 20. Backward compatibility

The feature should preserve existing behavior when disabled:

```json
"inspection": {
  "local_defect": {
    "enabled": false
  }
}
```

Given the full config parser requires all keys, `enabled = false` still lives inside a complete `local_defect` object.

When disabled:

- No local baseline is built.
- No local metrics are added.
- `RuleEngine` should behave like before, including positional calls and `min_violated_dimensions` semantics.
- Existing `min_violated_dimensions` behavior must remain intact.
- Existing tests for `rules.py` must still pass.

Migration should allow old `1.0.0` profiles to load by adding default local defect config and changing version to `1.1.0`.

---

## 21. Testing plan

The repository currently uses `unittest`. Do not assume `pytest` is available unless it is explicitly added to `pyproject.toml`.

Recommended test command:

```bash
uv run python -m unittest discover
```

If `uv` is unavailable in an environment:

```bash
python -m unittest discover
```

### 21.1 Existing tests that must continue to pass

Run all existing tests after implementation:

```bash
uv run python -m unittest discover
```

Especially protect:

- `tests.test_profile_rules`
- tests for `RuleEngine.evaluate(...)`
- tests for snapshot output
- tests for average-ratio mode

### 21.2 New config tests

Add tests for:

1. Loading profile `1.1.0` with `inspection.local_defect`.
2. Migrating profile `1.0.0` to `1.1.0` and adding default local defect config.
3. Rejecting unknown keys under `inspection.local_defect`.
4. Rejecting invalid zone pairs:
   - wrong length
   - non-numeric values
   - booleans
   - values outside `[0, 1]`
   - reversed pair
5. Rejecting invalid `morph_kernel_size` when even.
6. Rejecting invalid `max_template_area_ratio <= min_template_area_ratio`.
7. Exporting `LocalDefectConfig` from `drag_conveyor.config`.

### 21.3 New local defect unit tests

Create:

```text
tests/test_local_defects.py
```

Use synthetic images and masks.

Required tests:

1. `test_mask_roi_to_frame_mask_clips_to_frame`
2. `test_canonicalize_preserves_image_left_right_for_horizontal_bar`
3. `test_canonicalize_preserves_image_left_right_for_rotated_bar`
4. `test_canonicalize_rejects_low_aspect_ratio_contour`
5. `test_left_notch_produces_highest_left_score`
6. `test_right_notch_produces_highest_right_score`
7. `test_middle_hole_produces_highest_middle_score`
8. `test_both_sides_notches_produce_left_and_right_scores`
9. `test_template_median_resists_small_defect_contamination`
10. `test_lab_color_black_patch_increases_abnormal_ratio`
11. `test_dark_patch_increases_dark_pixel_ratio`
12. `test_zero_failed_contains_failure_flags`

### 21.4 Rule engine tests

Extend current `rules.py` tests.

Required tests:

1. Existing `min_violated_dimensions = 2` behavior remains unchanged.
   - One geometry reason should not mark result as `suspected_defect` if policy requires two dimensions.

2. Local defect independently marks result as `suspected_defect`.
   - Geometry normal + `left_shape_score` above threshold -> `deform_left`, result `suspected_defect`.

3. Multi-label local reasons.
   - Left + right + middle + color can all be present.

4. Low alignment severe fallback.
   - Low IoU + mild score should not produce local reason.
   - Low IoU + severe score should produce local reason.

5. Color threshold with dynamic baseline p95.
   - If baseline p95 is high, effective threshold increases.

6. Disabled local config preserves previous behavior.

### 21.5 Batch integration tests

Add or extend batch-level tests with stubs/mocks where appropriate:

1. `auto_baseline + local_defect.enabled = true` calls local baseline builder.
2. Local baseline failure returns `BatchInspectionResult(success=False)` with specific `failure_reason`.
3. `average_ratio` does not call local defect analysis.
4. `local_defect.enabled = false` does not call local defect analysis.
5. Defect summary includes local metrics when available.

### 21.6 Manual validation checklist

Use real videos if available.

Validate:

1. Normal video:
   - low local shape scores
   - low color abnormal ratio
   - no false `color_defect`

2. Left defect video/image:
   - `left_shape_score` highest
   - reason includes `deform_left`

3. Right defect video/image:
   - `right_shape_score` highest
   - reason includes `deform_right`

4. Middle defect:
   - `middle_shape_score` highest
   - reason includes `deform_middle`

5. Both sides:
   - both left and right scores high
   - reason includes `deform_both_sides`

6. Black/dark defect:
   - `color_abnormal_ratio` and/or `dark_pixel_ratio` high
   - reason includes `color_defect`

7. Orientation:
   - left/right match the image viewer perspective.
   - If inverted, set `orientation_flip_x = true` only after confirming the cause.

---

## 22. Performance and memory considerations

### 22.1 Memory

Current pipeline already stores `source_frame=frame.copy()` for each collected bar. Adding `mask_roi.copy()` increases memory usage.

Approximate cost:

```text
source_frame 1280x720x3 uint8 ~= 2.6 MB per bar
mask_roi 1280x720 uint8 ~= 0.9 MB per bar if ROI is full frame
```

For many bars, memory can grow quickly. This is already a batch-processing pipeline, not real-time streaming.

Phase 1 accepts this cost. If memory becomes an issue later, phase 2 can store cropped regions or compressed masks only.

### 22.2 Runtime

Additional per-bar operations:

- perspective warp for frame crop
- perspective warp for mask
- morphology
- LAB conversion
- pixel statistics

This is acceptable for batch processing. Keep debug image saving disabled by default.

### 22.3 Avoid repeated unnecessary work

During baseline building, canonical crops/masks are produced for candidates. It is acceptable to recompute per-bar analysis in phase 1 for simplicity.

If performance becomes an issue, cache canonical data by `track_id` or by `(frame_id, track_id)`.

---

## 23. Coding workflow requirements from `AGENTS.md`

This repository is indexed by GitNexus. Follow the project-specific workflow before editing code.

### 23.1 Before modifying symbols

Before editing any function, class, or method, run GitNexus impact analysis for the target symbol and record the blast radius.

Required impact analysis targets for this feature include at minimum:

```text
profile_from_dict
validate_profile
migrate_profile_dict
InspectionConfig
RuleEngine.evaluate
run_batch_inspection
_classify_with_auto_baseline
_classify_collected_bars
CollectedBar
_build_summary
```

If a target has HIGH or CRITICAL impact risk, warn the user before proceeding.

### 23.2 Before committing

Run:

```text
gitnexus_detect_changes()
```

Confirm that affected symbols and flows match the intended scope.

### 23.3 After commit

If needed, refresh the GitNexus index:

```bash
npx gitnexus analyze
```

If embeddings existed before, preserve them:

```bash
npx gitnexus analyze --embeddings
```

---

## 24. Implementation task checklist

### Task 1: Config schema and migration

Files:

```text
drag_conveyor/config/_core.py
drag_conveyor/config/__init__.py
config/base_profile.json
```

Work:

- Add `LocalDefectConfig` dataclass.
- Add `local_defect` to `InspectionConfig`.
- Add `DEFAULT_LOCAL_DEFECT_DICT_FOR_MIGRATION` with `enabled: false`.
- Bump `PROFILE_VERSION` to `1.1.0`.
- Add migration from `1.0.0` to `1.1.0` that preserves old behavior by disabling local defect on migrated profiles.
- Add `_required_float_pair(...)`.
- Parse and validate `inspection.local_defect`.
- Export `LocalDefectConfig` in `config/__init__.py`.
- Update `base_profile.json`.

Acceptance:

- Existing profiles load via migration.
- New base profile validates.
- Invalid local config fails with `ProfileError`.

### Task 2: Local defect module

File:

```text
drag_conveyor/pipeline/local_defects.py
```

Work:

- Add dataclasses:
  - `LocalDefectFeatures`
  - `LocalDefectBaseline`
- Add public functions:
  - `build_local_defect_baseline(...)`
  - `analyze_local_defects(...)`
- Add helpers:
  - mask conversion
  - canonicalization preserving image left/right
  - cleanup
  - zone slicing
  - shape scoring
  - LAB/dark color scoring
  - baseline validation
- Add `__all__`.

Acceptance:

- Synthetic tests verify left/right/middle scores.
- Canonicalization rejects bad contours.
- Color/dark tests pass.

### Task 3: Batch integration

File:

```text
drag_conveyor/app/batch.py
```

Work:

- Extend `CollectedBar` with `mask_roi` and `roi_origin_xy`.
- Store mask and ROI origin when collecting triggered bars.
- Add geometry-normal candidate selection.
- Build local baseline in `_classify_with_auto_baseline(...)` only when enabled.
- Merge local feature metrics into `measurements`.
- Keep `average_ratio` behavior unchanged.
- Fail job if local baseline fails.

Acceptance:

- `average_ratio` tests still pass.
- `auto_baseline` local failure returns `success=False`.
- Local metrics are present when enabled.

### Task 4: Rule engine

File:

```text
drag_conveyor/pipeline/rules.py
```

Work:

- Extend `evaluate(...)` signature.
- Preserve geometry `min_violated_dimensions` behavior.
- Add local shape reasons.
- Add color reason.
- Add dynamic color thresholds using baseline p95.
- Update score calculation.
- Add thresholds/margins outputs.

Acceptance:

- Existing geometry rule tests pass.
- New multi-label local tests pass.
- Local defect can mark geometry-normal bar as suspected defect.

### Task 5: Server summary

File:

```text
server/worker.py
```

Work:

- Add local metrics to defect summary.
- Do not implement report generation.

Acceptance:

- Summary remains backward compatible.
- Existing summary keys remain unchanged.

### Task 6: Tests

Files:

```text
tests/test_local_defects.py
tests/test_profile_rules.py or existing config/rules test files
```

Work:

- Add config/migration tests.
- Add local defect unit tests.
- Extend `RuleEngine` tests.
- Add batch integration tests if existing test infrastructure supports it.

Acceptance:

```bash
uv run python -m unittest discover
```

passes.

---

## 25. Reason labels

Backend should keep stable English reason codes:

```python
REASON_LABELS = {
    "length_too_short": "BAR TOO SHORT",
    "length_too_long": "BAR TOO LONG",
    "width_too_small": "BAR TOO NARROW",
    "width_too_large": "BAR TOO WIDE",
    "deform_left": "LEFT-SIDE DEFORMATION",
    "deform_right": "RIGHT-SIDE DEFORMATION",
    "deform_middle": "MIDDLE DEFORMATION",
    "deform_both_sides": "BOTH-SIDES DEFORMATION",
    "color_defect": "COLOR DEFECT",
}
```

Vietnamese labels can be applied manually or at UI/report time:

```python
VI_REASON_LABELS = {
    "deform_left": "BIẾN DẠNG BÊN TRÁI",
    "deform_right": "BIẾN DẠNG BÊN PHẢI",
    "deform_middle": "BIẾN DẠNG Ở GIỮA",
    "deform_both_sides": "BIẾN DẠNG 2 BÊN",
    "color_defect": "BIẾN DẠNG MÀU",
}
```

Do not store Vietnamese labels as backend reason codes.

---

## 26. Acceptance criteria

The feature is complete when all criteria below are true.

### Functional acceptance

- Existing YOLO segmentation pipeline still works.
- Existing geometry detection still works.
- `average_ratio` mode is unchanged.
- `auto_baseline + local_defect.enabled = true` builds local baseline.
- Job fails if local baseline is unstable.
- Local defect can produce:
  - `deform_left`
  - `deform_right`
  - `deform_middle`
  - `deform_both_sides`
  - `color_defect`
- One bar can contain multiple local reasons.
- Left/right are correct from image viewer perspective.
- Color defect is based on LAB masked-pixel comparison and dark-pixel ratio.

### Technical acceptance

- `LocalDefectConfig` is parsed, validated, migrated, and exported.
- `RuleEngine` preserves `min_violated_dimensions` semantics.
- Canonicalization has contour, size, and aspect-ratio guards.
- Local baseline has stability checks beyond sample count.
- `max_template_area_ratio` is not too low for tight canonical masks.
- Per-bar local analysis failure is visible via metrics.
- No report generator is added.
- `/api/runtime-config` is not expanded in phase 1.
- Existing tests pass.
- New tests cover config, canonicalization, shape scoring, color scoring, rules, and integration.

### Process acceptance

- GitNexus impact analysis is run before editing affected symbols.
- `gitnexus_detect_changes()` is run before committing.
- No HIGH/CRITICAL GitNexus warning is ignored.
- `uv run python -m unittest discover` passes.

---

## 27. Future phase ideas

Do not implement these in phase 1.

Potential future improvements:

1. Classifier-based defect classification on canonical crops.
2. More robust reference alignment using expected bar geometry instead of current contour `minAreaRect`.
3. Missing-bar detection using expected pitch and belt cycle length.
4. Encoder/speed-assisted full conveyor loop detection.
5. Long-term baseline persistence across videos.
6. UI controls for local-defect thresholds.
7. R2 upload for canonical debug artifacts.
8. More nuanced color categories such as `black_stain`, `burn_mark`, `oil_contamination`.

---

## 28. Final implementation summary

The phase-1 implementation should be understood as:

```text
Keep YOLO segmentation as the detector.
Use its mask to normalize each detected bar.
Build a per-video template mask and color baseline from geometry-normal bars.
Compare each bar against that template by left/middle/right zones.
Compare each bar's masked LAB pixels against the baseline white-bar color.
Add local reasons in RuleEngine without breaking existing geometry policy.
Fail the job if local baseline is required but unstable.
```

The most important implementation constraints are:

1. Preserve image-view left/right during canonicalization.
2. Preserve existing `min_violated_dimensions` semantics.
3. Validate the local baseline strongly before using it.
4. Treat LAB color and dark-pixel ratio as baseline-relative, not absolute black/white checks only.
5. Keep phase 1 limited to `auto_baseline`.
