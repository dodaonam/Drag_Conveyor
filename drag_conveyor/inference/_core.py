from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from ..config import ModelConfig

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ModelDiagnostics:
    model_path: str
    model_hash: str
    providers: list[str]
    input_names: list[str]
    input_shapes: list[list[int | str]]
    output_names: list[str]
    output_shapes: list[list[int | str]]


@dataclass(frozen=True, slots=True)
class PreprocessResult:
    tensor: np.ndarray
    roi_shape: tuple[int, int]
    roi_origin_xy: tuple[int, int]
    scale: float
    pad_x: float
    pad_y: float
    input_size: int


@dataclass(slots=True)
class Detection:
    class_id: int
    score: float
    bbox_roi_xyxy: tuple[float, float, float, float]
    bbox_frame_xyxy: tuple[float, float, float, float]
    centroid_frame_xy: tuple[float, float]
    mask_roi: np.ndarray
    contour_frame: np.ndarray


class InferenceEngine(Protocol):
    def load(self, model_path: str, model_spec: ModelConfig) -> ModelDiagnostics: ...

    def infer(self, input_tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...

    def close(self) -> None: ...


class OnnxRuntimeEngine:
    def __init__(self, providers: list[str] | None = None) -> None:
        self.providers = providers or ["CPUExecutionProvider"]
        self._session: Any | None = None
        self._input_name: str | None = None
        self._model_spec: ModelConfig | None = None

    def load(self, model_path: str, model_spec: ModelConfig) -> ModelDiagnostics:
        try:
            import onnxruntime as ort  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "onnxruntime is required to load ONNX models. "
                "Install with `uv add onnxruntime` in production runtime env."
            ) from exc

        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        self._session = ort.InferenceSession(str(path), providers=self.providers)
        self._input_name = self._session.get_inputs()[0].name
        self._model_spec = model_spec

        input_names = [x.name for x in self._session.get_inputs()]
        input_shapes = [list(x.shape) for x in self._session.get_inputs()]
        output_names = [x.name for x in self._session.get_outputs()]
        output_shapes = [list(x.shape) for x in self._session.get_outputs()]

        self._validate_model(model_spec, input_shapes, output_shapes)

        diagnostics = ModelDiagnostics(
            model_path=str(path),
            model_hash=_sha256(path),
            providers=list(self._session.get_providers()),
            input_names=input_names,
            input_shapes=input_shapes,
            output_names=output_names,
            output_shapes=output_shapes,
        )

        LOGGER.info("Model loaded: %s", diagnostics)
        return diagnostics

    def _validate_model(
        self,
        model_spec: ModelConfig,
        input_shapes: list[list[int | str]],
        output_shapes: list[list[int | str]],
    ) -> None:
        if model_spec.backend.lower() != "onnxruntime":
            raise ValueError(f"Unsupported backend for OnnxRuntimeEngine: {model_spec.backend}")
        if model_spec.task != "segmentation":
            raise ValueError(f"Expected segmentation task, got: {model_spec.task}")

        if not input_shapes:
            raise ValueError("Model has no input")
        first_input = input_shapes[0]
        if len(first_input) != 4:
            raise ValueError(f"Expected NCHW input shape, got: {first_input}")

        h = int(first_input[2]) if isinstance(first_input[2], int) else None
        w = int(first_input[3]) if isinstance(first_input[3], int) else None
        if h is not None and w is not None and (h != model_spec.input_size or w != model_spec.input_size):
            raise ValueError(
                f"Model input shape mismatch. Model={h}x{w}, ModelSpec={model_spec.input_size}x{model_spec.input_size}"
            )

        if len(output_shapes) < 2:
            raise ValueError("Expected at least 2 outputs for YOLO segmentation (dets + proto)")
        det_shape = output_shapes[0]
        proto_shape = output_shapes[1]

        if len(det_shape) != 3:
            raise ValueError(f"Unexpected detection output shape: {det_shape}")
        if len(proto_shape) != 4:
            raise ValueError(f"Unexpected proto output shape: {proto_shape}")

    def infer(self, input_tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._session is None or self._input_name is None:
            raise RuntimeError("Model session is not loaded")

        outputs = self._session.run(None, {self._input_name: input_tensor})
        if len(outputs) < 2:
            raise RuntimeError(f"Expected 2 outputs from model, got {len(outputs)}")
        return outputs[0], outputs[1]

    def close(self) -> None:
        self._session = None
        self._input_name = None
        self._model_spec = None


def preprocess_roi(
    roi_bgr: np.ndarray,
    roi_origin_xy: tuple[int, int],
    input_size: int,
    normalize: bool = True,
    color_format: str = "RGB",
) -> PreprocessResult:
    h, w = roi_bgr.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError("ROI must have positive shape")

    scale = min(input_size / w, input_size / h)
    resized_w = int(round(w * scale))
    resized_h = int(round(h * scale))

    resized = cv2.resize(roi_bgr, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((input_size, input_size, 3), 114, dtype=np.uint8)

    pad_x = (input_size - resized_w) / 2.0
    pad_y = (input_size - resized_h) / 2.0
    x1 = int(round(pad_x))
    y1 = int(round(pad_y))
    canvas[y1 : y1 + resized_h, x1 : x1 + resized_w] = resized

    if color_format.upper() == "RGB":
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    tensor = canvas.astype(np.float32)
    if normalize:
        tensor /= 255.0
    tensor = np.transpose(tensor, (2, 0, 1))[None, ...]

    return PreprocessResult(
        tensor=tensor,
        roi_shape=(h, w),
        roi_origin_xy=roi_origin_xy,
        scale=scale,
        pad_x=pad_x,
        pad_y=pad_y,
        input_size=input_size,
    )


def postprocess_segmentation(
    det_output: np.ndarray,
    proto_output: np.ndarray,
    preprocess: PreprocessResult,
    model_spec: ModelConfig,
    conf_threshold: float,
    iou_threshold: float,
) -> list[Detection]:
    from .yolo_seg_postprocess import YoloSegPostprocessor

    postprocessor = YoloSegPostprocessor()
    detections = postprocessor.decode(
        det_output=det_output,
        proto_output=proto_output,
        preprocess=preprocess,
        output_format=model_spec.output_format,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        detection_factory=Detection,
    )
    return detections  # type: ignore[return-value]


def _model_box_to_roi_xyxy(
    box_xyxy: np.ndarray,
    preprocess: PreprocessResult,
) -> tuple[float, float, float, float]:
    x1 = (float(box_xyxy[0]) - preprocess.pad_x) / preprocess.scale
    y1 = (float(box_xyxy[1]) - preprocess.pad_y) / preprocess.scale
    x2 = (float(box_xyxy[2]) - preprocess.pad_x) / preprocess.scale
    y2 = (float(box_xyxy[3]) - preprocess.pad_y) / preprocess.scale

    roi_h, roi_w = preprocess.roi_shape
    x1 = float(np.clip(x1, 0, roi_w - 1))
    y1 = float(np.clip(y1, 0, roi_h - 1))
    x2 = float(np.clip(x2, 0, roi_w))
    y2 = float(np.clip(y2, 0, roi_h))

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def _undo_letterbox_mask(mask_input: np.ndarray, preprocess: PreprocessResult) -> np.ndarray:
    roi_h, roi_w = preprocess.roi_shape
    resized_w = int(round(roi_w * preprocess.scale))
    resized_h = int(round(roi_h * preprocess.scale))
    x1 = int(round(preprocess.pad_x))
    y1 = int(round(preprocess.pad_y))
    crop = mask_input[y1 : y1 + resized_h, x1 : x1 + resized_w]
    if crop.size == 0:
        return np.zeros((roi_h, roi_w), dtype=np.float32)
    return cv2.resize(crop, (roi_w, roi_h), interpolation=cv2.INTER_LINEAR)


def _nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []

    while order.size > 0:
        i = int(order[0])
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter_w = np.maximum(0.0, xx2 - xx1)
        inter_h = np.maximum(0.0, yy2 - yy1)
        inter = inter_w * inter_h

        union = areas[i] + areas[order[1:]] - inter
        iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)

        remaining = np.where(iou <= iou_threshold)[0]
        order = order[remaining + 1]

    return np.array(keep, dtype=np.int64)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))
