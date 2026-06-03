from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import cv2
import numpy as np


class OutputFormatLike(Protocol):
    type: str
    box_format: str
    has_objectness: bool
    class_encoding: str
    num_classes: int
    num_mask_coeffs: int


@dataclass(frozen=True, slots=True)
class OutputInspection:
    det_shape: tuple[int, ...]
    proto_shape: tuple[int, ...]
    normalized_det_shape: tuple[int, ...]
    expected_feature_dim: int
    proto_channels: int


class YoloSegPostprocessor:
    """Decode YOLO segmentation outputs into contour-ready detections."""

    def inspect_outputs(
        self,
        det_output: np.ndarray,
        proto_output: np.ndarray,
        output_format: OutputFormatLike,
    ) -> OutputInspection:
        raw = self.normalize_detection_output_shape(det_output, output_format=output_format)
        if proto_output.ndim != 4:
            raise ValueError(f"Expected proto output shape [B,C,H,W], got {proto_output.shape}")
        return OutputInspection(
            det_shape=tuple(det_output.shape),
            proto_shape=tuple(proto_output.shape),
            normalized_det_shape=tuple(raw.shape),
            expected_feature_dim=_expected_feature_dim(output_format),
            proto_channels=int(proto_output.shape[1]),
        )

    def normalize_detection_output_shape(
        self,
        det_output: np.ndarray,
        *,
        output_format: OutputFormatLike,
    ) -> np.ndarray:
        if det_output.ndim != 3:
            raise ValueError(f"Expected detection output with 3 dims, got {det_output.shape}")

        expected_feature_dim = _expected_feature_dim(output_format)
        b, d1, d2 = det_output.shape
        if d2 == expected_feature_dim:
            return det_output
        if d1 == expected_feature_dim:
            return np.transpose(det_output, (0, 2, 1))

        raise ValueError(
            "Unexpected detection output shape. "
            f"shape={det_output.shape}, expected_feature_dim={expected_feature_dim}"
        )

    def decode(
        self,
        *,
        det_output: np.ndarray,
        proto_output: np.ndarray,
        preprocess,
        output_format: OutputFormatLike,
        conf_threshold: float,
        iou_threshold: float,
        detection_factory: Callable[..., object],
    ) -> list[object]:
        raw_det = self.normalize_detection_output_shape(det_output, output_format=output_format)
        if proto_output.ndim != 4:
            raise ValueError(f"Expected proto output shape [B,C,H,W], got {proto_output.shape}")

        raw = raw_det[0].astype(np.float32)
        proto = proto_output[0].astype(np.float32)
        proto_channels, mh, mw = proto.shape
        if proto_channels != int(output_format.num_mask_coeffs):
            raise ValueError(
                "Proto channels mismatch with model.output_format.num_mask_coeffs. "
                f"proto={proto_channels} spec={int(output_format.num_mask_coeffs)}"
            )

        decoded = self._decode_rows(
            raw=raw,
            output_format=output_format,
            conf_threshold=conf_threshold,
        )
        if decoded is None:
            return []

        boxes_model, class_ids, scores, mask_coeff = decoded
        if boxes_model.shape[0] == 0:
            return []

        # V1 inspects white_bar class (class_id=0).
        if int(output_format.num_classes) > 1:
            keep_cls = class_ids == 0
            boxes_model = boxes_model[keep_cls]
            class_ids = class_ids[keep_cls]
            scores = scores[keep_cls]
            mask_coeff = mask_coeff[keep_cls]
            if boxes_model.shape[0] == 0:
                return []

        keep_nms = _nms_xyxy(boxes_model, scores, iou_threshold)
        boxes_model = boxes_model[keep_nms]
        class_ids = class_ids[keep_nms]
        scores = scores[keep_nms]
        mask_coeff = mask_coeff[keep_nms]

        proto_flat = proto.reshape(proto_channels, -1)
        roi_h, roi_w = preprocess.roi_shape
        roi_x, roi_y = preprocess.roi_origin_xy

        detections: list[object] = []
        for i in range(len(scores)):
            box_model = boxes_model[i]
            score = float(scores[i])
            class_id = int(class_ids[i])
            coeff = mask_coeff[i]
            if coeff.shape[0] != proto_channels:
                raise ValueError(
                    f"Mask coeff channels mismatch: coeff={coeff.shape[0]} proto={proto_channels}"
                )

            mask_small = _sigmoid(coeff @ proto_flat).reshape(mh, mw)
            mask_input = cv2.resize(
                mask_small,
                (preprocess.input_size, preprocess.input_size),
                interpolation=cv2.INTER_LINEAR,
            )

            mask_roi_prob = _undo_letterbox_mask(mask_input, preprocess)
            mask_roi = (mask_roi_prob >= 0.5).astype(np.uint8)

            x1r, y1r, x2r, y2r = _model_box_to_roi_xyxy(box_model, preprocess)
            x1i, y1i, x2i, y2i = int(x1r), int(y1r), int(x2r), int(y2r)
            bbox_mask = np.zeros_like(mask_roi)
            bbox_mask[y1i:y2i, x1i:x2i] = mask_roi[y1i:y2i, x1i:x2i]
            mask_roi = bbox_mask

            contours, _ = cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            valid_contours = [contour for contour in contours if cv2.contourArea(contour) > 1.0]
            if not valid_contours:
                continue

            contour_roi = max(valid_contours, key=cv2.contourArea)
            x, y, w_box, h_box = cv2.boundingRect(contour_roi)
            x1r = float(x)
            y1r = float(y)
            x2r = float(x + w_box)
            y2r = float(y + h_box)

            contour_frame = contour_roi.copy().astype(np.float32)
            contour_frame[:, 0, 0] += roi_x
            contour_frame[:, 0, 1] += roi_y

            moments = cv2.moments(contour_roi)
            if moments["m00"] > 0:
                cx = float(moments["m10"] / moments["m00"])
                cy = float(moments["m01"] / moments["m00"])
            else:
                cx = float((x1r + x2r) / 2.0)
                cy = float((y1r + y2r) / 2.0)

            bbox_frame = (x1r + roi_x, y1r + roi_y, x2r + roi_x, y2r + roi_y)
            detections.append(
                detection_factory(
                    class_id=class_id,
                    score=score,
                    bbox_roi_xyxy=(x1r, y1r, x2r, y2r),
                    bbox_frame_xyxy=bbox_frame,
                    centroid_frame_xy=(cx + roi_x, cy + roi_y),
                    mask_roi=mask_roi,
                    contour_frame=contour_frame,
                )
            )

        return detections

    def _decode_rows(
        self,
        *,
        raw: np.ndarray,
        output_format: OutputFormatLike,
        conf_threshold: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        feature_dim = raw.shape[1]
        expected_feature_dim = _expected_feature_dim(output_format)
        if feature_dim != expected_feature_dim:
            raise ValueError(
                "Unexpected detection feature dimension. "
                f"raw={feature_dim}, expected={expected_feature_dim}"
            )

        num_classes = int(output_format.num_classes)
        num_mask_coeffs = int(output_format.num_mask_coeffs)
        has_objectness = bool(output_format.has_objectness)

        boxes = raw[:, :4].astype(np.float32)
        offset = 4
        if has_objectness:
            objectness = raw[:, offset].astype(np.float32)
            offset += 1
        else:
            objectness = np.ones((raw.shape[0],), dtype=np.float32)

        if output_format.class_encoding == "scores":
            class_scores = raw[:, offset : offset + num_classes].astype(np.float32)
            offset += num_classes
            class_ids = class_scores.argmax(axis=1).astype(np.int32)
            class_conf = class_scores[np.arange(class_scores.shape[0]), class_ids]
        elif output_format.class_encoding == "id":
            class_ids = raw[:, offset].astype(np.int32)
            offset += 1
            class_conf = np.ones((raw.shape[0],), dtype=np.float32)
        else:
            raise ValueError(f"Unsupported class_encoding: {output_format.class_encoding}")

        mask_coeff = raw[:, offset : offset + num_mask_coeffs].astype(np.float32)
        if mask_coeff.shape[1] != num_mask_coeffs:
            raise ValueError(
                "Mask coefficient dimension mismatch. "
                f"coeff={mask_coeff.shape[1]} spec={num_mask_coeffs}"
            )

        scores = (objectness * class_conf).astype(np.float32)

        boxes_xyxy = _convert_boxes_to_xyxy(boxes, output_format.box_format)

        keep = scores >= float(conf_threshold)
        if not np.any(keep):
            return None

        return (
            boxes_xyxy[keep],
            class_ids[keep],
            scores[keep],
            mask_coeff[keep],
        )


def _expected_feature_dim(output_format: OutputFormatLike) -> int:
    class_dims = int(output_format.num_classes) if output_format.class_encoding == "scores" else 1
    return (
        4
        + (1 if bool(output_format.has_objectness) else 0)
        + class_dims
        + int(output_format.num_mask_coeffs)
    )


def _convert_boxes_to_xyxy(boxes: np.ndarray, box_format: str) -> np.ndarray:
    out = boxes.copy().astype(np.float32)
    if box_format == "xyxy":
        return out
    if box_format == "xywh":
        cx = boxes[:, 0].astype(np.float32)
        cy = boxes[:, 1].astype(np.float32)
        w = np.maximum(boxes[:, 2].astype(np.float32), 0.0)
        h = np.maximum(boxes[:, 3].astype(np.float32), 0.0)
        out[:, 0] = cx - w / 2.0
        out[:, 1] = cy - h / 2.0
        out[:, 2] = cx + w / 2.0
        out[:, 3] = cy + h / 2.0
        return out
    raise ValueError(f"Unsupported box format: {box_format}")


def _model_box_to_roi_xyxy(
    box_xyxy: np.ndarray,
    preprocess,
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


def _undo_letterbox_mask(mask_input: np.ndarray, preprocess) -> np.ndarray:
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


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))
