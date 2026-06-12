from __future__ import annotations

import json
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from drag_conveyor.config import profile_from_dict
from drag_conveyor.inference import PreprocessResult, postprocess_segmentation

ROOT = Path(__file__).resolve().parents[1]


def _base_profile():
    raw = json.loads((ROOT / "config" / "base_profile.json").read_text(encoding="utf-8"))
    return profile_from_dict(raw)


def _preprocess() -> PreprocessResult:
    return PreprocessResult(
        tensor=np.zeros((1, 3, 8, 8), dtype=np.float32),
        roi_shape=(8, 8),
        roi_origin_xy=(0, 0),
        scale=1.0,
        pad_x=0.0,
        pad_y=0.0,
        input_size=8,
    )


def _outputs(*, class_id: int = 0) -> tuple[np.ndarray, np.ndarray]:
    row = np.zeros((38,), dtype=np.float32)
    row[:6] = [2.0, 2.0, 6.0, 6.0, 0.9, float(class_id)]
    det_output = row.reshape(1, 1, -1)
    proto_output = np.zeros((1, 32, 8, 8), dtype=np.float32)
    return det_output, proto_output


class PostprocessConfigTests(unittest.TestCase):
    def test_target_class_ids_filter_detections(self) -> None:
        profile = _base_profile()
        det_output, proto_output = _outputs(class_id=1)

        filtered = postprocess_segmentation(
            det_output,
            proto_output,
            _preprocess(),
            profile.model,
            postprocess_config=profile.model.postprocess,
        )
        allowed = postprocess_segmentation(
            det_output,
            proto_output,
            _preprocess(),
            profile.model,
            postprocess_config=replace(profile.model.postprocess, target_class_ids=[1]),
        )

        self.assertEqual(filtered, [])
        self.assertEqual(len(allowed), 1)

    def test_mask_threshold_and_bbox_crop_affect_mask(self) -> None:
        profile = _base_profile()
        det_output, proto_output = _outputs()

        strict_threshold = postprocess_segmentation(
            det_output,
            proto_output,
            _preprocess(),
            profile.model,
            postprocess_config=replace(profile.model.postprocess, mask_threshold=0.6),
        )
        cropped = postprocess_segmentation(
            det_output,
            proto_output,
            _preprocess(),
            profile.model,
            postprocess_config=replace(
                profile.model.postprocess,
                mask_threshold=0.5,
                crop_mask_to_bbox=True,
            ),
        )
        uncropped = postprocess_segmentation(
            det_output,
            proto_output,
            _preprocess(),
            profile.model,
            postprocess_config=replace(
                profile.model.postprocess,
                mask_threshold=0.5,
                crop_mask_to_bbox=False,
            ),
        )

        self.assertEqual(strict_threshold, [])
        self.assertEqual(len(cropped), 1)
        self.assertEqual(int(cropped[0].mask_roi.sum()), 16)
        self.assertEqual(len(uncropped), 1)
        self.assertEqual(int(uncropped[0].mask_roi.sum()), 64)


if __name__ == "__main__":
    unittest.main()
