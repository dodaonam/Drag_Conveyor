from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class Measurements:
    area: float
    length: float
    width: float
    aspect_ratio: float

    def to_dict(self) -> dict[str, float]:
        return {
            "area": self.area,
            "length": self.length,
            "width": self.width,
            "aspect_ratio": self.aspect_ratio,
        }


def measure_contour(contour: np.ndarray) -> Measurements:
    area = float(max(cv2.contourArea(contour), 0.0))

    rect = cv2.minAreaRect(contour)
    w, h = rect[1]
    length = float(max(w, h))
    width = float(min(w, h))
    aspect_ratio = float(length / width) if width > 1e-6 else float("inf")

    return Measurements(
        area=area,
        length=length,
        width=width,
        aspect_ratio=aspect_ratio,
    )

__all__ = ["Measurements", "measure_contour"]
