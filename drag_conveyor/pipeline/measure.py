from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class Measurements:
    length: float
    width: float

    def to_dict(self) -> dict[str, float]:
        return {
            "length": self.length,
            "width": self.width,
        }


def measure_contour(contour: np.ndarray) -> Measurements:
    rect = cv2.minAreaRect(contour)
    w, h = rect[1]
    length = float(max(w, h))
    width = float(min(w, h))
    return Measurements(length=length, width=width)

__all__ = ["Measurements", "measure_contour"]
