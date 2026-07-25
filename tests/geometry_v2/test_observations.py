from __future__ import annotations

import unittest

import numpy as np

from drag_conveyor.geometry_v2 import ChainCoordinates, Point, Roi
from drag_conveyor.geometry_v2.observations import (
    ComponentExtractionConfig,
    SideHint,
    deduplicate_components,
    extract_components,
)


def _coordinates() -> ChainCoordinates:
    return ChainCoordinates.create(Roi(0, 0, 100, 100), Point(50, 0), Point(50, 100))


def _config() -> ComponentExtractionConfig:
    return ComponentExtractionConfig(chain_band_half_width=4.0, boundary_margin_px=2)


def _extract(mask: np.ndarray, detection: str = "d01"):
    return extract_components(
        mask,
        source_frame_id=1,
        source_detection_id=detection,
        source_detection_score=0.9,
        source_model_output_row_index=3,
        class_id=0,
        coordinates=_coordinates(),
        config=_config(),
        model_bbox_roi_xyxy=(0.0, 0.0, 100.0, 100.0),
    )


class ComponentExtractionTests(unittest.TestCase):
    def test_connected_component_is_preserved_without_morphology(self) -> None:
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[40:60, 20:80] = 1

        components = _extract(mask)

        self.assertEqual(len(components), 1)
        self.assertEqual(components[0].side_hint, SideHint.SPANS_BOTH)
        self.assertEqual(components[0].area_px, 1200)

    def test_noise_smaller_than_bootstrap_threshold_is_excluded(self) -> None:
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[40:50, 10:20] = 1
        mask[70:72, 70:72] = 1

        components = _extract(mask)

        self.assertEqual(len(components), 1)
        self.assertEqual(components[0].area_px, 100)

    def test_left_and_right_components_are_not_deduplicated_just_for_near_anchor(self) -> None:
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[40:60, 10:30] = 1
        mask[40:60, 70:90] = 1

        components = _extract(mask)
        deduplicated = deduplicate_components(components, coordinates=_coordinates(), config=_config())

        self.assertEqual(len(deduplicated), 2)
        self.assertEqual({component.side_hint for component in deduplicated}, {SideHint.LEFT, SideHint.RIGHT})

    def test_containment_duplicate_is_aliased_even_when_iou_is_low(self) -> None:
        full = np.zeros((100, 100), dtype=np.uint8)
        full[35:65, 10:40] = 1
        subset = np.zeros((100, 100), dtype=np.uint8)
        subset[40:55, 15:35] = 1

        components = (*_extract(full, "d01"), *_extract(subset, "d02"))
        deduplicated = deduplicate_components(components, coordinates=_coordinates(), config=_config())

        self.assertEqual(len(deduplicated), 1)
        self.assertEqual(len(deduplicated[0].duplicate_aliases), 1)
        self.assertGreaterEqual(deduplicated[0].duplicate_aliases[0].ios, 0.85)
        self.assertLess(deduplicated[0].duplicate_aliases[0].iou, 0.70)

    def test_component_ids_do_not_depend_on_connected_component_label_order(self) -> None:
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[70:90, 10:30] = 1
        mask[20:40, 10:30] = 1

        components = _extract(mask)

        self.assertEqual([component.component_id for component in components], ["f000000001-d01-c01", "f000000001-d01-c02"])
        self.assertLess(components[0].s_anchor, components[1].s_anchor)


if __name__ == "__main__":
    unittest.main()
