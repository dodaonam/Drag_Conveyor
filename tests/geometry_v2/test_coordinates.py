from __future__ import annotations

import unittest

import numpy as np

from drag_conveyor.geometry_v2 import ChainCoordinates, GeometryValidationError, Point, Roi, mad, quantile_type7
from drag_conveyor.geometry_v2.coordinates import rasterized_crop_bbox


class ChainCoordinateTests(unittest.TestCase):
    def test_vertical_centerline_projects_pixel_centers_and_side_signs(self) -> None:
        coordinates = ChainCoordinates.create(Roi(0, 0, 100, 100), Point(50, 0), Point(50, 100))

        self.assertEqual(coordinates.project_pixel_center(50, 10), (10.5, 0.5))
        self.assertEqual(coordinates.project(Point(40, 10)), (10.0, -10.0))
        self.assertEqual(coordinates.project(Point(60, 10)), (10.0, 10.0))
        self.assertEqual((coordinates.s_min, coordinates.s_max), (0.0, 100.0))

    def test_rolled_centerline_uses_exact_line_rectangle_interval(self) -> None:
        coordinates = ChainCoordinates.create(Roi(0, 0, 100, 100), Point(40, 0), Point(50, 100))

        self.assertAlmostEqual(coordinates.s_min, 0.0)
        self.assertAlmostEqual(coordinates.s_max, np.hypot(10.0, 100.0))
        self.assertAlmostEqual(coordinates.project(Point(45, 50))[1], 0.0)

    def test_reversed_endpoints_are_canonicalized(self) -> None:
        coordinates = ChainCoordinates.create(Roi(0, 0, 100, 100), Point(50, 100), Point(50, 0))

        self.assertEqual(coordinates.centerline.top, Point(50, 0))
        self.assertEqual(coordinates.centerline.bottom, Point(50, 100))

    def test_available_fov_intersects_all_roi_boundaries(self) -> None:
        coordinates = ChainCoordinates.create(Roi(0, 0, 100, 100), Point(50, 0), Point(50, 100))

        self.assertEqual(coordinates.available_side_extents(20.0), (50.0, 50.0))

        off_center = ChainCoordinates.create(Roi(0, 0, 100, 100), Point(20, 0), Point(30, 100))
        left, right = off_center.available_side_extents(50.0)
        self.assertGreater(left, 0.0)
        self.assertGreater(right, left)

    def test_trigger_strip_is_expressed_in_chain_coordinate(self) -> None:
        coordinates = ChainCoordinates.create(Roi(0, 0, 100, 100), Point(50, 0), Point(50, 100))
        strip = coordinates.trigger_strip(0.5, 0.2)

        self.assertEqual((strip.top_s, strip.bottom_s), (40.0, 60.0))
        polygon = coordinates.trigger_strip_polygon(strip)
        self.assertEqual({point.y for point in polygon}, {40.0, 60.0})
        self.assertEqual({point.x for point in polygon}, {0.0, 100.0})

    def test_invalid_centerline_is_rejected(self) -> None:
        with self.assertRaises(GeometryValidationError):
            ChainCoordinates.create(Roi(0, 0, 100, 100), Point(50, 20), Point(50, 20))

    def test_type7_quantile_and_mad_are_deterministic(self) -> None:
        values = [0.0, 10.0, 20.0, 30.0]

        self.assertEqual(quantile_type7(values, 0.25), 7.5)
        self.assertEqual(quantile_type7(values, 0.5), 15.0)
        self.assertEqual(mad(values), 10.0)

    def test_rasterized_crop_bbox_clips_then_uses_half_open_floor(self) -> None:
        bbox = rasterized_crop_bbox((-1.0, 2.9, 10.99, 20.0), roi_width=10, roi_height=10)

        self.assertEqual(bbox, (0, 2, 10, 10))


if __name__ == "__main__":
    unittest.main()
