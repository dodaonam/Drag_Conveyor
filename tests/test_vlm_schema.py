from __future__ import annotations

import unittest

from pydantic import ValidationError

from drag_conveyor.vlm.schema import DefectReason


class VlmSchemaTests(unittest.TestCase):
    def test_normal_is_accepted_as_defect_type(self) -> None:
        reason = DefectReason(bar_id="b1", defect_type="normal")
        self.assertEqual(reason.defect_type, "normal")

    def test_known_defect_types_still_accepted(self) -> None:
        for label in ("bent_left", "bent_right", "bent_both", "broken", "other"):
            self.assertEqual(DefectReason(bar_id="b", defect_type=label).defect_type, label)

    def test_unknown_defect_type_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            DefectReason(bar_id="b", defect_type="exploded")


if __name__ == "__main__":
    unittest.main()
