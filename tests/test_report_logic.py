from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVER_DIR = ROOT / "server"
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

import report  # noqa: E402


def _summary() -> dict:
    return {
        "total_bars": 10,
        "defects": [
            {"track_id": 1, "defect_type": "bent_left", "snapshot_key": "results/j/snapshots/defects/a.jpg"},
            {"track_id": 2, "defect_type": "broken", "snapshot_key": "results/j/snapshots/defects/b.jpg"},
        ],
        "normals": [
            {"track_id": 3, "snapshot_key": "results/j/snapshots/normals/c.jpg"},
        ],
    }


class ReportLogicTests(unittest.TestCase):
    def test_groups_defects_and_counts(self) -> None:
        data = report.build_report_data(_summary(), corrections=[])
        self.assertEqual(data["total_bars"], 10)
        self.assertEqual(data["defect_count"], 2)
        self.assertEqual(data["defect_rate"], 20.0)
        self.assertEqual(len(data["defects_by_type"]["bent_left"]), 1)
        self.assertEqual(len(data["defects_by_type"]["broken"]), 1)

    def test_correction_changes_type(self) -> None:
        data = report.build_report_data(_summary(), corrections=[{"track_id": 2, "defect_type": "bent_right"}])
        self.assertEqual(len(data["defects_by_type"]["broken"]), 0)
        self.assertEqual(len(data["defects_by_type"]["bent_right"]), 1)

    def test_correction_normal_removes_defect(self) -> None:
        data = report.build_report_data(_summary(), corrections=[{"track_id": 1, "defect_type": "normal"}])
        self.assertEqual(data["defect_count"], 1)

    def test_promote_normal_to_defect(self) -> None:
        data = report.build_report_data(_summary(), corrections=[{"track_id": 3, "defect_type": "bent_both"}])
        self.assertEqual(data["defect_count"], 3)
        self.assertEqual(len(data["defects_by_type"]["bent_both"]), 1)

    def test_uncorrected_other_defect_raises(self) -> None:
        s = _summary()
        s["defects"].append({"track_id": 5, "defect_type": "other", "snapshot_key": "results/j/x.jpg"})
        with self.assertRaises(report.ReportError):
            report.build_report_data(s, corrections=[])

    def test_invalid_correction_type_raises(self) -> None:
        with self.assertRaises(report.ReportError):
            report.build_report_data(_summary(), corrections=[{"track_id": 1, "defect_type": "exploded"}])

    def test_unknown_track_id_raises(self) -> None:
        with self.assertRaises(report.ReportError):
            report.build_report_data(_summary(), corrections=[{"track_id": 999, "defect_type": "broken"}])


class GeometryV2ReportLogicTests(unittest.TestCase):
    def test_broken_location_uses_legacy_broken_report_group(self) -> None:
        summary = {
            "total_bars": 1,
            "inspection_mode": "geometry_v2",
            "defects": [{"track_id": 7, "vision_status": "broken_center"}],
            "normals": [],
        }
        data = report.build_report_data(summary, corrections=[])
        self.assertEqual(len(data["defects_by_type"]["broken"]), 1)
        self.assertEqual(data["resolved_statuses"], {7: "broken_center"})

    def test_uncertain_requires_human_correction_before_export(self) -> None:
        summary = {"total_bars": 1, "defects": [{"track_id": 7, "vision_status": "uncertain"}], "normals": []}
        with self.assertRaisesRegex(report.ReportError, "still unclassified"):
            report.build_report_data(summary, corrections=[])
        data = report.build_report_data(summary, corrections=[{"track_id": 7, "defect_type": "broken_center"}])
        self.assertEqual(data["resolved_statuses"], {7: "broken_center"})


if __name__ == "__main__":
    unittest.main()
