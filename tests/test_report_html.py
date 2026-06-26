from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVER_DIR = ROOT / "server"
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

import report  # noqa: E402


class ReportHtmlTests(unittest.TestCase):
    def test_format_ict(self) -> None:
        self.assertEqual(report.format_ict("2026-06-11T11:47:00+00:00"), "11/06/2026 · 18:47")

    def test_sanitize_filename_part(self) -> None:
        self.assertEqual(report.sanitize_filename_part("M515A"), "M515A")
        self.assertEqual(report.sanitize_filename_part("Băng / 1"), "Bang_1")

    def test_report_filename(self) -> None:
        # export time (ICT) + machine name
        name = report.report_filename("M515A", "2026-06-11T11:47:00+00:00")
        self.assertEqual(name, "20260611_184700_M515A.pdf")

    def test_build_html_contains_sections_and_totals(self) -> None:
        data = {
            "total_bars": 100,
            "defect_count": 10,
            "defect_rate": 10.0,
            "defects_by_type": {
                "bent_left": [{"track_id": 1, "snapshot_key": "k1"}],
                "bent_right": [],
                "bent_both": [],
                "broken": [{"track_id": 2, "snapshot_key": "k2"}],
            },
        }
        meta = {"inspector_name": "Trần Văn Trường", "conveyor_name": "M515A", "datetime_str": "11/06/2026 · 18:47"}
        html = report.build_html(data, meta, images_by_track={1: "data:image/jpeg;base64,AAA"})
        self.assertIn("PHIẾU KIỂM TRA XÍCH TẢI", html)
        self.assertIn("Trần Văn Trường", html)
        self.assertIn("M515A", html)
        self.assertIn("100", html)            # total
        self.assertIn("10.0%", html)          # rate
        self.assertIn("Biến dạng bên trái", html)
        self.assertIn("Gãy", html)
        self.assertNotIn("Biến dạng bên phải", html)  # empty section hidden
        self.assertIn("data:image/jpeg;base64,AAA", html)

    def test_build_html_zero_defects(self) -> None:
        data = {"total_bars": 50, "defect_count": 0, "defect_rate": 0.0,
                "defects_by_type": {t: [] for t in report.ALLOWED_DEFECT_TYPES}}
        meta = {"inspector_name": "A", "conveyor_name": "B", "datetime_str": "x"}
        html = report.build_html(data, meta, images_by_track={})
        self.assertIn("Không phát hiện cánh lỗi", html)


if __name__ == "__main__":
    unittest.main()
