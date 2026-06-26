from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "server" / "static"


class ReportFrontendContractTests(unittest.TestCase):
    def test_index_has_save_button(self) -> None:
        html = (STATIC / "index.html").read_text(encoding="utf-8")
        self.assertIn('id="btn-save-report"', html)

    def test_appjs_posts_to_report_endpoint(self) -> None:
        js = (STATIC / "app.js").read_text(encoding="utf-8")
        self.assertIn("/report", js)
        self.assertIn("collectCorrections", js)
        self.assertIn("saveReport", js)


if __name__ == "__main__":
    unittest.main()
