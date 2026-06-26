from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SERVER_DIR = ROOT / "server"
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

import report  # noqa: E402


def _weasyprint_available() -> bool:
    try:
        import weasyprint  # noqa: F401
        return True
    except Exception:
        return False


def _summary() -> dict:
    return {
        "total_bars": 4,
        "defects": [{"track_id": 1, "defect_type": "bent_left", "snapshot_key": "results/j/d/a.jpg"}],
        "normals": [{"track_id": 2, "snapshot_key": "results/j/n/b.jpg"}],
    }


_META = {"inspector_name": "A", "conveyor_name": "M515A", "datetime_str": "11/06/2026 · 18:47"}


class ReportSaveTests(unittest.TestCase):
    def test_save_writes_pdf_and_returns_filename(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(report, "render_pdf", return_value=b"%PDF-1.7 fake"):
                name = report.save_report(
                    summary=_summary(), corrections=[], meta=_META, reports_dir=Path(tmp),
                    fetch_image=lambda key: b"img",
                    exported_at_iso="2026-06-11T11:47:00+00:00",
                )
            self.assertEqual(name, "20260611_184700_M515A.pdf")
            saved = Path(tmp) / name
            self.assertTrue(saved.exists())
            self.assertTrue(saved.read_bytes().startswith(b"%PDF"))

    def test_save_dedupes_on_collision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(report, "render_pdf", return_value=b"%PDF x"):
                n1 = report.save_report(_summary(), [], _META, Path(tmp), lambda k: None,
                                        exported_at_iso="2026-06-11T11:47:00+00:00")
                n2 = report.save_report(_summary(), [], _META, Path(tmp), lambda k: None,
                                        exported_at_iso="2026-06-11T11:47:00+00:00")
            self.assertNotEqual(n1, n2)
            self.assertTrue(n2.endswith("_2.pdf"))

    def test_save_propagates_report_error(self) -> None:
        s = _summary()
        s["defects"].append({"track_id": 9, "defect_type": "other", "snapshot_key": "results/j/x.jpg"})
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(report.ReportError):
                report.save_report(s, [], _META, Path(tmp), lambda k: None,
                                   exported_at_iso="2026-06-11T11:47:00+00:00")

    @unittest.skipUnless(_weasyprint_available(), "WeasyPrint + system libs not installed")
    def test_render_pdf_smoke(self) -> None:
        pdf = report.render_pdf("<html><body><h1>hi</h1></body></html>")
        self.assertTrue(pdf.startswith(b"%PDF"))


if __name__ == "__main__":
    unittest.main()
