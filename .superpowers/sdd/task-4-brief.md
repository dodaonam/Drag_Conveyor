# Task 4 brief (from approved plan)

### Task 4: `report.py` — render + save orchestrator + WeasyPrint dependency

**Files:**
- Modify: `server/report.py`
- Modify: `pyproject.toml` (add `weasyprint`)
- Modify: `README.md` (system-lib note)
- Test: `tests/test_report_save.py` (create)

**Interfaces:**
- Consumes: `build_report_data`, `build_html`, `report_filename` (Task 2–3).
- Produces:
  - `render_pdf(html: str) -> bytes` (lazy-imports WeasyPrint).
  - `save_report(summary, corrections, meta, job_id, created_at_iso, reports_dir, fetch_image) -> str`
    where `fetch_image: Callable[[str], bytes | None]` takes a `snapshot_key` and returns JPEG bytes or None. Returns the saved filename. Writes the PDF into `reports_dir` (created if missing), de-duplicating the name with `_2`, `_3`… on collision.

- [ ] **Step 1: Write the failing test**

Create `tests/test_report_save.py`:

```python
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
                    summary=_summary(), corrections=[], meta=_META, job_id="job123",
                    created_at_iso="2026-06-11T11:47:00+00:00", reports_dir=Path(tmp),
                    fetch_image=lambda key: b"img",
                )
            self.assertEqual(name, "XT-100_M515A_20260611_1847_job123.pdf")
            saved = Path(tmp) / name
            self.assertTrue(saved.exists())
            self.assertTrue(saved.read_bytes().startswith(b"%PDF"))

    def test_save_dedupes_on_collision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(report, "render_pdf", return_value=b"%PDF x"):
                n1 = report.save_report(_summary(), [], _META, "job123",
                                        "2026-06-11T11:47:00+00:00", Path(tmp), lambda k: None)
                n2 = report.save_report(_summary(), [], _META, "job123",
                                        "2026-06-11T11:47:00+00:00", Path(tmp), lambda k: None)
            self.assertNotEqual(n1, n2)
            self.assertTrue(n2.endswith("_2.pdf"))

    def test_save_propagates_report_error(self) -> None:
        s = _summary()
        s["defects"].append({"track_id": 9, "defect_type": "other", "snapshot_key": "results/j/x.jpg"})
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(report.ReportError):
                report.save_report(s, [], _META, "j", "2026-06-11T11:47:00+00:00", Path(tmp), lambda k: None)

    @unittest.skipUnless(_weasyprint_available(), "WeasyPrint + system libs not installed")
    def test_render_pdf_smoke(self) -> None:
        pdf = report.render_pdf("<html><body><h1>hi</h1></body></html>")
        self.assertTrue(pdf.startswith(b"%PDF"))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m unittest discover -s tests -p "test_report_save.py" -v`
Expected: FAIL — `AttributeError: module 'report' has no attribute 'save_report'`.

- [ ] **Step 3: Add render + save to `server/report.py`**

Add `from typing import Callable` to the imports, then append:

```python
def render_pdf(html: str) -> bytes:
    from weasyprint import HTML  # lazy: avoid import cost / system-lib requirement at module load
    return HTML(string=html).write_pdf()


def _image_data_uri(data: bytes) -> str:
    import base64
    return "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")


def save_report(summary, corrections, meta, job_id, created_at_iso, reports_dir,
                fetch_image: "Callable[[str], bytes | None]") -> str:
    report_data = build_report_data(summary, corrections)

    images_by_track: dict[int, str | None] = {}
    for bars in report_data["defects_by_type"].values():
        for bar in bars:
            key = bar.get("snapshot_key")
            data = fetch_image(key) if key else None
            images_by_track[int(bar["track_id"])] = _image_data_uri(data) if data else None

    html = build_html(report_data, meta, images_by_track)
    pdf_bytes = render_pdf(html)

    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    base = report_filename(meta["conveyor_name"], created_at_iso, job_id)
    target = reports_dir / base
    if target.exists():
        stem, suffix = base[:-4], ".pdf"
        i = 2
        while (reports_dir / f"{stem}_{i}{suffix}").exists():
            i += 1
        target = reports_dir / f"{stem}_{i}{suffix}"
    target.write_bytes(pdf_bytes)
    return target.name
```

- [ ] **Step 4: Run logic tests to verify pass (without WeasyPrint installed yet)**

Run: `.venv/bin/python -m unittest discover -s tests -p "test_report_save.py" -v`
Expected: PASS — 3 tests pass, `test_render_pdf_smoke` SKIPPED.

- [ ] **Step 5: Add WeasyPrint dependency and install**

Edit `pyproject.toml` — add to `[project].dependencies`:

```
    "weasyprint>=62.0",
```

Install system libs (one-time, requires sudo — run manually in WSL):

```bash
sudo apt-get update && sudo apt-get install -y libpango-1.0-0 libpangoft2-1.0-0 libcairo2 libgdk-pixbuf-2.0-0
```

Then sync the venv:

```bash
uv sync --extra gui
```

- [ ] **Step 6: Run the render smoke test (now active)**

Run: `.venv/bin/python -m unittest discover -s tests -p "test_report_save.py" -v`
Expected: PASS — all 4 tests pass (smoke test no longer skipped). If `test_render_pdf_smoke` errors with a missing-library message, the apt step above was not completed.

- [ ] **Step 7: Document the system-lib requirement in `README.md`**

Append:

```markdown
## Xuất phiếu PDF (WeasyPrint)

Tính năng lưu phiếu kiểm tra PDF dùng WeasyPrint. Cài thư viện hệ thống một lần trong WSL:

    sudo apt-get install -y libpango-1.0-0 libpangoft2-1.0-0 libcairo2 libgdk-pixbuf-2.0-0

Đặt thư mục lưu phiếu trong `server/.env`:

    REPORTS_DIR=/mnt/c/Users/<bạn>/KetQuaKiemTra
```

- [ ] **Step 8: Commit**

```bash
git add server/report.py pyproject.toml uv.lock README.md tests/test_report_save.py
git commit -m "feat: render report PDF with WeasyPrint and save to disk" -m "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

