# PDF Inspection Report (XT-100) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a "Lưu kết quả" button that sends the inspector's corrected results to the server, which renders an XT-100 PDF report and saves it to a configured local Windows folder.

**Architecture:** New `POST /api/jobs/{job_id}/report` endpoint receives corrections keyed by `track_id`, applies them to the stored job summary, validates no defect remains unclassified, fetches snapshot images from R2, renders an HTML/CSS report to PDF with WeasyPrint, and writes it to `REPORTS_DIR`. PDF-building logic lives in a dependency-light `server/report.py` (pure functions + a thin WeasyPrint render wrapper), so the core is unit-testable without WeasyPrint installed.

**Tech Stack:** Python 3.12, FastAPI, Pydantic, boto3 (R2), WeasyPrint (HTML/CSS→PDF), vanilla JS frontend, `unittest`.

## Global Constraints

- Python ≥ 3.12.
- **Tests run with `unittest`, NOT pytest** (pytest is not installed). Run from repo root with the venv: `.venv/bin/python -m unittest discover -s tests -p "<file>.py" -v`.
- Server modules (`main`, `db`, `r2`, `settings`, `worker`, `report`) live in `server/` and are imported by inserting `server/` on `sys.path`. Server modules that read required env vars need the test env (see Task 5 loader). `report.py` must NOT import `settings`/`r2` at module top level so it can be imported standalone.
- WeasyPrint import must be lazy (inside the render function) so `import report` works without WeasyPrint or its system libs present.
- Commit messages end with: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- `docs/` is gitignored; commit plan/spec/doc files with `git add -f`.
- All user-facing UI/report strings are Vietnamese.
- Valid defect types (verbatim): `bent_left`, `bent_right`, `bent_both`, `broken`. Valid correction types add `normal`.
- Company header strings (verbatim): `CÔNG TY CỔ PHẦN C.P. VIỆT NAM`, `Chi nhánh Xuân Mai – Hà Nội • Phòng Kỹ Thuật`, `PHIẾU KT`, `Mã: XT-100`, title `PHIẾU KIỂM TRA XÍCH TẢI`, subtitle `Báo cáo kiểm tra cánh gạt băng tải`.
- Defect section labels (verbatim, in order): `bent_left`→`Biến dạng bên trái`, `bent_right`→`Biến dạng bên phải`, `bent_both`→`Biến dạng 2 bên`, `broken`→`Gãy`.

---

### Task 1: Infra — `REPORTS_DIR` setting + R2 `download_bytes`

**Files:**
- Modify: `server/settings.py` (append after existing settings)
- Modify: `server/r2.py` (add function)
- Modify: `server/.env` (add `REPORTS_DIR`)
- Test: `tests/test_report_infra.py` (create)

**Interfaces:**
- Produces: `settings.REPORTS_DIR: Path`; `r2.download_bytes(object_key: str) -> bytes`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_report_infra.py`:

```python
from __future__ import annotations

import importlib
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SERVER_DIR = ROOT / "server"


def _load(temp_dir: Path):
    env = {
        "R2_ENDPOINT_URL": "https://example.invalid",
        "R2_ACCESS_KEY_ID": "k",
        "R2_SECRET_ACCESS_KEY": "s",
        "R2_BUCKET_NAME": "b",
        "API_AUTH_TOKEN": "t",
        "REPORTS_DIR": str(temp_dir / "reports"),
    }
    os.environ.update(env)
    if str(SERVER_DIR) not in sys.path:
        sys.path.insert(0, str(SERVER_DIR))
    for name in ("settings", "r2"):
        sys.modules.pop(name, None)
    settings = importlib.import_module("settings")
    r2 = importlib.import_module("r2")
    return settings, r2


class ReportInfraTests(unittest.TestCase):
    def test_reports_dir_reads_env(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            settings, _ = _load(Path(tmp))
            self.assertEqual(settings.REPORTS_DIR, (Path(tmp) / "reports").resolve())

    def test_download_bytes_reads_object_body(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            _, r2 = _load(Path(tmp))
            fake_body = mock.Mock()
            fake_body.read.return_value = b"image-bytes"
            fake_client = mock.Mock()
            fake_client.get_object.return_value = {"Body": fake_body}
            with mock.patch.object(r2, "_client", return_value=fake_client):
                data = r2.download_bytes("results/job1/snapshots/defects/x.jpg")
            self.assertEqual(data, b"image-bytes")
            fake_client.get_object.assert_called_once()


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m unittest discover -s tests -p "test_report_infra.py" -v`
Expected: FAIL — `AttributeError: module 'settings' has no attribute 'REPORTS_DIR'` (and `r2` has no `download_bytes`).

- [ ] **Step 3: Add `REPORTS_DIR` to `server/settings.py`**

Append at the end of `server/settings.py`:

```python
REPORTS_DIR = Path(
    os.environ.get("REPORTS_DIR", str(_SERVER_DIR / "runtime" / "reports"))
).resolve()
```

- [ ] **Step 4: Add `download_bytes` to `server/r2.py`**

Add after `download_file`:

```python
def download_bytes(object_key: str) -> bytes:
    resp = _client().get_object(Bucket=settings.R2_BUCKET_NAME, Key=object_key)
    return resp["Body"].read()
```

- [ ] **Step 5: Add `REPORTS_DIR` to `server/.env`**

Append a line (adjust path to your Windows folder):

```
REPORTS_DIR=/mnt/c/Users/lebao/KetQuaKiemTra
```

- [ ] **Step 6: Run test to verify it passes**

Run: `.venv/bin/python -m unittest discover -s tests -p "test_report_infra.py" -v`
Expected: PASS (2 tests).

- [ ] **Step 7: Commit**

```bash
git add server/settings.py server/r2.py server/.env tests/test_report_infra.py
git commit -m "feat: add REPORTS_DIR setting and r2.download_bytes" -m "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: `report.py` — corrections logic (pure)

**Files:**
- Create: `server/report.py`
- Test: `tests/test_report_logic.py` (create)

**Interfaces:**
- Produces:
  - `ALLOWED_DEFECT_TYPES = ("bent_left", "bent_right", "bent_both", "broken")`
  - `class ReportError(Exception)`
  - `build_report_data(summary: dict, corrections: list[dict]) -> dict` returning
    `{"total_bars": int, "defect_count": int, "defect_rate": float, "defects_by_type": dict[str, list[dict]]}`.
    `defects_by_type` has one key per `ALLOWED_DEFECT_TYPES` (in order), each a list of bar dicts from the summary. Raises `ReportError` on invalid/unknown/unclassified.

- [ ] **Step 1: Write the failing test**

Create `tests/test_report_logic.py`:

```python
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


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m unittest discover -s tests -p "test_report_logic.py" -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'report'`.

- [ ] **Step 3: Create `server/report.py` with the logic**

```python
from __future__ import annotations

ALLOWED_DEFECT_TYPES: tuple[str, ...] = ("bent_left", "bent_right", "bent_both", "broken")
ALLOWED_CORRECTION_TYPES: tuple[str, ...] = ALLOWED_DEFECT_TYPES + ("normal",)


class ReportError(Exception):
    """Raised when corrected results cannot produce a valid report."""


def build_report_data(summary: dict, corrections: list[dict]) -> dict:
    bars_by_track: dict[int, dict] = {}
    final_type: dict[int, str] = {}

    for bar in summary.get("defects", []):
        tid = int(bar["track_id"])
        bars_by_track[tid] = bar
        final_type[tid] = bar.get("defect_type") or "_unclassified"
    for bar in summary.get("normals", []):
        tid = int(bar["track_id"])
        bars_by_track[tid] = bar
        final_type[tid] = "normal"

    for corr in corrections:
        tid = int(corr["track_id"])
        ctype = corr["defect_type"]
        if ctype not in ALLOWED_CORRECTION_TYPES:
            raise ReportError(f"invalid defect_type: {ctype}")
        if tid not in bars_by_track:
            raise ReportError(f"unknown track_id: {tid}")
        final_type[tid] = ctype

    defects_by_type: dict[str, list[dict]] = {t: [] for t in ALLOWED_DEFECT_TYPES}
    for tid, ftype in final_type.items():
        if ftype == "normal":
            continue
        if ftype not in ALLOWED_DEFECT_TYPES:
            raise ReportError(f"bar track {tid} is still unclassified ({ftype})")
        defects_by_type[ftype].append(bars_by_track[tid])

    defect_count = sum(len(v) for v in defects_by_type.values())
    total_bars = int(summary.get("total_bars", len(bars_by_track)))
    defect_rate = (defect_count / total_bars * 100.0) if total_bars else 0.0

    return {
        "total_bars": total_bars,
        "defect_count": defect_count,
        "defect_rate": defect_rate,
        "defects_by_type": defects_by_type,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m unittest discover -s tests -p "test_report_logic.py" -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add server/report.py tests/test_report_logic.py
git commit -m "feat: add report corrections/validation logic" -m "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: `report.py` — filename + HTML builder (pure)

**Files:**
- Modify: `server/report.py`
- Create: `server/report.css`
- Test: `tests/test_report_html.py` (create)

**Interfaces:**
- Produces:
  - `format_ict(iso_utc: str) -> str` → `"dd/MM/yyyy · HH:mm"` in ICT (UTC+7).
  - `sanitize_filename_part(name: str) -> str` → safe ascii-ish token.
  - `report_filename(conveyor_name: str, created_at_iso: str, job_id: str) -> str` → `XT-100_{conveyor}_{YYYYMMDD_HHMM}_{job_id}.pdf`.
  - `build_html(report_data: dict, meta: dict, images_by_track: dict[int, str | None], logo_data_uri: str | None = None) -> str`. `meta` keys: `inspector_name`, `conveyor_name`, `datetime_str`. `images_by_track` maps track_id → data URI (or None to skip the image).

- [ ] **Step 1: Write the failing test**

Create `tests/test_report_html.py`:

```python
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
        name = report.report_filename("M515A", "2026-06-11T11:47:00+00:00", "job123")
        self.assertEqual(name, "XT-100_M515A_20260611_1847_job123.pdf")

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m unittest discover -s tests -p "test_report_html.py" -v`
Expected: FAIL — `AttributeError: module 'report' has no attribute 'format_ict'`.

- [ ] **Step 3: Create `server/report.css`**

```css
@page { size: A4; margin: 18mm 14mm; }
* { box-sizing: border-box; }
body { font-family: "DejaVu Sans", sans-serif; color: #1f2937; font-size: 11px; margin: 0; }
.hdr { display: flex; justify-content: space-between; align-items: flex-start; border-bottom: 2px solid #166534; padding-bottom: 8px; }
.hdr .company { font-size: 15px; font-weight: 700; color: #166534; }
.hdr .branch { font-size: 10px; color: #6b7280; }
.hdr .code { text-align: right; font-size: 10px; color: #6b7280; }
.title { text-align: center; font-size: 22px; font-weight: 800; letter-spacing: 1px; margin: 14px 0 2px; }
.subtitle { text-align: center; color: #166534; font-size: 11px; margin-bottom: 14px; }
.meta { display: flex; gap: 10px; margin-bottom: 12px; }
.meta .box { flex: 1; border: 1px solid #e5e7eb; border-radius: 8px; padding: 8px 10px; }
.meta .lbl { font-size: 8px; letter-spacing: 1px; color: #9ca3af; text-transform: uppercase; }
.meta .val { font-size: 13px; font-weight: 700; margin-top: 2px; }
.stats { display: flex; gap: 10px; margin-bottom: 16px; }
.stats .card { flex: 1; border-radius: 8px; padding: 12px 14px; color: #fff; }
.stats .card.green { background: #15803d; }
.stats .card.orange { background: #c2410c; }
.stats .card.blue { background: #475569; }
.stats .num { font-size: 24px; font-weight: 800; }
.stats .lbl { font-size: 9px; opacity: .9; }
.section { border: 1px solid #e5e7eb; border-radius: 8px; padding: 10px 12px; margin-bottom: 10px; }
.section h3 { display: inline-block; color: #166534; font-size: 13px; margin: 0; }
.section .badge { float: right; color: #166534; font-size: 10px; border: 1px solid #bbf7d0; border-radius: 10px; padding: 1px 8px; }
.grid { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px; }
.grid img { width: 30%; border-radius: 6px; border: 1px solid #e5e7eb; }
.none { color: #6b7280; font-style: italic; }
```

- [ ] **Step 4: Add the builders to `server/report.py`**

Add at the top imports and helpers (keep WeasyPrint OUT of this file's top-level imports):

```python
import html as _html
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

_TZ_ICT = timezone(timedelta(hours=7))
_REPORT_CSS = (Path(__file__).parent / "report.css").read_text(encoding="utf-8")

_SECTION_LABELS = {
    "bent_left": "Biến dạng bên trái",
    "bent_right": "Biến dạng bên phải",
    "bent_both": "Biến dạng 2 bên",
    "broken": "Gãy",
}

_COMPANY = "CÔNG TY CỔ PHẦN C.P. VIỆT NAM"
_BRANCH = "Chi nhánh Xuân Mai – Hà Nội • Phòng Kỹ Thuật"


def format_ict(iso_utc: str) -> str:
    dt = datetime.fromisoformat(iso_utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(_TZ_ICT)
    return dt.strftime("%d/%m/%Y · %H:%M")


def sanitize_filename_part(name: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-zÀ-ỹ]+", "_", (name or "").strip()).strip("_")
    return cleaned or "x"


def report_filename(conveyor_name: str, created_at_iso: str, job_id: str) -> str:
    dt = datetime.fromisoformat(created_at_iso)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    stamp = dt.astimezone(_TZ_ICT).strftime("%Y%m%d_%H%M")
    return f"XT-100_{sanitize_filename_part(conveyor_name)}_{stamp}_{job_id}.pdf"


def build_html(report_data: dict, meta: dict, images_by_track: dict[int, str | None],
               logo_data_uri: str | None = None) -> str:
    def esc(s: object) -> str:
        return _html.escape(str(s))

    logo = f'<img src="{logo_data_uri}" style="height:40px">' if logo_data_uri else ""
    sections = []
    for dtype in ALLOWED_DEFECT_TYPES:
        bars = report_data["defects_by_type"].get(dtype, [])
        if not bars:
            continue
        imgs = ""
        for bar in bars:
            uri = images_by_track.get(int(bar["track_id"]))
            if uri:
                imgs += f'<img src="{uri}">'
        sections.append(
            f'<div class="section"><span class="badge">{len(bars)} cánh lỗi</span>'
            f'<h3>● {esc(_SECTION_LABELS[dtype])}</h3>'
            f'<div class="grid">{imgs}</div></div>'
        )
    body_sections = "".join(sections) or '<div class="none">Không phát hiện cánh lỗi</div>'

    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>{_REPORT_CSS}</style></head>
<body>
  <div class="hdr">
    <div>{logo}<div class="company">{esc(_COMPANY)}</div><div class="branch">{esc(_BRANCH)}</div></div>
    <div class="code"><b>PHIẾU KT</b><br>Mã: XT-100</div>
  </div>
  <div class="title">PHIẾU KIỂM TRA XÍCH TẢI</div>
  <div class="subtitle">Báo cáo kiểm tra cánh gạt băng tải</div>
  <div class="meta">
    <div class="box"><div class="lbl">Nhân viên kiểm tra</div><div class="val">{esc(meta['inspector_name'])}</div></div>
    <div class="box"><div class="lbl">Ngày & giờ kiểm tra</div><div class="val">{esc(meta['datetime_str'])}</div></div>
    <div class="box"><div class="lbl">Tên máy</div><div class="val">{esc(meta['conveyor_name'])}</div></div>
  </div>
  <div class="stats">
    <div class="card green"><div class="num">{report_data['total_bars']}</div><div class="lbl">Tổng số cánh đã kiểm tra</div></div>
    <div class="card orange"><div class="num">{report_data['defect_count']}</div><div class="lbl">Tổng số cánh lỗi</div></div>
    <div class="card blue"><div class="num">{report_data['defect_rate']:.1f}%</div><div class="lbl">Tỷ lệ lỗi</div></div>
  </div>
  {body_sections}
</body></html>"""
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m unittest discover -s tests -p "test_report_html.py" -v`
Expected: PASS (5 tests).

- [ ] **Step 6: Commit**

```bash
git add server/report.py server/report.css tests/test_report_html.py
git commit -m "feat: add report filename + HTML/CSS builder" -m "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

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

### Task 5: `main.py` — `POST /api/jobs/{job_id}/report` endpoint

**Files:**
- Modify: `server/main.py`
- Test: `tests/test_report_endpoint.py` (create)

**Interfaces:**
- Consumes: `report.save_report`, `report.format_ict`, `report.ReportError` (Task 2–4); `r2.download_bytes` (Task 1); `db.get_job`, `settings.REPORTS_DIR`.
- Produces: `main.ReportIn` model; `main.save_report(job_id: str, body: ReportIn) -> dict` returning `{"filename": str, "saved": True}`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_report_endpoint.py`:

```python
from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SERVER_DIR = ROOT / "server"


def _load(temp_dir: Path):
    os.environ.update({
        "R2_ENDPOINT_URL": "https://example.invalid", "R2_ACCESS_KEY_ID": "k",
        "R2_SECRET_ACCESS_KEY": "s", "R2_BUCKET_NAME": "b", "API_AUTH_TOKEN": "t",
        "REPORTS_DIR": str(temp_dir / "reports"),
    })
    if str(SERVER_DIR) not in sys.path:
        sys.path.insert(0, str(SERVER_DIR))
    for name in ("settings", "db", "r2", "worker", "main", "report"):
        sys.modules.pop(name, None)
    db = importlib.import_module("db")
    settings = importlib.import_module("settings")
    importlib.import_module("worker")
    main = importlib.import_module("main")
    db.DB_PATH = temp_dir / "jobs.db"
    settings.TEMP_DIR = temp_dir / "temp"
    settings.TEMP_DIR.mkdir(parents=True, exist_ok=True)
    settings.REPORTS_DIR = temp_dir / "reports"
    db.init_db()
    return db, settings, main


_SUMMARY = {
    "total_bars": 3,
    "defects": [{"track_id": 1, "defect_type": "bent_left", "snapshot_key": "results/job1/d/a.jpg"}],
    "normals": [{"track_id": 2, "snapshot_key": "results/job1/n/b.jpg"}],
}


def _seed_completed_job(db, summary=_SUMMARY, job_id="job1"):
    # create_job is keyword-only and needs status + roi_config (dict);
    # save_result only completes a job whose status is 'processing'.
    db.create_job(job_id=job_id, status="processing", object_key="vid.mp4",
                  content_type="video/mp4", size_bytes=1, inspection_mode="auto_baseline",
                  roi_config={}, now=db.now())
    db.save_result(job_id=job_id, summary=summary, now=db.now(), success=True)


class ReportEndpointTests(unittest.TestCase):
    def test_save_report_writes_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db, settings, main = _load(Path(tmp))
            _seed_completed_job(db)
            body = main.ReportIn.model_validate({
                "inspector_name": "Trần Văn Trường", "conveyor_name": "M515A",
                "corrections": [{"track_id": 1, "defect_type": "bent_left"}],
            })
            with mock.patch.object(main.r2, "download_bytes", return_value=b"img"), \
                 mock.patch.object(main.report, "render_pdf", return_value=b"%PDF fake"):
                out = main.save_report("job1", body)
            self.assertTrue(out["saved"])
            self.assertTrue(out["filename"].endswith(".pdf"))
            self.assertTrue((Path(tmp) / "reports" / out["filename"]).exists())

    def test_unclassified_returns_400(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db, settings, main = _load(Path(tmp))
            bad = dict(_SUMMARY)
            bad["defects"] = [{"track_id": 1, "defect_type": "other", "snapshot_key": "results/job1/x.jpg"}]
            _seed_completed_job(db, summary=bad)
            body = main.ReportIn.model_validate({"inspector_name": "A", "conveyor_name": "B", "corrections": []})
            with mock.patch.object(main.report, "render_pdf", return_value=b"%PDF"):
                with self.assertRaises(main.HTTPException) as ctx:
                    main.save_report("job1", body)
            self.assertEqual(ctx.exception.status_code, 400)

    def test_not_completed_returns_409(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db, settings, main = _load(Path(tmp))
            # Job exists but is not completed (status='uploaded') → 409.
            db.create_job(job_id="job1", status="uploaded", object_key="v", content_type="video/mp4",
                          size_bytes=1, inspection_mode="auto_baseline", roi_config={}, now=db.now())
            body = main.ReportIn.model_validate({"inspector_name": "A", "conveyor_name": "B", "corrections": []})
            with self.assertRaises(main.HTTPException) as ctx:
                main.save_report("job1", body)
            self.assertEqual(ctx.exception.status_code, 409)


if __name__ == "__main__":
    unittest.main()
```

(Note: confirm `db.create_job` parameter names by reading `server/db.py` before running; adjust the seed calls if the signature differs.)

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m unittest discover -s tests -p "test_report_endpoint.py" -v`
Expected: FAIL — `AttributeError: module 'main' has no attribute 'ReportIn'`.

- [ ] **Step 3: Add the model + endpoint to `server/main.py`**

Add `import report` near the other server imports (after `import worker`). Add the model after `StatusOut`:

```python
class CorrectionIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    track_id: int
    defect_type: str


class ReportIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    inspector_name: str
    conveyor_name: str
    corrections: list[CorrectionIn]
```

Add the endpoint after `get_result`:

```python
@app.post("/api/jobs/{job_id}/report", dependencies=[Depends(require_auth)])
def save_report(job_id: str, body: ReportIn) -> dict[str, Any]:
    row = db.get_job(job_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if row["status"] != "completed":
        raise HTTPException(status_code=409, detail=f"Job not completed: {row['status']}")

    summary = json.loads(row["result_summary_json"])
    key_prefix = f"results/{job_id}/"

    def fetch_image(snapshot_key: str) -> bytes | None:
        if not snapshot_key or not snapshot_key.startswith(key_prefix):
            return None
        try:
            return r2.download_bytes(snapshot_key)
        except Exception as exc:  # noqa: BLE001
            logging.getLogger(__name__).warning("report image fetch failed for %s: %s", snapshot_key, exc)
            return None

    meta = {
        "inspector_name": body.inspector_name,
        "conveyor_name": body.conveyor_name,
        "datetime_str": report.format_ict(row["created_at"]),
    }
    try:
        filename = report.save_report(
            summary=summary,
            corrections=[c.model_dump() for c in body.corrections],
            meta=meta,
            job_id=job_id,
            created_at_iso=row["created_at"],
            reports_dir=settings.REPORTS_DIR,
            fetch_image=fetch_image,
        )
    except report.ReportError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Cannot write report: {exc}") from exc

    return {"filename": filename, "saved": True}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m unittest discover -s tests -p "test_report_endpoint.py" -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Run the full suite (no regressions)**

Run: `.venv/bin/python -m unittest discover -s tests -v`
Expected: All pass (existing 24 + new report tests).

- [ ] **Step 6: Commit**

```bash
git add server/main.py tests/test_report_endpoint.py
git commit -m "feat: add POST /api/jobs/{id}/report endpoint" -m "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: Frontend — "Lưu kết quả" button, validation, API call

**Files:**
- Modify: `server/static/index.html` (result screen, near `#btn-new-job` ~line 163)
- Modify: `server/static/app.js`
- Modify: `server/static/styles.css`
- Test: `tests/test_report_frontend_contract.py` (create) — mirrors existing `test_frontend_trigger_preview_matches_runtime_contract`.

**Interfaces:**
- Consumes: `POST /api/jobs/{job_id}/report` (Task 5); `G.allDefects`, `G.allNormals`, `G.inspector`, `G.conveyor`, `G.jobId`, `api()`, `DEFECT_LABEL_KEYS`.

- [ ] **Step 1: Write the failing contract test**

Create `tests/test_report_frontend_contract.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m unittest discover -s tests -p "test_report_frontend_contract.py" -v`
Expected: FAIL — `AssertionError: 'id="btn-save-report"' not found`.

- [ ] **Step 3: Add the button + status element to `server/static/index.html`**

Find (around line 163): `<button class="btn btn-primary" id="btn-new-job">Kiểm tra tiếp</button>` and insert BEFORE it:

```html
          <button class="btn btn-success" id="btn-save-report">Lưu kết quả</button>
          <div id="msg-save" class="save-msg" style="display:none"></div>
```

- [ ] **Step 4: Add JS to `server/static/app.js`**

Add near the other result helpers (after `applyCorrection`):

```javascript
const VALID_DEFECT_TYPES = ['bent_left', 'bent_right', 'bent_both', 'broken'];

function collectCorrections() {
  const out = [];
  for (const d of G.allDefects) out.push({ track_id: d.track_id, defect_type: d.defect_type });
  for (const n of G.allNormals) out.push({ track_id: n.track_id, defect_type: 'normal' });
  return out;
}

function hasUnclassified() {
  return G.allDefects.some(d => !VALID_DEFECT_TYPES.includes(d.defect_type));
}

function setSaveMsg(text, isErr) {
  const el = document.getElementById('msg-save');
  el.style.display = 'block';
  el.textContent = text;
  el.classList.toggle('err', !!isErr);
}

async function saveReport() {
  if (hasUnclassified()) {
    setSaveMsg('Còn thanh chưa phân loại — hãy phân loại hết (trái/phải/2 bên/gãy) hoặc đánh dấu bình thường trước khi lưu.', true);
    return;
  }
  const btn = document.getElementById('btn-save-report');
  btn.disabled = true;
  btn.textContent = 'Đang lưu...';
  try {
    const res = await api('/api/jobs/' + G.jobId + '/report', {
      method: 'POST',
      body: JSON.stringify({
        inspector_name: G.inspector,
        conveyor_name: G.conveyor,
        corrections: collectCorrections(),
      }),
    });
    btn.textContent = 'Đã lưu ✓';
    setSaveMsg('Đã lưu: ' + res.filename, false);
  } catch (e) {
    btn.disabled = false;
    btn.textContent = 'Lưu kết quả';
    setSaveMsg('Lỗi khi lưu: ' + e.message, true);
  }
}

document.getElementById('btn-save-report').addEventListener('click', saveReport);
```

- [ ] **Step 5: Add styles to `server/static/styles.css`**

Append:

```css
.btn-success { background: #15803d; color: #fff; }
.btn-success:disabled { opacity: .6; }
.save-msg { margin-top: 8px; font-size: 13px; color: #166534; }
.save-msg.err { color: #b91c1c; }
```

- [ ] **Step 6: Run the contract test to verify it passes**

Run: `.venv/bin/python -m unittest discover -s tests -p "test_report_frontend_contract.py" -v`
Expected: PASS (2 tests).

- [ ] **Step 7: Manual verification**

Start the server and exercise the flow end-to-end:

```bash
cd server && uv run uvicorn main:app --host 127.0.0.1 --port 8000
```

Verify: complete a job → open results on the device → leave one defect as "Không xác định rõ" → tap "Lưu kết quả" → confirm it is blocked with the Vietnamese message. Reclassify all → tap again → confirm "Đã lưu: <filename>" and the PDF appears in `REPORTS_DIR` (open the Windows folder) with the XT-100 layout.

- [ ] **Step 8: Commit**

```bash
git add server/static/index.html server/static/app.js server/static/styles.css tests/test_report_frontend_contract.py
git commit -m "feat: add Lưu kết quả button + report save flow (frontend)" -m "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- §5 endpoint (track_id corrections, 400/409) → Task 5. ✓
- §6 PDF template (header/meta/stats/4 sections/zero-case/images) → Task 3 + 4. ✓
- §7 config & file saving (REPORTS_DIR, filename, dedup, weasyprint dep + apt) → Task 1, 3, 4. ✓
- §8 frontend (button, local validation, corrections, success/error) → Task 6. ✓
- §9 error handling (other→400, dir not writable→500, image fail→skip, prefix check, 409) → Task 4 (image skip), Task 5 (400/409/500, prefix). ✓
- §10 testing (pure logic, smoke render, endpoint) → Tasks 2–6. ✓

**Placeholder scan:** No TBD/TODO; every code step has real code. ✓

**Type consistency:** `build_report_data`→dict with `defects_by_type` consumed by `build_html` and `save_report`. `save_report(... fetch_image)` signature matches the closure built in `main.save_report`. `ReportIn.corrections: list[CorrectionIn]` → `c.model_dump()` → `{track_id, defect_type}` matches `build_report_data` expectation. `report.format_ict` / `report.ReportError` / `report.render_pdf` referenced in main match Task 3/2/4 definitions. ✓

**db signatures (verified):** `db.create_job(*, job_id, status, object_key, content_type, size_bytes, inspection_mode, roi_config: dict, now)` and `db.save_result(*, job_id, summary, now, success)` — both keyword-only. `save_result` only completes a job whose current status is `'processing'`, so the Task 5 seed creates the job with `status="processing"` before calling `save_result`. These are baked into the Task 5 test code.
