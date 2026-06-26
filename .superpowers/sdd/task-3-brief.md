# Task 3 brief (from approved plan)

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

