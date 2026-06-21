from __future__ import annotations

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


def format_ict(iso_utc: str) -> str:
    dt = datetime.fromisoformat(iso_utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(_TZ_ICT)
    return dt.strftime("%d/%m/%Y · %H:%M")


def sanitize_filename_part(name: str) -> str:
    import unicodedata
    name = (name or "").strip()
    # Normalize and remove diacritics
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    # Replace non-alphanumeric with underscore
    cleaned = re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_")
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

    logo = f'<img src="{esc(logo_data_uri)}" style="height:40px">' if logo_data_uri else ""
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
