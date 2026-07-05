from __future__ import annotations

import functools
import html as _html
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

_TZ_ICT = timezone(timedelta(hours=7))


@functools.lru_cache(maxsize=1)
def _report_css() -> str:
    return (Path(__file__).parent / "report.css").read_text(encoding="utf-8")

_SECTION_LABELS = {
    "bent_left": "Biến dạng bên trái",
    "bent_right": "Biến dạng bên phải",
    "bent_both": "Biến dạng 2 bên",
    "broken": "Gãy",
}

_COMPANY = "CÔNG TY CỔ PHẦN CHĂN NUÔI C.P. VIỆT NAM"
_BRANCH = "Chi nhánh Xuân Mai – Hà Nội"
_DEPARTMENT = "Phòng Kỹ Thuật"

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


def report_filename(conveyor_name: str, exported_at_iso: str) -> str:
    """Name = export timestamp + inspected machine name, e.g.
    ``20260625_143052_M515A.pdf``. Collisions are resolved by save_report."""
    dt = datetime.fromisoformat(exported_at_iso)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    stamp = dt.astimezone(_TZ_ICT).strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{sanitize_filename_part(conveyor_name)}.pdf"


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

    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>{_report_css()}</style></head>
<body>
  <div class="hdr">
    <div>{logo}<div class="company">{esc(_COMPANY)}</div><div class="branch">{esc(_BRANCH)}</div><div class="dept">{esc(_DEPARTMENT)}</div></div>
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


def render_pdf(html: str) -> bytes:
    import sys
    import pdfkit
    if globals().get("__compiled__", False):
        import platform
        from pathlib import Path as _Path
        exe_name = "wkhtmltopdf.exe" if platform.system() == "Windows" else "wkhtmltopdf"
        cfg = pdfkit.configuration(wkhtmltopdf=str(_Path(sys.executable).parent / "bin" / exe_name))
    else:
        cfg = pdfkit.configuration()
    return pdfkit.from_string(html, False, configuration=cfg,
                              options={"encoding": "UTF-8", "no-outline": None,
                                       "background": None, "image-quality": "100"})


def _image_data_uri(data: bytes) -> str:
    import base64
    return "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")


def save_report(summary, corrections, meta, reports_dir,
                fetch_image: "Callable[[str], bytes | None]",
                exported_at_iso: str | None = None) -> str:
    if exported_at_iso is None:
        exported_at_iso = datetime.now(timezone.utc).isoformat()
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
    base = report_filename(meta["conveyor_name"], exported_at_iso)
    target = reports_dir / base
    if target.exists():
        stem, suffix = base[:-4], ".pdf"
        i = 2
        while (reports_dir / f"{stem}_{i}{suffix}").exists():
            i += 1
        target = reports_dir / f"{stem}_{i}{suffix}"
    target.write_bytes(pdf_bytes)
    return target.name
