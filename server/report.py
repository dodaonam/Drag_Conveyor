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
