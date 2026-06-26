# Task 3 Completion Report

## STATUS
✅ COMPLETE

## Commit Hash
- `cd45215` — feat: add report filename + HTML/CSS builder

## Test Summary
All 5 tests pass (100%):
- `test_format_ict` — ICT timezone formatting verified
- `test_sanitize_filename_part` — diacritic stripping and filename sanitization verified
- `test_report_filename` — PDF filename generation verified
- `test_build_html_contains_sections_and_totals` — HTML output structure and content verified
- `test_build_html_zero_defects` — zero-defect case with fallback message verified

## Files Modified/Created
- `server/report.py` — Added 4 functions + imports + module-level constants (kept existing `ALLOWED_DEFECT_TYPES`, `ReportError`, `build_report_data`)
- `server/report.css` — Created (26 lines, A4 page layout with Vietnamese inspection report styling)
- `tests/test_report_html.py` — Created (5 test cases)

## Implementation Details
- **`format_ict(iso_utc: str) -> str`** — Parses ISO UTC, converts to ICT (UTC+7), formats as `dd/MM/yyyy · HH:mm`
- **`sanitize_filename_part(name: str) -> str`** — Normalizes Unicode, strips diacritics (Băng → Bang), replaces non-alphanumeric with underscore
- **`report_filename(conveyor_name, created_at_iso, job_id) -> str`** — Returns `XT-100_{sanitized_conveyor}_{YYYYMMDD_HHMM}_{job_id}.pdf`
- **`build_html(report_data, meta, images_by_track, logo_data_uri=None) -> str`** — Generates full HTML inspection report with embedded CSS, metadata boxes, defect statistics, and conditional image galleries (skips empty defect types)

## Concerns
None. All requirements met, all tests passing.

---

## Hardening Fix — Logo URI Escaping

**Commit:** `d77ba4e` — fix: escape logo data URI in report HTML

**Change:** Line 103 in `server/report.py`
```python
# Before:
logo = f'<img src="{logo_data_uri}" style="height:40px">' if logo_data_uri else ""

# After:
logo = f'<img src="{esc(logo_data_uri)}" style="height:40px">' if logo_data_uri else ""
```

**Test Result:** 5/5 passing (no regressions)
- All existing HTML and report tests confirmed green after escaping applied
