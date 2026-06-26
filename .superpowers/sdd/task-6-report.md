# Task 6 Report — "Lưu kết quả" button + report save flow (frontend)

## Status: COMPLETE

## Commit
`1c7d182` — feat: add Lưu kết quả button + report save flow (frontend)

Files changed:
- `server/static/index.html` — button `#btn-save-report` + `#msg-save` div inserted before `#btn-new-job`
- `server/static/app.js` — added `VALID_DEFECT_TYPES`, `collectCorrections`, `hasUnclassified`, `setSaveMsg`, `saveReport`, and `addEventListener` for `btn-save-report`
- `server/static/styles.css` — appended `.btn-success:disabled`, `.save-msg`, `.save-msg.err`
- `tests/test_report_frontend_contract.py` — new contract test (created)

## Contract test result
```
test_appjs_posts_to_report_endpoint ... ok
test_index_has_save_button ... ok
Ran 2 tests in 0.001s — OK
```

## Full suite result
```
Ran 47 tests in 0.571s — OK
```

## TDD steps followed
1. Wrote `tests/test_report_frontend_contract.py` first → both tests FAILED (expected)
2. Added button + msg div to `index.html`
3. Added JS functions to `app.js` after `applyCorrection`
4. Appended `.save-msg` styles to `styles.css` (`.btn-success` was already present; added `:disabled` override + `.save-msg` rules)
5. Re-ran contract test → PASSED (2/2)
6. Ran full suite → PASSED (47/47)
7. Committed with two `-m` flags per spec

## Notes
- `.btn-success` background/color were already defined in `styles.css`; the append adds the `:disabled` opacity override and `.save-msg` colour rules without conflict.
- `hasUnclassified` blocks save if any defect has a `defect_type` not in `['bent_left','bent_right','bent_both','broken']` (covers `'other'` and `'_unclassified'`).
- `saveReport` POSTs to `/api/jobs/{G.jobId}/report` with `inspector_name`, `conveyor_name`, and `corrections` (defects as `{track_id, defect_type}`, normals as `defect_type: 'normal'`).

## Manual verification — PENDING (human action required)
Step 7 from the brief requires a live device/browser with a configured R2 bucket and a running uvicorn server:
```bash
cd server && uv run uvicorn main:app --host 127.0.0.1 --port 8000
```
Verify: complete a job → open results → leave one defect as "Không xác định rõ" → tap "Lưu kết quả" → confirm Vietnamese block message. Reclassify all → tap again → confirm "Đã lưu: <filename>" and PDF appears in `REPORTS_DIR`.
