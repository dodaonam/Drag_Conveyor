# Task 5 Report — POST /api/jobs/{job_id}/report endpoint

## Task tests (test_report_endpoint.py)
```
Ran 3 tests in 0.519s — OK
```

## Full suite
```
Ran 45 tests in 0.514s — OK
```

## Commit
`8b84df9` — feat: add POST /api/jobs/{id}/report endpoint

## Files changed
- `server/main.py` — added `import report`, `CorrectionIn` + `ReportIn` models, `save_report` endpoint
- `tests/test_report_endpoint.py` — created (3 tests: writes file, 400 on unclassified, 409 on non-completed)

## Concerns
None. All 45 tests pass, no regressions.

## Security Hardening
`28529f1` — fix: reject snapshot keys containing .. in report image fetch
- Added `".." in snapshot_key` check to `fetch_image` closure in `save_report` endpoint (line 326)
- Tests: 3/3 passed
