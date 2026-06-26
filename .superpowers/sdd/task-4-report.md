# Task 4 Report

**STATUS:** DONE_WITH_CONCERNS

**Commit:** `bde06f3` — "feat: render report PDF with WeasyPrint and save to disk"

**Test summary:** 4/4 passed, 0 skipped — smoke test (`test_render_pdf_smoke`) RAN and PASSED (WeasyPrint Python package installed successfully via `uv sync --extra gui`).

**Concerns:**
- WeasyPrint system libraries (`libpango-1.0-0`, `libpangoft2-1.0-0`, `libcairo2`, `libgdk-pixbuf-2.0-0`) were NOT explicitly installed via `sudo apt-get` (sudo was skipped as instructed). The smoke test passed, which means these libs were already present on this WSL Ubuntu instance.
- On a fresh WSL environment without those system libs, `import weasyprint` will fail and `test_render_pdf_smoke` will SKIP (expected). The human must run `sudo apt-get install -y libpango-1.0-0 libpangoft2-1.0-0 libcairo2 libgdk-pixbuf-2.0-0` on any new environment to enable real PDF rendering.

---

## Dependency Fix (2026-06-22)

**Change:** Moved `weasyprint>=62.0` from `[project.optional-dependencies].gui` to core `[project] dependencies` so that `uv sync` (without `--extra gui`) includes it and the report endpoint does not fail at runtime.

**Test result:** Ran 4 tests in 0.080s — OK (4/4 passed)

**Commit:** `82112fe` — "fix: make weasyprint a core dependency (server renders PDFs)"

**Files changed:** `pyproject.toml`, `uv.lock`

---

**Files changed (original):**
- `server/report.py` — added `from typing import Callable`, `render_pdf`, `_image_data_uri`, `save_report`
- `pyproject.toml` — added `weasyprint>=62.0` to `[project.optional-dependencies].gui`
- `uv.lock` — updated by `uv sync --extra gui`
- `README.md` — appended WeasyPrint system-lib install note and `REPORTS_DIR` env var documentation
- `tests/test_report_save.py` — created with 4 tests (3 logic + 1 render smoke)
