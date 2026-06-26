# Task 5 brief (from approved plan)

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

