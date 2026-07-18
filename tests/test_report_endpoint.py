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
            self.assertTrue(out["excel_filename"].endswith(".xlsx"))
            self.assertTrue((Path(tmp) / "reports" / out["filename"]).exists())
            self.assertTrue((Path(tmp) / "reports" / out["excel_filename"]).exists())

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
