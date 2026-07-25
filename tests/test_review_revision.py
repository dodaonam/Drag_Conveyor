from __future__ import annotations

import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SERVER_DIR = ROOT / "server"
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))


class ReviewRevisionTests(unittest.TestCase):
    def test_cas_persists_review_without_overwriting_vision_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db = importlib.import_module("db")
            previous = db.DB_PATH
            try:
                db.DB_PATH = Path(directory) / "jobs.db"
                db.init_db()
                db.create_job(job_id="job", status="processing", object_key="x", content_type="video/mp4", size_bytes=1, inspection_mode="geometry_v2", roi_config={}, now=db.now())
                summary = {"inspection_mode": "geometry_v2", "defects": [{"track_id": 1, "vision_status": "uncertain"}], "normals": []}
                db.save_result(job_id="job", summary=summary, now=db.now(), success=True)
                self.assertTrue(db.save_review_revision(job_id="job", expected_revision=0, reviewed_statuses={1: "broken_center"}, reviewer="reviewer", pdf_filename="a.pdf", excel_filename="a.xlsx", now=db.now()))
                saved = json.loads(db.get_job("job")["result_summary_json"])
                self.assertEqual(saved["defects"][0]["vision_status"], "uncertain")
                self.assertEqual(saved["defects"][0]["final_reviewed_status"], "broken_center")
                self.assertEqual(saved["review"]["revision"], 1)
                self.assertFalse(db.save_review_revision(job_id="job", expected_revision=0, reviewed_statuses={1: "normal"}, reviewer="reviewer", pdf_filename="b.pdf", excel_filename="b.xlsx", now=db.now()))
            finally:
                db.DB_PATH = previous


if __name__ == "__main__":
    unittest.main()
