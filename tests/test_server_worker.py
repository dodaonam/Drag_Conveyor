from __future__ import annotations

import importlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]
SERVER_DIR = ROOT / "server"


def _load_server_modules(temp_dir: Path):
    env = {
        "R2_ENDPOINT_URL": "https://example.invalid",
        "R2_ACCESS_KEY_ID": "test-access-key",
        "R2_SECRET_ACCESS_KEY": "test-secret-key",
        "R2_BUCKET_NAME": "test-bucket",
        "API_AUTH_TOKEN": "test-token",
    }
    for key, value in env.items():
        os.environ[key] = value

    if str(SERVER_DIR) not in sys.path:
        sys.path.insert(0, str(SERVER_DIR))

    for name in ("main", "worker", "db", "r2", "settings"):
        sys.modules.pop(name, None)

    db = importlib.import_module("db")
    settings = importlib.import_module("settings")
    worker = importlib.import_module("worker")
    main = importlib.import_module("main")

    db.DB_PATH = temp_dir / "jobs.db"
    settings.TEMP_DIR = temp_dir / "temp"
    settings.TEMP_DIR.mkdir(parents=True, exist_ok=True)
    db.init_db()
    return db, worker, main


class ServerWorkerTests(unittest.TestCase):
    def test_roi_validation_rejects_negative_or_out_of_bounds_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _, _, main = _load_server_modules(Path(tmp))

            valid = main.RoiIn.model_validate(
                {
                    "x": 0,
                    "y": 0,
                    "w": 100,
                    "h": 80,
                    "frame_width": 320,
                    "frame_height": 240,
                }
            )
            self.assertEqual(valid.frame_width, 320)

            for payload in (
                {
                    "x": -1,
                    "y": 0,
                    "w": 100,
                    "h": 80,
                    "frame_width": 320,
                    "frame_height": 240,
                },
                {
                    "x": 0,
                    "y": 0,
                    "w": 100,
                    "h": 80,
                    "frame_width": 0,
                    "frame_height": 240,
                },
                {
                    "x": 250,
                    "y": 0,
                    "w": 100,
                    "h": 80,
                    "frame_width": 320,
                    "frame_height": 240,
                },
            ):
                with self.assertRaises(ValidationError):
                    main.RoiIn.model_validate(payload)

            with self.assertRaises(ValidationError):
                main.RoiIn.model_validate(
                    {
                        "x": 0,
                        "y": 0,
                        "w": 100,
                        "h": 80,
                        "frame_width": 320,
                        "frame_height": 240,
                        "position_ratio": 0.5,
                        "thickness_ratio": 0.25,
                    }
                )

    def test_create_job_persists_roi_without_name_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db, _, main = _load_server_modules(Path(tmp))
            body = main.CreateJobIn.model_validate(
                {
                    "content_type": "video/mp4",
                    "size_bytes": 123,
                    "roi": {
                        "x": 1,
                        "y": 2,
                        "w": 100,
                        "h": 80,
                        "frame_width": 320,
                        "frame_height": 240,
                    },
                }
            )

            with mock.patch.object(main.r2, "presigned_put_url", return_value="https://example.invalid/upload"):
                result = main.create_job(body)

            self.assertEqual(result.presigned_put_url, "https://example.invalid/upload")
            row = db.get_job(result.job_id)
            self.assertIsNotNone(row)
            assert row is not None
            self.assertIn('"x": 1', row["roi_config_json"])
            self.assertEqual(row["status"], "waiting_upload")
            self.assertEqual(row["inspection_mode"], "auto_baseline")

    def test_create_job_accepts_both_supported_inspection_modes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _, _, main = _load_server_modules(Path(tmp))
            base_payload = {
                "content_type": "video/mp4",
                "size_bytes": 123,
                "roi": {
                    "x": 1,
                    "y": 2,
                    "w": 100,
                    "h": 80,
                    "frame_width": 320,
                    "frame_height": 240,
                },
            }

            default_body = main.CreateJobIn.model_validate(base_payload)
            average_mode_body = main.CreateJobIn.model_validate(
                {
                    **base_payload,
                    "inspection_mode": "average_ratio",
                }
            )
            auto_baseline_body = main.CreateJobIn.model_validate(
                {
                    **base_payload,
                    "inspection_mode": "auto_baseline",
                }
            )

            self.assertIsNone(default_body.inspection_mode)
            self.assertEqual(average_mode_body.inspection_mode, "average_ratio")
            self.assertEqual(auto_baseline_body.inspection_mode, "auto_baseline")

    def test_runtime_config_exposes_profile_trigger_band(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _, _, main = _load_server_modules(Path(tmp))

            cfg = main.runtime_config()

            self.assertEqual(cfg["inspection"]["mode"], "auto_baseline")
            band = cfg["collection"]["trigger_band"]
            self.assertEqual(band["position_ratio"], 0.5)
            self.assertEqual(band["thickness_ratio"], 0.25)

    def test_uploaded_jobs_are_claimed_from_sqlite_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db, worker, _ = _load_server_modules(Path(tmp))
            now = "2026-06-09T00:00:00+00:00"
            db.create_job(
                job_id="job-1",
                status="waiting_upload",
                object_key="uploads/job-1/input.mp4",
                content_type="video/mp4",
                size_bytes=123,
                inspection_mode="auto_baseline",
                roi_config={
                    "x": 0,
                    "y": 0,
                    "w": 100,
                    "h": 80,
                    "frame_width": 320,
                    "frame_height": 240,
                },
                now=now,
            )
            db.mark_uploaded("job-1", "2026-06-09T00:00:05+00:00")

            processed: list[str] = []
            with mock.patch.object(worker, "_process_job", side_effect=processed.append):
                self.assertTrue(worker._claim_and_process_next_job())
                self.assertFalse(worker._claim_and_process_next_job())

            self.assertEqual(processed, ["job-1"])
            self.assertEqual(db.get_job("job-1")["status"], "downloading")


if __name__ == "__main__":
    unittest.main()
