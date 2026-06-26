from __future__ import annotations

import importlib
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SERVER_DIR = ROOT / "server"


def _load(temp_dir: Path):
    env = {
        "R2_ENDPOINT_URL": "https://example.invalid",
        "R2_ACCESS_KEY_ID": "k",
        "R2_SECRET_ACCESS_KEY": "s",
        "R2_BUCKET_NAME": "b",
        "API_AUTH_TOKEN": "t",
        "REPORTS_DIR": str(temp_dir / "reports"),
    }
    os.environ.update(env)
    if str(SERVER_DIR) not in sys.path:
        sys.path.insert(0, str(SERVER_DIR))
    for name in ("settings", "r2"):
        sys.modules.pop(name, None)
    settings = importlib.import_module("settings")
    r2 = importlib.import_module("r2")
    return settings, r2


class ReportInfraTests(unittest.TestCase):
    def test_reports_dir_reads_env(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            settings, _ = _load(Path(tmp))
            self.assertEqual(settings.REPORTS_DIR, (Path(tmp) / "reports").resolve())

    def test_download_bytes_reads_object_body(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            _, r2 = _load(Path(tmp))
            fake_body = mock.Mock()
            fake_body.read.return_value = b"image-bytes"
            fake_client = mock.Mock()
            fake_client.get_object.return_value = {"Body": fake_body}
            with mock.patch.object(r2, "_client", return_value=fake_client):
                data = r2.download_bytes("results/job1/snapshots/defects/x.jpg")
            self.assertEqual(data, b"image-bytes")
            fake_client.get_object.assert_called_once()


if __name__ == "__main__":
    unittest.main()
