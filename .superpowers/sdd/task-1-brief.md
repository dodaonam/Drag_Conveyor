# Task 1 brief (from approved plan)

### Task 1: Infra — `REPORTS_DIR` setting + R2 `download_bytes`

**Files:**
- Modify: `server/settings.py` (append after existing settings)
- Modify: `server/r2.py` (add function)
- Modify: `server/.env` (add `REPORTS_DIR`)
- Test: `tests/test_report_infra.py` (create)

**Interfaces:**
- Produces: `settings.REPORTS_DIR: Path`; `r2.download_bytes(object_key: str) -> bytes`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_report_infra.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m unittest discover -s tests -p "test_report_infra.py" -v`
Expected: FAIL — `AttributeError: module 'settings' has no attribute 'REPORTS_DIR'` (and `r2` has no `download_bytes`).

- [ ] **Step 3: Add `REPORTS_DIR` to `server/settings.py`**

Append at the end of `server/settings.py`:

```python
REPORTS_DIR = Path(
    os.environ.get("REPORTS_DIR", str(_SERVER_DIR / "runtime" / "reports"))
).resolve()
```

- [ ] **Step 4: Add `download_bytes` to `server/r2.py`**

Add after `download_file`:

```python
def download_bytes(object_key: str) -> bytes:
    resp = _client().get_object(Bucket=settings.R2_BUCKET_NAME, Key=object_key)
    return resp["Body"].read()
```

- [ ] **Step 5: Add `REPORTS_DIR` to `server/.env`**

Append a line (adjust path to your Windows folder):

```
REPORTS_DIR=/mnt/c/Users/lebao/KetQuaKiemTra
```

- [ ] **Step 6: Run test to verify it passes**

Run: `.venv/bin/python -m unittest discover -s tests -p "test_report_infra.py" -v`
Expected: PASS (2 tests).

- [ ] **Step 7: Commit**

```bash
git add server/settings.py server/r2.py server/.env tests/test_report_infra.py
git commit -m "feat: add REPORTS_DIR setting and r2.download_bytes" -m "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

