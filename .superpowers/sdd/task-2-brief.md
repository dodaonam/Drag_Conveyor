# Task 2 brief (from approved plan)

### Task 2: `report.py` — corrections logic (pure)

**Files:**
- Create: `server/report.py`
- Test: `tests/test_report_logic.py` (create)

**Interfaces:**
- Produces:
  - `ALLOWED_DEFECT_TYPES = ("bent_left", "bent_right", "bent_both", "broken")`
  - `class ReportError(Exception)`
  - `build_report_data(summary: dict, corrections: list[dict]) -> dict` returning
    `{"total_bars": int, "defect_count": int, "defect_rate": float, "defects_by_type": dict[str, list[dict]]}`.
    `defects_by_type` has one key per `ALLOWED_DEFECT_TYPES` (in order), each a list of bar dicts from the summary. Raises `ReportError` on invalid/unknown/unclassified.

- [ ] **Step 1: Write the failing test**

Create `tests/test_report_logic.py`:

```python
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVER_DIR = ROOT / "server"
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

import report  # noqa: E402


def _summary() -> dict:
    return {
        "total_bars": 10,
        "defects": [
            {"track_id": 1, "defect_type": "bent_left", "snapshot_key": "results/j/snapshots/defects/a.jpg"},
            {"track_id": 2, "defect_type": "broken", "snapshot_key": "results/j/snapshots/defects/b.jpg"},
        ],
        "normals": [
            {"track_id": 3, "snapshot_key": "results/j/snapshots/normals/c.jpg"},
        ],
    }


class ReportLogicTests(unittest.TestCase):
    def test_groups_defects_and_counts(self) -> None:
        data = report.build_report_data(_summary(), corrections=[])
        self.assertEqual(data["total_bars"], 10)
        self.assertEqual(data["defect_count"], 2)
        self.assertEqual(data["defect_rate"], 20.0)
        self.assertEqual(len(data["defects_by_type"]["bent_left"]), 1)
        self.assertEqual(len(data["defects_by_type"]["broken"]), 1)

    def test_correction_changes_type(self) -> None:
        data = report.build_report_data(_summary(), corrections=[{"track_id": 2, "defect_type": "bent_right"}])
        self.assertEqual(len(data["defects_by_type"]["broken"]), 0)
        self.assertEqual(len(data["defects_by_type"]["bent_right"]), 1)

    def test_correction_normal_removes_defect(self) -> None:
        data = report.build_report_data(_summary(), corrections=[{"track_id": 1, "defect_type": "normal"}])
        self.assertEqual(data["defect_count"], 1)

    def test_promote_normal_to_defect(self) -> None:
        data = report.build_report_data(_summary(), corrections=[{"track_id": 3, "defect_type": "bent_both"}])
        self.assertEqual(data["defect_count"], 3)
        self.assertEqual(len(data["defects_by_type"]["bent_both"]), 1)

    def test_uncorrected_other_defect_raises(self) -> None:
        s = _summary()
        s["defects"].append({"track_id": 5, "defect_type": "other", "snapshot_key": "results/j/x.jpg"})
        with self.assertRaises(report.ReportError):
            report.build_report_data(s, corrections=[])

    def test_invalid_correction_type_raises(self) -> None:
        with self.assertRaises(report.ReportError):
            report.build_report_data(_summary(), corrections=[{"track_id": 1, "defect_type": "exploded"}])

    def test_unknown_track_id_raises(self) -> None:
        with self.assertRaises(report.ReportError):
            report.build_report_data(_summary(), corrections=[{"track_id": 999, "defect_type": "broken"}])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m unittest discover -s tests -p "test_report_logic.py" -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'report'`.

- [ ] **Step 3: Create `server/report.py` with the logic**

```python
from __future__ import annotations

ALLOWED_DEFECT_TYPES: tuple[str, ...] = ("bent_left", "bent_right", "bent_both", "broken")
ALLOWED_CORRECTION_TYPES: tuple[str, ...] = ALLOWED_DEFECT_TYPES + ("normal",)


class ReportError(Exception):
    """Raised when corrected results cannot produce a valid report."""


def build_report_data(summary: dict, corrections: list[dict]) -> dict:
    bars_by_track: dict[int, dict] = {}
    final_type: dict[int, str] = {}

    for bar in summary.get("defects", []):
        tid = int(bar["track_id"])
        bars_by_track[tid] = bar
        final_type[tid] = bar.get("defect_type") or "_unclassified"
    for bar in summary.get("normals", []):
        tid = int(bar["track_id"])
        bars_by_track[tid] = bar
        final_type[tid] = "normal"

    for corr in corrections:
        tid = int(corr["track_id"])
        ctype = corr["defect_type"]
        if ctype not in ALLOWED_CORRECTION_TYPES:
            raise ReportError(f"invalid defect_type: {ctype}")
        if tid not in bars_by_track:
            raise ReportError(f"unknown track_id: {tid}")
        final_type[tid] = ctype

    defects_by_type: dict[str, list[dict]] = {t: [] for t in ALLOWED_DEFECT_TYPES}
    for tid, ftype in final_type.items():
        if ftype == "normal":
            continue
        if ftype not in ALLOWED_DEFECT_TYPES:
            raise ReportError(f"bar track {tid} is still unclassified ({ftype})")
        defects_by_type[ftype].append(bars_by_track[tid])

    defect_count = sum(len(v) for v in defects_by_type.values())
    total_bars = int(summary.get("total_bars", len(bars_by_track)))
    defect_rate = (defect_count / total_bars * 100.0) if total_bars else 0.0

    return {
        "total_bars": total_bars,
        "defect_count": defect_count,
        "defect_rate": defect_rate,
        "defects_by_type": defects_by_type,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m unittest discover -s tests -p "test_report_logic.py" -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add server/report.py tests/test_report_logic.py
git commit -m "feat: add report corrections/validation logic" -m "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

