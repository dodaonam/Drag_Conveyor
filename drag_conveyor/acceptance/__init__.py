from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class AcceptanceResult:
    run_id: str
    total_rows: int
    suspected_rows: int
    snapshots_found: int
    a2_pass: bool
    issues: list[str]


def evaluate_run_outputs(*, app_root: Path, run_id: str) -> AcceptanceResult:
    logs_path = app_root / "logs" / f"{run_id}_inspection.csv"
    snapshots_dir = app_root / "output" / "defect_snapshots" / run_id

    if not logs_path.exists():
        return AcceptanceResult(
            run_id=run_id,
            total_rows=0,
            suspected_rows=0,
            snapshots_found=0,
            a2_pass=False,
            issues=[f"Missing CSV: {logs_path}"],
        )

    rows: list[dict[str, str]] = []
    with logs_path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        rows = list(reader)

    issues: list[str] = []
    suspected = [r for r in rows if r.get("result") == "suspected_defect"]

    snapshots = list(snapshots_dir.glob("track_*.jpg")) if snapshots_dir.exists() else []

    # A2 proxy: each suspected defect must have reasons + score + snapshot.
    for row in suspected:
        reasons = row.get("reasons", "")
        score = row.get("score", "")
        if not reasons:
            issues.append(f"track_id={row.get('track_id')} missing reasons")
        else:
            try:
                parsed = json.loads(reasons)
                if not isinstance(parsed, list):
                    issues.append(f"track_id={row.get('track_id')} reasons is not JSON list")
            except json.JSONDecodeError:
                issues.append(f"track_id={row.get('track_id')} reasons is not valid JSON")
        if score == "":
            issues.append(f"track_id={row.get('track_id')} missing score")

    if len(snapshots) < len(suspected):
        issues.append(
            f"snapshot count {len(snapshots)} < suspected rows {len(suspected)}"
        )

    return AcceptanceResult(
        run_id=run_id,
        total_rows=len(rows),
        suspected_rows=len(suspected),
        snapshots_found=len(snapshots),
        a2_pass=(len(issues) == 0),
        issues=issues,
    )
