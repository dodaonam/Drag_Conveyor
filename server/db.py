from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

DB_PATH = Path(__file__).parent / "jobs.db"
_DEFAULT_INSPECTION_MODE = "auto_baseline"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id              TEXT PRIMARY KEY,
    status              TEXT NOT NULL,
    created_at          TEXT NOT NULL,
    updated_at          TEXT NOT NULL,
    object_key          TEXT,
    content_type        TEXT,
    size_bytes          INTEGER,
    inspection_mode     TEXT NOT NULL DEFAULT 'auto_baseline',
    roi_config_json     TEXT,
    upload_completed_at TEXT,
    result_summary_json TEXT,
    error_message       TEXT
);
"""


def init_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript(_SCHEMA)
        _migrate_jobs_table(conn)
        _ensure_inspection_mode_column(conn)
        _strip_csv_keys_from_summaries(conn)


def _job_columns(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute("PRAGMA table_info(jobs)").fetchall()
    return [str(row[1]) for row in rows]


def _migrate_jobs_table(conn: sqlite3.Connection) -> None:
    columns = _job_columns(conn)
    if "result_csv_key" not in columns:
        return

    conn.executescript(
        """
        ALTER TABLE jobs RENAME TO jobs_old;
        CREATE TABLE jobs (
            job_id              TEXT PRIMARY KEY,
            status              TEXT NOT NULL,
            created_at          TEXT NOT NULL,
            updated_at          TEXT NOT NULL,
            object_key          TEXT,
            content_type        TEXT,
            size_bytes          INTEGER,
            inspection_mode     TEXT NOT NULL DEFAULT 'auto_baseline',
            roi_config_json     TEXT,
            upload_completed_at TEXT,
            result_summary_json TEXT,
            error_message       TEXT
        );
        INSERT INTO jobs (
            job_id,
            status,
            created_at,
            updated_at,
            object_key,
            content_type,
            size_bytes,
            inspection_mode,
            roi_config_json,
            upload_completed_at,
            result_summary_json,
            error_message
        )
        SELECT
            job_id,
            status,
            created_at,
            updated_at,
            object_key,
            content_type,
            size_bytes,
            'auto_baseline',
            roi_config_json,
            upload_completed_at,
            result_summary_json,
            error_message
        FROM jobs_old;
        DROP TABLE jobs_old;
        """
    )


def _ensure_inspection_mode_column(conn: sqlite3.Connection) -> None:
    columns = _job_columns(conn)
    if "inspection_mode" in columns:
        return
    conn.execute(
        "ALTER TABLE jobs ADD COLUMN inspection_mode TEXT NOT NULL DEFAULT 'auto_baseline'"
    )


def _strip_csv_keys_from_summaries(conn: sqlite3.Connection) -> None:
    rows = conn.execute(
        "SELECT job_id, result_summary_json FROM jobs WHERE result_summary_json IS NOT NULL"
    ).fetchall()
    updates: list[tuple[str, str]] = []
    for job_id, payload in rows:
        if payload is None:
            continue
        try:
            summary = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if not isinstance(summary, dict):
            continue
        changed = summary.pop("csv_key", None) is not None
        changed = summary.pop("csv_url", None) is not None or changed
        if changed:
            updates.append((json.dumps(summary), str(job_id)))
    if updates:
        conn.executemany(
            "UPDATE jobs SET result_summary_json=? WHERE job_id=?",
            updates,
        )


@contextmanager
def _conn() -> Generator[sqlite3.Connection, None, None]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def create_job(
    *,
    job_id: str,
    status: str,
    object_key: str,
    content_type: str,
    size_bytes: int,
    inspection_mode: str = _DEFAULT_INSPECTION_MODE,
    roi_config: dict,
    now: str,
) -> None:
    with _conn() as conn:
        conn.execute(
            """INSERT INTO jobs
               (job_id, status, created_at, updated_at, object_key,
                content_type, size_bytes, inspection_mode, roi_config_json)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                job_id, status, now, now, object_key,
                content_type, size_bytes, inspection_mode, json.dumps(roi_config),
            ),
        )


def get_job(job_id: str) -> sqlite3.Row | None:
    with _conn() as conn:
        return conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()


def update_status(
    job_id: str,
    status: str,
    now: str,
    error_message: str | None = None,
) -> None:
    with _conn() as conn:
        conn.execute(
            "UPDATE jobs SET status=?, updated_at=?, error_message=? WHERE job_id=?",
            (status, now, error_message, job_id),
        )


def mark_uploaded(job_id: str, now: str) -> None:
    with _conn() as conn:
        conn.execute(
            """UPDATE jobs
               SET status='uploaded', updated_at=?, upload_completed_at=?
               WHERE job_id=?""",
            (now, now, job_id),
        )


def claim_next_uploaded_job(now: str) -> str | None:
    with _conn() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            """SELECT job_id
               FROM jobs
               WHERE status='uploaded'
               ORDER BY COALESCE(upload_completed_at, created_at), created_at
               LIMIT 1"""
        ).fetchone()
        if row is None:
            return None

        job_id = str(row["job_id"])
        updated = conn.execute(
            """UPDATE jobs
               SET status='downloading', updated_at=?
               WHERE job_id=? AND status='uploaded'""",
            (now, job_id),
        )
        return job_id if updated.rowcount == 1 else None


def save_result(
    *,
    job_id: str,
    summary: dict,
    now: str,
    success: bool,
) -> None:
    status = "completed" if success else "failed"
    error = None if success else summary.get("failure_reason")
    with _conn() as conn:
        conn.execute(
            """UPDATE jobs
               SET result_summary_json=?, status=?, error_message=?, updated_at=?
               WHERE job_id=?""",
            (json.dumps(summary), status, error, now, job_id),
        )


def expire_stale_uploads(cutoff_iso: str, now: str) -> list[str]:
    """waiting_upload jobs older than cutoff → upload_expired. Returns affected job_ids."""
    with _conn() as conn:
        rows = conn.execute(
            "SELECT job_id FROM jobs WHERE status='waiting_upload' AND created_at < ?",
            (cutoff_iso,),
        ).fetchall()
        ids = [r["job_id"] for r in rows]
        if ids:
            conn.executemany(
                "UPDATE jobs SET status='upload_expired', updated_at=? WHERE job_id=?",
                [(now, jid) for jid in ids],
            )
        return ids


def timeout_processing(cutoff_iso: str, now: str) -> list[str]:
    """downloading/processing jobs stuck past cutoff → failed. Returns affected job_ids."""
    with _conn() as conn:
        rows = conn.execute(
            "SELECT job_id FROM jobs"
            " WHERE status IN ('downloading', 'processing') AND updated_at < ?",
            (cutoff_iso,),
        ).fetchall()
        ids = [r["job_id"] for r in rows]
        if ids:
            conn.executemany(
                "UPDATE jobs SET status='failed', updated_at=?,"
                " error_message='Xử lý quá thời gian cho phép' WHERE job_id=?",
                [(now, jid) for jid in ids],
            )
        return ids
