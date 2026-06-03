from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WorkerStatus:
    name: str
    state: str
    message: str
    last_error: str | None = None

