from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from ..pipeline_worker import DisplayPacket
from ..runtime import RunSummary, RuntimeStatus


@dataclass(frozen=True, slots=True)
class RuntimeSnapshot:
    status: RuntimeStatus | None
    summary: RunSummary | None
    display_packet: DisplayPacket | None
    last_error: str | None
    updated_at: float


class RuntimeStore:
    """Thread-safe snapshot store updated by RuntimeController monitor."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snapshot = RuntimeSnapshot(
            status=None,
            summary=None,
            display_packet=None,
            last_error=None,
            updated_at=time.monotonic(),
        )

    def update(
        self,
        *,
        status: RuntimeStatus | None,
        summary: RunSummary | None,
        display_packet: DisplayPacket | None,
        last_error: str | None,
    ) -> None:
        with self._lock:
            self._snapshot = RuntimeSnapshot(
                status=status,
                summary=summary,
                display_packet=display_packet,
                last_error=last_error,
                updated_at=time.monotonic(),
            )

    def snapshot(self) -> RuntimeSnapshot:
        with self._lock:
            return RuntimeSnapshot(
                status=self._snapshot.status,
                summary=self._snapshot.summary,
                display_packet=self._snapshot.display_packet,
                last_error=self._snapshot.last_error,
                updated_at=self._snapshot.updated_at,
            )
