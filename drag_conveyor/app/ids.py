from __future__ import annotations

from datetime import datetime
from uuid import uuid4


def generate_run_id() -> str:
    """Generate unique run ID with millisecond precision plus short suffix."""
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    millis = f"{int(now.microsecond / 1000):03d}"
    suffix = uuid4().hex[:8]
    return f"{timestamp}_{millis}_{suffix}"
