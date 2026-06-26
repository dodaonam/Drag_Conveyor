from __future__ import annotations

import sys
from pathlib import Path

if getattr(sys, "frozen", False):
    # sys._MEIPASS = _internal/ — chứa bundled assets (read-only)
    BUNDLE_DIR = Path(sys._MEIPASS)
    # cạnh DragConveyor.exe — writable, tồn tại sau khi user giải nén
    APP_DIR = Path(sys.executable).parent
else:
    BUNDLE_DIR = Path(__file__).resolve().parent.parent
    APP_DIR = Path(__file__).resolve().parent.parent

RUNTIME_DIR = APP_DIR / "runtime"
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)


def ensure_repo_root_on_path() -> None:
    root = str(APP_DIR)
    if root not in sys.path:
        sys.path.insert(0, root)


__all__ = ["BUNDLE_DIR", "APP_DIR", "RUNTIME_DIR", "ensure_repo_root_on_path"]
