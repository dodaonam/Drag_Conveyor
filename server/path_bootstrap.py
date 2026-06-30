from __future__ import annotations

import sys
from pathlib import Path

if getattr(sys, "frozen", False):
    # PyInstaller sets sys._MEIPASS; Nuitka does not — fall back to exe directory.
    _meipass = getattr(sys, "_MEIPASS", None)
    BUNDLE_DIR = Path(_meipass) if _meipass else Path(sys.executable).parent
    APP_DIR    = Path(sys.executable).parent
else:
    BUNDLE_DIR = Path(__file__).resolve().parent.parent
    APP_DIR    = Path(__file__).resolve().parent.parent

RUNTIME_DIR = APP_DIR / "runtime"
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)


def ensure_repo_root_on_path() -> None:
    root = str(APP_DIR)
    if root not in sys.path:
        sys.path.insert(0, root)


__all__ = ["BUNDLE_DIR", "APP_DIR", "RUNTIME_DIR", "ensure_repo_root_on_path"]
