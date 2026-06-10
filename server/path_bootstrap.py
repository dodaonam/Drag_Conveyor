from __future__ import annotations

import sys
from pathlib import Path


def ensure_repo_root_on_path() -> None:
    server_dir = Path(__file__).resolve().parent
    repo_root = server_dir.parent
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


__all__ = ["ensure_repo_root_on_path"]
