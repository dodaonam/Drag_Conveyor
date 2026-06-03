from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class DeploymentPaths:
    root: Path
    weights_dir: Path
    config_dir: Path
    logs_dir: Path
    output_dir: Path
    defect_snapshots_dir: Path
    runtime_dir: Path

    def ensure_runtime_dirs(self) -> None:
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.defect_snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.runtime_dir.mkdir(parents=True, exist_ok=True)


def detect_app_root(override: str | Path | None = None) -> Path:
    if override is not None:
        return Path(override).resolve()

    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path.cwd().resolve()


def resolve_paths(app_root: str | Path | None = None) -> DeploymentPaths:
    root = detect_app_root(app_root)
    output_dir = root / "output"
    return DeploymentPaths(
        root=root,
        weights_dir=root / "weights",
        config_dir=root / "config",
        logs_dir=root / "logs",
        output_dir=output_dir,
        defect_snapshots_dir=output_dir / "defect_snapshots",
        runtime_dir=root / "runtime",
    )


def resolve_model_path(root: Path, configured_path: str) -> Path:
    path = Path(configured_path)
    if path.is_absolute() and path.exists():
        return path

    candidates = [root / configured_path]
    # Smooth migration between early local layout and deployment layout.
    if configured_path == "weights/best.onnx":
        candidates.append(root / "model" / "best.onnx")
    if configured_path == "model/best.onnx":
        candidates.append(root / "weights" / "best.onnx")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]
