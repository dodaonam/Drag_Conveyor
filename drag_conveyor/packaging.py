from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

BUNDLE_NAME = "WhiteBarInspection"


def build_onefolder_exe(*, app_root: Path, clean: bool = True) -> None:
    if not sys.platform.startswith("win"):
        raise RuntimeError(
            "Packaging production executable chỉ hỗ trợ Windows. "
            "Hãy build trên máy Windows clean theo Technical Design."
        )

    pyinstaller = shutil.which("pyinstaller")
    if pyinstaller is None:
        raise RuntimeError(
            "PyInstaller chưa được cài. Cài bằng `uv pip install pyinstaller` trước khi đóng gói."
        )

    dist = app_root / "dist"
    build = app_root / "build"
    spec = app_root / "runtime" / "whitebar.spec"

    if clean:
        if dist.exists():
            shutil.rmtree(dist)
        if build.exists():
            shutil.rmtree(build)

    cmd = [
        pyinstaller,
        "--noconfirm",
        "--clean",
        "--name",
        BUNDLE_NAME,
        "--onedir",
        "--paths",
        str(app_root),
        str(app_root / "runtime" / "entrypoint.py"),
        "--add-data",
        f"{_weights_source_dir(app_root)};weights",
        "--add-data",
        f"{app_root / 'config'};config",
    ]

    subprocess.run(cmd, cwd=app_root, check=True)
    bundle_root = _ensure_dist_layout(app_root)
    validation = _validate_bundle_layout(bundle_root)
    if not validation["ok"]:
        raise RuntimeError(
            "Bundle layout validation failed: " + ", ".join(str(x) for x in validation["missing"])
        )
    report = _build_packaging_report(app_root=app_root, bundle_root=bundle_root)
    report_path = app_root / "runtime" / "packaging_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    # Save a reference spec file for reproducible builds.
    spec.parent.mkdir(parents=True, exist_ok=True)
    spec.write_text(
        "\n".join(
            [
                "# Reference build command:",
                "# " + " ".join(cmd),
                f"# Packaging report: {report_path}",
            ]
        ),
        encoding="utf-8",
    )

def _weights_source_dir(app_root: Path) -> Path:
    weights = app_root / "weights"
    if weights.exists():
        return weights

    legacy_model_dir = app_root / "model"
    if legacy_model_dir.exists():
        return legacy_model_dir

    raise RuntimeError(
        "Không tìm thấy thư mục weights/ (hoặc model/ legacy). "
        "Hãy đặt best.onnx vào weights/best.onnx trước khi đóng gói."
    )


def _ensure_dist_layout(app_root: Path) -> Path:
    bundle_root = app_root / "dist" / BUNDLE_NAME
    if not bundle_root.exists():
        raise RuntimeError(f"Không tìm thấy output bundle sau khi build: {bundle_root}")

    _ensure_dir(bundle_root / "weights")
    _ensure_dir(bundle_root / "config")
    _ensure_dir(bundle_root / "logs")
    _ensure_dir(bundle_root / "output")
    _ensure_dir(bundle_root / "output" / "defect_snapshots")
    _ensure_dir(bundle_root / "runtime")

    src_weights = _weights_source_dir(app_root)
    _sync_dir(src_weights, bundle_root / "weights")
    _sync_dir(app_root / "config", bundle_root / "config")
    return bundle_root


def _build_packaging_report(*, app_root: Path, bundle_root: Path) -> dict[str, object]:
    layout = _validate_bundle_layout(bundle_root)
    smoke = _run_smoke_checks(bundle_root)
    all_ok = all(item["ok"] for item in smoke)
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "bundle_name": BUNDLE_NAME,
        "bundle_dir": str(bundle_root),
        "layout_validated": bool(layout["ok"]),
        "layout_missing": list(layout["missing"]),
        "smoke_checks": smoke,
        "status": "pass" if all_ok and bool(layout["ok"]) else "fail",
        "notes": (
            "Smoke checks run in package host environment. "
            "For production gate, execute on clean Windows machine."
        ),
        "technical_design_target": "Windows 10/11",
        "app_root": str(app_root),
    }


def _run_smoke_checks(bundle_root: Path) -> list[dict[str, object]]:
    exe_path = bundle_root / f"{BUNDLE_NAME}.exe"

    checks: list[tuple[str, list[str]]] = [
        ("help", [str(exe_path), "--help"]),
        ("scan_cameras", [str(exe_path), "scan-cameras", "--max-index", "1"]),
        ("self_check", [str(exe_path), "self-check", "--profile", "config/profile.json", "--load-model"]),
        ("gui_launch_smoke", [str(exe_path), "gui", "--profile", "config/profile.json", "--smoke-seconds", "1.5"]),
    ]
    results: list[dict[str, object]] = []

    for name, cmd in checks:
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=bundle_root,
                timeout=20,
                check=False,
            )
            results.append(
                {
                    "name": name,
                    "ok": proc.returncode == 0,
                    "returncode": proc.returncode,
                    "stdout_tail": proc.stdout[-800:],
                    "stderr_tail": proc.stderr[-800:],
                }
            )
        except Exception as exc:  # noqa: BLE001
            results.append(
                {
                    "name": name,
                    "ok": False,
                    "returncode": None,
                    "stdout_tail": "",
                    "stderr_tail": str(exc),
                }
            )

    return results


def _validate_bundle_layout(bundle_root: Path) -> dict[str, object]:
    missing: list[str] = []
    exe_win = bundle_root / f"{BUNDLE_NAME}.exe"
    if not exe_win.exists():
        missing.append(f"{BUNDLE_NAME}.exe")

    required_paths = [
        "weights",
        "config",
        "logs",
        "output",
        "output/defect_snapshots",
        "runtime",
        "config/profile.json",
        "weights/best.onnx",
    ]
    for rel in required_paths:
        if not (bundle_root / rel).exists():
            missing.append(rel)

    return {
        "ok": len(missing) == 0,
        "missing": missing,
    }


def validate_packaging_report_data(report: dict[str, object]) -> tuple[bool, list[str]]:
    required_checks = {"help", "scan_cameras", "self_check", "gui_launch_smoke"}
    issues: list[str] = []

    if str(report.get("technical_design_target", "")) != "Windows 10/11":
        issues.append("technical_design_target must be 'Windows 10/11'")

    if bool(report.get("layout_validated")) is not True:
        issues.append("layout_validated must be true")

    smoke_checks = report.get("smoke_checks")
    if not isinstance(smoke_checks, list):
        issues.append("smoke_checks must be a list")
        return False, issues

    by_name: dict[str, dict[str, object]] = {}
    for item in smoke_checks:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", ""))
        by_name[name] = item

    missing_checks = sorted(required_checks - set(by_name.keys()))
    if missing_checks:
        issues.append("missing required smoke checks: " + ", ".join(missing_checks))

    for check_name in sorted(required_checks):
        item = by_name.get(check_name)
        if item is None:
            continue
        if bool(item.get("ok")) is not True:
            issues.append(f"smoke check failed: {check_name}")

    if str(report.get("status", "")).lower() != "pass":
        issues.append("status must be 'pass'")

    return len(issues) == 0, issues


def validate_packaging_report_file(report_path: Path) -> tuple[bool, list[str]]:
    if not report_path.exists():
        return False, [f"report not found: {report_path}"]
    raw = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return False, ["report root must be an object"]
    return validate_packaging_report_data(raw)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _sync_dir(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)
