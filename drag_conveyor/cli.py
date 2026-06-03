from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

from .acceptance import evaluate_run_outputs
from .benchmark import run_benchmark, save_benchmark_report
from .camera_io import is_camera_index_source, open_capture_source, probe_camera_indices
from .calibration_runner import build_calibration_artifacts_dir, run_auto_calibration
from .config import Profile, default_profile, load_profile, save_profile
from .packaging import BUNDLE_NAME, build_onefolder_exe, validate_packaging_report_file
from .paths import resolve_model_path, resolve_paths
from .runtime import InspectionRuntime


def _parse_bool_flag(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="White Bar Inspection V1 runtime")
    parser.add_argument("--app-root", default=".", help="App root path for relative layout")
    parser.add_argument("--log-level", default="INFO", help="Logging level")

    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init-profile", help="Create default profile.json")
    p_init.add_argument("--profile", default="config/profile.json")
    p_init.add_argument("--model", default="weights/best.onnx")
    p_init.add_argument("--source", default=None, help="Optional source to auto-fill frame size/ROI")

    p_cal = sub.add_parser("calibrate", help="Auto baseline calibration")
    p_cal.add_argument("--profile", default="config/profile.json")
    p_cal.add_argument("--source", required=True, help="Video path or camera index")
    p_cal.add_argument("--max-frames", type=int, default=6000)
    p_cal.add_argument("--show", action="store_true", help="Show preview")
    p_cal.add_argument("--artifacts-dir", default=None, help="Optional output directory for calibration artifacts")
    p_cal.add_argument("--loop-video", action="store_true", help="Loop video file sources during calibration")

    p_run = sub.add_parser("run", help="Run detection")
    p_run.add_argument("--profile", default="config/profile.json")
    p_run.add_argument("--source", required=True, help="Video path or camera index")
    p_run.add_argument("--max-frames", type=int, default=None)
    p_run.add_argument("--show", action="store_true", help="Show preview")
    p_run.add_argument("--loop-video", action="store_true", help="Loop video file sources during detection")

    p_gui = sub.add_parser("gui", help="Run desktop UI")
    p_gui.add_argument("--profile", default="config/profile.json")
    p_gui.add_argument("--smoke-seconds", type=float, default=None, help="Auto-close GUI after N seconds")

    p_scan = sub.add_parser("scan-cameras", help="Probe local camera indices")
    p_scan.add_argument("--max-index", type=int, default=10)
    p_scan.add_argument("--width", type=int, default=None)
    p_scan.add_argument("--height", type=int, default=None)
    p_scan.add_argument("--fps", type=int, default=None)

    p_bench = sub.add_parser("benchmark", help="Benchmark input sizes")
    p_bench.add_argument("--profile", default="config/profile.json")
    p_bench.add_argument("--source", required=True)
    p_bench.add_argument("--sizes", nargs="+", type=int, default=[416, 512, 640])
    p_bench.add_argument("--frames", type=int, default=300)
    p_bench.add_argument("--output", default=None)

    p_pack = sub.add_parser("package", help="Build one-folder executable using PyInstaller")
    p_pack.add_argument("--no-clean", action="store_true")

    p_verify_pack = sub.add_parser(
        "verify-packaging-report",
        help="Verify clean-machine packaging report JSON against required checklist",
    )
    p_verify_pack.add_argument(
        "--report",
        default="runtime/packaging_report.json",
        help="Path to packaging_report.json",
    )

    p_self_check = sub.add_parser("self-check", help="Check profile loading and model path resolution")
    p_self_check.add_argument("--profile", default="config/profile.json")
    p_self_check.add_argument(
        "--load-model",
        action="store_true",
        help="Also load ONNX model via runtime engine to verify onnxruntime packaging/runtime dependencies",
    )

    p_uat = sub.add_parser("uat", help="Generate acceptance report (A1/A2/A3/A4/A5/A7/A9/A10/A11)")
    p_uat.add_argument("--run-id", required=True)
    p_uat.add_argument("--output-dir", default="runtime")
    p_uat.add_argument("--evidence-json", default=None)
    p_uat.add_argument("--manual-count", type=int, default=None)
    p_uat.add_argument("--system-count", type=int, default=None)
    p_uat.add_argument("--a3-duration-minutes", type=float, default=None)
    p_uat.add_argument("--a3-crashed", type=_parse_bool_flag, default=None)
    p_uat.add_argument("--a3-latency-drift-ratio", type=float, default=None)
    p_uat.add_argument("--a4-mapping-tests-passed", type=_parse_bool_flag, default=None)
    p_uat.add_argument("--a5-recovered", type=_parse_bool_flag, default=None)
    p_uat.add_argument("--a5-crash", type=_parse_bool_flag, default=None)
    p_uat.add_argument("--a5-notes", default=None)
    p_uat.add_argument("--a7-blocked-detection", type=_parse_bool_flag, default=None)
    p_uat.add_argument("--a7-warning-present", type=_parse_bool_flag, default=None)
    p_uat.add_argument("--a9-success", type=_parse_bool_flag, default=None)
    p_uat.add_argument("--a9-records-collected", type=int, default=None)
    p_uat.add_argument("--a9-reason", default=None)
    p_uat.add_argument("--recall", type=float, default=None)
    p_uat.add_argument("--false-positive-ratio", type=float, default=None)
    p_uat.add_argument("--defects-truth", default=None)
    p_uat.add_argument("--normal-truth", default=None)

    p_uat_def = sub.add_parser("uat-defects", help="Compute A10 recall from run CSV + defect truth")
    p_uat_def.add_argument("--run-id", required=True)
    p_uat_def.add_argument("--truth", required=True, help="Path to defect truth JSON")

    p_uat_norm = sub.add_parser("uat-normal", help="Compute A11 false-positive ratio from run CSV + normal truth")
    p_uat_norm.add_argument("--run-id", required=True)
    p_uat_norm.add_argument("--truth", required=True, help="Path to normal truth JSON")
    p_uat_norm.add_argument(
        "--defects-truth",
        default=None,
        help="Optional defect truth JSON to also print recall + full confusion matrix",
    )

    return parser


def cmd_init_profile(args: argparse.Namespace) -> int:
    root = Path(args.app_root).resolve()
    profile_path = (root / args.profile).resolve()

    profile = default_profile(model_path=args.model)

    if args.source is not None:
        source = str(args.source).strip()
        cap, backend_name = open_capture_source(
            source,
            width=profile.camera.width,
            height=profile.camera.height,
            fps=profile.camera.fps,
            buffersize=1,
        )

        ok, frame = cap.read()
        cap.release()
        if ok and frame is not None:
            h, w = frame.shape[:2]
            if is_camera_index_source(source):
                profile.camera.index = int(source)
            profile.camera.backend = backend_name
            profile.camera.width = w
            profile.camera.height = h
            profile.inspection_region.frame_width = w
            profile.inspection_region.frame_height = h
            profile.inspection_region.x = 0
            profile.inspection_region.y = 0
            profile.inspection_region.w = w
            profile.inspection_region.h = h

    save_profile(profile, profile_path)
    print(f"Created profile: {profile_path}")
    return 0


def cmd_scan_cameras(args: argparse.Namespace) -> int:
    probes = probe_camera_indices(
        max_index=args.max_index,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )
    if not probes:
        print("No camera found in tested indices.")
        return 0

    print("Detected cameras:")
    for probe in probes:
        print(f"- source={probe.index} backend={probe.backend} frame={probe.width}x{probe.height}")
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    root = Path(args.app_root).resolve()
    profile_path = (root / args.profile).resolve()
    profile = load_profile(profile_path)

    model_path = resolve_model_path(root, profile.model.path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    artifacts_dir = (
        Path(args.artifacts_dir).resolve()
        if args.artifacts_dir is not None
        else build_calibration_artifacts_dir(root / "runtime")
    )

    result = run_auto_calibration(
        profile=profile,
        source=args.source,
        model_path=str(model_path),
        max_frames=args.max_frames,
        show_preview=args.show,
        artifacts_dir=artifacts_dir,
        loop_video=args.loop_video,
    )

    outcome = result.outcome
    if not outcome.success or outcome.updated_profile is None:
        print(
            "Calibration failed.",
            f"records_collected={result.records_collected}",
            f"reason={outcome.reason}",
            f"records_csv={result.records_csv_path}",
            f"report_json={result.report_json_path}",
        )
        return 2

    save_profile(outcome.updated_profile, profile_path)
    print(
        "Calibration success.",
        f"records_collected={result.records_collected}",
        f"frames_scanned={result.frames_scanned}",
        f"valid_records={outcome.calibration_result.valid_records if outcome.calibration_result else 0}",
        f"inlier_ratio={outcome.calibration_result.inlier_ratio if outcome.calibration_result else 0.0:.3f}",
        f"profile_saved={profile_path}",
        f"records_csv={result.records_csv_path}",
        f"report_json={result.report_json_path}",
    )
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    root = Path(args.app_root).resolve()
    paths = resolve_paths(root)

    profile_path = (root / args.profile).resolve()
    profile = load_profile(profile_path)

    model_path = resolve_model_path(root, profile.model.path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"best.onnx not found: {model_path}. "
            "Provide a valid model path in profile.model.path."
        )

    runtime = InspectionRuntime(profile=profile, deployment_paths=paths, model_path=model_path)
    runtime.load_model()

    summary = runtime.run(
        source=args.source,
        max_frames=args.max_frames,
        show_preview=args.show,
        loop_file=args.loop_video,
    )
    print(
        "Run complete.",
        f"run_id={summary.run_id}",
        f"frames_captured={summary.total_frames}",
        f"frames_inferred={summary.frames_inferred}",
        f"processed={summary.total_processed_bars}",
        f"normal={summary.normal_bars}",
        f"suspected_defect={summary.suspected_defect_bars}",
        f"runtime_throughput_fps={summary.avg_fps:.2f}",
        f"inference_fps_estimate={summary.inference_fps_estimate:.2f}",
        f"avg_latency_ms={summary.avg_latency_ms:.2f}",
        f"p95_latency_ms={summary.p95_latency_ms:.2f}",
        f"elapsed_sec={summary.elapsed_sec:.2f}",
    )
    return 0


def cmd_gui(args: argparse.Namespace) -> int:
    try:
        from .ui import run_gui
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PySide6 is not available in this Python environment. "
            "Install PySide6 or use CLI commands instead."
        ) from exc

    root = Path(args.app_root).resolve()
    profile_path = (root / args.profile).resolve()
    run_gui(app_root=root, profile_path=profile_path, smoke_seconds=args.smoke_seconds)
    return 0


def cmd_self_check(args: argparse.Namespace) -> int:
    root = Path(args.app_root).resolve()
    profile_path = (root / args.profile).resolve()
    profile = load_profile(profile_path)
    model_path = resolve_model_path(root, profile.model.path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    providers = "not_checked"
    if bool(getattr(args, "load_model", False)):
        from .inference import OnnxRuntimeEngine

        engine = OnnxRuntimeEngine()
        try:
            diagnostics = engine.load(str(model_path), profile.model)
            providers = ",".join(diagnostics.providers)
        finally:
            engine.close()

    print(
        "Self-check OK:",
        f"profile={profile_path}",
        f"model={model_path}",
        f"load_model={1 if bool(getattr(args, 'load_model', False)) else 0}",
        f"providers={providers}",
    )
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    root = Path(args.app_root).resolve()
    profile_path = (root / args.profile).resolve()
    profile = load_profile(profile_path)

    report = run_benchmark(
        app_root=root,
        profile=profile,
        source=args.source,
        input_sizes=args.sizes,
        frames=args.frames,
    )

    output = args.output
    if output is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = str(root / "runtime" / f"benchmark_{ts}.json")
    output_path = Path(output).resolve()
    save_benchmark_report(report, output_path)

    print(f"Benchmark saved: {output_path}")
    for item in report.results:
        throughput = f"{item.throughput_fps:.2f}" if item.throughput_fps is not None else "n/a"
        infer_fps = f"{item.inference_fps_estimate:.2f}" if item.inference_fps_estimate is not None else "n/a"
        latency = f"{item.avg_latency_ms:.2f}" if item.avg_latency_ms is not None else "n/a"
        p95 = f"{item.p95_latency_ms:.2f}" if item.p95_latency_ms is not None else "n/a"
        cpu = f"{item.cpu_process_time_ratio:.2f}" if item.cpu_process_time_ratio is not None else "n/a"
        mem = f"{item.memory_peak_mb:.2f}" if item.memory_peak_mb is not None else "n/a"
        print(
            f"imgsz={item.input_size} frames_captured={item.frames_captured} "
            f"frames_inferred={item.frames_inferred} frames_processed={item.frames_processed} "
            f"throughput_fps={throughput} infer_fps_est={infer_fps} "
            f"latency_ms={latency} p95_latency_ms={p95} cpu_ratio={cpu} "
            f"memory_peak_mb={mem} status={item.status}"
        )
    return 0


def cmd_package(args: argparse.Namespace) -> int:
    root = Path(args.app_root).resolve()
    build_onefolder_exe(app_root=root, clean=not args.no_clean)
    print(f"Packaged executable at: {root / 'dist' / BUNDLE_NAME}")
    print(f"Packaging report: {root / 'runtime' / 'packaging_report.json'}")
    return 0


def cmd_verify_packaging_report(args: argparse.Namespace) -> int:
    root = Path(args.app_root).resolve()
    report_path = (root / args.report).resolve()
    ok, issues = validate_packaging_report_file(report_path)
    if ok:
        print(f"Packaging report PASS: {report_path}")
        return 0

    print(f"Packaging report FAIL: {report_path}")
    for issue in issues:
        print(f"- {issue}")
    return 2


def cmd_uat(args: argparse.Namespace) -> int:
    root = Path(args.app_root).resolve()
    evidence: dict[str, object] = {}
    if args.evidence_json:
        evidence_path = Path(args.evidence_json).resolve()
        loaded = json.loads(evidence_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise RuntimeError("evidence-json root must be an object")
        evidence = loaded

    def _value(name: str, cli_value):
        if cli_value is not None:
            return cli_value
        return evidence.get(name)

    try:
        from acceptance.report import build_report, save_markdown
    except ModuleNotFoundError:
        # Fallback to A2-only proxy when acceptance suite package is unavailable.
        result = evaluate_run_outputs(app_root=root, run_id=args.run_id)
        print(
            "UAT result (A2 proxy):",
            f"run_id={result.run_id}",
            f"rows={result.total_rows}",
            f"suspected_rows={result.suspected_rows}",
            f"snapshots={result.snapshots_found}",
            f"A2_pass={result.a2_pass}",
        )
        if result.issues:
            for issue in result.issues:
                print("-", issue)
            return 2
        return 0

    report = build_report(
        app_root=root,
        run_id=args.run_id,
        manual_count=_value("manual_count", args.manual_count),
        system_count=_value("system_count", args.system_count),
        a3_duration_minutes=_value("a3_duration_minutes", args.a3_duration_minutes),
        a3_crashed=_value("a3_crashed", args.a3_crashed),
        a3_latency_drift_ratio=_value("a3_latency_drift_ratio", args.a3_latency_drift_ratio),
        a4_mapping_tests_passed=_value("a4_mapping_tests_passed", args.a4_mapping_tests_passed),
        a5_recovered=_value("a5_recovered", args.a5_recovered),
        a5_crash=_value("a5_crash", args.a5_crash),
        a5_notes=_value("a5_notes", args.a5_notes),
        a7_blocked_detection=_value("a7_blocked_detection", args.a7_blocked_detection),
        a7_warning_present=_value("a7_warning_present", args.a7_warning_present),
        a9_success=_value("a9_success", args.a9_success),
        a9_records_collected=_value("a9_records_collected", args.a9_records_collected),
        a9_reason=_value("a9_reason", args.a9_reason),
        recall=_value("recall", args.recall),
        false_positive_ratio=_value("false_positive_ratio", args.false_positive_ratio),
        defects_truth_path=(
            Path(_value("defects_truth", args.defects_truth)).resolve()
            if _value("defects_truth", args.defects_truth) is not None
            else None
        ),
        normal_truth_path=(
            Path(_value("normal_truth", args.normal_truth)).resolve()
            if _value("normal_truth", args.normal_truth) is not None
            else None
        ),
    )

    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "acceptance_report.json"
    md_path = output_dir / "acceptance_report.md"
    report.save_json(json_path)
    save_markdown(report, md_path)

    print(f"Saved: {json_path}")
    print(f"Saved: {md_path}")
    print(f"Overall: {'PASS' if report.passed else 'FAIL'}")
    for item in report.results:
        print(f"- {item.criterion}: {'PASS' if item.passed else 'FAIL'} | {item.details}")

    return 0 if report.passed else 2


def cmd_uat_defects(args: argparse.Namespace) -> int:
    root = Path(args.app_root).resolve()
    truth_path = Path(args.truth).resolve()
    from acceptance.test_a10_defect_recall import evaluate_from_run

    criterion, evidence = evaluate_from_run(
        app_root=root,
        run_id=args.run_id,
        truth_path=truth_path,
    )
    print(
        "A10 defect recall:",
        f"run_id={args.run_id}",
        f"passed={criterion.passed}",
        criterion.details,
    )
    print(json.dumps(evidence, indent=2, ensure_ascii=False))
    return 0 if criterion.passed else 2


def cmd_uat_normal(args: argparse.Namespace) -> int:
    root = Path(args.app_root).resolve()
    truth_path = Path(args.truth).resolve()
    from acceptance.test_a11_false_positive import evaluate_from_run
    from acceptance.test_a10_defect_recall import evaluate_from_run as evaluate_defect_recall

    criterion, evidence = evaluate_from_run(
        app_root=root,
        run_id=args.run_id,
        truth_path=truth_path,
    )
    print(
        "A11 false positive:",
        f"run_id={args.run_id}",
        f"passed={criterion.passed}",
        criterion.details,
    )
    print(json.dumps(evidence, indent=2, ensure_ascii=False))
    if args.defects_truth:
        defects_truth_path = Path(args.defects_truth).resolve()
        defect_criterion, defect_evidence = evaluate_defect_recall(
            app_root=root,
            run_id=args.run_id,
            truth_path=defects_truth_path,
        )
        confusion = {
            "tp": int(defect_evidence.get("true_positive", 0)),
            "fn": int(defect_evidence.get("false_negative", 0)),
            "fp": int(evidence.get("false_positive_count", 0)),
            "tn": int(evidence.get("true_negative_count", 0)),
        }
        print(
            "A10 recall (optional):",
            f"passed={defect_criterion.passed}",
            defect_criterion.details,
        )
        print(
            json.dumps(
                {
                    "recall": float(defect_evidence.get("recall", 0.0)),
                    "false_positive_rate": float(evidence.get("false_positive_ratio", 0.0)),
                    "confusion_matrix": confusion,
                    "missed_defect_track_ids": defect_evidence.get("missed_defect_track_ids", []),
                    "false_positive_track_ids": evidence.get("false_positive_track_ids", []),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    return 0 if criterion.passed else 2


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    level = getattr(logging, str(args.log_level).upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s - %(message)s")

    try:
        if args.command == "init-profile":
            return cmd_init_profile(args)
        if args.command == "calibrate":
            return cmd_calibrate(args)
        if args.command == "run":
            return cmd_run(args)
        if args.command == "gui":
            return cmd_gui(args)
        if args.command == "scan-cameras":
            return cmd_scan_cameras(args)
        if args.command == "self-check":
            return cmd_self_check(args)
        if args.command == "benchmark":
            return cmd_benchmark(args)
        if args.command == "package":
            return cmd_package(args)
        if args.command == "verify-packaging-report":
            return cmd_verify_packaging_report(args)
        if args.command == "uat":
            return cmd_uat(args)
        if args.command == "uat-defects":
            return cmd_uat_defects(args)
        if args.command == "uat-normal":
            return cmd_uat_normal(args)
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}")
        return 2

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
