from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

from ..tools.benchmark import run_benchmark, save_benchmark_report
from ..config import default_profile, load_profile, save_profile
from ..video_io import open_video_source
from .batch import run_batch_inspection
from .paths import resolve_model_path, resolve_paths


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
    p_init.add_argument("--profile", default="config/base_profile.json")
    p_init.add_argument("--model", default="weights/best.onnx")
    p_init.add_argument("--source", default=None, help="Optional source to auto-fill frame size/ROI")

    p_inspect = sub.add_parser("inspect", help="1-pass inspect: collect all bars, calibrate, classify")
    p_inspect.add_argument("--profile", default="config/base_profile.json")
    p_inspect.add_argument("--source", required=True, help="Video path")
    p_inspect.add_argument("--run-id", default=None, help="Optional fixed run ID")
    p_inspect.add_argument("--no-snapshot", action="store_true", help="Skip defect snapshot writing")

    p_bench = sub.add_parser("benchmark", help="Benchmark input sizes")
    p_bench.add_argument("--profile", default="config/base_profile.json")
    p_bench.add_argument("--source", required=True)
    p_bench.add_argument("--sizes", nargs="+", type=int, default=[416, 512, 640])
    p_bench.add_argument("--frames", type=int, default=300)
    p_bench.add_argument("--output", default=None)

    p_self_check = sub.add_parser("self-check", help="Check profile loading and model path resolution")
    p_self_check.add_argument("--profile", default="config/base_profile.json")
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

    return parser


def cmd_init_profile(args: argparse.Namespace) -> int:
    root = Path(args.app_root).resolve()
    profile_path = (root / args.profile).resolve()

    profile = default_profile(model_path=args.model)

    if args.source is not None:
        source = str(args.source).strip()
        cap, _ = open_video_source(source)
        ok, frame = cap.read()
        cap.release()
        if ok and frame is not None:
            h, w = frame.shape[:2]
            profile.inspection_region.frame_width = w
            profile.inspection_region.frame_height = h
            profile.inspection_region.x = 0
            profile.inspection_region.y = 0
            profile.inspection_region.w = w
            profile.inspection_region.h = h

    save_profile(profile, profile_path)
    print(f"Created profile: {profile_path}")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    root = Path(args.app_root).resolve()
    paths = resolve_paths(root)

    profile_path = (root / args.profile).resolve()
    profile = load_profile(profile_path)

    model_path = resolve_model_path(root, profile.model.path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    result = run_batch_inspection(
        profile=profile,
        source=args.source,
        model_path=str(model_path),
        run_id=getattr(args, "run_id", None),
        defect_snapshots_root=paths.defect_snapshots_dir,
        save_defect_snapshot=not getattr(args, "no_snapshot", False),
    )

    if not result.success:
        print(f"Inspection failed: {result.failure_reason}")
        print(f"bars_collected={result.total_bars}")
        print(f"frames_scanned={result.frames_scanned}")
        return 2

    print(
        "Inspection complete.",
        f"run_id={result.run_id}",
        f"frames_scanned={result.frames_scanned}",
        f"total_bars={result.total_bars}",
        f"normal={result.normal_bars}",
        f"suspected_defect={result.defect_bars}",
        f"inlier_ratio={result.inlier_ratio:.3f}",
        f"inlier_count={result.inlier_count}",
        f"outlier_count={result.outlier_count}",
        f"snapshots={result.defect_snapshots_dir}",
    )
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
        from ..inference import OnnxRuntimeEngine

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
            f"frames_inferred={item.frames_inferred} "
            f"throughput_fps={throughput} infer_fps_est={infer_fps} "
            f"latency_ms={latency} p95_latency_ms={p95} cpu_ratio={cpu} "
            f"memory_peak_mb={mem} status={item.status}"
        )
    return 0


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
        raise RuntimeError("acceptance.report module is required")

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


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    level = getattr(logging, str(args.log_level).upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s - %(message)s")

    try:
        if args.command == "init-profile":
            return cmd_init_profile(args)
        if args.command == "inspect":
            return cmd_inspect(args)
        if args.command == "self-check":
            return cmd_self_check(args)
        if args.command == "benchmark":
            return cmd_benchmark(args)
        if args.command == "uat":
            return cmd_uat(args)
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}")
        return 2

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
