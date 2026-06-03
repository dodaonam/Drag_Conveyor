from __future__ import annotations

import json
import time
import tracemalloc
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .config import Profile
from .paths import resolve_model_path, resolve_paths
from .runtime import InspectionRuntime


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    input_size: int
    frames_captured: int
    frames_inferred: int
    frames_processed: int
    throughput_fps: float | None
    inference_fps_estimate: float | None
    avg_latency_ms: float | None
    p95_latency_ms: float | None
    cpu_process_time_ratio: float | None
    memory_peak_mb: float | None
    elapsed_sec: float
    status: str
    message: str


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    source: str
    results: list[BenchmarkResult]


def run_benchmark(
    *,
    app_root: Path,
    profile: Profile,
    source: str,
    input_sizes: list[int],
    frames: int,
) -> BenchmarkReport:
    paths = resolve_paths(app_root)
    results: list[BenchmarkResult] = []
    base_input_size = int(profile.model.input_size)

    for size in input_sizes:
        test_profile = profile.clone()
        test_profile.model.input_size = int(size)

        model_path = _resolve_model_for_size(
            app_root=app_root,
            configured_model_path=test_profile.model.path,
            requested_size=int(size),
            base_input_size=base_input_size,
        )
        if model_path is None:
            results.append(
                BenchmarkResult(
                    input_size=int(size),
                    frames_captured=0,
                    frames_inferred=0,
                    frames_processed=0,
                    throughput_fps=None,
                    inference_fps_estimate=None,
                    avg_latency_ms=None,
                    p95_latency_ms=None,
                    cpu_process_time_ratio=None,
                    memory_peak_mb=None,
                    elapsed_sec=0.0,
                    status="skipped",
                    message=f"Missing model for input size {size}. Provide best_{size}.onnx.",
                )
            )
            continue

        runtime = InspectionRuntime(profile=test_profile, deployment_paths=paths, model_path=model_path)
        try:
            runtime.load_model()
        except Exception as exc:  # noqa: BLE001
            results.append(
                BenchmarkResult(
                    input_size=int(size),
                    frames_captured=0,
                    frames_inferred=0,
                    frames_processed=0,
                    throughput_fps=None,
                    inference_fps_estimate=None,
                    avg_latency_ms=None,
                    p95_latency_ms=None,
                    cpu_process_time_ratio=None,
                    memory_peak_mb=None,
                    elapsed_sec=0.0,
                    status="failed",
                    message=str(exc),
                )
            )
            continue

        cpu_start = time.process_time()
        wall_start = time.perf_counter()
        tracemalloc.start()
        summary = runtime.run(source=source, max_frames=frames, show_preview=False)
        _current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        wall_end = time.perf_counter()
        cpu_end = time.process_time()

        elapsed = wall_end - wall_start
        cpu_ratio = (cpu_end - cpu_start) / elapsed if elapsed > 0 else 0.0
        peak_mem_mb = float(peak_mem) / (1024.0 * 1024.0)

        results.append(
            BenchmarkResult(
                input_size=int(size),
                frames_captured=summary.total_frames,
                frames_inferred=summary.frames_inferred,
                frames_processed=summary.total_processed_bars,
                throughput_fps=summary.avg_fps,
                inference_fps_estimate=summary.inference_fps_estimate,
                avg_latency_ms=summary.avg_latency_ms,
                p95_latency_ms=summary.p95_latency_ms,
                cpu_process_time_ratio=cpu_ratio,
                memory_peak_mb=peak_mem_mb,
                elapsed_sec=elapsed,
                status="ok",
                message="ok",
            )
        )

    return BenchmarkReport(source=source, results=results)


def save_benchmark_report(report: BenchmarkReport, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "source": report.source,
                "results": [asdict(_normalize_nan(r)) for r in report.results],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _resolve_model_for_size(
    *,
    app_root: Path,
    configured_model_path: str,
    requested_size: int,
    base_input_size: int,
) -> Path | None:
    base = resolve_model_path(app_root, configured_model_path)
    if _model_name_has_size(base, requested_size):
        return base

    parent = base.parent
    stem = base.stem
    suffix = base.suffix

    candidates = [
        parent / f"{stem}_{requested_size}{suffix}",
        parent / f"best_{requested_size}{suffix}",
        app_root / "model" / f"best_{requested_size}{suffix}",
        app_root / "weights" / f"best_{requested_size}{suffix}",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if requested_size == base_input_size:
        return base
    return None


def _model_name_has_size(path: Path, size: int) -> bool:
    token = str(size)
    return token in path.stem


def _normalize_nan(result: BenchmarkResult) -> BenchmarkResult:
    def norm(v: float | None) -> float | None:
        if v is None:
            return None
        return None if isinstance(v, float) and np.isnan(v) else v

    return BenchmarkResult(
        input_size=result.input_size,
        frames_captured=result.frames_captured,
        frames_inferred=result.frames_inferred,
        frames_processed=result.frames_processed,
        throughput_fps=norm(result.throughput_fps),
        inference_fps_estimate=norm(result.inference_fps_estimate),
        avg_latency_ms=norm(result.avg_latency_ms),
        p95_latency_ms=norm(result.p95_latency_ms),
        cpu_process_time_ratio=norm(result.cpu_process_time_ratio),
        memory_peak_mb=norm(result.memory_peak_mb),
        elapsed_sec=result.elapsed_sec,
        status=result.status,
        message=result.message,
    )
