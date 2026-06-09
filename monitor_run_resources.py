#!/usr/bin/env python3
"""Monitor CPU/GPU usage for one field delineation run.

The monitor is intentionally external to the pipeline so it can be attached to
an already-running nohup job. It samples Linux process CPU/RSS with ps,
samples NVIDIA GPUs with nvidia-smi when available, and infers the current
pipeline stage/tile from the run logs.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import shutil
import signal
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


PIPELINE_STEP_RE = re.compile(r"\|\s+(?:INFO|ERROR)\s+\|\s+(START|DONE|FAILED)\s+([A-Za-z0-9_.-]+)")
TILE_RE = re.compile(r"\b[0-9]{2}[A-Z]{3}\b")
RESOURCE_PROCESS_RE = re.compile(
    r"(pipeline\.py|delineate\.py|postprocess_instance_rasters\.py|vector_tile_viewer\.py|scripts/infer\.py)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path, help="Run directory to monitor.")
    parser.add_argument(
        "--output-dir",
        default=None,
        type=Path,
        help="Output directory. Defaults to <run-dir>/logs/resource_monitor.",
    )
    parser.add_argument("--interval", default=10.0, type=float, help="Sampling interval in seconds.")
    parser.add_argument("--max-samples", default=None, type=int, help="Optional stop after N samples.")
    parser.add_argument(
        "--gpu-util-threshold",
        default=5.0,
        type=float,
        help="GPU utilization percent considered busy for summary accounting.",
    )
    parser.add_argument(
        "--cpu-threshold",
        default=10.0,
        type=float,
        help="Tracked process CPU percent sum considered active for summary accounting.",
    )
    parser.add_argument(
        "--process-command-filter",
        default=None,
        help="Optional extra regex. Commands must match this or the run dir/run name.",
    )
    parser.add_argument("--no-gpu", action="store_true", help="Skip nvidia-smi sampling.")
    parser.add_argument("--print-status", action="store_true", help="Print one compact line per sample.")
    return parser.parse_args()


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def run_command(command: list[str]) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(command, check=False, text=True, capture_output=True)
    except (FileNotFoundError, PermissionError, OSError):
        return None


def parse_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value.strip())
    except (TypeError, ValueError):
        return default


def parse_int(value: str, default: int = 0) -> int:
    try:
        return int(float(value.strip()))
    except (TypeError, ValueError):
        return default


def list_processes(run_dir: Path, extra_filter: re.Pattern[str] | None) -> list[dict[str, Any]]:
    result = run_command(["ps", "-eo", "pid=,ppid=,pcpu=,pmem=,rss=,etimes=,args="])
    if result is None or result.returncode != 0:
        return []

    run_dir_text = str(run_dir)
    run_name = run_dir.name
    rows: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        parts = line.strip().split(None, 6)
        if len(parts) < 7:
            continue
        pid, ppid, pcpu, pmem, rss_kb, etimes, command = parts
        command_matches = (
            run_dir_text in command
            or run_name in command
            or (extra_filter is not None and extra_filter.search(command) is not None)
        )
        if not command_matches:
            continue
        if RESOURCE_PROCESS_RE.search(command) is None and run_name not in command and run_dir_text not in command:
            continue
        rows.append(
            {
                "pid": parse_int(pid),
                "ppid": parse_int(ppid),
                "cpu_percent": parse_float(pcpu),
                "mem_percent": parse_float(pmem),
                "rss_mb": parse_int(rss_kb) / 1024.0,
                "elapsed_process_s": parse_int(etimes),
                "command": command,
            }
        )
    return rows


def query_gpu_devices(no_gpu: bool) -> tuple[list[dict[str, Any]], str | None]:
    if no_gpu:
        return [], None
    if shutil.which("nvidia-smi") is None:
        return [], "nvidia-smi not found"

    command = [
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw",
        "--format=csv,noheader,nounits",
    ]
    result = run_command(command)
    if result is None or result.returncode != 0:
        error = result.stderr.strip() if result is not None else "nvidia-smi failed"
        return [], error

    rows: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 6:
            continue
        rows.append(
            {
                "gpu_index": parse_int(fields[0]),
                "gpu_util_percent": parse_float(fields[1]),
                "gpu_mem_util_percent": parse_float(fields[2]),
                "gpu_mem_used_mb": parse_float(fields[3]),
                "gpu_mem_total_mb": parse_float(fields[4]),
                "gpu_power_w": parse_float(fields[5]),
            }
        )
    return rows, None


def query_gpu_apps(no_gpu: bool) -> list[dict[str, Any]]:
    if no_gpu or shutil.which("nvidia-smi") is None:
        return []

    commands = [
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory,gpu_uuid",
            "--format=csv,noheader,nounits",
        ],
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
    ]
    result = None
    for command in commands:
        result = run_command(command)
        if result is not None and result.returncode == 0:
            break
    if result is None or result.returncode != 0:
        return []

    rows: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 3:
            continue
        rows.append(
            {
                "pid": parse_int(fields[0]),
                "process_name": fields[1],
                "gpu_used_memory_mb": parse_float(fields[2]),
                "gpu_uuid": fields[3] if len(fields) > 3 else "",
            }
        )
    return rows


def infer_current_status(run_dir: Path) -> dict[str, Any]:
    pipeline_log = run_dir / "logs" / "pipeline.log"
    active_steps: list[str] = []
    latest_line = ""
    current_stage = "unknown"
    current_tile = ""

    if pipeline_log.exists():
        try:
            lines = pipeline_log.read_text(errors="replace").splitlines()
        except OSError:
            lines = []
        for line in lines:
            match = PIPELINE_STEP_RE.search(line)
            if not match:
                continue
            action, step = match.groups()
            latest_line = line
            if action == "START":
                active_steps.append(step)
            elif action in {"DONE", "FAILED"}:
                for index in range(len(active_steps) - 1, -1, -1):
                    if active_steps[index] == step:
                        del active_steps[index]
                        break
                if action == "FAILED":
                    active_steps.append(f"FAILED:{step}")

    if active_steps:
        current_stage = active_steps[-1]
    elif latest_line:
        match = PIPELINE_STEP_RE.search(latest_line)
        if match:
            current_stage = match.group(2)

    tile_match = TILE_RE.search(current_stage)
    if tile_match:
        current_tile = tile_match.group(0)

    # During the monolithic delineation stage, the backend log often contains
    # the freshest tile/region clue.
    latest_backend_line = ""
    backend_logs = sorted(run_dir.glob("*delineation*.log")) + sorted(run_dir.glob("*detectron2*.log"))
    for log_path in backend_logs:
        try:
            with log_path.open("rb") as handle:
                handle.seek(0, 2)
                size = handle.tell()
                handle.seek(max(0, size - 65536), 0)
                tail = handle.read().decode(errors="replace").splitlines()
        except OSError:
            continue
        for line in tail:
            if line.strip():
                latest_backend_line = line
                tile_match = TILE_RE.search(line)
                if tile_match:
                    current_tile = tile_match.group(0)

    return {
        "current_stage": current_stage,
        "current_tile": current_tile,
        "latest_pipeline_line": latest_line,
        "latest_backend_line": latest_backend_line,
    }


def open_csv(path: Path, fieldnames: list[str]) -> tuple[Any, csv.DictWriter]:
    exists = path.exists() and path.stat().st_size > 0
    handle = path.open("a", encoding="utf-8", newline="")
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    if not exists:
        writer.writeheader()
    return handle, writer


def write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def summarize(
    *,
    started_at: float,
    sample_count: int,
    cpu_active_seconds: float,
    gpu_attached_seconds: float,
    gpu_busy_seconds: float,
    stage_seconds: dict[str, float],
    max_cpu_percent_sum: float,
    max_rss_mb_sum: float,
    max_gpu_mem_used_mb_sum: float,
    max_gpu_util_percent: float,
    current_status: dict[str, Any],
) -> dict[str, Any]:
    elapsed = max(0.0, time.time() - started_at)
    return {
        "updated_at": utc_now(),
        "elapsed_s": elapsed,
        "sample_count": sample_count,
        "cpu_active_seconds": cpu_active_seconds,
        "gpu_attached_seconds": gpu_attached_seconds,
        "gpu_busy_seconds_global": gpu_busy_seconds,
        "max_cpu_percent_sum": max_cpu_percent_sum,
        "max_rss_mb_sum": max_rss_mb_sum,
        "max_gpu_mem_used_mb_sum": max_gpu_mem_used_mb_sum,
        "max_gpu_util_percent_global": max_gpu_util_percent,
        "stage_seconds": dict(sorted(stage_seconds.items())),
        "current": current_status,
        "notes": [
            "gpu_busy_seconds_global is based on device utilization and may include other jobs.",
            "gpu_attached_seconds counts samples where a tracked process appeared in nvidia-smi compute apps.",
            "current_tile is inferred from log lines; backend postprocessing may not always expose tile-level progress.",
        ],
    }


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.exists():
        print(f"Run directory does not exist: {run_dir}", file=sys.stderr)
        return 2

    output_dir = (args.output_dir.expanduser().resolve() if args.output_dir else run_dir / "logs" / "resource_monitor")
    output_dir.mkdir(parents=True, exist_ok=True)

    extra_filter = re.compile(args.process_command_filter) if args.process_command_filter else None
    started_at = time.time()
    stop_requested = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    summary_fields = [
        "timestamp",
        "elapsed_s",
        "current_stage",
        "current_tile",
        "process_count",
        "tracked_pids",
        "cpu_percent_sum",
        "rss_mb_sum",
        "gpu_process_pids",
        "gpu_attached",
        "gpu_util_percent_max",
        "gpu_util_percent_avg",
        "gpu_mem_used_mb_sum",
        "gpu_power_w_sum",
    ]
    process_fields = [
        "timestamp",
        "pid",
        "ppid",
        "cpu_percent",
        "mem_percent",
        "rss_mb",
        "elapsed_process_s",
        "command",
    ]
    gpu_fields = [
        "timestamp",
        "gpu_index",
        "gpu_util_percent",
        "gpu_mem_util_percent",
        "gpu_mem_used_mb",
        "gpu_mem_total_mb",
        "gpu_power_w",
    ]

    summary_handle, summary_writer = open_csv(output_dir / "resource_samples.csv", summary_fields)
    process_handle, process_writer = open_csv(output_dir / "process_samples.csv", process_fields)
    gpu_handle, gpu_writer = open_csv(output_dir / "gpu_samples.csv", gpu_fields)

    sample_count = 0
    cpu_active_seconds = 0.0
    gpu_attached_seconds = 0.0
    gpu_busy_seconds = 0.0
    stage_seconds: dict[str, float] = defaultdict(float)
    max_cpu_percent_sum = 0.0
    max_rss_mb_sum = 0.0
    max_gpu_mem_used_mb_sum = 0.0
    max_gpu_util_percent = 0.0
    last_sample_time = time.time()
    current_status: dict[str, Any] = {}

    try:
        while not stop_requested:
            now = time.time()
            delta = max(0.0, now - last_sample_time)
            last_sample_time = now
            timestamp = utc_now()
            elapsed_s = now - started_at

            processes = list_processes(run_dir, extra_filter)
            gpu_devices, gpu_error = query_gpu_devices(args.no_gpu)
            gpu_apps = query_gpu_apps(args.no_gpu)
            current_status = infer_current_status(run_dir)

            tracked_pids = {row["pid"] for row in processes}
            gpu_process_pids = {row["pid"] for row in gpu_apps if row["pid"] in tracked_pids}
            cpu_percent_sum = sum(row["cpu_percent"] for row in processes)
            rss_mb_sum = sum(row["rss_mb"] for row in processes)
            gpu_util_values = [row["gpu_util_percent"] for row in gpu_devices]
            gpu_util_max = max(gpu_util_values) if gpu_util_values else 0.0
            gpu_util_avg = sum(gpu_util_values) / len(gpu_util_values) if gpu_util_values else 0.0
            gpu_mem_used_mb_sum = sum(row["gpu_mem_used_mb"] for row in gpu_devices)
            gpu_power_w_sum = sum(row["gpu_power_w"] for row in gpu_devices)

            sample_count += 1
            if cpu_percent_sum >= args.cpu_threshold:
                cpu_active_seconds += delta
            if gpu_process_pids:
                gpu_attached_seconds += delta
            if gpu_util_max >= args.gpu_util_threshold:
                gpu_busy_seconds += delta
            stage_seconds[current_status["current_stage"]] += delta

            max_cpu_percent_sum = max(max_cpu_percent_sum, cpu_percent_sum)
            max_rss_mb_sum = max(max_rss_mb_sum, rss_mb_sum)
            max_gpu_mem_used_mb_sum = max(max_gpu_mem_used_mb_sum, gpu_mem_used_mb_sum)
            max_gpu_util_percent = max(max_gpu_util_percent, gpu_util_max)

            summary_row = {
                "timestamp": timestamp,
                "elapsed_s": round(elapsed_s, 3),
                "current_stage": current_status["current_stage"],
                "current_tile": current_status["current_tile"],
                "process_count": len(processes),
                "tracked_pids": " ".join(str(pid) for pid in sorted(tracked_pids)),
                "cpu_percent_sum": round(cpu_percent_sum, 3),
                "rss_mb_sum": round(rss_mb_sum, 3),
                "gpu_process_pids": " ".join(str(pid) for pid in sorted(gpu_process_pids)),
                "gpu_attached": bool(gpu_process_pids),
                "gpu_util_percent_max": round(gpu_util_max, 3),
                "gpu_util_percent_avg": round(gpu_util_avg, 3),
                "gpu_mem_used_mb_sum": round(gpu_mem_used_mb_sum, 3),
                "gpu_power_w_sum": round(gpu_power_w_sum, 3),
            }
            summary_writer.writerow(summary_row)

            for row in processes:
                process_writer.writerow({"timestamp": timestamp, **row})
            for row in gpu_devices:
                gpu_writer.writerow({"timestamp": timestamp, **row})

            summary_handle.flush()
            process_handle.flush()
            gpu_handle.flush()

            live_status = {
                **summary_row,
                "run_dir": str(run_dir),
                "gpu_error": gpu_error,
                "latest_pipeline_line": current_status["latest_pipeline_line"],
                "latest_backend_line": current_status["latest_backend_line"],
            }
            write_json(output_dir / "current_status.json", live_status)
            write_json(
                output_dir / "summary.json",
                summarize(
                    started_at=started_at,
                    sample_count=sample_count,
                    cpu_active_seconds=cpu_active_seconds,
                    gpu_attached_seconds=gpu_attached_seconds,
                    gpu_busy_seconds=gpu_busy_seconds,
                    stage_seconds=stage_seconds,
                    max_cpu_percent_sum=max_cpu_percent_sum,
                    max_rss_mb_sum=max_rss_mb_sum,
                    max_gpu_mem_used_mb_sum=max_gpu_mem_used_mb_sum,
                    max_gpu_util_percent=max_gpu_util_percent,
                    current_status=current_status,
                ),
            )

            if args.print_status:
                tile = current_status["current_tile"] or "-"
                gpu_text = "gpu=na" if gpu_error else f"gpu_max={gpu_util_max:.0f}%"
                print(
                    f"{timestamp} stage={current_status['current_stage']} tile={tile} "
                    f"cpu_sum={cpu_percent_sum:.0f}% rss={rss_mb_sum:.0f}MB {gpu_text}",
                    flush=True,
                )

            if args.max_samples is not None and sample_count >= args.max_samples:
                break
            time.sleep(args.interval)
    finally:
        summary_handle.close()
        process_handle.close()
        gpu_handle.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
