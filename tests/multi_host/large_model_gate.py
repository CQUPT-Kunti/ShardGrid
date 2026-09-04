from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from examples.models.large_residual_transformer import (
    LargeResidualTransformerConfig,
    build_large_residual_transformer,
    make_large_residual_batch,
)
from shardgrid.common.config import load_cluster_config, load_config_data
from shardgrid.control.job_manager import JobManager
from shardgrid.control.status_store import StatusStore
from shardgrid.planner.memory import MemoryEstimationConfig, build_model_profile

BYTES_PER_MB = 1024 * 1024
FIXED_SEED = 42


def emit_gate_marker(marker: str, **payload: object) -> None:
    event = {
        "marker": marker,
        "pid": os.getpid(),
        "thread": threading.current_thread().name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    print(json.dumps(event, sort_keys=True), flush=True)


def pytorch_pipeline_cluster_config(tmp_path: Path) -> Path:
    payload = load_config_data("examples/workers.yaml")
    payload.setdefault("backend_preference", {})["parallel_engine"] = "pytorch_pipeline"
    path = tmp_path / "workers-pytorch-pipeline.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def live_worker_inventory(config_path: Path) -> list[dict[str, Any]]:
    config = load_cluster_config(config_path)
    manager = JobManager(config, source_root=Path(__file__).resolve().parents[2])
    inventory: list[dict[str, Any]] = []
    for worker in config.workers:
        result = manager._probe_worker(worker)
        resource = result.worker_resource
        if not resource.enabled:
            continue
        total_bytes = (
            None
            if resource.gpu_total_memory is None
            else int(resource.gpu_total_memory) * BYTES_PER_MB
        )
        free_bytes = (
            None
            if resource.gpu_free_memory is None
            else int(resource.gpu_free_memory) * BYTES_PER_MB
        )
        used_bytes = (
            None
            if total_bytes is None or free_bytes is None
            else max(total_bytes - free_bytes, 0)
        )
        inventory.append(
            {
                "worker_id": str(resource.worker_id),
                "gpu_index": 0,
                "gpu_name": resource.gpu_name,
                "gpu_total_memory_mb": resource.gpu_total_memory,
                "gpu_free_memory_mb": resource.gpu_free_memory,
                "gpu_used_memory_mb": (
                    None
                    if resource.gpu_total_memory is None or resource.gpu_free_memory is None
                    else max(int(resource.gpu_total_memory) - int(resource.gpu_free_memory), 0)
                ),
                "gpu_total_memory_bytes": total_bytes,
                "gpu_free_memory_bytes": free_bytes,
                "gpu_used_memory_bytes": used_bytes,
                "gpu_utilization": resource.gpu_utilization,
            }
        )
    return inventory


def live_worker_memory(config_path: Path) -> dict[str, int]:
    return {
        str(item["worker_id"]): int(item["gpu_free_memory_bytes"])
        for item in live_worker_inventory(config_path)
        if item["gpu_free_memory_bytes"] is not None
    }


def large_b_parameters(memory: dict[str, int]) -> dict[str, object]:
    usable = sorted(memory.values(), reverse=True)
    if len(usable) < 2:
        raise RuntimeError("Large-B requires at least two live GPU workers")
    max_single = usable[0]
    min_pair = usable[1]
    lower = int(max_single * 1.08 / 4)
    upper = int(min_pair * 0.80 / 2)
    if lower >= upper:
        raise RuntimeError("BLOCKED_BY_AVAILABLE_HARDWARE: no safe Large-B memory window")
    hidden_size = 512
    num_layers = 4
    bank_bytes = lower
    rows = math.ceil(bank_bytes / (hidden_size * 4 * num_layers))
    return {
        "vocab_size": 2048,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "num_heads": 8,
        "ffn_size": 2048,
        "sequence_length": 8,
        "batch_size": 4,
        "microbatch_count": 2,
        "training_steps": 1,
        "learning_rate": "1e-3",
        "memory_bank_rows": rows,
        "memory_bank_touch_rows": 1,
    }


def estimate_large_model(parameters: dict[str, object]) -> dict[str, object]:
    config = LargeResidualTransformerConfig.from_mapping(parameters)
    model = build_large_residual_transformer(config, seed=FIXED_SEED, device="meta")
    inputs, _targets = make_large_residual_batch(
        config,
        seed=FIXED_SEED,
        step=0,
        device="meta",
    )
    memory_config = MemoryEstimationConfig(
        optimizer_type="adamw",
        gradient_dtype="float32",
        optimizer_state_dtype="float32",
        runtime_overhead_bytes=1024,
        communication_buffer_bytes=2048,
        safety_headroom_bytes=4096,
        temporary_buffer_factor=0.25,
    )
    profile = build_model_profile(
        model,
        engine_id="pytorch_pipeline",
        model_name="large-residual-transformer",
        sample_args=(inputs,),
        memory_config=memory_config,
        required_backends=("nccl",),
    )
    return {
        "parameters": model.parameter_count(),
        "estimated_training_peak": profile.total_memory.planner_required_bytes,
        "config": config,
    }


def _rows_for_fraction(
    smallest_free_bytes: int,
    *,
    hidden_size: int,
    num_layers: int,
    fraction: float,
    minimum_rows: int,
) -> int:
    bytes_per_row = hidden_size * 4 * num_layers
    return max(minimum_rows, int((smallest_free_bytes * fraction) // bytes_per_row))


def three_worker_model_candidates(memory: dict[str, int]) -> list[dict[str, object]]:
    usable = sorted(memory.values())
    if len(usable) < 3:
        raise RuntimeError("three-worker gate requires three live GPU workers")
    smallest = usable[0]
    candidates: list[dict[str, object]] = []
    seen: set[tuple[int, int, int, int]] = set()
    templates = (
        (512, 4, 2048, (0.58, 0.54, 0.50, 0.46), 120000),
        (512, 3, 1536, (0.68, 0.60, 0.52), 90000),
        (384, 4, 1536, (0.72, 0.64, 0.56), 90000),
    )
    for hidden_size, num_layers, ffn_size, fractions, minimum_rows in templates:
        for fraction in fractions:
            rows = _rows_for_fraction(
                smallest,
                hidden_size=hidden_size,
                num_layers=num_layers,
                fraction=fraction,
                minimum_rows=minimum_rows,
            )
            key = (hidden_size, num_layers, ffn_size, rows)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                {
                    "vocab_size": 2048,
                    "hidden_size": hidden_size,
                    "num_layers": num_layers,
                    "num_heads": max(4, hidden_size // 64),
                    "ffn_size": ffn_size,
                    "sequence_length": 8,
                    "batch_size": 4,
                    "microbatch_count": 2,
                    "training_steps": 8,
                    "learning_rate": "1e-3",
                    "memory_bank_rows": rows,
                    "memory_bank_touch_rows": 1,
                }
            )
    return candidates


def medium_model_candidates(memory: dict[str, int]) -> list[dict[str, object]]:
    usable = sorted(memory.values())
    if len(usable) < 2:
        raise RuntimeError("multi-job sharing requires at least two live GPU workers")
    smallest = usable[0]
    candidates: list[dict[str, object]] = []
    for hidden_size, num_layers, ffn_size, fractions in (
        (512, 3, 1536, (0.42, 0.34)),
        (384, 3, 1536, (0.46, 0.38)),
        (384, 2, 1024, (0.50, 0.42)),
    ):
        for fraction in fractions:
            candidates.append(
                {
                    "vocab_size": 2048,
                    "hidden_size": hidden_size,
                    "num_layers": num_layers,
                    "num_heads": max(4, hidden_size // 64),
                    "ffn_size": ffn_size,
                    "sequence_length": 8,
                    "batch_size": 4,
                    "microbatch_count": 2,
                    "training_steps": 40,
                    "learning_rate": "1e-3",
                    "memory_bank_rows": _rows_for_fraction(
                        smallest,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        fraction=fraction,
                        minimum_rows=40000,
                    ),
                    "memory_bank_touch_rows": 1,
                }
            )
    return candidates


def small_model_candidates(memory: dict[str, int]) -> list[dict[str, object]]:
    usable = sorted(memory.values())
    if len(usable) < 2:
        raise RuntimeError("multi-job sharing requires at least two live GPU workers")
    smallest = usable[0]
    candidates: list[dict[str, object]] = []
    for hidden_size, num_layers, ffn_size, fractions in (
        (384, 2, 1024, (0.28, 0.22)),
        (256, 3, 768, (0.34, 0.28)),
        (256, 2, 512, (0.38, 0.30)),
    ):
        for fraction in fractions:
            candidates.append(
                {
                    "vocab_size": 2048,
                    "hidden_size": hidden_size,
                    "num_layers": num_layers,
                    "num_heads": max(4, hidden_size // 64),
                    "ffn_size": ffn_size,
                    "sequence_length": 8,
                    "batch_size": 4,
                    "microbatch_count": 2,
                    "training_steps": 40,
                    "learning_rate": "1e-3",
                    "memory_bank_rows": _rows_for_fraction(
                        smallest,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        fraction=fraction,
                        minimum_rows=16000,
                    ),
                    "memory_bank_touch_rows": 1,
                }
            )
    return candidates


def write_training_config(
    tmp_path: Path,
    parameters: dict[str, object],
    *,
    name: str,
    stage_count: int = 2,
    world_size: int | None = None,
    preferred_workers: list[str] | None = None,
) -> Path:
    selected_world_size = world_size or stage_count
    selected_workers = preferred_workers or ["gpu4060", "gpu1060", "gpu4060-cqupt"]
    path = tmp_path / f"{name}.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "job": {
                    "name": name,
                    "backend": "ssh",
                    "communication_backend": "auto",
                },
                "model": {
                    "name": name,
                    "type": "large_residual_transformer",
                    "stage_count": stage_count,
                    "max_train_minutes": 15,
                    "min_loss_decrease_percent": 0.0,
                    "parameters": parameters,
                },
                "resources": {
                    "world_size": selected_world_size,
                    "preferred_workers": selected_workers,
                },
                "planning": {"mode": "automatic"},
                "artifacts": {
                    "snapshot_name": name,
                    "keep_failed_snapshots": True,
                    "transport": "auto",
                    "checkpoint": {
                        "consolidation": {
                            "enabled": False,
                            "device": "auto",
                            "required": False,
                        }
                    },
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def train_command(config_path: Path, training_path: Path, *, dry_run: bool = False) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "shardgrid.cli.app",
        "--config",
        str(config_path),
        "train",
        str(training_path),
        "--json",
    ]
    if dry_run:
        command.append("--dry-run")
    return command


def train_env(extra_env: dict[str, str] | None = None) -> dict[str, str]:
    repo = Path(__file__).resolve().parents[2]
    return {**os.environ, "PYTHONPATH": str(repo / "src"), **(extra_env or {})}


def run_train(
    config_path: Path,
    training_path: Path,
    *,
    timeout: int = 900,
    dry_run: bool = False,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    repo = Path(__file__).resolve().parents[2]
    return subprocess.run(
        train_command(config_path, training_path, dry_run=dry_run),
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=train_env(extra_env),
    )


def start_train(
    config_path: Path,
    training_path: Path,
    *,
    extra_env: dict[str, str] | None = None,
) -> subprocess.Popen[str]:
    repo = Path(__file__).resolve().parents[2]
    return subprocess.Popen(
        train_command(config_path, training_path),
        cwd=repo,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
        env=train_env(extra_env),
    )


def jobs_root(config_path: Path) -> Path:
    return load_cluster_config(config_path).jobs_root


def wait_for_reservation(root: Path, *, timeout: int = 180) -> list[dict[str, Any]]:
    reservation_path = root / "resource-reservations.json"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if reservation_path.exists():
            payload = json.loads(reservation_path.read_text(encoding="utf-8"))
            reservations = payload.get("reservations", [])
            if reservations:
                return list(reservations)
        time.sleep(1)
    return []


def active_reservations(root: Path) -> list[dict[str, Any]]:
    return StatusStore(root).active_reservations()


def job_snapshot_root(root: Path, job_id: str) -> Path:
    return root / str(job_id)


def job_status_payload(root: Path, job_id: str) -> dict[str, Any]:
    path = root / str(job_id) / "job-status.json"
    return json.loads(path.read_text(encoding="utf-8"))


def snapshot_metadata_payload(root: Path, job_id: str) -> dict[str, Any]:
    path = job_snapshot_root(root, job_id) / "diagnostics" / "snapshot-metadata.json"
    return json.loads(path.read_text(encoding="utf-8"))


def execution_plan_payload(root: Path, job_id: str) -> dict[str, Any]:
    path = job_snapshot_root(root, job_id) / "plan" / "execution-plan.json"
    return json.loads(path.read_text(encoding="utf-8"))


def monitor_payloads(root: Path, job_id: str) -> list[dict[str, Any]]:
    diagnostics = job_snapshot_root(root, job_id) / "diagnostics"
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(diagnostics.glob("monitor-*.json"))
    ]


def _rank_steps(monitors: list[dict[str, Any]]) -> dict[int, int]:
    steps: dict[int, int] = {}
    for payload in monitors:
        train = payload.get("train")
        if not isinstance(train, dict):
            continue
        rank = payload.get("rank")
        if rank is None:
            rank = train.get("rank")
        if rank is None:
            continue
        steps[int(rank)] = int(train.get("steps", 0))
    return steps


def stop_payloads(root: Path, job_id: str) -> list[dict[str, Any]]:
    diagnostics = job_snapshot_root(root, job_id) / "diagnostics"
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(diagnostics.glob("stop-*.json"))
    ]


def find_feasible_plan(
    tmp_path: Path,
    *,
    config_path: Path,
    name_prefix: str,
    candidates: list[dict[str, object]],
    expected_worker_count: str | None = None,
    extra_env: dict[str, str] | None = None,
    stage_count: int = 2,
    world_size: int | None = None,
    preferred_workers: list[str] | None = None,
    timeout: int = 900,
) -> tuple[dict[str, object], dict[str, Any], list[dict[str, Any]]]:
    attempts: list[dict[str, Any]] = []
    for index, parameters in enumerate(candidates):
        training_path = write_training_config(
            tmp_path,
            parameters,
            name=f"{name_prefix}-scan-{index}",
            stage_count=stage_count,
            world_size=world_size,
            preferred_workers=preferred_workers,
        )
        result = run_train(
            config_path,
            training_path,
            timeout=timeout,
            dry_run=True,
            extra_env=extra_env,
        )
        payload = json.loads(result.stdout)
        planning = payload.get("planning", {})
        attempt = {
            "candidate_index": index,
            "job_id": payload.get("job_id"),
            "parameters": parameters,
            "returncode": result.returncode,
            "state": payload.get("state"),
            "failure": payload.get("failure"),
            "selected_worker_count": planning.get("selected_worker_count"),
            "attempted_worker_counts": planning.get("attempted_worker_counts"),
        }
        attempts.append(attempt)
        if result.returncode != 0:
            continue
        if expected_worker_count is not None and planning.get("selected_worker_count") != expected_worker_count:
            continue
        return parameters, payload, attempts
    raise RuntimeError(
        json.dumps(
            {
                "expected_worker_count": expected_worker_count,
                "attempts": attempts,
            },
            indent=2,
            sort_keys=True,
        )
    )


def wait_for_job_ids(
    root: Path,
    *,
    expected: int,
    known: set[str] | None = None,
    timeout: int = 180,
) -> list[str]:
    known_ids = set(known or ())
    deadline = time.time() + timeout
    while time.time() < deadline:
        discovered = sorted(
            path.name
            for path in root.iterdir()
            if path.is_dir()
            and path.name not in known_ids
            and (path / "job-status.json").exists()
        )
        if len(discovered) >= expected:
            return discovered[:expected]
        time.sleep(1)
    return []


def wait_for_job_training_steps(
    root: Path,
    job_id: str,
    *,
    min_steps: int,
    timeout: int = 600,
) -> list[dict[str, Any]]:
    emit_gate_marker(
        "WAIT_FOR_JOB_TRAINING_STEPS_START",
        job_id=job_id,
        min_steps=min_steps,
        timeout=timeout,
    )
    deadline = time.time() + timeout
    last_state: str | None = None
    last_steps: dict[int, int] = {}
    last_process_states: dict[int, str | None] = {}
    last_monitor_count = 0
    while time.time() < deadline:
        status = job_status_payload(root, job_id)
        monitors = monitor_payloads(root, job_id)
        expected_ranks = max(
            int(status.get("world_size") or 0),
            len(status.get("assignments") or ()),
            len(status.get("workers") or ()),
        )
        steps = _rank_steps(monitors)
        last_state = str(status.get("state"))
        last_steps = steps
        last_monitor_count = len(monitors)
        last_process_states = {
            int(payload.get("rank")): (
                None if payload.get("process_state") is None else str(payload.get("process_state"))
            )
            for payload in monitors
            if payload.get("rank") is not None
        }
        if (
            last_state == "training"
            and expected_ranks > 0
            and len(steps) >= expected_ranks
            and min(steps.values()) >= min_steps
        ):
            emit_gate_marker(
                "WAIT_FOR_JOB_TRAINING_STEPS_READY",
                job_id=job_id,
                expected_ranks=expected_ranks,
                steps=steps,
            )
            return monitors
        time.sleep(2)
    emit_gate_marker(
        "WAIT_FOR_JOB_TRAINING_STEPS_TIMEOUT",
        job_id=job_id,
        min_steps=min_steps,
        state=last_state,
        steps=last_steps,
        process_states=last_process_states,
        monitor_count=last_monitor_count,
    )
    return []


def wait_for_job_terminal(
    root: Path,
    job_id: str,
    *,
    timeout: int = 1800,
) -> dict[str, Any] | None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            status = job_status_payload(root, job_id)
        except FileNotFoundError:
            time.sleep(1)
            continue
        if status.get("state") in {"completed", "failed", "stopped"}:
            return status
        time.sleep(2)
    return None


def stop_job(config_path: Path, job_id: str) -> subprocess.CompletedProcess[str]:
    repo = Path(__file__).resolve().parents[2]
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "shardgrid.cli.app",
            "--config",
            str(config_path),
            "stop",
            job_id,
            "--yes",
            "--json",
        ],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
        env=train_env(),
    )
