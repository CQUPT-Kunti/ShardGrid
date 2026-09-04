from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
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
from shardgrid.planner.memory import MemoryEstimationConfig, build_model_profile

BYTES_PER_MB = 1024 * 1024
FIXED_SEED = 42


def pytorch_pipeline_cluster_config(tmp_path: Path) -> Path:
    payload = load_config_data("examples/workers.yaml")
    payload.setdefault("backend_preference", {})["parallel_engine"] = "pytorch_pipeline"
    path = tmp_path / "workers-pytorch-pipeline.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def live_worker_memory(config_path: Path) -> dict[str, int]:
    config = load_cluster_config(config_path)
    manager = JobManager(config, source_root=Path(__file__).resolve().parents[2])
    memory: dict[str, int] = {}
    for worker in config.workers:
        result = manager._probe_worker(worker)
        resource = result.worker_resource
        if not resource.enabled or resource.gpu_free_memory is None:
            continue
        memory[str(resource.worker_id)] = int(resource.gpu_free_memory) * BYTES_PER_MB
    return memory


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


def write_training_config(tmp_path: Path, parameters: dict[str, object], *, name: str) -> Path:
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
                    "stage_count": 2,
                    "max_train_minutes": 15,
                    "min_loss_decrease_percent": 0.0,
                    "parameters": parameters,
                },
                "resources": {
                    "world_size": 4,
                    "preferred_workers": ["gpu4060", "gpu1060", "gpu4060-cqupt"],
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


def train_env() -> dict[str, str]:
    repo = Path(__file__).resolve().parents[2]
    return {**os.environ, "PYTHONPATH": str(repo / "src")}


def run_train(
    config_path: Path,
    training_path: Path,
    *,
    timeout: int = 900,
    dry_run: bool = False,
) -> subprocess.CompletedProcess[str]:
    repo = Path(__file__).resolve().parents[2]
    return subprocess.run(
        train_command(config_path, training_path, dry_run=dry_run),
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=train_env(),
    )


def start_train(config_path: Path, training_path: Path) -> subprocess.Popen[str]:
    repo = Path(__file__).resolve().parents[2]
    return subprocess.Popen(
        train_command(config_path, training_path),
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=train_env(),
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
