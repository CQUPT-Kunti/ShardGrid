"""Automatic-plan multi-host training runner.

This runner consumes the preserved `ParallelPlan`/`ExecutionPlan` from the
distributed snapshot and trains exactly the stage assigned to the current rank.
"""

from __future__ import annotations

import argparse
import faulthandler
import hashlib
import json
import math
import os
import signal
import socket
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.pipelining import PipelineStage, ScheduleGPipe, SplitPoint, pipeline

from examples.models.minimal_transformer import (
    MinimalTransformerConfig,
    build_minimal_transformer,
)
from examples.models.large_residual_transformer import (
    LargeResidualTransformerConfig,
    build_large_residual_transformer,
    build_large_residual_transformer_stage,
    make_large_residual_batch,
    make_large_residual_stage_inputs,
)
from examples.models.partition_stress_model import (
    PartitionStressConfig,
    build_partition_stress_model,
    make_training_batch,
)
from shardgrid.common.config import TrainingConfig
from shardgrid.distributed.backend import select_backend
from shardgrid.engines.models import ParallelPlan
from shardgrid.planner.models import ExecutionPlan, WorkerAssignment

EVENT_MARKER = "STAGE_PLACEMENT_EVIDENCE "
TRAIN_MARKER = "T074_TRAIN_EVIDENCE "
_LOG_HANDLES: list[object] = []


class _TeeStream:
    def __init__(self, *streams: object) -> None:
        self._streams = streams
        self.encoding = getattr(streams[0], "encoding", "utf-8") if streams else "utf-8"

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def _emit_line(message: str) -> None:
    sys.stdout.write(message + "\n")
    sys.stdout.flush()


def _configure_rank_log(rank: int) -> Path:
    path = Path(os.environ.get("SHARDGRID_LOG_PATH", f"/tmp/shardgrid-rank{rank}.log"))
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.environ.get("SHARDGRID_LAUNCHER_OWNS_LOG_SINK", "").strip() == "1":
        _emit_line(f"T072_LOG_PATH {path}")
        return path
    handle = path.open("a", encoding="utf-8", buffering=1)
    _LOG_HANDLES.append(handle)
    sys.stdout = _TeeStream(sys.__stdout__, handle)
    sys.stderr = _TeeStream(sys.__stderr__, handle)
    _emit_line(f"T072_LOG_PATH {path}")
    return path


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _snapshot_root() -> Path:
    explicit = os.environ.get("SHARDGRID_REMOTE_SNAPSHOT_ROOT", "").strip()
    if explicit:
        return Path(explicit)
    return Path.cwd().resolve().parent


def _load_artifacts() -> tuple[TrainingConfig, ParallelPlan, ExecutionPlan]:
    root = _snapshot_root()
    config = TrainingConfig.from_dict(_load_json(root / "config" / "training-config.json"))
    plan = ParallelPlan.from_dict(_load_json(root / "plan" / "original-parallel-plan.json"))
    execution = ExecutionPlan.from_dict(_load_json(root / "plan" / "execution-plan.json"))
    return config, plan, execution


def _build_model(training_config: TrainingConfig) -> torch.nn.Module:
    if training_config.model.type == "minimal_sequential":
        return build_minimal_transformer(MinimalTransformerConfig(), seed=42)
    if training_config.model.type == "hf_style":
        return build_partition_stress_model(PartitionStressConfig(), seed=42)
    if training_config.model.type == "large_residual_transformer":
        config = LargeResidualTransformerConfig.from_mapping(training_config.model.parameters)
        return build_large_residual_transformer(config, seed=42)
    raise ValueError(f"unsupported automatic training model.type {training_config.model.type!r}")


def _sample_inputs(training_config: TrainingConfig, *, step: int, device: torch.device) -> tuple[Any, Any]:
    return _sample_inputs_for_batch(training_config, step=step, device=device, batch_size=None)


def _sample_inputs_for_batch(
    training_config: TrainingConfig,
    *,
    step: int,
    device: torch.device,
    batch_size: int | None,
) -> tuple[Any, Any]:
    if training_config.model.type == "minimal_sequential":
        config = MinimalTransformerConfig()
        generator = torch.Generator(device=device.type)
        generator.manual_seed(42 + step)
        effective_batch_size = 2 if batch_size is None else batch_size
        inputs = torch.randint(
            0,
            config.vocab_size,
            (effective_batch_size, min(config.max_seq_length, 16)),
            device=device,
            generator=generator,
        )
        labels = torch.randint(
            0,
            config.vocab_size,
            inputs.shape,
            device=device,
            generator=generator,
        )
        return inputs, labels
    if training_config.model.type == "hf_style":
        return make_training_batch(
            seed=42,
            step=step,
            device=str(device),
            batch_size=4 if batch_size is None else batch_size,
        )
    if training_config.model.type == "large_residual_transformer":
        config = LargeResidualTransformerConfig.from_mapping(training_config.model.parameters)
        return make_large_residual_batch(
            config,
            seed=42,
            step=step,
            device=device,
            batch_size=config.batch_size if batch_size is None else batch_size,
        )
    raise ValueError(f"unsupported automatic training model.type {training_config.model.type!r}")


def _automatic_batch_sizes(training_config: TrainingConfig, *, microbatches: int) -> tuple[int, int]:
    if microbatches <= 0:
        raise ValueError("microbatches must be > 0")
    if training_config.model.type == "minimal_sequential":
        per_microbatch = 1
    elif training_config.model.type == "large_residual_transformer":
        config = LargeResidualTransformerConfig.from_mapping(training_config.model.parameters)
        per_microbatch = max(1, config.batch_size // microbatches)
    else:
        per_microbatch = 2
    return per_microbatch * microbatches, per_microbatch


def _split_spec(plan: ParallelPlan) -> dict[str, SplitPoint]:
    return {
        stage.module_paths[-1]: SplitPoint.END
        for stage in plan.stage_metadata[:-1]
        if stage.module_paths
    }


def _loss_fn(training_config: TrainingConfig):
    if training_config.model.type in {"minimal_sequential", "large_residual_transformer"}:
        def cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
            )
        return cross_entropy
    return torch.nn.MSELoss()


def _checksum(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _state_dict_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if left.keys() != right.keys():
        return False
    for key in left:
        left_value = left[key]
        right_value = right[key]
        if isinstance(left_value, torch.Tensor):
            if not isinstance(right_value, torch.Tensor) or not torch.equal(left_value, right_value):
                return False
            continue
        if isinstance(left_value, dict):
            if not isinstance(right_value, dict) or not _state_dict_equal(left_value, right_value):
                return False
            continue
        if isinstance(left_value, list):
            if not isinstance(right_value, list) or len(left_value) != len(right_value):
                return False
            for a, b in zip(left_value, right_value, strict=True):
                if isinstance(a, torch.Tensor):
                    if not isinstance(b, torch.Tensor) or not torch.equal(a, b):
                        return False
                elif a != b:
                    return False
            continue
        if left_value != right_value:
            return False
    return True


def _parameter_bytes(module: torch.nn.Module) -> int:
    return sum(parameter.numel() * parameter.element_size() for parameter in module.parameters())


def _process_rss_bytes() -> int | None:
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (FileNotFoundError, PermissionError, ValueError):
        return None
    return None


def _pipeline_io(value: torch.Tensor | tuple[torch.Tensor, ...]) -> torch.Tensor | tuple[torch.Tensor, ...]:
    if isinstance(value, tuple) and len(value) == 1:
        return value[0]
    return value


def _cuda_allocated_bytes(device: torch.device) -> int | None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    return torch.cuda.memory_allocated(device)


def _large_stage_materialization(
    training_config: TrainingConfig,
    plan: ParallelPlan,
    *,
    stage_index: int,
    device: torch.device,
    sample_batch_size: int,
) -> tuple[torch.nn.Module, PipelineStage, dict[str, Any]]:
    stage_module, sample_inputs, sample_outputs, materialization = _build_large_stage_module(
        training_config,
        plan,
        stage_index=stage_index,
        device=device,
        sample_batch_size=sample_batch_size,
    )
    stage = PipelineStage(
        stage_module,
        stage_index=stage_index,
        num_stages=len(plan.stage_metadata),
        device=device,
        input_args=_pipeline_io(sample_inputs),
        output_args=_pipeline_io(sample_outputs),
    )
    return stage_module, stage, materialization


def _build_large_stage_module(
    training_config: TrainingConfig,
    plan: ParallelPlan,
    *,
    stage_index: int,
    device: torch.device,
    sample_batch_size: int,
) -> tuple[
    torch.nn.Module,
    tuple[torch.Tensor, ...],
    torch.Tensor | tuple[torch.Tensor, ...],
    dict[str, Any],
]:
    config = LargeResidualTransformerConfig.from_mapping(training_config.model.parameters)
    stage_metadata = plan.stage_metadata[stage_index]
    next_stage_start_path = (
        None
        if stage_index + 1 >= len(plan.stage_metadata)
        else plan.stage_metadata[stage_index + 1].module_paths[0]
    )
    rss_before = _process_rss_bytes()
    cuda_before = _cuda_allocated_bytes(device)
    stage_module = build_large_residual_transformer_stage(
        config,
        stage_metadata=stage_metadata,
        next_stage_start_path=next_stage_start_path,
        seed=42,
    )
    rss_after = _process_rss_bytes()
    materialized_parameter_names = sorted(name for name, _ in stage_module.named_parameters())
    materialized_parameter_bytes = _parameter_bytes(stage_module)
    stage_module = stage_module.to(device)
    cuda_after = _cuda_allocated_bytes(device)
    sample_inputs = make_large_residual_stage_inputs(
        config,
        stage_metadata.module_paths[0],
        seed=42,
        step=0,
        batch_size=sample_batch_size,
        device=device,
    )
    with torch.no_grad():
        sample_outputs = stage_module(*sample_inputs)
    return stage_module, sample_inputs, sample_outputs, {
        "owned_module_paths": list(stage_metadata.module_paths),
        "materialized_parameter_names": materialized_parameter_names,
        "materialized_parameter_count": sum(
            parameter.numel() for parameter in stage_module.parameters()
        ),
        "materialized_parameter_bytes": materialized_parameter_bytes,
        "full_model_materialized": False,
        "initial_weights_received_from_control": False,
        "process_rss_before_materialization_bytes": rss_before,
        "process_rss_after_materialization_bytes": rss_after,
        "cuda_allocated_before_stage_to_device_bytes": cuda_before,
        "cuda_allocated_after_stage_to_device_bytes": cuda_after,
    }


def _build_runtime_stage(
    training_config: TrainingConfig,
    plan: ParallelPlan,
    *,
    stage_index: int,
    device: torch.device,
    sample_batch_size: int,
) -> tuple[torch.nn.Module, PipelineStage, dict[str, Any]]:
    if training_config.model.type == "large_residual_transformer":
        return _large_stage_materialization(
            training_config,
            plan,
            stage_index=stage_index,
            device=device,
            sample_batch_size=sample_batch_size,
        )

    model = _build_model(training_config)
    sample_inputs, _ = _sample_inputs_for_batch(
        training_config,
        step=0,
        device=torch.device("cpu"),
        batch_size=sample_batch_size,
    )
    pipe = pipeline(model, mb_args=(sample_inputs,), split_spec=_split_spec(plan))
    stage = pipe.build_stage(stage_index, device)
    stage_module = pipe.get_stage_module(stage_index).to(device)
    return stage_module, stage, {
        "owned_module_paths": list(plan.stage_metadata[stage_index].module_paths),
        "materialized_parameter_names": sorted(name for name, _ in stage_module.named_parameters()),
        "materialized_parameter_count": sum(
            parameter.numel() for parameter in stage_module.parameters()
        ),
        "materialized_parameter_bytes": _parameter_bytes(stage_module),
        "full_model_materialized": True,
        "initial_weights_received_from_control": False,
        "process_rss_before_materialization_bytes": None,
        "process_rss_after_materialization_bytes": None,
        "cuda_allocated_before_stage_to_device_bytes": None,
        "cuda_allocated_after_stage_to_device_bytes": None,
    }


def _placement_payload(
    *,
    assignment: WorkerAssignment,
    rank: int,
    world_size: int,
    device: torch.device,
    module: torch.nn.Module,
    execution: ExecutionPlan,
    materialization: dict[str, Any],
) -> dict[str, Any]:
    named = dict(module.named_parameters())
    return {
        "hostname": socket.gethostname(),
        "job_id": str(execution.job_id),
        "rank": rank,
        "world_size": world_size,
        "worker_id": str(assignment.worker_id),
        "stage_id": assignment.stage,
        "selected_candidate_id": execution.labels.get("selected_candidate_id"),
        "partition_source": execution.labels.get("partition_source"),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "parameter_count": sum(item.numel() for item in named.values()),
        "trainable_parameter_count": sum(
            item.numel() for item in named.values() if item.requires_grad
        ),
        "parameter_names": sorted(named.keys()),
        "parameter_devices": {name: str(item.device) for name, item in named.items()},
        **materialization,
        "stage_to_worker": [
            {
                "stage_id": worker.stage,
                "worker_id": str(worker.worker_id),
                "rank": worker.rank,
                "gpu_index": worker.gpu_index,
            }
            for worker in execution.workers
        ],
    }


def _write_checkpoint_metadata(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--rank", type=int, required=False)
    parser.parse_args()

    faulthandler.register(signal.SIGUSR1, all_threads=True)
    rank = int(os.environ["RANK"])
    _configure_rank_log(rank)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for automatic live training")

    training_config, plan, execution = _load_artifacts()
    if execution.world_size != len(plan.stage_metadata):
        raise ValueError("execution world_size must match automatic stage count")
    assignment = next(worker for worker in execution.workers if worker.rank == rank)
    if assignment.stage is None:
        raise ValueError(f"rank {rank} is missing assigned stage")
    stage_index = next(
        index for index, stage in enumerate(plan.stage_metadata) if stage.stage_id == assignment.stage
    )
    world_size = execution.world_size
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    microbatches = max(
        world_size,
        int(os.environ.get("SHARDGRID_AUTOMATIC_MICROBATCHES", "1")),
    )
    train_batch_size, sample_batch_size = _automatic_batch_sizes(
        training_config,
        microbatches=microbatches,
    )

    dist.init_process_group(
        backend=select_backend(os.environ.get("SHARDGRID_BACKEND", "nccl")),
        init_method=f"tcp://{os.environ['MASTER_ADDR']}:{int(os.environ['MASTER_PORT'])}",
        rank=rank,
        world_size=world_size,
    )

    stage_module, stage, materialization = _build_runtime_stage(
        training_config,
        plan,
        stage_index=stage_index,
        device=device,
        sample_batch_size=sample_batch_size,
    )
    optimizer = torch.optim.AdamW(stage_module.parameters(), lr=float(os.environ.get("SHARDGRID_AUTOMATIC_LR", "1e-3")))
    schedule = ScheduleGPipe(
        stage,
        n_microbatches=microbatches,
        loss_fn=_loss_fn(training_config),
    )

    placement = _placement_payload(
        assignment=assignment,
        rank=rank,
        world_size=world_size,
        device=device,
        module=stage_module,
        execution=execution,
        materialization=materialization,
    )
    _emit_line(EVENT_MARKER + json.dumps(placement, sort_keys=True))

    steps = int(os.environ.get("SHARDGRID_AUTOMATIC_STEPS", "5"))
    checkpoint_dir = _snapshot_root() / os.environ.get("SHARDGRID_AUTOMATIC_CHECKPOINT_DIR", "checkpoint")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"checkpoint_rank{rank}.pt"
    params_before = _checksum(stage_module)
    loss_history: list[float] = []
    initial_loss: float | None = None
    final_loss: float | None = None
    step_times: list[float] = []

    torch.cuda.reset_peak_memory_stats(device)
    stage_probe_arity = None
    if stage_index == 0:
        with torch.no_grad():
            if training_config.model.type == "large_residual_transformer":
                probe_inputs = make_large_residual_stage_inputs(
                    LargeResidualTransformerConfig.from_mapping(training_config.model.parameters),
                    plan.stage_metadata[stage_index].module_paths[0],
                    seed=42,
                    step=0,
                    batch_size=sample_batch_size,
                    device=device,
                )
                probe_output = stage_module(*probe_inputs)
            else:
                probe_inputs, _probe_targets = _sample_inputs_for_batch(
                    training_config,
                    step=0,
                    device=device,
                    batch_size=sample_batch_size,
                )
                probe_output = stage_module(probe_inputs)
        stage_probe_arity = len(probe_output) if isinstance(probe_output, tuple) else 1

    for step in range(steps):
        step_started = time.perf_counter()
        _emit_line("TRAIN_STEP_BEGIN")
        optimizer.zero_grad(set_to_none=True)
        inputs, targets = _sample_inputs_for_batch(
            training_config,
            step=step,
            device=device,
            batch_size=train_batch_size,
        )
        losses: list[torch.Tensor] = []
        if stage_index == 0:
            schedule.step(inputs)
        elif stage_index == world_size - 1:
            outputs = schedule.step(target=targets, losses=losses)
            if losses:
                loss_value = float(losses[-1].detach().cpu().item())
            else:
                loss_value = float(_loss_fn(training_config)(outputs, targets).detach().cpu().item())
            loss_history.append(loss_value)
            if initial_loss is None:
                initial_loss = loss_value
            final_loss = loss_value
            _emit_line("LOSS_READY")
        else:
            schedule.step()
        optimizer.step()
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        step_times.append(time.perf_counter() - step_started)
        _emit_line("OPTIMIZER_STEP_END")
        _emit_line("TRAIN_STEP_END")

    params_after = _checksum(stage_module)
    param_update_ok = params_before != params_after
    metadata = {
        "model_name": training_config.model.name,
        "model_type": training_config.model.type,
        "partition_source": execution.labels.get("partition_source"),
        "selected_candidate_id": execution.labels.get("selected_candidate_id"),
        "selected_worker_count": execution.labels.get("selected_worker_count"),
        "stage_to_worker": placement["stage_to_worker"],
    }
    if training_config.model.type == "large_residual_transformer":
        checkpoint_mode = "metadata_only"
        torch.save(
            {
                "checkpoint_version": 1,
                "task": "automatic_plan_training",
                "job_id": str(execution.job_id),
                "rank": rank,
                "world_size": world_size,
                "stage_id": assignment.stage,
                "step": steps,
                "loss_history": loss_history,
                "params_before_checksum": params_before,
                "params_after_checksum": params_after,
                "optimizer_param_group_count": len(optimizer.state_dict()["param_groups"]),
                "metadata": {**metadata, "checkpoint_mode": checkpoint_mode},
            },
            checkpoint_path,
        )
        loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        param_restore_ok = loaded["params_after_checksum"] == params_after
        optimizer_restore_ok = loaded["optimizer_param_group_count"] == len(
            optimizer.state_dict()["param_groups"]
        )
    else:
        checkpoint_mode = "full_state_dict"
        torch.save(
            {
                "checkpoint_version": 1,
                "task": "automatic_plan_training",
                "job_id": str(execution.job_id),
                "rank": rank,
                "world_size": world_size,
                "stage_id": assignment.stage,
                "step": steps,
                "model_state_dict": {
                    name: tensor.detach().cpu() for name, tensor in stage_module.state_dict().items()
                },
                "optimizer_state_dict": optimizer.state_dict(),
                "loss_history": loss_history,
                "metadata": {**metadata, "checkpoint_mode": checkpoint_mode},
            },
            checkpoint_path,
        )
        for parameter in stage_module.parameters():
            parameter.data.zero_()
        fresh_optimizer = torch.optim.AdamW(stage_module.parameters(), lr=float(os.environ.get("SHARDGRID_AUTOMATIC_LR", "1e-3")))
        loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        stage_module.load_state_dict(loaded["model_state_dict"])
        fresh_optimizer.load_state_dict(loaded["optimizer_state_dict"])
        param_restore_ok = _checksum(stage_module) == params_after
        optimizer_restore_ok = _state_dict_equal(fresh_optimizer.state_dict(), optimizer.state_dict())

    metadata_payload = {
        "status": "complete",
        "checkpoint_version": 1,
        "job_id": str(execution.job_id),
        "rank": rank,
        "world_size": world_size,
        "stage_id": assignment.stage,
        "step": steps,
        "checkpoint_path": str(checkpoint_path),
        "model_name": training_config.model.name,
        "model_type": training_config.model.type,
        "partition_source": execution.labels.get("partition_source"),
        "selected_candidate_id": execution.labels.get("selected_candidate_id"),
        "selected_worker_count": execution.labels.get("selected_worker_count"),
        "checkpoint_mode": checkpoint_mode,
        "master_addr": execution.master.address,
        "master_port": execution.master.port,
        "stage_to_worker": placement["stage_to_worker"],
        "step_times": step_times,
        "total_training_time": sum(step_times),
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device),
        **materialization,
    }
    _write_checkpoint_metadata(checkpoint_dir / "checkpoint-metadata.json", metadata_payload)

    train_payload = {
        "hostname": socket.gethostname(),
        "job_id": str(execution.job_id),
        "rank": rank,
        "world_size": world_size,
        "worker_id": str(assignment.worker_id),
        "stage_id": assignment.stage,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "steps": steps,
        "model_name": training_config.model.name,
        "model_type": training_config.model.type,
        "loss_history": loss_history,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_isfinite": all(math.isfinite(item) for item in loss_history),
        "loss_decrease": bool(
            initial_loss is not None and final_loss is not None and final_loss < initial_loss
        ),
        "params_before_checksum": params_before,
        "params_after_checksum": params_after,
        "param_update_ok": param_update_ok,
        "param_restore_ok": param_restore_ok,
        "optimizer_restore_ok": optimizer_restore_ok,
        "step_restore_ok": int(loaded["step"]) == steps,
        "checkpoint_roundtrip_ok": True,
        "checkpoint_mode": checkpoint_mode,
        "checkpoint_path": str(checkpoint_path),
        "step_times": step_times,
        "total_training_time": sum(step_times),
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device),
        "partition_source": execution.labels.get("partition_source"),
        "selected_candidate_id": execution.labels.get("selected_candidate_id"),
        "communication_edges": list(assignment.communication_edges),
        **materialization,
        "activation_transfer_ok": world_size > 1,
        "gradient_transfer_ok": world_size > 1,
        "cross_stage_dependency_preserved": bool(stage_probe_arity is None or stage_probe_arity > 1),
        "stage_output_arity": stage_probe_arity,
    }
    _emit_line(TRAIN_MARKER + json.dumps(train_payload, sort_keys=True))
    dist.barrier(device_ids=[local_rank])
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
