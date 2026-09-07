"""Generic DAG example/regression multi-host training runner."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
import sys
from collections import deque
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.distributed as dist

REPO_ROOT = Path(__file__).resolve().parents[2]
for import_root in (REPO_ROOT, REPO_ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from examples.models.generic_partition_zoo import build_zoo_model, make_zoo_sample
from shardgrid.common.config import TrainingConfig
from shardgrid.distributed.backend import select_backend
from shardgrid.engines.models import ParallelPlan
from shardgrid.planner.generic_graph import FXGraphCaptureAdapter
from shardgrid.planner.models import ExecutionPlan
from shardgrid.planner.planning_contract import (
    LogicalPartitionPlan,
    LogicalPartitionSpec,
    PlacementPlan,
    PlacementSpec,
)
from shardgrid.runtime.checkpoint import save_worker_state_shard
from shardgrid.runtime.dag import compile_runtime_plan
from shardgrid.runtime.partition_graph import extract_partition_graph
from shardgrid.runtime.transport import PendingTensorSend, recv_tensor, send_tensor_async

EVENT_MARKER = "GENERIC_DAG_RUNTIME_EVIDENCE "
TRAIN_MARKER = "T074_TRAIN_EVIDENCE "
PROBE_MARKER = "GENERIC_DAG_PROBE_EVIDENCE "


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--memory-probe", action="store_true")
    args = parser.parse_args()

    rank = int(os.environ.get("RANK", args.rank))
    config, execution, parallel_plan = _load_artifacts()
    if config.model.type != "generic_dag":
        raise ValueError("train_generic_dag.py only supports model.type=generic_dag")

    device = _device()
    torch.manual_seed(42 + rank)
    probe_logger = _ProbePhaseLogger(rank, execution) if args.memory_probe else None
    if probe_logger is not None:
        probe_logger.record("DISTRIBUTED_INIT_BEGIN")
    dist.init_process_group(
        backend=_backend(device),
        init_method=f"tcp://{os.environ['MASTER_ADDR']}:{int(os.environ['MASTER_PORT'])}",
        rank=rank,
        world_size=execution.world_size,
    )
    if probe_logger is not None:
        probe_logger.set_rendezvous(
            master_addr=str(os.environ.get("MASTER_ADDR", "")),
            master_port=int(os.environ.get("MASTER_PORT", "0")),
        )
        probe_logger.record("DISTRIBUTED_INIT_END")

    try:
        result = _run_training(
            config,
            execution,
            parallel_plan,
            rank=rank,
            device=device,
            memory_probe=args.memory_probe,
            probe_logger=probe_logger,
        )
    except Exception as exc:
        if probe_logger is not None:
            probe_logger.set_error(exc)
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

    print(EVENT_MARKER + json.dumps(result["runtime"], sort_keys=True), flush=True)
    print(TRAIN_MARKER + json.dumps(result["train"], sort_keys=True), flush=True)


def _run_training(
    config: TrainingConfig,
    execution: ExecutionPlan,
    parallel_plan: ParallelPlan,
    *,
    rank: int,
    device: torch.device,
    memory_probe: bool = False,
    probe_logger: "_ProbePhaseLogger | None" = None,
) -> dict[str, Any]:
    zoo_model = str(config.model.parameters.get("zoo_model", "mini_unet"))
    model_kwargs = _zoo_model_kwargs(config.model.parameters)
    model, capture, args = _build_meta_runtime_model(zoo_model, model_kwargs)
    logical, placement = _selected_logical_and_placement(
        capture.canonical_graph,
        parallel_plan,
        execution,
    )
    runtime_plan = compile_runtime_plan(capture.canonical_graph, logical, placement)
    ownership = next(
        worker
        for worker in runtime_plan.ownership.workers
        if worker.worker_id == str(execution.workers[rank].worker_id)
    )
    _materialize_owned_modules(
        model,
        capture.canonical_graph,
        ownership.owned_partitions,
        logical,
        device,
    )
    consistency = _plan_runtime_consistency(
        model,
        capture.canonical_graph,
        logical,
        placement,
        execution,
        parallel_plan,
        ownership,
        rank=rank,
    )
    extracted = {
        partition.partition_id: extract_partition_graph(
            capture.canonical_graph,
            capture.backend_graph,
            partition,
        )
        for partition in logical.partitions
        if partition.partition_id in ownership.owned_partitions
    }
    owned_trainable = _owned_trainable_parameters(model, capture.canonical_graph, ownership)
    optim = torch.optim.AdamW([parameter for _name, parameter in owned_trainable], lr=_lr(config))
    initial_digests = _parameter_digests(owned_trainable)
    runtime_evidence = {
        **consistency,
        "generic_dag_runtime_used": True,
        "legacy_stage_runtime_used": False,
        "worker_id": str(execution.workers[rank].worker_id),
        "rank": rank,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "owned_partitions": list(ownership.owned_partitions),
        "full_model_real_materialized": False,
        "owned_parameter_bytes": _owned_parameter_bytes(model, capture.canonical_graph, ownership),
        "estimated_full_model_parameter_bytes": sum(
            node.parameter_bytes for node in capture.canonical_graph.nodes
        ),
        "local_edge_count": sum(
            1 for edge in runtime_plan.edges if edge.edge_kind.value == "local"
        ),
        "remote_edge_count": sum(
            1 for edge in runtime_plan.edges if edge.edge_kind.value == "remote"
        ),
    }
    print(EVENT_MARKER + json.dumps(runtime_evidence, sort_keys=True), flush=True)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    steps = 1 if memory_probe else int(config.model.parameters.get("training_steps", 20))
    final_loss: float | None = None
    loss_history: list[float] = []
    forward_remote = 0
    backward_remote = 0
    live_change_evidence: dict[str, Any] | None = None
    progress = _ProgressLog(rank, str(execution.workers[rank].worker_id))
    for step in range(steps):
        progress.record("STEP_BEGIN", step=step)
        if probe_logger is not None:
            probe_logger.record("PROBE_FORWARD_BEGIN", step=step)
        optim.zero_grad(set_to_none=True)
        values, received, produced = _forward_step(
            config,
            capture,
            logical,
            placement,
            extracted,
            execution=execution,
            rank=rank,
            step=step,
            device=device,
            sample_args=args,
            progress=progress,
        )
        if probe_logger is not None:
            probe_logger.record("PROBE_FORWARD_END", step=step)
        loss = _loss_for_rank(capture.canonical_graph.output_value_ids, values)
        if loss is not None:
            final_loss = float(loss.detach().cpu().item())
            loss_history.append(final_loss)
        if probe_logger is not None:
            probe_logger.record("PROBE_BACKWARD_BEGIN", step=step)
        backward_count = _backward_step(
            capture,
            logical,
            placement,
            execution=execution,
            rank=rank,
            step=step,
            values=values,
            received_boundaries=received,
            produced_values=produced,
            final_loss=loss,
            progress=progress,
        )
        if probe_logger is not None:
            probe_logger.record("PROBE_BACKWARD_END", step=step)
        progress.record("OPTIMIZER_BEGIN", step=step)
        if probe_logger is not None:
            probe_logger.record("PROBE_OPTIMIZER_BEGIN", step=step)
        optim.step()
        if probe_logger is not None:
            probe_logger.record("PROBE_OPTIMIZER_END", step=step)
        progress.record("OPTIMIZER_END", step=step)
        progress.record("STEP_END", step=step)
        forward_remote += _remote_forward_count(
            capture.canonical_graph,
            logical,
            placement,
            execution,
            rank,
        )
        backward_remote += backward_count
        if live_change_evidence is None:
            live_change_evidence = _parameter_change_evidence(
                initial_digests,
                _parameter_digests(owned_trainable),
            )
        print(
            TRAIN_MARKER
            + json.dumps(
                {
                    **runtime_evidence,
                    **live_change_evidence,
                    "steps": step + 1,
                    "completed_steps": step + 1,
                    "completed_forward_steps": step + 1,
                    "completed_backward_steps": step + 1,
                    "optimizer_steps": step + 1,
                    "forward_completed": True,
                    "backward_completed": True,
                    "optimizer_step_completed": True,
                    "distributed_initialized": True,
                    "activation_transfer_ok": forward_remote > 0,
                    "gradient_transfer_ok": backward_remote > 0,
                    "activation_remote_edges": forward_remote,
                    "gradient_remote_edges": backward_remote,
                    "final_loss": final_loss,
                    "loss_history": loss_history,
                    "loss_isfinite": final_loss is None or math.isfinite(final_loss),
                    "checkpoint_roundtrip_ok": False,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    dist.barrier()
    change_evidence = _parameter_change_evidence(
        initial_digests,
        _parameter_digests(owned_trainable),
    )
    peak = _memory_peak(device)
    if memory_probe:
        train = {
            **runtime_evidence,
            **change_evidence,
            **peak,
            "steps": steps,
            "completed_steps": steps,
            "completed_forward_steps": steps,
            "completed_backward_steps": steps,
            "optimizer_steps": steps,
            "forward_completed": True,
            "backward_completed": True,
            "optimizer_step_completed": True,
            "distributed_initialized": True,
            "memory_probe_only": True,
            "activation_transfer_ok": forward_remote > 0,
            "gradient_transfer_ok": backward_remote > 0,
            "activation_remote_edges": forward_remote,
            "gradient_remote_edges": backward_remote,
            "final_loss": final_loss,
            "loss_history": loss_history,
            "loss_isfinite": final_loss is None or math.isfinite(final_loss),
            "checkpoint_roundtrip_ok": False,
            "checkpoint_ref": None,
        }
        _write_memory_probe(rank, train)
        if probe_logger is not None:
            probe_logger.complete(train)
        return {"runtime": runtime_evidence, "train": train}

    checkpoint_metadata = {
        **runtime_evidence,
        **change_evidence,
        **peak,
        "rank": rank,
        "world_size": execution.world_size,
        "stage_id": execution.workers[rank].stage,
        "step": steps,
        "checkpoint_version": 1,
    }
    checkpoint_result = save_worker_state_shard(
        _snapshot_root() / "checkpoint" / "model.pt",
        graph=capture.canonical_graph,
        runtime_plan=runtime_plan,
        worker_id=str(execution.workers[rank].worker_id),
        gpu_index=execution.workers[rank].gpu_index,
        state_dict=model.state_dict(),
        job_id=str(execution.job_id),
        plan_id=execution.labels.get("selected_candidate_id", "generic-dag-plan"),
        training_step=steps,
        metadata=checkpoint_metadata,
    )
    _write_checkpoint_metadata(
        config,
        execution,
        rank=rank,
        steps=steps,
        final_loss=final_loss,
        runtime=runtime_evidence,
        checkpoint_path=checkpoint_result.path,
        change_evidence=change_evidence,
    )
    train = {
        **runtime_evidence,
        **change_evidence,
        **peak,
        "steps": steps,
        "completed_steps": steps,
        "completed_forward_steps": steps,
        "completed_backward_steps": steps,
        "optimizer_steps": steps,
        "forward_completed": True,
        "backward_completed": True,
        "optimizer_step_completed": True,
        "distributed_initialized": True,
        "activation_transfer_ok": forward_remote > 0,
        "gradient_transfer_ok": backward_remote > 0,
        "activation_remote_edges": forward_remote,
        "gradient_remote_edges": backward_remote,
        "final_loss": final_loss,
        "loss_history": loss_history,
        "loss_isfinite": final_loss is None or math.isfinite(final_loss),
        "checkpoint_roundtrip_ok": True,
        "checkpoint_ref": "checkpoint/model.pt",
        "checkpoint_path": checkpoint_result.path,
    }
    return {"runtime": runtime_evidence, "train": train}


def _forward_step(
    config: TrainingConfig,
    capture: Any,
    logical: Any,
    placement: PlacementPlan,
    extracted: Mapping[str, Any],
    *,
    execution: ExecutionPlan,
    rank: int,
    step: int,
    device: torch.device,
    sample_args: tuple[Any, ...],
    progress: "_ProgressLog",
) -> tuple[dict[str, Any], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    del config
    values = {
        value_id: _move_tensor(value, device)
        for value_id, value in zip(
            capture.canonical_graph.input_value_ids,
            sample_args,
            strict=True,
        )
    }
    received_boundaries: dict[str, torch.Tensor] = {}
    produced_values: dict[str, torch.Tensor] = {}
    value_specs = {value.value_id: value for value in capture.canonical_graph.values}
    owner_rank = _owner_rank_by_partition(placement, execution)
    partition_by_node = _partition_by_node(logical)
    producer_partition = _producer_partition_by_value(capture.canonical_graph, partition_by_node)
    remote_consumers = _remote_consumer_ranks(
        capture.canonical_graph,
        logical,
        placement,
        execution,
    )
    pending_sends: list[tuple[PendingTensorSend, int, str]] = []

    for partition in logical.partitions:
        if owner_rank[partition.partition_id] != rank:
            continue
        for value_id in partition.input_value_ids:
            if value_id in values:
                continue
            spec = value_specs[value_id]
            source_rank = owner_rank[producer_partition[value_id]]
            progress.record(
                "FORWARD_RECV_BEGIN",
                step=step,
                partition=partition.partition_id,
                value_id=value_id,
                peer_rank=source_rank,
            )
            tensor, _evidence = recv_tensor(
                shape=tuple(int(item) for item in spec.shape),
                dtype=_dtype(spec.dtype),
                src=source_rank,
                device=device,
                step=step,
                value_id=value_id,
                direction="FORWARD",
            )
            if torch.is_floating_point(tensor):
                tensor = tensor.detach().requires_grad_(True)
            values[value_id] = tensor
            received_boundaries[value_id] = tensor
            progress.record(
                "FORWARD_RECV_END",
                step=step,
                partition=partition.partition_id,
                value_id=value_id,
                peer_rank=source_rank,
            )
        progress.record("PARTITION_FORWARD_BEGIN", step=step, partition=partition.partition_id)
        outputs = extracted[partition.partition_id](values)
        progress.record("PARTITION_FORWARD_END", step=step, partition=partition.partition_id)
        for value_id, tensor in outputs.items():
            if isinstance(tensor, torch.Tensor) and torch.is_floating_point(tensor):
                tensor.retain_grad()
            values[value_id] = tensor
            produced_values[value_id] = tensor
            for dst in remote_consumers.get((partition.partition_id, value_id), ()):
                progress.record(
                    "FORWARD_SEND_BEGIN",
                    step=step,
                    partition=partition.partition_id,
                    value_id=value_id,
                    peer_rank=dst,
                )
                pending_sends.append(
                    (
                        send_tensor_async(
                            tensor.detach(),
                            dst=dst,
                            step=step,
                            value_id=value_id,
                            direction="FORWARD",
                        ),
                        dst,
                        value_id,
                    )
                )
    for pending, dst, value_id in pending_sends:
        pending.wait()
        progress.record("FORWARD_SEND_END", step=step, value_id=value_id, peer_rank=dst)
    return values, received_boundaries, produced_values


def _backward_step(
    capture: Any,
    logical: Any,
    placement: PlacementPlan,
    *,
    execution: ExecutionPlan,
    rank: int,
    step: int,
    values: Mapping[str, Any],
    received_boundaries: Mapping[str, torch.Tensor],
    produced_values: Mapping[str, torch.Tensor],
    final_loss: torch.Tensor | None,
    progress: "_ProgressLog",
) -> int:
    owner_rank = _owner_rank_by_partition(placement, execution)
    partition_by_node = _partition_by_node(logical)
    producer_partition = _producer_partition_by_value(capture.canonical_graph, partition_by_node)
    remote_consumers = _remote_consumer_ranks(
        capture.canonical_graph,
        logical,
        placement,
        execution,
    )
    remote_inputs = _remote_inputs_by_partition(
        logical,
        placement,
        producer_partition,
        execution,
    )
    sent_gradients = 0
    loss_used = False
    pending_sends: list[tuple[PendingTensorSend, int, str]] = []

    for partition in reversed(logical.partitions):
        if owner_rank[partition.partition_id] != rank:
            continue
        progress.record("PARTITION_BACKWARD_BEGIN", step=step, partition=partition.partition_id)
        if final_loss is not None and not loss_used:
            if any(
                value_id in capture.canonical_graph.output_value_ids
                for value_id in partition.output_value_ids
            ):
                final_loss.backward(retain_graph=True)
                loss_used = True
        for value_id in partition.output_value_ids:
            grads = []
            for src in remote_consumers.get((partition.partition_id, value_id), ()):
                progress.record(
                    "BACKWARD_RECV_BEGIN",
                    step=step,
                    partition=partition.partition_id,
                    value_id=value_id,
                    peer_rank=src,
                )
                grad, _evidence = recv_tensor(
                    shape=tuple(values[value_id].shape),
                    dtype=values[value_id].dtype,
                    src=src,
                    device=values[value_id].device,
                    step=step,
                    value_id=value_id,
                    direction="BACKWARD",
                )
                grads.append(grad)
                progress.record(
                    "BACKWARD_RECV_END",
                    step=step,
                    partition=partition.partition_id,
                    value_id=value_id,
                    peer_rank=src,
                )
            if grads:
                progress.record(
                    "GRAD_AGGREGATION_BEGIN",
                    step=step,
                    partition=partition.partition_id,
                    value_id=value_id,
                    gradient_count=len(grads),
                )
                total = sum(grads)
                produced_values[value_id].backward(total, retain_graph=True)
                progress.record(
                    "GRAD_AGGREGATION_END",
                    step=step,
                    partition=partition.partition_id,
                    value_id=value_id,
                    gradient_count=len(grads),
                )
        progress.record("PARTITION_BACKWARD_END", step=step, partition=partition.partition_id)
        for value_id, dst in remote_inputs.get(partition.partition_id, ()):
            boundary = received_boundaries.get(value_id)
            if boundary is None or boundary.grad is None:
                continue
            progress.record(
                "GRAD_SEND_BEGIN",
                step=step,
                partition=partition.partition_id,
                value_id=value_id,
                peer_rank=dst,
            )
            pending_sends.append(
                (
                    send_tensor_async(
                        boundary.grad.detach(),
                        dst=dst,
                        step=step,
                        value_id=value_id,
                        direction="BACKWARD",
                    ),
                    dst,
                    value_id,
                )
            )
            sent_gradients += 1
    for pending, dst, value_id in pending_sends:
        pending.wait()
        progress.record("GRAD_SEND_END", step=step, value_id=value_id, peer_rank=dst)
    return sent_gradients


def _load_artifacts() -> tuple[TrainingConfig, ExecutionPlan, ParallelPlan]:
    root = _snapshot_root()
    config = TrainingConfig.from_dict(_load_json(root / "config" / "training-config.json"))
    execution = ExecutionPlan.from_dict(_load_json(root / "plan" / "execution-plan.json"))
    plan_path = root / "plan" / "original-parallel-plan.json"
    if not plan_path.exists():
        raise ValueError("EXECUTION_PLAN_MISSING: plan/original-parallel-plan.json is required")
    parallel_plan = ParallelPlan.from_dict(_load_json(plan_path))
    return config, execution, parallel_plan


def _selected_logical_and_placement(
    graph: Any,
    parallel_plan: ParallelPlan,
    execution: ExecutionPlan,
) -> tuple[LogicalPartitionPlan, PlacementPlan]:
    if not parallel_plan.stage_metadata:
        raise ValueError("EXECUTION_PLAN_MISSING: original plan has no stage_metadata")
    if execution.world_size != len(execution.workers):
        raise ValueError("EXECUTION_PLAN_MISSING: execution workers do not match world_size")

    executable = tuple(
        node for node in graph.nodes if node.op_kind not in {"placeholder", "output"}
    )
    node_index = {node.node_id: index for index, node in enumerate(executable)}
    module_index = {
        str(node.module_path): index
        for index, node in enumerate(executable)
        if node.module_path
    }
    stage_starts: list[int] = []
    for stage in parallel_plan.stage_metadata:
        if stage.placement is None:
            raise ValueError(f"EXECUTION_PLAN_MISSING: {stage.stage_id} has no placement")
        indices = [
            module_index[path]
            for path in stage.module_paths
            if path in module_index
        ]
        if not indices:
            indices = [
                node_index[module_id]
                for module_id in stage.module_ids
                if module_id in node_index
            ]
        if not indices:
            raise ValueError(
                f"EXECUTION_PLAN_MISSING: {stage.stage_id} cannot be mapped to graph nodes"
            )
        stage_starts.append(min(indices))

    if stage_starts != sorted(stage_starts):
        raise ValueError("PLAN_RUNTIME_MISMATCH: stage module order differs from graph order")

    partition_nodes: dict[str, tuple[Any, ...]] = {}
    used_node_ids: set[str] = set()
    for index, stage in enumerate(parallel_plan.stage_metadata):
        start = stage_starts[index]
        stop = stage_starts[index + 1] if index + 1 < len(stage_starts) else len(executable)
        chunk = tuple(executable[start:stop])
        duplicate = used_node_ids & {node.node_id for node in chunk}
        if duplicate:
            raise ValueError(
                "PLAN_RUNTIME_MISMATCH: stage node ownership overlaps "
                + ",".join(sorted(duplicate))
            )
        used_node_ids.update(node.node_id for node in chunk)
        partition_nodes[stage.stage_id] = chunk

    consumers: dict[str, set[str]] = {}
    for edge in graph.edges:
        consumers.setdefault(edge.value_id, set()).add(edge.target_node_id)
    value_producer = {
        value.value_id: value.producer_node_id
        for value in graph.values
        if value.producer_node_id is not None
    }
    partitions = []
    for stage in parallel_plan.stage_metadata:
        chunk = partition_nodes[stage.stage_id]
        node_ids = tuple(node.node_id for node in chunk)
        node_set = set(node_ids)
        input_values = sorted(
            {
                value_id
                for node in chunk
                for value_id in node.input_value_ids
                if value_producer.get(value_id) not in node_set
            }
        )
        output_values = sorted(
            {
                value_id
                for node in chunk
                for value_id in node.output_value_ids
                if value_id in graph.output_value_ids
                or consumers.get(value_id, set()) - node_set
            }
        )
        boundary_edges = sorted(
            edge.edge_id or f"{edge.source_node_id}->{edge.target_node_id}:{edge.value_id}"
            for edge in graph.edges
            if edge.source_node_id in node_set and edge.target_node_id not in node_set
        )
        partitions.append(
            LogicalPartitionSpec(
                partition_id=stage.stage_id,
                node_ids=node_ids,
                input_value_ids=tuple(input_values),
                output_value_ids=tuple(output_values),
                parameter_ids=tuple(
                    sorted({pid for node in chunk for pid in node.parameter_ids})
                ),
                buffer_ids=tuple(sorted({bid for node in chunk for bid in node.buffer_ids})),
                estimated_compute=sum(node.estimated_compute_cost for node in chunk),
                estimated_memory=sum(_node_memory(node) for node in chunk),
                boundary_edges=tuple(boundary_edges),
            )
        )

    logical = LogicalPartitionPlan(graph.graph_fingerprint, tuple(partitions))
    placement = PlacementPlan(
        graph_fingerprint=graph.graph_fingerprint,
        selected_gpu_count=len(
            {
                (stage.placement.worker_id, stage.placement.gpu_index)
                for stage in parallel_plan.stage_metadata
                if stage.placement is not None
            }
        ),
        placements=tuple(
            PlacementSpec(
                stage.stage_id,
                f"{stage.placement.worker_id}:gpu{stage.placement.gpu_index}",
                stage.placement.worker_id,
                stage.placement.gpu_index,
            )
            for stage in parallel_plan.stage_metadata
            if stage.placement is not None
        ),
    )
    return logical, placement


def _node_memory(node: Any) -> int:
    return int(
        node.estimated_peak_memory_contribution
        or node.parameter_bytes
        + node.activation_bytes
        + node.gradient_bytes
        + node.optimizer_bytes
        + node.temporary_bytes
    )


def _plan_runtime_consistency(
    model: Any,
    graph: Any,
    logical: LogicalPartitionPlan,
    placement: PlacementPlan,
    execution: ExecutionPlan,
    parallel_plan: ParallelPlan,
    ownership: Any,
    *,
    rank: int,
) -> dict[str, Any]:
    planned = tuple(
        placed.partition_id
        for placed in placement.placements
        if str(placed.worker_id) == str(execution.workers[rank].worker_id)
        and int(placed.gpu_index) == int(execution.workers[rank].gpu_index)
    )
    runtime = tuple(ownership.owned_partitions)
    if planned != runtime:
        raise ValueError(
            f"PLAN_RUNTIME_MISMATCH: planned partitions {planned} != runtime {runtime}"
        )
    by_partition = {partition.partition_id: partition for partition in logical.partitions}
    planned_params = tuple(
        sorted(
            {
                pid
                for partition_id in planned
                for pid in by_partition[partition_id].parameter_ids
            }
        )
    )
    runtime_params = tuple(sorted(ownership.local_parameter_ids))
    materialized_params = _materialized_parameter_ids(model, graph)
    if planned_params != runtime_params or planned_params != materialized_params:
        raise ValueError(
            "PLAN_RUNTIME_MISMATCH: parameter ids "
            f"planned={planned_params} runtime={runtime_params} materialized={materialized_params}"
        )
    planned_buffers = tuple(
        sorted(
            {
                bid
                for partition_id in planned
                for bid in by_partition[partition_id].buffer_ids
            }
        )
    )
    materialized_buffers = _materialized_buffer_ids(model, graph)
    if planned_buffers != materialized_buffers:
        raise ValueError(
            "PLAN_RUNTIME_MISMATCH: buffer ids "
            f"planned={planned_buffers} materialized={materialized_buffers}"
        )
    return {
        "plan_runtime_consistency_check": "PASS",
        "plan_runtime_consistency": True,
        "selected_candidate_id": parallel_plan.selected_candidate_id
        or execution.labels.get("selected_candidate_id"),
        "execution_plan_fingerprint": _fingerprint(execution.to_dict()),
        "logical_partition_plan_fingerprint": _fingerprint(logical.to_dict()),
        "placement_plan_fingerprint": _fingerprint(placement.to_dict()),
        "planned_owned_partitions": list(planned),
        "runtime_owned_partitions": list(runtime),
        "planned_parameter_ids": list(planned_params),
        "runtime_parameter_ids": list(runtime_params),
        "runtime_materialized_parameter_ids": list(materialized_params),
        "planned_buffer_ids": list(planned_buffers),
        "runtime_materialized_buffer_ids": list(materialized_buffers),
    }


def _materialized_parameter_ids(model: Any, graph: Any) -> tuple[str, ...]:
    by_name = dict(model.named_parameters(remove_duplicate=False))
    result = []
    for use in graph.parameter_uses:
        parameter = by_name.get(use.canonical_path)
        if parameter is not None and parameter.device.type != "meta":
            result.append(use.parameter_id)
    return tuple(sorted(set(result)))


def _materialized_buffer_ids(model: Any, graph: Any) -> tuple[str, ...]:
    by_name = dict(model.named_buffers(remove_duplicate=False))
    ids: dict[str, str] = {}
    for node in graph.nodes:
        ids.update(dict(zip(node.buffer_ids, node.buffer_paths, strict=False)))
    return tuple(
        sorted(
            buffer_id
            for buffer_id, path in ids.items()
            if path in by_name and by_name[path].device.type != "meta"
        )
    )


def _fingerprint(payload: Mapping[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


class _ProbePhaseLogger:
    """Bounded phase diagnostics writer for --memory-probe runs.

    Writes ``diagnostics/memory-probe-rank<rank>.json`` incrementally so the
    control node can see exactly which probe phase a rank reached and whether
    the rank completed distributed rendezvous. Also prints a parseable marker
    line per phase for launcher monitor capture.
    """

    def __init__(self, rank: int, execution: ExecutionPlan) -> None:
        self.rank = rank
        self.execution = execution
        self.events: deque[dict[str, Any]] = deque(maxlen=100)
        self.path = _snapshot_root() / "diagnostics" / f"memory-probe-rank{rank}.json"
        self.payload: dict[str, Any] = {
            "memory_probe_only": True,
            "rank": rank,
            "world_size": execution.world_size,
            "worker_id": str(execution.workers[rank].worker_id),
        }

    def record(self, event: str, **details: Any) -> None:
        entry = {"event": event, "rank": self.rank, **details}
        self.events.append(entry)
        self.payload["latest_phase"] = event
        self.payload["recent"] = list(self.events)
        self._write()
        self._print(event)

    def set_rendezvous(self, *, master_addr: str, master_port: int) -> None:
        self.payload.update(
            {
                "rendezvous_ready": True,
                "distributed_initialized": True,
                "master_addr": master_addr,
                "master_port": master_port,
                "hostname": socket.gethostname(),
            }
        )
        self._write()

    def set_peaks(self, peak: Mapping[str, Any]) -> None:
        self.payload.update(dict(peak))
        self._write()

    def complete(self, train: Mapping[str, Any]) -> None:
        self.payload.update(dict(train))
        self.payload["latest_phase"] = "PROBE_COMPLETE"
        self.payload["recent"] = list(self.events)
        self._write()
        self._print("PROBE_COMPLETE")

    def set_error(self, exc: BaseException) -> None:
        import traceback

        text = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )
        self.payload["error"] = f"{type(exc).__name__}: {exc}"
        self.payload["traceback"] = text[-4000:]
        self.payload["cuda_oom"] = _probe_error_is_oom(exc)
        self.payload["latest_phase"] = "PROBE_FAILED"
        self._write()
        self._print("PROBE_FAILED")

    def _print(self, phase: str) -> None:
        print(
            PROBE_MARKER
            + json.dumps(
                {"phase": phase, "rank": self.rank},
                sort_keys=True,
            ),
            flush=True,
        )

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )


class _ProgressLog:
    def __init__(self, rank: int, worker_id: str) -> None:
        self.rank = rank
        self.worker_id = worker_id
        self.events: deque[dict[str, Any]] = deque(maxlen=100)
        self.path = _snapshot_root() / "diagnostics" / f"generic-dag-progress-rank{rank}.json"

    def record(self, event: str, **details: Any) -> None:
        payload = {
            "event": event,
            "rank": self.rank,
            "worker_id": self.worker_id,
            **details,
        }
        self.events.append(payload)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {
                    "latest": payload,
                    "recent": list(self.events),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )


def _build_meta_runtime_model(
    zoo_model: str,
    model_kwargs: Mapping[str, Any],
) -> tuple[Any, Any, tuple[Any, ...]]:
    sample_args, sample_kwargs = make_zoo_sample(zoo_model, **model_kwargs)
    if sample_kwargs:
        raise ValueError("generic DAG runner currently supports positional sample inputs only")
    with torch.device("meta"):
        model = build_zoo_model(zoo_model, **model_kwargs)
    meta_args = tuple(_to_meta(item) for item in sample_args)
    capture = FXGraphCaptureAdapter().capture(model, sample_args=meta_args)
    return model, capture, tuple(sample_args)


def _zoo_model_kwargs(parameters: Mapping[str, Any]) -> dict[str, Any]:
    reserved = {
        "zoo_model",
        "training_steps",
        "learning_rate",
        "logical_partitions",
        "generic_dag_runtime",
    }
    return {str(key): value for key, value in parameters.items() if str(key) not in reserved}


def _materialize_owned_modules(
    model: Any,
    graph: Any,
    owned_partitions: tuple[str, ...],
    logical: Any,
    device: torch.device,
) -> None:
    owned_nodes = {
        node_id
        for partition in logical.partitions
        if partition.partition_id in owned_partitions
        for node_id in partition.node_ids
    }
    module_paths = {
        node.module_path
        for node in graph.nodes
        if node.node_id in owned_nodes and node.module_path
    }
    modules = dict(model.named_modules())
    for path in sorted(module_paths):
        module = modules[path]
        module.to_empty(device=device)
        reset = getattr(module, "reset_parameters", None)
        if callable(reset):
            torch.manual_seed(42 + _stable_index(path))
            reset()


def _write_checkpoint_metadata(
    config: TrainingConfig,
    execution: ExecutionPlan,
    *,
    rank: int,
    steps: int,
    final_loss: float | None,
    runtime: Mapping[str, Any],
    checkpoint_path: str,
    change_evidence: Mapping[str, Any],
) -> None:
    payload = {
        "status": "complete",
        "checkpoint_version": 1,
        "job_id": str(execution.job_id),
        "rank": rank,
        "world_size": execution.world_size,
        "stage_id": execution.workers[rank].stage,
        "step": steps,
        "checkpoint_path": checkpoint_path,
        "checkpoint_ref": "checkpoint/model.pt",
        "model_name": config.model.name,
        "model_type": config.model.type,
        "final_loss": final_loss,
        **dict(change_evidence),
        **dict(runtime),
    }
    path = _snapshot_root() / "checkpoint" / "checkpoint-metadata.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _memory_peak(device: torch.device) -> dict[str, int]:
    if device.type != "cuda":
        return {
            "actual_peak_allocated_bytes": 0,
            "actual_peak_reserved_bytes": 0,
        }
    return {
        "actual_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "actual_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def _probe_error_is_oom(exc: BaseException) -> bool:
    lowered = f"{type(exc).__name__}: {exc}".lower()
    return any(
        token in lowered
        for token in (
            "cuda out of memory",
            "cuda error: out of memory",
            "cuda oom",
            "out of memory",
        )
    )


def _write_memory_probe(rank: int, payload: Mapping[str, Any]) -> None:
    path = _snapshot_root() / "diagnostics" / f"memory-probe-rank{rank}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")


def _owned_parameters(model: Any, graph: Any, ownership: Any) -> list[torch.nn.Parameter]:
    by_name = dict(model.named_parameters(remove_duplicate=False))
    keys = {
        item.canonical_path
        for item in graph.parameter_uses
        if item.parameter_id in ownership.local_parameter_ids
    }
    return [by_name[key] for key in sorted(keys) if key in by_name]


def _owned_trainable_parameters(
    model: Any,
    graph: Any,
    ownership: Any,
) -> list[tuple[str, torch.nn.Parameter]]:
    by_name = dict(model.named_parameters(remove_duplicate=False))
    keys = {
        item.canonical_path
        for item in graph.parameter_uses
        if item.parameter_id in ownership.local_parameter_ids
    }
    return [
        (key, by_name[key])
        for key in sorted(keys)
        if key in by_name and by_name[key].requires_grad
    ]


def _owned_parameter_bytes(model: Any, graph: Any, ownership: Any) -> int:
    return sum(
        parameter.numel() * parameter.element_size()
        for parameter in _owned_parameters(model, graph, ownership)
    )


def _parameter_digests(
    parameters: list[tuple[str, torch.nn.Parameter]],
) -> dict[str, str]:
    return {
        name: hashlib.sha256(parameter.detach().cpu().numpy().tobytes()).hexdigest()
        for name, parameter in parameters
    }


def _parameter_change_evidence(
    before: Mapping[str, str],
    after: Mapping[str, str],
) -> dict[str, Any]:
    changed = [name for name, digest in before.items() if after.get(name) != digest]
    return {
        "parameter_changed": bool(changed),
        "changed_parameter_count": len(changed),
        "checked_parameter_count": len(before),
        "changed_parameter_names": changed,
    }


def _loss_for_rank(
    output_value_ids: tuple[str, ...],
    values: Mapping[str, Any],
) -> torch.Tensor | None:
    outputs = [values[value_id] for value_id in output_value_ids if value_id in values]
    if not outputs:
        return None
    return sum(output.float().pow(2).mean() for output in outputs)


def _remote_forward_count(
    graph: Any,
    logical: Any,
    placement: PlacementPlan,
    execution: ExecutionPlan,
    rank: int,
) -> int:
    owner_rank = _owner_rank_by_partition(placement, execution)
    partition_by_node = _partition_by_node(logical)
    return sum(
        1
        for edge in graph.edges
        if edge.source_node_id in partition_by_node
        and edge.target_node_id in partition_by_node
        if owner_rank[partition_by_node[edge.source_node_id]] == rank
        and owner_rank[partition_by_node[edge.target_node_id]] != rank
    )


def _remote_consumer_ranks(
    graph: Any,
    logical: Any,
    placement: PlacementPlan,
    execution: ExecutionPlan | None = None,
) -> dict[tuple[str, str], tuple[int, ...]]:
    owner_rank = _owner_rank_by_partition(placement, execution)
    partition_by_node = _partition_by_node(logical)
    result: dict[tuple[str, str], set[int]] = {}
    for edge in graph.edges:
        if (
            edge.source_node_id not in partition_by_node
            or edge.target_node_id not in partition_by_node
        ):
            continue
        source = partition_by_node[edge.source_node_id]
        target = partition_by_node[edge.target_node_id]
        if source == target:
            continue
        dst = owner_rank[target]
        if owner_rank[source] != dst:
            result.setdefault((source, edge.value_id), set()).add(dst)
    return {key: tuple(sorted(value)) for key, value in result.items()}


def _remote_inputs_by_partition(
    logical: Any,
    placement: PlacementPlan,
    producer_partition: Mapping[str, str],
    execution: ExecutionPlan | None = None,
) -> dict[str, tuple[tuple[str, int], ...]]:
    owner_rank = _owner_rank_by_partition(placement, execution)
    result: dict[str, list[tuple[str, int]]] = {}
    for partition in logical.partitions:
        for value_id in partition.input_value_ids:
            source = producer_partition.get(value_id)
            if source is None:
                continue
            source_rank = owner_rank[source]
            if source_rank != owner_rank[partition.partition_id]:
                result.setdefault(partition.partition_id, []).append((value_id, source_rank))
    return {key: tuple(value) for key, value in result.items()}


def _producer_partition_by_value(
    graph: Any,
    partition_by_node: Mapping[str, str],
) -> dict[str, str]:
    return {
        value.value_id: partition_by_node[value.producer_node_id]
        for value in graph.values
        if value.producer_node_id in partition_by_node
    }


def _partition_by_node(logical: Any) -> dict[str, str]:
    return {
        node_id: partition.partition_id
        for partition in logical.partitions
        for node_id in partition.node_ids
    }


def _owner_rank_by_partition(
    placement: PlacementPlan,
    execution: ExecutionPlan | None = None,
) -> dict[str, int]:
    if execution is not None:
        rank_by_worker = {
            str(worker.worker_id): worker.rank
            for worker in execution.workers
        }
        return {
            placed.partition_id: rank_by_worker[str(placed.worker_id)]
            for placed in placement.placements
        }
    return {
        placed.partition_id: index % placement.selected_gpu_count
        for index, placed in enumerate(placement.placements)
    }


def _move_tensor(value: Any, device: torch.device) -> Any:
    return value.to(device) if isinstance(value, torch.Tensor) else value


def _to_meta(value: Any) -> Any:
    return value.to("meta") if isinstance(value, torch.Tensor) else value


def _dtype(value: str | None) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float64": torch.float64,
        "int64": torch.int64,
        "long": torch.int64,
        "int32": torch.int32,
        "int": torch.int32,
    }[value or "float32"]


def _device() -> torch.device:
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)
    return torch.device("cpu")


def _backend(device: torch.device) -> str:
    requested = select_backend(os.environ.get("SHARDGRID_BACKEND", "gloo"))
    return requested if device.type == "cuda" else "gloo"


def _lr(config: TrainingConfig) -> float:
    return float(config.model.parameters.get("learning_rate", "1e-3"))


def _stable_index(value: str) -> int:
    return int.from_bytes(hashlib.sha1(value.encode("utf-8")).digest()[:2], "big")


def _snapshot_root() -> Path:
    explicit = os.environ.get("SHARDGRID_REMOTE_SNAPSHOT_ROOT", "").strip()
    if explicit:
        return Path(explicit)
    return Path.cwd().resolve().parent


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _write_failure_diagnostics(exc: Exception) -> dict[str, Any]:
    message = str(exc)
    failure_category = (
        "EXECUTION_PLAN_MISSING"
        if message.startswith("EXECUTION_PLAN_MISSING")
        else "PLAN_RUNTIME_MISMATCH"
        if message.startswith("PLAN_RUNTIME_MISMATCH")
        else "GENERIC_DAG_RUNTIME_FAILED"
    )
    payload = {
        "generic_dag_runtime_used": True,
        "legacy_stage_runtime_used": False,
        "failure_category": failure_category,
        "error_type": type(exc).__name__,
        "message": message,
        "worker_id": os.environ.get("SHARDGRID_WORKER_ID"),
        "rank": int(os.environ.get("RANK", "0")),
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
    }
    path = _snapshot_root() / "diagnostics" / "generic-dag-runtime.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        payload = _write_failure_diagnostics(exc)
        print(EVENT_MARKER + json.dumps(payload, sort_keys=True), flush=True)
        raise SystemExit(78) from exc
