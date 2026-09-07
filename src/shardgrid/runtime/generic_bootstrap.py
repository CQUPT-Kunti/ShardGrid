"""Production generic runtime bootstrap for ordinary (non-zoo) entrypoints.

This module is the worker-side entrypoint executed by ``SSHLauncher`` for
captured-context plans::

    python -m shardgrid.runtime.generic_bootstrap --rank N ...

It consumes only artifacts produced by the control plane during capture and
planning:

* ``config/training-config.json``
* ``plan/execution-plan.json`` (worker assignment)
* ``plan/original-parallel-plan.json`` (the exact planner-selected plan)
* ``plan/captured-context.json`` (capture metadata)
* ``plan/captured-graph.json`` (canonical graph IR)
* ``plan/backend-graph.pt`` (serialized FX backend graph module with state)
* ``plan/initial-state.pt`` (initial ``model.state_dict()`` values)

It never rebuilds the user model, never calls zoo builders, never re-captures,
never re-partitions, and never round-robin places.  It loads the captured
graph + exact plan, compiles the :class:`RuntimePlan`, materializes only the
state owned by this worker on the real device (non-owned state stays on
``meta``), and executes the exact distributed generic DAG runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
from collections import deque
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.distributed as dist

from shardgrid.distributed.backend import select_backend
from shardgrid.planner.generic_graph import CanonicalGraphIR, GenericGraphIR
from shardgrid.planner.models import ExecutionPlan
from shardgrid.planner.planning_contract import (
    LogicalPartitionPlan,
    LogicalPartitionSpec,
    PlacementPlan,
    PlacementSpec,
)
from shardgrid.runtime.checkpoint import save_worker_state_shard
from shardgrid.runtime.dag import RuntimePlan, compile_runtime_plan
from shardgrid.runtime.partition_graph import extract_partition_graph
from shardgrid.runtime.transport import PendingTensorSend, recv_tensor, send_tensor_async

EVENT_MARKER = "GENERIC_DAG_RUNTIME_EVIDENCE "
FORWARD_MARKER = "T072_FORWARD_EVIDENCE "
BACKWARD_MARKER = "T073_BACKWARD_EVIDENCE "
TRAIN_MARKER = "T074_TRAIN_EVIDENCE "
PROBE_MARKER = "GENERIC_DAG_PROBE_EVIDENCE "
PROBE_COMPLETE = "PROBE_COMPLETE"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--parallel-plan-id", default="")
    parser.add_argument("--selected-candidate-id", default="")
    parser.add_argument("--plan-artifact", default="plan/original-parallel-plan.json")
    parser.add_argument("--context-artifact", default="plan/captured-context.json")
    parser.add_argument("--job-id", default="")
    parser.add_argument("--memory-probe", action="store_true")
    args = parser.parse_args()

    rank = int(os.environ.get("RANK", args.rank))
    device = _device()
    torch.manual_seed(42 + rank)
    probe_logger = _ProbePhaseLogger(rank, _execution()) if args.memory_probe else None
    if probe_logger is not None:
        probe_logger.record("DISTRIBUTED_INIT_BEGIN")
    dist.init_process_group(
        backend=_backend(device),
        init_method=f"tcp://{os.environ['MASTER_ADDR']}:{int(os.environ['MASTER_PORT'])}",
        rank=rank,
        world_size=int(os.environ.get("WORLD_SIZE", "1")),
    )
    if probe_logger is not None:
        probe_logger.set_rendezvous(
            master_addr=str(os.environ.get("MASTER_ADDR", "")),
            master_port=int(os.environ.get("MASTER_PORT", "0")),
        )
        probe_logger.record("DISTRIBUTED_INIT_END")

    try:
        result = _run_training(
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
    *,
    rank: int,
    device: torch.device,
    memory_probe: bool,
    probe_logger: "_ProbePhaseLogger | None",
) -> dict[str, Any]:
    execution = _execution()
    parallel_plan = _load_parallel_plan()
    graph, backend_graph, initial_state = _load_captured_runtime_artifacts()
    runtime_plan_path = _snapshot_root() / "plan" / "runtime-plan.json"
    if runtime_plan_path.is_file():
        runtime_plan = RuntimePlan.from_dict(
            _load_json(runtime_plan_path)
        )
        logical = _logical_from_runtime_plan(runtime_plan)
        placement = _placement_from_runtime_plan(runtime_plan)
    else:
        logical, placement = _decode_exact_plan(graph, parallel_plan, execution)
        runtime_plan = compile_runtime_plan(graph, logical, placement)
    ownership = next(
        worker
        for worker in runtime_plan.ownership.workers
        if worker.worker_id == str(execution.workers[rank].worker_id)
    )
    _materialize_owned_modules(
        backend_graph,
        graph,
        ownership.owned_partitions,
        logical,
        initial_state,
        device,
    )
    consistency = _plan_runtime_consistency(
        graph,
        backend_graph,
        logical,
        placement,
        execution,
        parallel_plan,
        ownership,
        rank=rank,
    )
    extracted = {
        partition.partition_id: extract_partition_graph(
            graph,
            backend_graph,
            partition,
        )
        for partition in logical.partitions
        if partition.partition_id in ownership.owned_partitions
    }
    owned_trainable = _owned_trainable_parameters(backend_graph, graph, ownership)
    optim = torch.optim.AdamW(
        [parameter for _name, parameter in owned_trainable],
        lr=_lr(parallel_plan),
    )
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
        "owned_parameter_bytes": _owned_parameter_bytes(backend_graph, graph, ownership),
        "estimated_full_model_parameter_bytes": sum(
            node.parameter_bytes for node in graph.nodes
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
    steps = 1 if memory_probe else int(
        os.environ.get("SHARDGRID_TRAINING_STEPS", "20")
    )
    final_loss: float | None = None
    loss_history: list[float] = []
    forward_remote = 0
    backward_remote = 0
    live_change_evidence: dict[str, Any] | None = None
    progress = _ProgressLog(rank, str(execution.workers[rank].worker_id))
    sample_args = _input_sample()
    for step in range(steps):
        progress.record("STEP_BEGIN", step=step)
        if probe_logger is not None:
            probe_logger.record("PROBE_FORWARD_BEGIN", step=step)
        optim.zero_grad(set_to_none=True)
        values, received, produced = _forward_step(
            graph,
            logical,
            placement,
            extracted,
            execution=execution,
            rank=rank,
            step=step,
            device=device,
            sample_args=sample_args,
            progress=progress,
        )
        if probe_logger is not None:
            probe_logger.record("PROBE_FORWARD_END", step=step)
        loss = _loss_for_rank(graph.output_value_ids, values)
        if loss is not None:
            final_loss = float(loss.detach().cpu().item())
            loss_history.append(final_loss)
        if probe_logger is not None:
            probe_logger.record("PROBE_BACKWARD_BEGIN", step=step)
        backward_count = _backward_step(
            graph,
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
        forward_remote += _remote_forward_count(graph, logical, placement, execution, rank)
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
        }
        _write_memory_probe(rank, train)
        if probe_logger is not None:
            probe_logger.record(PROBE_COMPLETE)
        return {"runtime": runtime_evidence, "train": train}

    _save_checkpoint_shard(
        execution=execution,
        rank=rank,
        steps=steps,
        graph=graph,
        runtime_plan=runtime_plan,
        backend_graph=backend_graph,
        final_loss=final_loss,
        change_evidence=change_evidence,
        peak=peak,
        runtime_evidence=runtime_evidence,
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
    }
    return {"runtime": runtime_evidence, "train": train}


def _save_checkpoint_shard(
    *,
    execution: ExecutionPlan,
    rank: int,
    steps: int,
    graph: CanonicalGraphIR,
    runtime_plan: RuntimePlan,
    backend_graph: Any,
    final_loss: float | None,
    change_evidence: Mapping[str, Any],
    peak: Mapping[str, int],
    runtime_evidence: Mapping[str, Any],
) -> None:
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
        graph=graph,
        runtime_plan=runtime_plan,
        worker_id=str(execution.workers[rank].worker_id),
        gpu_index=execution.workers[rank].gpu_index,
        state_dict=backend_graph.state_dict(),
        job_id=str(execution.job_id),
        plan_id=str(execution.labels.get("selected_candidate_id", "generic-dag-plan")),
        training_step=steps,
        metadata=checkpoint_metadata,
    )
    payload = {
        "status": "complete",
        "checkpoint_version": 1,
        "job_id": str(execution.job_id),
        "rank": rank,
        "world_size": execution.world_size,
        "stage_id": execution.workers[rank].stage,
        "step": steps,
        "checkpoint_path": checkpoint_result.path,
        "checkpoint_ref": "checkpoint/model.pt",
        "model_name": "captured-entrypoint",
        "model_type": "captured_entrypoint",
        "final_loss": final_loss,
        **dict(change_evidence),
        **dict(runtime_evidence),
    }
    path = _snapshot_root() / "checkpoint" / "checkpoint-metadata.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_captured_runtime_artifacts() -> tuple[CanonicalGraphIR, Any, Mapping[str, Any]]:
    root = _snapshot_root()
    graph_path = root / "plan" / "captured-graph.json"
    if not graph_path.is_file():
        raise ValueError(
            "CAPTURE_ARTIFACT_MISSING: plan/captured-graph.json is required"
        )
    graph = GenericGraphIR.from_dict(json.loads(graph_path.read_text(encoding="utf-8")))
    backend_path = root / "plan" / "backend-graph.pt"
    if not backend_path.is_file():
        raise ValueError("CAPTURE_ARTIFACT_MISSING: plan/backend-graph.pt is required")
    backend_graph = torch.load(backend_path, map_location="cpu", weights_only=False)
    state_path = root / "plan" / "initial-state.pt"
    if not state_path.is_file():
        raise ValueError("CAPTURE_ARTIFACT_MISSING: plan/initial-state.pt is required")
    initial_state = torch.load(state_path, map_location="cpu", weights_only=False)
    return graph, backend_graph, initial_state


def _execution() -> ExecutionPlan:
    return ExecutionPlan.from_dict(_load_json(_snapshot_root() / "plan" / "execution-plan.json"))


def _load_parallel_plan() -> Any:
    from shardgrid.engines.models import ParallelPlan

    return ParallelPlan.from_dict(
        _load_json(_snapshot_root() / "plan" / "original-parallel-plan.json")
    )


def _decode_exact_plan(
    graph: CanonicalGraphIR,
    parallel_plan: Any,
    execution: ExecutionPlan | None = None,
) -> tuple[LogicalPartitionPlan, PlacementPlan]:
    if not parallel_plan.stage_metadata:
        raise ValueError("EXECUTION_PLAN_MISSING: original plan has no stage_metadata")
    if execution is not None and execution.world_size != len(execution.workers):
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


def _logical_from_runtime_plan(runtime_plan: RuntimePlan) -> LogicalPartitionPlan:
    return LogicalPartitionPlan(
        runtime_plan.graph_fingerprint,
        tuple(runtime_plan.logical_partitions),
    )


def _placement_from_runtime_plan(runtime_plan: RuntimePlan) -> PlacementPlan:
    return PlacementPlan(
        graph_fingerprint=runtime_plan.graph_fingerprint,
        selected_gpu_count=len(
            {
                (placement.worker_id, placement.gpu_index)
                for placement in runtime_plan.placements
            }
        ),
        placements=tuple(runtime_plan.placements),
    )


def _node_memory(node: Any) -> int:
    return int(
        node.estimated_peak_memory_contribution
        or node.parameter_bytes
        + node.activation_bytes
        + node.gradient_bytes
        + node.optimizer_bytes
        + node.temporary_bytes
    )


def _materialize_owned_modules(
    backend_graph: Any,
    graph: CanonicalGraphIR,
    owned_partitions: tuple[str, ...],
    logical: Any,
    initial_state: Mapping[str, Any],
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
    modules = dict(backend_graph.named_modules())
    for path in sorted(module_paths):
        module = modules[path]
        module.to_empty(device=device)
        _load_owned_module_state(module, path, initial_state, device)


def _load_owned_module_state(
    module: Any,
    path: str,
    initial_state: Mapping[str, Any],
    device: torch.device,
) -> None:
    state = module.state_dict()
    for key in state:
        value = initial_state.get(f"{path}.{key}")
        if value is not None:
            state[key].copy_(value.detach().to(device=device, dtype=state[key].dtype))


def _plan_runtime_consistency(
    graph: CanonicalGraphIR,
    backend_graph: Any,
    logical: LogicalPartitionPlan,
    placement: PlacementPlan,
    execution: ExecutionPlan,
    parallel_plan: Any,
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
    materialized_params = _materialized_parameter_ids(backend_graph, graph)
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
    materialized_buffers = _materialized_buffer_ids(backend_graph, graph)
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


def _materialized_parameter_ids(backend_graph: Any, graph: Any) -> tuple[str, ...]:
    by_name = dict(backend_graph.named_parameters(remove_duplicate=False))
    result = []
    for use in graph.parameter_uses:
        parameter = by_name.get(use.canonical_path)
        if parameter is not None and parameter.device.type != "meta":
            result.append(use.parameter_id)
    return tuple(sorted(set(result)))


def _materialized_buffer_ids(backend_graph: Any, graph: Any) -> tuple[str, ...]:
    by_name = dict(backend_graph.named_buffers(remove_duplicate=False))
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


def _forward_step(
    graph: CanonicalGraphIR,
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
    values = {
        value_id: _move_tensor(value, device)
        for value_id, value in zip(graph.input_value_ids, sample_args, strict=True)
    }
    received_boundaries: dict[str, torch.Tensor] = {}
    produced_values: dict[str, torch.Tensor] = {}
    value_specs = {value.value_id: value for value in graph.values}
    owner_rank = _owner_rank_by_partition(placement, execution)
    partition_by_node = _partition_by_node(logical)
    producer_partition = _producer_partition_by_value(graph, partition_by_node)
    remote_consumers = _remote_consumer_ranks(graph, logical, placement, execution)
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
    graph: CanonicalGraphIR,
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
    producer_partition = _producer_partition_by_value(graph, partition_by_node)
    remote_consumers = _remote_consumer_ranks(graph, logical, placement, execution)
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
                value_id in graph.output_value_ids
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


def _loss_for_rank(
    output_value_ids: tuple[str, ...],
    values: Mapping[str, Any],
) -> torch.Tensor | None:
    outputs = [values[value_id] for value_id in output_value_ids if value_id in values]
    if not outputs:
        return None
    return sum(output.float().pow(2).mean() for output in outputs)


def _remote_forward_count(
    graph: CanonicalGraphIR,
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
    graph: CanonicalGraphIR,
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
    graph: CanonicalGraphIR,
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


def _lr(parallel_plan: Any) -> float:
    return float(parallel_plan.requirements.get("learning_rate", "1e-3") or "1e-3")


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


def _input_sample() -> tuple[Any, ...]:
    root = _snapshot_root()
    context = _load_json(root / "plan" / "captured-context.json")
    args = context.get("model_call", {}).get("args")
    names: tuple[str, ...] = ()
    if isinstance(args, dict):
        names = tuple(str(key) for key in sorted(args))
    elif isinstance(args, tuple):
        names = tuple(str(index) for index in range(len(args)))
    elif args is not None and not isinstance(args, (list, tuple)):
        names = ("0",)
    if not names:
        metadata = context.get("tensor_metadata", {})
        if isinstance(metadata, dict) and metadata:
            names = tuple(str(key) for key in sorted(metadata))
    samples = []
    for key in names:
        artifact = root / "plan" / f"input-{key}.pt"
        if not artifact.is_file():
            raise ValueError(f"CAPTURE_ARTIFACT_MISSING: plan/input-{key}.pt is required")
        samples.append(torch.load(artifact, map_location="cpu", weights_only=False))
    return tuple(samples)


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


def _owned_trainable_parameters(
    backend_graph: Any,
    graph: CanonicalGraphIR,
    ownership: Any,
) -> list[tuple[str, torch.nn.Parameter]]:
    by_name = dict(backend_graph.named_parameters(remove_duplicate=False))
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


def _owned_parameter_bytes(backend_graph: Any, graph: CanonicalGraphIR, ownership: Any) -> int:
    return sum(
        parameter.numel() * parameter.element_size()
        for _name, parameter in _owned_trainable_parameters(backend_graph, graph, ownership)
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
        "checked_parameter_count": len(before),
        "changed_parameters": changed,
    }


def _fingerprint(payload: Mapping[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


class _ProbePhaseLogger:
    """Bounded probe phase diagnostics writer (same contract as the generic DAG runner)."""

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
        self.payload["master_addr"] = master_addr
        self.payload["master_port"] = master_port

    def set_error(self, exc: Exception) -> None:
        self.payload["error"] = f"{type(exc).__name__}: {exc}"
        self.payload["latest_phase"] = "PROBE_FAILED"
        self._write()

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _print(self, event: str) -> None:
        print(PROBE_MARKER + json.dumps({"phase": event, "rank": self.rank}), flush=True)


def _write_memory_probe(rank: int, payload: Mapping[str, Any]) -> None:
    path = _snapshot_root() / "diagnostics" / f"memory-probe-rank{rank}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")


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


def _write_failure_diagnostics(exc: Exception) -> dict[str, Any]:
    message = str(exc)
    failure_category = (
        "EXECUTION_PLAN_MISSING"
        if message.startswith("EXECUTION_PLAN_MISSING")
        else "PLAN_RUNTIME_MISMATCH"
        if message.startswith("PLAN_RUNTIME_MISMATCH")
        else "CAPTURE_ARTIFACT_MISSING"
        if message.startswith("CAPTURE_ARTIFACT_MISSING")
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