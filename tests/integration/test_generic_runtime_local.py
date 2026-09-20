from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from shardgrid.bootstrap.runner import CaptureResult, capture_entrypoint_workload
from shardgrid.planner.generic_graph import CanonicalGraphIR, FXGraphCaptureAdapter
from shardgrid.planner.planning_contract import (
    LogicalPartitionPlan,
    LogicalPartitionSpec,
    PlacementPlan,
    PlacementSpec,
)
from shardgrid.runtime.checkpoint import save_worker_state_shard
from shardgrid.runtime.dag import (
    LocalDAGRuntime,
    RuntimePartition,
    compile_runtime_plan,
    materialize_worker_owned_state,
)
from shardgrid.runtime.partition_graph import extract_partition_graph

FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "ordinary_training_scripts"


def test_local_generic_runtime_runs_ordinary_entrypoint_training_step(tmp_path: Path) -> None:
    workload = capture_entrypoint_workload(
        FIXTURE_ROOT / "positional_tuple_train.py",
        argv=("--epochs", "1", "--checkpoint", "out/tuple.pt"),
        cwd=FIXTURE_ROOT,
        environment={"SHARDGRID_TEST_CAPTURE": "1"},
    )
    assert not isinstance(workload, CaptureResult)
    assert workload.context.model_identity["class_name"] == "TupleBatchModel"
    assert workload.context.graph_capture["backend"]

    model = workload.model
    sample_args = tuple(_detach(value) for value in workload.sample_args)
    capture = FXGraphCaptureAdapter().capture(model, sample_args=sample_args)
    logical = _two_stage_plan(capture.canonical_graph)
    placement = PlacementPlan(
        graph_fingerprint=capture.graph_fingerprint,
        selected_gpu_count=2,
        placements=(
            PlacementSpec("P0", "gpu0", "worker0", 0),
            PlacementSpec("P1", "gpu1", "worker1", 0),
        ),
    )
    runtime_plan = compile_runtime_plan(capture.canonical_graph, logical, placement)
    runtime = LocalDAGRuntime(
        runtime_plan,
        {
            partition.partition_id: RuntimePartition(
                partition,
                extract_partition_graph(
                    capture.canonical_graph,
                    capture.backend_graph,
                    partition,
                ),
            )
            for partition in logical.partitions
        },
    )
    state_objects = _canonical_state_objects(capture.canonical_graph, model)
    before = {
        state_id: parameter.detach().clone()
        for state_id, parameter in state_objects.items()
        if isinstance(parameter, torch.nn.Parameter)
    }
    model.zero_grad(set_to_none=True)

    outputs, evidence = runtime.forward(
        dict(zip(capture.canonical_graph.input_value_ids, sample_args, strict=True))
    )
    output = outputs[capture.canonical_graph.output_value_ids[0]]
    loss = output.pow(2).mean()
    loss.backward()

    optimizers = {}
    for worker in runtime_plan.ownership.workers:
        materialized = materialize_worker_owned_state(
            runtime_plan,
            worker_id=worker.worker_id,
            gpu_index=worker.gpu_index,
            state_objects=state_objects,
        )
        assert tuple(materialized.parameters) == worker.local_parameter_ids
        if materialized.parameters:
            optimizers[worker.worker_id] = torch.optim.SGD(
                materialized.parameters.values(),
                lr=0.05,
            )
    evidence = runtime.optimizer_step(optimizers, evidence)

    shard_results = []
    for worker in runtime_plan.ownership.workers:
        changed = any(
            not torch.equal(before[state_id], state_objects[state_id])
            for state_id in worker.local_parameter_ids
        )
        shard_results.append(
            save_worker_state_shard(
                tmp_path / f"{worker.worker_id}.pt",
                graph=capture.canonical_graph,
                runtime_plan=runtime_plan,
                worker_id=worker.worker_id,
                gpu_index=worker.gpu_index,
                state_dict=model.state_dict(),
                job_id="job-local-runtime",
                plan_id="plan-local-runtime",
                training_step=1,
                metadata={
                    "checked_parameter_count": len(worker.local_parameter_ids),
                    "parameter_changed": changed,
                },
            )
        )

    assert evidence.executed_partition_sequence == ("P0", "P1")
    assert evidence.remote_edges_used == 1
    assert evidence.optimizer_step_completed is True
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(result.parameter_count > 0 for result in shard_results)
    assert all(result.bytes > 0 for result in shard_results)
    assert any(
        not torch.equal(before[state_id], state_objects[state_id])
        for state_id in before
    )


def _two_stage_plan(graph: CanonicalGraphIR) -> LogicalPartitionPlan:
    modules = tuple(node for node in graph.nodes if node.op_kind == "call_module")
    assert len(modules) >= 3
    left = modules[:2]
    right = modules[2:]
    boundary = left[-1].output_value_ids[0]
    partitions = (
        _partition("P0", left, graph.input_value_ids, (boundary,)),
        _partition("P1", right, (boundary,), graph.output_value_ids),
    )
    return LogicalPartitionPlan(graph.graph_fingerprint, partitions)


def _partition(
    partition_id: str,
    nodes: tuple[Any, ...],
    input_value_ids: tuple[str, ...],
    output_value_ids: tuple[str, ...],
) -> LogicalPartitionSpec:
    parameter_ids = tuple(
        state_id for node in nodes for state_id in node.parameter_ids
    )
    buffer_ids = tuple(state_id for node in nodes for state_id in node.buffer_ids)
    return LogicalPartitionSpec(
        partition_id=partition_id,
        node_ids=tuple(node.node_id for node in nodes),
        input_value_ids=input_value_ids,
        output_value_ids=output_value_ids,
        parameter_ids=parameter_ids,
        buffer_ids=buffer_ids,
        estimated_compute=1,
        estimated_memory=1,
        boundary_edges=output_value_ids,
        owned_state_ids=parameter_ids + buffer_ids,
    )


def _canonical_state_objects(graph: CanonicalGraphIR, model: Any) -> dict[str, Any]:
    named_parameters = dict(model.named_parameters())
    named_buffers = dict(model.named_buffers())
    return {
        use.parameter_id: named_parameters[use.canonical_path]
        for use in graph.parameter_uses
    } | {
        buffer_id: named_buffers[path]
        for node in graph.nodes
        for buffer_id, path in zip(node.buffer_ids, node.buffer_paths, strict=True)
    }


def _detach(value: Any) -> Any:
    return value.detach().clone() if isinstance(value, torch.Tensor) else value
