from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from shardgrid.planner.generic_graph import (
    GenericGraphIR,
    GraphEdgeSpec,
    GraphNodeSpec,
    GraphValueSpec,
    StateObjectSpec,
)
from shardgrid.planner.planning_contract import (
    LogicalPartitionPlan,
    LogicalPartitionSpec,
    PlacementPlan,
    PlacementSpec,
)
from shardgrid.runtime.dag import EdgeKind, compile_runtime_plan, materialize_worker_owned_state


def test_runtime_compiles_exact_planner_ownership_and_placement() -> None:
    graph, logical, placement = _exact_plan_fixture()

    runtime_plan = compile_runtime_plan(graph, logical, placement)

    assert runtime_plan.graph_fingerprint == "graph-exact"
    assert [worker.worker_id for worker in runtime_plan.ownership.workers] == [
        "worker-a",
        "worker-b",
    ]
    worker_a, worker_b = runtime_plan.ownership.workers
    assert worker_a.gpu_id == "gpu-a1"
    assert worker_a.gpu_index == 1
    assert worker_a.owned_partitions == ("partition-b",)
    assert worker_a.local_parameter_ids == ("p1",)
    assert worker_a.local_buffer_ids == ("b0",)
    assert worker_a.read_only_state_ids == ("p0",)
    assert worker_b.gpu_id == "gpu-b0"
    assert worker_b.gpu_index == 0
    assert worker_b.owned_partitions == ("partition-a", "partition-c")
    assert worker_b.local_parameter_ids == ("p0", "p2")
    assert runtime_plan.logical_partitions == logical.partitions
    assert runtime_plan.placements == placement.placements


def test_runtime_edges_preserve_planner_boundary_values() -> None:
    graph, logical, placement = _exact_plan_fixture()

    runtime_plan = compile_runtime_plan(graph, logical, placement)

    edges = {
        (edge.producer_partition, edge.consumer_partition, edge.value_id): edge
        for edge in runtime_plan.edges
    }
    first = edges[("partition-a", "partition-b", "v0")]
    second = edges[("partition-b", "partition-c", "v1")]
    assert first.edge_kind is EdgeKind.REMOTE
    assert first.producer_worker_id == "worker-b"
    assert first.consumer_worker_id == "worker-a"
    assert first.producer_gpu_id == "gpu-b0"
    assert first.consumer_gpu_id == "gpu-a1"
    assert first.shape == (4, 8)
    assert first.dtype == "float32"
    assert first.forward_transfer_bytes == 128
    assert first.backward_transfer_bytes == 128
    assert second.edge_kind is EdgeKind.REMOTE
    assert second.producer_worker_id == "worker-a"
    assert second.consumer_worker_id == "worker-b"


def test_runtime_does_not_round_robin_by_rank_or_world_size() -> None:
    graph, logical, placement = _exact_plan_fixture()

    runtime_plan = compile_runtime_plan(graph, logical, placement)

    by_partition = {
        partition_id: worker.worker_id
        for worker in runtime_plan.ownership.workers
        for partition_id in worker.owned_partitions
    }
    assert by_partition == {
        "partition-a": "worker-b",
        "partition-b": "worker-a",
        "partition-c": "worker-b",
    }
    assert by_partition["partition-a"] == by_partition["partition-c"]


@pytest.mark.parametrize("field", ["logical", "placement"])
def test_runtime_rejects_graph_fingerprint_mismatch(field: str) -> None:
    graph, logical, placement = _exact_plan_fixture()
    if field == "logical":
        logical = replace(logical, graph_fingerprint="other-graph")
    else:
        placement = replace(placement, graph_fingerprint="other-graph")

    with pytest.raises(ValueError, match="PLAN_VALIDATION_FAILURE.*fingerprint mismatch"):
        compile_runtime_plan(graph, logical, placement)


def test_runtime_rejects_non_exact_placement_coverage() -> None:
    graph, logical, placement = _exact_plan_fixture()
    placement = replace(placement, placements=placement.placements[:-1])

    with pytest.raises(ValueError, match="PLAN_VALIDATION_FAILURE.*cover logical partitions"):
        compile_runtime_plan(graph, logical, placement)


def test_worker_materializes_only_explicitly_owned_parameter_and_buffer_state() -> None:
    graph, logical, placement = _exact_plan_fixture()
    runtime_plan = compile_runtime_plan(graph, logical, placement)
    state_objects = {
        "p0": torch.nn.Parameter(torch.ones(2, 2)),
        "p1": torch.nn.Parameter(torch.ones(2, 2)),
        "p2": torch.nn.Parameter(torch.ones(2, 2)),
        "b0": torch.ones(2),
        "unused": torch.nn.Parameter(torch.zeros(1)),
    }

    materialized = materialize_worker_owned_state(
        runtime_plan,
        worker_id="worker-a",
        gpu_index=1,
        state_objects=state_objects,
    )

    assert materialized.worker.owned_partitions == ("partition-b",)
    assert materialized.parameters == {"p1": state_objects["p1"]}
    assert materialized.buffers == {"b0": state_objects["b0"]}
    assert materialized.read_only_state_ids == ("p0",)
    assert materialized.materialized_state_ids == ("p1", "b0")
    assert "p0" not in materialized.parameters
    assert "p0" not in materialized.buffers
    assert "p2" not in materialized.parameters
    assert "unused" not in materialized.parameters


def test_worker_materialization_preserves_canonical_owner_for_shared_state() -> None:
    graph, logical, placement = _exact_plan_fixture()
    runtime_plan = compile_runtime_plan(graph, logical, placement)
    state_objects = {
        "p0": torch.nn.Parameter(torch.ones(2, 2)),
        "p1": torch.nn.Parameter(torch.ones(2, 2)),
        "p2": torch.nn.Parameter(torch.ones(2, 2)),
        "b0": torch.ones(2),
    }

    worker_a = materialize_worker_owned_state(
        runtime_plan,
        worker_id="worker-a",
        gpu_index=1,
        state_objects=state_objects,
    )
    worker_b = materialize_worker_owned_state(
        runtime_plan,
        worker_id="worker-b",
        gpu_index=0,
        state_objects=state_objects,
    )

    assert worker_a.read_only_state_ids == ("p0",)
    assert "p0" not in worker_a.materialized_state_ids
    assert worker_b.worker.owned_partitions == ("partition-a", "partition-c")
    assert worker_b.parameters == {"p0": state_objects["p0"], "p2": state_objects["p2"]}
    assert worker_b.buffers == {}


def test_worker_materialization_keeps_parameterless_execution_nodes_state_free() -> None:
    graph, logical, placement = _exact_plan_fixture()
    runtime_plan = compile_runtime_plan(graph, logical, placement)

    worker_b = materialize_worker_owned_state(
        runtime_plan,
        worker_id="worker-b",
        gpu_index=0,
        state_objects={
            "p0": torch.nn.Parameter(torch.ones(2, 2)),
            "p1": torch.nn.Parameter(torch.ones(2, 2)),
            "p2": torch.nn.Parameter(torch.ones(2, 2)),
            "b0": torch.ones(2),
        },
    )

    partition_c = next(
        partition
        for partition in runtime_plan.logical_partitions
        if partition.partition_id == "partition-c"
    )
    parameterless = next(node for node in graph.nodes if node.node_id == "n3")
    assert partition_c.node_ids == ("n2", "n3")
    assert parameterless.parameter_ids == ()
    assert parameterless.buffer_ids == ()
    assert worker_b.materialized_state_ids == ("p0", "p2")


def test_worker_materialization_rejects_missing_owned_state() -> None:
    graph, logical, placement = _exact_plan_fixture()
    runtime_plan = compile_runtime_plan(graph, logical, placement)

    with pytest.raises(ValueError, match="missing owned worker state.*p1"):
        materialize_worker_owned_state(
            runtime_plan,
            worker_id="worker-a",
            gpu_index=1,
            state_objects={"b0": torch.ones(2)},
        )


def _exact_plan_fixture() -> tuple[GenericGraphIR, LogicalPartitionPlan, PlacementPlan]:
    graph = GenericGraphIR(
        graph_fingerprint="graph-exact",
        capture_backend="unit",
        input_value_ids=("input",),
        output_value_ids=("v3",),
        parameter_owners={},
        nodes=(
            GraphNodeSpec(
                node_id="n0",
                op_kind="call_module",
                target="encoder",
                module_path="encoder",
                input_value_ids=("input",),
                output_value_ids=("v0",),
                parameter_ids=("p0",),
            ),
            GraphNodeSpec(
                node_id="n1",
                op_kind="call_module",
                target="middle",
                module_path="middle",
                input_value_ids=("v0",),
                output_value_ids=("v1",),
                parameter_ids=("p1",),
                buffer_ids=("b0",),
            ),
            GraphNodeSpec(
                node_id="n2",
                op_kind="call_module",
                target="decoder",
                module_path="decoder",
                input_value_ids=("v1", "v0"),
                output_value_ids=("v2",),
                parameter_ids=("p2",),
            ),
            GraphNodeSpec(
                node_id="n3",
                op_kind="call_function",
                target="torch.relu",
                module_path=None,
                input_value_ids=("v2",),
                output_value_ids=("v3",),
            ),
        ),
        values=(
            GraphValueSpec("input", None, ("n0",)),
            GraphValueSpec(
                "v0",
                "n0",
                ("n1", "n2"),
                shape=(4, 8),
                dtype="float32",
                requires_grad=True,
                estimated_bytes=128,
            ),
            GraphValueSpec(
                "v1",
                "n1",
                ("n2",),
                shape=(4, 8),
                dtype="float32",
                requires_grad=True,
                estimated_bytes=128,
            ),
            GraphValueSpec("v2", "n2", ("n3",), shape=(4, 2), dtype="float32"),
            GraphValueSpec("v3", "n3", (), shape=(4, 2), dtype="float32"),
        ),
        edges=(
            GraphEdgeSpec(
                "n0",
                "n1",
                "v0",
                forward_transfer_bytes=128,
                backward_transfer_bytes=128,
            ),
            GraphEdgeSpec(
                "n0",
                "n2",
                "v0",
                forward_transfer_bytes=128,
                backward_transfer_bytes=128,
            ),
            GraphEdgeSpec(
                "n1",
                "n2",
                "v1",
                forward_transfer_bytes=128,
                backward_transfer_bytes=128,
            ),
            GraphEdgeSpec("n2", "n3", "v2"),
        ),
        states=(
            StateObjectSpec("p0", "parameter", "encoder.weight"),
            StateObjectSpec("p1", "parameter", "middle.weight"),
            StateObjectSpec("p2", "parameter", "decoder.weight"),
            StateObjectSpec("b0", "buffer", "middle.running_mean"),
        ),
    )
    logical = LogicalPartitionPlan(
        graph_fingerprint="graph-exact",
        partitions=(
            LogicalPartitionSpec(
                partition_id="partition-a",
                node_ids=("n0",),
                input_value_ids=("input",),
                output_value_ids=("v0",),
                parameter_ids=("p0",),
                buffer_ids=(),
                estimated_compute=1,
                estimated_memory=10,
                boundary_edges=("v0",),
                owned_state_ids=("p0",),
            ),
            LogicalPartitionSpec(
                partition_id="partition-b",
                node_ids=("n1",),
                input_value_ids=("v0",),
                output_value_ids=("v1",),
                parameter_ids=("p1",),
                buffer_ids=("b0",),
                estimated_compute=1,
                estimated_memory=10,
                boundary_edges=("v0", "v1"),
                owned_state_ids=("p1", "b0"),
                read_only_state_ids=("p0",),
            ),
            LogicalPartitionSpec(
                partition_id="partition-c",
                node_ids=("n2", "n3"),
                input_value_ids=("v1", "v0"),
                output_value_ids=("v3",),
                parameter_ids=("p2",),
                buffer_ids=(),
                estimated_compute=1,
                estimated_memory=10,
                boundary_edges=("v1",),
                owned_state_ids=("p2",),
                read_only_state_ids=("p0",),
            ),
        ),
    )
    placement = PlacementPlan(
        graph_fingerprint="graph-exact",
        selected_gpu_count=2,
        placements=(
            PlacementSpec("partition-a", "gpu-b0", "worker-b", 0),
            PlacementSpec("partition-b", "gpu-a1", "worker-a", 1),
            PlacementSpec("partition-c", "gpu-b0", "worker-b", 0),
        ),
    )
    return graph, logical, placement
