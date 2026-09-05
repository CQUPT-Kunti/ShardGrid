from __future__ import annotations

import pytest
import torch
from examples.models.generic_partition_zoo import build_zoo_model, make_zoo_sample
from torch import nn

from shardgrid.planner.generic_graph import (
    FXGraphCaptureAdapter,
    GenericGraphIR,
    GraphEdgeSpec,
    GraphNodeSpec,
)
from shardgrid.planner.planning_contract import (
    LogicalPartitionPlan,
    LogicalPartitionSpec,
    PlacementPlan,
    PlacementSpec,
)
from shardgrid.runtime.checkpoint import (
    consolidate_worker_state_shards,
    save_worker_state_shard,
)
from shardgrid.runtime.dag import (
    EdgeKind,
    LocalDAGRuntime,
    RuntimePartition,
    ValueStore,
    compile_runtime_plan,
)
from shardgrid.runtime.partition_graph import extract_partition_graph


def test_worker_ownership_allows_non_contiguous_partitions_on_same_gpu() -> None:
    graph, logical, placement = _runtime_fixture()

    runtime_plan = compile_runtime_plan(graph, logical, placement)

    worker0 = next(
        worker for worker in runtime_plan.ownership.workers if worker.worker_id == "worker0"
    )
    assert worker0.owned_partitions == ("P0", "P4")
    assert set(worker0.local_parameter_ids) == {"p0", "p4"}
    assert any(
        edge.producer_partition == "P0"
        and edge.consumer_partition == "P4"
        and edge.edge_kind is EdgeKind.LOCAL
        for edge in runtime_plan.edges
    )
    assert any(edge.edge_kind is EdgeKind.REMOTE for edge in runtime_plan.edges)


def test_value_store_releases_values_after_last_consumer() -> None:
    value = torch.ones(2, 2)
    store = ValueStore({"v0": 2})

    store.put("v0", value)
    store.consume("v0")
    assert store.get("v0") is value
    store.consume("v0")

    assert store.current_bytes == 0
    assert store.peak_bytes == value.numel() * value.element_size()


def test_local_dag_runtime_forward_backward_optimizer_non_contiguous_placement() -> None:
    graph, logical, placement = _runtime_fixture()
    runtime_plan = compile_runtime_plan(graph, logical, placement)
    modules = [nn.Linear(4, 4) for _ in range(5)]
    before = [module.weight.detach().clone() for module in modules]

    runtime = LocalDAGRuntime(
        runtime_plan,
        {
            "P0": RuntimePartition(
                logical.partitions[0],
                lambda values: {"v0": modules[0](values["input"])},
            ),
            "P1": RuntimePartition(
                logical.partitions[1],
                lambda values: {"v1": modules[1](values["v0"])},
            ),
            "P2": RuntimePartition(
                logical.partitions[2],
                lambda values: {"v2": modules[2](values["v1"])},
            ),
            "P3": RuntimePartition(
                logical.partitions[3],
                lambda values: {"v3": modules[3](values["v2"])},
            ),
            "P4": RuntimePartition(
                logical.partitions[4],
                lambda values: {"v4": modules[4](values["v3"] + values["v0"])},
            ),
        },
    )

    outputs, evidence = runtime.forward({"input": torch.randn(2, 4)})
    loss = outputs["v4"].sum()
    loss.backward()
    optimizers = {
        "worker0": torch.optim.SGD(_unique_parameters((modules[0], modules[4])), lr=0.01),
        "worker1": torch.optim.SGD(_unique_parameters((modules[1], modules[3])), lr=0.01),
        "worker2": torch.optim.SGD(_unique_parameters((modules[2],)), lr=0.01),
    }
    evidence = runtime.optimizer_step(optimizers, evidence)

    assert evidence.executed_partition_sequence == ("P0", "P1", "P2", "P3", "P4")
    assert evidence.local_edges_used == 1
    assert evidence.remote_edges_used == 4
    assert evidence.optimizer_step_completed is True
    assert modules[0].weight.grad is not None
    assert torch.isfinite(modules[0].weight.grad).all()
    assert any(not torch.equal(old, module.weight) for old, module in zip(before, modules))


def test_extract_partition_graph_residual_mlp_matches_full_backward_optimizer() -> None:
    _assert_extracted_partitions_match_full("residual_mlp_dag", partition_count=3)


def test_extract_partition_graph_unet_matches_full_backward_optimizer() -> None:
    _assert_extracted_partitions_match_full("mini_unet", partition_count=4)


def test_extract_partition_graph_densenet_matches_full_backward_optimizer() -> None:
    _assert_extracted_partitions_match_full("mini_densenet", partition_count=4)


def test_checkpoint_shards_consolidate_complete_state_dict(tmp_path) -> None:
    model = build_zoo_model("mini_unet")
    args, _kwargs = make_zoo_sample("mini_unet")
    capture = FXGraphCaptureAdapter().capture(model, sample_args=args)
    logical = _logical_plan_for_extractor(capture.canonical_graph, 4)
    placement = PlacementPlan(
        graph_fingerprint=capture.graph_fingerprint,
        selected_gpu_count=2,
        placements=tuple(
            PlacementSpec(
                partition.partition_id,
                f"gpu{index % 2}",
                f"worker{index % 2}",
                0,
            )
            for index, partition in enumerate(logical.partitions)
        ),
    )
    runtime_plan = compile_runtime_plan(capture.canonical_graph, logical, placement)
    shard_paths = []
    for worker in runtime_plan.ownership.workers:
        path = tmp_path / f"{worker.worker_id}.pt"
        result = save_worker_state_shard(
            path,
            graph=capture.canonical_graph,
            runtime_plan=runtime_plan,
            worker_id=worker.worker_id,
            gpu_index=worker.gpu_index,
            state_dict=model.state_dict(),
            job_id="job-test",
            plan_id="plan-test",
            training_step=20,
            metadata={
                "parameter_changed": True,
                "changed_parameter_count": 1,
                "checked_parameter_count": 1,
            },
        )
        assert result.bytes > 0
        assert result.parameter_count + result.buffer_count > 0
        shard_paths.append(path)

    consolidated = consolidate_worker_state_shards(
        shard_paths,
        tmp_path / "full-model-state-dict.pt",
        expected_state_keys=tuple(model.state_dict()),
    )

    assert set(consolidated["state_dict"]) == set(model.state_dict())
    assert consolidated["job_id"] == "job-test"
    assert consolidated["graph_fingerprint"] == capture.graph_fingerprint
    assert consolidated["plan_id"] == "plan-test"
    assert consolidated["training_step"] == 20
    assert consolidated["training_evidence"] == {
        "worker_parameter_changed": {"worker0": True, "worker1": True},
        "any_parameter_changed": True,
        "all_trainable_workers_parameter_changed": True,
    }
    assert all(shard["metadata"]["parameter_changed"] is True for shard in consolidated["shards"])
    for key, tensor in model.state_dict().items():
        assert torch.equal(consolidated["state_dict"][key], tensor)

    broken = torch.load(shard_paths[-1], map_location="cpu", weights_only=False)
    broken["job_id"] = "other-job"
    torch.save(broken, shard_paths[-1])
    with pytest.raises(ValueError, match="checkpoint metadata disagrees"):
        consolidate_worker_state_shards(
            shard_paths,
            tmp_path / "broken.pt",
            expected_state_keys=tuple(model.state_dict()),
        )


def _runtime_fixture() -> tuple[GenericGraphIR, LogicalPartitionPlan, PlacementPlan]:
    nodes = tuple(
        GraphNodeSpec(
            node_id=f"n{index}",
            op_kind="call_module",
            target=f"partition{index}",
            module_path=f"partition{index}",
            input_value_ids=("input",) if index == 0 else (f"v{index - 1}",),
            output_value_ids=(f"v{index}",),
            parameter_ids=(f"p{index}",),
        )
        for index in range(5)
    )
    nodes = (
        nodes[0],
        nodes[1],
        nodes[2],
        nodes[3],
        GraphNodeSpec(
            node_id="n4",
            op_kind="call_module",
            target="partition4",
            module_path="partition4",
            input_value_ids=("v3", "v0"),
            output_value_ids=("v4",),
            parameter_ids=("p4",),
        ),
    )
    graph = GenericGraphIR(
        nodes=nodes,
        values=(),
        edges=(
            GraphEdgeSpec("n0", "n1", "v0", edge_id="e0"),
            GraphEdgeSpec("n1", "n2", "v1", edge_id="e1"),
            GraphEdgeSpec("n2", "n3", "v2", edge_id="e2"),
            GraphEdgeSpec("n3", "n4", "v3", edge_id="e3"),
            GraphEdgeSpec("n0", "n4", "v0", edge_id="e4"),
        ),
        input_value_ids=("input",),
        output_value_ids=("v4",),
        parameter_owners={f"p{index}": f"n{index}" for index in range(5)},
        capture_backend="test",
        graph_fingerprint="fixture",
    )
    logical = LogicalPartitionPlan(
        graph_fingerprint="fixture",
        partitions=tuple(
            LogicalPartitionSpec(
                partition_id=f"P{index}",
                node_ids=(f"n{index}",),
                input_value_ids=nodes[index].input_value_ids,
                output_value_ids=nodes[index].output_value_ids,
                parameter_ids=(f"p{index}",),
                buffer_ids=(),
                estimated_compute=1,
                estimated_memory=1,
                boundary_edges=(),
            )
            for index in range(5)
        ),
    )
    placement = PlacementPlan(
        graph_fingerprint="fixture",
        selected_gpu_count=3,
        placements=(
            PlacementSpec("P0", "gpu0", "worker0", 0),
            PlacementSpec("P1", "gpu1", "worker1", 0),
            PlacementSpec("P2", "gpu2", "worker2", 0),
            PlacementSpec("P3", "gpu1", "worker1", 0),
            PlacementSpec("P4", "gpu0", "worker0", 0),
        ),
    )
    return graph, logical, placement


def _unique_parameters(modules: tuple[nn.Module, ...]) -> list[nn.Parameter]:
    seen: set[int] = set()
    result: list[nn.Parameter] = []
    for module in modules:
        for parameter in module.parameters():
            if id(parameter) in seen:
                continue
            seen.add(id(parameter))
            result.append(parameter)
    return result


def _assert_extracted_partitions_match_full(name: str, *, partition_count: int) -> None:
    model = build_zoo_model(name)
    reference = build_zoo_model(name)
    reference.load_state_dict(model.state_dict())
    args, kwargs = make_zoo_sample(name)
    if kwargs:
        raise AssertionError("partition extractor tests only use positional zoo samples")
    capture = FXGraphCaptureAdapter().capture(model, sample_args=args)
    logical = _logical_plan_for_extractor(capture.canonical_graph, partition_count)
    placement = PlacementPlan(
        graph_fingerprint=capture.graph_fingerprint,
        selected_gpu_count=2,
        placements=tuple(
            PlacementSpec(
                partition.partition_id,
                f"gpu{index % 2}",
                f"worker{index % 2}",
                0,
            )
            for index, partition in enumerate(logical.partitions)
        ),
    )
    runtime_plan = compile_runtime_plan(capture.canonical_graph, logical, placement)
    extracted = {
        partition.partition_id: extract_partition_graph(
            capture.canonical_graph,
            capture.backend_graph,
            partition,
        )
        for partition in logical.partitions
    }
    runtime = LocalDAGRuntime(
        runtime_plan,
        {
            partition.partition_id: RuntimePartition(partition, extracted[partition.partition_id])
            for partition in logical.partitions
        },
    )

    expected = reference(*tuple(item.detach().clone() for item in args))
    outputs, evidence = runtime.forward(
        {
            value_id: value.detach().clone()
            for value_id, value in zip(capture.canonical_graph.input_value_ids, args, strict=True)
        }
    )
    actual = tuple(outputs[value_id] for value_id in capture.canonical_graph.output_value_ids)
    actual_value = actual[0] if len(actual) == 1 else actual

    assert torch.allclose(actual_value, expected, atol=1e-5, rtol=1e-5)
    loss = (
        actual_value.sum()
        if isinstance(actual_value, torch.Tensor)
        else sum(value.sum() for value in actual_value)
    )
    loss.backward()
    optimizer = torch.optim.SGD(_unique_parameters((model,)), lr=0.01)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    evidence = runtime.optimizer_step({"local": optimizer}, evidence)

    assert evidence.optimizer_step_completed is True
    assert any(parameter.grad is not None for parameter in model.parameters())
    assert any(
        not torch.equal(old, parameter)
        for old, parameter in zip(before, model.parameters(), strict=True)
    )


def _logical_plan_for_extractor(
    graph: GenericGraphIR,
    partition_count: int,
) -> LogicalPartitionPlan:
    executable = tuple(
        node for node in graph.nodes if node.op_kind not in {"placeholder", "output"}
    )
    size = max(1, (len(executable) + partition_count - 1) // partition_count)
    chunks = tuple(executable[index : index + size] for index in range(0, len(executable), size))
    producer = {
        value.value_id: value.producer_node_id
        for value in graph.values
        if value.producer_node_id is not None
    }
    consumers: dict[str, set[str]] = {}
    for edge in graph.edges:
        consumers.setdefault(edge.value_id, set()).add(edge.target_node_id)
    partitions = []
    for index, chunk in enumerate(chunks):
        node_ids = {node.node_id for node in chunk}
        later_node_ids = {
            node.node_id
            for later_chunk in chunks[index + 1 :]
            for node in later_chunk
        }
        input_value_ids = tuple(
            sorted(
                {
                    value_id
                    for value in graph.values
                    for value_id in (value.value_id,)
                    if producer.get(value_id) not in node_ids
                    and consumers.get(value_id, set()) & node_ids
                }
            )
        )
        output_value_ids = tuple(
            sorted(
                {
                    value_id
                    for node in chunk
                    for value_id in node.output_value_ids
                    if value_id in graph.output_value_ids
                    or consumers.get(value_id, set()) & later_node_ids
                }
            )
        )
        partitions.append(
            LogicalPartitionSpec(
                partition_id=f"P{index}",
                node_ids=tuple(node.node_id for node in chunk),
                input_value_ids=input_value_ids,
                output_value_ids=output_value_ids,
                parameter_ids=tuple(sorted({pid for node in chunk for pid in node.parameter_ids})),
                buffer_ids=tuple(sorted({bid for node in chunk for bid in node.buffer_ids})),
                estimated_compute=1,
                estimated_memory=1,
                boundary_edges=(),
            )
        )
    return LogicalPartitionPlan(graph.graph_fingerprint, tuple(partitions))
