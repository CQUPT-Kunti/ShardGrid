from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import torch
from torch import nn

from shardgrid.planner.generic_graph import (
    FXGraphCaptureAdapter,
    GenericGraphIR,
    GraphEdgeSpec,
    GraphNodeSpec,
    StateObjectSpec,
    GraphValueSpec,
    ModelFactorySpec,
    capture_generic_graph,
)
from shardgrid.planner.planning_contract import (
    GPUResourceSpec,
    LogicalPartitionPlan,
    LogicalPartitionSpec,
    PlacementSpec,
    PlanningConstraints,
    ResourceSnapshot,
    RuntimeCapabilities,
    build_logical_partition_plan,
    final_plan_fingerprint_inputs,
    generate_logical_partition_candidates,
    generate_placement_candidates,
    plan,
    validate_final_plan,
)


class RenamedModelA(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(4, 4)
        self.b = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.b(torch.relu(self.a(x)))


class RenamedModelB(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(4, 4)
        self.b = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.b(torch.relu(self.a(x)))


class ChangedGraphModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(4, 4)
        self.extra = nn.Linear(4, 4)
        self.b = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.b(torch.relu(self.extra(torch.relu(self.a(x)))))


class SharedParameterModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shared = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shared(self.shared(x))


def _resources(count: int) -> ResourceSnapshot:
    return ResourceSnapshot(
        tuple(
            GPUResourceSpec(
                gpu_id=f"gpu-{index}",
                worker_id=f"worker-{index}",
                gpu_index=0,
                total_memory_bytes=1_000_000_000,
                free_memory_bytes=1_000_000_000,
            )
            for index in range(count)
        )
    )


def test_capture_adapter_returns_canonical_schema_and_metadata() -> None:
    result = FXGraphCaptureAdapter().capture(
        RenamedModelA(),
        sample_args=(torch.ones(1, 4),),
    )

    assert result.canonical_graph.schema_version == "shardgrid.canonical_graph.v1"
    assert result.graph_fingerprint == result.canonical_graph.graph_fingerprint
    assert result.metadata["control_plane_full_real_model_materialized"] is True
    assert result.backend_graph is not None


def test_capture_factory_can_use_meta_parameters() -> None:
    def factory() -> nn.Module:
        return RenamedModelA()

    def sample(device: str) -> tuple[tuple[torch.Tensor, ...], dict[str, object]]:
        return (torch.empty(1, 4, device=device),), {}

    result = FXGraphCaptureAdapter().capture_factory(
        ModelFactorySpec(factory, sample_input_builder=sample)
    )

    assert result.metadata["control_plane_parameter_real_storage_bytes"] == 0
    assert result.metadata["control_plane_full_real_model_materialized"] is False


def test_graph_fingerprint_ignores_class_rename() -> None:
    sample = torch.ones(1, 4)

    assert capture_generic_graph(RenamedModelA(), sample_args=(sample,)).graph_fingerprint == (
        capture_generic_graph(RenamedModelB(), sample_args=(sample,)).graph_fingerprint
    )


def test_graph_fingerprint_changes_when_graph_changes() -> None:
    sample = torch.ones(1, 4)

    assert capture_generic_graph(RenamedModelA(), sample_args=(sample,)).graph_fingerprint != (
        capture_generic_graph(ChangedGraphModel(), sample_args=(sample,)).graph_fingerprint
    )


def test_logical_partition_and_placement_are_separate_artifacts() -> None:
    graph = _weighted_graph([3] * 12)
    result = plan(
        graph,
        ResourceSnapshot(
            (
                GPUResourceSpec("gpu-large", "worker-a", 0, 100, 24),
                GPUResourceSpec("gpu-mid", "worker-b", 0, 100, 12),
                GPUResourceSpec("gpu-small", "worker-c", 0, 100, 6),
            )
        ),
        PlanningConstraints(max_partition_candidates=8),
        RuntimeCapabilities(supports_multiple_partitions_per_device=True),
    )

    assert result.logical_partition_plan is not None
    assert result.placement_plan is not None
    assert any(
        len(candidate.logical_partition_plan.partitions)
        > candidate.placement_plan.selected_gpu_count
        for candidate in result.plan_candidates
    )
    assert (
        result.logical_partition_plan.to_dict()["schema_version"]
        == "shardgrid.dag_planning.v1"
    )
    assert result.placement_plan.to_dict()["schema_version"] == "shardgrid.dag_planning.v1"
    assert result.estimated_cost["total_graph_edge_bytes"] > 0
    assert result.search_diagnostics is not None
    assert result.search_diagnostics["evaluated_plans"] > 1


def test_shared_parameter_is_rejected_by_capabilities() -> None:
    graph = capture_generic_graph(
        SharedParameterModel(),
        sample_args=(torch.ones(1, 4),),
    )
    result = plan(
        graph,
        _resources(2),
        PlanningConstraints(max_logical_partitions=2),
        RuntimeCapabilities(supports_shared_parameter=False),
    )

    assert graph.shared_parameter_ids
    assert result.placement_plan is None
    assert "AUTO_PARTITION_UNSUPPORTED_SHARED_PARAMETER" in result.diagnostics


def test_generic_planner_core_has_no_model_specific_names() -> None:
    root = Path(__file__).resolve().parents[2]
    files = (
        root / "src/shardgrid/planner/generic_graph.py",
        root / "src/shardgrid/planner/planning_contract.py",
        root / "src/shardgrid/planner/partitioning.py",
        root / "src/shardgrid/planner/placement.py",
    )
    banned = (
        "examples.models",
        "unet",
        "vit",
        "densenet",
        "resnet",
        "large_residual_transformer",
        "transformer",
    )

    for path in files:
        source = path.read_text(encoding="utf-8").lower()
        assert not any(name in source for name in banned), path


def test_partition_candidates_can_exceed_eight_without_fixed_cap() -> None:
    graph = _weighted_graph([100] * 16)
    resources = ResourceSnapshot(
        tuple(
            GPUResourceSpec(
                gpu_id=f"gpu-{index}",
                worker_id=f"worker-{index}",
                gpu_index=0,
                total_memory_bytes=100,
                free_memory_bytes=100,
            )
            for index in range(16)
        )
    )

    candidates = generate_logical_partition_candidates(
        graph,
        resources,
        PlanningConstraints(max_partition_candidates=32),
    )

    assert max(len(candidate.partitions) for candidate in candidates) > 8


def test_plan_search_respects_candidate_budgets() -> None:
    graph = _weighted_graph([10] * 20)
    constraints = PlanningConstraints(
        max_partition_candidates=3,
        max_placement_candidates_per_partition=2,
        max_total_plan_candidates=4,
        top_k_plans=2,
        beam_width=4,
    )

    result = plan(
        graph,
        ResourceSnapshot(
            tuple(
                GPUResourceSpec(f"gpu-{index}", f"worker-{index}", 0, 100, 100)
                for index in range(4)
            )
        ),
        constraints,
        RuntimeCapabilities(supports_multiple_partitions_per_device=True),
    )

    assert result.search_diagnostics is not None
    assert result.search_diagnostics["partition_candidates"] <= 3
    assert result.search_diagnostics["evaluated_plans"] <= 4
    assert len(result.plan_candidates) <= 2


def test_placement_beam_generates_multiple_bounded_candidates() -> None:
    graph = _weighted_graph([10] * 8)
    logical = generate_logical_partition_candidates(
        graph,
        ResourceSnapshot(
            tuple(
                GPUResourceSpec(f"gpu-{index}", f"worker-{index}", 0, 100, 100)
                for index in range(3)
            )
        ),
        PlanningConstraints(max_partition_candidates=1),
    )[0]

    placements = generate_placement_candidates(
        graph,
        logical,
        ResourceSnapshot(
            tuple(
                GPUResourceSpec(f"gpu-{index}", f"worker-{index}", 0, 100, 100)
                for index in range(3)
            )
        ),
        PlanningConstraints(max_placement_candidates_per_partition=2, beam_width=3),
        RuntimeCapabilities(supports_multiple_partitions_per_device=True),
    )

    assert 1 <= len(placements) <= 2


def test_free_memory_drives_uneven_partitioning() -> None:
    graph = _weighted_graph([4] * 12)
    result = plan(
        graph,
        ResourceSnapshot(
            (
                GPUResourceSpec("gpu-large", "worker-a", 0, 100, 24),
                GPUResourceSpec("gpu-mid", "worker-b", 0, 100, 12),
                GPUResourceSpec("gpu-small", "worker-c", 0, 100, 6),
            )
        ),
        PlanningConstraints(max_partition_candidates=8),
        RuntimeCapabilities(supports_multiple_partitions_per_device=True),
    )

    assert result.logical_partition_plan is not None
    sizes = [partition.estimated_memory for partition in result.logical_partition_plan.partitions]
    assert max(sizes) > min(sizes)


def test_fresh_free_memory_changes_selected_plan() -> None:
    graph = _weighted_graph([5] * 8)
    capabilities = RuntimeCapabilities(supports_multiple_partitions_per_device=True)
    constraints = PlanningConstraints(max_partition_candidates=8)

    before = plan(
        graph,
        ResourceSnapshot(
            (
                GPUResourceSpec("gpu-a", "worker-a", 0, 100, 40),
                GPUResourceSpec("gpu-b", "worker-b", 0, 100, 8),
            )
        ),
        constraints,
        capabilities,
    )
    after = plan(
        graph,
        ResourceSnapshot(
            (
                GPUResourceSpec("gpu-a", "worker-a", 0, 100, 20),
                GPUResourceSpec("gpu-b", "worker-b", 0, 100, 28),
            )
        ),
        constraints,
        capabilities,
    )

    assert before.placement_plan is not None
    assert after.placement_plan is not None
    assert before.placement_plan.to_dict() != after.placement_plan.to_dict()


def test_logical_partition_old_constructor_remains_compatible() -> None:
    partition = LogicalPartitionSpec(
        "P0",
        ("n0000",),
        ("v-input",),
        ("v-output",),
        ("p0000",),
        ("b0000",),
        7,
        11,
        ("e0000",),
    )

    assert partition.node_ids == ("n0000",)
    assert partition.parameter_ids == ("p0000",)
    assert partition.buffer_ids == ("b0000",)
    assert partition.owned_state_ids == ()
    assert partition.read_only_state_ids == ()
    assert partition.validation_evidence is None


def test_logical_partition_records_nodes_values_state_and_evidence() -> None:
    logical = build_logical_partition_plan(
        _shared_state_graph(),
        max_partitions=2,
    )
    first, second = logical.partitions

    assert first.node_ids == ("n0000",)
    assert first.output_value_ids == ("v0000",)
    assert first.owned_state_ids == ("p0000",)
    assert first.read_only_state_ids == ()
    assert first.estimated_transfer_bytes == 10
    assert first.validation_evidence == {
        "node_count": 1,
        "input_value_count": 0,
        "output_value_count": 1,
        "boundary_edge_count": 1,
    }
    assert second.node_ids == ("n0001",)
    assert second.input_value_ids == ("v0000",)
    assert second.owned_state_ids == ()
    assert second.read_only_state_ids == ("p0000",)


def test_logical_partition_serialization_round_trip_keeps_explicit_fields() -> None:
    logical = build_logical_partition_plan(_shared_state_graph(), max_partitions=2)

    restored = LogicalPartitionPlan.from_dict(logical.to_dict())

    assert restored == logical
    assert restored.to_dict()["partitions"][1]["read_only_state_ids"] == ["p0000"]


def test_logical_partition_state_coverage_does_not_depend_on_module_order() -> None:
    graph = capture_generic_graph(
        SharedParameterModel(),
        sample_args=(torch.ones(1, 4),),
    )
    logical = build_logical_partition_plan(graph, max_partitions=2)
    first, second = logical.partitions

    assert [node.module_path for node in graph.nodes if node.module_path] == [
        "shared",
        "shared",
    ]
    assert first.owned_state_ids == ("p0000", "p0001")
    assert second.read_only_state_ids == ("p0000", "p0001")
    assert set(second.parameter_ids) == {"p0000", "p0001"}


def test_validate_final_plan_accepts_legal_plan_and_stable_fingerprint_inputs() -> None:
    graph, logical, placement = _valid_final_plan()

    result = validate_final_plan(graph, logical, placement)
    fingerprint_inputs = final_plan_fingerprint_inputs(graph, logical, placement)

    assert result.valid
    assert result.diagnostics == ()
    assert fingerprint_inputs == final_plan_fingerprint_inputs(graph, logical, placement)
    assert json.dumps(fingerprint_inputs, sort_keys=True)


def test_validate_final_plan_rejects_graph_logical_and_placement_mismatch() -> None:
    graph, logical, placement = _valid_final_plan()

    logical_result = validate_final_plan(
        graph,
        replace(logical, graph_fingerprint="different-logical"),
        placement,
    )
    placement_result = validate_final_plan(
        graph,
        logical,
        replace(placement, graph_fingerprint="different-placement"),
    )

    assert logical_result.diagnostics[0] == "PLAN_VALIDATION_FAILURE"
    assert "graph_logical_fingerprint_mismatch" in logical_result.diagnostics
    assert placement_result.diagnostics[0] == "PLAN_VALIDATION_FAILURE"
    assert "graph_placement_fingerprint_mismatch" in placement_result.diagnostics


def test_validate_final_plan_rejects_missing_duplicate_and_unknown_placement() -> None:
    graph, logical, placement = _valid_final_plan()
    first = placement.placements[0]
    missing = replace(placement, placements=placement.placements[:1])
    duplicate = replace(placement, placements=placement.placements + (first,))
    unknown = replace(
        placement,
        placements=placement.placements
        + (PlacementSpec("PX", first.gpu_id, first.worker_id, first.gpu_index),),
    )

    assert "missing_placement_partition" in validate_final_plan(
        graph,
        logical,
        missing,
    ).diagnostics
    assert "duplicate_placement_partition" in validate_final_plan(
        graph,
        logical,
        duplicate,
    ).diagnostics
    assert "unknown_placement_partition" in validate_final_plan(
        graph,
        logical,
        unknown,
    ).diagnostics


def test_validate_final_plan_rejects_node_state_and_boundary_invariants() -> None:
    graph, logical, placement = _valid_final_plan()
    first, second = logical.partitions
    duplicate_node = replace(
        logical,
        partitions=(replace(first, node_ids=first.node_ids + second.node_ids[:1]), second),
    )
    missing_state = replace(
        logical,
        partitions=(
            replace(first, owned_state_ids=()),
            replace(second, read_only_state_ids=()),
        ),
    )
    missing_boundary = replace(
        logical,
        partitions=(replace(first, output_value_ids=()), second),
    )

    assert "duplicate_partition_node" in validate_final_plan(
        graph,
        duplicate_node,
        placement,
    ).diagnostics
    assert "missing_state_owner" in validate_final_plan(
        graph,
        missing_state,
        placement,
    ).diagnostics
    assert "missing_boundary_value" in validate_final_plan(
        graph,
        missing_boundary,
        placement,
    ).diagnostics


def test_resource_failure_is_not_plan_validation_failure() -> None:
    result = plan(
        _weighted_graph([10] * 4),
        ResourceSnapshot(()),
        PlanningConstraints(),
        RuntimeCapabilities(),
    )

    assert "NO_AVAILABLE_GPUS" in result.diagnostics
    assert "PLAN_VALIDATION_FAILURE" not in result.diagnostics


def _valid_final_plan():
    graph = _shared_state_graph()
    logical = build_logical_partition_plan(graph, max_partitions=2)
    placement = generate_placement_candidates(
        graph,
        logical,
        _resources(2),
        PlanningConstraints(),
        RuntimeCapabilities(),
    )[0]
    return graph, logical, placement


def _weighted_graph(weights: list[int]):
    nodes = tuple(
        GraphNodeSpec(
            node_id=f"n{index:04d}",
            op_kind="call_module",
            target="linear",
            module_path=f"m{index}",
            input_value_ids=() if index == 0 else (f"v{index - 1:04d}",),
            output_value_ids=(f"v{index:04d}",),
            parameter_ids=(f"p{index:04d}",),
            estimated_peak_memory_contribution=weight,
        )
        for index, weight in enumerate(weights)
    )
    edges = tuple(
        GraphEdgeSpec(
            source_node_id=f"n{index:04d}",
            target_node_id=f"n{index + 1:04d}",
            value_id=f"v{index:04d}",
            edge_id=f"e{index:04d}",
            forward_transfer_bytes=weights[index],
            backward_transfer_bytes=weights[index],
            communication_weight=weights[index] * 2,
        )
        for index in range(len(weights) - 1)
    )
    values = tuple(
        GraphValueSpec(
            value_id=f"v{index:04d}",
            producer_node_id=f"n{index:04d}",
            estimated_bytes=weight,
            dtype="float32",
            shape=(1, weight // 4 or 1),
        )
        for index, weight in enumerate(weights)
    )
    return GenericGraphIR(
        nodes=nodes,
        values=values,
        edges=edges,
        input_value_ids=("v0000",),
        output_value_ids=(f"v{len(weights) - 1:04d}",),
        parameter_owners={
            f"p{index:04d}": f"n{index:04d}" for index in range(len(weights))
        },
        capture_backend="test",
        graph_fingerprint="",
    )


def _shared_state_graph() -> GenericGraphIR:
    nodes = (
        GraphNodeSpec(
            node_id="n0000",
            op_kind="call_function",
            target="linear",
            module_path=None,
            output_value_ids=("v0000",),
            parameter_ids=("p0000",),
            estimated_peak_memory_contribution=5,
        ),
        GraphNodeSpec(
            node_id="n0001",
            op_kind="call_function",
            target="linear",
            module_path=None,
            input_value_ids=("v0000",),
            output_value_ids=("v0001",),
            parameter_ids=("p0000",),
            estimated_peak_memory_contribution=5,
        ),
    )
    return GenericGraphIR(
        nodes=nodes,
        values=(
            GraphValueSpec("v0000", "n0000", ("n0001",), estimated_bytes=5),
            GraphValueSpec("v0001", "n0001", (), estimated_bytes=5),
        ),
        edges=(
            GraphEdgeSpec(
                "n0000",
                "n0001",
                "v0000",
                edge_id="e0000",
                communication_weight=10,
            ),
        ),
        input_value_ids=("v-input",),
        output_value_ids=("v0001",),
        parameter_owners={"p0000": "n0000"},
        capture_backend="test",
        states=(
            StateObjectSpec(
                canonical_state_id="p0000",
                kind="parameter",
                state_dict_key="weight",
                owner_node_ids=("n0000",),
                use_node_ids=("n0000", "n0001"),
            ),
        ),
    )
