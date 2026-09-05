from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from shardgrid.planner.generic_graph import (
    FXGraphCaptureAdapter,
    GenericGraphIR,
    GraphEdgeSpec,
    GraphNodeSpec,
    GraphValueSpec,
    ModelFactorySpec,
    capture_generic_graph,
)
from shardgrid.planner.planning_contract import (
    GPUResourceSpec,
    PlanningConstraints,
    ResourceSnapshot,
    RuntimeCapabilities,
    generate_logical_partition_candidates,
    generate_placement_candidates,
    plan,
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


def _weighted_graph(weights: list[int]):
    nodes = tuple(
        GraphNodeSpec(
            node_id=f"n{index:04d}",
            op_kind="call_module",
            target="linear",
            module_path=f"m{index}",
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
