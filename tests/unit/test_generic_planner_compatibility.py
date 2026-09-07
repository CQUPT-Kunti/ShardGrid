from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shardgrid.planner.generic_graph import capture_generic_graph
from shardgrid.planner.memory import MemoryEstimationConfig, build_model_profile
from shardgrid.planner.partitioning import generate_partition_candidates
from shardgrid.planner.planning_contract import (
    GPUResourceSpec,
    PlanningConstraints,
    ResourceSnapshot,
    RuntimeCapabilities,
    build_logical_partition_plan,
    generate_placement_candidates,
    validate_final_plan,
)
from shardgrid.planner.requirements import FeasibilityStatus


FAMILIES = (
    "sequential",
    "residual",
    "cnn",
    "unet_like",
    "dense",
    "transformer",
    "attention",
    "encoder_decoder",
    "multi_branch",
    "shared_module",
)


def _fixtures() -> ModuleType:
    module_name = "generic_training_models"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    path = Path(__file__).resolve().parents[1] / "fixtures" / "generic_training_models.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load fixture module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("family", FAMILIES)
def test_cpu_planner_pipeline_supports_ordinary_pytorch_family(family: str) -> None:
    case = _fixtures().generic_training_case(family)
    kwargs = dict(case.kwargs or {})
    graph = capture_generic_graph(case.module.eval(), sample_args=case.args, sample_kwargs=kwargs)
    profile = build_model_profile(
        case.module,
        engine_id="pytorch_pipeline",
        model_name=family,
        sample_args=case.args,
        sample_kwargs=kwargs,
        memory_config=_memory_config(),
    )
    partition = generate_partition_candidates(
        profile,
        graph=graph,
        memory_config=_memory_config(),
        min_stage_count=2,
        max_stage_count=2,
    )
    logical = build_logical_partition_plan(graph, max_partitions=2)
    placement = generate_placement_candidates(
        graph,
        logical,
        _resources(len(logical.partitions)),
        PlanningConstraints(),
        RuntimeCapabilities(
            supports_multiple_partitions_per_device=True,
            supports_shared_parameter=True,
        ),
    )
    validation = validate_final_plan(graph, logical, placement[0])

    assert graph.nodes
    assert graph.states
    assert profile.state_memory
    assert partition.status == FeasibilityStatus.FEASIBLE
    assert partition.candidates
    assert _covered_nodes(logical) == {
        node.node_id for node in graph.nodes if node.op_kind != "output"
    }
    assert _owned_states(logical) == {state.canonical_state_id for state in graph.states}
    assert validation.valid, validation.diagnostics


def _memory_config() -> MemoryEstimationConfig:
    return MemoryEstimationConfig(
        optimizer_type="adamw",
        gradient_dtype="float32",
        optimizer_state_dtype="float32",
        runtime_overhead_bytes=1024,
        communication_buffer_bytes=2048,
        safety_headroom_bytes=4096,
        temporary_buffer_factor=0.25,
    )


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
            for index in range(max(count, 2))
        )
    )


def _covered_nodes(logical: Any) -> set[str]:
    return {node_id for partition in logical.partitions for node_id in partition.node_ids}


def _owned_states(logical: Any) -> set[str]:
    return {
        state_id
        for partition in logical.partitions
        for state_id in partition.owned_state_ids
    }
