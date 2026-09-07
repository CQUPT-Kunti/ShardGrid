from __future__ import annotations

import importlib.util
import math
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.models.minimal_transformer import (
    MinimalTransformerConfig,
    build_minimal_transformer,
)
from examples.models.partition_stress_model import (
    PartitionStressConfig,
    build_partition_stress_model,
    make_training_batch,
    train_step,
)
from torch import nn
from torch.fx import wrap

from shardgrid.common.enums import FailureCode
from shardgrid.planner.generic_graph import capture_generic_graph
from shardgrid.planner.memory import MemoryEstimationConfig, build_model_profile
from shardgrid.planner.partitioning import (
    build_partition_profile,
    discover_partition_support,
    generate_partition_candidates,
    validate_partition_candidate,
)
from shardgrid.planner.requirements import FeasibilityStatus


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


def _generic_fixture_module() -> ModuleType:
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


def _ordinary_case(name: str) -> Any:
    return _generic_fixture_module().generic_training_case(name)


def _profile_case(case: Any, name: str):
    return build_model_profile(
        case.module,
        engine_id="pytorch_pipeline",
        model_name=name,
        sample_args=case.args,
        sample_kwargs=dict(case.kwargs or {}),
        memory_config=_memory_config(),
    )


def _ordinary_profile_and_support(name: str):
    case = _ordinary_case(name)
    profile = _profile_case(case, name)
    support = discover_partition_support(
        case.module,
        profile,
        sample_args=case.args,
        sample_kwargs=dict(case.kwargs or {}),
    )
    return case, profile, support


def test_partition_stress_model_forward_is_deterministic() -> None:
    model_a = build_partition_stress_model(seed=42).eval()
    model_b = build_partition_stress_model(seed=42).eval()
    inputs_a, targets_a = make_training_batch(seed=7, step=0)
    inputs_b, targets_b = make_training_batch(seed=7, step=0)

    output_a = model_a(inputs_a)
    output_b = model_b(inputs_b)

    assert torch.equal(inputs_a, inputs_b)
    assert torch.equal(targets_a, targets_b)
    assert torch.allclose(output_a, output_b)


def test_partition_stress_model_does_not_generate_training_data_in_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = build_partition_stress_model(seed=42).eval()
    inputs, _targets = make_training_batch(seed=11, step=0)

    def fail(*_args: object, **_kwargs: object) -> torch.Tensor:
        raise AssertionError("random generation inside forward is forbidden")

    monkeypatch.setattr(torch, "randn", fail)
    monkeypatch.setattr(torch, "rand", fail)
    monkeypatch.setattr(torch, "randint", fail)

    with torch.inference_mode():
        output = model(inputs)

    assert output.shape[0] == inputs.shape[0]


def test_partition_stress_model_short_training_smoke() -> None:
    model = build_partition_stress_model(seed=42).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    losses: list[float] = []

    for step in range(5):
        loss = train_step(model, make_training_batch(seed=19, step=step), optimizer)
        losses.append(float(loss.detach().item()))

    assert len(losses) == 5
    assert all(math.isfinite(loss) for loss in losses)
    assert any(
        not torch.equal(original, current)
        for original, current in zip(before, model.parameters())
    )


def test_minimal_transformer_generates_automatic_candidates_without_stage_fixtures() -> None:
    model = build_minimal_transformer(
        MinimalTransformerConfig(
            vocab_size=128,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_seq_length=16,
        ),
        seed=42,
    )
    input_ids = torch.randint(0, 128, (2, 15), dtype=torch.long)
    profile = build_model_profile(
        model,
        engine_id="pytorch_pipeline",
        model_name="minimal-transformer",
        sample_args=(input_ids,),
        memory_config=_memory_config(),
        required_backends=("nccl", "gloo"),
    )
    support = discover_partition_support(model, profile, sample_args=(input_ids,))

    result_a = build_partition_profile(
        model,
        profile,
        sample_args=(input_ids,),
        memory_config=_memory_config(),
        original_engine_plan_ref="/tmp/engine-plan.json",
    )
    result_b = build_partition_profile(
        model,
        profile,
        sample_args=(input_ids,),
        memory_config=_memory_config(),
        original_engine_plan_ref="/tmp/engine-plan.json",
    )

    assert support.status.value == "supported"
    assert any(
        "embed->blocks.0.ln1" in boundary.forward_dependencies
        for boundary in support.boundaries
    )
    assert all(
        "embed->pos" not in boundary.forward_dependencies
        for boundary in support.boundaries
    )
    assert result_a.status == FeasibilityStatus.FEASIBLE
    assert len(result_a.candidates) >= 1
    assert [candidate.candidate_id for candidate in result_a.candidates] == [
        candidate.candidate_id for candidate in result_b.candidates
    ]
    assert all(
        candidate.original_engine_plan_ref == "/tmp/engine-plan.json"
        for candidate in result_a.candidates
    )


def test_residual_skip_model_generates_candidates_and_preserves_skip_dependencies() -> None:
    model = build_partition_stress_model(PartitionStressConfig(), seed=42)
    inputs, _targets = make_training_batch(seed=23, step=0)
    profile = build_model_profile(
        model,
        engine_id="pytorch_pipeline",
        model_name="partition-stress-model",
        sample_args=(inputs,),
        memory_config=_memory_config(),
    )

    result = build_partition_profile(
        model,
        profile,
        sample_args=(inputs,),
        memory_config=_memory_config(),
        max_stage_count=3,
    )

    assert result.status == FeasibilityStatus.FEASIBLE
    feasible = [candidate for candidate in result.candidates if not candidate.rejection_reasons]
    assert feasible
    assert {candidate.stage_count for candidate in feasible} >= {2, 3}
    assert any(
        abs(int(edge.target_stage_id[-1]) - int(edge.source_stage_id[-1])) > 1
        for candidate in feasible
        for edge in candidate.communication_edges
    )


def test_feasible_candidates_cover_each_parameter_exactly_once() -> None:
    model = build_partition_stress_model(seed=42)
    inputs, _targets = make_training_batch(seed=31, step=0)
    profile = build_model_profile(
        model,
        engine_id="pytorch_pipeline",
        model_name="partition-stress-model",
        sample_args=(inputs,),
        memory_config=_memory_config(),
    )

    result = build_partition_profile(
        model,
        profile,
        sample_args=(inputs,),
        memory_config=_memory_config(),
        max_stage_count=3,
    )
    expected = sorted(name for name, _parameter in model.named_parameters())

    for candidate in result.candidates:
        if candidate.hard_constraint_status is not FeasibilityStatus.FEASIBLE:
            continue
        actual = sorted(
            name for stage in candidate.stages for name in stage.parameter_names_or_ranges
        )
        assert actual == expected


def test_partition_candidates_respect_usable_memory_margin() -> None:
    model = build_partition_stress_model(seed=42)
    inputs, _targets = make_training_batch(seed=41, step=0)
    profile = build_model_profile(
        model,
        engine_id="pytorch_pipeline",
        model_name="partition-stress-model",
        sample_args=(inputs,),
        memory_config=_memory_config(),
    )

    result = build_partition_profile(
        model,
        profile,
        sample_args=(inputs,),
        memory_config=_memory_config(),
        usable_memory_bytes=(120_000, 120_000, 120_000),
        max_stage_count=3,
    )

    by_stage_count = {candidate.stage_count: candidate for candidate in result.candidates}

    assert result.status == FeasibilityStatus.FEASIBLE
    assert by_stage_count[2].hard_constraint_status == FeasibilityStatus.INFEASIBLE
    assert any(
        "usable GPU memory after headroom" in reason
        for reason in by_stage_count[2].rejection_reasons
    )
    assert by_stage_count[3].hard_constraint_status == FeasibilityStatus.FEASIBLE
    assert all(
        stage.estimated_peak_training_memory.planner_required_bytes <= 120_000
        for stage in by_stage_count[3].stages
    )


def test_equal_capacity_partition_avoids_extreme_imbalance() -> None:
    model = build_partition_stress_model(seed=42)
    inputs, _targets = make_training_batch(seed=43, step=0)
    profile = build_model_profile(
        model,
        engine_id="pytorch_pipeline",
        model_name="partition-stress-model",
        sample_args=(inputs,),
        memory_config=_memory_config(),
    )

    result = build_partition_profile(
        model,
        profile,
        sample_args=(inputs,),
        memory_config=_memory_config(),
        usable_memory_bytes=(400_000, 400_000),
        max_stage_count=2,
    )
    candidate = next(
        candidate for candidate in result.candidates if candidate.stage_count == 2
    )

    assert candidate.hard_constraint_status == FeasibilityStatus.FEASIBLE
    assert 1 <= candidate.stages[0].stop_index < candidate.stages[1].stop_index
    stage_bytes = [
        stage.estimated_peak_training_memory.planner_required_bytes
        for stage in candidate.stages
    ]
    assert max(stage_bytes) - min(stage_bytes) < 80_000


def test_ordinary_unet_functional_execution_nodes_are_not_partition_slots() -> None:
    case, profile, support = _ordinary_profile_and_support("unet_like")
    graph = capture_generic_graph(
        case.module.eval(),
        sample_args=case.args,
        sample_kwargs=dict(case.kwargs or {}),
    )
    functional_targets = {
        node.target
        for node in graph.nodes
        if node.module_path is None and node.op_kind == "call_function"
    }
    profile_paths = {module.module_path for module in profile.modules}

    result = generate_partition_candidates(
        profile,
        partition_support=support,
        memory_config=_memory_config(),
        max_stage_count=3,
    )

    assert {"enc1", "enc2", "dec1", "out"} == profile_paths
    assert any("avg_pool2d" in target for target in functional_targets)
    assert any("interpolate" in target for target in functional_targets)
    assert any("cat" in target for target in functional_targets)
    assert result.status == FeasibilityStatus.FEASIBLE
    assert all(
        path in profile_paths
        for candidate in result.candidates
        for stage in candidate.stages
        for path in stage.module_paths
    )
    assert all(
        "cat" not in path and "interpolate" not in path and "avg_pool2d" not in path
        for candidate in result.candidates
        for stage in candidate.stages
        for path in stage.module_paths
    )


def test_ordinary_transformer_state_owner_without_execution_node_is_unsupported() -> None:
    case, profile, support = _ordinary_profile_and_support("transformer")
    graph = capture_generic_graph(
        case.module.eval(),
        sample_args=case.args,
        sample_kwargs=dict(case.kwargs or {}),
    )
    profile_paths = {module.module_path for module in profile.modules}
    executed_paths = {node.module_path for node in graph.nodes if node.module_path}

    result = generate_partition_candidates(
        profile,
        partition_support=support,
        memory_config=_memory_config(),
    )

    assert "attn.out_proj" in profile_paths
    assert "attn.out_proj" not in executed_paths
    assert any(
        "out_proj" in parameter
        for module in profile.modules
        if module.module_path == "attn.out_proj"
        for parameter in module.parameter_names
    )
    assert support.status.value == "unsupported"
    assert result.status == FeasibilityStatus.UNSUPPORTED
    assert any("attn.out_proj" in reason for reason in support.reasons)


def test_ordinary_shared_module_is_collapsed_to_one_partition_module_slot() -> None:
    case, profile, support = _ordinary_profile_and_support("shared_module")
    graph = capture_generic_graph(
        case.module.eval(),
        sample_args=case.args,
        sample_kwargs=dict(case.kwargs or {}),
    )
    shared_executions = [
        node
        for node in graph.nodes
        if node.op_kind == "call_module" and node.module_path == "shared"
    ]

    result = generate_partition_candidates(
        profile,
        partition_support=support,
        memory_config=_memory_config(),
    )

    assert len(shared_executions) == 2
    assert [module.module_path for module in profile.modules].count("shared") == 1
    assert support.status.value == "supported"
    assert result.status == FeasibilityStatus.FEASIBLE
    assert all(
        [path for stage in candidate.stages for path in stage.module_paths].count("shared")
        == 1
        for candidate in result.candidates
    )


def test_ordinary_dense_multi_consumer_values_are_reduced_to_module_edges() -> None:
    case, profile, support = _ordinary_profile_and_support("dense")
    graph = capture_generic_graph(
        case.module.eval(),
        sample_args=case.args,
        sample_kwargs=dict(case.kwargs or {}),
    )
    multi_consumer_values = [
        value for value in graph.values if len(set(value.consumer_node_ids)) > 1
    ]

    result = generate_partition_candidates(
        profile,
        partition_support=support,
        memory_config=_memory_config(),
        min_stage_count=3,
        max_stage_count=3,
    )
    candidate = result.candidates[0]
    edge_pairs = {
        (edge.source_module_id, edge.target_module_id)
        for edge in candidate.communication_edges
    }

    assert result.status == FeasibilityStatus.FEASIBLE
    assert any(len(value.consumer_node_ids) >= 2 for value in multi_consumer_values)
    assert any(
        "input->head" in boundary.forward_dependencies for boundary in support.boundaries
    )
    assert any(
        "layer1->head" in boundary.forward_dependencies for boundary in support.boundaries
    )
    assert [stage.module_paths for stage in candidate.stages] == [
        ("input", "layer1"),
        ("layer2",),
        ("head",),
    ]
    assert ("m0000", "m0003") in edge_pairs
    assert ("m0001", "m0003") in edge_pairs


def test_graph_partitioning_residual_preserves_skip_boundary_values() -> None:
    case = _ordinary_case("residual")
    graph = capture_generic_graph(case.module.eval(), sample_args=case.args)
    profile = _profile_case(case, "residual")

    result = generate_partition_candidates(
        profile,
        graph=graph,
        memory_config=_memory_config(),
        min_stage_count=2,
        max_stage_count=2,
    )
    candidate = result.candidates[0]
    stage_paths = [path for stage in candidate.stages for path in stage.module_paths]

    assert result.status == FeasibilityStatus.FEASIBLE
    assert any("add" in path for path in stage_paths)
    assert any(
        edge.source_module_id == "n0001" and edge.target_module_id == "n0005"
        for edge in candidate.communication_edges
    )
    assert "v0001" in candidate.stages[0].output_value_ids
    assert "v0001" in candidate.stages[1].input_value_ids


def test_graph_partitioning_multibranch_uses_execution_order_not_registration_order() -> None:
    case = _ordinary_case("multi_branch")
    graph = capture_generic_graph(case.module.eval(), sample_args=case.args)
    profile = _profile_case(case, "multi_branch")

    result = generate_partition_candidates(
        profile,
        graph=graph,
        memory_config=_memory_config(),
        min_stage_count=2,
        max_stage_count=2,
    )
    candidate_paths = [
        path
        for stage in result.candidates[0].stages
        for path in stage.module_paths
    ]
    profile_paths = [module.module_path for module in profile.modules]

    assert profile_paths.index("right") < profile_paths.index("gate")
    assert candidate_paths.index("gate") < candidate_paths.index("right")
    assert any("cat" in path for path in candidate_paths)
    assert result.candidates[0].stages[0].node_ids[0] == "n0000"


def test_graph_partitioning_attention_and_dense_keep_value_dependencies() -> None:
    transformer = _ordinary_case("transformer")
    dense = _ordinary_case("dense")
    transformer_graph = capture_generic_graph(
        transformer.module.eval(),
        sample_args=transformer.args,
    )
    dense_graph = capture_generic_graph(dense.module.eval(), sample_args=dense.args)
    transformer_result = generate_partition_candidates(
        _profile_case(transformer, "transformer"),
        graph=transformer_graph,
        memory_config=_memory_config(),
        min_stage_count=3,
        max_stage_count=3,
    )
    dense_result = generate_partition_candidates(
        _profile_case(dense, "dense"),
        graph=dense_graph,
        memory_config=_memory_config(),
        min_stage_count=3,
        max_stage_count=3,
    )

    assert transformer_result.status == FeasibilityStatus.FEASIBLE
    attention_value_consumers = {
        edge.target_module_id
        for edge in transformer_result.candidates[0].communication_edges
        if edge.activation and edge.activation[0].name == "v0007"
    }
    assert len(attention_value_consumers) >= 2
    assert dense_result.status == FeasibilityStatus.FEASIBLE
    assert any(
        edge.activation and edge.activation[0].name == "v0002"
        for edge in dense_result.candidates[0].communication_edges
    )
    assert "v0002" in dense_result.candidates[0].stages[-1].input_value_ids


def test_graph_partitioning_shared_module_state_is_owned_once_and_read_only_later() -> None:
    case = _ordinary_case("shared_module")
    graph = capture_generic_graph(case.module.eval(), sample_args=case.args)
    profile = _profile_case(case, "shared_module")

    result = generate_partition_candidates(
        profile,
        graph=graph,
        memory_config=_memory_config(),
        min_stage_count=3,
        max_stage_count=3,
    )
    stages = result.candidates[0].stages

    assert result.status == FeasibilityStatus.FEASIBLE
    assert stages[0].owned_state_ids == ("p0000", "p0001")
    assert stages[1].read_only_state_ids == ("p0000", "p0001")
    assert stages[2].owned_state_ids == ("p0002", "p0003")
    assert not (set(stages[1].owned_state_ids) & set(stages[1].read_only_state_ids))


def _validation_fixture(name: str, stage_count: int = 2):
    case = _ordinary_case(name)
    graph = capture_generic_graph(case.module.eval(), sample_args=case.args)
    profile = _profile_case(case, name)
    support = discover_partition_support(
        case.module,
        profile,
        sample_args=case.args,
        sample_kwargs=dict(case.kwargs or {}),
    )
    result = generate_partition_candidates(
        profile,
        partition_support=support,
        graph=graph,
        memory_config=_memory_config(),
        min_stage_count=stage_count,
        max_stage_count=stage_count,
    )
    assert result.candidates
    return profile, support, graph, result.candidates[0]


def _validation_codes(result) -> set[str]:
    return {violation.code for violation in result.violations}


def test_validate_graph_candidate_accepts_legal_candidate() -> None:
    profile, support, graph, candidate = _validation_fixture("residual")

    result = validate_partition_candidate(profile, candidate, support, graph=graph)

    assert result.valid
    assert result.status == FeasibilityStatus.FEASIBLE


def test_validate_graph_candidate_rejects_duplicate_execution_node() -> None:
    profile, support, graph, candidate = _validation_fixture("residual")
    stages = list(candidate.stages)
    stages[1] = replace(
        stages[1],
        node_ids=stages[1].node_ids + (stages[0].node_ids[0],),
    )

    result = validate_partition_candidate(
        profile,
        replace(candidate, stages=tuple(stages)),
        support,
        graph=graph,
    )

    assert "duplicate_execution_node" in _validation_codes(result)


def test_validate_graph_candidate_rejects_missing_execution_node() -> None:
    profile, support, graph, candidate = _validation_fixture("residual")
    stages = list(candidate.stages)
    stages[0] = replace(stages[0], node_ids=stages[0].node_ids[1:])

    result = validate_partition_candidate(
        profile,
        replace(candidate, stages=tuple(stages)),
        support,
        graph=graph,
    )

    assert "missing_execution_node" in _validation_codes(result)


def test_validate_graph_candidate_rejects_duplicate_and_missing_state_owner() -> None:
    profile, support, graph, candidate = _validation_fixture("shared_module", 3)
    stages = list(candidate.stages)
    stages[1] = replace(
        stages[1],
        owned_state_ids=stages[1].owned_state_ids + ("p0000",),
        read_only_state_ids=tuple(
            state_id for state_id in stages[1].read_only_state_ids if state_id != "p0000"
        ),
    )
    duplicate = validate_partition_candidate(
        profile,
        replace(candidate, stages=tuple(stages)),
        support,
        graph=graph,
    )

    stages = list(candidate.stages)
    stages[0] = replace(
        stages[0],
        owned_state_ids=tuple(
            state_id for state_id in stages[0].owned_state_ids if state_id != "p0000"
        ),
    )
    missing = validate_partition_candidate(
        profile,
        replace(candidate, stages=tuple(stages)),
        support,
        graph=graph,
    )

    assert "duplicate_state_owner" in _validation_codes(duplicate)
    assert "missing_state_owner" in _validation_codes(missing)


def test_validate_graph_candidate_rejects_read_only_owner_conflict() -> None:
    profile, support, graph, candidate = _validation_fixture("shared_module", 3)
    stages = list(candidate.stages)
    stages[0] = replace(
        stages[0],
        read_only_state_ids=stages[0].read_only_state_ids + ("p0000",),
    )

    result = validate_partition_candidate(
        profile,
        replace(candidate, stages=tuple(stages)),
        support,
        graph=graph,
    )

    assert "state_read_owner_conflict" in _validation_codes(result)


def test_validate_graph_candidate_rejects_ambiguous_shared_state_owner() -> None:
    profile, support, graph, candidate = _validation_fixture("shared_tied_parameter", 2)
    shared_state_id = next(state.canonical_state_id for state in graph.states)
    stages = list(candidate.stages)
    stages[1] = replace(
        stages[1],
        owned_state_ids=stages[1].owned_state_ids + (shared_state_id,),
        read_only_state_ids=tuple(
            state_id
            for state_id in stages[1].read_only_state_ids
            if state_id != shared_state_id
        ),
    )

    result = validate_partition_candidate(
        profile,
        replace(candidate, stages=tuple(stages)),
        support,
        graph=graph,
    )

    codes = _validation_codes(result)
    assert "duplicate_state_owner" in codes
    assert "ambiguous_shared_state_owner" in codes


def test_validate_graph_candidate_rejects_missing_boundary_value_metadata() -> None:
    profile, support, graph, candidate = _validation_fixture("residual")

    result = validate_partition_candidate(
        profile,
        replace(candidate, communication_edges=()),
        support,
        graph=graph,
    )

    assert "missing_boundary_value" in _validation_codes(result)


class DynamicControlFlowModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.left = nn.Linear(4, 4)
        self.right = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.sum() > 0:
            return self.left(x)
        return self.right(x)


def custom_square(x: torch.Tensor) -> torch.Tensor:
    return x.square()


wrap("custom_square")


class CustomOpModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return custom_square(self.proj(x))


class SharedWeightLinear(nn.Module):
    def __init__(self, weight: nn.Parameter) -> None:
        super().__init__()
        self.weight = weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.transpose(0, 1)


class SharedParameterModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        weight = nn.Parameter(torch.randn(4, 4))
        self.left = SharedWeightLinear(weight)
        self.right = SharedWeightLinear(weight)
        self.out = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.left(x)
        hidden = self.right(hidden)
        return self.out(hidden)


class ImbalancedTwoStageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.small = nn.Linear(4, 4)
        self.big = nn.Linear(4, 4096)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.big(self.small(x))


def test_untraceable_dynamic_control_flow_returns_structured_unsupported() -> None:
    model = DynamicControlFlowModel()
    sample = torch.ones((2, 4))
    profile = build_model_profile(
        model,
        engine_id="pytorch_pipeline",
        model_name="dynamic-control-flow",
        sample_args=(sample,),
        memory_config=_memory_config(),
    )

    result = build_partition_profile(
        model,
        profile,
        sample_args=(sample,),
        memory_config=_memory_config(),
    )

    assert result.status == FeasibilityStatus.UNSUPPORTED
    assert any("untraceable graph" in reason for reason in result.reasons)
    assert result.failure_code is FailureCode.GRAPH_BREAK_UNSUPPORTED


def test_custom_op_returns_structured_unsupported() -> None:
    model = CustomOpModel()
    sample = torch.ones((2, 4))
    profile = build_model_profile(
        model,
        engine_id="pytorch_pipeline",
        model_name="custom-op-model",
        sample_args=(sample,),
        memory_config=_memory_config(),
    )

    result = build_partition_profile(
        model,
        profile,
        sample_args=(sample,),
        memory_config=_memory_config(),
    )

    assert result.status == FeasibilityStatus.UNSUPPORTED
    assert any("unsupported custom op" in reason for reason in result.reasons)
    assert result.failure_code is FailureCode.CUSTOM_OP_UNSUPPORTED


def test_partition_failure_codes_distinguish_budget_from_no_feasible_plan() -> None:
    case = _ordinary_case("sequential")
    profile = _profile_case(case, "sequential")
    support = discover_partition_support(
        case.module,
        profile,
        sample_args=case.args,
    )
    capacity = (1000, 1000)

    budget_result = generate_partition_candidates(
        profile,
        partition_support=support,
        memory_config=_memory_config(),
        min_stage_count=2,
        max_stage_count=3,
        usable_memory_bytes=capacity,
        max_candidates=1,
    )
    full_result = generate_partition_candidates(
        profile,
        partition_support=support,
        memory_config=_memory_config(),
        min_stage_count=2,
        max_stage_count=3,
        usable_memory_bytes=capacity,
        max_candidates=1000,
    )

    assert budget_result.status == FeasibilityStatus.INFEASIBLE
    assert budget_result.failure_code is FailureCode.SEARCH_BUDGET_LIMIT
    assert full_result.status == FeasibilityStatus.INFEASIBLE
    assert full_result.failure_code is FailureCode.NO_FEASIBLE_PLAN


def test_shared_parameter_boundary_is_rejected_explicitly() -> None:
    torch.manual_seed(42)
    model = SharedParameterModel()
    sample = torch.ones((2, 4))
    profile = build_model_profile(
        model,
        engine_id="pytorch_pipeline",
        model_name="shared-parameter-model",
        sample_args=(sample,),
        memory_config=_memory_config(),
    )
    support = discover_partition_support(model, profile, sample_args=(sample,))
    result = generate_partition_candidates(
        profile,
        partition_support=support,
        memory_config=_memory_config(),
    )

    assert result.candidates
    assert any(
        candidate.hard_constraint_status is FeasibilityStatus.FEASIBLE
        for candidate in result.candidates
    )
    assert any(
        "shared/tied parameter crosses boundary" in reason
        for candidate in result.candidates
        for reason in candidate.rejection_reasons
    )
    assert all(
        "b0000" not in candidate.selected_boundary_ids
        for candidate in result.candidates
        if candidate.hard_constraint_status is FeasibilityStatus.FEASIBLE
    )


def test_extreme_imbalance_is_infeasible_for_similar_gpu_capacities() -> None:
    model = ImbalancedTwoStageModel()
    sample = torch.ones((2, 4))
    profile = build_model_profile(
        model,
        engine_id="pytorch_pipeline",
        model_name="imbalanced-two-stage-model",
        sample_args=(sample,),
        memory_config=_memory_config(),
    )

    result = build_partition_profile(
        model,
        profile,
        sample_args=(sample,),
        memory_config=_memory_config(),
        usable_memory_bytes=(100_000_000, 100_000_000),
        min_stage_count=2,
        max_stage_count=2,
    )

    assert result.status == FeasibilityStatus.INFEASIBLE
    assert result.candidates
    assert result.candidates[0].hard_constraint_status == FeasibilityStatus.INFEASIBLE
    assert any(
        "capacity-aware imbalance is too extreme" in reason
        for reason in result.candidates[0].rejection_reasons
    )
