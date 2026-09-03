from __future__ import annotations

import math
import sys
from pathlib import Path

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

from shardgrid.planner.memory import MemoryEstimationConfig, build_model_profile
from shardgrid.planner.partitioning import (
    build_partition_profile,
    discover_partition_support,
    generate_partition_candidates,
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

    assert result_a.status == FeasibilityStatus.FEASIBLE
    assert len(result_a.candidates) >= 1
    assert [candidate.candidate_id for candidate in result_a.candidates] == [
        candidate.candidate_id for candidate in result_b.candidates
    ]
    assert all(candidate.original_engine_plan_ref == "/tmp/engine-plan.json" for candidate in result_a.candidates)


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
    assert all(
        candidate.hard_constraint_status is not FeasibilityStatus.FEASIBLE
        for candidate in result.candidates
    )
    assert any(
        "shared/tied parameter crosses boundary" in reason
        for candidate in result.candidates
        for reason in candidate.rejection_reasons
    )
