from __future__ import annotations

import torch
from torch import nn

from shardgrid.common.enums import BackendStatus, Health, PhysicalOS, RuntimeOS
from shardgrid.common.models import as_hostname, as_worker_id
from shardgrid.engines.models import EstimateKind, ModelProfile, ProfileResult
from shardgrid.planner.memory import (
    MemoryEstimationConfig,
    build_model_profile,
    dtype_bytes,
    estimate_stage_memory,
    evaluate_stage_memory_fit,
)
from shardgrid.resources.models import WorkerResource


class TinyTransformerModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(256, 32)
        self.pos = nn.Embedding(32, 32)
        layer = nn.TransformerEncoderLayer(
            d_model=32,
            nhead=4,
            dim_feedforward=64,
            dropout=0.0,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.ln = nn.LayerNorm(32)
        self.head = nn.Linear(32, 256, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        hidden = self.embed(input_ids) + self.pos(positions)
        hidden = self.encoder(hidden)
        hidden = self.ln(hidden)
        return self.head(hidden)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class TinySequenceMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16, 32, bias=False),
            nn.ReLU(),
            nn.Linear(32, 8, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def test_minimal_transformer_profile_captures_reusable_memory_breakdown() -> None:
    torch.manual_seed(42)
    model = TinyTransformerModel()
    input_ids = torch.randint(0, 256, (2, 31), dtype=torch.long)
    config = MemoryEstimationConfig(
        optimizer_type="adamw",
        gradient_dtype="float32",
        optimizer_state_dtype="float32",
        runtime_overhead_bytes=8 * 1024 * 1024,
        communication_buffer_bytes=2 * 1024 * 1024,
        safety_headroom_bytes=256 * 1024 * 1024,
        temporary_buffer_factor=0.25,
    )

    profile = build_model_profile(
        model,
        engine_id="pytorch_pipeline",
        model_name="minimal-transformer",
        sample_args=(input_ids,),
        memory_config=config,
        required_backends=("nccl", "gloo"),
    )

    assert profile.model_name == "minimal-transformer"
    assert profile.required_backends == ("nccl", "gloo")
    assert len(profile.modules) > 6
    assert profile.communication_edges
    assert any(module.module_path == "embed" for module in profile.modules)
    assert any(module.module_path == "head" for module in profile.modules)

    parameter_count = model.parameter_count()
    parameter_bytes = parameter_count * dtype_bytes(torch.float32)
    assert profile.total_memory.parameter_bytes == parameter_bytes
    assert profile.total_memory.gradient_bytes == parameter_bytes
    assert profile.total_memory.optimizer_bytes == parameter_count * 8
    assert profile.total_memory.activation_bytes is not None
    assert profile.total_memory.activation_bytes > 0
    assert profile.total_memory.temporary_bytes is not None
    assert profile.total_memory.temporary_bytes > 0
    assert (
        profile.total_memory.planner_required_bytes
        == profile.total_memory.estimated_peak_bytes + config.safety_headroom_bytes
    )
    assert profile.total_memory.estimate_kind == EstimateKind.ESTIMATED


def test_stage_range_estimation_respects_precision_and_master_weights() -> None:
    model = TinySequenceMLP().half()
    sample = torch.ones((4, 16), dtype=torch.float16)
    config = MemoryEstimationConfig(
        optimizer_type="adamw",
        activation_dtype="float16",
        gradient_dtype="float16",
        optimizer_state_dtype="float32",
        master_weight_dtype="float32",
        temporary_buffer_factor=0.5,
        runtime_overhead_bytes=1024,
        communication_buffer_bytes=2048,
        safety_headroom_bytes=4096,
    )

    profile = build_model_profile(
        model,
        engine_id="pytorch_pipeline",
        model_name="tiny-sequence-mlp",
        sample_args=(sample,),
        memory_config=config,
    )
    stage0 = estimate_stage_memory(profile, (0, 2), config)

    linear0 = profile.modules[0]
    relu = profile.modules[1]
    assert linear0.module_type == "Linear"
    assert relu.module_type == "ReLU"

    expected_parameter_bytes = 16 * 32 * 2
    expected_gradient_bytes = 16 * 32 * 2
    expected_optimizer_bytes = 16 * 32 * (8 + 4)
    expected_activation_bytes = (4 * 32 * 2) * 2
    expected_temporary_bytes = expected_activation_bytes // 2
    expected_peak = (
        expected_parameter_bytes
        + expected_gradient_bytes
        + expected_optimizer_bytes
        + expected_activation_bytes
        + expected_temporary_bytes
        + 1024
        + 2048
    )

    assert stage0.parameter_bytes == expected_parameter_bytes
    assert stage0.gradient_bytes == expected_gradient_bytes
    assert stage0.optimizer_bytes == expected_optimizer_bytes
    assert stage0.activation_bytes == expected_activation_bytes
    assert stage0.temporary_bytes == expected_temporary_bytes
    assert stage0.estimated_peak_bytes == expected_peak
    assert stage0.planner_required_bytes == expected_peak + 4096


def test_engine_profile_result_is_reused_when_available() -> None:
    model = TinySequenceMLP()
    sample = torch.ones((2, 16), dtype=torch.float32)
    expected = build_model_profile(
        model,
        engine_id="galvatron",
        model_name="engine-profile",
        sample_args=(sample,),
    )
    profile_result = ProfileResult(
        engine_id="galvatron",
        status=BackendStatus.AVAILABLE,
        model_profile=expected,
    )

    actual = build_model_profile(
        model,
        engine_id="galvatron",
        model_name="ignored",
        sample_args=(sample,),
        profile_result=profile_result,
    )

    assert actual is expected


def test_stage_memory_fit_rejects_worker_after_headroom() -> None:
    model = TinySequenceMLP().half()
    sample = torch.ones((4, 16), dtype=torch.float16)
    config = MemoryEstimationConfig(
        optimizer_type="adamw",
        activation_dtype="float16",
        gradient_dtype="float16",
        optimizer_state_dtype="float32",
        master_weight_dtype="float32",
        safety_headroom_bytes=4 * 1024 * 1024 * 1024,
    )
    profile = build_model_profile(
        model,
        engine_id="pytorch_pipeline",
        model_name="tiny-sequence-mlp",
        sample_args=(sample,),
        memory_config=config,
    )
    estimate = estimate_stage_memory(profile, (0, len(profile.modules)), config)
    worker = WorkerResource(
        worker_id=as_worker_id("gpu1650"),
        hostname=as_hostname("worker-d"),
        physical_os=PhysicalOS.WINDOWS,
        runtime_os=RuntimeOS.WSL2_LINUX,
        gpu_name="GTX 1650",
        gpu_total_memory=4096,
        health=Health.HEALTHY,
    )

    fit = evaluate_stage_memory_fit(worker, estimate)

    assert fit.fits is False
    assert fit.shortfall_bytes is not None
    assert fit.shortfall_bytes > 0
    assert "after headroom" in (fit.reason or "")


def test_unsupported_optimizer_stays_explicitly_unsupported() -> None:
    model = TinySequenceMLP()
    sample = torch.ones((2, 16), dtype=torch.float32)
    profile = build_model_profile(
        model,
        engine_id="pytorch_pipeline",
        model_name="tiny-sequence-mlp",
        sample_args=(sample,),
    )

    estimate = estimate_stage_memory(
        profile,
        (0, len(profile.modules)),
        MemoryEstimationConfig(optimizer_type="lion"),
    )

    assert estimate.estimate_kind == EstimateKind.UNSUPPORTED
    assert estimate.estimated_peak_bytes is None
    assert estimate.planner_required_bytes is None
    assert "unsupported" in estimate.notes[0]


def test_model_profile_round_trip_keeps_memory_metadata() -> None:
    model = TinySequenceMLP()
    sample = torch.ones((2, 16), dtype=torch.float32)
    profile = build_model_profile(
        model,
        engine_id="pytorch_pipeline",
        model_name="tiny-sequence-mlp",
        sample_args=(sample,),
    )

    restored = ModelProfile.from_dict(profile.to_dict())

    assert restored == profile
