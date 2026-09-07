from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
from torch import nn

from shardgrid.common.enums import BackendStatus, Health, PhysicalOS, RuntimeOS
from shardgrid.common.models import as_hostname, as_worker_id
from shardgrid.engines.models import EstimateKind, ModelProfile, ProfileResult
from shardgrid.planner.generic_graph import capture_generic_graph
from shardgrid.planner.memory import (
    MemoryEstimationConfig,
    _iter_target_modules,
    build_model_profile,
    dtype_bytes,
    estimate_stage_memory,
    evaluate_stage_memory_fit,
)
from shardgrid.resources.models import WorkerResource


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


def _profile_ordinary_case(name: str) -> ModelProfile:
    case = _ordinary_case(name)
    return build_model_profile(
        case.module,
        engine_id="pytorch_pipeline",
        model_name=name,
        sample_args=case.args,
        sample_kwargs=dict(case.kwargs or {}),
    )


def _parameter_names(profile: ModelProfile) -> set[str]:
    return {
        parameter_name
        for module in profile.modules
        for parameter_name in module.parameter_names
    }


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


class BufferMemoryModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(4)
        self.proj = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.bn(x))


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


def test_model_profile_module_order_follows_target_module_registration_order() -> None:
    case = _ordinary_case("multi_branch")
    profile = _profile_ordinary_case("multi_branch")
    graph = capture_generic_graph(case.module.eval(), sample_args=case.args)

    target_paths = [name for name, _module in _iter_target_modules(case.module)]
    profile_paths = [module.module_path for module in profile.modules]
    executed_paths = [node.module_path for node in graph.nodes if node.module_path]

    assert target_paths == ["left", "right", "gate", "head"]
    assert profile_paths == target_paths
    assert profile.module_order == tuple(module.module_id for module in profile.modules)
    assert executed_paths.index("gate") < executed_paths.index("right")
    assert profile_paths != executed_paths[: len(profile_paths)]


def test_model_profile_has_no_slot_for_parameterless_functional_execution_nodes() -> None:
    case = _ordinary_case("unet_like")
    profile = _profile_ordinary_case("unet_like")
    graph = capture_generic_graph(case.module.eval(), sample_args=case.args)

    profile_paths = [module.module_path for module in profile.modules]
    functional_targets = [
        node.target
        for node in graph.nodes
        if node.module_path is None and node.op_kind not in {"placeholder", "output"}
    ]

    assert profile_paths == ["enc1", "enc2", "dec1", "out"]
    assert any("avg_pool2d" in target for target in functional_targets)
    assert any("interpolate" in target for target in functional_targets)
    assert any("cat" in target for target in functional_targets)
    assert all("avg_pool2d" not in path for path in profile_paths)
    assert all("interpolate" not in path for path in profile_paths)
    assert all("cat" not in path for path in profile_paths)
    assert any(
        node.module_path is None
        and "cat" in node.target
        and node.state_ids == ()
        and node.activation_bytes > 0
        for node in profile.execution_costs
    )
    assert profile.graph_value_activation_bytes is not None
    assert profile.graph_value_activation_bytes > 0


def test_model_profile_tracks_root_owned_state_without_standalone_module_entry() -> None:
    case = _ordinary_case("transformer")
    profile = _profile_ordinary_case("transformer")

    model_parameter_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in case.module.parameters()
    )
    state_by_key = {state.state_dict_key: state for state in profile.state_memory}

    assert "position" in dict(case.module.named_parameters())
    assert "position" not in _parameter_names(profile)
    assert "" not in [module.module_path for module in profile.modules]
    assert profile.total_memory.parameter_bytes < model_parameter_bytes
    assert state_by_key["position"].kind == "parameter"
    assert state_by_key["position"].bytes == case.module.position.numel() * 4
    assert profile.state_parameter_bytes == model_parameter_bytes


def test_model_profile_currently_counts_tied_parameters_per_module_owner() -> None:
    case = _ordinary_case("shared_tied_parameter")
    profile = _profile_ordinary_case("shared_tied_parameter")

    unique_model_parameter_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in case.module.parameters()
    )

    assert case.module.embedding.weight is case.module.decoder.weight
    assert profile.shared_parameter_groups == (("embedding.weight", "decoder.weight"),)
    assert _parameter_names(profile) == {"embedding.weight", "decoder.weight"}
    assert profile.total_memory.parameter_bytes == unique_model_parameter_bytes * 2
    assert profile.state_parameter_bytes == unique_model_parameter_bytes
    assert {
        state.checkpoint_owner_key
        for state in profile.state_memory
        if state.shared_group_id is not None
    } == {"embedding.weight"}
    assert "shared/tied parameters detected in model profile" in profile.diagnostics


def test_model_profile_tracks_buffer_state_memory_separately() -> None:
    model = BufferMemoryModel().eval()
    profile = build_model_profile(
        model,
        engine_id="pytorch_pipeline",
        model_name="buffer-memory-model",
        sample_args=(torch.randn(3, 4),),
    )
    state_by_key = {state.state_dict_key: state for state in profile.state_memory}

    expected_buffer_bytes = sum(
        buffer.numel() * buffer.element_size() for buffer in model.buffers()
    )

    assert {"bn.running_mean", "bn.running_var", "bn.num_batches_tracked"} <= set(
        state_by_key
    )
    assert state_by_key["bn.running_mean"].kind == "buffer"
    assert profile.state_buffer_bytes == expected_buffer_bytes
    assert profile.state_parameter_bytes == sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    )


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
