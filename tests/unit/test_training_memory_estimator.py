from __future__ import annotations

import pytest

from shardgrid.engines.models import (
    EstimateKind,
    ModelProfile,
    ModuleProfile,
    StateMemoryProfile,
    TensorMetadata,
    TrainingMemoryEstimate,
)
from shardgrid.planner.memory import MemoryEstimationConfig, estimate_stage_memory


def _module(
    module_id: str,
    *,
    parameter_count: int = 0,
    parameter_bytes: int = 0,
    trainable_parameter_count: int | None = None,
    trainable_parameter_bytes: int | None = None,
    output_bytes: tuple[int | None, ...] = (),
    output_shape: tuple[int | str, ...] = (1,),
    memory: TrainingMemoryEstimate | None = None,
) -> ModuleProfile:
    trainable_count = parameter_count if trainable_parameter_count is None else trainable_parameter_count
    trainable_bytes = parameter_bytes if trainable_parameter_bytes is None else trainable_parameter_bytes
    return ModuleProfile(
        module_id=module_id,
        module_path=module_id,
        module_type="SyntheticLayer",
        parameter_names=(f"{module_id}.weight",) if parameter_count else (),
        parameter_count=parameter_count,
        parameter_bytes=parameter_bytes,
        trainable_parameter_count=trainable_count,
        trainable_parameter_bytes=trainable_bytes,
        output_tensors=tuple(
            TensorMetadata(
                name=f"{module_id}:output{index}",
                shape=output_shape,
                dtype="float32",
                estimated_bytes=value,
                estimate_kind=EstimateKind.ESTIMATED,
                source="metadata",
            )
            for index, value in enumerate(output_bytes)
        ),
        memory=memory or TrainingMemoryEstimate(),
    )


def _profile(
    *modules: ModuleProfile,
    shared_parameter_groups: tuple[tuple[str, ...], ...] = (),
    state_memory: tuple[StateMemoryProfile, ...] = (),
) -> ModelProfile:
    return ModelProfile(
        profile_id="synthetic-profile",
        engine_id="pytorch_pipeline",
        model_name="synthetic",
        modules=tuple(modules),
        module_order=tuple(module.module_id for module in modules),
        shared_parameter_groups=shared_parameter_groups,
        state_memory=state_memory,
        state_parameter_bytes=sum(
            item.bytes for item in state_memory if item.kind == "parameter"
        ),
        state_buffer_bytes=sum(item.bytes for item in state_memory if item.kind == "buffer"),
    )


def test_training_memory_estimate_covers_core_state_gradient_optimizer_activation_and_buffers() -> None:
    module = _module(
        "encoder",
        parameter_count=128,
        parameter_bytes=512,
        output_bytes=(256, 128),
    )
    profile = _profile(
        module,
        state_memory=(
            StateMemoryProfile("state:0000", "parameter", "encoder.weight", bytes=512),
            StateMemoryProfile("state:0001", "buffer", "encoder.running", bytes=64),
        ),
    )
    config = MemoryEstimationConfig(
        optimizer_type="adamw",
        gradient_dtype="float32",
        optimizer_state_dtype="float32",
        runtime_overhead_bytes=32,
        communication_buffer_bytes=16,
        safety_headroom_bytes=8,
        source="unit-test-metadata",
    )

    estimate = estimate_stage_memory(profile, (0, 1), config)

    assert estimate.parameter_bytes == 512
    assert profile.state_buffer_bytes == 64
    assert estimate.gradient_bytes == 512
    assert estimate.optimizer_bytes == 128 * 4 * 2
    assert estimate.activation_bytes == 384
    assert estimate.temporary_bytes == 0
    assert estimate.runtime_overhead_bytes == 32
    assert estimate.communication_buffer_bytes == 16
    assert estimate.estimated_peak_bytes == 512 + 512 + 1024 + 384 + 32 + 16
    assert estimate.planner_required_bytes == estimate.estimated_peak_bytes + 8
    assert estimate.estimate_kind == EstimateKind.ESTIMATED
    assert estimate.source == "unit-test-metadata"


def test_estimator_accounts_for_activation_liveness_backward_saved_tensors_and_workspace_separately() -> None:
    activation = 1_000
    backward_saved = 2_000
    workspace = 700
    module = _module(
        "attention",
        parameter_count=10,
        parameter_bytes=40,
        output_bytes=(activation,),
        memory=TrainingMemoryEstimate(
            activation_bytes=backward_saved,
            temporary_bytes=workspace,
        ),
    )
    profile = _profile(module)
    config = MemoryEstimationConfig(
        optimizer_type="sgd",
        optimizer_kwargs={"momentum": 0.9},
        temporary_buffer_factor=0.25,
        communication_buffer_bytes=300,
    )

    estimate = estimate_stage_memory(profile, (0, 1), config)

    assert estimate.activation_bytes == activation
    assert estimate.temporary_bytes == max(workspace, int(activation * 0.25))
    assert estimate.optimizer_bytes == 10 * 4
    assert estimate.communication_buffer_bytes == 300
    assert estimate.estimated_peak_bytes == (
        40
        + 40
        + 40
        + activation
        + max(workspace, int(activation * 0.25))
        + 300
    )


def test_dtype_mixed_precision_and_batch_shape_drive_component_estimates() -> None:
    module = _module(
        "mlp",
        parameter_count=64,
        parameter_bytes=128,
        output_bytes=(None,),
        output_shape=(4, 8),
    )
    profile = _profile(module)
    config = MemoryEstimationConfig(
        optimizer_type="adamw",
        activation_dtype="float16",
        gradient_dtype="float16",
        optimizer_state_dtype="float32",
        master_weight_dtype="float32",
        temporary_buffer_factor=0.5,
    )

    estimate = estimate_stage_memory(profile, (0, 1), config)

    assert estimate.parameter_bytes == 128
    assert estimate.gradient_bytes == 64 * 2
    assert estimate.optimizer_bytes == (64 * 4 * 2) + (64 * 4)
    assert estimate.activation_bytes == 4 * 8 * 2
    assert estimate.temporary_bytes == (4 * 8 * 2) // 2


def test_unknown_batch_shape_preserves_unknown_activation_without_cpu_or_gpu_execution() -> None:
    module = _module(
        "dynamic",
        parameter_count=2,
        parameter_bytes=8,
        output_bytes=(None,),
        output_shape=("batch", 8),
    )
    profile = _profile(module)

    estimate = estimate_stage_memory(
        profile,
        (0, 1),
        MemoryEstimationConfig(activation_dtype="float16"),
    )

    assert estimate.activation_bytes is None
    assert estimate.temporary_bytes == 0
    assert estimate.estimated_peak_bytes == estimate.parameter_bytes + estimate.gradient_bytes + estimate.optimizer_bytes


def test_unsupported_optimizer_returns_structured_unsupported_estimate() -> None:
    profile = _profile(_module("decoder", parameter_count=16, parameter_bytes=64))

    estimate = estimate_stage_memory(
        profile,
        (0, 1),
        MemoryEstimationConfig(optimizer_type="8bit-adam"),
    )

    assert estimate.estimate_kind == EstimateKind.UNSUPPORTED
    assert estimate.estimated_peak_bytes is None
    assert estimate.planner_required_bytes is None
    assert estimate.notes == ("optimizer 8bit-adam is unsupported for memory estimation",)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "T074 coverage gap: module-slice estimates still count tied/shared "
        "parameter owners per module instead of charging one canonical state object."
    ),
)
def test_shared_tied_state_is_not_double_counted_in_module_slice_estimate() -> None:
    profile = _profile(
        _module("embedding", parameter_count=32, parameter_bytes=128),
        _module("decoder", parameter_count=32, parameter_bytes=128),
        shared_parameter_groups=(("embedding.weight", "decoder.weight"),),
        state_memory=(
            StateMemoryProfile(
                "state:0000",
                "parameter",
                "embedding.weight",
                bytes=128,
                shared_group_id="shared:0000",
                checkpoint_owner_key="embedding.weight",
            ),
        ),
    )

    estimate = estimate_stage_memory(profile, (0, 2))

    assert estimate.parameter_bytes == 128
