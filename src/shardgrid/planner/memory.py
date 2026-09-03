"""Model profiling and training-memory estimation helpers for T109.

The planner consumes stable metadata and conservative estimates. This module
does not generate partition candidates or perform placement search.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from shardgrid.common.models import WorkerId
from shardgrid.engines.models import (
    AutomaticPartitionSupport,
    CommunicationEdge,
    EstimateKind,
    ModelProfile,
    ModuleProfile,
    ProfileResult,
    TensorMetadata,
    TrainingMemoryEstimate,
)
from shardgrid.resources.models import WorkerResource

if TYPE_CHECKING:
    import torch
    from torch import nn

_DTYPE_BYTES: dict[str, int] = {
    "bool": 1,
    "uint8": 1,
    "int8": 1,
    "float16": 2,
    "half": 2,
    "bfloat16": 2,
    "int16": 2,
    "short": 2,
    "float32": 4,
    "float": 4,
    "int32": 4,
    "int": 4,
    "float64": 8,
    "double": 8,
    "int64": 8,
    "long": 8,
}
_DTYPE_ALIASES = {
    "fp16": "float16",
    "bf16": "bfloat16",
    "fp32": "float32",
    "fp64": "float64",
}


@dataclass(frozen=True)
class MemoryEstimationConfig:
    optimizer_type: str = "adamw"
    optimizer_kwargs: Mapping[str, Any] = field(default_factory=dict)
    activation_dtype: str | None = None
    gradient_dtype: str | None = None
    optimizer_state_dtype: str | None = "float32"
    master_weight_dtype: str | None = None
    temporary_buffer_factor: float = 0.0
    runtime_overhead_bytes: int = 0
    communication_buffer_bytes: int = 0
    safety_headroom_bytes: int = 0
    source: str = "shardgrid.heuristic"

    def __post_init__(self) -> None:
        if self.temporary_buffer_factor < 0:
            raise ValueError("temporary_buffer_factor must be >= 0")
        for field_name in (
            "runtime_overhead_bytes",
            "communication_buffer_bytes",
            "safety_headroom_bytes",
        ):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} must be >= 0")
        if not self.optimizer_type.strip():
            raise ValueError("optimizer_type must be a non-empty string")
        if not self.source.strip():
            raise ValueError("source must be a non-empty string")


@dataclass(frozen=True)
class StageMemoryFit:
    worker_id: WorkerId
    fits: bool
    usable_bytes: int | None
    estimated_peak_bytes: int | None
    safety_headroom_bytes: int
    planner_required_bytes: int | None
    shortfall_bytes: int | None
    reason: str | None = None


def normalize_dtype_name(dtype: Any) -> str | None:
    if dtype is None:
        return None
    normalized = str(dtype).replace("torch.", "").strip().lower()
    if not normalized:
        return None
    return _DTYPE_ALIASES.get(normalized, normalized)


def dtype_bytes(dtype: Any) -> int | None:
    normalized = normalize_dtype_name(dtype)
    if normalized is None:
        return None
    return _DTYPE_BYTES.get(normalized)


def build_model_profile(
    model: "nn.Module",
    *,
    engine_id: str,
    model_name: str | None = None,
    sample_args: Sequence[Any] = (),
    sample_kwargs: Mapping[str, Any] | None = None,
    memory_config: MemoryEstimationConfig | None = None,
    profile_result: ProfileResult | None = None,
    partition_support: AutomaticPartitionSupport | None = None,
    required_runtime: str | None = None,
    required_backends: Sequence[str] = (),
) -> ModelProfile:
    if profile_result is not None and profile_result.model_profile is not None:
        return profile_result.model_profile

    if not engine_id.strip():
        raise ValueError("engine_id must be a non-empty string")

    config = memory_config or MemoryEstimationConfig()
    sample_kwargs = dict(sample_kwargs or {})
    modules, communication_edges = _profile_modules(
        model,
        sample_args=sample_args,
        sample_kwargs=sample_kwargs,
        config=config,
    )
    diagnostics = []
    if profile_result is not None:
        diagnostics.extend(profile_result.diagnostics)
        diagnostics.extend(profile_result.notes)

    shared_groups = _shared_parameter_groups(model)
    if shared_groups:
        diagnostics.append("shared/tied parameters detected in model profile")

    total_memory = estimate_stage_memory(
        ModelProfile(
            profile_id=_profile_id(engine_id, model_name or type(model).__name__),
            engine_id=engine_id,
            model_name=model_name or type(model).__name__,
            modules=modules,
            module_order=tuple(module.module_id for module in modules),
            partition_support=partition_support,
            communication_edges=communication_edges,
            shared_parameter_groups=shared_groups,
            required_runtime=required_runtime,
            required_backends=tuple(required_backends),
            total_memory=TrainingMemoryEstimate(),
            evidence_paths=tuple(profile_result.evidence_paths) if profile_result else (),
            diagnostics=tuple(diagnostics),
        ),
        (0, len(modules)),
        config,
    )
    return ModelProfile(
        profile_id=_profile_id(engine_id, model_name or type(model).__name__),
        engine_id=engine_id,
        model_name=model_name or type(model).__name__,
        modules=modules,
        module_order=tuple(module.module_id for module in modules),
        partition_support=partition_support,
        communication_edges=communication_edges,
        shared_parameter_groups=shared_groups,
        required_runtime=required_runtime,
        required_backends=tuple(required_backends),
        total_memory=total_memory,
        evidence_paths=tuple(profile_result.evidence_paths) if profile_result else (),
        diagnostics=tuple(diagnostics),
    )


def estimate_stage_memory(
    profile: ModelProfile,
    module_range: tuple[int, int] | slice | Sequence[str],
    config: MemoryEstimationConfig | None = None,
) -> TrainingMemoryEstimate:
    selected = _select_modules(profile, module_range)
    if not selected:
        raise ValueError("module_range must select at least one module")
    estimate_config = config or MemoryEstimationConfig()

    parameter_bytes = sum(module.parameter_bytes for module in selected)
    trainable_parameter_bytes = sum(
        module.trainable_parameter_bytes for module in selected
    )
    parameter_count = sum(module.parameter_count for module in selected)
    trainable_parameter_count = sum(
        module.trainable_parameter_count for module in selected
    )

    activation_total = 0
    activation_known = False
    for module in selected:
        value = _activation_bytes_for_module(module, estimate_config)
        if value is not None:
            activation_total += value
            activation_known = True

    temporary_total = sum(
        _temporary_bytes_for_module(module, estimate_config) for module in selected
    )

    gradient_bytes = _gradient_bytes(
        trainable_parameter_count,
        trainable_parameter_bytes,
        estimate_config,
    )
    optimizer_bytes = _optimizer_bytes(
        parameter_count,
        parameter_bytes,
        trainable_parameter_count,
        trainable_parameter_bytes,
        estimate_config,
    )
    if gradient_bytes is None or optimizer_bytes is None:
        note = f"optimizer {estimate_config.optimizer_type} is unsupported for memory estimation"
        return TrainingMemoryEstimate(
            parameter_bytes=parameter_bytes,
            gradient_bytes=gradient_bytes,
            optimizer_bytes=optimizer_bytes,
            activation_bytes=activation_total if activation_known else None,
            temporary_bytes=temporary_total,
            runtime_overhead_bytes=estimate_config.runtime_overhead_bytes,
            communication_buffer_bytes=estimate_config.communication_buffer_bytes,
            estimated_peak_bytes=None,
            safety_headroom_bytes=estimate_config.safety_headroom_bytes,
            planner_required_bytes=None,
            estimate_kind=EstimateKind.UNSUPPORTED,
            source=estimate_config.source,
            notes=(note,),
        )

    activation_bytes = activation_total if activation_known else None
    peak = (
        parameter_bytes
        + gradient_bytes
        + optimizer_bytes
        + (activation_bytes or 0)
        + temporary_total
        + estimate_config.runtime_overhead_bytes
        + estimate_config.communication_buffer_bytes
    )
    return TrainingMemoryEstimate(
        parameter_bytes=parameter_bytes,
        gradient_bytes=gradient_bytes,
        optimizer_bytes=optimizer_bytes,
        activation_bytes=activation_bytes,
        temporary_bytes=temporary_total,
        runtime_overhead_bytes=estimate_config.runtime_overhead_bytes,
        communication_buffer_bytes=estimate_config.communication_buffer_bytes,
        estimated_peak_bytes=peak,
        safety_headroom_bytes=estimate_config.safety_headroom_bytes,
        planner_required_bytes=peak + estimate_config.safety_headroom_bytes,
        estimate_kind=EstimateKind.ESTIMATED,
        source=estimate_config.source,
    )


def evaluate_stage_memory_fit(
    worker: WorkerResource,
    estimate: TrainingMemoryEstimate,
) -> StageMemoryFit:
    usable_bytes = _worker_usable_bytes(worker)
    required = estimate.planner_required_bytes
    if usable_bytes is None:
        return StageMemoryFit(
            worker_id=worker.worker_id,
            fits=False,
            usable_bytes=None,
            estimated_peak_bytes=estimate.estimated_peak_bytes,
            safety_headroom_bytes=estimate.safety_headroom_bytes,
            planner_required_bytes=required,
            shortfall_bytes=None,
            reason="worker usable GPU memory is unknown",
        )
    if required is None:
        return StageMemoryFit(
            worker_id=worker.worker_id,
            fits=False,
            usable_bytes=usable_bytes,
            estimated_peak_bytes=estimate.estimated_peak_bytes,
            safety_headroom_bytes=estimate.safety_headroom_bytes,
            planner_required_bytes=None,
            shortfall_bytes=None,
            reason="stage memory estimate is unavailable",
        )
    if required <= usable_bytes:
        return StageMemoryFit(
            worker_id=worker.worker_id,
            fits=True,
            usable_bytes=usable_bytes,
            estimated_peak_bytes=estimate.estimated_peak_bytes,
            safety_headroom_bytes=estimate.safety_headroom_bytes,
            planner_required_bytes=required,
            shortfall_bytes=0,
        )
    return StageMemoryFit(
        worker_id=worker.worker_id,
        fits=False,
        usable_bytes=usable_bytes,
        estimated_peak_bytes=estimate.estimated_peak_bytes,
        safety_headroom_bytes=estimate.safety_headroom_bytes,
        planner_required_bytes=required,
        shortfall_bytes=required - usable_bytes,
        reason="estimated peak training memory exceeds usable GPU memory after headroom",
    )


def _profile_modules(
    model: "nn.Module",
    *,
    sample_args: Sequence[Any],
    sample_kwargs: Mapping[str, Any],
    config: MemoryEstimationConfig,
) -> tuple[tuple[ModuleProfile, ...], tuple[CommunicationEdge, ...]]:
    import torch

    targets = list(_iter_target_modules(model))
    inputs: dict[str, tuple[TensorMetadata, ...]] = {}
    outputs: dict[str, tuple[TensorMetadata, ...]] = {}
    call_order: list[str] = []
    handles = []

    def _hook(name: str):
        def capture(_module: Any, args: tuple[Any, ...], output: Any) -> None:
            if name not in call_order:
                call_order.append(name)
            inputs[name] = _extract_tensors(args, prefix=f"{name}:input")
            outputs[name] = _extract_tensors(output, prefix=f"{name}:output")

        return capture

    was_training = model.training
    try:
        for name, module in targets:
            handles.append(module.register_forward_hook(_hook(name)))
        model.eval()
        with torch.inference_mode():
            model(*sample_args, **sample_kwargs)
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)

    module_by_name = dict(targets)
    ordered_names = [name for name, _module in targets]
    modules: list[ModuleProfile] = []
    for index, name in enumerate(ordered_names):
        module = module_by_name[name]
        param_info = list(module.named_parameters(recurse=False))
        parameter_names = tuple(
            f"{name}.{local_name}" if name else local_name
            for local_name, _parameter in param_info
        )
        parameter_count = sum(parameter.numel() for _, parameter in param_info)
        parameter_bytes = sum(
            parameter.numel() * parameter.element_size()
            for _, parameter in param_info
        )
        trainable = [
            parameter for _, parameter in param_info if parameter.requires_grad
        ]
        trainable_parameter_count = sum(parameter.numel() for parameter in trainable)
        trainable_parameter_bytes = sum(
            parameter.numel() * parameter.element_size()
            for parameter in trainable
        )
        input_tensors = inputs.get(name, ())
        output_tensors = outputs.get(name, ())
        memory = estimate_stage_memory(
            ModelProfile(
                profile_id="scratch",
                engine_id="scratch",
                model_name="scratch",
                modules=(
                    ModuleProfile(
                        module_id=f"m{index:04d}",
                        module_path=name,
                        module_type=type(module).__name__,
                        parameter_names=parameter_names,
                        parameter_count=parameter_count,
                        parameter_bytes=parameter_bytes,
                        trainable_parameter_count=trainable_parameter_count,
                        trainable_parameter_bytes=trainable_parameter_bytes,
                        input_tensors=input_tensors,
                        output_tensors=output_tensors,
                    ),
                ),
                module_order=(f"m{index:04d}",),
            ),
            (0, 1),
            config,
        )
        modules.append(
            ModuleProfile(
                module_id=f"m{index:04d}",
                module_path=name,
                module_type=type(module).__name__,
                parameter_names=parameter_names,
                parameter_count=parameter_count,
                parameter_bytes=parameter_bytes,
                trainable_parameter_count=trainable_parameter_count,
                trainable_parameter_bytes=trainable_parameter_bytes,
                input_tensors=input_tensors,
                output_tensors=output_tensors,
                memory=memory,
            )
        )

    edges: list[CommunicationEdge] = []
    for current, following in zip(modules, modules[1:]):
        edges.append(
            CommunicationEdge(
                source_module_id=current.module_id,
                target_module_id=following.module_id,
                activation=current.output_tensors,
                gradient=current.output_tensors,
                estimate_kind=EstimateKind.MEASURED,
                source="lightweight_forward",
            )
        )
    return tuple(modules), tuple(edges)


def _iter_target_modules(model: "nn.Module") -> Sequence[tuple[str, "nn.Module"]]:
    targets: list[tuple[str, Any]] = []
    for name, module in model.named_modules():
        if not name:
            continue
        children = tuple(module.children())
        direct_params = tuple(module.named_parameters(recurse=False))
        direct_buffers = tuple(module.named_buffers(recurse=False))
        if children and not direct_params and not direct_buffers:
            continue
        targets.append((name, module))
    return targets


def _extract_tensors(value: Any, *, prefix: str) -> tuple[TensorMetadata, ...]:
    import torch

    tensors: list[TensorMetadata] = []

    def visit(item: Any, name: str) -> None:
        if isinstance(item, torch.Tensor):
            tensors.append(
                TensorMetadata(
                    name=name,
                    shape=tuple(int(dim) for dim in item.shape),
                    dtype=normalize_dtype_name(item.dtype),
                    estimated_bytes=int(item.numel() * item.element_size()),
                    estimate_kind=EstimateKind.MEASURED,
                    source="lightweight_forward",
                )
            )
            return
        if isinstance(item, Mapping):
            for key, inner in item.items():
                visit(inner, f"{name}.{key}")
            return
        if isinstance(item, (list, tuple)):
            for index, inner in enumerate(item):
                visit(inner, f"{name}[{index}]")

    visit(value, prefix)
    return tuple(tensors)


def _select_modules(
    profile: ModelProfile,
    module_range: tuple[int, int] | slice | Sequence[str],
) -> tuple[ModuleProfile, ...]:
    if isinstance(module_range, slice):
        selected = profile.modules[module_range]
    elif isinstance(module_range, tuple) and len(module_range) == 2:
        start, stop = module_range
        selected = profile.modules[start:stop]
    else:
        wanted = list(module_range)
        by_id = {module.module_id: module for module in profile.modules}
        selected = tuple(by_id[module_id] for module_id in wanted)
    return tuple(selected)


def _activation_bytes_for_module(
    module: ModuleProfile,
    config: MemoryEstimationConfig,
) -> int | None:
    if module.output_tensors:
        if config.activation_dtype is not None:
            total = 0
            for tensor in module.output_tensors:
                numel = _tensor_numel(tensor.shape)
                size = dtype_bytes(config.activation_dtype)
                if numel is None or size is None:
                    return module.memory.activation_bytes
                total += numel * size
            return total
        total = 0
        for tensor in module.output_tensors:
            if tensor.estimated_bytes is None:
                return module.memory.activation_bytes
            total += tensor.estimated_bytes
        return total
    return module.memory.activation_bytes


def _temporary_bytes_for_module(
    module: ModuleProfile,
    config: MemoryEstimationConfig,
) -> int:
    if (
        module.memory.temporary_bytes is not None
        and config.temporary_buffer_factor <= 0
    ):
        return module.memory.temporary_bytes
    activation = _activation_bytes_for_module(module, config) or 0
    estimated = ceil(activation * config.temporary_buffer_factor)
    if module.memory.temporary_bytes is None:
        return estimated
    return max(module.memory.temporary_bytes, estimated)


def _gradient_bytes(
    trainable_parameter_count: int,
    trainable_parameter_bytes: int,
    config: MemoryEstimationConfig,
) -> int | None:
    if config.gradient_dtype is None:
        return trainable_parameter_bytes
    size = dtype_bytes(config.gradient_dtype)
    if size is None:
        return None
    return trainable_parameter_count * size


def _optimizer_bytes(
    parameter_count: int,
    parameter_bytes: int,
    trainable_parameter_count: int,
    trainable_parameter_bytes: int,
    config: MemoryEstimationConfig,
) -> int | None:
    optimizer = config.optimizer_type.strip().lower()
    master_weight_bytes = 0
    if config.master_weight_dtype is not None:
        master_size = dtype_bytes(config.master_weight_dtype)
        if master_size is None:
            return None
        master_weight_bytes = trainable_parameter_count * master_size
    if optimizer in {"adam", "adamw"}:
        state_size = dtype_bytes(config.optimizer_state_dtype)
        if state_size is None:
            return None
        return trainable_parameter_count * state_size * 2 + master_weight_bytes
    if optimizer == "sgd":
        momentum = float(config.optimizer_kwargs.get("momentum", 0.0) or 0.0)
        if momentum <= 0:
            return master_weight_bytes
        state_size = dtype_bytes(config.optimizer_state_dtype)
        if state_size is None:
            return None
        return trainable_parameter_count * state_size + master_weight_bytes
    if optimizer in {"adafactor", "lion", "8bit-adam"}:
        return None
    del parameter_count, parameter_bytes, trainable_parameter_bytes
    return None


def _tensor_numel(shape: Sequence[int | str]) -> int | None:
    total = 1
    for dim in shape:
        if not isinstance(dim, int) or dim < 0:
            return None
        total *= dim
    return total


def _worker_usable_bytes(worker: WorkerResource) -> int | None:
    memory_mb = worker.gpu_total_memory
    if memory_mb is None:
        memory_mb = worker.gpu_free_memory
    if memory_mb is None:
        return None
    return int(memory_mb) * 1024 * 1024


def _shared_parameter_groups(model: "nn.Module") -> tuple[tuple[str, ...], ...]:
    groups: dict[int, list[str]] = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        groups.setdefault(int(parameter.data_ptr()), []).append(name)
    shared = [tuple(names) for names in groups.values() if len(names) > 1]
    return tuple(sorted(shared))


def _profile_id(engine_id: str, model_name: str) -> str:
    slug = model_name.strip().lower().replace(" ", "-").replace("/", "-")
    return f"{engine_id}:{slug}"
