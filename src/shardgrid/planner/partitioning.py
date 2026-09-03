"""Automatic partition boundary discovery and candidate generation for T110."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Callable, Iterable, Mapping, Sequence

from shardgrid.engines.models import (
    AutomaticPartitionSupport,
    BoundaryTensorSpec,
    CommunicationEdge,
    EstimateKind,
    ModelProfile,
    PartitionBoundary,
    PartitionSupportStatus,
    TensorMetadata,
    TrainingMemoryEstimate,
)
from shardgrid.planner.memory import MemoryEstimationConfig, estimate_stage_memory
from shardgrid.planner.requirements import (
    ConstraintViolation,
    FeasibilityStatus,
    validate_partition_boundary,
)


@dataclass(frozen=True)
class StagePartition:
    stage_id: str
    module_ids: tuple[str, ...]
    module_paths: tuple[str, ...]
    start_index: int
    stop_index: int
    boundary_before_id: str | None = None
    boundary_after_id: str | None = None
    parameter_names_or_ranges: tuple[str, ...] = ()
    parameter_bytes: int = 0
    gradient_bytes: int | None = None
    activation_bytes: int | None = None
    estimated_compute_units: int = 0
    estimated_peak_training_memory: TrainingMemoryEstimate = TrainingMemoryEstimate()
    required_runtime: str | None = None
    required_backends: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.stage_id.strip():
            raise ValueError("stage_id must be a non-empty string")
        if not self.module_ids:
            raise ValueError("stage must include at least one module")
        if self.start_index < 0 or self.stop_index <= self.start_index:
            raise ValueError("stage indices must define a non-empty forward range")
        if self.parameter_bytes < 0 or self.estimated_compute_units < 0:
            raise ValueError("stage counters must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "module_ids": list(self.module_ids),
            "module_paths": list(self.module_paths),
            "start_index": self.start_index,
            "stop_index": self.stop_index,
            "boundary_before_id": self.boundary_before_id,
            "boundary_after_id": self.boundary_after_id,
            "parameter_names_or_ranges": list(self.parameter_names_or_ranges),
            "parameter_bytes": self.parameter_bytes,
            "gradient_bytes": self.gradient_bytes,
            "activation_bytes": self.activation_bytes,
            "estimated_compute_units": self.estimated_compute_units,
            "estimated_peak_training_memory": self.estimated_peak_training_memory.to_dict(),
            "required_runtime": self.required_runtime,
            "required_backends": list(self.required_backends),
        }


@dataclass(frozen=True)
class StageCommunicationEdge:
    source_stage_id: str
    target_stage_id: str
    source_module_id: str
    target_module_id: str
    activation: tuple[TensorMetadata, ...] = ()
    gradient: tuple[TensorMetadata, ...] = ()
    estimated_bytes_per_step: int | None = None
    estimate_kind: EstimateKind = EstimateKind.UNKNOWN

    def __post_init__(self) -> None:
        if not self.source_stage_id.strip() or not self.target_stage_id.strip():
            raise ValueError("stage edge ids must be non-empty")
        if not self.source_module_id.strip() or not self.target_module_id.strip():
            raise ValueError("module edge ids must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_stage_id": self.source_stage_id,
            "target_stage_id": self.target_stage_id,
            "source_module_id": self.source_module_id,
            "target_module_id": self.target_module_id,
            "activation": [item.to_dict() for item in self.activation],
            "gradient": [item.to_dict() for item in self.gradient],
            "estimated_bytes_per_step": self.estimated_bytes_per_step,
            "estimate_kind": self.estimate_kind.value,
        }


@dataclass(frozen=True)
class CandidateValidationResult:
    candidate_id: str
    valid: bool
    status: FeasibilityStatus
    violations: tuple[ConstraintViolation, ...] = ()


@dataclass(frozen=True)
class PartitionCandidate:
    candidate_id: str
    model_profile_id: str
    stage_count: int
    stages: tuple[StagePartition, ...]
    communication_edges: tuple[StageCommunicationEdge, ...]
    estimated_bytes_per_step: int | None
    required_worker_count: int
    hard_constraint_status: FeasibilityStatus
    rejection_reasons: tuple[str, ...] = ()
    engine_id: str | None = None
    required_runtime: str | None = None
    required_backends: tuple[str, ...] = ()
    original_engine_plan_ref: str | None = None
    selected_boundary_ids: tuple[str, ...] = ()
    score_breakdown: Mapping[str, int | str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "model_profile_id": self.model_profile_id,
            "stage_count": self.stage_count,
            "stages": [stage.to_dict() for stage in self.stages],
            "communication_edges": [edge.to_dict() for edge in self.communication_edges],
            "estimated_bytes_per_step": self.estimated_bytes_per_step,
            "required_worker_count": self.required_worker_count,
            "hard_constraint_status": self.hard_constraint_status.value,
            "rejection_reasons": list(self.rejection_reasons),
            "engine_id": self.engine_id,
            "required_runtime": self.required_runtime,
            "required_backends": list(self.required_backends),
            "original_engine_plan_ref": self.original_engine_plan_ref,
            "selected_boundary_ids": list(self.selected_boundary_ids),
            "score_breakdown": dict(self.score_breakdown),
        }


@dataclass(frozen=True)
class PartitionGenerationResult:
    model_profile_id: str
    engine_id: str
    status: FeasibilityStatus
    candidates: tuple[PartitionCandidate, ...] = ()
    partition_support: AutomaticPartitionSupport | None = None
    reasons: tuple[str, ...] = ()
    diagnostics: tuple[str, ...] = ()


def discover_partition_support(
    model: Any,
    profile: ModelProfile,
    *,
    sample_args: Sequence[Any] = (),
    sample_kwargs: Mapping[str, Any] | None = None,
    source: str = "torch.fx.symbolic_trace",
) -> AutomaticPartitionSupport:
    sample_kwargs = dict(sample_kwargs or {})
    try:
        dependencies, custom_op_reasons = _symbolic_module_dependencies(
            model,
        )
    except Exception as exc:
        if _export_supports_model(
            model,
            sample_args=sample_args,
            sample_kwargs=sample_kwargs,
        ):
            dependencies = _adjacent_module_dependencies(profile)
            custom_op_reasons = ()
        else:
            return AutomaticPartitionSupport(
                engine_id=profile.engine_id,
                status=PartitionSupportStatus.UNSUPPORTED,
                supported_runtime=profile.required_runtime,
                supported_backends=profile.required_backends,
                reasons=(f"untraceable graph: {type(exc).__name__}: {exc}",),
            )

    if custom_op_reasons:
        return AutomaticPartitionSupport(
            engine_id=profile.engine_id,
            status=PartitionSupportStatus.UNSUPPORTED,
            supported_runtime=profile.required_runtime,
            supported_backends=profile.required_backends,
            reasons=tuple(custom_op_reasons),
        )

    module_ids_by_path = {module.module_path: module.module_id for module in profile.modules}
    ordered_paths = [module.module_path for module in profile.modules]
    shared_groups = profile.shared_parameter_groups
    boundaries: list[PartitionBoundary] = []
    for split_index in range(len(ordered_paths) - 1):
        left_paths = tuple(ordered_paths[: split_index + 1])
        right_paths = tuple(ordered_paths[split_index + 1 :])
        cross_dependencies = sorted(
            dep for dep in dependencies if dep[0] in left_paths and dep[1] in right_paths
        )
        shared_names = tuple(
            name
            for group in shared_groups
            for name in group
            if _shared_group_crosses(group, left_paths, right_paths)
        )
        reasons: list[str] = []
        status = PartitionSupportStatus.SUPPORTED
        if not cross_dependencies:
            status = PartitionSupportStatus.UNSUPPORTED
            reasons.append("no traced dependency crosses this boundary")
        if shared_names:
            status = PartitionSupportStatus.UNSUPPORTED
            reasons.append("shared/tied parameter crosses boundary")
        source_path = ordered_paths[split_index]
        target_path = ordered_paths[split_index + 1]
        boundary_tensors = _boundary_tensors_for_dependencies(
            cross_dependencies, profile, module_ids_by_path, source
        )
        boundaries.append(
            PartitionBoundary(
                boundary_id=f"b{split_index:04d}",
                source_module=source_path,
                target_module=target_path,
                parameter_owners=tuple(module_ids_by_path[path] for path in left_paths),
                shared_parameter_names=shared_names,
                boundary_tensors=boundary_tensors,
                forward_dependencies=tuple(f"{src}->{dst}" for src, dst in cross_dependencies),
                backward_dependencies=tuple(f"{dst}->{src}" for src, dst in cross_dependencies),
                status=status,
                reasons=tuple(reasons),
            )
        )
    return AutomaticPartitionSupport(
        engine_id=profile.engine_id,
        status=PartitionSupportStatus.SUPPORTED,
        supported_runtime=profile.required_runtime,
        supported_backends=profile.required_backends,
        boundaries=tuple(boundaries),
    )


def generate_partition_candidates(
    profile: ModelProfile,
    *,
    partition_support: AutomaticPartitionSupport | None = None,
    memory_config: MemoryEstimationConfig | None = None,
    original_engine_plan_ref: str | None = None,
    min_stage_count: int = 2,
    max_stage_count: int | None = None,
    max_candidates: int = 128,
) -> PartitionGenerationResult:
    support = partition_support or profile.partition_support
    if support is None:
        return PartitionGenerationResult(
            model_profile_id=profile.profile_id,
            engine_id=profile.engine_id,
            status=FeasibilityStatus.UNSUPPORTED,
            reasons=("partition support metadata is required",),
        )
    if support.status is not PartitionSupportStatus.SUPPORTED:
        return PartitionGenerationResult(
            model_profile_id=profile.profile_id,
            engine_id=profile.engine_id,
            status=_support_to_feasibility(support.status),
            partition_support=support,
            reasons=support.reasons,
        )

    boundaries = tuple(support.boundaries)
    if not boundaries:
        return PartitionGenerationResult(
            model_profile_id=profile.profile_id,
            engine_id=profile.engine_id,
            status=FeasibilityStatus.UNSUPPORTED,
            partition_support=support,
            reasons=("no partition boundaries available",),
        )

    upper_stage_count = max_stage_count or min(len(profile.modules), len(boundaries) + 1)
    candidates: list[PartitionCandidate] = []
    for stage_count in range(min_stage_count, upper_stage_count + 1):
        split_count = stage_count - 1
        for split_indices in combinations(range(len(boundaries)), split_count):
            selected_boundaries = tuple(boundaries[index] for index in split_indices)
            candidate = _build_candidate(
                profile,
                support,
                selected_boundaries,
                memory_config=memory_config or MemoryEstimationConfig(),
                original_engine_plan_ref=original_engine_plan_ref,
            )
            candidates.append(candidate)
            if len(candidates) >= max_candidates:
                break
        if len(candidates) >= max_candidates:
            break

    if not candidates:
        return PartitionGenerationResult(
            model_profile_id=profile.profile_id,
            engine_id=profile.engine_id,
            status=FeasibilityStatus.UNSUPPORTED,
            partition_support=support,
            reasons=("no stage combinations can be generated from discovered boundaries",),
        )

    ordered = tuple(
        sorted(
            candidates,
            key=lambda item: (
                item.stage_count,
                item.selected_boundary_ids,
                item.candidate_id,
            ),
        )
    )
    status = (
        FeasibilityStatus.FEASIBLE
        if any(candidate.hard_constraint_status is FeasibilityStatus.FEASIBLE for candidate in ordered)
        else FeasibilityStatus.INFEASIBLE
    )
    reasons = ()
    if status is not FeasibilityStatus.FEASIBLE:
        reasons = tuple(
            sorted(
                {
                    reason
                    for candidate in ordered
                    for reason in candidate.rejection_reasons
                }
            )
        )
    return PartitionGenerationResult(
        model_profile_id=profile.profile_id,
        engine_id=profile.engine_id,
        status=status,
        candidates=ordered,
        partition_support=support,
        reasons=reasons,
    )


def validate_partition_candidate(
    profile: ModelProfile,
    candidate: PartitionCandidate,
    support: AutomaticPartitionSupport,
) -> CandidateValidationResult:
    violations: list[ConstraintViolation] = []
    if candidate.stage_count != len(candidate.stages):
        violations.append(
            ConstraintViolation(
                "stage_count",
                "candidate stage_count does not match number of stages",
            )
        )

    covered_module_ids: list[str] = []
    covered_parameter_names: list[str] = []
    expected_modules = [module.module_id for module in profile.modules]
    current_index = 0
    for stage in candidate.stages:
        if stage.start_index != current_index:
            violations.append(
                ConstraintViolation(
                    "stage_order",
                    "candidate stages are not contiguous and ordered",
                )
            )
        covered_module_ids.extend(stage.module_ids)
        covered_parameter_names.extend(stage.parameter_names_or_ranges)
        current_index = stage.stop_index
        estimate = stage.estimated_peak_training_memory
        if (
            estimate.planner_required_bytes is None
            or estimate.estimated_peak_bytes is None
        ):
            violations.append(
                ConstraintViolation(
                    "memory_metadata",
                    f"stage {stage.stage_id} does not have complete memory metadata",
                )
            )

    if tuple(covered_module_ids) != tuple(expected_modules):
        violations.append(
            ConstraintViolation(
                "module_coverage",
                "candidate module ranges do not cover the full model exactly once",
            )
        )

    expected_parameters = sorted(
        name for module in profile.modules for name in module.parameter_names
    )
    if sorted(covered_parameter_names) != expected_parameters:
        violations.append(
            ConstraintViolation(
                "parameter_coverage",
                "candidate parameters do not cover the full model exactly once",
            )
        )

    boundary_map = {boundary.boundary_id: boundary for boundary in support.boundaries}
    for boundary_id in candidate.selected_boundary_ids:
        boundary = boundary_map[boundary_id]
        result = validate_partition_boundary(boundary, support)
        if not result.feasible:
            violations.extend(result.violations)

    if any(
        edge.source_stage_id == edge.target_stage_id
        for edge in candidate.communication_edges
    ):
        violations.append(
            ConstraintViolation(
                "communication_edges",
                "communication edges must cross stage boundaries",
            )
        )

    status = FeasibilityStatus.FEASIBLE if not violations else FeasibilityStatus.INFEASIBLE
    return CandidateValidationResult(
        candidate_id=candidate.candidate_id,
        valid=not violations,
        status=status,
        violations=tuple(violations),
    )


def build_partition_profile(
    model: Any,
    profile: ModelProfile,
    *,
    sample_args: Sequence[Any] = (),
    sample_kwargs: Mapping[str, Any] | None = None,
    memory_config: MemoryEstimationConfig | None = None,
    original_engine_plan_ref: str | None = None,
    min_stage_count: int = 2,
    max_stage_count: int | None = None,
) -> PartitionGenerationResult:
    support = discover_partition_support(
        model,
        profile,
        sample_args=sample_args,
        sample_kwargs=sample_kwargs,
    )
    return generate_partition_candidates(
        _with_partition_support(profile, support),
        partition_support=support,
        memory_config=memory_config,
        original_engine_plan_ref=original_engine_plan_ref,
        min_stage_count=min_stage_count,
        max_stage_count=max_stage_count,
    )


def _build_candidate(
    profile: ModelProfile,
    support: AutomaticPartitionSupport,
    selected_boundaries: Sequence[PartitionBoundary],
    *,
    memory_config: MemoryEstimationConfig,
    original_engine_plan_ref: str | None,
) -> PartitionCandidate:
    split_positions = [int(boundary.boundary_id[1:]) + 1 for boundary in selected_boundaries]
    boundaries_by_position = {int(boundary.boundary_id[1:]) + 1: boundary for boundary in selected_boundaries}
    stops = split_positions + [len(profile.modules)]
    starts = [0] + split_positions
    stages: list[StagePartition] = []
    module_to_stage: dict[str, str] = {}
    for stage_index, (start, stop) in enumerate(zip(starts, stops, strict=True)):
        modules = profile.modules[start:stop]
        boundary_before = boundaries_by_position.get(start)
        boundary_after = boundaries_by_position.get(stop)
        estimate = estimate_stage_memory(profile, (start, stop), memory_config)
        stage_id = f"stage{stage_index}"
        for module in modules:
            module_to_stage[module.module_id] = stage_id
        stages.append(
            StagePartition(
                stage_id=stage_id,
                module_ids=tuple(module.module_id for module in modules),
                module_paths=tuple(module.module_path for module in modules),
                start_index=start,
                stop_index=stop,
                boundary_before_id=None if boundary_before is None else boundary_before.boundary_id,
                boundary_after_id=None if boundary_after is None else boundary_after.boundary_id,
                parameter_names_or_ranges=tuple(
                    name for module in modules for name in module.parameter_names
                ),
                parameter_bytes=sum(module.parameter_bytes for module in modules),
                gradient_bytes=estimate.gradient_bytes,
                activation_bytes=estimate.activation_bytes,
                estimated_compute_units=sum(
                    module.trainable_parameter_count for module in modules
                ),
                estimated_peak_training_memory=estimate,
                required_runtime=profile.required_runtime,
                required_backends=profile.required_backends,
            )
        )

    candidate_edges = _candidate_communication_edges(
        profile,
        support,
        module_to_stage,
    )
    validation = validate_partition_candidate(
        profile,
        PartitionCandidate(
            candidate_id=_candidate_id(profile, stages),
            model_profile_id=profile.profile_id,
            stage_count=len(stages),
            stages=tuple(stages),
            communication_edges=candidate_edges,
            estimated_bytes_per_step=_sum_edge_bytes(candidate_edges),
            required_worker_count=len(stages),
            hard_constraint_status=FeasibilityStatus.FEASIBLE,
            engine_id=profile.engine_id,
            required_runtime=profile.required_runtime,
            required_backends=profile.required_backends,
            original_engine_plan_ref=original_engine_plan_ref,
            selected_boundary_ids=tuple(boundary.boundary_id for boundary in selected_boundaries),
            score_breakdown={"stage_count": len(stages)},
        ),
        support,
    )
    reasons = tuple(violation.reason for violation in validation.violations)
    return PartitionCandidate(
        candidate_id=_candidate_id(profile, stages),
        model_profile_id=profile.profile_id,
        stage_count=len(stages),
        stages=tuple(stages),
        communication_edges=candidate_edges,
        estimated_bytes_per_step=_sum_edge_bytes(candidate_edges),
        required_worker_count=len(stages),
        hard_constraint_status=validation.status,
        rejection_reasons=reasons,
        engine_id=profile.engine_id,
        required_runtime=profile.required_runtime,
        required_backends=profile.required_backends,
        original_engine_plan_ref=original_engine_plan_ref,
        selected_boundary_ids=tuple(boundary.boundary_id for boundary in selected_boundaries),
        score_breakdown={"stage_count": len(stages)},
    )


def _candidate_communication_edges(
    profile: ModelProfile,
    support: AutomaticPartitionSupport,
    module_to_stage: Mapping[str, str],
) -> tuple[StageCommunicationEdge, ...]:
    module_by_path = {module.module_path: module for module in profile.modules}
    module_id_by_path = {module.module_path: module.module_id for module in profile.modules}
    seen: dict[tuple[str, str, str, str], StageCommunicationEdge] = {}
    dependency_pairs = sorted(
        {
            pair
            for boundary in support.boundaries
            for pair in _parse_dependency_pairs(boundary.forward_dependencies)
        }
    )
    for source_path, target_path in dependency_pairs:
        source_module = module_by_path[source_path]
        target_module = module_by_path[target_path]
        source_stage_id = module_to_stage[source_module.module_id]
        target_stage_id = module_to_stage[target_module.module_id]
        if source_stage_id == target_stage_id:
            continue
        activation = tuple(
            TensorMetadata(
                name=f"{source_path}->{target_path}:{tensor.name}",
                shape=tensor.shape,
                dtype=tensor.dtype,
                estimated_bytes=tensor.estimated_bytes,
                estimate_kind=tensor.estimate_kind,
                source=tensor.source,
            )
            for tensor in source_module.output_tensors
        )
        gradient = activation
        seen[(source_stage_id, target_stage_id, source_path, target_path)] = StageCommunicationEdge(
            source_stage_id=source_stage_id,
            target_stage_id=target_stage_id,
            source_module_id=module_id_by_path[source_path],
            target_module_id=module_id_by_path[target_path],
            activation=activation,
            gradient=gradient,
            estimated_bytes_per_step=_tensor_bytes(activation) * 2
            if _tensor_bytes(activation) is not None
            else None,
            estimate_kind=EstimateKind.MEASURED if activation else EstimateKind.UNKNOWN,
        )
    return tuple(
        sorted(
            seen.values(),
            key=lambda edge: (
                edge.source_stage_id,
                edge.target_stage_id,
                edge.source_module_id,
                edge.target_module_id,
            ),
        )
    )


def _trace_module_dependencies(
    model: Any,
) -> tuple[tuple[tuple[str, str], ...], tuple[str, ...]]:
    from torch.fx import Node, symbolic_trace

    traced = symbolic_trace(model)
    module_targets = {str(name) for name, _module in traced.named_modules() if name}
    dependencies: set[tuple[str, str]] = set()
    custom_reasons: set[str] = set()

    def source_modules(value: Any) -> set[str]:
        if isinstance(value, Node):
            if value.op == "call_module" and str(value.target) in module_targets:
                return {str(value.target)}
            nested: set[str] = set()
            nested.update(source_modules(value.args))
            nested.update(source_modules(value.kwargs))
            return nested
        if isinstance(value, Mapping):
            result: set[str] = set()
            for item in value.values():
                result.update(source_modules(item))
            return result
        if isinstance(value, (list, tuple)):
            result: set[str] = set()
            for item in value:
                result.update(source_modules(item))
            return result
        return set()

    for node in traced.graph.nodes:
        if node.op == "call_function":
            if not _is_allowed_function(node.target):
                custom_reasons.add(
                    f"unsupported custom op: {_callable_name(node.target)}"
                )
        if node.op != "call_module":
            continue
        current = str(node.target)
        for source in sorted(source_modules(node.args) | source_modules(node.kwargs)):
            if source != current:
                dependencies.add((source, current))

    return tuple(sorted(dependencies)), tuple(sorted(custom_reasons))


def _symbolic_module_dependencies(
    model: Any,
) -> tuple[tuple[tuple[str, str], ...], tuple[str, ...]]:
    return _trace_module_dependencies(model)


def _adjacent_module_dependencies(profile: ModelProfile) -> tuple[tuple[str, str], ...]:
    return tuple(
        (left.module_path, right.module_path)
        for left, right in zip(profile.modules, profile.modules[1:])
    )


def _export_supports_model(
    model: Any,
    *,
    sample_args: Sequence[Any],
    sample_kwargs: Mapping[str, Any],
) -> bool:
    import torch

    try:
        if sample_kwargs:
            torch.export.export(model, args=tuple(sample_args), kwargs=dict(sample_kwargs))
        else:
            torch.export.export(model, tuple(sample_args))
    except Exception:
        return False
    return True


def _is_allowed_function(target: Any) -> bool:
    module = getattr(target, "__module__", "")
    name = getattr(target, "__name__", "")
    if module.startswith("torch"):
        return True
    if module in {"operator", "_operator", "builtins", "math"}:
        return True
    if "VariableFunctionsClass" in type(target).__qualname__:
        return True
    if name in {"getitem", "getattr"}:
        return True
    return False


def _callable_name(target: Any) -> str:
    module = getattr(target, "__module__", "")
    name = getattr(target, "__qualname__", None) or getattr(target, "__name__", repr(target))
    return f"{module}.{name}".strip(".")


def _shared_group_crosses(
    group: Sequence[str],
    left_paths: Sequence[str],
    right_paths: Sequence[str],
) -> bool:
    return any(_param_in_paths(name, left_paths) for name in group) and any(
        _param_in_paths(name, right_paths) for name in group
    )


def _param_in_paths(parameter_name: str, module_paths: Sequence[str]) -> bool:
    for path in module_paths:
        if parameter_name == path or parameter_name.startswith(f"{path}."):
            return True
    return False


def _boundary_tensors_for_dependencies(
    dependencies: Sequence[tuple[str, str]],
    profile: ModelProfile,
    module_ids_by_path: Mapping[str, str],
    source: str,
) -> tuple[BoundaryTensorSpec, ...]:
    module_by_id = {module.module_id: module for module in profile.modules}
    tensors: list[BoundaryTensorSpec] = []
    seen: set[tuple[str, tuple[int | str, ...], str | None]] = set()
    for source_path, target_path in dependencies:
        module_id = module_ids_by_path[source_path]
        module = module_by_id[module_id]
        for tensor in module.output_tensors:
            key = (
                f"{source_path}->{target_path}:{tensor.name}",
                tensor.shape,
                tensor.dtype,
            )
            if key in seen:
                continue
            seen.add(key)
            tensors.append(
                BoundaryTensorSpec(
                    name=key[0],
                    shape=tensor.shape,
                    dtype=tensor.dtype,
                )
            )
    del source
    return tuple(tensors)


def _parse_dependency_pairs(values: Iterable[str]) -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    for value in values:
        source, _, target = value.partition("->")
        if source and target:
            pairs.append((source, target))
    return tuple(pairs)


def _sum_edge_bytes(edges: Sequence[StageCommunicationEdge]) -> int | None:
    total = 0
    for edge in edges:
        if edge.estimated_bytes_per_step is None:
            return None
        total += edge.estimated_bytes_per_step
    return total


def _tensor_bytes(tensors: Sequence[TensorMetadata]) -> int | None:
    total = 0
    for tensor in tensors:
        if tensor.estimated_bytes is None:
            return None
        total += tensor.estimated_bytes
    return total


def _candidate_id(profile: ModelProfile, stages: Sequence[StagePartition]) -> str:
    ranges = "-".join(f"{stage.start_index}:{stage.stop_index}" for stage in stages)
    return f"{profile.profile_id}:{ranges}"


def _with_partition_support(
    profile: ModelProfile,
    support: AutomaticPartitionSupport,
) -> ModelProfile:
    return ModelProfile(
        profile_id=profile.profile_id,
        engine_id=profile.engine_id,
        model_name=profile.model_name,
        modules=profile.modules,
        module_order=profile.module_order,
        partition_support=support,
        communication_edges=profile.communication_edges,
        shared_parameter_groups=profile.shared_parameter_groups,
        required_runtime=profile.required_runtime,
        required_backends=profile.required_backends,
        total_memory=profile.total_memory,
        evidence_paths=profile.evidence_paths,
        diagnostics=profile.diagnostics,
    )


def _support_to_feasibility(status: PartitionSupportStatus) -> FeasibilityStatus:
    if status is PartitionSupportStatus.SUPPORTED:
        return FeasibilityStatus.FEASIBLE
    if status is PartitionSupportStatus.UNSUPPORTED:
        return FeasibilityStatus.UNSUPPORTED
    return FeasibilityStatus.INFEASIBLE
