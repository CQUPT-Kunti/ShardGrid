"""Automatic partition boundary discovery and candidate generation for T110."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from shardgrid.engines.models import (
    AutomaticPartitionSupport,
    BoundaryTensorSpec,
    EstimateKind,
    ModelProfile,
    PartitionBoundary,
    PartitionSupportStatus,
    TensorMetadata,
    TrainingMemoryEstimate,
)
from shardgrid.planner.generic_graph import (
    FXGraphCaptureAdapter,
    module_dependencies_from_graph,
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


@dataclass(frozen=True)
class _DependencyEdge:
    source_path: str
    target_path: str
    source_module_id: str
    target_module_id: str
    activation: tuple[TensorMetadata, ...]
    gradient: tuple[TensorMetadata, ...]
    estimated_bytes_per_step: int | None
    estimate_kind: EstimateKind


def discover_partition_support(
    model: Any,
    profile: ModelProfile,
    *,
    sample_args: Sequence[Any] = (),
    sample_kwargs: Mapping[str, Any] | None = None,
    source: str = "torch.fx.symbolic_trace",
) -> AutomaticPartitionSupport:
    sample_kwargs = dict(sample_kwargs or {})
    target_paths = {module.module_path for module in profile.modules}
    ordered_paths: tuple[str, ...]
    fx_error: str | None = None
    try:
        dependencies, traced_order, custom_reasons = _symbolic_module_dependencies(model)
        trace_source = source
    except Exception as fx_exc:
        fx_error = f"torch.fx failed with {type(fx_exc).__name__}: {fx_exc}"
    else:
        try:
            ordered_paths = _normalize_traced_order(profile, traced_order)
        except ValueError as exc:
            fx_error = f"torch.fx order was incomplete ({exc})"
        else:
            fx_error = None

    if fx_error is not None:
        try:
            dependencies, traced_order, custom_reasons = _export_module_dependencies(
                model,
                sample_args=sample_args,
                sample_kwargs=sample_kwargs,
                target_paths=target_paths,
            )
            trace_source = "torch.export.export"
            ordered_paths = _normalize_traced_order(profile, traced_order)
        except Exception as export_exc:
            return AutomaticPartitionSupport(
                engine_id=profile.engine_id,
                status=PartitionSupportStatus.UNSUPPORTED,
                supported_runtime=profile.required_runtime,
                supported_backends=profile.required_backends,
                reasons=(
                    "untraceable graph: "
                    f"{fx_error}; "
                    f"torch.export failed with {type(export_exc).__name__}: {export_exc}",
                ),
            )

    if custom_reasons:
        return AutomaticPartitionSupport(
            engine_id=profile.engine_id,
            status=PartitionSupportStatus.UNSUPPORTED,
            supported_runtime=profile.required_runtime,
            supported_backends=profile.required_backends,
            reasons=tuple(custom_reasons),
        )

    module_ids_by_path = {module.module_path: module.module_id for module in profile.modules}
    boundaries: list[PartitionBoundary] = []
    for split_index in range(len(ordered_paths) - 1):
        left_paths = tuple(ordered_paths[: split_index + 1])
        right_paths = tuple(ordered_paths[split_index + 1 :])
        cross_dependencies = sorted(
            dep for dep in dependencies if dep[0] in left_paths and dep[1] in right_paths
        )
        shared_names = tuple(
            name
            for group in profile.shared_parameter_groups
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
        boundaries.append(
            PartitionBoundary(
                boundary_id=f"b{split_index:04d}",
                source_module=ordered_paths[split_index],
                target_module=ordered_paths[split_index + 1],
                parameter_owners=tuple(module_ids_by_path[path] for path in left_paths),
                shared_parameter_names=shared_names,
                boundary_tensors=_boundary_tensors_for_dependencies(
                    cross_dependencies,
                    profile,
                    module_ids_by_path,
                    trace_source,
                ),
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
    usable_memory_bytes: Sequence[int] | int | None = None,
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

    try:
        ordered_modules = _ordered_modules_for_support(profile, support)
    except ValueError as exc:
        return PartitionGenerationResult(
            model_profile_id=profile.profile_id,
            engine_id=profile.engine_id,
            status=FeasibilityStatus.UNSUPPORTED,
            partition_support=support,
            reasons=(str(exc),),
        )

    if not support.boundaries:
        return PartitionGenerationResult(
            model_profile_id=profile.profile_id,
            engine_id=profile.engine_id,
            status=FeasibilityStatus.UNSUPPORTED,
            partition_support=support,
            reasons=("no partition boundaries available",),
        )

    capacities = _normalize_capacity_bytes(usable_memory_bytes)
    if capacities and len(capacities) < min_stage_count:
        return PartitionGenerationResult(
            model_profile_id=profile.profile_id,
            engine_id=profile.engine_id,
            status=FeasibilityStatus.INFEASIBLE,
            partition_support=support,
            reasons=("insufficient usable-memory capacity slots for requested stage count",),
        )

    upper_stage_count = max_stage_count or min(len(ordered_modules), len(support.boundaries) + 1)
    if capacities:
        upper_stage_count = min(upper_stage_count, len(capacities))

    memory = memory_config or MemoryEstimationConfig()
    dependency_edges = _dependency_edges(profile, support)
    stage_estimates = _precompute_stage_estimates(profile, ordered_modules, memory)
    stage_weights = tuple(_partition_weight_bytes(module) for module in ordered_modules)
    internal_edge_bytes = _precompute_internal_edge_bytes(ordered_modules, dependency_edges)

    candidates: list[PartitionCandidate] = []
    failures: list[str] = []
    for stage_count in range(min_stage_count, upper_stage_count + 1):
        ranges = _plan_stage_ranges(
            ordered_modules,
            support.boundaries,
            stage_estimates,
            stage_weights,
            internal_edge_bytes,
            stage_count=stage_count,
            capacities=capacities[:stage_count] if capacities else (),
        )
        if ranges is None:
            failures.append(f"unable to derive a contiguous {stage_count}-stage partition")
            continue
        candidates.append(
            _build_candidate(
                profile,
                support,
                ordered_modules,
                dependency_edges,
                ranges,
                memory_config=memory,
                original_engine_plan_ref=original_engine_plan_ref,
                stage_capacities=capacities[:stage_count] if capacities else (),
            )
        )
        if len(candidates) >= max_candidates:
            break

    if not candidates:
        return PartitionGenerationResult(
            model_profile_id=profile.profile_id,
            engine_id=profile.engine_id,
            status=FeasibilityStatus.INFEASIBLE,
            partition_support=support,
            reasons=tuple(sorted(set(failures))) or (
                "no stage candidates can be generated from discovered boundaries",
            ),
        )

    ordered = tuple(sorted(candidates, key=_candidate_sort_key))
    status = (
        FeasibilityStatus.FEASIBLE
        if any(
            candidate.hard_constraint_status is FeasibilityStatus.FEASIBLE
            for candidate in ordered
        )
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

    try:
        expected_modules = _ordered_modules_for_support(profile, support)
    except ValueError as exc:
        violations.append(ConstraintViolation("module_order", str(exc)))
        expected_modules = profile.modules

    covered_module_ids: list[str] = []
    covered_parameter_names: list[str] = []
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
        if estimate.planner_required_bytes is None or estimate.estimated_peak_bytes is None:
            violations.append(
                ConstraintViolation(
                    "memory_metadata",
                    f"stage {stage.stage_id} does not have complete memory metadata",
                )
            )

    if tuple(covered_module_ids) != tuple(module.module_id for module in expected_modules):
        violations.append(
            ConstraintViolation(
                "module_coverage",
                "candidate module ranges do not cover the traced module order exactly once",
            )
        )

    expected_parameters = sorted(
        name for module in expected_modules for name in module.parameter_names
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
        result = validate_partition_boundary(boundary_map[boundary_id], support)
        if not result.feasible:
            violations.extend(result.violations)

    if any(edge.source_stage_id == edge.target_stage_id for edge in candidate.communication_edges):
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
    usable_memory_bytes: Sequence[int] | int | None = None,
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
        usable_memory_bytes=usable_memory_bytes,
        min_stage_count=min_stage_count,
        max_stage_count=max_stage_count,
    )


def _build_candidate(
    profile: ModelProfile,
    support: AutomaticPartitionSupport,
    ordered_modules: Sequence[Any],
    dependency_edges: Sequence[_DependencyEdge],
    ranges: Sequence[tuple[int, int]],
    *,
    memory_config: MemoryEstimationConfig,
    original_engine_plan_ref: str | None,
    stage_capacities: Sequence[int],
) -> PartitionCandidate:
    boundaries_by_position = {
        index + 1: boundary for index, boundary in enumerate(support.boundaries)
    }
    stages: list[StagePartition] = []
    module_to_stage: dict[str, str] = {}
    for stage_index, (start, stop) in enumerate(ranges):
        modules = ordered_modules[start:stop]
        estimate = estimate_stage_memory(
            profile,
            [module.module_id for module in modules],
            memory_config,
        )
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
                boundary_before_id=(
                    None if start == 0 else boundaries_by_position[start].boundary_id
                ),
                boundary_after_id=(
                    None
                    if stop == len(ordered_modules)
                    else boundaries_by_position[stop].boundary_id
                ),
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

    selected_boundaries = tuple(
        boundaries_by_position[stop]
        for _, stop in ranges
        if stop < len(ordered_modules)
    )
    candidate_edges = _candidate_communication_edges(dependency_edges, module_to_stage)
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
            score_breakdown={},
        ),
        support,
    )

    rejection_reasons = tuple(
        sorted(
            {violation.reason for violation in validation.violations}
            | set(_capacity_reasons(tuple(stages), stage_capacities))
        )
    )
    communication_bytes = _sum_edge_bytes(candidate_edges)
    return PartitionCandidate(
        candidate_id=_candidate_id(profile, stages),
        model_profile_id=profile.profile_id,
        stage_count=len(stages),
        stages=tuple(stages),
        communication_edges=candidate_edges,
        estimated_bytes_per_step=communication_bytes,
        required_worker_count=len(stages),
        hard_constraint_status=(
            FeasibilityStatus.FEASIBLE
            if not rejection_reasons
            else FeasibilityStatus.INFEASIBLE
        ),
        rejection_reasons=rejection_reasons,
        engine_id=profile.engine_id,
        required_runtime=profile.required_runtime,
        required_backends=profile.required_backends,
        original_engine_plan_ref=original_engine_plan_ref,
        selected_boundary_ids=tuple(boundary.boundary_id for boundary in selected_boundaries),
        score_breakdown={
            "stage_count": len(stages),
            "balance_penalty_bytes": _capacity_balance_penalty(tuple(stages), stage_capacities),
            "communication_bytes": communication_bytes or -1,
            "max_stage_required_bytes": max(
                (
                    stage.estimated_peak_training_memory.planner_required_bytes or 0
                    for stage in stages
                ),
                default=0,
            ),
            "memory_shortfall_bytes": _capacity_shortfall(tuple(stages), stage_capacities),
        },
    )


def _candidate_communication_edges(
    dependency_edges: Sequence[_DependencyEdge],
    module_to_stage: Mapping[str, str],
) -> tuple[StageCommunicationEdge, ...]:
    seen: dict[tuple[str, str, str, str], StageCommunicationEdge] = {}
    for edge in dependency_edges:
        source_stage_id = module_to_stage[edge.source_module_id]
        target_stage_id = module_to_stage[edge.target_module_id]
        if source_stage_id == target_stage_id:
            continue
        seen[(source_stage_id, target_stage_id, edge.source_path, edge.target_path)] = (
            StageCommunicationEdge(
                source_stage_id=source_stage_id,
                target_stage_id=target_stage_id,
                source_module_id=edge.source_module_id,
                target_module_id=edge.target_module_id,
                activation=edge.activation,
                gradient=edge.gradient,
                estimated_bytes_per_step=edge.estimated_bytes_per_step,
                estimate_kind=edge.estimate_kind,
            )
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


def _symbolic_module_dependencies(
    model: Any,
) -> tuple[tuple[tuple[str, str], ...], tuple[str, ...], tuple[str, ...]]:
    result = FXGraphCaptureAdapter().capture(model)
    dependencies, ordered = module_dependencies_from_graph(result.canonical_graph)
    return dependencies, ordered, result.diagnostics


def _export_module_dependencies(
    model: Any,
    *,
    sample_args: Sequence[Any],
    sample_kwargs: Mapping[str, Any],
    target_paths: set[str],
) -> tuple[tuple[tuple[str, str], ...], tuple[str, ...], tuple[str, ...]]:
    import torch
    from torch.fx import Node

    exported = (
        torch.export.export(model, args=tuple(sample_args), kwargs=dict(sample_kwargs))
        if sample_kwargs
        else torch.export.export(model, tuple(sample_args))
    )
    dependencies: set[tuple[str, str]] = set()
    custom_reasons: set[str] = set()
    ordered: list[str] = []
    producers: dict[Node, set[str]] = {}

    def walk_nodes(value: Any) -> Iterable[Node]:
        if isinstance(value, Node):
            yield value
            return
        if isinstance(value, Mapping):
            for item in value.values():
                yield from walk_nodes(item)
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                yield from walk_nodes(item)

    for node in exported.graph_module.graph.nodes:
        if node.op == "call_function" and not _is_allowed_function(node.target):
            custom_reasons.add(f"unsupported custom op: {_callable_name(node.target)}")
        input_modules: set[str] = set()
        for parent in walk_nodes(node.args):
            input_modules.update(producers.get(parent, set()))
        for parent in walk_nodes(node.kwargs):
            input_modules.update(producers.get(parent, set()))
        current = _node_module_path(node, target_paths)
        if current is None:
            producers[node] = input_modules
            continue
        if current not in ordered:
            ordered.append(current)
        for source in sorted(input_modules):
            if source != current:
                dependencies.add((source, current))
        producers[node] = {current}

    return tuple(sorted(dependencies)), tuple(ordered), tuple(sorted(custom_reasons))


def _node_module_path(node: Any, target_paths: set[str]) -> str | None:
    target = getattr(node, "target", None)
    if getattr(node, "op", None) == "call_module" and str(target) in target_paths:
        return str(target)
    stack = getattr(node, "meta", {}).get("nn_module_stack")
    if isinstance(stack, Mapping):
        for value in reversed(tuple(stack.values())):
            path = value[0] if isinstance(value, tuple) and value else None
            if path and path in target_paths:
                return str(path)
    return None


def _normalize_traced_order(
    profile: ModelProfile,
    traced_order: Sequence[str],
) -> tuple[str, ...]:
    expected = {module.module_path for module in profile.modules}
    ordered: list[str] = []
    seen: set[str] = set()
    for path in traced_order:
        if path in expected and path not in seen:
            ordered.append(path)
            seen.add(path)
    missing = sorted(expected - seen)
    if missing:
        raise ValueError(
            "traced graph did not include all partitionable modules: "
            + ", ".join(missing)
        )
    return tuple(ordered)


def _ordered_modules_for_support(
    profile: ModelProfile,
    support: AutomaticPartitionSupport,
) -> tuple[Any, ...]:
    module_by_path = {module.module_path: module for module in profile.modules}
    if len(profile.modules) == 1:
        return profile.modules
    if len(support.boundaries) != len(profile.modules) - 1:
        raise ValueError("partition support boundaries do not match traced module order")
    ordered_paths = [support.boundaries[0].source_module]
    ordered_paths.extend(boundary.target_module for boundary in support.boundaries)
    if len(set(ordered_paths)) != len(ordered_paths):
        raise ValueError("partition support contains duplicate module order entries")
    missing = [path for path in module_by_path if path not in ordered_paths]
    if missing:
        raise ValueError(
            "partition support is missing traced modules: " + ", ".join(sorted(missing))
        )
    return tuple(module_by_path[path] for path in ordered_paths)


def _dependency_edges(
    profile: ModelProfile,
    support: AutomaticPartitionSupport,
) -> tuple[_DependencyEdge, ...]:
    module_by_path = {module.module_path: module for module in profile.modules}
    dependency_pairs = sorted(
        {
            pair
            for boundary in support.boundaries
            for pair in _parse_dependency_pairs(boundary.forward_dependencies)
        }
    )
    edges: list[_DependencyEdge] = []
    for source_path, target_path in dependency_pairs:
        source_module = module_by_path[source_path]
        target_module = module_by_path[target_path]
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
        gradient = tuple(
            TensorMetadata(
                name=f"{source_path}->{target_path}:grad:{tensor.name}",
                shape=tensor.shape,
                dtype=tensor.dtype,
                estimated_bytes=tensor.estimated_bytes,
                estimate_kind=EstimateKind.ESTIMATED,
                source="activation-sized backward approximation",
            )
            for tensor in source_module.output_tensors
        )
        activation_bytes = _tensor_bytes(activation)
        gradient_bytes = _tensor_bytes(gradient)
        edges.append(
            _DependencyEdge(
                source_path=source_path,
                target_path=target_path,
                source_module_id=source_module.module_id,
                target_module_id=target_module.module_id,
                activation=activation,
                gradient=gradient,
                estimated_bytes_per_step=(
                    None
                    if activation_bytes is None or gradient_bytes is None
                    else activation_bytes + gradient_bytes
                ),
                estimate_kind=(
                    EstimateKind.ESTIMATED
                    if activation_bytes is not None and gradient_bytes is not None
                    else EstimateKind.UNKNOWN
                ),
            )
        )
    return tuple(edges)


def _precompute_stage_estimates(
    profile: ModelProfile,
    ordered_modules: Sequence[Any],
    config: MemoryEstimationConfig,
) -> tuple[tuple[TrainingMemoryEstimate | None, ...], ...]:
    module_ids = [module.module_id for module in ordered_modules]
    estimates: list[list[TrainingMemoryEstimate | None]] = [
        [None for _ in range(len(ordered_modules) + 1)]
        for _ in range(len(ordered_modules))
    ]
    for start in range(len(ordered_modules)):
        for stop in range(start + 1, len(ordered_modules) + 1):
            estimates[start][stop] = estimate_stage_memory(
                profile,
                module_ids[start:stop],
                config,
            )
    return tuple(tuple(row) for row in estimates)


def _precompute_internal_edge_bytes(
    ordered_modules: Sequence[Any],
    dependency_edges: Sequence[_DependencyEdge],
) -> tuple[tuple[int, ...], ...]:
    index_by_path = {
        module.module_path: index for index, module in enumerate(ordered_modules)
    }
    internal: list[list[int]] = [
        [0 for _ in range(len(ordered_modules) + 1)]
        for _ in range(len(ordered_modules))
    ]
    for start in range(len(ordered_modules)):
        for stop in range(start + 1, len(ordered_modules) + 1):
            total = 0
            for edge in dependency_edges:
                source_index = index_by_path[edge.source_path]
                target_index = index_by_path[edge.target_path]
                if start <= source_index < stop and start <= target_index < stop:
                    total += edge.estimated_bytes_per_step or 0
            internal[start][stop] = total
    return tuple(tuple(row) for row in internal)


def _plan_stage_ranges(
    ordered_modules: Sequence[Any],
    boundaries: Sequence[PartitionBoundary],
    stage_estimates: Sequence[Sequence[TrainingMemoryEstimate | None]],
    stage_weights: Sequence[int],
    internal_edge_bytes: Sequence[Sequence[int]],
    *,
    stage_count: int,
    capacities: Sequence[int],
) -> tuple[tuple[int, int], ...] | None:
    module_count = len(ordered_modules)
    if stage_count <= 0 or stage_count > module_count:
        return None

    prefix_weights = [0]
    for weight in stage_weights:
        prefix_weights.append(prefix_weights[-1] + weight)
    targets = _target_prefix_weights(prefix_weights[-1], stage_count, capacities)
    max_capacity = max(capacities) if capacities else None
    boundary_by_position = {index + 1: boundary for index, boundary in enumerate(boundaries)}
    large_penalty = max(prefix_weights[-1], sum(stage_weights), 1) * (
        module_count + stage_count + 1
    )
    best_scores: list[list[int | None]] = [
        [None for _ in range(module_count + 1)] for _ in range(stage_count + 1)
    ]
    previous: list[list[int | None]] = [
        [None for _ in range(module_count + 1)] for _ in range(stage_count + 1)
    ]
    best_scores[0][0] = 0

    for stage_index in range(stage_count):
        for start in range(module_count + 1):
            current = best_scores[stage_index][start]
            if current is None:
                continue
            min_stop = start + 1
            max_stop = module_count - (stage_count - stage_index - 1)
            for stop in range(min_stop, max_stop + 1):
                objective = internal_edge_bytes[start][stop]
                objective -= _balance_penalty(
                    prefix_weights[stop],
                    targets[stage_index],
                    stage_index,
                    stage_count,
                )
                objective -= _boundary_penalty(
                    boundary_by_position.get(stop),
                    stop,
                    module_count,
                    large_penalty,
                )
                objective -= _memory_penalty(
                    stage_estimates[start][stop],
                    max_capacity,
                    large_penalty,
                )
                score = current + objective
                if (
                    best_scores[stage_index + 1][stop] is None
                    or score > best_scores[stage_index + 1][stop]
                ):
                    best_scores[stage_index + 1][stop] = score
                    previous[stage_index + 1][stop] = start

    if best_scores[stage_count][module_count] is None:
        return None

    ranges: list[tuple[int, int]] = []
    end = module_count
    for stage_index in range(stage_count, 0, -1):
        start = previous[stage_index][end]
        if start is None:
            return None
        ranges.append((start, end))
        end = start
    ranges.reverse()
    return tuple(ranges)


def _target_prefix_weights(
    total_weight: int,
    stage_count: int,
    capacities: Sequence[int],
) -> tuple[int, ...]:
    if capacities:
        selected = tuple(sorted(capacities, reverse=True)[:stage_count])
        capacity_total = sum(selected)
        if capacity_total > 0:
            running = 0
            targets: list[int] = []
            for capacity in selected:
                running += capacity
                targets.append((total_weight * running) // capacity_total)
            return tuple(targets)
    return tuple((total_weight * (index + 1)) // stage_count for index in range(stage_count))


def _balance_penalty(
    prefix_weight: int,
    target_weight: int,
    stage_index: int,
    stage_count: int,
) -> int:
    if stage_index == stage_count - 1:
        return 0
    delta = abs(prefix_weight - target_weight)
    tolerance = max(target_weight // 10, 1)
    return max(0, delta - tolerance)


def _boundary_penalty(
    boundary: PartitionBoundary | None,
    stop: int,
    module_count: int,
    large_penalty: int,
) -> int:
    if stop == module_count or boundary is None:
        return 0
    return 0 if boundary.status is PartitionSupportStatus.SUPPORTED else large_penalty


def _memory_penalty(
    estimate: TrainingMemoryEstimate | None,
    max_capacity: int | None,
    large_penalty: int,
) -> int:
    if estimate is None or estimate.planner_required_bytes is None:
        return large_penalty
    if max_capacity is None or estimate.planner_required_bytes <= max_capacity:
        return 0
    return estimate.planner_required_bytes - max_capacity


def _capacity_reasons(
    stages: Sequence[StagePartition],
    capacities: Sequence[int],
) -> tuple[str, ...]:
    if not capacities:
        return ()
    sorted_requirements = sorted(
        (
            (stage.stage_id, stage.estimated_peak_training_memory.planner_required_bytes)
            for stage in stages
        ),
        key=lambda item: -1 if item[1] is None else -item[1],
    )
    sorted_capacities = sorted(capacities, reverse=True)
    reasons: list[str] = []
    if len(stages) > len(sorted_capacities):
        reasons.append("candidate requires more stages than usable-memory capacity slots")
    for (stage_id, required), capacity in zip(sorted_requirements, sorted_capacities, strict=False):
        if required is None:
            reasons.append(f"stage {stage_id} memory estimate is unavailable")
            continue
        if required > capacity:
            reasons.append(
                "stage "
                f"{stage_id} training peak exceeds usable GPU memory after headroom "
                f"({required} > {capacity})"
            )
    reasons.extend(_capacity_imbalance_reasons(sorted_requirements, sorted_capacities))
    return tuple(reasons)


def _capacity_imbalance_reasons(
    requirements: Sequence[tuple[str, int | None]],
    capacities: Sequence[int],
) -> tuple[str, ...]:
    known = [
        (stage_id, required, capacity)
        for (stage_id, required), capacity in zip(requirements, capacities, strict=False)
        if required is not None
    ]
    if len(known) < 2:
        return ()
    smallest = min(capacity for _, _, capacity in known)
    largest = max(capacity for _, _, capacity in known)
    if smallest <= 0 or largest > int(smallest * 1.25):
        return ()
    total_required = sum(required for _, required, _ in known)
    total_capacity = sum(capacity for _, _, capacity in known)
    if total_required <= 0 or total_capacity <= 0:
        return ()
    max_delta = max(
        abs((required / total_required) - (capacity / total_capacity))
        for _, required, capacity in known
    )
    if max_delta <= 0.30:
        return ()
    return (
        "capacity-aware imbalance is too extreme for similar GPU memory capacity",
    )


def _capacity_balance_penalty(
    stages: Sequence[StagePartition],
    capacities: Sequence[int],
) -> int:
    requirements = sorted(
        (
            stage.estimated_peak_training_memory.planner_required_bytes or 0
            for stage in stages
        ),
        reverse=True,
    )
    if not capacities:
        if not requirements:
            return 0
        ideal = sum(requirements) // len(requirements)
        return sum(
            max(0, abs(required - ideal) - max(ideal // 10, 1))
            for required in requirements
        )

    penalty = 0
    for required, capacity in zip(requirements, sorted(capacities, reverse=True), strict=False):
        penalty += max(0, abs(required - capacity) - max(capacity // 10, 1))
    return penalty


def _capacity_shortfall(
    stages: Sequence[StagePartition],
    capacities: Sequence[int],
) -> int:
    if not capacities:
        return 0
    return sum(
        max(0, required - capacity)
        for required, capacity in zip(
            sorted(
                (
                    stage.estimated_peak_training_memory.planner_required_bytes or 0
                    for stage in stages
                ),
                reverse=True,
            ),
            sorted(capacities, reverse=True),
            strict=False,
        )
    )


def _candidate_sort_key(
    candidate: PartitionCandidate,
) -> tuple[int, int, int, int, int, tuple[str, ...], str]:
    return (
        0 if candidate.hard_constraint_status is FeasibilityStatus.FEASIBLE else 1,
        candidate.required_worker_count,
        int(candidate.score_breakdown.get("memory_shortfall_bytes", 0)),
        int(candidate.score_breakdown.get("balance_penalty_bytes", 0)),
        (
            candidate.estimated_bytes_per_step
            if candidate.estimated_bytes_per_step is not None
            else 2**63 - 1
        ),
        candidate.selected_boundary_ids,
        candidate.candidate_id,
    )


def _partition_weight_bytes(module: Any) -> int:
    return sum(
        value or 0
        for value in (
            module.memory.parameter_bytes,
            module.memory.gradient_bytes,
            module.memory.optimizer_bytes,
            module.memory.activation_bytes,
            module.memory.temporary_bytes,
        )
    )


def _normalize_capacity_bytes(
    usable_memory_bytes: Sequence[int] | int | None,
) -> tuple[int, ...]:
    if usable_memory_bytes is None:
        return ()
    if isinstance(usable_memory_bytes, int):
        if usable_memory_bytes <= 0:
            raise ValueError("usable_memory_bytes must be > 0")
        return (usable_memory_bytes,)
    capacities = tuple(int(value) for value in usable_memory_bytes)
    if any(value <= 0 for value in capacities):
        raise ValueError("usable_memory_bytes must contain only positive values")
    return tuple(sorted(capacities, reverse=True))


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
                    shape=key[1],
                    dtype=key[2],
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
