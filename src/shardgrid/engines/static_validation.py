"""Static validation for the supported two-stage ParallelPlan (T070).

Loads and validates ``examples/models/static_parallel_plan.yaml`` against the
real T069 Stage0 / Stage1 parameter ownership.  The plan is limited-support
by definition: it describes the explicit two-stage split of the supported
model, never arbitrary graph partitioning.

Validation returns a list of problems; an empty list means the plan matches
reality (world size, stage count, disjoint ranks, non-empty per-stage
parameters, complete non-overlapping coverage of the full model, and
matching tensor boundary metadata).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

LIMITED_SUPPORT_LABEL = "limited_support"


class StaticPlanValidationError(ValueError):
    """Raised when a static plan fails validation."""


@dataclass(frozen=True)
class StagePlan:
    id: str
    rank: int
    parameter_count: int
    parameter_ownership: tuple[str, ...] = ()
    activation_shape: tuple[int | str, ...] = ()
    activation_dtype: str | None = None
    worker_id: str | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StagePlan":
        activation = data.get("activation_out") or data.get("activation_in") or {}
        activation_shape = tuple(activation.get("shape", ()))
        activation_dtype = activation.get("dtype")
        return cls(
            id=str(data["id"]),
            rank=int(data["rank"]),
            parameter_count=int(data.get("parameter_count", 0)),
            parameter_ownership=tuple(
                str(item) for item in data.get("parameter_ownership", [])
            ),
            activation_shape=activation_shape,
            activation_dtype=activation_dtype,
            worker_id=data.get("worker_id"),
        )


@dataclass(frozen=True)
class StaticParallelPlan:
    plan_id: str
    plan_mode: str
    support_label: str
    engine: str
    engine_plan_path: str | None
    world_size: int
    stages: tuple[StagePlan, ...]
    model_total_parameter_count: int
    limitations: tuple[str, ...] = ()

    @property
    def stage_count(self) -> int:
        return len(self.stages)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StaticParallelPlan":
        model = data.get("model") or {}
        return cls(
            plan_id=str(data["plan_id"]),
            plan_mode=str(data.get("plan_mode", "static")),
            support_label=str(data.get("support_label", LIMITED_SUPPORT_LABEL)),
            engine=str(data.get("engine", "galvatron")),
            engine_plan_path=data.get("engine_plan_path"),
            world_size=int(data.get("world_size", 0)),
            stages=tuple(
                StagePlan.from_dict(item) for item in data.get("stages", [])
            ),
            model_total_parameter_count=int(
                model.get("total_parameter_count", 0)
            ),
            limitations=tuple(str(item) for item in data.get("limitations", [])),
        )


def load_static_parallel_plan(path: str | Path) -> StaticParallelPlan:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise StaticPlanValidationError("plan file must contain a mapping")
    return StaticParallelPlan.from_dict(payload)


def validate_parameter_coverage(
    *,
    stage_parameter_counts: Mapping[str, int],
    full_model_parameter_count: int,
) -> list[str]:
    """Check non-empty, disjoint (by construction), complete coverage."""
    problems: list[str] = []
    if len(stage_parameter_counts) == 0:
        problems.append("no stage parameter counts provided")
        return problems
    total = 0
    for stage_id, count in stage_parameter_counts.items():
        if count <= 0:
            problems.append(f"stage {stage_id} has parameter_count {count} (must be > 0)")
        total += int(count)
    if total != full_model_parameter_count:
        problems.append(
            f"stage parameter total {total} != full model parameter count "
            f"{full_model_parameter_count}"
        )
    return problems


def validate_parameter_ownership(
    *,
    stage_ownership: Mapping[str, Sequence[str]],
    full_parameter_names: Sequence[str],
) -> list[str]:
    """Parameter names must be disjoint and together cover the full model."""
    problems: list[str] = []
    full_set = set(full_parameter_names)
    seen: set[str] = set()
    union: set[str] = set()
    for stage_id, names in stage_ownership.items():
        name_set = set(names)
        duplicates = name_set & seen
        if duplicates:
            problems.append(
                f"stage {stage_id} repeats parameters: {sorted(duplicates)}"
            )
        unknown = name_set - full_set
        if unknown:
            problems.append(
                f"stage {stage_id} owns unknown parameters: {sorted(unknown)}"
            )
        seen |= name_set
        union |= name_set
    missing = full_set - union
    if missing:
        problems.append(f"parameters missing from all stages: {sorted(missing)}")
    if union != full_set:
        problems.append("stage parameter ownership does not cover the full model")
    return problems


def validate_tensor_boundary(
    *,
    activation_out: tuple[int | str, ...] | None,
    activation_in: tuple[int | str, ...] | None,
    dtype_out: str | None,
    dtype_in: str | None,
) -> list[str]:
    """Stage0 activation output must match Stage1 activation input metadata."""
    problems: list[str] = []
    if activation_out is None or activation_in is None:
        problems.append("tensor boundary metadata missing (activation)")
        return problems
    if activation_out != activation_in:
        problems.append(
            f"activation shape mismatch: {activation_out} vs {activation_in}"
        )
    if dtype_out is not None and dtype_in is not None and dtype_out != dtype_in:
        problems.append(f"activation dtype mismatch: {dtype_out} vs {dtype_in}")
    return problems


def validate_static_plan(
    plan: StaticParallelPlan,
    *,
    stage_parameter_counts: Mapping[str, int] | None = None,
    full_model_parameter_count: int | None = None,
    stage_ownership: Mapping[str, Sequence[str]] | None = None,
    full_parameter_names: Sequence[str] | None = None,
) -> list[str]:
    """Validate the static plan and (optionally) real parameter coverage."""
    problems: list[str] = []
    if plan.world_size <= 0:
        problems.append(f"world_size {plan.world_size} must be > 0")
    if plan.stage_count <= 0:
        problems.append(f"stage_count {plan.stage_count} must be > 0")
    if plan.world_size > 0 and plan.stage_count != plan.world_size:
        problems.append(
            f"stage_count {plan.stage_count} != world_size {plan.world_size}"
        )
    ranks = [stage.rank for stage in plan.stages]
    if len(ranks) != len(set(ranks)):
        problems.append(f"stages must map to distinct ranks, got {ranks}")
    for stage in plan.stages:
        if stage.parameter_count <= 0:
            problems.append(
                f"stage {stage.id} parameter_count {stage.parameter_count} must be > 0"
            )
    if plan.support_label != LIMITED_SUPPORT_LABEL:
        problems.append(
            f"support_label {plan.support_label!r} != {LIMITED_SUPPORT_LABEL!r}"
        )
    if plan.engine_plan_path is None:
        problems.append("external engine plan reference missing")
    if plan.plan_mode != "static":
        problems.append(f"plan_mode {plan.plan_mode!r} != 'static'")

    if stage_parameter_counts is not None:
        problems.extend(
            validate_parameter_coverage(
                stage_parameter_counts=stage_parameter_counts,
                full_model_parameter_count=(
                    full_model_parameter_count
                    if full_model_parameter_count is not None
                    else plan.model_total_parameter_count
                ),
            )
        )
    if stage_ownership is not None and full_parameter_names is not None:
        problems.extend(
            validate_parameter_ownership(
                stage_ownership=stage_ownership,
                full_parameter_names=full_parameter_names,
            )
        )
    return problems


def validate_static_plan_or_raise(
    plan: StaticParallelPlan,
    *,
    stage_parameter_counts: Mapping[str, int] | None = None,
    full_model_parameter_count: int | None = None,
    stage_ownership: Mapping[str, Sequence[str]] | None = None,
    full_parameter_names: Sequence[str] | None = None,
) -> None:
    problems = validate_static_plan(
        plan,
        stage_parameter_counts=stage_parameter_counts,
        full_model_parameter_count=full_model_parameter_count,
        stage_ownership=stage_ownership,
        full_parameter_names=full_parameter_names,
    )
    if problems:
        raise StaticPlanValidationError("; ".join(problems))
