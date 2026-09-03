"""Planner package skeleton for ShardGrid."""

from .memory import (
    MemoryEstimationConfig,
    StageMemoryFit,
    build_model_profile,
    dtype_bytes,
    estimate_stage_memory,
    evaluate_stage_memory_fit,
    normalize_dtype_name,
)
from .requirements import (
    CommunicationRequirement,
    ConstraintViolation,
    FeasibilityStatus,
    MemoryConstraint,
    ParallelPlanningRequirements,
    PlacementFeasibilityResult,
    PlacementRequirements,
    WorkerEligibilityRequirements,
    WorkerEligibilityResult,
    evaluate_worker_eligibility,
    validate_partition_boundary,
    validate_placement_feasibility,
)

__all__ = [
    "MemoryEstimationConfig",
    "StageMemoryFit",
    "CommunicationRequirement",
    "ConstraintViolation",
    "build_model_profile",
    "dtype_bytes",
    "estimate_stage_memory",
    "evaluate_stage_memory_fit",
    "FeasibilityStatus",
    "MemoryConstraint",
    "normalize_dtype_name",
    "ParallelPlanningRequirements",
    "PlacementFeasibilityResult",
    "PlacementRequirements",
    "WorkerEligibilityRequirements",
    "WorkerEligibilityResult",
    "evaluate_worker_eligibility",
    "validate_partition_boundary",
    "validate_placement_feasibility",
]
