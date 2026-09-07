"""Shared types and helpers for ShardGrid."""

from shardgrid.common.enums import (
    BackendStatus,
    FailureStage,
    Health,
    JobState,
    MachineRole,
    PhysicalOS,
    RuntimeOS,
)
from shardgrid.common.models import (
    BackendName,
    EngineName,
    Hostname,
    JobId,
    MachineId,
    WorkerId,
    as_backend_name,
    as_engine_name,
    as_hostname,
    as_job_id,
    as_machine_id,
    as_worker_id,
)

__all__ = [
    "BackendName",
    "BackendStatus",
    "EngineName",
    "FailureStage",
    "Health",
    "Hostname",
    "JobId",
    "JobState",
    "MachineId",
    "MachineRole",
    "PhysicalOS",
    "RuntimeOS",
    "WorkerId",
    "as_backend_name",
    "as_engine_name",
    "as_hostname",
    "as_job_id",
    "as_machine_id",
    "as_worker_id",
]
