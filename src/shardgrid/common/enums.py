"""Shared enums with stable string serialization."""

from __future__ import annotations

from enum import StrEnum
from typing import Self


class SerializableStrEnum(StrEnum):
    """Enum base class that round-trips as a plain string value."""

    def to_json(self) -> str:
        return self.value

    @classmethod
    def from_value(cls, value: str) -> Self:
        return cls(value)


class MachineRole(SerializableStrEnum):
    CONTROL = "control"
    GPU_WORKER = "gpu_worker"
    CLIENT = "client"
    DEV_TEST = "dev_test"
    BACKUP_LOGIN = "backup_login"


class PhysicalOS(SerializableStrEnum):
    LINUX = "linux"
    WINDOWS = "windows"


class RuntimeOS(SerializableStrEnum):
    LINUX = "linux"
    WSL2_LINUX = "wsl2_linux"
    WINDOWS = "windows"
    UNKNOWN = "unknown"


class Health(SerializableStrEnum):
    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    BLOCKED_MANUAL_ACTION = "blocked_manual_action"
    UNREACHABLE = "unreachable"
    FAILED = "failed"


class BackendStatus(SerializableStrEnum):
    NOT_CHECKED = "not_checked"
    AVAILABLE = "available"
    FAILED = "failed"
    FALLBACK_USED = "fallback_used"
    EXPERIMENTAL = "experimental"
    BLOCKED = "blocked"


class JobState(SerializableStrEnum):
    CREATED = "created"
    PROBING = "probing"
    PLANNING = "planning"
    SNAPSHOTTING = "snapshotting"
    DISTRIBUTING = "distributing"
    LAUNCHING = "launching"
    RENDEZVOUS = "rendezvous"
    TRAINING = "training"
    CHECKPOINTING = "checkpointing"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPING = "stopping"
    STOPPED = "stopped"


class FailureStage(SerializableStrEnum):
    BOOTSTRAP = "BOOTSTRAP"
    PROBE = "PROBE"
    NETWORK = "NETWORK"
    PROFILE = "PROFILE"
    PLAN = "PLAN"
    DISTRIBUTE = "DISTRIBUTE"
    LAUNCH = "LAUNCH"
    RENDEZVOUS = "RENDEZVOUS"
    TRAIN = "TRAIN"
    CHECKPOINT = "CHECKPOINT"
    SCHEDULE = "SCHEDULE"
    GPU_SHARE = "GPU_SHARE"
    STOP = "STOP"
    CLEANUP = "CLEANUP"


class FailureCode(SerializableStrEnum):
    """Precise failure codes from the failure contract (contracts/failures.md).

    Broad :class:`FailureStage` remains the coarse-grained compatibility layer;
    ``FailureCode`` provides the exact machine-readable taxonomy.
    """

    MODEL_CAPTURE_UNSUPPORTED = "MODEL_CAPTURE_UNSUPPORTED"
    GRAPH_BREAK_UNSUPPORTED = "GRAPH_BREAK_UNSUPPORTED"
    CUSTOM_OP_UNSUPPORTED = "CUSTOM_OP_UNSUPPORTED"
    DYNAMIC_CONTROL_FLOW_UNSUPPORTED = "DYNAMIC_CONTROL_FLOW_UNSUPPORTED"

    PROFILE_FAILURE = "PROFILE_FAILURE"
    PARTITION_FAILURE = "PARTITION_FAILURE"
    PLAN_VALIDATION_FAILURE = "PLAN_VALIDATION_FAILURE"
    NO_FEASIBLE_PLAN = "NO_FEASIBLE_PLAN"
    SEARCH_BUDGET_LIMIT = "SEARCH_BUDGET_LIMIT"

    MEMORY_REJECT = "MEMORY_REJECT"
    FORMAL_TRAINING_OOM = "FORMAL_TRAINING_OOM"
    RESOURCE_CHANGED = "RESOURCE_CHANGED"

    NETWORK_FAILURE = "NETWORK_FAILURE"
    RENDEZVOUS_FAILURE = "RENDEZVOUS_FAILURE"
    PROCESS_LAUNCH_FAILURE = "PROCESS_LAUNCH_FAILURE"
    INFRA_FAILURE = "INFRA_FAILURE"
    RUNTIME_FAILURE = "RUNTIME_FAILURE"

    CPU_PROCESS_SATURATION = "CPU_PROCESS_SATURATION"
    GPU_MEMORY_SATURATION = "GPU_MEMORY_SATURATION"
    TEST_LIMIT_REACHED = "TEST_LIMIT_REACHED"
    SATURATION_NOT_PROVEN = "SATURATION_NOT_PROVEN"
