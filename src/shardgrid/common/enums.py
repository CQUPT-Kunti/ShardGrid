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
