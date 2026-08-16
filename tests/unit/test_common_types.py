from __future__ import annotations

from typing import Any, Callable

import pytest

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
    as_backend_name,
    as_engine_name,
    as_hostname,
    as_job_id,
    as_machine_id,
    as_worker_id,
)


@pytest.mark.parametrize(
    ("enum_type", "member", "raw_value"),
    [
        (MachineRole, MachineRole.CONTROL, "control"),
        (MachineRole, MachineRole.GPU_WORKER, "gpu_worker"),
        (PhysicalOS, PhysicalOS.LINUX, "linux"),
        (RuntimeOS, RuntimeOS.WSL2_LINUX, "wsl2_linux"),
        (Health, Health.BLOCKED_MANUAL_ACTION, "blocked_manual_action"),
        (BackendStatus, BackendStatus.FALLBACK_USED, "fallback_used"),
        (JobState, JobState.CHECKPOINTING, "checkpointing"),
        (FailureStage, FailureStage.RENDEZVOUS, "RENDEZVOUS"),
    ],
)
def test_enums_round_trip_as_strings(
    enum_type: Any, member: Any, raw_value: str
) -> None:
    assert member.value == raw_value
    assert member.to_json() == raw_value
    assert enum_type.from_value(raw_value) is member
    assert enum_type(raw_value) is member


@pytest.mark.parametrize(
    ("enum_type", "raw_value"),
    [
        (MachineRole, "trainer"),
        (PhysicalOS, "macos"),
        (RuntimeOS, "wsl1"),
        (Health, "passing"),
        (BackendStatus, "ready"),
        (JobState, "queued"),
        (FailureStage, "launch"),
    ],
)
def test_invalid_enum_values_are_rejected(enum_type: Any, raw_value: str) -> None:
    with pytest.raises(ValueError):
        enum_type(raw_value)


@pytest.mark.parametrize(
    ("factory", "raw_value", "expected"),
    [
        (as_machine_id, " machine-a ", "machine-a"),
        (as_worker_id, "worker-c", "worker-c"),
        (as_job_id, "job-0001", "job-0001"),
        (as_hostname, "gpu4060.local", "gpu4060.local"),
        (as_backend_name, "ssh", "ssh"),
        (as_engine_name, "galvatron", "galvatron"),
    ],
)
def test_identifier_factories_normalize_non_empty_values(
    factory: Callable[[str], str], raw_value: str, expected: str
) -> None:
    assert factory(raw_value) == expected


@pytest.mark.parametrize(
    ("factory", "raw_value"),
    [
        (as_machine_id, ""),
        (as_worker_id, "   "),
        (as_job_id, "\n"),
        (as_hostname, "\t"),
        (as_backend_name, ""),
        (as_engine_name, " "),
    ],
)
def test_identifier_factories_reject_blank_values(
    factory: Callable[[str], str], raw_value: str
) -> None:
    with pytest.raises(ValueError):
        factory(raw_value)
