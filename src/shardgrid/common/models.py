"""Shared identifier types used across ShardGrid models."""

from __future__ import annotations

from typing import NewType, TypeVar

MachineId = NewType("MachineId", str)
WorkerId = NewType("WorkerId", str)
JobId = NewType("JobId", str)
Hostname = NewType("Hostname", str)
BackendName = NewType("BackendName", str)
EngineName = NewType("EngineName", str)

IdentifierType = TypeVar(
    "IdentifierType",
    MachineId,
    WorkerId,
    JobId,
    Hostname,
    BackendName,
    EngineName,
)


def _as_identifier(
    value: str,
    *,
    field_name: str,
    identifier_type: type[IdentifierType],
) -> IdentifierType:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string")
    return identifier_type(normalized)


def as_machine_id(value: str) -> MachineId:
    return _as_identifier(value, field_name="machine_id", identifier_type=MachineId)


def as_worker_id(value: str) -> WorkerId:
    return _as_identifier(value, field_name="worker_id", identifier_type=WorkerId)


def as_job_id(value: str) -> JobId:
    return _as_identifier(value, field_name="job_id", identifier_type=JobId)


def as_hostname(value: str) -> Hostname:
    return _as_identifier(value, field_name="hostname", identifier_type=Hostname)


def as_backend_name(value: str) -> BackendName:
    return _as_identifier(value, field_name="backend_name", identifier_type=BackendName)


def as_engine_name(value: str) -> EngineName:
    return _as_identifier(value, field_name="engine_name", identifier_type=EngineName)
