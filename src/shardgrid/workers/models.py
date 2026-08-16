"""Worker and machine data models for ShardGrid."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, cast

from shardgrid.common.enums import Health, MachineRole, PhysicalOS, RuntimeOS
from shardgrid.common.models import (
    Hostname,
    MachineId,
    WorkerId,
    as_hostname,
    as_machine_id,
    as_worker_id,
)

PathStyle = str
RuntimeName = str



def _serialize(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "value"):
        return value.value
    if is_dataclass(value):
        return {key: _serialize(item) for key, item in asdict(cast(Any, value)).items()}
    if isinstance(value, dict):
        return {str(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    return value


@dataclass(frozen=True)
class Machine:
    machine_id: MachineId
    role: MachineRole
    physical_os: PhysicalOS
    hostname: Hostname
    configured_host: str
    required_for_mvp: bool = False
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Machine":
        return cls(
            machine_id=as_machine_id(str(data["machine_id"])),
            role=MachineRole(data["role"]),
            physical_os=PhysicalOS(data["physical_os"]),
            hostname=as_hostname(str(data["hostname"])),
            configured_host=str(data["configured_host"]),
            required_for_mvp=bool(data.get("required_for_mvp", False)),
            notes=data.get("notes"),
        )


@dataclass(frozen=True)
class ControlNode:
    machine_id: MachineId
    hostname: Hostname
    os_version: str
    python_version: str
    ssh_available: bool
    git_available: bool
    iperf3_available: bool
    jobs_root: Path
    disk_free_bytes: int
    health: Health = Health.UNKNOWN

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ControlNode":
        return cls(
            machine_id=as_machine_id(str(data["machine_id"])),
            hostname=as_hostname(str(data["hostname"])),
            os_version=str(data["os_version"]),
            python_version=str(data["python_version"]),
            ssh_available=bool(data["ssh_available"]),
            git_available=bool(data["git_available"]),
            iperf3_available=bool(data["iperf3_available"]),
            jobs_root=Path(str(data["jobs_root"])),
            disk_free_bytes=int(data["disk_free_bytes"]),
            health=Health(data.get("health", Health.UNKNOWN.value)),
        )


@dataclass(frozen=True)
class Worker:
    worker_id: WorkerId
    machine_id: MachineId
    hostname: Hostname
    physical_os: PhysicalOS
    runtime_os: RuntimeOS
    host: str
    ssh_user_ref: str
    runtime: RuntimeName
    runtime_distro: str | None = None
    local_world_size: int = 1
    enabled: bool = True
    health: Health = Health.UNKNOWN

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Worker":
        return cls(
            worker_id=as_worker_id(str(data["worker_id"])),
            machine_id=as_machine_id(str(data["machine_id"])),
            hostname=as_hostname(str(data["hostname"])),
            physical_os=PhysicalOS(data["physical_os"]),
            runtime_os=RuntimeOS(data["runtime_os"]),
            host=str(data["host"]),
            ssh_user_ref=str(data["ssh_user_ref"]),
            runtime=str(data["runtime"]),
            runtime_distro=data.get("runtime_distro"),
            local_world_size=int(data.get("local_world_size", 1)),
            enabled=bool(data.get("enabled", True)),
            health=Health(data.get("health", Health.UNKNOWN.value)),
        )


@dataclass(frozen=True)
class WorkerRuntime:
    worker_id: WorkerId
    runtime_os: RuntimeOS
    runtime_version: str | None = None
    python_version: str | None = None
    torch_version: str | None = None
    torch_cuda_version: str | None = None
    cuda_available: bool = False
    nccl_available: bool = False
    gloo_available: bool = False
    nvidia_smi_path: str | None = None
    path_style: PathStyle = "posix"
    health: Health = Health.UNKNOWN

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkerRuntime":
        return cls(
            worker_id=as_worker_id(str(data["worker_id"])),
            runtime_os=RuntimeOS(data["runtime_os"]),
            runtime_version=data.get("runtime_version"),
            python_version=data.get("python_version"),
            torch_version=data.get("torch_version"),
            torch_cuda_version=data.get("torch_cuda_version"),
            cuda_available=bool(data.get("cuda_available", False)),
            nccl_available=bool(data.get("nccl_available", False)),
            gloo_available=bool(data.get("gloo_available", False)),
            nvidia_smi_path=data.get("nvidia_smi_path"),
            path_style=str(data.get("path_style", "posix")),
            health=Health(data.get("health", Health.UNKNOWN.value)),
        )
