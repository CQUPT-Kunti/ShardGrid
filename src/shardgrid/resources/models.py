"""Resource models used by inventory, probing, and planning."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, cast

from shardgrid.common.enums import Health, PhysicalOS, RuntimeOS
from shardgrid.common.models import Hostname, WorkerId, as_hostname, as_worker_id


def _serialize(value: Any) -> Any:
    if hasattr(value, "value"):
        return value.value
    if is_dataclass(value):
        return {key: _serialize(item) for key, item in asdict(cast(Any, value)).items()}
    if isinstance(value, dict):
        return {str(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    return value


def _optional_int(data: dict[str, Any], key: str) -> int | None:
    value = data.get(key)
    return None if value is None else int(value)


def _optional_float(data: dict[str, Any], key: str) -> float | None:
    value = data.get(key)
    return None if value is None else float(value)


@dataclass(frozen=True)
class GPUResource:
    worker_id: WorkerId
    gpu_index: int = 0
    gpu_name: str | None = None
    total_memory_mb: int | None = None
    free_memory_mb: int | None = None
    utilization_percent: float | None = None
    compute_capability: str | None = None
    driver_version: str | None = None
    cuda_version: str | None = None
    health: Health = Health.UNKNOWN

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GPUResource":
        return cls(
            worker_id=as_worker_id(str(data["worker_id"])),
            gpu_index=int(data.get("gpu_index", 0)),
            gpu_name=data.get("gpu_name"),
            total_memory_mb=_optional_int(data, "total_memory_mb"),
            free_memory_mb=_optional_int(data, "free_memory_mb"),
            utilization_percent=_optional_float(data, "utilization_percent"),
            compute_capability=data.get("compute_capability"),
            driver_version=data.get("driver_version"),
            cuda_version=data.get("cuda_version"),
            health=Health(data.get("health", Health.UNKNOWN.value)),
        )


@dataclass(frozen=True)
class WorkerResource:
    worker_id: WorkerId
    hostname: Hostname
    physical_os: PhysicalOS
    runtime_os: RuntimeOS
    environment_manager: str = "conda"
    conda_environment: str | None = None
    conda_prefix: str | None = None
    python_executable: str | None = None
    ip: str | None = None
    gpu_name: str | None = None
    gpu_total_memory: int | None = None
    gpu_free_memory: int | None = None
    gpu_utilization: float | None = None
    compute_capability: str | None = None
    driver_version: str | None = None
    cuda_version: str | None = None
    torch_version: str | None = None
    torch_cuda_version: str | None = None
    nccl_available: bool = False
    gloo_available: bool = False
    network_interface: str | None = None
    network_bandwidth: float | None = None
    network_latency: float | None = None
    health: Health = Health.UNKNOWN
    last_probe_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkerResource":
        return cls(
            worker_id=as_worker_id(str(data["worker_id"])),
            hostname=as_hostname(str(data["hostname"])),
            physical_os=PhysicalOS(data["physical_os"]),
            runtime_os=RuntimeOS(data["runtime_os"]),
            environment_manager=str(data.get("environment_manager", "conda")),
            conda_environment=data.get("conda_environment"),
            conda_prefix=data.get("conda_prefix"),
            python_executable=data.get("python_executable"),
            ip=data.get("ip"),
            gpu_name=data.get("gpu_name"),
            gpu_total_memory=_optional_int(data, "gpu_total_memory"),
            gpu_free_memory=_optional_int(data, "gpu_free_memory"),
            gpu_utilization=_optional_float(data, "gpu_utilization"),
            compute_capability=data.get("compute_capability"),
            driver_version=data.get("driver_version"),
            cuda_version=data.get("cuda_version"),
            torch_version=data.get("torch_version"),
            torch_cuda_version=data.get("torch_cuda_version"),
            nccl_available=bool(data.get("nccl_available", False)),
            gloo_available=bool(data.get("gloo_available", False)),
            network_interface=data.get("network_interface"),
            network_bandwidth=_optional_float(data, "network_bandwidth"),
            network_latency=_optional_float(data, "network_latency"),
            health=Health(data.get("health", Health.UNKNOWN.value)),
            last_probe_at=data.get("last_probe_at"),
        )


@dataclass(frozen=True)
class NetworkLink:
    source_worker_id: WorkerId
    target_worker_id: WorkerId
    source_ip: str
    target_ip: str
    interface: str
    tcp_reachable: bool
    latency_ms: float | None = None
    bandwidth_mbps: float | None = None
    interface_mtu: int | None = None
    expected_mtu: int | None = None
    mtu_status: str | None = None
    port: int = 29500
    measured_at: str | None = None
    failure_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NetworkLink":
        return cls(
            source_worker_id=as_worker_id(str(data["source_worker_id"])),
            target_worker_id=as_worker_id(str(data["target_worker_id"])),
            source_ip=str(data["source_ip"]),
            target_ip=str(data["target_ip"]),
            interface=str(data["interface"]),
            tcp_reachable=bool(data["tcp_reachable"]),
            latency_ms=_optional_float(data, "latency_ms"),
            bandwidth_mbps=_optional_float(data, "bandwidth_mbps"),
            interface_mtu=_optional_int(data, "interface_mtu"),
            expected_mtu=_optional_int(data, "expected_mtu"),
            mtu_status=data.get("mtu_status"),
            port=int(data.get("port", 29500)),
            measured_at=data.get("measured_at"),
            failure_reason=data.get("failure_reason"),
        )


@dataclass(frozen=True)
class NetworkState:
    network_id: str
    workers: list[WorkerId] = field(default_factory=list)
    links: list[NetworkLink] = field(default_factory=list)
    created_at: str | None = None
    selected_interfaces: dict[str, str] = field(default_factory=dict)
    diagnostics_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NetworkState":
        return cls(
            network_id=str(data["network_id"]),
            workers=[as_worker_id(str(item)) for item in data.get("workers", [])],
            links=[NetworkLink.from_dict(item) for item in data.get("links", [])],
            created_at=data.get("created_at"),
            selected_interfaces={
                str(key): str(value)
                for key, value in data.get("selected_interfaces", {}).items()
            },
            diagnostics_path=data.get("diagnostics_path"),
        )
