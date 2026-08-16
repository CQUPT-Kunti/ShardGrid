"""Configuration loading and validation for ShardGrid."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path, PurePath
from typing import Any

import yaml

from shardgrid.common.enums import PhysicalOS, RuntimeOS
from shardgrid.common.models import (
    BackendName,
    EngineName,
    Hostname,
    MachineId,
    WorkerId,
    as_backend_name,
    as_engine_name,
    as_hostname,
    as_machine_id,
    as_worker_id,
)


class ConfigValidationError(ValueError):
    """Raised when a ShardGrid configuration is invalid."""


def _require_mapping(value: object, *, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigValidationError(f"{field_name} must be a mapping")
    return value


def _require_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigValidationError(f"{field_name} must be a non-empty string")
    return value.strip()


def _optional_string(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    return _require_string(value, field_name=field_name)


def _require_int(value: object, *, field_name: str, minimum: int = 1) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigValidationError(f"{field_name} must be an integer")
    if value < minimum:
        raise ConfigValidationError(f"{field_name} must be >= {minimum}")
    return value


def _require_bool(value: object, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigValidationError(f"{field_name} must be a boolean")
    return value


def _require_string_list(value: object, *, field_name: str) -> list[str]:
    if not isinstance(value, list):
        raise ConfigValidationError(f"{field_name} must be a list of strings")
    return [_require_string(item, field_name=field_name) for item in value]


def _validate_jobs_root(value: object) -> Path:
    raw_path = _require_string(value, field_name="jobs_root")
    path = Path(raw_path)
    pure_path = PurePath(raw_path)
    if not path.is_absolute():
        raise ConfigValidationError("jobs_root must be an absolute path")
    if raw_path == path.anchor:
        raise ConfigValidationError("jobs_root must not be the filesystem root")
    if ".." in pure_path.parts:
        raise ConfigValidationError("jobs_root must not contain parent traversal")
    return path


def _require_string_mapping(value: object, *, field_name: str) -> dict[str, str]:
    mapping = _require_mapping(value, field_name=field_name)
    return {
        _require_string(key, field_name=field_name): _require_string(
            item, field_name=field_name
        )
        for key, item in mapping.items()
    }


def _serialize(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "value"):
        return value.value
    if is_dataclass(value):
        return {field.name: _serialize(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {str(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    return value


@dataclass(frozen=True)
class ControlNodeConfig:
    machine_id: MachineId
    hostname: Hostname

    @classmethod
    def from_dict(cls, data: object) -> ControlNodeConfig:
        payload = _require_mapping(data, field_name="control")
        return cls(
            machine_id=as_machine_id(
                _require_string(payload.get("machine_id"), field_name="control.machine_id")
            ),
            hostname=as_hostname(
                _require_string(payload.get("hostname"), field_name="control.hostname")
            ),
        )


@dataclass(frozen=True)
class SSHConfig:
    default_port: int = 22
    connect_timeout_seconds: int = 15
    strict_host_key_checking: bool = True
    private_key_path: str | None = None
    known_hosts_path: str | None = None

    @classmethod
    def from_dict(cls, data: object) -> SSHConfig:
        payload = _require_mapping(data, field_name="ssh")
        return cls(
            default_port=_require_int(
                payload.get("default_port", 22), field_name="ssh.default_port"
            ),
            connect_timeout_seconds=_require_int(
                payload.get("connect_timeout_seconds", 15),
                field_name="ssh.connect_timeout_seconds",
            ),
            strict_host_key_checking=_require_bool(
                payload.get("strict_host_key_checking", True),
                field_name="ssh.strict_host_key_checking",
            ),
            private_key_path=_optional_string(
                payload.get("private_key_path"), field_name="ssh.private_key_path"
            ),
            known_hosts_path=_optional_string(
                payload.get("known_hosts_path"), field_name="ssh.known_hosts_path"
            ),
        )


@dataclass(frozen=True)
class RuntimeConfig:
    python_executable: str = "python3"
    environment_manager: str = "conda"
    conda_executable: str | None = None
    conda_environment: str | None = None
    conda_prefix: str | None = None
    linux_shell: str = "/bin/bash"
    windows_shell: str = "powershell.exe"
    wsl_shell: str = "/bin/bash"
    default_wsl_distro: str | None = None
    environment: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: object) -> RuntimeConfig:
        payload = _require_mapping(data, field_name="runtime")
        return cls(
            python_executable=_require_string(
                payload.get("python_executable", "python3"),
                field_name="runtime.python_executable",
            ),
            environment_manager=_require_string(
                payload.get("environment_manager", "conda"),
                field_name="runtime.environment_manager",
            ),
            conda_executable=_optional_string(
                payload.get("conda_executable"),
                field_name="runtime.conda_executable",
            ),
            conda_environment=_optional_string(
                payload.get("conda_environment"),
                field_name="runtime.conda_environment",
            ),
            conda_prefix=_optional_string(
                payload.get("conda_prefix"),
                field_name="runtime.conda_prefix",
            ),
            linux_shell=_require_string(
                payload.get("linux_shell", "/bin/bash"), field_name="runtime.linux_shell"
            ),
            windows_shell=_require_string(
                payload.get("windows_shell", "powershell.exe"),
                field_name="runtime.windows_shell",
            ),
            wsl_shell=_require_string(
                payload.get("wsl_shell", "/bin/bash"), field_name="runtime.wsl_shell"
            ),
            default_wsl_distro=_optional_string(
                payload.get("default_wsl_distro"),
                field_name="runtime.default_wsl_distro",
            ),
            environment=_require_string_mapping(
                payload.get("environment", {}), field_name="runtime.environment"
            ),
        )


@dataclass(frozen=True)
class NetworkConfig:
    rendezvous_port: int = 29500
    iperf3_port: int = 5201
    interface_preference: str | None = None

    @classmethod
    def from_dict(cls, data: object) -> NetworkConfig:
        payload = _require_mapping(data, field_name="network")
        return cls(
            rendezvous_port=_require_int(
                payload.get("rendezvous_port", 29500),
                field_name="network.rendezvous_port",
            ),
            iperf3_port=_require_int(
                payload.get("iperf3_port", 5201), field_name="network.iperf3_port"
            ),
            interface_preference=_optional_string(
                payload.get("interface_preference"),
                field_name="network.interface_preference",
            ),
        )


@dataclass(frozen=True)
class BackendPreferenceConfig:
    launcher: BackendName
    communication_backend: BackendName
    parallel_engine: EngineName

    @classmethod
    def from_dict(cls, data: object) -> BackendPreferenceConfig:
        payload = _require_mapping(data, field_name="backend_preference")
        return cls(
            launcher=as_backend_name(
                _require_string(
                    payload.get("launcher", "ssh"),
                    field_name="backend_preference.launcher",
                )
            ),
            communication_backend=as_backend_name(
                _require_string(
                    payload.get("communication_backend", "auto"),
                    field_name="backend_preference.communication_backend",
                )
            ),
            parallel_engine=as_engine_name(
                _require_string(
                    payload.get("parallel_engine", "auto"),
                    field_name="backend_preference.parallel_engine",
                )
            ),
        )


@dataclass(frozen=True)
class ManualOverrideConfig:
    preferred_workers: list[WorkerId] = field(default_factory=list)
    disabled_workers: list[WorkerId] = field(default_factory=list)
    worker_address_overrides: dict[WorkerId, str] = field(default_factory=dict)
    rendezvous_port: int | None = None

    @classmethod
    def from_dict(cls, data: object) -> ManualOverrideConfig:
        payload = _require_mapping(data, field_name="manual_override")
        return cls(
            preferred_workers=[
                as_worker_id(item)
                for item in _require_string_list(
                    payload.get("preferred_workers", []),
                    field_name="manual_override.preferred_workers",
                )
            ],
            disabled_workers=[
                as_worker_id(item)
                for item in _require_string_list(
                    payload.get("disabled_workers", []),
                    field_name="manual_override.disabled_workers",
                )
            ],
            worker_address_overrides={
                as_worker_id(key): _require_string(
                    item, field_name="manual_override.worker_address_overrides"
                )
                for key, item in _require_mapping(
                    payload.get("worker_address_overrides", {}),
                    field_name="manual_override.worker_address_overrides",
                ).items()
            },
            rendezvous_port=(
                None
                if payload.get("rendezvous_port") is None
                else _require_int(
                    payload["rendezvous_port"],
                    field_name="manual_override.rendezvous_port",
                )
            ),
        )


@dataclass(frozen=True)
class WorkerConfig:
    worker_id: WorkerId
    machine_id: MachineId
    physical_os: PhysicalOS
    runtime_os: RuntimeOS
    runtime: str
    host: Hostname
    ssh_user: str
    ssh_port: int = 22
    runtime_distro: str | None = None
    conda_environment: str | None = None
    conda_prefix: str | None = None
    local_world_size: int = 1
    enabled: bool = True
    labels: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: object) -> WorkerConfig:
        payload = _require_mapping(data, field_name="worker")
        return cls(
            worker_id=as_worker_id(
                _require_string(payload.get("id"), field_name="worker.id")
            ),
            machine_id=as_machine_id(
                _require_string(payload.get("machine_id"), field_name="worker.machine_id")
            ),
            physical_os=PhysicalOS(
                _require_string(payload.get("physical_os"), field_name="worker.physical_os")
            ),
            runtime_os=RuntimeOS(
                _require_string(payload.get("runtime_os"), field_name="worker.runtime_os")
            ),
            runtime=_require_string(payload.get("runtime"), field_name="worker.runtime"),
            host=as_hostname(
                _require_string(payload.get("host"), field_name="worker.host")
            ),
            ssh_user=_require_string(payload.get("ssh_user"), field_name="worker.ssh_user"),
            ssh_port=_require_int(
                payload.get("ssh_port", 22), field_name="worker.ssh_port"
            ),
            runtime_distro=_optional_string(
                payload.get("runtime_distro"), field_name="worker.runtime_distro"
            ),
            conda_environment=_optional_string(
                payload.get("conda_environment"), field_name="worker.conda_environment"
            ),
            conda_prefix=_optional_string(
                payload.get("conda_prefix"), field_name="worker.conda_prefix"
            ),
            local_world_size=_require_int(
                payload.get("local_world_size", 1),
                field_name="worker.local_world_size",
            ),
            enabled=_require_bool(payload.get("enabled", True), field_name="worker.enabled"),
            labels=_require_string_mapping(payload.get("labels", {}), field_name="worker.labels"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.worker_id),
            "machine_id": str(self.machine_id),
            "physical_os": self.physical_os.value,
            "runtime_os": self.runtime_os.value,
            "runtime": self.runtime,
            "host": str(self.host),
            "ssh_user": self.ssh_user,
            "ssh_port": self.ssh_port,
            "runtime_distro": self.runtime_distro,
            "conda_environment": self.conda_environment,
            "conda_prefix": self.conda_prefix,
            "local_world_size": self.local_world_size,
            "enabled": self.enabled,
            "labels": dict(self.labels),
        }


@dataclass(frozen=True)
class ClusterConfig:
    control: ControlNodeConfig
    jobs_root: Path
    ssh: SSHConfig
    runtime: RuntimeConfig
    network: NetworkConfig
    backend_preference: BackendPreferenceConfig
    manual_override: ManualOverrideConfig
    workers: list[WorkerConfig]

    @classmethod
    def from_dict(cls, data: object) -> ClusterConfig:
        payload = _require_mapping(data, field_name="cluster_config")
        workers_data = payload.get("workers")
        if not isinstance(workers_data, list) or not workers_data:
            raise ConfigValidationError("workers must be a non-empty list")
        workers = [WorkerConfig.from_dict(item) for item in workers_data]
        worker_ids = [worker.worker_id for worker in workers]
        if len(worker_ids) != len(set(worker_ids)):
            raise ConfigValidationError("worker.id values must be unique")
        return cls(
            control=ControlNodeConfig.from_dict(payload.get("control")),
            jobs_root=_validate_jobs_root(payload.get("jobs_root")),
            ssh=SSHConfig.from_dict(payload.get("ssh", {})),
            runtime=RuntimeConfig.from_dict(payload.get("runtime", {})),
            network=NetworkConfig.from_dict(payload.get("network", {})),
            backend_preference=BackendPreferenceConfig.from_dict(
                payload.get("backend_preference", {})
            ),
            manual_override=ManualOverrideConfig.from_dict(
                payload.get("manual_override", {})
            ),
            workers=workers,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "control": _serialize(self.control),
            "jobs_root": str(self.jobs_root),
            "ssh": _serialize(self.ssh),
            "runtime": _serialize(self.runtime),
            "network": _serialize(self.network),
            "backend_preference": _serialize(self.backend_preference),
            "manual_override": _serialize(self.manual_override),
            "workers": [worker.to_dict() for worker in self.workers],
        }


@dataclass(frozen=True)
class TrainingJobConfig:
    name: str
    backend: BackendName
    communication_backend: BackendName


@dataclass(frozen=True)
class TrainingModelConfig:
    name: str
    type: str
    stage_count: int = 2
    max_train_minutes: int = 15
    min_loss_decrease_percent: float = 5.0


@dataclass(frozen=True)
class TrainingResourcesConfig:
    world_size: int
    preferred_workers: list[WorkerId] = field(default_factory=list)
    worker_overrides: dict[WorkerId, dict[str, str]] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingArtifactsConfig:
    snapshot_name: str | None = None
    keep_failed_snapshots: bool = True


@dataclass(frozen=True)
class TrainingConfig:
    job: TrainingJobConfig
    model: TrainingModelConfig
    resources: TrainingResourcesConfig
    artifacts: TrainingArtifactsConfig

    @classmethod
    def from_dict(cls, data: object) -> TrainingConfig:
        payload = _require_mapping(data, field_name="training_config")
        job_payload = _require_mapping(payload.get("job"), field_name="job")
        model_payload = _require_mapping(payload.get("model"), field_name="model")
        resources_payload = _require_mapping(payload.get("resources"), field_name="resources")
        artifacts_payload = _require_mapping(
            payload.get("artifacts", {}), field_name="artifacts"
        )
        return cls(
            job=TrainingJobConfig(
                name=_require_string(job_payload.get("name"), field_name="job.name"),
                backend=as_backend_name(
                    _require_string(job_payload.get("backend", "ssh"), field_name="job.backend")
                ),
                communication_backend=as_backend_name(
                    _require_string(
                        job_payload.get("communication_backend", "auto"),
                        field_name="job.communication_backend",
                    )
                ),
            ),
            model=TrainingModelConfig(
                name=_require_string(model_payload.get("name"), field_name="model.name"),
                type=_require_string(model_payload.get("type"), field_name="model.type"),
                stage_count=_require_int(
                    model_payload.get("stage_count", 2), field_name="model.stage_count"
                ),
                max_train_minutes=_require_int(
                    model_payload.get("max_train_minutes", 15),
                    field_name="model.max_train_minutes",
                ),
                min_loss_decrease_percent=float(
                    model_payload.get("min_loss_decrease_percent", 5.0)
                ),
            ),
            resources=TrainingResourcesConfig(
                world_size=_require_int(
                    resources_payload.get("world_size", 2),
                    field_name="resources.world_size",
                ),
                preferred_workers=[
                    as_worker_id(item)
                    for item in _require_string_list(
                        resources_payload.get("preferred_workers", []),
                        field_name="resources.preferred_workers",
                    )
                ],
                worker_overrides={
                    as_worker_id(key): _require_string_mapping(
                        value, field_name="resources.worker_overrides"
                    )
                    for key, value in _require_mapping(
                        resources_payload.get("worker_overrides", {}),
                        field_name="resources.worker_overrides",
                    ).items()
                },
            ),
            artifacts=TrainingArtifactsConfig(
                snapshot_name=_optional_string(
                    artifacts_payload.get("snapshot_name"),
                    field_name="artifacts.snapshot_name",
                ),
                keep_failed_snapshots=_require_bool(
                    artifacts_payload.get("keep_failed_snapshots", True),
                    field_name="artifacts.keep_failed_snapshots",
                ),
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


def load_config_data(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    text = config_path.read_text()
    if config_path.suffix == ".json":
        payload = json.loads(text)
    else:
        payload = yaml.safe_load(text)
    return _require_mapping(payload, field_name=str(config_path))


def load_cluster_config(path: str | Path) -> ClusterConfig:
    return ClusterConfig.from_dict(load_config_data(path))


def load_training_config(path: str | Path) -> TrainingConfig:
    return TrainingConfig.from_dict(load_config_data(path))
