"""Persistent environment reports for control nodes and GPU Workers.

Environment reports reuse environment evidence already collected by doctor,
probe, or bootstrap flows.  The module never re-probes hardware itself; it
persists the evidence passed to it and records an explicit ``evidence_status``
so that missing or unverified platform evidence is never presented as a pass.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import sys
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

from shardgrid.common.config import WorkerConfig
from shardgrid.common.enums import Health, PhysicalOS, RuntimeOS, SerializableStrEnum
from shardgrid.common.models import Hostname, MachineId, as_hostname, as_machine_id
from shardgrid.common.process import run_process
from shardgrid.jobs.models import EnvironmentSnapshot, FailureRecord

_EVIDENCE_STATUSES = frozenset({"live", "mock", "pending"})


class ReportScope(SerializableStrEnum):
    CONTROL = "control"
    WORKER = "worker"


def _serialize(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "value"):
        return value.value
    if is_dataclass(value):
        return {key: _serialize(item) for key, item in asdict(cast(Any, value)).items()}
    if isinstance(value, dict):
        return {str(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(item) for item in value]
    return value


def _optional_int(data: dict[str, Any], key: str) -> int | None:
    value = data.get(key)
    return None if value is None else int(value)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class EnvironmentReport:
    """Stable, JSON-serializable record of detected environment evidence."""

    report_id: str
    scope: ReportScope
    target: str
    machine_id: MachineId
    hostname: Hostname
    physical_os: PhysicalOS
    runtime_os: RuntimeOS
    timestamp: str
    health: Health
    environment_manager: str = "conda"
    conda_executable: str | None = None
    conda_environment: str | None = None
    conda_prefix: str | None = None
    python_executable: str | None = None
    python_version: str | None = None
    torch_version: str | None = None
    torch_cuda_version: str | None = None
    cuda_version: str | None = None
    runtime_version: str | None = None
    gpu_name: str | None = None
    driver_version: str | None = None
    gpu_total_memory_mb: int | None = None
    compute_capability: str | None = None
    evidence_path: str | None = None
    commands: tuple[str, ...] = ()
    manual_actions: tuple[str, ...] = ()
    components: dict[str, str] = field(default_factory=dict)
    failure: FailureRecord | None = None
    evidence_status: str = "pending"

    def __post_init__(self) -> None:
        if not self.report_id or not self.target or not self.timestamp:
            raise ValueError("environment report requires report_id, target, and timestamp")
        if self.evidence_status not in _EVIDENCE_STATUSES:
            raise ValueError("evidence_status must be one of live, mock, pending")

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EnvironmentReport:
        required = (
            "report_id",
            "scope",
            "target",
            "machine_id",
            "hostname",
            "physical_os",
            "runtime_os",
            "timestamp",
            "health",
        )
        missing = [name for name in required if name not in data]
        if missing:
            raise ValueError(
                f"environment report missing required fields: {', '.join(sorted(missing))}"
            )
        failure = data.get("failure")
        return cls(
            report_id=str(data["report_id"]),
            scope=ReportScope(str(data["scope"])),
            target=str(data["target"]),
            machine_id=as_machine_id(str(data["machine_id"])),
            hostname=as_hostname(str(data["hostname"])),
            physical_os=PhysicalOS(str(data["physical_os"])),
            runtime_os=RuntimeOS(str(data["runtime_os"])),
            timestamp=str(data["timestamp"]),
            health=Health(str(data["health"])),
            environment_manager=str(data.get("environment_manager", "conda")),
            conda_executable=data.get("conda_executable"),
            conda_environment=data.get("conda_environment"),
            conda_prefix=data.get("conda_prefix"),
            python_executable=data.get("python_executable"),
            python_version=data.get("python_version"),
            torch_version=data.get("torch_version"),
            torch_cuda_version=data.get("torch_cuda_version"),
            cuda_version=data.get("cuda_version"),
            runtime_version=data.get("runtime_version"),
            gpu_name=data.get("gpu_name"),
            driver_version=data.get("driver_version"),
            gpu_total_memory_mb=_optional_int(data, "gpu_total_memory_mb"),
            compute_capability=data.get("compute_capability"),
            evidence_path=data.get("evidence_path"),
            commands=tuple(str(item) for item in data.get("commands", [])),
            manual_actions=tuple(str(item) for item in data.get("manual_actions", [])),
            components={
                str(key): str(value) for key, value in data.get("components", {}).items()
            },
            failure=None if failure is None else FailureRecord.from_dict(failure),
            evidence_status=str(data.get("evidence_status", "pending")),
        )

    @classmethod
    def from_snapshot(
        cls,
        snapshot: EnvironmentSnapshot,
        *,
        scope: ReportScope,
        target: str,
        machine_id: MachineId,
        hostname: Hostname,
        physical_os: PhysicalOS,
        runtime_os: RuntimeOS,
        timestamp: str | None = None,
        health: Health = Health.UNKNOWN,
        commands: Sequence[str] = (),
        manual_actions: Sequence[str] = (),
        evidence_status: str = "pending",
        components: Mapping[str, str] | None = None,
    ) -> EnvironmentReport:
        report_components = {"snapshot_id": snapshot.snapshot_id}
        report_components.update(
            {str(key): str(value) for key, value in (components or {}).items()}
        )
        return cls(
            report_id=f"{scope.value}-{target}",
            scope=scope,
            target=target,
            machine_id=machine_id,
            hostname=hostname,
            physical_os=physical_os,
            runtime_os=runtime_os,
            timestamp=timestamp or now_utc(),
            health=health,
            environment_manager=snapshot.environment_manager,
            conda_executable=snapshot.conda_executable,
            conda_environment=snapshot.conda_environment,
            conda_prefix=snapshot.conda_prefix,
            python_executable=snapshot.python_executable,
            python_version=snapshot.python_version,
            torch_version=snapshot.torch_version,
            torch_cuda_version=snapshot.torch_cuda_version,
            cuda_version=snapshot.cuda_version,
            commands=tuple(commands),
            manual_actions=tuple(manual_actions),
            components=report_components,
            evidence_status=evidence_status,
        )


def environment_report_filename(report: EnvironmentReport) -> str:
    return f"{report.scope.value}-{report.target}-environment-report.json"


def write_environment_report(report: EnvironmentReport, reports_root: str | Path) -> Path:
    root = Path(reports_root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / environment_report_filename(report)
    path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return path


def load_environment_report(path: str | Path) -> EnvironmentReport:
    return EnvironmentReport.from_dict(json.loads(Path(path).read_text()))


def find_environment_reports(reports_root: str | Path) -> list[Path]:
    return sorted(Path(reports_root).glob("*-environment-report.json"))


def detect_python() -> tuple[str | None, str | None]:
    return sys.executable, platform.python_version()


def detect_conda() -> tuple[str | None, str | None, str | None]:
    conda_executable = shutil.which("conda")
    conda_environment = os.environ.get("CONDA_DEFAULT_ENV")
    conda_prefix = os.environ.get("CONDA_PREFIX")
    return conda_executable, conda_environment, conda_prefix


_TOOL_VERSION_COMMANDS: dict[str, list[str]] = {
    "ssh": ["ssh", "-V"],
    "git": ["git", "--version"],
    "iperf3": ["iperf3", "--version"],
}


def detect_tool_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for tool, command in _TOOL_VERSION_COMMANDS.items():
        if shutil.which(tool) is None:
            versions[tool] = "not_installed"
            continue
        probe = run_process(command, timeout=10)
        if probe.ok:
            text = (probe.stdout or probe.stderr).strip()
            versions[tool] = text.splitlines()[0] if text else "installed"
        else:
            versions[tool] = "installed"
    return versions


def build_control_report(
    machine_id: MachineId,
    hostname: Hostname,
    *,
    timestamp: str | None = None,
    jobs_root: str | Path | None = None,
    commands: Sequence[str] = (),
    manual_actions: Sequence[str] = (),
) -> EnvironmentReport:
    """Build a live control-node environment report from real local detection."""
    conda_executable, conda_environment, conda_prefix = detect_conda()
    python_executable, python_version = detect_python()
    components = detect_tool_versions()
    components["os"] = platform.platform()
    try:
        components["disk_free_bytes_home"] = str(shutil.disk_usage(Path.home()).free)
    except OSError:
        components["disk_free_bytes_home"] = "unavailable"
    if jobs_root is not None:
        root = Path(jobs_root)
        components["jobs_root"] = str(root)
        components["jobs_root_state"] = "exists" if root.exists() else "not_created"

    return EnvironmentReport(
        report_id=f"control-{machine_id}",
        scope=ReportScope.CONTROL,
        target=f"control:{machine_id}",
        machine_id=machine_id,
        hostname=hostname,
        physical_os=PhysicalOS.LINUX,
        runtime_os=RuntimeOS.LINUX,
        timestamp=timestamp or now_utc(),
        health=Health.HEALTHY,
        conda_executable=conda_executable,
        conda_environment=conda_environment,
        conda_prefix=conda_prefix,
        python_executable=python_executable,
        python_version=python_version,
        commands=tuple(commands),
        manual_actions=tuple(manual_actions),
        components=components,
        evidence_status="live",
    )


def build_worker_report(
    worker: WorkerConfig,
    *,
    distro: str | None = None,
    health: Health = Health.UNKNOWN,
    evidence_status: str = "pending",
    runtime_facts: Mapping[str, str | None] | None = None,
    commands: Sequence[str] = (),
    manual_actions: Sequence[str] = (),
    failure: FailureRecord | None = None,
    evidence_path: str | None = None,
    components: Mapping[str, str] | None = None,
) -> EnvironmentReport:
    """Build a Worker environment report from identity plus runtime facts.

    The identity (machine_id, hostname, physical_os, runtime_os) comes from the
    configured Worker; runtime facts must be real evidence collected by earlier
    probe/bootstrap flows.  ``evidence_status`` keeps unverified platform
    evidence explicit instead of presenting it as a pass.
    """
    facts = dict(runtime_facts or {})
    gpu_memory = facts.get("gpu_total_memory_mb")
    return EnvironmentReport(
        report_id=f"worker-{worker.worker_id}",
        scope=ReportScope.WORKER,
        target=str(worker.worker_id),
        machine_id=worker.machine_id,
        hostname=as_hostname(worker.host),
        physical_os=worker.physical_os,
        runtime_os=worker.runtime_os,
        timestamp=now_utc(),
        health=health,
        conda_executable=facts.get("conda_executable"),
        conda_environment=facts.get("conda_environment"),
        conda_prefix=facts.get("conda_prefix"),
        python_executable=facts.get("python_executable"),
        python_version=facts.get("python_version"),
        torch_version=facts.get("torch_version"),
        torch_cuda_version=facts.get("torch_cuda_version"),
        cuda_version=facts.get("cuda_version"),
        runtime_version=distro or worker.runtime_distro,
        gpu_name=facts.get("gpu_name"),
        driver_version=facts.get("driver_version"),
        gpu_total_memory_mb=None if gpu_memory is None else int(gpu_memory),
        compute_capability=facts.get("compute_capability"),
        evidence_path=evidence_path,
        commands=tuple(commands),
        manual_actions=tuple(manual_actions),
        components={str(key): str(value) for key, value in (components or {}).items()},
        failure=failure,
        evidence_status=evidence_status,
    )


_SCHEMA_PATH = Path(
    "specs/001-multi-host-training-mvp/contracts/environment-report.schema.yaml"
)


def load_environment_report_schema() -> dict[str, Any]:
    import yaml

    return cast(dict[str, Any], yaml.safe_load(_SCHEMA_PATH.read_text()))


def validate_environment_report_payload(payload: dict[str, Any]) -> None:
    from jsonschema import Draft202012Validator

    from shardgrid.common.serialization import SchemaValidationError

    validator = Draft202012Validator(load_environment_report_schema())
    errors = sorted(
        validator.iter_errors(payload), key=lambda error: list(error.path)
    )
    if errors:
        message = "; ".join(error.message for error in errors)
        raise SchemaValidationError(message)


def validate_environment_report(report: EnvironmentReport) -> None:
    validate_environment_report_payload(report.to_dict())