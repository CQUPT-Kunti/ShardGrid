"""ShardGrid doctor readiness reports for control and WSL-backed GPU Workers."""

from __future__ import annotations

import json
import os
import platform
import shutil
import socket
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from shardgrid.bootstrap.runner import (
    BootstrapExecution,
    run_control_bootstrap,
    run_worker_runtime_bootstrap,
)
from shardgrid.common.config import ClusterConfig, WorkerConfig
from shardgrid.common.enums import Health, PhysicalOS, RuntimeOS
from shardgrid.common.process import ProcessResult, run_process
from shardgrid.transport.remote_access import RemoteAccessResult, run_remote_access_check
from shardgrid.transport.runtime import WSLRuntimeConfig, WSLRuntimeWrapper
from shardgrid.transport.ssh import SSHOptions, SSHTransport
from shardgrid.workers.environment_report import detect_conda, detect_python, detect_tool_versions

MIN_DISK_FREE_BYTES = 10 * 1024**3
PROJECT_DEPS = ("shardgrid", "yaml", "pytest", "ruff", "mypy")

EXIT_OK = 0
EXIT_DEGRADED_OR_FAILED = 1
EXIT_MANUAL_ACTION = 2

STATUS_PASS = "PASS"
STATUS_WARNING = "WARNING"
STATUS_FAIL = "FAIL"
STATUS_BLOCKED = "BLOCKED"
STATUS_UNAVAILABLE = "UNAVAILABLE"
STATUS_PENDING = "PENDING"


@dataclass(frozen=True)
class DoctorCheck:
    name: str
    status: str
    subject: str
    host: str
    runtime: str
    layer: str
    detected_value: Any = None
    expected_value: Any = None
    failure_reason: str | None = None
    recommended_action: str | None = None
    command: str | None = None
    manual_action_required: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "subject": self.subject,
            "host": self.host,
            "runtime": self.runtime,
            "layer": self.layer,
            "detected_value": self.detected_value,
            "expected_value": self.expected_value,
            "failure_reason": self.failure_reason,
            "recommended_action": self.recommended_action,
            "command": self.command,
            "manual_action_required": self.manual_action_required,
        }


@dataclass(frozen=True)
class DoctorSubjectReport:
    subject: str
    subject_type: str
    host: str
    runtime: str
    physical_os: str
    runtime_os: str
    timestamp: str
    checks: tuple[DoctorCheck, ...]
    environment: dict[str, Any]
    health: Health
    manual_actions: tuple[str, ...]
    commands_run: tuple[str, ...]
    exit_code: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "subject_type": self.subject_type,
            "host": self.host,
            "runtime": self.runtime,
            "physical_os": self.physical_os,
            "runtime_os": self.runtime_os,
            "timestamp": self.timestamp,
            "checks": [check.to_dict() for check in self.checks],
            "environment": self.environment,
            "health": self.health.value,
            "manual_actions": list(self.manual_actions),
            "commands_run": list(self.commands_run),
            "exit_code": self.exit_code,
        }


ControlDoctorReport = DoctorSubjectReport


@dataclass(frozen=True)
class DoctorReport:
    target: str
    generated_at: str
    subjects: tuple[DoctorSubjectReport, ...]
    health: Health
    exit_code: int
    checks: tuple[DoctorCheck, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "generated_at": self.generated_at,
            "subjects": [subject.to_dict() for subject in self.subjects],
            "health": self.health.value,
            "exit_code": self.exit_code,
            "checks": [check.to_dict() for check in self.checks],
        }


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run(command: list[str]) -> ProcessResult:
    return run_process(command, timeout=15)


def _check_dependencies(python_executable: str | None) -> dict[str, bool]:
    if not python_executable:
        return {dep: False for dep in PROJECT_DEPS}
    repo_root = _repo_root()
    result: dict[str, bool] = {}
    for dep in PROJECT_DEPS:
        env = dict(os.environ)
        if dep == "shardgrid":
            python_path = str(repo_root / "src")
            existing = env.get("PYTHONPATH")
            env["PYTHONPATH"] = f"{python_path}{os.pathsep}{existing}" if existing else python_path
        probe = run_process([python_executable, "-c", f"import {dep}"], env=env, timeout=15)
        result[dep] = probe.ok
    return result


def _local_address() -> str | None:
    probe = run_process(["hostname", "-I"], timeout=10)
    if probe.ok:
        addresses = [item for item in probe.stdout.split() if not item.startswith("127.")]
        if addresses:
            return addresses[0]
    try:
        hostname = socket.gethostname()
        address = socket.gethostbyname(hostname)
        return None if address.startswith("127.") else address
    except OSError:
        return None


def _status_for_subject(checks: list[DoctorCheck]) -> tuple[Health, int]:
    statuses = {check.status for check in checks}
    if any(check.manual_action_required for check in checks) or STATUS_BLOCKED in statuses:
        return Health.BLOCKED_MANUAL_ACTION, EXIT_MANUAL_ACTION
    if STATUS_FAIL in statuses:
        return Health.FAILED, EXIT_DEGRADED_OR_FAILED
    if STATUS_WARNING in statuses or STATUS_UNAVAILABLE in statuses or STATUS_PENDING in statuses:
        return Health.DEGRADED, EXIT_DEGRADED_OR_FAILED
    return Health.HEALTHY, EXIT_OK


def _aggregate_report(target: str, subjects: list[DoctorSubjectReport]) -> DoctorReport:
    checks = tuple(check for subject in subjects for check in subject.checks)
    if any(check.manual_action_required for check in checks) or any(
        check.status == STATUS_BLOCKED for check in checks
    ):
        health = Health.BLOCKED_MANUAL_ACTION
        exit_code = EXIT_MANUAL_ACTION
    elif any(check.status == STATUS_FAIL for check in checks):
        health = Health.FAILED
        exit_code = EXIT_DEGRADED_OR_FAILED
    elif any(
        check.status in {STATUS_WARNING, STATUS_UNAVAILABLE, STATUS_PENDING} for check in checks
    ):
        health = Health.DEGRADED
        exit_code = EXIT_DEGRADED_OR_FAILED
    else:
        health = Health.HEALTHY
        exit_code = EXIT_OK
    return DoctorReport(
        target=target,
        generated_at=_now(),
        subjects=tuple(subjects),
        health=health,
        exit_code=exit_code,
        checks=checks,
    )


def _add_check(
    checks: list[DoctorCheck],
    *,
    name: str,
    status: str,
    subject: str,
    host: str,
    runtime: str,
    layer: str,
    detected_value: Any = None,
    expected_value: Any = None,
    failure_reason: str | None = None,
    recommended_action: str | None = None,
    command: str | None = None,
    manual_action_required: bool = False,
) -> None:
    checks.append(
        DoctorCheck(
            name=name,
            status=status,
            subject=subject,
            host=host,
            runtime=runtime,
            layer=layer,
            detected_value=detected_value,
            expected_value=expected_value,
            failure_reason=failure_reason,
            recommended_action=recommended_action,
            command=command,
            manual_action_required=manual_action_required,
        )
    )


def run_control_doctor(
    config: ClusterConfig | None = None, *, fix: bool = False
) -> ControlDoctorReport:
    checks: list[DoctorCheck] = []
    manual_actions: list[str] = []
    commands_run: list[str] = []

    os_version = platform.platform()
    host = os.uname().nodename if hasattr(os, "uname") else "unknown"
    _add_check(
        checks,
        name="identity",
        status=STATUS_PASS,
        subject="control",
        host=host,
        runtime="control",
        layer="control",
        detected_value=os_version,
    )

    bootstrap = run_control_bootstrap(fix=fix)
    if bootstrap.before_state is None:
        _add_check(
            checks,
            name="bootstrap_runner",
            status=STATUS_BLOCKED,
            subject="control",
            host=host,
            runtime="control",
            layer="control",
            failure_reason=bootstrap.failure_reason,
            recommended_action="inspect bootstrap-linux.sh output and rerun doctor",
            command=bootstrap.command,
            manual_action_required=True,
        )
        if bootstrap.failure_reason:
            manual_actions.append(bootstrap.failure_reason)
    else:
        commands_run.extend(bootstrap.commands_run)
        bootstrap_state = bootstrap.effective_state or {}
        bootstrap_health = bootstrap_state.get("health")
        status = (
            STATUS_BLOCKED
            if bootstrap.execution == "blocked"
            else (
                STATUS_PASS
                if bootstrap.verified and bootstrap_health == Health.HEALTHY.value
                else STATUS_WARNING
            )
        )
        _add_check(
            checks,
            name="bootstrap_runner",
            status=status,
            subject="control",
            host=host,
            runtime="control",
            layer="control",
            detected_value={
                "action": bootstrap.action,
                "execution": bootstrap.execution,
                "health": bootstrap_health,
                "verified": bootstrap.verified,
            },
            failure_reason=bootstrap.failure_reason,
            recommended_action=bootstrap.manual_action,
            command=bootstrap.command,
            manual_action_required=bootstrap.execution == "blocked",
        )
        if bootstrap.manual_action:
            manual_actions.append(bootstrap.manual_action)

    conda_executable, conda_environment, conda_prefix = detect_conda()
    env_names: list[str] = []
    env_paths: dict[str, str] = {}
    if conda_executable:
        version_result = _run([conda_executable, "--version"])
        commands_run.append(f"{conda_executable} --version")
        _add_check(
            checks,
            name="conda_executable",
            status=STATUS_PASS,
            subject="control",
            host=host,
            runtime="control",
            layer="control",
            detected_value=conda_executable,
            command=f"{conda_executable} --version",
        )
        _add_check(
            checks,
            name="conda_version",
            status=STATUS_PASS,
            subject="control",
            host=host,
            runtime="control",
            layer="control",
            detected_value=(version_result.stdout or version_result.stderr).strip(),
        )
        env_result = _run([conda_executable, "env", "list"])
        commands_run.append(f"{conda_executable} env list")
        for line in env_result.stdout.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            env_names.append(parts[0])
            env_paths[parts[0]] = parts[-1]
        _add_check(
            checks,
            name="conda_environment_inventory",
            status=STATUS_PASS,
            subject="control",
            host=host,
            runtime="control",
            layer="control",
            detected_value=env_names,
            command=f"{conda_executable} env list",
        )
        _add_check(
            checks,
            name="conda_active_environment",
            status=STATUS_PASS if conda_environment else STATUS_WARNING,
            subject="control",
            host=host,
            runtime="control",
            layer="control",
            detected_value=conda_environment,
            failure_reason=(
                None if conda_environment else "CONDA_DEFAULT_ENV is unset on the control node"
            ),
            recommended_action=(
                None if conda_environment else "activate or select a compatible Conda environment"
            ),
        )
    else:
        action = "install Conda (requires elevated administrator action; doctor never installs it)"
        _add_check(
            checks,
            name="conda_executable",
            status=STATUS_BLOCKED,
            subject="control",
            host=host,
            runtime="control",
            layer="control",
            failure_reason="conda not found on PATH",
            recommended_action=action,
            manual_action_required=True,
        )
        manual_actions.append(action)

    python_executable, python_version = detect_python()
    if python_executable:
        _add_check(
            checks,
            name="python",
            status=STATUS_PASS,
            subject="control",
            host=host,
            runtime="control",
            layer="control",
            detected_value={"executable": python_executable, "version": python_version},
        )
    else:
        _add_check(
            checks,
            name="python",
            status=STATUS_FAIL,
            subject="control",
            host=host,
            runtime="control",
            layer="control",
            failure_reason="no python executable found",
            recommended_action="install Python in the selected Conda environment",
        )

    selected_environment: str | None = None
    if conda_executable and conda_prefix and python_executable:
        prefix_python = Path(conda_prefix).joinpath("bin", "python")
        if prefix_python.exists() and prefix_python.resolve() == Path(python_executable).resolve():
            selected_environment = conda_environment or "active"
    if selected_environment is None and conda_executable and env_paths:
        for name, path in env_paths.items():
            env_python = Path(path).joinpath("bin", "python")
            if env_python.exists() and _check_dependencies(str(env_python))["yaml"]:
                selected_environment = name
                break
    if selected_environment:
        _add_check(
            checks,
            name="selected_conda_environment",
            status=STATUS_PASS,
            subject="control",
            host=host,
            runtime="control",
            layer="control",
            detected_value=selected_environment,
            expected_value="reuse existing compatible environment",
        )
    elif conda_executable:
        action = "create a ShardGrid Conda environment (conda create is a manual action)"
        _add_check(
            checks,
            name="selected_conda_environment",
            status=STATUS_BLOCKED,
            subject="control",
            host=host,
            runtime="control",
            layer="control",
            failure_reason="no existing environment satisfies project dependencies",
            recommended_action=action,
            manual_action_required=True,
        )
        manual_actions.append(action)

    ssh_detail = detect_tool_versions().get("ssh", "not_checked")
    if ssh_detail != "not_installed":
        _add_check(
            checks,
            name="openssh",
            status=STATUS_PASS,
            subject="control",
            host=host,
            runtime="control",
            layer="control",
            detected_value=ssh_detail,
        )
    else:
        action = "install OpenSSH client (operator action: sudo apt-get install -y openssh-client)"
        _add_check(
            checks,
            name="openssh",
            status=STATUS_BLOCKED,
            subject="control",
            host=host,
            runtime="control",
            layer="control",
            failure_reason="OpenSSH client not installed",
            recommended_action=action,
            manual_action_required=True,
        )
        manual_actions.append(action)

    address = _local_address()
    _add_check(
        checks,
        name="network_readiness",
        status=STATUS_PASS if address else STATUS_WARNING,
        subject="control",
        host=host,
        runtime="control",
        layer="control",
        detected_value=address,
        failure_reason=None if address else "no routable IPv4 address detected",
        recommended_action=None
        if address
        else "verify the control node LAN interface and IPv4 routing",
    )

    disk_path = Path(config.jobs_root) if config else Path.home()
    try:
        free_bytes = shutil.disk_usage(disk_path).free
    except OSError:
        free_bytes = None
    _add_check(
        checks,
        name="disk_readiness",
        status=STATUS_PASS
        if free_bytes is not None and free_bytes >= MIN_DISK_FREE_BYTES
        else STATUS_WARNING,
        subject="control",
        host=host,
        runtime="control",
        layer="control",
        detected_value={"path": str(disk_path), "free_bytes": free_bytes},
        expected_value={"minimum_free_bytes": MIN_DISK_FREE_BYTES},
        failure_reason=(
            None
            if free_bytes is not None and free_bytes >= MIN_DISK_FREE_BYTES
            else "control node free disk is below the minimum readiness threshold"
        ),
        recommended_action=(
            None
            if free_bytes is not None and free_bytes >= MIN_DISK_FREE_BYTES
            else "free disk space on the control node"
        ),
    )

    deps = _check_dependencies(python_executable)
    for dep, present in deps.items():
        _add_check(
            checks,
            name=f"dependency:{dep}",
            status=STATUS_PASS if present else STATUS_WARNING,
            subject="control",
            host=host,
            runtime="control",
            layer="control",
            detected_value="present" if present else "missing",
            recommended_action=(
                None
                if present
                else (
                    "install project dependencies into the selected Conda "
                    "environment (pip install -e '.[dev,test]')"
                )
            ),
        )

    if config is None:
        _add_check(
            checks,
            name="jobs_root",
            status=STATUS_PENDING,
            subject="control",
            host=host,
            runtime="control",
            layer="control",
            failure_reason="no --config provided; jobs root was not validated",
            recommended_action="pass --config to validate the configured jobs root",
        )
    else:
        jobs_root = config.jobs_root
        exists = jobs_root.exists()
        writable = os.access(jobs_root, os.W_OK) if exists else False
        if exists and writable:
            _add_check(
                checks,
                name="jobs_root",
                status=STATUS_PASS,
                subject="control",
                host=host,
                runtime="control",
                layer="control",
                detected_value=str(jobs_root),
            )
        elif not exists:
            action = f"create jobs root directory: mkdir -p {jobs_root}"
            _add_check(
                checks,
                name="jobs_root",
                status=STATUS_WARNING,
                subject="control",
                host=host,
                runtime="control",
                layer="control",
                detected_value=str(jobs_root),
                failure_reason=f"{jobs_root} does not exist yet",
                recommended_action=action,
            )
            manual_actions.append(action)
        else:
            action = f"make jobs root writable: {jobs_root}"
            _add_check(
                checks,
                name="jobs_root",
                status=STATUS_BLOCKED,
                subject="control",
                host=host,
                runtime="control",
                layer="control",
                detected_value=str(jobs_root),
                failure_reason=f"{jobs_root} exists but is not writable",
                recommended_action=action,
                manual_action_required=True,
            )
            manual_actions.append(action)

    health, exit_code = _status_for_subject(checks)
    return ControlDoctorReport(
        subject="control",
        subject_type="control",
        host=host,
        runtime="control",
        physical_os=PhysicalOS.LINUX.value,
        runtime_os=RuntimeOS.LINUX.value,
        timestamp=_now(),
        checks=tuple(checks),
        environment={
            "conda_executable": conda_executable,
            "conda_environments": env_names,
            "active_environment": conda_environment,
            "conda_prefix": conda_prefix,
            "selected_environment": selected_environment,
            "python_executable": python_executable,
            "python_version": python_version,
        },
        health=health,
        manual_actions=tuple(dict.fromkeys(manual_actions)),
        commands_run=tuple(commands_run),
        exit_code=exit_code,
    )


def _build_transport(config: ClusterConfig, worker: WorkerConfig) -> SSHTransport:
    return SSHTransport(
        SSHOptions.from_ssh_config(
            config.ssh,
            host=str(worker.host),
            user=worker.ssh_user,
            port=worker.ssh_port,
        )
    )


def _bootstrap_worker_runtime(
    wrapper: WSLRuntimeWrapper,
    *,
    peer_ip: str,
    expected_mtu: int,
    fix: bool,
) -> BootstrapExecution:
    return run_worker_runtime_bootstrap(
        wrapper,
        peer_ip=peer_ip,
        expected_mtu=expected_mtu,
        fix=fix,
    )


def _runtime_probe_script() -> str:
    return """
import ctypes.util
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

payload = {
    "python_executable": sys.executable,
    "python_version": platform.python_version(),
    "torch_version": None,
    "torch_cuda_version": None,
    "cuda_available": False,
    "nccl_available": False,
    "nccl_version": None,
    "nccl_lib_path": None,
    "gloo_available": False,
    "gpu_name": None,
    "gpu_total_memory_mb": None,
    "compute_capability": None,
    "driver_version": None,
    "nvidia_smi_path": shutil.which("nvidia-smi"),
    "disk_free_bytes": shutil.disk_usage("/").free,
}

try:
    import torch
    payload["torch_version"] = getattr(torch, "__version__", None)
    payload["torch_cuda_version"] = getattr(getattr(torch, "version", None), "cuda", None)
    payload["cuda_available"] = bool(torch.cuda.is_available())
    try:
        import torch.distributed as dist
        payload["gloo_available"] = bool(dist.is_gloo_available())
    except Exception:
        payload["gloo_available"] = False
    if payload["cuda_available"]:
        try:
            nccl_version = torch.cuda.nccl.version()
            payload["nccl_available"] = bool(nccl_version)
            if nccl_version:
                payload["nccl_version"] = ".".join(str(item) for item in nccl_version)
        except Exception:
            payload["nccl_available"] = False
        try:
            props = torch.cuda.get_device_properties(0)
            payload["gpu_name"] = props.name
            payload["gpu_total_memory_mb"] = props.total_memory // (1024 * 1024)
            payload["compute_capability"] = ".".join(
                str(item) for item in torch.cuda.get_device_capability(0)
            )
        except Exception:
            pass
except Exception:
    pass

candidate = ctypes.util.find_library("nccl")
search_paths = [
    Path(sys.prefix) / "lib",
    Path("/usr/lib/x86_64-linux-gnu"),
    Path("/usr/local/cuda/lib64"),
    Path("/usr/lib/wsl/lib"),
]
for directory in search_paths:
    try:
        matches = sorted(directory.glob("libnccl.so*"))
    except Exception:
        continue
    if matches:
        payload["nccl_lib_path"] = str(matches[0])
        break
if payload["nccl_lib_path"] is None and candidate and "/" in candidate:
    payload["nccl_lib_path"] = candidate
if payload["nvidia_smi_path"]:
    try:
        result = subprocess.run(
            [
                payload["nvidia_smi_path"],
                "--query-gpu=name,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        lines = (result.stdout or result.stderr).strip().splitlines()
        if lines:
            parts = [item.strip() for item in lines[0].split(",")]
            if parts and not payload["gpu_name"]:
                payload["gpu_name"] = parts[0]
            if len(parts) > 1:
                payload["driver_version"] = parts[1]
    except Exception:
        pass
print(json.dumps(payload))
"""


def _probe_runtime_details(wrapper: WSLRuntimeWrapper) -> dict[str, Any]:
    result = wrapper.run_script(_runtime_probe_script(), timeout=60)
    if not result.ok:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "runtime probe failed")
    return json.loads(result.stdout)


def _build_runtime_wrapper(
    transport: SSHTransport,
    worker: WorkerConfig,
    access: RemoteAccessResult,
) -> WSLRuntimeWrapper:
    identity = access.runtime_identity
    if identity is None:
        raise ValueError("runtime identity unavailable")
    return WSLRuntimeWrapper(
        WSLRuntimeConfig(
            distro=identity.wsl_distro,
            user=worker.ssh_user,
            conda_executable=identity.conda_executable,
            conda_environment=identity.conda_environment,
            conda_prefix=identity.conda_prefix,
        ),
        transport,
    )


def _worker_peers(config: ClusterConfig, worker: WorkerConfig) -> list[WorkerConfig]:
    return [
        candidate
        for candidate in config.workers
        if candidate.enabled and str(candidate.worker_id) != str(worker.worker_id)
    ]


def _worker_environment(
    access: RemoteAccessResult,
    bootstrap: dict[str, Any] | None,
    runtime_probe: dict[str, Any] | None,
) -> dict[str, Any]:
    conda = (bootstrap or {}).get("conda") or {}
    python = (bootstrap or {}).get("python") or {}
    return {
        "windows_identity": access.windows_identity,
        "wsl_distro": access.wsl_distro,
        "conda_executable": conda.get("executable")
        or (access.runtime_identity.conda_executable if access.runtime_identity else None),
        "conda_environments": conda.get("environments") or [],
        "active_environment": conda.get("active_environment"),
        "conda_prefix": conda.get("selected_prefix")
        or (access.runtime_identity.conda_prefix if access.runtime_identity else None),
        "selected_environment": conda.get("selected_environment")
        or (access.runtime_identity.conda_environment if access.runtime_identity else None),
        "python_executable": (runtime_probe or {}).get("python_executable")
        or python.get("executable")
        or (access.runtime_identity.python_executable if access.runtime_identity else None),
        "python_version": (runtime_probe or {}).get("python_version")
        or python.get("version")
        or (access.runtime_identity.python_version if access.runtime_identity else None),
    }


def _run_worker_doctor(
    config: ClusterConfig,
    worker: WorkerConfig,
    *,
    fix: bool,
) -> DoctorSubjectReport:
    subject = str(worker.worker_id)
    host = str(worker.host)
    runtime = worker.runtime or "wsl2"
    checks: list[DoctorCheck] = []
    manual_actions: list[str] = []
    commands_run: list[str] = []

    transport = _build_transport(config, worker)
    access = run_remote_access_check(
        transport,
        worker,
        worker_label=str(worker.labels.get("gpu") or worker.worker_id),
        preferred_environment=worker.conda_environment or config.runtime.conda_environment,
    )
    commands_run.extend(access.commands)

    _add_check(
        checks,
        name="windows_identity",
        status=STATUS_PASS if access.windows_identity else STATUS_UNAVAILABLE,
        subject=subject,
        host=host,
        runtime=runtime,
        layer="windows_host",
        detected_value=access.windows_identity,
        failure_reason=None if access.windows_identity else "Windows host identity unavailable",
    )
    _add_check(
        checks,
        name="ssh_access",
        status=STATUS_PASS if access.status == "PASS" else STATUS_BLOCKED,
        subject=subject,
        host=host,
        runtime=runtime,
        layer="windows_host",
        detected_value=access.status,
        failure_reason=access.failure_reason,
        recommended_action=(
            None
            if access.failure_record is None
            else access.failure_record.get("recommended_action")
        ),
        manual_action_required=access.status != "PASS",
    )
    _add_check(
        checks,
        name="wsl_runtime_access",
        status=STATUS_PASS if access.wsl_distro else STATUS_PENDING,
        subject=subject,
        host=host,
        runtime=runtime,
        layer="wsl_runtime",
        detected_value=access.wsl_distro,
        failure_reason=None if access.wsl_distro else "WSL runtime was not reached",
        recommended_action=(
            None
            if access.failure_record is None
            else access.failure_record.get("recommended_action")
        ),
        manual_action_required=access.status == "BLOCKED",
    )

    bootstrap: dict[str, Any] | None = None
    runtime_probe: dict[str, Any] | None = None
    if access.failure_record is not None:
        message = access.failure_record.get("message")
        if message:
            manual_actions.append(str(message))

    if access.status == "PASS" and access.runtime_identity is not None:
        wrapper = _build_runtime_wrapper(transport, worker, access)
        peers = _worker_peers(config, worker)
        if not peers:
            _add_check(
                checks,
                name="peer_route",
                status=STATUS_PENDING,
                subject=subject,
                host=host,
                runtime=runtime,
                layer="network",
                failure_reason="no peer worker configured; NCCL path MTU was not checked",
            )
        for peer in peers:
            peer_ip = str(peer.host)
            bootstrap_run = _bootstrap_worker_runtime(
                wrapper,
                peer_ip=peer_ip,
                expected_mtu=config.network.nccl_mtu,
                fix=False,
            )
            bootstrap = bootstrap_run.effective_state or {}
            commands_run.extend(bootstrap_run.commands_run)
            conda = bootstrap.get("conda") or {}
            python = bootstrap.get("python") or {}
            runtime_tools = bootstrap.get("runtime_tools") or {}
            project_dependencies = bootstrap.get("project_dependencies") or {}
            mtu = bootstrap.get("nccl_path_mtu") or {}
            peer_manual_actions = [str(item) for item in bootstrap.get("manual_actions", [])]

            _add_check(
                checks,
                name="conda_executable",
                status=STATUS_PASS if conda.get("executable") else STATUS_BLOCKED,
                subject=subject,
                host=host,
                runtime=runtime,
                layer="wsl_runtime",
                detected_value=conda.get("executable"),
                failure_reason=None
                if conda.get("executable")
                else "Conda unavailable in WSL runtime",
                recommended_action=(
                    None
                    if conda.get("executable")
                    else "install or expose Conda inside the WSL training runtime"
                ),
                manual_action_required=not bool(conda.get("executable")),
            )
            _add_check(
                checks,
                name="selected_conda_environment",
                status=STATUS_PASS if conda.get("selected_environment") else STATUS_BLOCKED,
                subject=subject,
                host=host,
                runtime=runtime,
                layer="wsl_runtime",
                detected_value={
                    "environment": conda.get("selected_environment"),
                    "prefix": conda.get("selected_prefix"),
                },
                expected_value="reuse existing compatible shardgrid environment",
                failure_reason=(
                    None
                    if conda.get("selected_environment")
                    else "no compatible Conda environment selected for WSL runtime"
                ),
                recommended_action=(
                    None
                    if conda.get("selected_environment")
                    else "select or create a compatible shardgrid Conda environment manually"
                ),
                manual_action_required=not bool(conda.get("selected_environment")),
            )
            _add_check(
                checks,
                name="python",
                status=STATUS_PASS if python.get("executable") else STATUS_FAIL,
                subject=subject,
                host=host,
                runtime=runtime,
                layer="wsl_runtime",
                detected_value={
                    "executable": python.get("executable"),
                    "version": python.get("version"),
                },
                failure_reason=None if python.get("executable") else "runtime Python unavailable",
                recommended_action=None
                if python.get("executable")
                else "repair the selected Conda environment Python",
            )
            _add_check(
                checks,
                name="basic_runtime_readiness",
                status=STATUS_PASS
                if runtime_tools.get("iperf3") not in {None, "not_installed"}
                else STATUS_WARNING,
                subject=subject,
                host=host,
                runtime=runtime,
                layer="wsl_runtime",
                detected_value=runtime_tools,
                failure_reason=(
                    None
                    if runtime_tools.get("iperf3") not in {None, "not_installed"}
                    else "iperf3 not installed in WSL runtime"
                ),
                recommended_action=(
                    None
                    if runtime_tools.get("iperf3") not in {None, "not_installed"}
                    else "install iperf3 inside the Ubuntu WSL2 runtime"
                ),
            )
            _add_check(
                checks,
                name="project_dependencies",
                status=STATUS_PASS
                if project_dependencies.get("status") == "present"
                else STATUS_WARNING,
                subject=subject,
                host=host,
                runtime=runtime,
                layer="wsl_runtime",
                detected_value=project_dependencies,
                failure_reason=(
                    None
                    if project_dependencies.get("status") == "present"
                    else "ShardGrid runtime dependencies are incomplete"
                ),
                recommended_action=(
                    None
                    if project_dependencies.get("status") == "present"
                    else "install ShardGrid dependencies into the selected WSL Conda environment"
                ),
            )
            _add_check(
                checks,
                name="peer_route",
                status=STATUS_PASS if mtu.get("interface") else STATUS_FAIL,
                subject=subject,
                host=host,
                runtime=runtime,
                layer="network",
                detected_value={
                    "peer": peer_ip,
                    "interface": mtu.get("interface"),
                    "route_output": mtu.get("route_output"),
                },
                failure_reason=None
                if mtu.get("interface")
                else f"ip route get {peer_ip} did not resolve a device",
                recommended_action=None
                if mtu.get("interface")
                else f"verify WSL routing to peer {peer_ip}",
            )

            mtu_status = (
                STATUS_PASS if str(mtu.get("status")).upper() == STATUS_PASS else STATUS_FAIL
            )
            mtu_failure_reason = (
                None
                if mtu_status == STATUS_PASS
                else (
                    f"NCCL_PATH_MTU_UNSAFE: peer={peer_ip} "
                    f"interface={mtu.get('interface')} "
                    "interface_mtu="
                    f"{mtu.get('interface_mtu_after') or mtu.get('interface_mtu_before')} "
                    f"expected_mtu={config.network.nccl_mtu}"
                )
            )
            mtu_recommended_action = (
                None
                if mtu_status == STATUS_PASS
                else (
                    "rerun `shardgrid doctor --target workers --fix` "
                    "or set the actual peer interface MTU to 1500"
                )
            )
            mtu_manual_action_required = False
            if mtu_status != STATUS_PASS and fix:
                fix_run = _bootstrap_worker_runtime(
                    wrapper,
                    peer_ip=peer_ip,
                    expected_mtu=config.network.nccl_mtu,
                    fix=True,
                )
                commands_run.extend(fix_run.commands_run)
                bootstrap = fix_run.effective_state or bootstrap
                mtu = bootstrap.get("nccl_path_mtu") or {}
                peer_manual_actions = [str(item) for item in bootstrap.get("manual_actions", [])]
                if (
                    fix_run.verified
                    and fix_run.execution != "blocked"
                    and str(mtu.get("status")).upper() == STATUS_PASS
                ):
                    mtu_status = STATUS_PASS
                    mtu_failure_reason = None
                    mtu_recommended_action = None
                else:
                    mtu_status = (
                        STATUS_BLOCKED if fix_run.execution == "blocked" else STATUS_WARNING
                    )
                    mtu_failure_reason = (
                        fix_run.failure_reason
                        or (
                            f"NCCL_PATH_MTU_UNSAFE: peer={peer_ip} "
                            f"interface={mtu.get('interface')} "
                            "interface_mtu="
                            f"{mtu.get('interface_mtu_after') or mtu.get('interface_mtu_before')} "
                            f"expected_mtu={config.network.nccl_mtu}"
                        )
                    )
                    mtu_recommended_action = (
                        fix_run.manual_action
                        or (
                            "rerun with root/CAP_NET_ADMIN or set the actual peer "
                            "interface MTU manually"
                        )
                    )
                    mtu_manual_action_required = fix_run.execution == "blocked"

            manual_actions.extend(peer_manual_actions)

            _add_check(
                checks,
                name="nccl_path_mtu",
                status=mtu_status,
                subject=subject,
                host=host,
                runtime=runtime,
                layer="network",
                detected_value={
                    "peer": peer_ip,
                    "interface": mtu.get("interface"),
                    "interface_mtu": mtu.get("interface_mtu_after")
                    or mtu.get("interface_mtu_before"),
                },
                expected_value={"interface_mtu": config.network.nccl_mtu},
                failure_reason=mtu_failure_reason,
                recommended_action=mtu_recommended_action,
                manual_action_required=mtu_manual_action_required,
            )
            _add_check(
                checks,
                name="network_readiness",
                status=STATUS_PASS if mtu_status == STATUS_PASS else STATUS_WARNING,
                subject=subject,
                host=host,
                runtime=runtime,
                layer="network",
                detected_value={
                    "peer": peer_ip,
                    "df_1472": mtu.get("df_1472"),
                    "df_1473": mtu.get("df_1473"),
                    "mtu_status": mtu.get("status"),
                },
                failure_reason=None
                if mtu_status == STATUS_PASS
                else "peer route or MTU is not ready for NCCL",
                recommended_action=None
                if mtu_status == STATUS_PASS
                else "fix the real peer route/interface MTU before NCCL workloads",
            )

        runtime_probe = _probe_runtime_details(wrapper)
        _add_check(
            checks,
            name="pytorch",
            status=STATUS_PASS if runtime_probe.get("torch_version") else STATUS_FAIL,
            subject=subject,
            host=host,
            runtime=runtime,
            layer="wsl_runtime",
            detected_value=runtime_probe.get("torch_version"),
            failure_reason=None
            if runtime_probe.get("torch_version")
            else "PyTorch unavailable in the selected runtime",
            recommended_action=None
            if runtime_probe.get("torch_version")
            else "install PyTorch into the selected WSL Conda environment",
        )
        _add_check(
            checks,
            name="cuda_availability",
            status=STATUS_PASS if runtime_probe.get("cuda_available") else STATUS_FAIL,
            subject=subject,
            host=host,
            runtime=runtime,
            layer="wsl_runtime",
            detected_value=runtime_probe.get("cuda_available"),
            expected_value=True,
            failure_reason=None
            if runtime_probe.get("cuda_available")
            else "torch.cuda.is_available() returned False",
            recommended_action=None
            if runtime_probe.get("cuda_available")
            else "repair CUDA visibility inside the WSL training runtime",
        )
        _add_check(
            checks,
            name="gpu",
            status=STATUS_PASS if runtime_probe.get("gpu_name") else STATUS_UNAVAILABLE,
            subject=subject,
            host=host,
            runtime=runtime,
            layer="wsl_runtime",
            detected_value={
                "name": runtime_probe.get("gpu_name"),
                "memory_mb": runtime_probe.get("gpu_total_memory_mb"),
                "compute_capability": runtime_probe.get("compute_capability"),
                "driver_version": runtime_probe.get("driver_version"),
            },
            failure_reason=None
            if runtime_probe.get("gpu_name")
            else "GPU was not verified from the real WSL runtime",
        )
        _add_check(
            checks,
            name="nccl_availability",
            status=STATUS_PASS if runtime_probe.get("nccl_available") else STATUS_FAIL,
            subject=subject,
            host=host,
            runtime=runtime,
            layer="wsl_runtime",
            detected_value={
                "available": runtime_probe.get("nccl_available"),
                "version": runtime_probe.get("nccl_version"),
                "lib_path": runtime_probe.get("nccl_lib_path"),
            },
            failure_reason=None
            if runtime_probe.get("nccl_available")
            else "NCCL unavailable in the selected runtime",
            recommended_action=None
            if runtime_probe.get("nccl_available")
            else "repair the PyTorch CUDA/NCCL runtime inside the selected Conda environment",
        )
        _add_check(
            checks,
            name="gloo_availability",
            status=STATUS_PASS if runtime_probe.get("gloo_available") else STATUS_FAIL,
            subject=subject,
            host=host,
            runtime=runtime,
            layer="wsl_runtime",
            detected_value=runtime_probe.get("gloo_available"),
            failure_reason=None
            if runtime_probe.get("gloo_available")
            else "Gloo unavailable in the selected runtime",
        )
        disk_free = runtime_probe.get("disk_free_bytes")
        _add_check(
            checks,
            name="disk_readiness",
            status=STATUS_PASS
            if isinstance(disk_free, int) and disk_free >= MIN_DISK_FREE_BYTES
            else STATUS_WARNING,
            subject=subject,
            host=host,
            runtime=runtime,
            layer="wsl_runtime",
            detected_value={"free_bytes": disk_free},
            expected_value={"minimum_free_bytes": MIN_DISK_FREE_BYTES},
            failure_reason=None
            if isinstance(disk_free, int) and disk_free >= MIN_DISK_FREE_BYTES
            else "WSL runtime free disk is below the minimum readiness threshold",
            recommended_action=None
            if isinstance(disk_free, int) and disk_free >= MIN_DISK_FREE_BYTES
            else "free disk space inside the WSL runtime before training",
        )

    health, exit_code = _status_for_subject(checks)
    return DoctorSubjectReport(
        subject=subject,
        subject_type="worker",
        host=host,
        runtime=runtime,
        physical_os=worker.physical_os.value,
        runtime_os=worker.runtime_os.value,
        timestamp=_now(),
        checks=tuple(checks),
        environment=_worker_environment(access, bootstrap, runtime_probe),
        health=health,
        manual_actions=tuple(dict.fromkeys(manual_actions)),
        commands_run=tuple(dict.fromkeys(commands_run)),
        exit_code=exit_code,
    )


def run_doctor(
    target: str,
    *,
        config: ClusterConfig | None = None,
        fix: bool = False,
    ) -> DoctorReport:
    subjects: list[DoctorSubjectReport] = []
    if target in {"control", "all"}:
        try:
            subjects.append(run_control_doctor(config, fix=fix))
        except TypeError:
            subjects.append(run_control_doctor(config))
    if target in {"workers", "all"}:
        if config is None:
            raise ValueError("doctor target workers/all requires --config")
        for worker in config.workers:
            if worker.enabled:
                subjects.append(_run_worker_doctor(config, worker, fix=fix))
    return _aggregate_report(target, subjects)
