"""Control-node doctor checks for Ubuntu Machine A.

Doctor is detect-and-report only.  It never installs, creates, or replaces
anything, and it reuses the existing config, process, and environment-detection
primitives instead of reimplementing a shell execution framework.
"""

from __future__ import annotations

import os
import platform
import shutil
import socket
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from shardgrid.common.config import ClusterConfig
from shardgrid.common.enums import Health
from shardgrid.common.process import ProcessResult, run_process
from shardgrid.workers.environment_report import detect_conda, detect_python, detect_tool_versions

MIN_DISK_FREE_BYTES = 10 * 1024**3
PROJECT_DEPS = ("shardgrid", "yaml", "pytest", "ruff", "mypy")

EXIT_OK = 0
EXIT_DEGRADED_OR_FAILED = 1
EXIT_MANUAL_ACTION = 2


@dataclass(frozen=True)
class DoctorCheck:
    name: str
    status: str
    detail: str | None = None
    manual_action_required: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
            "manual_action_required": self.manual_action_required,
        }


@dataclass(frozen=True)
class ControlDoctorReport:
    target: str
    host: str
    os_version: str
    timestamp: str
    environment: dict[str, Any]
    checks: tuple[DoctorCheck, ...]
    health: Health
    manual_actions: tuple[str, ...]
    commands_run: tuple[str, ...]
    exit_code: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "host": self.host,
            "os_version": self.os_version,
            "timestamp": self.timestamp,
            "environment": self.environment,
            "checks": [check.to_dict() for check in self.checks],
            "health": self.health.value,
            "manual_actions": list(self.manual_actions),
            "commands_run": list(self.commands_run),
            "exit_code": self.exit_code,
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
            env["PYTHONPATH"] = (
                f"{python_path}{os.pathsep}{existing}" if existing else python_path
            )
        probe = run_process(
            [python_executable, "-c", f"import {dep}"], env=env, timeout=15
        )
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


def run_control_doctor(config: ClusterConfig | None = None) -> ControlDoctorReport:
    checks: list[DoctorCheck] = []
    manual_actions: list[str] = []
    commands_run: list[str] = []

    os_version = platform.platform()
    host = os.uname().nodename if hasattr(os, "uname") else "unknown"
    checks.append(DoctorCheck(name="os", status="ok", detail=os_version))

    conda_executable, conda_environment, conda_prefix = detect_conda()
    env_names: list[str] = []
    env_paths: dict[str, str] = {}
    if conda_executable:
        version_result = _run([conda_executable, "--version"])
        commands_run.append(f"{conda_executable} --version")
        version = (version_result.stdout or version_result.stderr).strip()
        checks.append(
            DoctorCheck(
                name="conda",
                status="ok",
                detail=f"{conda_executable} {version}".strip(),
            )
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
        checks.append(
            DoctorCheck(
                name="conda_environments",
                status="ok",
                detail=", ".join(env_names) if env_names else "none",
            )
        )
        checks.append(
            DoctorCheck(
                name="conda_active",
                status="ok" if conda_environment else "degraded",
                detail=conda_environment or "no active environment (CONDA_DEFAULT_ENV unset)",
            )
        )
    else:
        checks.append(
            DoctorCheck(
                name="conda",
                status="fail",
                detail="conda not found on PATH",
                manual_action_required=True,
            )
        )
        manual_actions.append(
            "install Conda (requires elevated administrator action; doctor never installs it)"
        )

    python_executable, python_version = detect_python()
    if python_executable:
        checks.append(
            DoctorCheck(
                name="python",
                status="ok",
                detail=f"{python_executable} ({python_version})",
            )
        )
    else:
        checks.append(
            DoctorCheck(name="python", status="fail", detail="no python executable found")
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
        checks.append(
            DoctorCheck(name="selected_environment", status="ok", detail=selected_environment)
        )
    elif conda_executable:
        checks.append(
            DoctorCheck(
                name="selected_environment",
                status="degraded",
                detail="no existing environment satisfies project dependencies",
                manual_action_required=True,
            )
        )
        manual_actions.append(
            "create a ShardGrid Conda environment (conda create is a manual action)"
        )

    tools = detect_tool_versions()
    ssh_detail = tools.get("ssh", "not_checked")
    if ssh_detail != "not_installed":
        checks.append(DoctorCheck(name="openssh", status="ok", detail=ssh_detail))
    else:
        checks.append(
            DoctorCheck(
                name="openssh",
                status="fail",
                detail="openssh client not installed",
                manual_action_required=True,
            )
        )
        manual_actions.append(
            "install OpenSSH client (operator action: sudo apt-get install -y openssh-client)"
        )

    address = _local_address()
    checks.append(
        DoctorCheck(
            name="network",
            status="ok" if address else "degraded",
            detail=address or "no routable IPv4 address detected",
        )
    )

    disk_path = Path(config.jobs_root) if config else Path.home()
    try:
        free_bytes = shutil.disk_usage(disk_path).free
    except OSError:
        free_bytes = None
    if free_bytes is not None and free_bytes >= MIN_DISK_FREE_BYTES:
        disk_status = "ok"
    else:
        disk_status = "degraded"
    disk_detail = (
        f"{disk_path}: {free_bytes // (1024**3)} GiB free"
        if free_bytes is not None
        else f"{disk_path}: disk usage unavailable"
    )
    checks.append(DoctorCheck(name="disk", status=disk_status, detail=disk_detail))

    deps = _check_dependencies(python_executable)
    for dep, present in deps.items():
        checks.append(
            DoctorCheck(
                name=f"dependency:{dep}",
                status="ok" if present else "degraded",
                detail="present" if present else "missing",
            )
        )
    if not all(deps.values()):
        checks.append(
            DoctorCheck(
                name="project_dependencies",
                status="degraded",
                detail=(
                    "install project dependencies into the selected Conda environment "
                    "(pip install -e '.[dev,test]')"
                ),
            )
        )

    if config is None:
        checks.append(
            DoctorCheck(
                name="jobs_root",
                status="not_checked",
                detail="no --config provided; pass --config to validate the jobs root",
            )
        )
    else:
        jobs_root = config.jobs_root
        exists = jobs_root.exists()
        writable = os.access(jobs_root, os.W_OK) if exists else False
        if exists and writable:
            checks.append(DoctorCheck(name="jobs_root", status="ok", detail=str(jobs_root)))
        elif not exists:
            checks.append(
                DoctorCheck(
                    name="jobs_root",
                    status="degraded",
                    detail=(
                        f"{jobs_root} does not exist yet; operator creates it with: "
                        f"mkdir -p {jobs_root}"
                    ),
                )
            )
            manual_actions.append(f"create jobs root directory: mkdir -p {jobs_root}")
        else:
            checks.append(
                DoctorCheck(
                    name="jobs_root",
                    status="fail",
                    detail=f"{jobs_root} exists but is not writable",
                    manual_action_required=True,
                )
            )
            manual_actions.append(f"make jobs root writable: {jobs_root}")

    blocked = any(check.manual_action_required for check in checks)
    failed = any(check.status == "fail" for check in checks)
    degraded = any(check.status == "degraded" for check in checks)

    if blocked:
        health = Health.BLOCKED_MANUAL_ACTION
        exit_code = EXIT_MANUAL_ACTION
    elif failed:
        health = Health.FAILED
        exit_code = EXIT_DEGRADED_OR_FAILED
    elif degraded:
        health = Health.DEGRADED
        exit_code = EXIT_DEGRADED_OR_FAILED
    else:
        health = Health.HEALTHY
        exit_code = EXIT_OK

    environment = {
        "conda_executable": conda_executable,
        "conda_environments": env_names,
        "active_environment": conda_environment,
        "conda_prefix": conda_prefix,
        "selected_environment": selected_environment,
        "python_executable": python_executable,
        "python_version": python_version,
    }

    return ControlDoctorReport(
        target="control",
        host=host,
        os_version=os_version,
        timestamp=_now(),
        environment=environment,
        checks=tuple(checks),
        health=health,
        manual_actions=tuple(dict.fromkeys(manual_actions)),
        commands_run=tuple(commands_run),
        exit_code=exit_code,
    )