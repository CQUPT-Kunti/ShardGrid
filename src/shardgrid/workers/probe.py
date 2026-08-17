"""Worker probe orchestrator for Windows GPU Workers with WSL2 training runtimes.

The probe keeps the Windows physical host and the WSL2 Linux training runtime
strictly separate: physical identity is recorded on ``WorkerResource`` /
``WindowsHostInfo`` while the training environment (Conda, Python, PyTorch,
CUDA, NCCL, Gloo, GPU, interface) comes only from the selected WSL2 Conda
runtime.  Failures are preserved with their real reasons; partial failures
never silently fall back to healthy values.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any

from shardgrid.common.config import WorkerConfig
from shardgrid.common.enums import Health, PhysicalOS, RuntimeOS
from shardgrid.common.models import as_hostname
from shardgrid.resources.models import WorkerResource
from shardgrid.workers import runtime_checks as checks
from shardgrid.workers.models import WorkerRuntime

FAILED = Health.FAILED


@dataclass(frozen=True)
class WindowsHostInfo:
    os_version: str | None
    openssh_available: bool
    wsl_available: bool
    nvidia_driver_visible: bool
    driver_name: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "os_version": self.os_version,
            "openssh_available": self.openssh_available,
            "wsl_available": self.wsl_available,
            "nvidia_driver_visible": self.nvidia_driver_visible,
            "driver_name": self.driver_name,
        }


@dataclass(frozen=True)
class ProbeFailure:
    layer: str
    check: str
    message: str
    exit_code: int | None = None
    output: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "check": self.check,
            "message": self.message,
            "exit_code": self.exit_code,
            "output": self.output,
        }


@dataclass(frozen=True)
class WorkerProbeResult:
    worker_resource: WorkerResource
    worker_runtime: WorkerRuntime
    windows_host: WindowsHostInfo
    failures: tuple[ProbeFailure, ...]
    health: Health
    probe_status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "worker_resource": self.worker_resource.to_dict(),
            "worker_runtime": self.worker_runtime.to_dict(),
            "windows_host": self.windows_host.to_dict(),
            "failures": [failure.to_dict() for failure in self.failures],
            "health": self.health.value,
            "probe_status": self.probe_status,
        }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _record_failure(
    failures: list[ProbeFailure],
    outcome: checks.CheckOutcome,
    *,
    layer: str,
    check: str,
    message: str,
) -> None:
    if not outcome.ok:
        failures.append(
            ProbeFailure(
                layer=layer,
                check=check,
                message=message,
                exit_code=outcome.exit_code,
                output=outcome.value,
            )
        )


def _select_environment(
    conda_environment: str | None,
    conda_prefix: str | None,
    env_names: list[str],
) -> tuple[str | None, str | None]:
    if conda_environment and conda_prefix:
        return conda_environment, conda_prefix
    if env_names:
        selected = env_names[0]
        return selected, None
    return None, None


def probe_worker(
    worker: WorkerConfig,
    windows_runner: checks.Runner,
    wsl_runner: checks.Runner,
    *,
    distro: str | None = None,
    probe_status: str = "live",
) -> WorkerProbeResult:
    failures: list[ProbeFailure] = []
    selected_distro = distro or worker.runtime_distro

    windows_os = checks.probe_windows_os(windows_runner)
    openssh = checks.probe_windows_openssh(windows_runner)
    wsl_available = checks.probe_windows_wsl_available(windows_runner)
    nvidia_driver = checks.probe_windows_nvidia_driver(windows_runner)

    _record_failure(
        failures, windows_os, layer="windows_host", check="os", message="OS detection failed"
    )
    _record_failure(
        failures,
        openssh,
        layer="windows_host",
        check="openssh",
        message="OpenSSH detection failed",
    )
    _record_failure(
        failures,
        wsl_available,
        layer="windows_host",
        check="wsl_available",
        message="WSL2 is not available on the Windows host",
    )
    _record_failure(
        failures,
        nvidia_driver,
        layer="windows_host",
        check="nvidia_driver",
        message="NVIDIA driver is not visible on the Windows host",
    )

    host_info = WindowsHostInfo(
        os_version=windows_os.value if windows_os.ok else None,
        openssh_available=openssh.ok and bool(openssh.value),
        wsl_available=wsl_available.ok,
        nvidia_driver_visible=nvidia_driver.ok and bool(nvidia_driver.value),
        driver_name=nvidia_driver.value if nvidia_driver.ok else None,
    )

    runtime = WorkerRuntime(
        worker_id=worker.worker_id,
        runtime_os=RuntimeOS.WSL2_LINUX,
        runtime_version=selected_distro,
        path_style="posix",
        health=FAILED,
    )
    resource = WorkerResource(
        worker_id=worker.worker_id,
        hostname=as_hostname(worker.host),
        physical_os=PhysicalOS.WINDOWS,
        runtime_os=RuntimeOS.WSL2_LINUX,
        health=FAILED,
    )

    if not host_info.wsl_available:
        return WorkerProbeResult(
            worker_resource=resource,
            worker_runtime=runtime,
            windows_host=host_info,
            failures=tuple(failures),
            health=FAILED,
            probe_status=probe_status,
        )

    conda_exe = checks.probe_wsl_conda_executable(wsl_runner)
    env_list = checks.probe_wsl_conda_env_list(wsl_runner)
    active = checks.probe_wsl_conda_active(wsl_runner)

    _record_failure(
        failures,
        conda_exe,
        layer="wsl_runtime",
        check="conda_executable",
        message="Conda is not available in the WSL2 training runtime",
    )
    _record_failure(
        failures,
        env_list,
        layer="wsl_runtime",
        check="conda_env_list",
        message="Conda environment list could not be read",
    )
    _record_failure(
        failures,
        active,
        layer="wsl_runtime",
        check="conda_active",
        message="Active Conda environment could not be read",
    )

    env_names = checks.parse_env_list(env_list.value) if env_list.ok else []
    active_env, active_prefix = checks.parse_active_env(active.value) if active.ok else (None, None)
    selected_env, selected_prefix = _select_environment(active_env, active_prefix, env_names)

    runtime = WorkerRuntime(
        worker_id=worker.worker_id,
        runtime_os=RuntimeOS.WSL2_LINUX,
        runtime_version=selected_distro,
        conda_executable=conda_exe.value if conda_exe.ok else None,
        conda_environment=selected_env,
        conda_prefix=selected_prefix,
        conda_active=active_env is not None,
        health=FAILED,
    )
    resource = WorkerResource(
        worker_id=worker.worker_id,
        hostname=as_hostname(worker.host),
        physical_os=PhysicalOS.WINDOWS,
        runtime_os=RuntimeOS.WSL2_LINUX,
        conda_environment=selected_env,
        conda_prefix=selected_prefix,
        health=FAILED,
    )

    if conda_exe.ok and selected_env:
        python_executable = (
            f"{selected_prefix}/bin/python" if selected_prefix else "python"
        )
        python_version = checks.probe_wsl_python_version(wsl_runner, python_executable)
        torch = checks.probe_wsl_torch(wsl_runner, python_executable)

        _record_failure(
            failures,
            python_version,
            layer="wsl_runtime",
            check="python",
            message=f"Python could not be detected in Conda environment {selected_env}",
        )
        _record_failure(
            failures,
            torch,
            layer="wsl_runtime",
            check="pytorch",
            message="PyTorch is not available in the selected Conda environment",
        )

        torch_values = checks.parse_key_values(torch.value) if torch.ok else {}
        cuda_available = torch_values.get("CUDA_AVAILABLE") == "True"
        torch_version = torch_values.get("TORCH_VERSION")
        torch_cuda_version = torch_values.get("CUDA_VERSION")

        runtime = WorkerRuntime(
            worker_id=worker.worker_id,
            runtime_os=RuntimeOS.WSL2_LINUX,
            runtime_version=selected_distro,
            conda_executable=conda_exe.value if conda_exe.ok else None,
            conda_environment=selected_env,
            conda_prefix=selected_prefix,
            conda_active=active_env is not None,
            python_executable=python_executable if python_version.ok else None,
            python_version=python_version.value if python_version.ok else None,
            torch_version=torch_version,
            torch_cuda_version=torch_cuda_version,
            cuda_available=cuda_available,
            health=FAILED,
        )
        resource = WorkerResource(
            worker_id=worker.worker_id,
            hostname=as_hostname(worker.host),
            physical_os=PhysicalOS.WINDOWS,
            runtime_os=RuntimeOS.WSL2_LINUX,
            conda_environment=selected_env,
            conda_prefix=selected_prefix,
            python_executable=python_executable if python_version.ok else None,
            torch_version=torch_version,
            torch_cuda_version=torch_cuda_version,
            cuda_version=torch_cuda_version,
            health=FAILED,
        )

        if not cuda_available:
            failures.append(
                ProbeFailure(
                    layer="wsl_runtime",
                    check="cuda",
                    message="CUDA is not available from the selected Conda environment "
                    "(torch.cuda.is_available() returned False)",
                )
            )
        else:
            gpu = checks.probe_wsl_gpu(wsl_runner, python_executable)
            nccl = checks.probe_wsl_nccl(wsl_runner, python_executable)
            _record_failure(
                failures,
                gpu,
                layer="wsl_runtime",
                check="gpu",
                message="GPU properties could not be read from PyTorch",
            )
            _record_failure(
                failures,
                nccl,
                layer="wsl_runtime",
                check="nccl",
                message="NCCL is not available in the selected Conda environment",
            )
            gpu_values = checks.parse_key_values(gpu.value) if gpu.ok else {}
            resource = WorkerResource(
                worker_id=worker.worker_id,
                hostname=as_hostname(worker.host),
                physical_os=PhysicalOS.WINDOWS,
                runtime_os=RuntimeOS.WSL2_LINUX,
                conda_environment=selected_env,
                conda_prefix=selected_prefix,
                python_executable=python_executable if python_version.ok else None,
                gpu_name=gpu_values.get("GPU_NAME"),
                gpu_total_memory=_parse_int(gpu_values.get("GPU_MEM_MB")),
                compute_capability=gpu_values.get("GPU_CC"),
                cuda_version=torch_cuda_version,
                torch_version=torch_version,
                torch_cuda_version=torch_cuda_version,
                nccl_available=nccl.ok,
                gloo_available=False,
                health=FAILED,
            )
            runtime = WorkerRuntime(
                worker_id=worker.worker_id,
                runtime_os=RuntimeOS.WSL2_LINUX,
                runtime_version=selected_distro,
                conda_executable=conda_exe.value if conda_exe.ok else None,
                conda_environment=selected_env,
                conda_prefix=selected_prefix,
                conda_active=active_env is not None,
                python_executable=python_executable if python_version.ok else None,
                python_version=python_version.value if python_version.ok else None,
                torch_version=torch_version,
                torch_cuda_version=torch_cuda_version,
                cuda_available=True,
                nccl_available=nccl.ok,
                gloo_available=False,
                health=FAILED,
            )

        gloo = checks.probe_wsl_gloo(wsl_runner, python_executable)
        _record_failure(
            failures,
            gloo,
            layer="wsl_runtime",
            check="gloo",
            message="Gloo availability could not be determined",
        )
        gloo_values = checks.parse_key_values(gloo.value) if gloo.ok else {}
        runtime = replace(runtime, gloo_available=gloo_values.get("GLOO") == "True")
        resource = replace(resource, gloo_available=gloo_values.get("GLOO") == "True")

    interface = checks.probe_wsl_interface(wsl_runner)
    _record_failure(
        failures,
        interface,
        layer="wsl_runtime",
        check="interface",
        message="Network interface could not be determined in the WSL2 runtime",
    )
    resource = replace(resource, network_interface=interface.value if interface.ok else None)

    health = FAILED if failures else Health.HEALTHY
    return WorkerProbeResult(
        worker_resource=replace(resource, health=health, last_probe_at=_now()),
        worker_runtime=replace(runtime, health=health),
        windows_host=host_info,
        failures=tuple(failures),
        health=health,
        probe_status=probe_status,
    )


def _parse_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None