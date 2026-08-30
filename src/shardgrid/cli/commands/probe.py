"""`shardgrid probe` Worker/GPU/runtime probe CLI (T078)."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from typing import Any

from shardgrid.cli.context import (
    EXIT_CONFIG_ERROR,
    EXIT_OK,
    EXIT_RUNTIME_ERROR,
    EXIT_USAGE,
)
from shardgrid.common.config import ClusterConfig, WorkerConfig
from shardgrid.common.enums import FailureStage, Health
from shardgrid.common.errors import make_failure_record
from shardgrid.jobs.models import FailureRecord
from shardgrid.resources.models import WorkerResource
from shardgrid.transport.remote_access import RemoteAccessResult, run_remote_access_check
from shardgrid.transport.runtime import WSLRuntimeConfig, WSLRuntimeWrapper
from shardgrid.transport.ssh import SSHOptions, SSHTransport
from shardgrid.workers.gpu_probe import GPUProbeResult, probe_gpu
from shardgrid.workers.models import WorkerRuntime


@dataclass(frozen=True)
class WorkerProbeReport:
    worker: dict[str, Any]
    runtime: WorkerRuntime | None
    resource: WorkerResource
    reachability: str
    status: str
    failure: FailureRecord | None
    reason: str | None
    windows_identity: str | None
    runtime_identity: dict[str, Any] | None
    gpu: dict[str, Any]
    cuda: dict[str, Any]
    pytorch: dict[str, Any]
    backends: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "worker": dict(self.worker),
            "runtime": None if self.runtime is None else self.runtime.to_dict(),
            "resource": self.resource.to_dict(),
            "reachability": self.reachability,
            "status": self.status,
            "failure": None if self.failure is None else self.failure.to_dict(),
            "reason": self.reason,
            "windows_identity": self.windows_identity,
            "runtime_identity": self.runtime_identity,
            "gpu": dict(self.gpu),
            "cuda": dict(self.cuda),
            "pytorch": dict(self.pytorch),
            "backends": dict(self.backends),
        }


def register_probe_command(
    subparsers: argparse._SubParsersAction[Any],
) -> None:
    parser = subparsers.add_parser("probe", help="Probe Worker runtime and GPU state")
    parser.add_argument("--worker", help="Only probe one worker by worker_id")
    parser.add_argument(
        "--json",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Emit structured JSON output",
    )
    parser.set_defaults(handler=run_probe_command, command_name="probe")


def _build_transport(config: ClusterConfig, worker: WorkerConfig) -> SSHTransport:
    return SSHTransport(
        SSHOptions.from_ssh_config(
            config.ssh,
            host=str(worker.host),
            user=worker.ssh_user,
            port=worker.ssh_port,
        )
    )


def _runtime_wrapper(
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


def _parse_gpu_payload(gpu_result: GPUProbeResult) -> dict[str, Any]:
    if not gpu_result.raw_output:
        return {}
    for line in gpu_result.raw_output.splitlines():
        stripped = line.strip()
        if not stripped.startswith("{"):
            continue
        try:
            payload = json.loads(stripped)
        except ValueError:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _resolve_targets(config: ClusterConfig, worker_id: str | None) -> list[WorkerConfig]:
    if not worker_id:
        return list(config.workers)
    for worker in config.workers:
        if str(worker.worker_id) == worker_id:
            return [worker]
    raise ValueError(f"unknown worker id: {worker_id}")


def _probe_failure_record(
    worker: WorkerConfig,
    access: RemoteAccessResult,
    message: str,
) -> FailureRecord:
    return make_failure_record(
        stage=FailureStage.PROBE,
        host=str(worker.host),
        worker_id=str(worker.worker_id),
        command=access.commands[-1] if access.commands else None,
        exit_code=access.exit_code,
        conda_environment=worker.conda_environment,
        conda_prefix=worker.conda_prefix,
        message=message,
        recommended_action="inspect the worker probe diagnostics and rerun the probe",
    )


def _probe_report(worker: WorkerConfig, config: ClusterConfig) -> WorkerProbeReport:
    transport = _build_transport(config, worker)
    access = run_remote_access_check(
        transport,
        worker,
        worker_label=str(worker.labels.get("gpu") or worker.worker_id),
        preferred_environment=worker.conda_environment or config.runtime.conda_environment,
    )
    worker_info = {
        "worker_id": str(worker.worker_id),
        "hostname": access.windows_identity or str(worker.host),
        "configured_host": str(worker.host),
        "physical_os": worker.physical_os.value,
        "runtime_os": worker.runtime_os.value,
        "reachability": "reachable" if access.status != "BLOCKED" else "unreachable",
    }
    if access.status != "PASS" or access.runtime_identity is None:
        failure = (
            FailureRecord.from_dict(access.failure_record)
            if access.failure_record
            else _probe_failure_record(
                worker,
                access,
                access.failure_reason or "worker probe failed before runtime validation",
            )
        )
        resource = WorkerResource(
            worker_id=worker.worker_id,
            hostname=worker.host,
            physical_os=worker.physical_os,
            runtime_os=worker.runtime_os,
            conda_environment=worker.conda_environment or config.runtime.conda_environment,
            conda_prefix=worker.conda_prefix or config.runtime.conda_prefix,
            ip=str(worker.host),
            health=Health.UNREACHABLE if access.status == "BLOCKED" else Health.FAILED,
        )
        return WorkerProbeReport(
            worker=worker_info,
            runtime=None,
            resource=resource,
            reachability=worker_info["reachability"],
            status="FAILED",
            failure=failure,
            reason=failure.message,
            windows_identity=access.windows_identity,
            runtime_identity=None,
            gpu={"selected_gpu": None, "gpu_count": None},
            cuda={"available": False, "runtime_version": None},
            pytorch={"version": None},
            backends={
                "nccl_available": False,
                "nccl_version": None,
                "gloo_available": False,
                "backend_capability": [],
            },
        )

    gpu_result = getattr(access, "gpu_probe_result", None)
    if gpu_result is None:
        wrapper = _runtime_wrapper(transport, worker, access)
        gpu_result = probe_gpu(
            wrapper,
            worker,
            probe_status="live",
            timeout=float(config.ssh.probe_timeout_seconds),
        )
    payload = _parse_gpu_payload(gpu_result)
    torch_data = payload.get("torch") if isinstance(payload.get("torch"), dict) else {}
    resource = replace(
        gpu_result.worker_resource,
        ip=str(worker.host),
    )
    runtime = replace(
        gpu_result.worker_runtime,
        conda_executable=access.runtime_identity.conda_executable,
        python_executable=access.runtime_identity.python_executable,
        python_version=access.runtime_identity.python_version,
    )
    nccl_version = torch_data.get("nccl")
    backend_capability = []
    if resource.nccl_available:
        backend_capability.append("nccl")
    if resource.gloo_available:
        backend_capability.append("gloo")
    failure = None
    reason = None
    status = "PASS"
    if gpu_result.failures:
        first = gpu_result.failures[0]
        failure = make_failure_record(
            stage=FailureStage.PROBE,
            host=str(worker.host),
            worker_id=str(worker.worker_id),
            python_executable=runtime.python_executable,
            conda_environment=runtime.conda_environment,
            conda_prefix=runtime.conda_prefix,
            message=first.message,
            recommended_action="inspect the selected WSL Conda runtime and rerun the probe",
        )
        reason = "; ".join(item.message for item in gpu_result.failures)
        status = "FAILED"
    return WorkerProbeReport(
        worker=worker_info,
        runtime=runtime,
        resource=resource,
        reachability=worker_info["reachability"],
        status=status,
        failure=failure,
        reason=reason,
        windows_identity=access.windows_identity,
        runtime_identity={
            "wsl_distro": access.runtime_identity.wsl_distro,
            "conda_executable": access.runtime_identity.conda_executable,
            "conda_environment": access.runtime_identity.conda_environment,
            "conda_prefix": access.runtime_identity.conda_prefix,
            "python_executable": access.runtime_identity.python_executable,
            "python_version": access.runtime_identity.python_version,
        },
        gpu={
            "name": resource.gpu_name,
            "gpu_count": torch_data.get("device_count"),
            "selected_gpu": 0 if torch_data.get("device_count") else None,
            "total_memory_mb": resource.gpu_total_memory,
            "free_memory_mb": resource.gpu_free_memory,
            "compute_capability": resource.compute_capability,
            "driver_version": resource.driver_version,
        },
        cuda={
            "available": runtime.cuda_available,
            "runtime_version": resource.cuda_version,
            "torch_cuda_is_available": runtime.cuda_available,
        },
        pytorch={
            "version": resource.torch_version,
            "python_version": runtime.python_version,
        },
        backends={
            "nccl_available": resource.nccl_available,
            "nccl_version": nccl_version,
            "gloo_available": resource.gloo_available,
            "backend_capability": backend_capability,
        },
    )


def _human_output(reports: list[WorkerProbeReport]) -> str:
    lines = ["ShardGrid probe", ""]
    for report in reports:
        runtime_line = (
            "Runtime: "
            f"host={report.worker['hostname']} ({report.worker['configured_host']}) | "
            f"physical={report.worker['physical_os']} | "
            f"runtime={report.worker['runtime_os']}"
        )
        gpu_line = (
            "GPU: "
            f"{report.gpu.get('name') or 'n/a'} | "
            f"count={report.gpu.get('gpu_count') or 'n/a'} | "
            "selected="
            f"{_selected_gpu_text(report.gpu)}"
        )
        nccl_text = report.backends.get("nccl_version") or (
            "available" if report.backends.get("nccl_available") else "unavailable"
        )
        pytorch_line = (
            "PyTorch: "
            f"{report.pytorch.get('version') or 'n/a'} | "
            f"NCCL={nccl_text} | "
            f"Gloo={'available' if report.backends.get('gloo_available') else 'unavailable'}"
        )
        lines.extend(
            [
                f"Worker: {report.worker['worker_id']}",
                f"Probe status: {report.status} | Reachability: {report.reachability}",
                runtime_line,
                (
                    "Conda: "
                    f"{(report.runtime_identity or {}).get('conda_environment') or 'n/a'} "
                    f"[{(report.runtime_identity or {}).get('conda_prefix') or 'n/a'}]"
                ),
                (
                    "Python: "
                    f"{(report.runtime_identity or {}).get('python_executable') or 'n/a'} | "
                    f"{(report.runtime_identity or {}).get('python_version') or 'n/a'}"
                ),
                gpu_line,
                (
                    "CUDA: "
                    f"available={'yes' if report.cuda.get('available') else 'no'} | "
                    f"runtime={report.cuda.get('runtime_version') or 'n/a'}"
                ),
                pytorch_line,
                f"Reason: {report.reason or 'none'}",
                "",
            ]
        )
    return "\n".join(lines).rstrip()


def _selected_gpu_text(gpu: dict[str, Any]) -> str:
    selected = gpu.get("selected_gpu")
    return "n/a" if selected is None else str(selected)


def run_probe_command(args: argparse.Namespace) -> int:
    context = getattr(args, "context", None)
    config = getattr(context, "config", None)
    json_output = bool(getattr(args, "json", False)) or bool(
        getattr(context, "json_output", False)
    )
    if config is None:
        print(
            "probe requires a cluster config: "
            "shardgrid --config examples/workers.yaml probe"
        )
        return EXIT_CONFIG_ERROR

    try:
        targets = _resolve_targets(config, getattr(args, "worker", None))
    except ValueError as error:
        print(f"probe: {error}")
        return EXIT_USAGE

    reports = [_probe_report(worker, config) for worker in targets]
    payload = {
        "target": "probe",
        "worker_count": len(reports),
        "workers": [report.to_dict() for report in reports],
    }

    if json_output:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(_human_output(reports))

    if any(report.failure is not None for report in reports):
        return EXIT_RUNTIME_ERROR
    return EXIT_OK
