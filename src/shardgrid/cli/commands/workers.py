"""`shardgrid workers` inventory CLI (T077).

This command is an inventory view over the existing Worker/WorkerRuntime/
WorkerResource probe contracts.  It does not reimplement doctor or GPU/network
probing: it loads configured Workers, optionally refreshes them through the
existing SSH -> WSL -> Conda runtime probes, and renders the latest known
resource state with explicit health/staleness semantics.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from shardgrid.cli.context import EXIT_CONFIG_ERROR, EXIT_OK, EXIT_RUNTIME_ERROR
from shardgrid.common.config import ClusterConfig, WorkerConfig
from shardgrid.common.enums import Health
from shardgrid.resources.models import WorkerResource
from shardgrid.transport.remote_access import RemoteAccessResult, run_remote_access_check
from shardgrid.transport.runtime import WSLRuntimeConfig, WSLRuntimeWrapper
from shardgrid.transport.ssh import SSHOptions, SSHTransport
from shardgrid.workers.gpu_probe import GPUProbeResult, probe_gpu
from shardgrid.workers.inventory import load_inventory
from shardgrid.workers.models import Worker, WorkerRuntime

DEFAULT_CACHE_DIR = Path.home() / ".shardgrid" / "workers"
DEFAULT_CACHE_PATH = DEFAULT_CACHE_DIR / "inventory-latest.json"
STALE_AFTER = timedelta(hours=24)


@dataclass(frozen=True)
class WorkerInventoryEntry:
    worker: Worker
    resource: WorkerResource
    runtime: WorkerRuntime | None
    reachability: str
    inventory_health: str
    eligible: bool
    stale: bool
    reason: str | None
    failure: dict[str, Any] | None
    source: str
    host_identity: str | None = None
    runtime_identity: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "worker": self.worker.to_dict(),
            "resource": self.resource.to_dict(),
            "runtime": None if self.runtime is None else self.runtime.to_dict(),
            "reachability": self.reachability,
            "health": self.inventory_health,
            "eligible": self.eligible,
            "stale": self.stale,
            "reason": self.reason,
            "failure": self.failure,
            "source": self.source,
            "host_identity": self.host_identity,
            "runtime_identity": self.runtime_identity,
        }


def register_workers_command(
    subparsers: argparse._SubParsersAction[Any],
) -> None:
    parser = subparsers.add_parser("workers", help="Show current Worker inventory")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Run live Worker probes and update the local inventory cache",
    )
    parser.add_argument(
        "--require-healthy",
        action="store_true",
        help="Exit non-zero when any displayed Worker is not healthy and eligible",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Emit structured JSON output",
    )
    parser.set_defaults(handler=run_workers_command, command_name="workers")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _is_stale(resource: WorkerResource) -> bool:
    stamp = _parse_timestamp(resource.last_probe_at)
    if stamp is None:
        return True
    return _now() - stamp > STALE_AFTER


def _health_label(
    resource: WorkerResource,
    *,
    stale: bool,
    reason: str | None,
    enabled: bool,
) -> tuple[str, bool]:
    if not enabled:
        return "UNAVAILABLE", False
    if resource.health == Health.UNREACHABLE:
        return "UNREACHABLE", False
    if stale:
        return "STALE", False
    if resource.health == Health.HEALTHY:
        return "HEALTHY", True
    if resource.health == Health.DEGRADED:
        return "DEGRADED", False
    if resource.health == Health.BLOCKED_MANUAL_ACTION:
        return "BLOCKED", False
    if resource.health == Health.FAILED:
        return "UNHEALTHY", False
    if reason:
        return "UNAVAILABLE", False
    return "STALE", False


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
    config: ClusterConfig,
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
        _build_transport(config, worker),
    )


def _refresh_worker(
    config: ClusterConfig,
    worker: WorkerConfig,
) -> tuple[
    WorkerResource,
    WorkerRuntime | None,
    str,
    str | None,
    dict[str, Any] | None,
    str | None,
    dict[str, Any] | None,
]:
    transport = _build_transport(config, worker)
    access = run_remote_access_check(
        transport,
        worker,
        worker_label=str(worker.labels.get("gpu") or worker.worker_id),
        preferred_environment=worker.conda_environment or config.runtime.conda_environment,
    )
    if access.status == "BLOCKED":
        resource = WorkerResource(
            worker_id=worker.worker_id,
            hostname=worker.host,
            physical_os=worker.physical_os,
            runtime_os=worker.runtime_os,
            conda_environment=worker.conda_environment or config.runtime.conda_environment,
            conda_prefix=worker.conda_prefix or config.runtime.conda_prefix,
            health=Health.UNREACHABLE,
            last_probe_at=_now().isoformat(),
        )
        return (
            resource,
            None,
            "UNREACHABLE",
            access.failure_reason,
            access.failure_record,
            access.windows_identity,
            None,
        )
    if access.runtime_identity is None:
        resource = WorkerResource(
            worker_id=worker.worker_id,
            hostname=worker.host,
            physical_os=worker.physical_os,
            runtime_os=worker.runtime_os,
            conda_environment=worker.conda_environment or config.runtime.conda_environment,
            conda_prefix=worker.conda_prefix or config.runtime.conda_prefix,
            health=Health.FAILED,
            last_probe_at=_now().isoformat(),
        )
        return (
            resource,
            None,
            "REACHABLE",
            access.failure_reason,
            access.failure_record,
            access.windows_identity,
            None,
        )

    wrapper = _runtime_wrapper(config, worker, access)
    gpu_result = probe_gpu(wrapper, worker, probe_status="live")
    resource = replace(
        gpu_result.worker_resource,
        ip=str(worker.host),
        last_probe_at=_now().isoformat(),
    )
    runtime = replace(
        gpu_result.worker_runtime,
        python_version=access.runtime_identity.python_version,
        python_executable=access.runtime_identity.python_executable,
        conda_executable=access.runtime_identity.conda_executable,
    )
    return (
        resource,
        runtime,
        "REACHABLE",
        _probe_reason(gpu_result),
        _probe_failure(gpu_result),
        access.windows_identity,
        {
            "wsl_distro": access.runtime_identity.wsl_distro,
            "conda_executable": access.runtime_identity.conda_executable,
            "conda_environment": access.runtime_identity.conda_environment,
            "conda_prefix": access.runtime_identity.conda_prefix,
            "python_executable": access.runtime_identity.python_executable,
            "python_version": access.runtime_identity.python_version,
        },
    )


def _probe_reason(result: GPUProbeResult) -> str | None:
    if not result.failures:
        return None
    return "; ".join(failure.message for failure in result.failures)


def _probe_failure(result: GPUProbeResult) -> dict[str, Any] | None:
    if not result.failures:
        return None
    failure = result.failures[0]
    return failure.to_dict()


def _configured_resource(worker: Worker) -> WorkerResource:
    return WorkerResource(
        worker_id=worker.worker_id,
        hostname=worker.hostname,
        physical_os=worker.physical_os,
        runtime_os=worker.runtime_os,
        conda_environment=worker.conda_environment,
        conda_prefix=worker.conda_prefix,
        ip=worker.host,
        health=Health.UNKNOWN if worker.enabled else Health.BLOCKED_MANUAL_ACTION,
    )


def _load_cache(path: Path) -> tuple[dict[str, WorkerResource], dict[str, WorkerRuntime]]:
    if not path.exists():
        return {}, {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    resources = {
        item["worker_id"]: WorkerResource.from_dict(item)
        for item in payload.get("resources", [])
    }
    runtimes = {
        item["worker_id"]: WorkerRuntime.from_dict(item)
        for item in payload.get("runtimes", [])
    }
    return resources, runtimes


def _save_cache(path: Path, entries: list[WorkerInventoryEntry]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": _now().isoformat(),
        "resources": [entry.resource.to_dict() for entry in entries],
        "runtimes": [
            entry.runtime.to_dict()
            for entry in entries
            if entry.runtime is not None
        ],
        "meta": [
            {
                "worker_id": str(entry.worker.worker_id),
                "host_identity": entry.host_identity,
                "runtime_identity": entry.runtime_identity,
            }
            for entry in entries
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _entry_from_sources(
    worker: Worker,
    resource: WorkerResource,
    runtime: WorkerRuntime | None,
    *,
    source: str,
    reason: str | None,
    failure: dict[str, Any] | None,
    host_identity: str | None = None,
    runtime_identity: dict[str, Any] | None = None,
) -> WorkerInventoryEntry:
    stale = _is_stale(resource)
    reachability = "reachable"
    if resource.health == Health.UNREACHABLE:
        reachability = "unreachable"
    elif source == "config-only":
        reachability = "unknown"
    label, eligible = _health_label(resource, stale=stale, reason=reason, enabled=worker.enabled)
    return WorkerInventoryEntry(
        worker=worker,
        resource=resource,
        runtime=runtime,
        reachability=reachability,
        inventory_health=label,
        eligible=eligible,
        stale=stale,
        reason=reason,
        failure=failure,
        source=source,
        host_identity=host_identity,
        runtime_identity=runtime_identity,
    )


def _collect_entries(
    config: ClusterConfig,
    *,
    refresh: bool,
    cache_path: Path,
) -> list[WorkerInventoryEntry]:
    inventory = load_inventory(config)
    cached_resources, cached_runtimes = _load_cache(cache_path)
    cached_payload = (
        json.loads(cache_path.read_text(encoding="utf-8"))
        if cache_path.exists()
        else {}
    )
    cached_meta = {
        item["worker_id"]: item
        for item in cached_payload.get("meta", [])
        if isinstance(item, dict) and item.get("worker_id")
    }
    entries: list[WorkerInventoryEntry] = []
    for worker in inventory.workers:
        if refresh:
            refreshed = _refresh_worker(
                config,
                next(item for item in config.workers if item.worker_id == worker.worker_id),
            )
            if len(refreshed) == 5:
                resource, runtime, _, reason, failure = refreshed
                host_identity = None
                runtime_identity = None
            else:
                (
                    resource,
                    runtime,
                    _,
                    reason,
                    failure,
                    host_identity,
                    runtime_identity,
                ) = refreshed
            entry = _entry_from_sources(
                worker,
                resource,
                runtime,
                source="live",
                reason=reason,
                failure=failure,
                host_identity=host_identity,
                runtime_identity=runtime_identity,
            )
        else:
            resource = cached_resources.get(str(worker.worker_id), _configured_resource(worker))
            runtime = cached_runtimes.get(str(worker.worker_id))
            source = "cache" if str(worker.worker_id) in cached_resources else "config-only"
            reason = None
            failure = None
            if source == "config-only":
                reason = "no live probe has been recorded for this worker yet"
            elif _is_stale(resource):
                reason = "last live probe is stale; rerun with --refresh"
            entry = _entry_from_sources(
                worker,
                resource,
                runtime,
                source=source,
                reason=reason,
                failure=failure,
                host_identity=cached_meta.get(str(worker.worker_id), {}).get("host_identity"),
                runtime_identity=cached_meta.get(str(worker.worker_id), {}).get(
                    "runtime_identity"
                ),
            )
        entries.append(entry)
    if refresh:
        _save_cache(cache_path, entries)
    return entries


def _human_output(entries: list[WorkerInventoryEntry]) -> str:
    lines = ["ShardGrid workers", ""]
    for entry in entries:
        runtime = entry.runtime
        backend = []
        if entry.resource.nccl_available:
            backend.append("nccl")
        if entry.resource.gloo_available:
            backend.append("gloo")
        backend_text = ",".join(backend) if backend else "none"
        gpu = entry.resource.gpu_name or "n/a"
        if entry.resource.gpu_total_memory is not None:
            gpu = f"{gpu} ({entry.resource.gpu_total_memory} MiB)"
        if entry.resource.compute_capability:
            gpu = f"{gpu} cc={entry.resource.compute_capability}"
        conda = entry.resource.conda_environment or "n/a"
        if entry.resource.conda_prefix:
            conda = f"{conda} [{entry.resource.conda_prefix}]"
        runtime_identity = (
            runtime.python_version
            if runtime and runtime.python_version
            else entry.resource.python_executable or "n/a"
        )
        runtime_text = f"{entry.worker.runtime_os.value} | {runtime_identity}"
        host_text = entry.host_identity or str(entry.worker.hostname)
        health_line = (
            f"Health: {entry.inventory_health} | Reachability: {entry.reachability} | "
            f"Eligible: {'yes' if entry.eligible else 'no'}"
        )
        runtime_line = (
            f"Runtime: host={host_text} ({entry.worker.host}) | "
            f"physical={entry.worker.physical_os.value} | runtime={runtime_text}"
        )
        network_line = (
            f"Network/backend: iface={entry.resource.network_interface or 'n/a'} | "
            f"ip={entry.resource.ip or 'n/a'} | backends={backend_text}"
        )
        lines.extend(
            [
                f"Worker: {entry.worker.worker_id}",
                health_line,
                runtime_line,
                f"GPU: {gpu}",
                f"Conda: {conda}",
                network_line,
                f"Reason: {entry.reason or 'none'}",
                "",
            ]
        )
    return "\n".join(lines).rstrip()


def _json_output(entries: list[WorkerInventoryEntry], *, refreshed: bool) -> dict[str, Any]:
    return {
        "target": "workers",
        "refreshed": refreshed,
        "worker_count": len(entries),
        "healthy_worker_count": sum(1 for entry in entries if entry.eligible),
        "workers": [entry.to_dict() for entry in entries],
    }


def run_workers_command(args: argparse.Namespace) -> int:
    context = getattr(args, "context", None)
    config = getattr(context, "config", None)
    json_output = bool(getattr(args, "json", False)) or bool(
        getattr(context, "json_output", False)
    )
    if config is None:
        print(
            "workers requires a cluster config: "
            "shardgrid --config examples/workers.yaml workers --refresh"
        )
        return EXIT_CONFIG_ERROR

    cache_path = DEFAULT_CACHE_PATH
    entries = _collect_entries(config, refresh=bool(args.refresh), cache_path=cache_path)
    if json_output:
        print(
            json.dumps(
                _json_output(entries, refreshed=bool(args.refresh)),
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(_human_output(entries))

    if bool(args.require_healthy) and any(
        not entry.eligible for entry in entries if entry.worker.enabled
    ):
        return EXIT_RUNTIME_ERROR
    return EXIT_OK
