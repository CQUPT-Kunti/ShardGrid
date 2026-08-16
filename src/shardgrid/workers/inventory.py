"""Configuration-driven worker inventory."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import yaml

from shardgrid.common.config import (
    ClusterConfig,
    ConfigValidationError,
    WorkerConfig,
    load_cluster_config,
)
from shardgrid.common.enums import Health
from shardgrid.common.models import Hostname, WorkerId
from shardgrid.workers.models import Worker


def _as_worker(config: WorkerConfig) -> Worker:
    return Worker(
        worker_id=config.worker_id,
        machine_id=config.machine_id,
        hostname=Hostname(str(config.host)),
        physical_os=config.physical_os,
        runtime_os=config.runtime_os,
        host=str(config.host),
        ssh_user_ref=config.ssh_user,
        runtime=config.runtime,
        runtime_distro=config.runtime_distro,
        local_world_size=config.local_world_size,
        enabled=config.enabled,
        health=Health.UNKNOWN,
    )


def _validate_workers(workers: list[Worker]) -> None:
    seen: set[WorkerId] = set()
    for worker in workers:
        if worker.worker_id in seen:
            raise ConfigValidationError(f"duplicate worker_id: {worker.worker_id}")
        seen.add(worker.worker_id)
        if worker.local_world_size != 1:
            raise ConfigValidationError("worker.local_world_size must be 1 in the current stage")


@dataclass(frozen=True)
class WorkerInventory:
    workers: tuple[Worker, ...]
    preferred_workers: tuple[WorkerId, ...] = ()

    def __post_init__(self) -> None:
        _validate_workers(list(self.workers))

    def enabled_workers(self) -> list[Worker]:
        return [worker for worker in self.workers if worker.enabled]

    def identify_worker(self, worker_id: WorkerId | str) -> Worker:
        target = WorkerId(str(worker_id))
        for worker in self.workers:
            if worker.worker_id == target:
                return worker
        raise KeyError(f"unknown worker: {target}")

    def enable_worker(self, worker_id: WorkerId | str) -> WorkerInventory:
        target = WorkerId(str(worker_id))
        return WorkerInventory(
            workers=tuple(
                replace(worker, enabled=True) if worker.worker_id == target else worker
                for worker in self.workers
            ),
            preferred_workers=self.preferred_workers,
        )

    def disable_worker(self, worker_id: WorkerId | str) -> WorkerInventory:
        target = WorkerId(str(worker_id))
        return WorkerInventory(
            workers=tuple(
                replace(worker, enabled=False) if worker.worker_id == target else worker
                for worker in self.workers
            ),
            preferred_workers=self.preferred_workers,
        )

    def merge(self, config: ClusterConfig) -> WorkerInventory:
        disabled = set(config.manual_override.disabled_workers)
        host_overrides = config.manual_override.worker_address_overrides
        workers = tuple(
            replace(
                worker,
                enabled=worker.enabled and worker.worker_id not in disabled,
                host=host_overrides.get(worker.worker_id, worker.host),
                hostname=Hostname(host_overrides.get(worker.worker_id, worker.host)),
            )
            for worker in self.workers
        )
        return WorkerInventory(
            workers=workers,
            preferred_workers=tuple(config.manual_override.preferred_workers),
        )


def load_inventory(source: Path | str | ClusterConfig) -> WorkerInventory:
    if isinstance(source, ClusterConfig):
        config = source
    elif isinstance(source, Path):
        config = load_cluster_config(source)
    else:
        candidate = Path(source)
        if candidate.exists():
            config = load_cluster_config(candidate)
        else:
            payload = yaml.safe_load(source)
            config = ClusterConfig.from_dict(payload)
    inventory = WorkerInventory(
        workers=tuple(_as_worker(worker) for worker in config.workers)
    )
    return inventory.merge(config)


def merge_inventory(inventory: WorkerInventory, config: ClusterConfig) -> WorkerInventory:
    return inventory.merge(config)
