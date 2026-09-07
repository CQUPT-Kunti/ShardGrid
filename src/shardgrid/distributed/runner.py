"""Multi-host distributed runner (T047).

Builds a per-Worker launch plan for the two-GPU model:
RTX 4060 Worker -> rank 0, GTX 1650 Worker -> rank 1, with one process per
physical Worker (``local_world_size = 1``, ``local_rank = 0``).  The runner
only performs dry-run command generation; it never starts a process group.
All paths and addresses come from configuration / runtime metadata, never from
hard-coded literals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from shardgrid.common.config import RuntimeConfig, WorkerConfig
from shardgrid.transport.runtime import WSLRuntimeWrapper

VALID_BACKENDS = ("nccl", "gloo")


class LaunchError(ValueError):
    """Raised when a multi-host launch plan is invalid."""


@dataclass(frozen=True)
class WorkerLaunch:
    worker_id: str
    rank: int
    world_size: int
    local_rank: int
    local_world_size: int
    python_executable: str
    conda_environment: str | None
    conda_prefix: str | None
    log_path: str
    argv: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "rank": self.rank,
            "world_size": self.world_size,
            "local_rank": self.local_rank,
            "local_world_size": self.local_world_size,
            "python_executable": self.python_executable,
            "conda_environment": self.conda_environment,
            "conda_prefix": self.conda_prefix,
            "log_path": self.log_path,
            "argv": list(self.argv),
        }


@dataclass(frozen=True)
class MultiHostPlan:
    world_size: int
    master_addr: str
    master_port: int
    backend: str
    launches: tuple[WorkerLaunch, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "world_size": self.world_size,
            "master_addr": self.master_addr,
            "master_port": self.master_port,
            "backend": self.backend,
            "launches": [launch.to_dict() for launch in self.launches],
        }


def validate_rank(rank: int, world_size: int) -> None:
    if world_size < 1:
        raise LaunchError(f"world_size must be >= 1, got {world_size}")
    if rank < 0 or rank >= world_size:
        raise LaunchError(
            f"rank must be in [0, world_size), got rank={rank} world_size={world_size}"
        )


def _resolve_conda_prefix(worker: WorkerConfig, runtime: RuntimeConfig) -> str | None:
    return worker.conda_prefix or runtime.conda_prefix


def build_launch_plan(
    workers: Sequence[WorkerConfig],
    *,
    runtime: RuntimeConfig,
    smoke_program: str,
    master_addr: str | None = None,
    master_port: int = 29500,
    backend: str = "gloo",
    log_dir: str = "/var/tmp/shardgrid/logs",
) -> MultiHostPlan:
    if len(workers) < 1:
        raise LaunchError("at least one worker is required")
    world_size = len(workers)
    resolved_master = master_addr or str(workers[0].host)
    if not resolved_master:
        raise LaunchError("master_addr must not be empty")
    if not (0 < master_port < 65536):
        raise LaunchError(f"master_port must be in (0, 65536), got {master_port}")
    if backend not in VALID_BACKENDS:
        raise LaunchError(f"backend must be one of {VALID_BACKENDS}, got {backend}")

    launches: list[WorkerLaunch] = []
    for rank, worker in enumerate(workers):
        prefix = _resolve_conda_prefix(worker, runtime)
        if not prefix:
            raise LaunchError(
                f"worker {worker.worker_id} has no Conda prefix; "
                "cannot select a training Python"
            )
        conda_environment = worker.conda_environment or runtime.conda_environment
        python_executable = f"{prefix}/bin/python"
        log_path = f"{log_dir}/rank-{rank}.log"
        argv = (
            python_executable,
            smoke_program,
            "--rank",
            str(rank),
            "--world-size",
            str(world_size),
            "--master-addr",
            resolved_master,
            "--master-port",
            str(master_port),
            "--backend",
            backend,
            "--local-rank",
            "0",
        )
        launches.append(
            WorkerLaunch(
                worker_id=str(worker.worker_id),
                rank=rank,
                world_size=world_size,
                local_rank=0,
                local_world_size=1,
                python_executable=python_executable,
                conda_environment=conda_environment,
                conda_prefix=prefix,
                log_path=log_path,
                argv=argv,
            )
        )
    return MultiHostPlan(
        world_size=world_size,
        master_addr=resolved_master,
        master_port=master_port,
        backend=backend,
        launches=tuple(launches),
    )


def remote_launch_command(wrapper: WSLRuntimeWrapper, launch: WorkerLaunch) -> str:
    """Return the T040 wrapper remote command string for this worker launch."""
    payload = wrapper.build_payload(launch.argv)
    return wrapper.build_remote_command(payload)