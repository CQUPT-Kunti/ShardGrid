"""T061 — automatic fresh resource discovery regression tests.

These tests validate the discovery contract that every job launch refreshes
worker/GPU resource state from live probes (not address book totals, not
stale snapshots, not a fixed host/GPU count).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from shardgrid.common.config import ClusterConfig
from shardgrid.common.enums import Health, PhysicalOS, RuntimeOS
from shardgrid.common.models import as_backend_name, as_hostname, as_job_id, as_worker_id
from shardgrid.control.job_manager import JobManager, create_training_job
from shardgrid.control.resource_manager import ResourceManager
from shardgrid.resources.models import (
    NetworkLink,
    NetworkState,
    WorkerResource,
)
from shardgrid.workers.models import WorkerRuntime
from shardgrid.workers.probe import ProbeFailure, WorkerProbeResult, WindowsHostInfo


def _worker_resource(
    worker_id: str,
    *,
    health: Health = Health.HEALTHY,
    free_memory: int | None = 8192,
) -> WorkerResource:
    from datetime import UTC, datetime

    return WorkerResource(
        worker_id=as_worker_id(worker_id),
        hostname=as_hostname(f"{worker_id}.local"),
        physical_os=PhysicalOS.WINDOWS,
        runtime_os=RuntimeOS.WSL2_LINUX,
        conda_environment="shardgrid",
        conda_prefix="/opt/conda/envs/shardgrid",
        python_executable="/opt/conda/envs/shardgrid/bin/python",
        ip="127.0.0.1",
        gpu_name="RTX 4060",
        gpu_total_memory=8192,
        gpu_free_memory=free_memory,
        compute_capability="8.9",
        driver_version="566.07",
        cuda_version="11.8",
        torch_version="2.7.1+cu118",
        torch_cuda_version="11.8",
        nccl_available=True,
        gloo_available=True,
        health=health,
        last_probe_at=datetime.now(tz=UTC).isoformat(),
    )


def _probe(worker_id: str, *, health: Health = Health.HEALTHY, free_memory: int | None = 8192):
    from datetime import UTC, datetime

    resource = WorkerResource(
        worker_id=as_worker_id(worker_id),
        hostname=as_hostname(f"{worker_id}.local"),
        physical_os=PhysicalOS.WINDOWS,
        runtime_os=RuntimeOS.WSL2_LINUX,
        conda_environment="shardgrid",
        conda_prefix="/opt/conda/envs/shardgrid",
        python_executable="/opt/conda/envs/shardgrid/bin/python",
        ip="127.0.0.1",
        gpu_name="RTX 4060",
        gpu_total_memory=8192,
        gpu_free_memory=free_memory,
        compute_capability="8.9",
        driver_version="566.07",
        cuda_version="11.8",
        torch_version="2.7.1+cu118",
        torch_cuda_version="11.8",
        nccl_available=True,
        gloo_available=True,
        health=health,
        last_probe_at=datetime.now(tz=UTC).isoformat(),
    )
    return WorkerProbeResult(
        worker_resource=resource,
        worker_runtime=WorkerRuntime(
            worker_id=as_worker_id(worker_id),
            runtime_os=RuntimeOS.WSL2_LINUX,
            runtime_version="Ubuntu-22.04",
            conda_environment="shardgrid",
            conda_prefix="/opt/conda/envs/shardgrid",
            health=health,
        ),
        windows_host=WindowsHostInfo(
            os_version="Windows",
            openssh_available=True,
            wsl_available=True,
            nvidia_driver_visible=health is Health.HEALTHY,
            driver_name="566.07",
        ),
        failures=(
            ()
            if health is Health.HEALTHY
            else (
                ProbeFailure(
                    layer="remote_access",
                    check="unreachable" if health is Health.UNREACHABLE else "unhealthy",
                    message=f"{worker_id} is not healthy",
                ),
            )
        ),
        health=health,
        probe_status="live",
    )


def _link(source: str, target: str) -> NetworkLink:
    from datetime import UTC, datetime

    return NetworkLink(
        source_worker_id=source,
        target_worker_id=target,
        source_ip="127.0.0.1",
        target_ip="127.0.0.1",
        interface="eth0",
        tcp_reachable=True,
        bandwidth_mbps=900.0,
        latency_ms=1.5,
        measured_at=datetime.now(tz=UTC).isoformat(),
    )


def _network_state(worker_ids: tuple[str, ...]) -> NetworkState:
    from datetime import UTC, datetime

    now = datetime.now(tz=UTC).isoformat()
    links = [
        item
        for source in worker_ids
        for target in worker_ids
        if source != target
        for item in (_link(source, target), _link(target, source))
    ]
    return NetworkState(
        network_id="net-1",
        workers=list(worker_ids),
        links=links,
        created_at=now,
        selected_interfaces={worker_id: "eth0" for worker_id in worker_ids},
    )


def _cluster_config(worker_ids: tuple[str, ...]) -> ClusterConfig:
    return ClusterConfig.from_dict(
        {
            "control": {"machine_id": "machine-a", "hostname": "control-a.local"},
            "jobs_root": "/var/tmp/shardgrid/jobs",
            "ssh": {},
            "runtime": {
                "conda_environment": "shardgrid",
                "conda_prefix": "/opt/conda/envs/shardgrid",
            },
            "network": {"rendezvous_port": 29500},
            "backend_preference": {},
            "manual_override": {},
            "workers": [
                {
                    "id": worker_id,
                    "machine_id": f"machine-{worker_id}",
                    "physical_os": "windows",
                    "runtime_os": "wsl2_linux",
                    "runtime": "wsl2",
                    "host": f"10.0.0.{index + 1}",
                    "ssh_user": "shardgrid",
                }
                for index, worker_id in enumerate(worker_ids)
            ],
        }
    )


def _network_probe(workers):
    del workers
    return _network_state(("worker-a", "worker-b"))


def _manager(worker_ids: tuple[str, ...]) -> JobManager:
    manager = JobManager(_cluster_config(worker_ids))
    manager._probe_network = _network_probe
    return manager


def test_unreachable_worker_is_excluded_from_automatic_candidate_pool() -> None:
    manager = _manager(("worker-a", "worker-b"))

    def probe(worker):
        return (
            _probe("worker-a")
            if str(worker.worker_id) == "worker-a"
            else _probe("worker-b", health=Health.UNREACHABLE)
        )

    manager._probe_worker = probe

    refreshed = manager._refresh_planning_state(
        training_config=None,
        selected_workers=_cluster_config(("worker-a", "worker-b")).workers,
        current_job_id=as_job_id("job-discovery-unreachable"),
    )

    assert refreshed is not None
    fresh_workers, _, _, cluster_state, network_state = refreshed
    assert [str(worker.worker_id) for worker in fresh_workers] == ["worker-a"]
    assert cluster_state.summary["eligible_workers"] == 1
    assert network_state is not None


def test_unhealthy_gpu_worker_is_excluded_from_candidate_pool() -> None:
    manager = _manager(("worker-a", "worker-b"))

    def probe(worker):
        return (
            _probe("worker-a")
            if str(worker.worker_id) == "worker-a"
            else _probe("worker-b", health=Health.FAILED)
        )

    manager._probe_worker = probe

    refreshed = manager._refresh_planning_state(
        training_config=None,
        selected_workers=_cluster_config(("worker-a", "worker-b")).workers,
        current_job_id=as_job_id("job-discovery-unhealthy"),
    )

    assert refreshed is not None
    fresh_workers, _, _, cluster_state, _ = refreshed
    assert [str(worker.worker_id) for worker in fresh_workers] == ["worker-a"]
    assert cluster_state.summary["eligible_workers"] == 1


def test_no_healthy_workers_returns_none_signal() -> None:
    manager = _manager(("worker-a",))

    manager._probe_worker = lambda worker: _probe("worker-a", health=Health.UNREACHABLE)

    refreshed = manager._refresh_planning_state(
        training_config=None,
        selected_workers=_cluster_config(("worker-a",)).workers,
        current_job_id=as_job_id("job-discovery-none-healthy"),
    )

    assert refreshed is None


def test_free_memory_comes_from_current_snapshot_not_total_memory() -> None:
    resource_manager = ResourceManager()
    total_only = _worker_resource("worker-a", free_memory=None)
    current = _worker_resource("worker-b", free_memory=2048)

    cluster = resource_manager.build_cluster_state(
        [total_only, current], minimum_gpu_memory_mb=2048
    )

    assert cluster.workers[0].eligible is False
    assert cluster.workers[1].eligible is True
    assert cluster.workers[0].resource.gpu_free_memory is None
    assert cluster.workers[1].resource.gpu_free_memory == 2048


def test_resource_snapshot_is_refreshed_before_next_job() -> None:
    manager = _manager(("worker-a",))
    calls: list[int] = []

    def probe(worker):
        calls.append(len(calls))
        return _probe("worker-a", free_memory=8192 - len(calls) * 1024)

    manager._probe_worker = probe

    first = manager._refresh_planning_state(
        training_config=None,
        selected_workers=_cluster_config(("worker-a",)).workers,
        current_job_id=as_job_id("job-discovery-refresh-1"),
    )
    second = manager._refresh_planning_state(
        training_config=None,
        selected_workers=_cluster_config(("worker-a",)).workers,
        current_job_id=as_job_id("job-discovery-refresh-2"),
    )

    assert len(calls) == 2
    first_free = first[3].workers[0].resource.gpu_free_memory
    second_free = second[3].workers[0].resource.gpu_free_memory
    assert second_free == first_free - 1024


def test_automatic_mode_accepts_variable_worker_counts() -> None:
    for worker_ids in (("worker-a",), ("worker-a", "worker-b")):
        manager = _manager(worker_ids)

        def probe(worker, worker_ids=worker_ids):
            return _probe(str(worker.worker_id), free_memory=8192)

        manager._probe_worker = probe
        refreshed = manager._refresh_planning_state(
            training_config=None,
            selected_workers=_cluster_config(worker_ids).workers,
            current_job_id=as_job_id(f"job-discovery-{len(worker_ids)}"),
        )

        assert refreshed is not None
        fresh_workers, _, _, cluster_state, _ = refreshed
        assert len(fresh_workers) == len(worker_ids)
        assert cluster_state.summary["eligible_workers"] == len(worker_ids)


def test_refresh_drops_newly_unreachable_worker() -> None:
    manager = _manager(("worker-a", "worker-b"))
    state = {"healthy": {"worker-a", "worker-b"}}

    def probe(worker):
        if str(worker.worker_id) not in state["healthy"]:
            return _probe(str(worker.worker_id), health=Health.UNREACHABLE)
        return _probe(str(worker.worker_id))

    manager._probe_worker = probe
    first = manager._refresh_planning_state(
        training_config=None,
        selected_workers=_cluster_config(("worker-a", "worker-b")).workers,
        current_job_id=as_job_id("job-discovery-ok"),
    )
    assert first is not None
    assert len(first[0]) == 2

    state["healthy"] = {"worker-a"}
    second = manager._refresh_planning_state(
        training_config=None,
        selected_workers=_cluster_config(("worker-a", "worker-b")).workers,
        current_job_id=as_job_id("job-discovery-lost-b"),
    )
    assert second is not None
    assert [str(w.worker_id) for w in second[0]] == ["worker-a"]


def test_resource_manager_retains_unreachable_worker_without_eligibility() -> None:
    resource_manager = ResourceManager()
    workers = [
        _worker_resource("worker-a"),
        _worker_resource("worker-b", health=Health.UNREACHABLE),
    ]

    cluster = resource_manager.build_cluster_state(workers)

    assert [str(entry.worker_id) for entry in cluster.workers] == [
        "worker-a",
        "worker-b",
    ]
    assert cluster.workers[1].eligible is False
