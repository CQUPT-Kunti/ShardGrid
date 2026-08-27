from __future__ import annotations

from datetime import UTC, datetime, timedelta

from shardgrid.common.enums import Health, PhysicalOS, RuntimeOS
from shardgrid.common.models import as_hostname, as_worker_id
from shardgrid.control.resource_manager import ResourceManager
from shardgrid.resources.models import NetworkLink, NetworkState, WorkerResource


def _timestamp(hours_ago: int = 0) -> str:
    return (datetime.now(tz=UTC) - timedelta(hours=hours_ago)).isoformat()


def _fixed_now() -> datetime:
    return datetime(2026, 8, 27, 0, 0, tzinfo=UTC)


def _worker(
    worker_id: str,
    *,
    health: Health = Health.HEALTHY,
    free_memory: int | None = 4096,
    last_probe_at: str | None = None,
    gpu_name: str | None = "GPU",
    python_executable: str | None = "/opt/conda/bin/python",
    torch_version: str | None = "2.7.1+cu118",
    cuda_version: str | None = "11.8",
) -> WorkerResource:
    return WorkerResource(
        worker_id=as_worker_id(worker_id),
        hostname=as_hostname(f"{worker_id}.local"),
        physical_os=PhysicalOS.WINDOWS,
        runtime_os=RuntimeOS.WSL2_LINUX,
        conda_environment="shardgrid",
        conda_prefix="/opt/conda/envs/shardgrid",
        python_executable=python_executable,
        ip=f"10.0.0.{1 if worker_id.endswith('a') else 2 if worker_id.endswith('b') else 3}",
        gpu_name=gpu_name,
        gpu_total_memory=8192,
        gpu_free_memory=free_memory,
        compute_capability="8.9",
        driver_version="566.07",
        cuda_version=cuda_version,
        torch_version=torch_version,
        torch_cuda_version="11.8",
        nccl_available=True,
        gloo_available=True,
        network_interface="eth0",
        health=health,
        last_probe_at=last_probe_at or _timestamp(),
    )


def _link(
    source: str,
    target: str,
    *,
    reachable: bool = True,
    measured_at: str | None = None,
    failure_reason: str | None = None,
) -> NetworkLink:
    return NetworkLink(
        source_worker_id=as_worker_id(source),
        target_worker_id=as_worker_id(target),
        source_ip=f"10.0.0.{1 if source.endswith('a') else 2 if source.endswith('b') else 3}",
        target_ip=f"10.0.0.{1 if target.endswith('a') else 2 if target.endswith('b') else 3}",
        interface="eth0",
        tcp_reachable=reachable,
        bandwidth_mbps=900.0 if reachable else None,
        latency_ms=1.5 if reachable else None,
        measured_at=measured_at or _timestamp(),
        failure_reason=failure_reason,
    )


def _network_state(*links: NetworkLink, created_at: str | None = None) -> NetworkState:
    return NetworkState(
        network_id="net-1",
        workers=sorted(
            {
                link.source_worker_id for link in links
            }
            | {link.target_worker_id for link in links}
        ),
        links=list(links),
        created_at=created_at or _timestamp(),
        selected_interfaces={str(link.source_worker_id): link.interface for link in links},
    )


def test_healthy_workers_and_links_are_eligible() -> None:
    manager = ResourceManager()
    workers = [_worker("worker-a"), _worker("worker-b")]
    network = _network_state(_link("worker-a", "worker-b"), _link("worker-b", "worker-a"))

    cluster = manager.build_cluster_state(workers, network_state=network)

    assert cluster.summary["eligible_workers"] == 2
    assert [entry.worker_id for entry in cluster.eligible_workers] == ["worker-a", "worker-b"]


def test_degraded_worker_is_kept_but_marked_ineligible() -> None:
    manager = ResourceManager()
    cluster = manager.build_cluster_state(
        [_worker("worker-a", health=Health.DEGRADED), _worker("worker-b")],
        network_state=_network_state(_link("worker-a", "worker-b"), _link("worker-b", "worker-a")),
    )

    degraded = cluster.workers[0]
    assert degraded.worker_id == "worker-a"
    assert degraded.eligible is False
    assert "degraded" in " ".join(degraded.exclusion_reasons).lower()


def test_unhealthy_and_unreachable_workers_are_not_selectable() -> None:
    manager = ResourceManager()
    cluster = manager.build_cluster_state(
        [_worker("worker-a", health=Health.FAILED), _worker("worker-b", health=Health.UNREACHABLE)],
        network_state=None,
    )

    assert all(not entry.eligible for entry in cluster.workers)
    assert any(
        "unhealthy" in " ".join(entry.exclusion_reasons).lower()
        for entry in cluster.workers
    )
    assert any(
        "unreachable" in " ".join(entry.exclusion_reasons).lower()
        for entry in cluster.workers
    )


def test_stale_worker_resource_is_ineligible() -> None:
    manager = ResourceManager()
    cluster = manager.build_cluster_state(
        [_worker("worker-a", last_probe_at=( _fixed_now() - timedelta(hours=30)).isoformat())],
        now=_fixed_now(),
    )

    assert cluster.workers[0].stale is True
    assert cluster.workers[0].eligible is False


def test_stale_network_state_blocks_network_eligibility() -> None:
    manager = ResourceManager()
    workers = [_worker("worker-a"), _worker("worker-b")]
    network = _network_state(
        _link("worker-a", "worker-b", measured_at=_timestamp(hours_ago=30)),
        _link("worker-b", "worker-a", measured_at=_timestamp(hours_ago=30)),
        created_at=_timestamp(hours_ago=30),
    )

    cluster = manager.build_cluster_state(workers, network_state=network, require_network=True)

    assert cluster.network_stale is True
    assert all(not entry.eligible for entry in cluster.workers)
    assert any(
        "network state is stale" in " ".join(entry.exclusion_reasons).lower()
        for entry in cluster.workers
    )


def test_insufficient_gpu_memory_blocks_eligibility() -> None:
    manager = ResourceManager()
    cluster = manager.build_cluster_state(
        [_worker("worker-a", free_memory=1024), _worker("worker-b", free_memory=4096)],
        minimum_gpu_memory_mb=2048,
    )

    assert cluster.workers[0].eligible is False
    assert cluster.workers[1].eligible is True
    assert "gpu memory" in " ".join(cluster.workers[0].exclusion_reasons).lower()


def test_failed_network_link_blocks_both_workers() -> None:
    manager = ResourceManager()
    workers = [_worker("worker-a"), _worker("worker-b")]
    network = _network_state(
        _link("worker-a", "worker-b", reachable=False, failure_reason="timeout"),
        _link("worker-b", "worker-a"),
    )

    cluster = manager.build_cluster_state(workers, network_state=network, require_network=True)

    assert all(not entry.eligible for entry in cluster.workers)
    assert any(
        "required network link failed" in " ".join(entry.exclusion_reasons).lower()
        for entry in cluster.workers
    )


def test_missing_required_link_blocks_workers_without_hiding_them() -> None:
    manager = ResourceManager()
    workers = [_worker("worker-a"), _worker("worker-b")]
    network = _network_state(_link("worker-a", "worker-b"))

    cluster = manager.build_cluster_state(workers, network_state=network, require_network=True)

    assert len(cluster.workers) == 2
    assert all(not entry.eligible for entry in cluster.workers)
    assert any(
        "missing required network link" in " ".join(entry.exclusion_reasons).lower()
        for entry in cluster.workers
    )


def test_unhealthy_worker_is_retained_in_cluster_state() -> None:
    manager = ResourceManager()
    cluster = manager.build_cluster_state(
        [_worker("worker-b"), _worker("worker-a", health=Health.FAILED)]
    )

    assert [entry.worker_id for entry in cluster.workers] == ["worker-a", "worker-b"]
    assert cluster.workers[0].eligible is False


def test_missing_runtime_capability_is_ineligible() -> None:
    manager = ResourceManager()
    cluster = manager.build_cluster_state(
        [_worker("worker-a", gpu_name=None), _worker("worker-b", python_executable=None)]
    )

    assert all(not entry.eligible for entry in cluster.workers)
    assert "gpu/runtime evidence missing" in " ".join(cluster.workers[0].exclusion_reasons).lower()


def test_more_than_two_workers_are_supported_deterministically() -> None:
    manager = ResourceManager()
    workers = [_worker("worker-c"), _worker("worker-a"), _worker("worker-b")]
    network = _network_state(
        _link("worker-a", "worker-b"),
        _link("worker-b", "worker-a"),
        _link("worker-a", "worker-c"),
        _link("worker-c", "worker-a"),
        _link("worker-b", "worker-c"),
        _link("worker-c", "worker-b"),
    )

    cluster = manager.build_cluster_state(workers, network_state=network, require_network=True)

    assert [entry.worker_id for entry in cluster.workers] == ["worker-a", "worker-b", "worker-c"]
    assert [entry.worker_id for entry in cluster.eligible_workers] == [
        "worker-a",
        "worker-b",
        "worker-c",
    ]


def test_cluster_state_output_is_deterministic() -> None:
    manager = ResourceManager()
    workers = [_worker("worker-b"), _worker("worker-a")]
    network = _network_state(_link("worker-b", "worker-a"), _link("worker-a", "worker-b"))

    first = manager.build_cluster_state(workers, network_state=network, now=_fixed_now())
    second = manager.build_cluster_state(
        list(reversed(workers)),
        network_state=network,
        now=_fixed_now(),
    )

    assert first.to_dict() == second.to_dict()
