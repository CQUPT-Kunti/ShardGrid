from __future__ import annotations

from pathlib import Path

import pytest

from shardgrid.common.enums import Health, JobState, PhysicalOS, RuntimeOS
from shardgrid.common.models import as_backend_name, as_hostname, as_worker_id
from shardgrid.control.job_manager import (
    can_launch_job,
    create_training_job,
    load_training_job,
    save_training_job,
    transition_training_job,
)
from shardgrid.resources.models import NetworkLink, NetworkState, WorkerResource


def _worker(worker_id: str, *, health: Health = Health.HEALTHY) -> WorkerResource:
    return WorkerResource(
        worker_id=as_worker_id(worker_id),
        hostname=as_hostname(f"{worker_id}.local"),
        physical_os=PhysicalOS.WINDOWS,
        runtime_os=RuntimeOS.WSL2_LINUX,
        conda_environment="shardgrid",
        python_executable="python",
        health=health,
    )


def _network_state() -> NetworkState:
    return NetworkState(
        network_id="net-1",
        workers=[as_worker_id("worker-a"), as_worker_id("worker-b")],
        links=[
            NetworkLink(
                source_worker_id=as_worker_id("worker-a"),
                target_worker_id=as_worker_id("worker-b"),
                source_ip="10.0.0.1",
                target_ip="10.0.0.2",
                interface="eth0",
                tcp_reachable=True,
            ),
            NetworkLink(
                source_worker_id=as_worker_id("worker-b"),
                target_worker_id=as_worker_id("worker-a"),
                source_ip="10.0.0.2",
                target_ip="10.0.0.1",
                interface="eth0",
                tcp_reachable=True,
            ),
        ],
    )


def test_valid_job_creation_sets_identity_runtime_ref_and_timestamps() -> None:
    job = create_training_job(
        config_path="examples/train-minimal.yaml",
        model="tiny-sequential",
        requested_world_size=2,
        backend_preference=as_backend_name("ssh"),
        runtime_environment_ref="env:worker-pair/shardgrid",
    )

    assert str(job.job_id).startswith("job-")
    assert job.runtime_environment_ref == "env:worker-pair/shardgrid"
    assert job.created_at is not None
    assert job.updated_at == job.created_at


def test_invalid_job_creation_is_rejected() -> None:
    with pytest.raises(ValueError):
        create_training_job(
            config_path="examples/train-minimal.yaml",
            model="tiny-sequential",
            requested_world_size=0,
            backend_preference=as_backend_name("ssh"),
            runtime_environment_ref="env:worker-pair/shardgrid",
        )

    with pytest.raises(ValueError):
        create_training_job(
            config_path="examples/train-minimal.yaml",
            model="tiny-sequential",
            requested_world_size=2,
            backend_preference=as_backend_name("ssh"),
            runtime_environment_ref=" ",
        )


def test_job_id_is_unique_across_consecutive_creates() -> None:
    first = create_training_job(
        config_path="examples/train-minimal.yaml",
        model="tiny-sequential",
        requested_world_size=2,
        backend_preference=as_backend_name("ssh"),
        runtime_environment_ref="env:1",
    )
    second = create_training_job(
        config_path="examples/train-minimal.yaml",
        model="tiny-sequential",
        requested_world_size=2,
        backend_preference=as_backend_name("ssh"),
        runtime_environment_ref="env:1",
    )

    assert first.job_id != second.job_id


def test_valid_and_invalid_lifecycle_transitions() -> None:
    job = create_training_job(
        config_path="examples/train-minimal.yaml",
        model="tiny-sequential",
        requested_world_size=2,
        backend_preference=as_backend_name("ssh"),
        runtime_environment_ref="env:1",
    )
    probing = transition_training_job(job, JobState.PROBING)
    assert probing.state is JobState.PROBING
    assert probing.updated_at != job.updated_at

    completed = probing
    for state in (
        JobState.PLANNING,
        JobState.SNAPSHOTTING,
        JobState.DISTRIBUTING,
        JobState.LAUNCHING,
        JobState.RENDEZVOUS,
        JobState.TRAINING,
        JobState.CHECKPOINTING,
        JobState.COMPLETED,
    ):
        kwargs = {}
        if state is JobState.LAUNCHING:
            kwargs = {
                "workers": [_worker("worker-a"), _worker("worker-b")],
                "network_state": _network_state(),
            }
        completed = transition_training_job(completed, state, **kwargs)

    with pytest.raises(ValueError):
        transition_training_job(completed, JobState.TRAINING)


def test_launch_eligibility_requires_workers_network_and_runtime_evidence() -> None:
    job = create_training_job(
        config_path="examples/train-minimal.yaml",
        model="tiny-sequential",
        requested_world_size=2,
        backend_preference=as_backend_name("ssh"),
        runtime_environment_ref="env:1",
    )
    job = transition_training_job(job, JobState.PROBING)
    job = transition_training_job(job, JobState.PLANNING)
    job = transition_training_job(job, JobState.SNAPSHOTTING)
    job = transition_training_job(job, JobState.DISTRIBUTING)
    workers = [_worker("worker-a"), _worker("worker-b")]
    network_state = _network_state()

    assert can_launch_job(job, workers=workers, network_state=network_state) is True

    missing_runtime = create_training_job(
        config_path="examples/train-minimal.yaml",
        model="tiny-sequential",
        requested_world_size=2,
        backend_preference=as_backend_name("ssh"),
        runtime_environment_ref=None,
    )
    assert can_launch_job(
        missing_runtime, workers=workers, network_state=network_state
    ) is False
    assert can_launch_job(job, workers=workers[:1], network_state=network_state) is False
    assert can_launch_job(job, workers=workers, network_state=None) is False

    with pytest.raises(ValueError):
        transition_training_job(job, JobState.LAUNCHING, workers=workers[:1], network_state=None)


def test_persistence_round_trip_keeps_core_fields() -> None:
    job = create_training_job(
        config_path="examples/train-minimal.yaml",
        model="tiny-sequential",
        requested_world_size=2,
        backend_preference=as_backend_name("ssh"),
        runtime_environment_ref="env:worker-pair/shardgrid",
    )
    path = Path("tmp-job.json")
    try:
        save_training_job(job, path)
        restored = load_training_job(path)
    finally:
        path.unlink(missing_ok=True)

    assert restored == job
