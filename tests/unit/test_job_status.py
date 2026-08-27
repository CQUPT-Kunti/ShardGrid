from __future__ import annotations

from pathlib import Path

import pytest

from shardgrid.common.enums import FailureStage, JobState
from shardgrid.common.models import as_backend_name, as_job_id, as_worker_id
from shardgrid.control.status_store import StatusStore
from shardgrid.jobs.models import FailureRecord, JobStatus, TrainingJob
from shardgrid.planner.models import WorkerAssignment


def _job() -> TrainingJob:
    return TrainingJob(
        job_id=as_job_id("job-0091"),
        config_path="examples/train-minimal.yaml",
        model="tiny-sequential",
        requested_world_size=2,
        backend_preference=as_backend_name("ssh"),
        runtime_environment_ref="env:cluster/shardgrid",
        created_at="2026-08-27T09:00:00+00:00",
        updated_at="2026-08-27T09:00:00+00:00",
    )


def _assignments() -> list[WorkerAssignment]:
    return [
        WorkerAssignment(
            worker_id=as_worker_id("gpu4060"),
            rank=0,
            local_rank=0,
            stage="0",
            conda_environment="shardgrid",
            conda_prefix="/home/shardgrid/miniconda3/envs/shardgrid",
            python_executable="/home/shardgrid/miniconda3/envs/shardgrid/bin/python",
        ),
        WorkerAssignment(
            worker_id=as_worker_id("gpu1060"),
            rank=1,
            local_rank=0,
            stage="1",
            conda_environment="shardgrid",
            conda_prefix="/home/shardgrid/miniconda3/envs/shardgrid",
            python_executable="/home/shardgrid/miniconda3/envs/shardgrid/bin/python",
        ),
    ]


def _failure() -> FailureRecord:
    return FailureRecord(
        stage=FailureStage.TRAIN,
        host="ldj",
        worker_id=as_worker_id("gpu4060"),
        command="torchrun train.py",
        exit_code=1,
        runtime_environment={"worker": "gpu4060", "runtime": "wsl2"},
        python_executable="/home/shardgrid/miniconda3/envs/shardgrid/bin/python",
        conda_environment="shardgrid",
        conda_prefix="/home/shardgrid/miniconda3/envs/shardgrid",
        message="rank 0 loss diverged",
        recommended_action="inspect rank logs and CUDA runtime state",
    )


def test_status_store_create_save_and_load_round_trip(tmp_path: Path) -> None:
    store = StatusStore(tmp_path)
    job = _job()

    created = store.create_initial_status(job)
    running = store.save(
        JobStatus(
            job_id=job.job_id,
            state=JobState.TRAINING,
            phase="training",
            workers=[assignment.worker_id for assignment in _assignments()],
            assignments=_assignments(),
            runtime_environment_refs={"0": "env:gpu4060/shardgrid", "1": "env:gpu1060/shardgrid"},
            loss_history=[1.2, 0.8, 0.6],
            latest_loss=0.6,
            backend=as_backend_name("nccl"),
            fallback_used=False,
            started_at=created.started_at,
        )
    )

    restored = store.load(job.job_id)

    assert created.state is JobState.CREATED
    assert running == restored
    assert restored.assignments[1].stage == "1"
    assert restored.runtime_environment_refs["1"] == "env:gpu1060/shardgrid"


def test_status_store_persists_terminal_states_and_repeated_updates(tmp_path: Path) -> None:
    store = StatusStore(tmp_path)
    store.create_initial_status(_job())

    completed = store.save(
        JobStatus(
            job_id=as_job_id("job-0091"),
            state=JobState.COMPLETED,
            phase="checkpoint",
            workers=[assignment.worker_id for assignment in _assignments()],
            assignments=_assignments(),
            runtime_environment_refs={"0": "env:gpu4060/shardgrid", "1": "env:gpu1060/shardgrid"},
            loss_history=[1.0, 0.5, 0.25],
            latest_loss=0.25,
            final_metrics={"final_loss": 0.25},
            checkpoint_ref="jobs/job-0091/checkpoint/model.pt",
        )
    )

    repeated = store.save(completed)

    assert repeated.finished_at is not None
    assert store.load("job-0091") == repeated


def test_job_status_allows_legal_transitions_and_rejects_illegal_terminal_backtracking() -> None:
    status = JobStatus(job_id=as_job_id("job-0091"), state=JobState.CREATED, phase="created")

    for next_state in (
        JobState.PROBING,
        JobState.PLANNING,
        JobState.SNAPSHOTTING,
        JobState.DISTRIBUTING,
        JobState.LAUNCHING,
        JobState.RENDEZVOUS,
        JobState.TRAINING,
        JobState.CHECKPOINTING,
    ):
        status = status.transition_to(next_state, phase=next_state.value)

    status = status.transition_to(
        JobState.COMPLETED,
        phase="checkpoint",
        checkpoint_ref="jobs/job-0091/checkpoint/model.pt",
        final_metrics={"final_loss": 0.2},
    )
    assert status.state is JobState.COMPLETED

    with pytest.raises(ValueError, match="invalid job status transition"):
        status.transition_to(JobState.TRAINING, phase="training")


def test_failed_status_persists_failure_record_and_runtime_context(tmp_path: Path) -> None:
    store = StatusStore(tmp_path)
    failed = store.save(
        JobStatus(
            job_id=as_job_id("job-0091"),
            state=JobState.FAILED,
            phase="training",
            workers=[assignment.worker_id for assignment in _assignments()],
            assignments=_assignments(),
            runtime_environment_refs={"0": "env:gpu4060/shardgrid", "1": "env:gpu1060/shardgrid"},
            backend=as_backend_name("gloo"),
            fallback_used=True,
            failure=_failure(),
        )
    )

    restored = store.load("job-0091")

    assert failed.finished_at is not None
    assert restored.failure == _failure()
    assert restored.failure.runtime_environment["runtime"] == "wsl2"
    assert restored.fallback_used is True


def test_completed_status_requires_checkpoint_and_final_loss() -> None:
    with pytest.raises(ValueError, match="checkpoint_ref"):
        JobStatus(job_id=as_job_id("job-0091"), state=JobState.COMPLETED, phase="checkpoint")

    with pytest.raises(ValueError, match="final_metrics"):
        JobStatus(
            job_id=as_job_id("job-0091"),
            state=JobState.COMPLETED,
            phase="checkpoint",
            checkpoint_ref="jobs/job-0091/checkpoint/model.pt",
        )
