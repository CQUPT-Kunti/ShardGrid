"""Minimal TrainingJob creation, lifecycle, and persistence helpers."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from shardgrid.common.enums import Health, JobState
from shardgrid.common.models import BackendName, JobId, WorkerId, as_job_id
from shardgrid.jobs.models import TrainingJob
from shardgrid.resources.models import NetworkState, WorkerResource


def _now() -> str:
    return datetime.now(tz=UTC).isoformat()


def create_training_job(
    *,
    config_path: str,
    model: str,
    requested_world_size: int,
    backend_preference: BackendName,
    runtime_environment_ref: str | None,
    job_id: JobId | None = None,
) -> TrainingJob:
    created_at = _now()
    return TrainingJob(
        job_id=_new_job_id() if job_id is None else job_id,
        config_path=config_path,
        model=model,
        requested_world_size=requested_world_size,
        backend_preference=backend_preference,
        runtime_environment_ref=runtime_environment_ref,
        created_at=created_at,
        updated_at=created_at,
    )


def save_training_job(job: TrainingJob, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.write_text(json.dumps(job.to_dict(), indent=2, sort_keys=True))
    return output_path


def load_training_job(path: str | Path) -> TrainingJob:
    return TrainingJob.from_dict(json.loads(Path(path).read_text()))


def can_launch_job(
    job: TrainingJob,
    *,
    workers: list[WorkerResource],
    network_state: NetworkState | None,
) -> bool:
    if not job.runtime_environment_ref:
        return False
    if len(workers) < job.requested_world_size:
        return False
    worker_ids = [worker.worker_id for worker in workers if worker.health is Health.HEALTHY]
    if len(worker_ids) < job.requested_world_size:
        return False
    if network_state is None:
        return False
    if not _covers_workers(network_state, worker_ids[: job.requested_world_size]):
        return False
    return True


def transition_training_job(
    job: TrainingJob,
    next_state: JobState,
    *,
    workers: list[WorkerResource] | None = None,
    network_state: NetworkState | None = None,
) -> TrainingJob:
    if next_state is JobState.LAUNCHING:
        if workers is None or not can_launch_job(
            job, workers=workers, network_state=network_state
        ):
            raise ValueError(
                "job cannot enter launching without eligible workers, "
                "network state, and runtime environment evidence"
            )
    transitioned = job.transition_to(next_state)
    if transitioned.created_at is None:
        return replace(transitioned, created_at=_now())
    return transitioned


def _new_job_id() -> JobId:
    return as_job_id(f"job-{datetime.now(tz=UTC):%Y%m%d%H%M%S}-{uuid4().hex[:8]}")


def _covers_workers(network_state: NetworkState, worker_ids: list[WorkerId]) -> bool:
    if not worker_ids:
        return False
    network_workers = set(network_state.workers)
    if not set(worker_ids).issubset(network_workers):
        return False
    for source in worker_ids:
        for target in worker_ids:
            if source == target:
                continue
            if not any(
                link.source_worker_id == source
                and link.target_worker_id == target
                and link.tcp_reachable
                for link in network_state.links
            ):
                return False
    return True

