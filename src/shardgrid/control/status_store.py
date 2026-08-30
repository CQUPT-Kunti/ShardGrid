"""Persistent JobStatus storage."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from shardgrid.common.enums import JobState
from shardgrid.common.models import JobId
from shardgrid.jobs.models import JobStatus, TrainingJob


def _now() -> str:
    return datetime.now(tz=UTC).isoformat()


class StatusStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def status_path(self, job_id: JobId | str) -> Path:
        return self.root / str(job_id) / "job-status.json"

    def create_initial_status(self, job: TrainingJob) -> JobStatus:
        timestamp = _now()
        status = JobStatus(
            job_id=job.job_id,
            state=JobState.CREATED,
            phase=JobState.CREATED.value,
            started_at=job.created_at or timestamp,
        )
        return self.save(status)

    def save(self, status: JobStatus) -> JobStatus:
        return self.save_path(self.status_path(status.job_id), status)

    def save_path(self, path: str | Path, status: JobStatus) -> JobStatus:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        persisted = status
        if persisted.state in {JobState.COMPLETED, JobState.FAILED, JobState.STOPPED}:
            if persisted.finished_at is None:
                persisted = replace(persisted, finished_at=_now())
        path.write_text(json.dumps(persisted.to_dict(), indent=2, sort_keys=True))
        return persisted

    def load(self, job_id: JobId | str) -> JobStatus:
        return self.load_path(self.status_path(job_id))

    def load_path(self, path: str | Path) -> JobStatus:
        return JobStatus.from_dict(json.loads(Path(path).read_text()))
