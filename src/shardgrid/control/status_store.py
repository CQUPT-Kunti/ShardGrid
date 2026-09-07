"""Persistent JobStatus storage."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from datetime import UTC, datetime
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence

from shardgrid.common.enums import JobState
from shardgrid.common.models import JobId
from shardgrid.jobs.models import JobStatus, TrainingJob
from shardgrid.planner.models import WorkerAssignment


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

    def reserve_resources(
        self,
        job_id: JobId | str,
        assignments: Sequence[WorkerAssignment],
    ) -> list[dict[str, object]]:
        with self._reservation_lock():
            payload = self._load_reservations()
            existing = [
                item
                for item in payload.get("reservations", [])
                if str(item.get("job_id")) != str(job_id)
            ]
            timestamp = _now()
            payload["reservations"] = existing + [
                {
                    "job_id": str(job_id),
                    "worker_id": str(assignment.worker_id),
                    "rank": assignment.rank,
                    "stage": assignment.stage,
                    "gpu_index": assignment.gpu_index,
                    "estimated_peak_training_memory": assignment.estimated_peak_training_memory,
                    "created_at": timestamp,
                }
                for assignment in assignments
            ]
            self._write_reservations(payload)
            return []

    def release_resources(self, job_id: JobId | str) -> None:
        with self._reservation_lock():
            payload = self._load_reservations()
            payload["reservations"] = [
                item
                for item in payload.get("reservations", [])
                if str(item.get("job_id")) != str(job_id)
            ]
            self._write_reservations(payload)

    def active_reservations(self) -> list[dict[str, object]]:
        with self._reservation_lock():
            return list(self._load_reservations().get("reservations", []))

    def _reservations_path(self) -> Path:
        return self.root / "resource-reservations.json"

    def _load_reservations(self) -> dict[str, object]:
        path = self._reservations_path()
        if not path.exists():
            return {"reservations": []}
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return {"reservations": []}
        reservations = payload.get("reservations", [])
        if not isinstance(reservations, list):
            return {"reservations": []}
        return {"reservations": reservations}

    def _write_reservations(self, payload: dict[str, object]) -> None:
        path = self._reservations_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    @contextmanager
    def _reservation_lock(self) -> Iterator[None]:
        lock_path = self.root / "resource-reservations.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+", encoding="utf-8") as handle:
            if os.name == "posix":
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if os.name == "posix":
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
