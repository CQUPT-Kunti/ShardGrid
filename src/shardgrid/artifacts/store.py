"""Job snapshot path helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath
from typing import Mapping

from shardgrid.common.models import JobId, as_job_id
from shardgrid.jobs.models import JobSnapshot, TrainingJob

_JOB_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SNAPSHOT_DIRS = (
    "code",
    "config",
    "plan",
    "logs",
    "checkpoint",
    "environment",
    "diagnostics",
)


def validate_job_id(job_id: JobId | str) -> JobId:
    normalized = as_job_id(str(job_id))
    if not _JOB_ID_PATTERN.fullmatch(str(normalized)):
        raise ValueError("job_id contains unsupported path characters")
    if any(token in {".", ".."} for token in PurePath(str(normalized)).parts):
        raise ValueError("job_id must not contain path traversal")
    return normalized


@dataclass(frozen=True)
class JobSnapshotPaths:
    root: Path
    code: Path
    config: Path
    plan: Path
    logs: Path
    checkpoint: Path
    environment: Path
    diagnostics: Path

    def create(self) -> JobSnapshotPaths:
        for path in (
            self.root,
            self.code,
            self.config,
            self.plan,
            self.logs,
            self.checkpoint,
            self.environment,
            self.diagnostics,
        ):
            path.mkdir(parents=True, exist_ok=True)
        return self


class ArtifactStore:
    def __init__(self, jobs_root: Path | str) -> None:
        root = Path(jobs_root)
        if not root.is_absolute():
            raise ValueError("jobs_root must be an absolute path")
        self.jobs_root = root.resolve()

    def create_snapshot(self, job: TrainingJob) -> JobSnapshot:
        paths = self.snapshot_paths(job.job_id)
        expected_root = str(paths.root)
        if job.snapshot_path is not None and Path(job.snapshot_path).resolve() != paths.root:
            raise ValueError("job snapshot_path does not match configured jobs_root")
        self._prepare_job_root(paths)
        paths.create()
        created_at = job.created_at or datetime.now(tz=UTC).isoformat()
        return JobSnapshot(
            job_id=job.job_id,
            root_path=expected_root,
            code_path=str(paths.code),
            config_path=str(paths.config),
            plan_path=str(paths.plan),
            logs_path=str(paths.logs),
            environment_path=str(paths.environment),
            checkpoint_path=str(paths.checkpoint),
            diagnostics_path=str(paths.diagnostics),
            created_at=created_at,
        )

    def snapshot_paths(self, job_id: JobId | str) -> JobSnapshotPaths:
        normalized = validate_job_id(job_id)
        job_root = self.jobs_root / str(normalized)
        self._ensure_contained(job_root)
        return JobSnapshotPaths(
            root=job_root,
            code=job_root / "code",
            config=job_root / "config",
            plan=job_root / "plan",
            logs=job_root / "logs",
            checkpoint=job_root / "checkpoint",
            environment=job_root / "environment",
            diagnostics=job_root / "diagnostics",
        )

    def _ensure_contained(self, path: Path) -> None:
        candidate = path.resolve(strict=False)
        if self.jobs_root not in candidate.parents and candidate != self.jobs_root:
            raise ValueError("snapshot path escaped jobs_root")

    def _prepare_job_root(self, paths: JobSnapshotPaths) -> None:
        if paths.root.exists() and not paths.root.is_dir():
            raise ValueError("snapshot root is occupied by a non-directory path")
        for path in (
            paths.code,
            paths.config,
            paths.plan,
            paths.logs,
            paths.checkpoint,
            paths.environment,
            paths.diagnostics,
        ):
            if path.exists() and not path.is_dir():
                raise ValueError("snapshot path conflict with existing non-directory artifact")
            self._ensure_contained(path)


def build_job_snapshot_paths(
    jobs_root: PurePath | str,
    job_id: JobId | str,
) -> Mapping[str, PurePath]:
    root_path = jobs_root if isinstance(jobs_root, PurePath) else PurePath(jobs_root)
    pure_type = PureWindowsPath if isinstance(root_path, PureWindowsPath) else PurePosixPath
    normalized = str(validate_job_id(job_id))
    root = pure_type(root_path) / normalized
    return {name: root / name for name in _SNAPSHOT_DIRS} | {"root": root}
