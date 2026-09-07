from __future__ import annotations

from pathlib import Path

import pytest

from shardgrid.artifacts.store import ArtifactStore
from shardgrid.common.models import as_backend_name, as_job_id
from shardgrid.control.job_manager import create_training_job
from shardgrid.jobs.models import TrainingJob


def _job(*, job_id: str = "job-001", snapshot_path: str | None = None) -> TrainingJob:
    return create_training_job(
        job_id=as_job_id(job_id),
        config_path="examples/train-minimal.yaml",
        model="tiny-sequential",
        requested_world_size=2,
        backend_preference=as_backend_name("ssh"),
        runtime_environment_ref="env:worker-pair/shardgrid",
    ) if snapshot_path is None else TrainingJob(
        job_id=as_job_id(job_id),
        config_path="examples/train-minimal.yaml",
        model="tiny-sequential",
        requested_world_size=2,
        backend_preference=as_backend_name("ssh"),
        runtime_environment_ref="env:worker-pair/shardgrid",
        created_at="2026-08-27T00:00:00+00:00",
        updated_at="2026-08-27T00:00:00+00:00",
        snapshot_path=snapshot_path,
    )


def test_create_job_snapshot_builds_standard_layout(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "jobs")

    snapshot = store.create_snapshot(_job())

    assert Path(snapshot.root_path) == (tmp_path / "jobs" / "job-001")
    assert Path(snapshot.code_path).is_dir() is True
    assert Path(snapshot.config_path).is_dir() is True
    assert Path(snapshot.plan_path).is_dir() is True
    assert Path(snapshot.logs_path).is_dir() is True
    assert Path(snapshot.checkpoint_path).is_dir() is True
    assert Path(snapshot.environment_path).is_dir() is True
    assert Path(snapshot.diagnostics_path).is_dir() is True


def test_jobs_root_is_configurable_and_jobs_are_isolated(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "custom-jobs-root")

    first = store.create_snapshot(_job(job_id="job-a"))
    second = store.create_snapshot(_job(job_id="job-b"))

    assert Path(first.root_path).parent == tmp_path / "custom-jobs-root"
    assert Path(second.root_path).parent == tmp_path / "custom-jobs-root"
    assert first.root_path != second.root_path


def test_same_job_id_reuses_snapshot_identity(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "jobs")
    job = _job()

    first = store.create_snapshot(job)
    second = store.create_snapshot(job)

    assert first.root_path == second.root_path
    assert first.code_path == second.code_path


def test_path_traversal_and_escape_are_rejected(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "jobs")

    with pytest.raises(ValueError):
        store.snapshot_paths("../escape")

    with pytest.raises(ValueError):
        store.snapshot_paths("/tmp/escape")


def test_existing_valid_snapshot_is_reused(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "jobs")
    job = _job()
    (tmp_path / "jobs" / "job-001").mkdir(parents=True)

    snapshot = store.create_snapshot(job)

    assert Path(snapshot.root_path).is_dir() is True
    assert Path(snapshot.logs_path).is_dir() is True


def test_conflicting_existing_snapshot_is_not_overwritten(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "jobs")
    root = tmp_path / "jobs" / "job-001"
    root.mkdir(parents=True)
    (root / "logs").write_text("occupied")

    with pytest.raises(ValueError, match="conflict"):
        store.create_snapshot(_job())


def test_snapshot_path_mismatch_is_rejected(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "jobs")
    wrong_path = str(tmp_path / "other" / "job-001")

    with pytest.raises(ValueError, match="snapshot_path"):
        store.create_snapshot(_job(snapshot_path=wrong_path))
