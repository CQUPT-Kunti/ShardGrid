from __future__ import annotations

from pathlib import Path, PurePosixPath, PureWindowsPath

import pytest

from shardgrid.artifacts.store import ArtifactStore, build_job_snapshot_paths, validate_job_id


def test_valid_job_id_builds_snapshot_structure(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)

    snapshot = store.snapshot_paths("job-001").create()

    assert snapshot.root == tmp_path / "job-001"
    assert snapshot.code == tmp_path / "job-001" / "code"
    assert snapshot.config == tmp_path / "job-001" / "config"
    assert snapshot.plan == tmp_path / "job-001" / "plan"
    assert snapshot.logs == tmp_path / "job-001" / "logs"
    assert snapshot.checkpoint == tmp_path / "job-001" / "checkpoint"
    assert snapshot.environment == tmp_path / "job-001" / "environment"
    assert snapshot.diagnostics == tmp_path / "job-001" / "diagnostics"
    assert snapshot.logs.is_dir() is True


@pytest.mark.parametrize("job_id", ["../escape", "..", "job/../../evil", "job\\..\\evil"])
def test_invalid_job_id_is_rejected(job_id: str) -> None:
    with pytest.raises(ValueError):
        validate_job_id(job_id)


def test_snapshot_paths_stay_within_jobs_root(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "jobs")

    snapshot = store.snapshot_paths("safe-job")

    assert snapshot.root.parent == tmp_path / "jobs"
    assert (tmp_path / "jobs") in snapshot.diagnostics.parents


def test_linux_and_windows_path_behavior() -> None:
    linux_paths = build_job_snapshot_paths(PurePosixPath("/srv/jobs"), "job-001")
    windows_paths = build_job_snapshot_paths(PureWindowsPath(r"C:\jobs"), "job-001")

    assert linux_paths["root"] == PurePosixPath("/srv/jobs/job-001")
    assert linux_paths["logs"] == PurePosixPath("/srv/jobs/job-001/logs")
    assert windows_paths["root"] == PureWindowsPath(r"C:\jobs\job-001")
    assert windows_paths["logs"] == PureWindowsPath(r"C:\jobs\job-001\logs")


def test_jobs_root_must_be_absolute() -> None:
    with pytest.raises(ValueError, match="absolute"):
        ArtifactStore("relative/jobs")
