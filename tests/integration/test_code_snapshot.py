from __future__ import annotations

from pathlib import Path

import pytest

from shardgrid.artifacts.snapshot import create_code_snapshot
from shardgrid.artifacts.store import ArtifactStore
from shardgrid.common.models import as_backend_name, as_job_id
from shardgrid.control.job_manager import create_training_job

_SECRET = "TEST_PASSWORD_DO_NOT_LEAK"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _job_snapshot(tmp_path: Path, *, job_id: str = "job-001"):
    store = ArtifactStore((tmp_path / "jobs").resolve())
    job = create_training_job(
        job_id=as_job_id(job_id),
        config_path="examples/train-minimal.yaml",
        model="tiny-sequential",
        requested_world_size=2,
        backend_preference=as_backend_name("ssh"),
        runtime_environment_ref="env:worker-pair/shardgrid",
    )
    return store.create_snapshot(job)


def _source_tree(tmp_path: Path) -> Path:
    root = tmp_path / "source"
    _write(root / "src/shardgrid/cli/app.py", "print('hello')\n")
    _write(root / "examples/train-minimal.yaml", "job:\n  name: train-minimal\n")
    _write(root / "examples/models/train_pipeline.py", "PIPELINE = 'ok'\n")
    _write(root / "examples/models/stage0.py", "class Stage0: pass\n")
    _write(root / "examples/models/__pycache__/stage0.cpython-313.pyc", "x")
    _write(root / ".pytest_cache/v/cache/nodeids", "[]")
    _write(root / "jobs/old-job/debug.log", "skip me")
    _write(root / ".env", f"PASSWORD={_SECRET}\n")
    _write(root / "examples/models/token_secret.txt", _SECRET)
    _write(root / "examples/models/runtime.log", "debug")
    _write(root / "examples/models/id_rsa", "private")
    return root


def test_create_code_snapshot_includes_supported_training_code(tmp_path: Path) -> None:
    source_root = _source_tree(tmp_path)
    snapshot = _job_snapshot(tmp_path)

    result = create_code_snapshot(snapshot, source_root=source_root, secrets=[_SECRET])

    assert result.root_path == snapshot.code_path
    assert (Path(snapshot.code_path) / "src/shardgrid/cli/app.py").is_file() is True
    assert (Path(snapshot.code_path) / "examples/train-minimal.yaml").is_file() is True
    assert (Path(snapshot.code_path) / "examples/models/train_pipeline.py").is_file() is True
    assert result.checksum


def test_transient_and_secret_files_are_excluded(tmp_path: Path) -> None:
    source_root = _source_tree(tmp_path)
    snapshot = _job_snapshot(tmp_path)

    create_code_snapshot(snapshot, source_root=source_root, secrets=[_SECRET])
    code_root = Path(snapshot.code_path)

    assert (code_root / "examples/models/__pycache__").exists() is False
    assert (code_root / ".pytest_cache").exists() is False
    assert (code_root / "jobs").exists() is False
    assert (code_root / ".env").exists() is False
    assert (code_root / "examples/models/token_secret.txt").exists() is False
    assert (code_root / "examples/models/runtime.log").exists() is False
    assert (code_root / "examples/models/id_rsa").exists() is False


def test_checksum_is_stable_for_same_content_and_changes_when_content_changes(
    tmp_path: Path,
) -> None:
    source_root = _source_tree(tmp_path)

    first = create_code_snapshot(_job_snapshot(tmp_path, job_id="job-a"), source_root=source_root)
    second = create_code_snapshot(_job_snapshot(tmp_path, job_id="job-b"), source_root=source_root)
    assert first.checksum == second.checksum

    _write(source_root / "src/shardgrid/cli/app.py", "print('changed')\n")
    third = create_code_snapshot(_job_snapshot(tmp_path, job_id="job-c"), source_root=source_root)
    assert third.checksum != first.checksum


def test_existing_snapshot_is_immutable_and_repeat_create_reuses_it(tmp_path: Path) -> None:
    source_root = _source_tree(tmp_path)
    snapshot = _job_snapshot(tmp_path)

    first = create_code_snapshot(snapshot, source_root=source_root)
    original = (Path(snapshot.code_path) / "src/shardgrid/cli/app.py").read_text()
    _write(source_root / "src/shardgrid/cli/app.py", "print('mutated')\n")
    second = create_code_snapshot(snapshot, source_root=source_root)

    assert second.checksum == first.checksum
    assert (Path(snapshot.code_path) / "src/shardgrid/cli/app.py").read_text() == original


def test_existing_nonempty_snapshot_without_manifest_is_rejected(tmp_path: Path) -> None:
    source_root = _source_tree(tmp_path)
    snapshot = _job_snapshot(tmp_path)
    _write(Path(snapshot.code_path) / "stale.txt", "occupied")

    with pytest.raises(ValueError, match="refusing to overwrite"):
        create_code_snapshot(snapshot, source_root=source_root)


def test_include_path_traversal_is_rejected(tmp_path: Path) -> None:
    source_root = _source_tree(tmp_path)
    snapshot = _job_snapshot(tmp_path)

    with pytest.raises(ValueError, match="within the source root"):
        create_code_snapshot(snapshot, source_root=source_root, includes=("../outside",))


def test_symlink_escape_is_rejected(tmp_path: Path) -> None:
    source_root = _source_tree(tmp_path)
    outside = tmp_path / "outside.py"
    outside.write_text("print('x')\n")
    (source_root / "examples/models/linked.py").symlink_to(outside)
    snapshot = _job_snapshot(tmp_path)

    with pytest.raises(ValueError, match="symlink"):
        create_code_snapshot(snapshot, source_root=source_root)
