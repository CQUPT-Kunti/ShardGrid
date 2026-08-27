from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from shardgrid.artifacts.collector import (
    ArtifactCollectionState,
    ArtifactCollector,
    CollectionStatus,
    WorkerArtifactSource,
)
from shardgrid.artifacts.store import ArtifactStore
from shardgrid.artifacts.transport import (
    ArtifactTransferItemResult,
    ArtifactTransferResult,
    ArtifactTransferStatus,
    ArtifactTransportName,
    RemoteArtifactLocation,
)
from shardgrid.common.config import ClusterConfig
from shardgrid.common.enums import FailureStage
from shardgrid.common.models import as_backend_name, as_job_id
from shardgrid.control.job_manager import create_training_job
from shardgrid.planner.models import WorkerAssignment


def _config(tmp_path: Path) -> ClusterConfig:
    return ClusterConfig.from_dict(
        {
            "control": {"machine_id": "machine-a", "hostname": "control-a.local"},
            "jobs_root": str((tmp_path / "jobs").resolve()),
            "ssh": {"private_key_path": "/tmp/private.key"},
            "runtime": {},
            "network": {},
            "backend_preference": {},
            "manual_override": {},
            "workers": [
                {
                    "id": "gpu4060",
                    "machine_id": "machine-c",
                    "physical_os": "windows",
                    "runtime_os": "wsl2_linux",
                    "runtime": "wsl2",
                    "host": "10.0.0.10",
                    "ssh_user": "shardgrid",
                },
                {
                    "id": "gpu1060",
                    "machine_id": "machine-d",
                    "physical_os": "windows",
                    "runtime_os": "wsl2_linux",
                    "runtime": "wsl2",
                    "host": "10.0.0.11",
                    "ssh_user": "shardgrid",
                },
            ],
        }
    )


def _snapshot(tmp_path: Path):
    config = _config(tmp_path)
    job = create_training_job(
        job_id=as_job_id("job-089"),
        config_path="examples/train-minimal.yaml",
        model="tiny-sequential",
        requested_world_size=2,
        backend_preference=as_backend_name("ssh"),
        runtime_environment_ref="env:worker-pair/shardgrid",
    )
    snapshot = ArtifactStore(config.jobs_root).create_snapshot(job)
    return config, snapshot


def _assignment(worker_id: str, rank: int, stage: str) -> WorkerAssignment:
    return WorkerAssignment(worker_id=worker_id, rank=rank, stage=stage)


def _source(config: ClusterConfig, worker_id: str, rank: int, stage: str, remote_root: Path):
    worker = next(item for item in config.workers if str(item.worker_id) == worker_id)
    return WorkerArtifactSource.from_worker_assignment(
        worker=worker,
        assignment=_assignment(worker_id, rank, stage),
        remote_root=str(remote_root),
    )


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


class _FakePullTransport:
    name = ArtifactTransportName.SCP

    def __init__(self, failures: dict[str, str] | None = None) -> None:
        self.failures = failures or {}
        self.calls: list[str] = []

    def transfer(self, items, *, remote: RemoteArtifactLocation, secrets=()):
        results = []
        for item in items:
            self.calls.append(remote.private_key_path or "")
            remote_path = Path(item.source)
            label = item.label
            failure = self.failures.get(label)
            if failure:
                results.append(
                    ArtifactTransferItemResult(
                        label=label,
                        transport=self.name.value,
                        status=ArtifactTransferStatus.FAILED,
                        source=item.source,
                        destination=item.destination,
                        recorded_command=f"scp {remote.private_key_path or ''}",
                        exit_code=1,
                        stderr=failure,
                    )
                )
                continue
            if not remote_path.exists():
                results.append(
                    ArtifactTransferItemResult(
                        label=label,
                        transport=self.name.value,
                        status=ArtifactTransferStatus.FAILED,
                        source=item.source,
                        destination=item.destination,
                        recorded_command=f"scp {remote.private_key_path or ''}",
                        exit_code=1,
                        stderr="No such file or directory",
                    )
                )
                continue
            destination = Path(item.destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(remote_path, destination)
            results.append(
                ArtifactTransferItemResult(
                    label=label,
                    transport=self.name.value,
                    status=ArtifactTransferStatus.SUCCESS,
                    source=item.source,
                    destination=item.destination,
                    recorded_command=f"scp {remote.private_key_path or ''}",
                    exit_code=0,
                )
            )
        status = ArtifactTransferStatus.SUCCESS
        if any(item.status is ArtifactTransferStatus.FAILED for item in results):
            status = (
                ArtifactTransferStatus.PARTIAL
                if any(item.status is ArtifactTransferStatus.SUCCESS for item in results)
                else ArtifactTransferStatus.FAILED
            )
        return ArtifactTransferResult(transport=self.name.value, status=status, items=results)


def test_collects_rank_logs_diagnostics_metadata_and_optional_checkpoint(tmp_path: Path) -> None:
    config, snapshot = _snapshot(tmp_path)
    worker_a = tmp_path / "remote-a"
    worker_b = tmp_path / "remote-b"
    for root, rank in ((worker_a, 0), (worker_b, 1)):
        _write(root / "logs" / "stdout.log", f"stdout-{rank}")
        _write(root / "logs" / "stderr.log", f"stderr-{rank}")
        _write(root / "diagnostics" / "runtime.json", json.dumps({"rank": rank}))
        _write(root / "checkpoint" / "checkpoint-metadata.json", '{"status":"complete"}')
        _write(root / "checkpoint" / "model.pt", f"checkpoint-{rank}")

    result = ArtifactCollector(transport=_FakePullTransport()).collect(
        snapshot,
        sources=[
            _source(config, "gpu4060", 0, "stage0", worker_a),
            _source(config, "gpu1060", 1, "stage1", worker_b),
        ],
        secrets=["/tmp/private.key"],
    )

    assert result.status is CollectionStatus.SUCCESS
    assert len(result.workers) == 2
    assert "stdout-0" in (
        Path(snapshot.logs_path, "gpu4060", "rank0-stage0", "stdout.log").read_text()
    )
    assert "stderr-1" in (
        Path(snapshot.logs_path, "gpu1060", "rank1-stage1", "stderr.log").read_text()
    )
    assert json.loads(
        Path(snapshot.diagnostics_path, "gpu4060", "rank0-stage0", "runtime.json").read_text()
    )["rank"] == 0
    checkpoint = Path(
        snapshot.checkpoint_path, "files", "gpu1060", "rank1-stage1", "model.pt"
    )
    assert checkpoint.read_text() == "checkpoint-1"


def test_partial_failure_keeps_success_and_marks_partial_checkpoint(
    tmp_path: Path,
) -> None:
    config, snapshot = _snapshot(tmp_path)
    worker_a = tmp_path / "remote-a"
    worker_b = tmp_path / "remote-b"
    _write(worker_a / "logs" / "stdout.log", "stdout-0")
    _write(worker_a / "logs" / "stderr.log", "stderr-0")
    _write(worker_a / "diagnostics" / "runtime.json", '{"ok": true}')
    _write(worker_a / "checkpoint" / "checkpoint-metadata.json", '{"status":"complete"}')
    _write(worker_a / "checkpoint" / "model.pt", "checkpoint-0")

    _write(worker_b / "logs" / "stderr.log", "stderr-1")
    _write(worker_b / "diagnostics" / "failure.json", '{"reason":"rank failed"}')
    _write(worker_b / "checkpoint" / "checkpoint-metadata.json", '{"status":"complete"}')

    result = ArtifactCollector(transport=_FakePullTransport()).collect(
        snapshot,
        sources=[
            _source(config, "gpu4060", 0, "stage0", worker_a),
            _source(config, "gpu1060", 1, "stage1", worker_b),
        ],
    )

    assert result.status is CollectionStatus.PARTIAL
    worker_ok = next(item for item in result.workers if item.worker_id == "gpu4060")
    worker_fail = next(item for item in result.workers if item.worker_id == "gpu1060")
    assert worker_ok.status is CollectionStatus.SUCCESS
    assert worker_fail.status is CollectionStatus.PARTIAL
    assert worker_fail.checkpoint_state is ArtifactCollectionState.PARTIAL
    checkpoint = Path(
        snapshot.checkpoint_path, "files", "gpu4060", "rank0-stage0", "model.pt"
    )
    assert checkpoint.read_text() == "checkpoint-0"
    assert json.loads(
        Path(snapshot.diagnostics_path, "gpu1060", "rank1-stage1", "failure.json").read_text()
    )["reason"] == "rank failed"


def test_existing_completed_artifact_survives_empty_or_failed_transfer(
    tmp_path: Path,
) -> None:
    config, snapshot = _snapshot(tmp_path)
    worker = tmp_path / "remote-a"
    _write(worker / "checkpoint" / "checkpoint-metadata.json", '{"status":"complete"}')
    _write(worker / "checkpoint" / "model.pt", "")
    local_checkpoint = Path(
        snapshot.checkpoint_path, "files", "gpu4060", "rank0-stage0", "model.pt"
    )
    _write(local_checkpoint, "good-checkpoint")

    result = ArtifactCollector(
        transport=_FakePullTransport({"gpu4060-rank0-stage0-stdout": "permission denied"})
    ).collect(
        snapshot,
        sources=[_source(config, "gpu4060", 0, "stage0", worker)],
    )

    worker_result = result.workers[0]
    assert result.status is CollectionStatus.PARTIAL
    assert worker_result.checkpoint_state is ArtifactCollectionState.PARTIAL
    assert local_checkpoint.read_text() == "good-checkpoint"
    assert any(
        artifact.failure is not None and artifact.failure.stage is FailureStage.CHECKPOINT
        for artifact in worker_result.artifacts
    )


def test_repeated_collection_is_idempotent_and_preserves_identity(
    tmp_path: Path,
) -> None:
    config, snapshot = _snapshot(tmp_path)
    worker = tmp_path / "remote-a"
    _write(worker / "logs" / "stdout.log", "stdout-0")
    _write(worker / "logs" / "stderr.log", "stderr-0")
    _write(worker / "diagnostics" / "runtime.json", '{"rank":0}')
    _write(worker / "checkpoint" / "checkpoint-metadata.json", '{"status":"partial"}')
    _write(worker / "checkpoint" / "model.pt", "checkpoint-0")

    collector = ArtifactCollector(transport=_FakePullTransport())
    source = _source(config, "gpu4060", 0, "stage0", worker)

    first = collector.collect(snapshot, sources=[source])
    second = collector.collect(snapshot, sources=[source])

    assert first.status is CollectionStatus.PARTIAL
    assert second.status is CollectionStatus.PARTIAL
    assert second.workers[0].worker_id == "gpu4060"
    assert second.workers[0].rank == 0
    assert second.workers[0].stage == "stage0"
    assert all(
        "private.key" not in (artifact.recorded_command or "")
        for artifact in second.workers[0].artifacts
    )

    escaped = _source(config, "gpu4060", 0, "stage0", worker)
    with pytest.raises(ValueError, match="artifact path escaped"):
        collector.collect(
            snapshot,
            sources=[escaped],
            artifact_paths=("logs/../../escape.txt",),
        )
