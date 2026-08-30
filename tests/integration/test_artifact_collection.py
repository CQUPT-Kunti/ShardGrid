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
                    "physical_os": "linux",
                    "runtime_os": "linux",
                    "runtime": "ssh",
                    "host": "10.0.0.10",
                    "ssh_user": "shardgrid",
                },
                {
                    "id": "gpu1060",
                    "machine_id": "machine-d",
                    "physical_os": "linux",
                    "runtime_os": "linux",
                    "runtime": "ssh",
                    "host": "10.0.0.11",
                    "ssh_user": "shardgrid",
                },
            ],
        }
    )


def _wsl_config(tmp_path: Path) -> ClusterConfig:
    data = _config(tmp_path).to_dict()
    for worker in data["workers"]:
        worker["physical_os"] = "windows"
        worker["runtime_os"] = "wsl2_linux"
        worker["runtime"] = "wsl2"
        worker["runtime_distro"] = "Ubuntu-22.04"
        worker["conda_environment"] = "shardgrid"
        worker["conda_prefix"] = "/home/shardgrid/miniconda3/envs/shardgrid"
    return ClusterConfig.from_dict(data)


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


class _FakeSSH:
    def __init__(self, profile: str = r"C:\Users\shardgrid") -> None:
        self.profile = profile
        self.commands: list[str] = []

    def run(self, command, *, timeout=None, secrets=(), check=False, stdin=None):
        del timeout, secrets, check, stdin
        self.commands.append(str(command))
        stdout = self.profile if "%USERPROFILE%" in str(command) else ""
        return _process(stdout=stdout)


class _FakeRuntime:
    def __init__(
        self,
        wsl_root: Path,
        windows_root: Path,
        *,
        fail_stage: bool = False,
    ) -> None:
        self.wsl_root = wsl_root
        self.windows_root = windows_root
        self.fail_stage = fail_stage
        self.scripts: list[str] = []

    def run_script(self, script: str, *, timeout=None, secrets=()):
        del timeout, secrets
        self.scripts.append(script)
        if "path.unlink()" in script or "shutil.rmtree" in script:
            return _process()
        if self.fail_stage:
            return _process(stderr="copy failed", exit_code=12)
        artifacts = _script_artifacts(script)
        if artifacts is not None:
            staging_root = _script_staging_root(script)
            payload = {"artifacts": []}
            failed = False
            for artifact in artifacts:
                source = self.wsl_root / artifact["source"].removeprefix("/")
                destination = self.windows_root / staging_root / artifact["relative_path"]
                item = {
                    "relative_path": artifact["relative_path"],
                    "source": artifact["source"],
                    "destination": str(destination),
                    "status": "missing",
                    "size_bytes": None,
                    "checksum": None,
                }
                if not source.exists():
                    payload["artifacts"].append(item)
                    if not artifact.get("optional", False):
                        failed = True
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
                item["status"] = "staged"
                item["size_bytes"] = destination.stat().st_size
                payload["artifacts"].append(item)
            return _process(stdout=json.dumps(payload), exit_code=10 if failed else 0)
        payload = _script_paths(script)
        source = self.wsl_root / payload["source"].removeprefix("/")
        destination = self.windows_root / payload["destination"].removeprefix(
            "/mnt/c/Users/shardgrid/"
        )
        if not source.exists():
            return _process(
                stdout=json.dumps({"exists": False, "source": payload["source"]}),
                exit_code=10,
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        return _process(
            stdout=json.dumps(
                {
                    "exists": True,
                    "source": payload["source"],
                    "destination": payload["destination"],
                    "size_bytes": destination.stat().st_size,
                }
            )
        )


class _FakeWindowsStagingTransport:
    name = ArtifactTransportName.SCP

    def __init__(self, windows_root: Path, *, fail: bool = False) -> None:
        self.windows_root = windows_root
        self.fail = fail
        self.calls: list[str] = []

    def transfer(self, items, *, remote: RemoteArtifactLocation, secrets=()):
        del remote, secrets
        results = []
        for item in items:
            self.calls.append(item.source)
            if item.source.startswith("/var/tmp/"):
                raise AssertionError("Windows SCP tried to read a WSL /var/tmp path directly")
            if self.fail:
                results.append(
                    ArtifactTransferItemResult(
                        label=item.label,
                        transport=self.name.value,
                        status=ArtifactTransferStatus.FAILED,
                        source=item.source,
                        destination=item.destination,
                        recorded_command=f"scp host:{item.source} {item.destination}",
                        exit_code=1,
                        stderr="scp failed",
                    )
                )
                continue
            source = self.windows_root / item.source
            destination = Path(item.destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if item.recursive:
                shutil.copytree(source, destination)
            else:
                shutil.copyfile(source, destination)
            results.append(
                ArtifactTransferItemResult(
                    label=item.label,
                    transport=self.name.value,
                    status=ArtifactTransferStatus.SUCCESS,
                    source=item.source,
                    destination=item.destination,
                    recorded_command=f"scp host:{item.source} {item.destination}",
                    exit_code=0,
                )
            )
        status = (
            ArtifactTransferStatus.FAILED
            if any(item.status is ArtifactTransferStatus.FAILED for item in results)
            else ArtifactTransferStatus.SUCCESS
        )
        return ArtifactTransferResult(transport=self.name.value, status=status, items=results)


class _DestinationFailureTransport:
    name = ArtifactTransportName.SCP

    def transfer(self, items, *, remote: RemoteArtifactLocation, secrets=()):
        del remote, secrets
        item = tuple(items)[0]
        return ArtifactTransferResult(
            transport=self.name.value,
            status=ArtifactTransferStatus.FAILED,
            items=[
                ArtifactTransferItemResult(
                    label=item.label,
                    transport=self.name.value,
                    status=ArtifactTransferStatus.FAILED,
                    source=item.source,
                    destination=item.destination,
                    recorded_command="scp local destination preparation",
                    stderr="local destination preparation failed: permission denied",
                )
            ],
        )


def _process(
    *,
    stdout: str = "",
    stderr: str = "",
    exit_code: int = 0,
    timed_out: bool = False,
):
    from shardgrid.common.process import ProcessResult

    return ProcessResult(
        args=(),
        recorded_command="fake",
        shell=False,
        cwd=None,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        timed_out=timed_out,
        runtime_environment={},
    )


def _script_paths(script: str) -> dict[str, str]:
    import ast
    import re

    values = re.findall(r"Path\(([^\n]+?)\)", script)
    return {"source": ast.literal_eval(values[0]), "destination": ast.literal_eval(values[1])}


def _script_artifacts(script: str):
    import ast
    import re

    match = re.search(r"artifacts = ([^\n]+)", script)
    if match is None:
        return None
    return json.loads(ast.literal_eval(match.group(1)))


def _script_staging_root(script: str) -> str:
    import ast
    import re

    match = re.search(r"staging_root = Path\(([^\n]+?)\)", script)
    assert match is not None
    return ast.literal_eval(match.group(1)).removeprefix("/mnt/c/Users/shardgrid/")


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


def test_collects_runtime_log_and_checkpoint_from_assignment_references(tmp_path: Path) -> None:
    config, snapshot = _snapshot(tmp_path)
    remote_root = tmp_path / "remote-runtime"
    assignment = WorkerAssignment(
        worker_id="gpu4060",
        rank=0,
        stage="stage0",
        log_path=str(remote_root / "logs" / "gpu4060" / "rank0-stage0" / "combined.log"),
    )
    _write(
        remote_root / "logs" / "gpu4060" / "rank0-stage0" / "combined.log",
        "combined-rank0\n",
    )
    _write(remote_root / "checkpoint" / "checkpoint_rank0.pt", "checkpoint-rank0")

    source = WorkerArtifactSource.from_worker_assignment(
        worker=next(item for item in config.workers if str(item.worker_id) == "gpu4060"),
        assignment=assignment,
        remote_root=str(remote_root),
        checkpoint_paths=("checkpoint/checkpoint_rank0.pt",),
    )
    result = ArtifactCollector(transport=_FakePullTransport()).collect(snapshot, sources=[source])

    assert result.status is CollectionStatus.SUCCESS
    assert result.workers[0].checkpoint_state is ArtifactCollectionState.COMPLETE
    assert (
        Path(snapshot.logs_path, "gpu4060", "rank0-stage0", "combined.log").read_text()
        == "combined-rank0\n"
    )
    assert (
        Path(snapshot.checkpoint_path, "files", "gpu4060", "rank0-stage0", "checkpoint_rank0.pt")
        .read_text()
        == "checkpoint-rank0"
    )


def test_windows_wsl_pull_stages_artifact_before_scp(tmp_path: Path) -> None:
    config = _wsl_config(tmp_path)
    _, snapshot = _snapshot(tmp_path)
    wsl_root = tmp_path / "wsl"
    windows_root = tmp_path / "windows"
    remote_root = Path("/var/tmp/shardgrid/jobs/job-wsl")
    _write(
        wsl_root
        / "var/tmp/shardgrid/jobs/job-wsl/logs/gpu4060/rank0-stage0/combined.log",
        "combined\n",
    )
    _write(
        wsl_root / "var/tmp/shardgrid/jobs/job-wsl/checkpoint/checkpoint_rank0.pt",
        "checkpoint\n",
    )
    source = WorkerArtifactSource.from_worker_assignment(
        worker=next(item for item in config.workers if str(item.worker_id) == "gpu4060"),
        assignment=WorkerAssignment(
            worker_id="gpu4060",
            rank=0,
            stage="stage0",
            log_path="/var/tmp/shardgrid/jobs/job-wsl/logs/gpu4060/rank0-stage0/combined.log",
        ),
        remote_root=str(remote_root),
        checkpoint_paths=("checkpoint/checkpoint_rank0.pt",),
    )
    transport = _FakeWindowsStagingTransport(windows_root)

    result = ArtifactCollector(
        transport=transport,
        ssh_factory=lambda _: _FakeSSH(),
        runtime_factory=lambda _, __: _FakeRuntime(wsl_root, windows_root),
    ).collect(snapshot, sources=[source])

    assert result.status is CollectionStatus.SUCCESS
    assert all(not call.startswith("/var/tmp/") for call in transport.calls)
    assert transport.calls == [".shardgrid/artifacts/job-089/gpu4060/rank0-stage0"]
    checkpoint = result.workers[0].artifacts[1]
    assert checkpoint.wsl_source == "/var/tmp/shardgrid/jobs/job-wsl/checkpoint/checkpoint_rank0.pt"
    assert checkpoint.windows_staging_path is not None
    assert checkpoint.control_destination is not None
    assert checkpoint.bytes_received == len("checkpoint\n")


def test_windows_wsl_worker_batches_staging_and_scp_for_multiple_artifacts(
    tmp_path: Path,
) -> None:
    config = _wsl_config(tmp_path)
    _, snapshot = _snapshot(tmp_path)
    wsl_root = tmp_path / "wsl"
    windows_root = tmp_path / "windows"
    remote_root = Path("/var/tmp/shardgrid/jobs/job-wsl")
    _write(
        wsl_root / "var/tmp/shardgrid/jobs/job-wsl/logs/gpu4060/rank0-stage0/combined.log",
        "combined\n",
    )
    _write(
        wsl_root / "var/tmp/shardgrid/jobs/job-wsl/checkpoint/checkpoint_rank0.pt",
        "checkpoint-0\n",
    )
    _write(
        wsl_root / "var/tmp/shardgrid/jobs/job-wsl/checkpoint/extra_rank0.pt",
        "checkpoint-extra\n",
    )
    runtime = _FakeRuntime(wsl_root, windows_root)
    transport = _FakeWindowsStagingTransport(windows_root)
    source = WorkerArtifactSource.from_worker_assignment(
        worker=next(item for item in config.workers if str(item.worker_id) == "gpu4060"),
        assignment=WorkerAssignment(
            worker_id="gpu4060",
            rank=0,
            stage="stage0",
            log_path="/var/tmp/shardgrid/jobs/job-wsl/logs/gpu4060/rank0-stage0/combined.log",
        ),
        remote_root=str(remote_root),
        checkpoint_paths=("checkpoint/checkpoint_rank0.pt", "checkpoint/extra_rank0.pt"),
    )

    result = ArtifactCollector(
        transport=transport,
        ssh_factory=lambda _: _FakeSSH(),
        runtime_factory=lambda _, __: runtime,
    ).collect(snapshot, sources=[source])

    assert result.status is CollectionStatus.SUCCESS
    assert len([script for script in runtime.scripts if "artifacts =" in script]) == 1
    assert transport.calls == [".shardgrid/artifacts/job-089/gpu4060/rank0-stage0"]
    assert (
        Path(snapshot.checkpoint_path, "files", "gpu4060", "rank0-stage0", "extra_rank0.pt")
        .read_text()
        == "checkpoint-extra\n"
    )


def test_windows_wsl_source_missing_does_not_run_scp(tmp_path: Path) -> None:
    config = _wsl_config(tmp_path)
    _, snapshot = _snapshot(tmp_path)
    windows_root = tmp_path / "windows"
    source = WorkerArtifactSource.from_worker_assignment(
        worker=next(item for item in config.workers if str(item.worker_id) == "gpu4060"),
        assignment=WorkerAssignment(worker_id="gpu4060", rank=0, stage="stage0"),
        remote_root="/var/tmp/shardgrid/jobs/job-wsl",
        checkpoint_paths=("checkpoint/missing.pt",),
    )
    transport = _FakeWindowsStagingTransport(windows_root)

    result = ArtifactCollector(
        transport=transport,
        ssh_factory=lambda _: _FakeSSH(),
        runtime_factory=lambda _, __: _FakeRuntime(tmp_path / "wsl", windows_root),
    ).collect(snapshot, sources=[source])

    artifact = result.workers[0].artifacts[0]
    assert transport.calls == []
    assert artifact.failure_class == "ARTIFACT_SOURCE_MISSING"
    assert artifact.failure is not None
    assert artifact.failure.message == "source artifact is missing in the WSL runtime"


def test_windows_wsl_stage_failure_does_not_run_scp(tmp_path: Path) -> None:
    config = _wsl_config(tmp_path)
    _, snapshot = _snapshot(tmp_path)
    source = WorkerArtifactSource.from_worker_assignment(
        worker=next(item for item in config.workers if str(item.worker_id) == "gpu4060"),
        assignment=WorkerAssignment(worker_id="gpu4060", rank=0, stage="stage0"),
        remote_root="/var/tmp/shardgrid/jobs/job-wsl",
        checkpoint_paths=("checkpoint/checkpoint_rank0.pt",),
    )
    transport = _FakeWindowsStagingTransport(tmp_path / "windows")

    result = ArtifactCollector(
        transport=transport,
        ssh_factory=lambda _: _FakeSSH(),
        runtime_factory=lambda _, __: _FakeRuntime(
            tmp_path / "wsl",
            tmp_path / "windows",
            fail_stage=True,
        ),
    ).collect(snapshot, sources=[source])

    artifact = result.workers[0].artifacts[0]
    assert transport.calls == []
    assert artifact.failure_class == "ARTIFACT_STAGE_FAILED"
    assert artifact.stderr_summary == "copy failed"


def test_windows_wsl_scp_failure_keeps_artifact_diagnostics(tmp_path: Path) -> None:
    config = _wsl_config(tmp_path)
    _, snapshot = _snapshot(tmp_path)
    wsl_root = tmp_path / "wsl"
    remote_root = "/var/tmp/shardgrid/jobs/job-wsl"
    _write(
        wsl_root / "var/tmp/shardgrid/jobs/job-wsl/checkpoint/checkpoint_rank0.pt",
        "checkpoint\n",
    )
    source = WorkerArtifactSource.from_worker_assignment(
        worker=next(item for item in config.workers if str(item.worker_id) == "gpu4060"),
        assignment=WorkerAssignment(worker_id="gpu4060", rank=0, stage="stage0"),
        remote_root=remote_root,
        checkpoint_paths=("checkpoint/checkpoint_rank0.pt",),
    )

    result = ArtifactCollector(
        transport=_FakeWindowsStagingTransport(tmp_path / "windows", fail=True),
        ssh_factory=lambda _: _FakeSSH(),
        runtime_factory=lambda _, __: _FakeRuntime(wsl_root, tmp_path / "windows"),
    ).collect(snapshot, sources=[source])

    artifact = result.workers[0].artifacts[0]
    assert artifact.failure_class == "ARTIFACT_SCP_FAILED"
    assert artifact.exit_code == 1
    assert artifact.stderr_summary == "scp failed"
    assert artifact.recorded_command is not None
    assert "host:.shardgrid/artifacts/" in artifact.recorded_command
    assert "host:/var/tmp/" not in artifact.recorded_command


def test_local_destination_preparation_failure_is_classified(tmp_path: Path) -> None:
    config, snapshot = _snapshot(tmp_path)
    worker = tmp_path / "remote-a"
    _write(worker / "checkpoint" / "model.pt", "checkpoint")

    result = ArtifactCollector(transport=_DestinationFailureTransport()).collect(
        snapshot,
        sources=[_source(config, "gpu4060", 0, "stage0", worker)],
        artifact_paths=("checkpoint/model.pt",),
    )

    artifact = result.workers[0].artifacts[0]
    assert artifact.failure_class == "ARTIFACT_DESTINATION_FAILED"
    assert artifact.failure is not None
    assert "local destination preparation failed" in artifact.failure.message
