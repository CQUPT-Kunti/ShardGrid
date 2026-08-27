"""Distribute immutable job snapshots to SSH/WSL workers."""

from __future__ import annotations

import hashlib
import json
import tarfile
import tempfile
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Sequence, cast

from shardgrid.artifacts.transport import (
    ArtifactTransferSpec,
    ArtifactTransferStatus,
    RemoteArtifactLocation,
    build_transport_config,
    select_artifact_transport,
    serialize_transfer_result,
)
from shardgrid.common.config import ClusterConfig, WorkerConfig
from shardgrid.common.enums import FailureStage, PhysicalOS, SerializableStrEnum
from shardgrid.common.errors import make_failure_record
from shardgrid.common.logging import redact_mapping
from shardgrid.jobs.models import FailureRecord, JobSnapshot
from shardgrid.transport.runtime import WSLRuntimeConfig, WSLRuntimeWrapper
from shardgrid.transport.ssh import SSHOptions, SSHTransport

_REMOTE_METADATA_PATH = "diagnostics/snapshot-metadata.json"
_REMOTE_REQUIRED_RELATIVE_PATHS = (
    "code/.shardgrid-code-snapshot.json",
    "config/training-config.json",
    "plan/original-parallel-plan.json",
    "plan/execution-plan.json",
    "diagnostics/network-state.json",
    "diagnostics/job-status.json",
    "diagnostics/snapshot-metadata.json",
    "checkpoint/checkpoint-metadata.json",
)


class DistributionStatus(SerializableStrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True)
class RemoteSnapshotProbe:
    exists: bool
    empty: bool
    checksum: str | None = None
    job_id: str | None = None
    missing_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class WorkerDistributionResult:
    worker_id: str
    host: str
    transport: str
    status: DistributionStatus
    remote_snapshot_root: str
    control_checksum: str
    remote_checksum: str | None = None
    remote_job_id: str | None = None
    metadata_ready: bool = False
    skipped: bool = False
    transfer_result: dict[str, Any] | None = None
    failure: FailureRecord | None = None

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


@dataclass(frozen=True)
class SnapshotDistributionResult:
    job_id: str
    control_checksum: str
    status: DistributionStatus
    workers: list[WorkerDistributionResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


def distribute_job_snapshot(
    snapshot: JobSnapshot,
    *,
    cluster_config: ClusterConfig,
    workers: Sequence[WorkerConfig],
    secrets: Sequence[str] = (),
) -> SnapshotDistributionResult:
    control_checksum = snapshot_checksum(Path(snapshot.root_path))
    preferred_transport = _load_snapshot_transport_preference(snapshot)
    results: list[WorkerDistributionResult] = []
    with tempfile.TemporaryDirectory(prefix=f"shardgrid-{snapshot.job_id}-") as tmp_dir:
        archive_path = _build_snapshot_archive(snapshot, Path(tmp_dir))
        for worker in workers:
            results.append(
                distribute_job_snapshot_to_worker(
                    snapshot,
                    archive_path=archive_path,
                    control_checksum=control_checksum,
                    cluster_config=cluster_config,
                    worker=worker,
                    preferred_transport=preferred_transport,
                    secrets=secrets,
                )
            )
    return SnapshotDistributionResult(
        job_id=str(snapshot.job_id),
        control_checksum=control_checksum,
        status=_overall_status(results),
        workers=results,
    )


def distribute_job_snapshot_to_worker(
    snapshot: JobSnapshot,
    *,
    archive_path: Path,
    control_checksum: str,
    cluster_config: ClusterConfig,
    worker: WorkerConfig,
    preferred_transport: str = "auto",
    secrets: Sequence[str] = (),
) -> WorkerDistributionResult:
    try:
        transport = _select_worker_transport(
            cluster_config=cluster_config,
            worker=worker,
            preferred_transport=preferred_transport,
        )
        ssh = _build_ssh_transport(cluster_config, worker)
        runtime = WSLRuntimeWrapper(
            WSLRuntimeConfig.from_worker_and_runtime(worker, cluster_config.runtime),
            ssh,
        )
        remote_root = _remote_snapshot_root(cluster_config, snapshot)
        before = _probe_remote_snapshot(runtime, remote_root)
    except Exception as exc:
        return WorkerDistributionResult(
            worker_id=str(worker.worker_id),
            host=str(worker.host),
            transport=preferred_transport,
            status=DistributionStatus.BLOCKED,
            remote_snapshot_root=_remote_snapshot_root(cluster_config, snapshot),
            control_checksum=control_checksum,
            failure=make_failure_record(
                stage=FailureStage.DISTRIBUTE,
                host=str(worker.host),
                worker_id=str(worker.worker_id),
                message=f"remote snapshot preflight failed: {exc}",
                recommended_action="repair SSH/WSL runtime access on the Worker, then retry",
            ),
        )
    if before.exists and not before.empty:
        if (
            before.checksum == control_checksum
            and before.job_id == str(snapshot.job_id)
            and not before.missing_paths
        ):
            return WorkerDistributionResult(
                worker_id=str(worker.worker_id),
                host=str(worker.host),
                transport=transport.name.value,
                status=DistributionStatus.PASS,
                remote_snapshot_root=remote_root,
                control_checksum=control_checksum,
                remote_checksum=before.checksum,
                remote_job_id=before.job_id,
                metadata_ready=True,
                skipped=True,
            )
        return WorkerDistributionResult(
            worker_id=str(worker.worker_id),
            host=str(worker.host),
            transport=transport.name.value,
            status=DistributionStatus.FAIL,
            remote_snapshot_root=remote_root,
            control_checksum=control_checksum,
            remote_checksum=before.checksum,
            remote_job_id=before.job_id,
            failure=make_failure_record(
                stage=FailureStage.DISTRIBUTE,
                host=str(worker.host),
                worker_id=str(worker.worker_id),
                message="remote snapshot already exists with a different immutable identity",
                recommended_action=(
                    "remove the conflicting remote snapshot or choose a new job_id, then retry"
                ),
            ),
        )

    windows_profile = _read_windows_userprofile(ssh)
    staging_dir = str(PureWindowsPath(".shardgrid") / "snapshots" / str(snapshot.job_id))
    _prepare_windows_staging_dir(ssh, staging_dir)
    archive_name = f"{snapshot.job_id}.tar.gz"
    destination = str(
        PurePosixPath(".shardgrid") / "snapshots" / str(snapshot.job_id) / archive_name
    )
    transport_result = transport.transfer(
        [
            ArtifactTransferSpec(
                label="snapshot",
                source=str(archive_path),
                destination=destination,
                direction="push",
                local_root=str(archive_path.parent),
            )
        ],
        remote=RemoteArtifactLocation(
            host=str(worker.host),
            user=worker.ssh_user,
            port=worker.ssh_port,
            path=".",
            private_key_path=cluster_config.ssh.private_key_path,
        ),
        secrets=secrets,
    )
    item = transport_result.items[0]
    if item.status is not ArtifactTransferStatus.SUCCESS:
        return WorkerDistributionResult(
            worker_id=str(worker.worker_id),
            host=str(worker.host),
            transport=transport.name.value,
            status=(
                DistributionStatus.BLOCKED
                if _is_auth_or_permission_problem(item.stderr)
                else DistributionStatus.FAIL
            ),
            remote_snapshot_root=remote_root,
            control_checksum=control_checksum,
            transfer_result=serialize_transfer_result(transport_result),
            failure=make_failure_record(
                stage=FailureStage.DISTRIBUTE,
                host=str(worker.host),
                worker_id=str(worker.worker_id),
                command=item.recorded_command,
                exit_code=item.exit_code,
                message="artifact transport failed before the remote snapshot was ready",
                recommended_action=(
                    "inspect transport stderr and remote write permissions, then retry"
                ),
                retryable=item.retryable,
                secrets=secrets,
            ),
        )

    windows_archive = (
        PureWindowsPath(windows_profile)
        / ".shardgrid"
        / "snapshots"
        / str(snapshot.job_id)
        / archive_name
    )
    archive_wsl_path = _windows_to_wsl_path(str(windows_archive))
    unpack_result = _extract_remote_snapshot(runtime, archive_wsl_path, remote_root)
    if not unpack_result.ok:
        return WorkerDistributionResult(
            worker_id=str(worker.worker_id),
            host=str(worker.host),
            transport=transport.name.value,
            status=DistributionStatus.FAIL,
            remote_snapshot_root=remote_root,
            control_checksum=control_checksum,
            transfer_result=serialize_transfer_result(transport_result),
            failure=runtime.classify_runtime_failure(
                unpack_result,
                host=str(worker.host),
                worker_id=str(worker.worker_id),
                conda_environment=worker.conda_environment,
                conda_prefix=worker.conda_prefix,
            ),
        )

    after = _probe_remote_snapshot(runtime, remote_root)
    if (
        after.checksum != control_checksum
        or after.job_id != str(snapshot.job_id)
        or after.missing_paths
    ):
        return WorkerDistributionResult(
            worker_id=str(worker.worker_id),
            host=str(worker.host),
            transport=transport.name.value,
            status=DistributionStatus.FAIL,
            remote_snapshot_root=remote_root,
            control_checksum=control_checksum,
            remote_checksum=after.checksum,
            remote_job_id=after.job_id,
            transfer_result=serialize_transfer_result(transport_result),
            failure=make_failure_record(
                stage=FailureStage.DISTRIBUTE,
                host=str(worker.host),
                worker_id=str(worker.worker_id),
                message="remote snapshot verification failed after distribution",
                recommended_action=(
                    "compare control and remote snapshot contents, then retry the distribution"
                ),
            ),
        )

    return WorkerDistributionResult(
        worker_id=str(worker.worker_id),
        host=str(worker.host),
        transport=transport.name.value,
        status=DistributionStatus.PASS,
        remote_snapshot_root=remote_root,
        control_checksum=control_checksum,
        remote_checksum=after.checksum,
        remote_job_id=after.job_id,
        metadata_ready=True,
        transfer_result=serialize_transfer_result(transport_result),
    )


def snapshot_checksum(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _build_snapshot_archive(snapshot: JobSnapshot, directory: Path) -> Path:
    archive_path = directory / f"{snapshot.job_id}.tar.gz"
    root = Path(snapshot.root_path)
    with tarfile.open(archive_path, "w:gz") as handle:
        for path in sorted(root.rglob("*")):
            handle.add(path, arcname=path.relative_to(root))
    return archive_path


def _load_snapshot_transport_preference(snapshot: JobSnapshot) -> str:
    config_path = Path(snapshot.config_path) / "training-config.json"
    if not config_path.exists():
        return "auto"
    payload = json.loads(config_path.read_text())
    artifacts = payload.get("artifacts")
    if isinstance(artifacts, dict) and isinstance(artifacts.get("transport"), str):
        return str(artifacts["transport"])
    return "auto"


def _remote_snapshot_root(cluster_config: ClusterConfig, snapshot: JobSnapshot) -> str:
    return str(PurePosixPath(str(cluster_config.jobs_root), str(snapshot.job_id)))


def _select_worker_transport(
    *,
    cluster_config: ClusterConfig,
    worker: WorkerConfig,
    preferred_transport: str,
):
    if preferred_transport == "auto" and worker.physical_os is PhysicalOS.WINDOWS:
        for name in ("scp", "sftp", "rsync"):
            try:
                return select_artifact_transport(
                    build_transport_config(name),
                )
            except ValueError:
                continue
        raise ValueError("no supported artifact transport executable is available")
    return select_artifact_transport(build_transport_config(preferred_transport))


def _build_ssh_transport(cluster_config: ClusterConfig, worker: WorkerConfig) -> SSHTransport:
    return SSHTransport(
        SSHOptions.from_ssh_config(
            cluster_config.ssh,
            host=str(worker.host),
            user=worker.ssh_user,
            port=worker.ssh_port,
        )
    )


def _read_windows_userprofile(ssh: SSHTransport) -> str:
    result = ssh.run("cmd /c echo %USERPROFILE%")
    if not result.ok:
        raise RuntimeError(result.stderr or "failed to query remote user profile")
    profile = result.stdout.strip().strip('"')
    if not profile:
        raise RuntimeError("remote user profile path is empty")
    return profile


def _prepare_windows_staging_dir(ssh: SSHTransport, staging_dir: str) -> None:
    quoted = staging_dir.replace('"', "")
    result = ssh.run(f'cmd /c if not exist "{quoted}" mkdir "{quoted}"')
    if not result.ok:
        raise RuntimeError(result.stderr or "failed to prepare remote staging directory")


def _windows_to_wsl_path(path: str) -> str:
    windows_path = PureWindowsPath(path)
    drive = windows_path.drive.rstrip(":")
    if not drive:
        raise ValueError("windows path must include a drive letter")
    parts = ["/mnt", drive.lower()]
    for part in windows_path.parts[1:]:
        if part in {"\\", "/"}:
            continue
        parts.append(part)
    return str(PurePosixPath(*parts))


def _extract_remote_snapshot(
    runtime: WSLRuntimeWrapper, archive_wsl_path: str, remote_root: str
):
    command = (
        f"mkdir -p '{remote_root}' "
        f"&& tar -xzf '{archive_wsl_path}' -C '{remote_root}'"
    )
    return runtime.run(command)


def _probe_remote_snapshot(runtime: WSLRuntimeWrapper, remote_root: str) -> RemoteSnapshotProbe:
    script = """
import hashlib
import json
from pathlib import Path

root = Path(%(root)r)
required = %(required)r
payload = {
    "exists": root.exists(),
    "empty": False,
    "checksum": None,
    "job_id": None,
    "missing_paths": [],
}
if root.exists():
    if not root.is_dir():
        raise SystemExit("remote snapshot root is not a directory")
    payload["empty"] = not any(root.iterdir())
    files = sorted(candidate for candidate in root.rglob("*") if candidate.is_file())
    if files:
        payload["missing_paths"] = [path for path in required if not (root / path).exists()]
        if not payload["missing_paths"]:
            digest = hashlib.sha256()
            for path in files:
                digest.update(path.relative_to(root).as_posix().encode())
                digest.update(b"\\0")
                digest.update(path.read_bytes())
                digest.update(b"\\0")
            payload["checksum"] = digest.hexdigest()
            metadata = json.loads((root / "diagnostics/snapshot-metadata.json").read_text())
            payload["job_id"] = metadata["job_id"]
print(json.dumps(payload, sort_keys=True))
""" % {"root": remote_root, "required": list(_REMOTE_REQUIRED_RELATIVE_PATHS)}
    result = runtime.run_script(script)
    if not result.ok:
        raise RuntimeError(result.stderr or "failed to probe remote snapshot")
    payload = json.loads(result.stdout)
    return RemoteSnapshotProbe(
        exists=bool(payload["exists"]),
        empty=bool(payload["empty"]),
        checksum=payload.get("checksum"),
        job_id=payload.get("job_id"),
        missing_paths=tuple(str(item) for item in payload.get("missing_paths", [])),
    )


def _overall_status(results: Sequence[WorkerDistributionResult]) -> DistributionStatus:
    if any(result.status is DistributionStatus.FAIL for result in results):
        return DistributionStatus.FAIL
    if any(result.status is DistributionStatus.BLOCKED for result in results):
        return DistributionStatus.BLOCKED
    return DistributionStatus.PASS


def _is_auth_or_permission_problem(stderr: str) -> bool:
    lowered = stderr.lower()
    return "permission denied" in lowered or "authentication" in lowered


def _serialize(value: Any) -> Any:
    if hasattr(value, "value"):
        return value.value
    if is_dataclass(value):
        return {key: _serialize(item) for key, item in asdict(cast(Any, value)).items()}
    if isinstance(value, dict):
        return {str(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(item) for item in value]
    return value


def serialize_distribution_result(
    result: SnapshotDistributionResult, *, secrets: Sequence[str] = ()
) -> dict[str, Any]:
    return redact_mapping(result.to_dict(), secrets)
