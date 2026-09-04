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
    ArtifactTransport,
    RemoteArtifactLocation,
    build_transport_config,
    select_artifact_transport,
    serialize_transfer_result,
)
from shardgrid.common.config import ClusterConfig, WorkerConfig
from shardgrid.common.enums import FailureStage, PhysicalOS, SerializableStrEnum
from shardgrid.common.errors import make_failure_record
from shardgrid.common.logging import redact_mapping
from shardgrid.common.process import redact_text
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
    "job-status.json",
    "diagnostics/snapshot-metadata.json",
    "checkpoint/checkpoint-metadata.json",
)
_PREPARE_ONLY_TOP_LEVEL_DIRS = frozenset({"logs", "diagnostics", "checkpoint"})
_CHECKSUM_EXCLUDED_TOP_LEVEL_DIRS = frozenset({"logs", "diagnostics", "checkpoint"})


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
    top_level_entries: tuple[str, ...] = ()
    metadata_ready: bool = False
    parse_error: str | None = None
    command_summary: str | None = None
    exit_code: int | None = None
    stdout_summary: str | None = None
    stderr_summary: str | None = None


@dataclass(frozen=True)
class RemoteSnapshotProbeError(RuntimeError):
    message: str
    diagnostics: dict[str, Any]

    def __str__(self) -> str:
        return self.message


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
    details: dict[str, Any] | None = None

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
    transport: ArtifactTransport | None = None,
    ssh: SSHTransport | None = None,
    runtime: WSLRuntimeWrapper | None = None,
) -> WorkerDistributionResult:
    remote_root = _remote_snapshot_root(cluster_config, snapshot)
    try:
        transport = transport or _select_worker_transport(
            cluster_config=cluster_config,
            worker=worker,
            preferred_transport=preferred_transport,
        )
        ssh = ssh or _build_ssh_transport(cluster_config, worker)
        runtime = runtime or WSLRuntimeWrapper(
            WSLRuntimeConfig.from_worker_and_runtime(worker, cluster_config.runtime),
            ssh,
        )
        before = _probe_remote_snapshot(runtime, remote_root)
    except RemoteSnapshotProbeError as exc:
        details = redact_mapping(exc.diagnostics, secrets)
        return WorkerDistributionResult(
            worker_id=str(worker.worker_id),
            host=str(worker.host),
            transport=preferred_transport,
            status=DistributionStatus.BLOCKED,
            remote_snapshot_root=remote_root,
            control_checksum=control_checksum,
            failure=make_failure_record(
                stage=FailureStage.DISTRIBUTE,
                host=str(worker.host),
                worker_id=str(worker.worker_id),
                message=f"remote snapshot preflight failed: {exc}",
                recommended_action=str(
                    details.get("recommended_action")
                    or "repair SSH/WSL runtime access on the Worker, then retry"
                ),
                command=details.get("command_summary"),
                exit_code=cast(int | None, details.get("exit_code")),
            ),
            details=details,
        )
    except Exception as exc:
        return WorkerDistributionResult(
            worker_id=str(worker.worker_id),
            host=str(worker.host),
            transport=preferred_transport,
            status=DistributionStatus.BLOCKED,
            remote_snapshot_root=remote_root,
            control_checksum=control_checksum,
            failure=make_failure_record(
                stage=FailureStage.DISTRIBUTE,
                host=str(worker.host),
                worker_id=str(worker.worker_id),
                message=f"remote snapshot preflight failed: {exc}",
                recommended_action="repair SSH/WSL runtime access on the Worker, then retry",
            ),
            details=_distribution_details(
                worker=worker,
                remote_root=remote_root,
                substep="preflight_exception",
                expected_job_id=str(snapshot.job_id),
                expected_checksum=control_checksum,
                recommended_action="repair SSH/WSL runtime access on the Worker, then retry",
                command_summary="remote snapshot preflight",
                parse_error=f"{type(exc).__name__}: {exc}",
                metadata_ready=False,
                secrets=secrets,
            ),
        )
    if before.exists and not before.empty:
        if (
            before.checksum == control_checksum
            and before.job_id == str(snapshot.job_id)
            and not before.missing_paths
            and before.parse_error is None
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
                details=_probe_details(
                    worker=worker,
                    remote_root=remote_root,
                    substep="preflight_ready",
                    probe=before,
                    expected_job_id=str(snapshot.job_id),
                    expected_checksum=control_checksum,
                    recommended_action="none",
                    secrets=secrets,
                ),
            )
        if _is_prepare_only_layout(before):
            pass
        else:
            recommended_action = (
                "remote snapshot identity is not fresh; use a new unique job_id or inspect "
                "the stale remote root before retrying"
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
                metadata_ready=before.metadata_ready,
                failure=make_failure_record(
                    stage=FailureStage.DISTRIBUTE,
                    host=str(worker.host),
                    worker_id=str(worker.worker_id),
                    message="remote snapshot already exists with a different immutable identity",
                    recommended_action=recommended_action,
                ),
                details=_probe_details(
                    worker=worker,
                    remote_root=remote_root,
                    substep="preflight_identity_conflict",
                    probe=before,
                    expected_job_id=str(snapshot.job_id),
                    expected_checksum=control_checksum,
                    recommended_action=recommended_action,
                    secrets=secrets,
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
            connect_timeout_seconds=cluster_config.ssh.connect_timeout_seconds,
            command_timeout_seconds=float(cluster_config.ssh.command_timeout_seconds),
            known_host_policy=(
                "yes" if cluster_config.ssh.strict_host_key_checking else "accept-new"
            ),
            known_hosts_path=cluster_config.ssh.known_hosts_path,
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
            details=_distribution_details(
                worker=worker,
                remote_root=remote_root,
                substep="transfer_archive",
                expected_job_id=str(snapshot.job_id),
                expected_checksum=control_checksum,
                recommended_action=(
                    "inspect transport stderr and remote write permissions, then retry"
                ),
                command_summary=item.recorded_command,
                exit_code=item.exit_code,
                stderr_summary=item.stderr,
                metadata_ready=False,
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
                stage=FailureStage.DISTRIBUTE,
                host=str(worker.host),
                worker_id=str(worker.worker_id),
                conda_environment=worker.conda_environment,
                conda_prefix=worker.conda_prefix,
            ),
            details=_distribution_details(
                worker=worker,
                remote_root=remote_root,
                substep="materialize_archive",
                expected_job_id=str(snapshot.job_id),
                expected_checksum=control_checksum,
                recommended_action=(
                    "inspect remote extract stderr and remote snapshot path permissions, then retry"
                ),
                command_summary=unpack_result.recorded_command,
                exit_code=unpack_result.exit_code,
                stdout_summary=unpack_result.stdout,
                stderr_summary=unpack_result.stderr,
                metadata_ready=False,
                secrets=secrets,
            ),
        )

    after = _probe_remote_snapshot(runtime, remote_root)
    if (
        after.checksum != control_checksum
        or after.job_id != str(snapshot.job_id)
        or after.missing_paths
        or after.parse_error is not None
    ):
        recommended_action = (
            "compare control and remote snapshot contents, inspect the first failed "
            "verification substep, then retry the distribution"
        )
        return WorkerDistributionResult(
            worker_id=str(worker.worker_id),
            host=str(worker.host),
            transport=transport.name.value,
            status=DistributionStatus.FAIL,
            remote_snapshot_root=remote_root,
            control_checksum=control_checksum,
            remote_checksum=after.checksum,
            remote_job_id=after.job_id,
            metadata_ready=after.metadata_ready,
            transfer_result=serialize_transfer_result(transport_result),
            failure=make_failure_record(
                stage=FailureStage.DISTRIBUTE,
                host=str(worker.host),
                worker_id=str(worker.worker_id),
                message="remote snapshot verification failed after distribution",
                recommended_action=recommended_action,
            ),
            details=_probe_details(
                worker=worker,
                remote_root=remote_root,
                substep="verify_post_transfer",
                probe=after,
                expected_job_id=str(snapshot.job_id),
                expected_checksum=control_checksum,
                recommended_action=recommended_action,
                secrets=secrets,
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
        details=_probe_details(
            worker=worker,
            remote_root=remote_root,
            substep="verify_success",
            probe=after,
            expected_job_id=str(snapshot.job_id),
            expected_checksum=control_checksum,
            recommended_action="none",
            secrets=secrets,
        ),
    )


def snapshot_checksum(root: Path) -> str:
    digest = hashlib.sha256()
    for path in _snapshot_checksum_files(root):
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
excluded = %(excluded)r
payload = {
    "exists": root.exists(),
    "empty": False,
    "checksum": None,
    "job_id": None,
    "missing_paths": [],
    "top_level_entries": [],
    "metadata_ready": False,
    "parse_error": None,
}
if root.exists():
    if not root.is_dir():
        payload["parse_error"] = "remote snapshot root is not a directory"
    else:
        payload["empty"] = not any(root.iterdir())
        payload["top_level_entries"] = sorted(path.name for path in root.iterdir())
        files = sorted(
            candidate
            for candidate in root.rglob("*")
            if candidate.is_file()
            and candidate.relative_to(root).parts[0] not in excluded
            and "__pycache__" not in candidate.relative_to(root).parts
            and candidate.suffix != ".pyc"
        )
        if files:
            digest = hashlib.sha256()
            for path in files:
                digest.update(path.relative_to(root).as_posix().encode())
                digest.update(b"\\0")
                digest.update(path.read_bytes())
                digest.update(b"\\0")
            payload["checksum"] = digest.hexdigest()
            payload["missing_paths"] = [path for path in required if not (root / path).exists()]
            metadata_path = root / "diagnostics/snapshot-metadata.json"
            if metadata_path.exists():
                try:
                    metadata = json.loads(metadata_path.read_text())
                    job_id = metadata.get("job_id")
                    if isinstance(job_id, str) and job_id:
                        payload["job_id"] = job_id
                    else:
                        payload["parse_error"] = "snapshot metadata missing job_id"
                except Exception as exc:
                    payload["parse_error"] = f"{type(exc).__name__}: {exc}"
            if (
                not payload["missing_paths"]
                and payload["job_id"]
                and payload["parse_error"] is None
            ):
                payload["metadata_ready"] = True
print(json.dumps(payload, sort_keys=True))
""" % {
        "root": remote_root,
        "required": list(_REMOTE_REQUIRED_RELATIVE_PATHS),
        "excluded": sorted(_CHECKSUM_EXCLUDED_TOP_LEVEL_DIRS),
    }
    result = runtime.run_script(script)
    if not result.ok:
        raise RemoteSnapshotProbeError(
            "failed to execute remote snapshot probe",
            _distribution_details(
                remote_root=remote_root,
                substep="probe_command",
                expected_job_id=None,
                expected_checksum=None,
                recommended_action="inspect SSH/WSL runtime command execution and retry",
                command_summary=result.recorded_command,
                exit_code=result.exit_code,
                stdout_summary=result.stdout,
                stderr_summary=result.stderr or "failed to probe remote snapshot",
                metadata_ready=False,
            ),
        )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RemoteSnapshotProbeError(
            "remote snapshot probe did not return valid JSON",
            _distribution_details(
                remote_root=remote_root,
                substep="probe_parse_json",
                expected_job_id=None,
                expected_checksum=None,
                recommended_action="repair the remote probe script output formatting and retry",
                command_summary=result.recorded_command,
                exit_code=result.exit_code,
                stdout_summary=result.stdout,
                stderr_summary=result.stderr,
                parse_error=f"{type(exc).__name__}: {exc}",
                metadata_ready=False,
            ),
        ) from exc
    if not isinstance(payload, dict):
        raise RemoteSnapshotProbeError(
            "remote snapshot probe returned a non-object payload",
            _distribution_details(
                remote_root=remote_root,
                substep="probe_payload_shape",
                expected_job_id=None,
                expected_checksum=None,
                recommended_action="repair the remote probe payload shape and retry",
                command_summary=result.recorded_command,
                exit_code=result.exit_code,
                stdout_summary=result.stdout,
                stderr_summary=result.stderr,
                parse_error="probe payload must be a JSON object",
                metadata_ready=False,
            ),
        )
    return RemoteSnapshotProbe(
        exists=bool(payload["exists"]),
        empty=bool(payload["empty"]),
        checksum=payload.get("checksum"),
        job_id=payload.get("job_id"),
        missing_paths=tuple(str(item) for item in payload.get("missing_paths", [])),
        top_level_entries=tuple(str(item) for item in payload.get("top_level_entries", [])),
        metadata_ready=bool(payload.get("metadata_ready")),
        parse_error=(
            None
            if payload.get("parse_error") in {None, ""}
            else str(payload.get("parse_error"))
        ),
        command_summary=result.recorded_command,
        exit_code=result.exit_code,
        stdout_summary=_summary_text(result.stdout),
        stderr_summary=_summary_text(result.stderr),
    )


def _is_prepare_only_layout(probe: RemoteSnapshotProbe) -> bool:
    if (
        probe.empty
        or probe.job_id
        or probe.missing_paths
        or probe.parse_error is not None
        or probe.metadata_ready
    ):
        return False
    entries = set(probe.top_level_entries)
    return bool(entries) and entries <= _PREPARE_ONLY_TOP_LEVEL_DIRS


def _summary_text(text: str | None, *, limit: int = 400) -> str | None:
    if text is None:
        return None
    squashed = " ".join(text.split())
    if not squashed:
        return None
    return squashed[:limit]


def _probe_state(probe: RemoteSnapshotProbe) -> str:
    if not probe.exists:
        return "ABSENT"
    if _is_prepare_only_layout(probe):
        return "PREPARE_ONLY"
    if probe.metadata_ready:
        return "READY"
    if probe.parse_error is not None:
        return "INVALID"
    if probe.missing_paths:
        return "PARTIAL"
    if probe.empty:
        return "EMPTY"
    return "PRESENT"


def _probe_details(
    *,
    worker: WorkerConfig,
    remote_root: str,
    substep: str,
    probe: RemoteSnapshotProbe,
    expected_job_id: str,
    expected_checksum: str,
    recommended_action: str,
    secrets: Sequence[str] = (),
) -> dict[str, Any]:
    return _distribution_details(
        worker=worker,
        remote_root=remote_root,
        substep=substep,
        expected_job_id=expected_job_id,
        detected_job_id=probe.job_id,
        expected_checksum=expected_checksum,
        detected_checksum=probe.checksum,
        recommended_action=recommended_action,
        command_summary=probe.command_summary,
        exit_code=probe.exit_code,
        stdout_summary=probe.stdout_summary,
        stderr_summary=probe.stderr_summary,
        parse_error=probe.parse_error,
        metadata_ready=probe.metadata_ready,
        probe_state=_probe_state(probe),
        missing_paths=list(probe.missing_paths),
        top_level_entries=list(probe.top_level_entries),
        secrets=secrets,
    )


def _distribution_details(
    *,
    remote_root: str,
    substep: str,
    expected_job_id: str | None,
    expected_checksum: str | None,
    recommended_action: str,
    metadata_ready: bool,
    worker: WorkerConfig | None = None,
    detected_job_id: str | None = None,
    detected_checksum: str | None = None,
    command_summary: str | None = None,
    exit_code: int | None = None,
    stdout_summary: str | None = None,
    stderr_summary: str | None = None,
    parse_error: str | None = None,
    probe_state: str | None = None,
    missing_paths: list[str] | None = None,
    top_level_entries: list[str] | None = None,
    secrets: Sequence[str] = (),
) -> dict[str, Any]:
    payload = {
        "worker_id": None if worker is None else str(worker.worker_id),
        "stage": FailureStage.DISTRIBUTE.value,
        "action": "remote_snapshot",
        "substep": substep,
        "remote_path": remote_root,
        "command_summary": redact_text(command_summary, secrets),
        "exit_code": exit_code,
        "stdout_summary": redact_text(_summary_text(stdout_summary), secrets),
        "stderr_summary": redact_text(_summary_text(stderr_summary), secrets),
        "parse_error": redact_text(parse_error, secrets),
        "expected_job_id": redact_text(expected_job_id, secrets),
        "detected_job_id": redact_text(detected_job_id, secrets),
        "expected_checksum": expected_checksum,
        "detected_checksum": detected_checksum,
        "metadata_ready": metadata_ready,
        "probe_state": probe_state,
        "missing_paths": missing_paths or [],
        "top_level_entries": top_level_entries or [],
        "recommended_action": redact_text(recommended_action, secrets),
    }
    return redact_mapping(payload, secrets)


def _snapshot_checksum_files(root: Path) -> list[Path]:
    return sorted(
        candidate
        for candidate in root.rglob("*")
        if candidate.is_file()
        and candidate.relative_to(root).parts[0] not in _CHECKSUM_EXCLUDED_TOP_LEVEL_DIRS
        and "__pycache__" not in candidate.relative_to(root).parts
        and candidate.suffix != ".pyc"
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
