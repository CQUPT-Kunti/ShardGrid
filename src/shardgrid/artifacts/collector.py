"""Collect job artifacts from Workers back into the control snapshot."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Callable, Sequence, cast

from shardgrid.artifacts.ssh_transport import (
    _prepare_windows_staging_dir,
    _read_windows_userprofile,
    _windows_to_wsl_path,
)
from shardgrid.artifacts.store import validate_job_id
from shardgrid.artifacts.transport import (
    ArtifactTransferItemResult,
    ArtifactTransferSpec,
    ArtifactTransferStatus,
    ArtifactTransport,
    RemoteArtifactLocation,
)
from shardgrid.common.enums import FailureStage, PhysicalOS, RuntimeOS, SerializableStrEnum
from shardgrid.common.errors import make_failure_record
from shardgrid.common.process import ProcessResult, redact_text
from shardgrid.jobs.models import FailureRecord, JobSnapshot
from shardgrid.planner.models import WorkerAssignment


class ArtifactCollectionState(SerializableStrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    MISSING = "missing"
    FAILED = "failed"
    SKIPPED = "skipped"


class CollectionStatus(SerializableStrEnum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


@dataclass(frozen=True)
class ArtifactRule:
    label: str
    relative_path: str
    artifact_type: str
    optional: bool = False


@dataclass(frozen=True)
class WorkerArtifactSource:
    worker_id: str
    host: str
    ssh_user: str | None
    ssh_port: int
    rank: int
    stage: str | None
    remote_root: str
    physical_os: PhysicalOS | None = None
    runtime_os: RuntimeOS | None = None
    runtime_distro: str | None = None
    conda_environment: str | None = None
    conda_prefix: str | None = None
    private_key_path: str | None = None
    connect_timeout_seconds: int | None = None
    command_timeout_seconds: float | None = None
    known_host_policy: str | None = None
    known_hosts_path: str | None = None
    log_path: str | None = None
    checkpoint_paths: tuple[str, ...] = ()

    @classmethod
    def from_worker_assignment(
        cls,
        *,
        worker,
        assignment: WorkerAssignment,
        remote_root: str,
        checkpoint_paths: Sequence[str] = (),
        private_key_path: str | None = None,
        connect_timeout_seconds: int | None = None,
        command_timeout_seconds: float | None = None,
        known_host_policy: str | None = None,
        known_hosts_path: str | None = None,
    ) -> "WorkerArtifactSource":
        return cls(
            worker_id=str(assignment.worker_id),
            host=str(worker.host),
            ssh_user=worker.ssh_user,
            ssh_port=worker.ssh_port,
            physical_os=worker.physical_os,
            runtime_os=worker.runtime_os,
            runtime_distro=worker.runtime_distro,
            conda_environment=worker.conda_environment,
            conda_prefix=worker.conda_prefix,
            private_key_path=private_key_path,
            connect_timeout_seconds=connect_timeout_seconds,
            command_timeout_seconds=command_timeout_seconds,
            known_host_policy=known_host_policy,
            known_hosts_path=known_hosts_path,
            rank=assignment.rank,
            stage=assignment.stage,
            remote_root=remote_root,
            log_path=_relative_remote_path(remote_root, assignment.log_path),
            checkpoint_paths=tuple(
                _validate_relative_path(str(path))
                for path in checkpoint_paths
                if str(path).strip()
            ),
        )


@dataclass(frozen=True)
class CollectedArtifact:
    worker_id: str
    rank: int
    stage: str | None
    artifact_type: str
    relative_path: str
    remote_path: str
    local_path: str
    status: ArtifactCollectionState
    optional: bool = False
    size_bytes: int | None = None
    checksum: str | None = None
    recorded_command: str | None = None
    transport: str | None = None
    failure_class: str | None = None
    wsl_source: str | None = None
    windows_staging_path: str | None = None
    control_destination: str | None = None
    exit_code: int | None = None
    timed_out: bool = False
    stdout_summary: str | None = None
    stderr_summary: str | None = None
    bytes_received: int | None = None
    checksum_actual: str | None = None
    failure: FailureRecord | None = None

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


@dataclass(frozen=True)
class WorkerArtifactCollection:
    worker_id: str
    host: str
    rank: int
    stage: str | None
    status: CollectionStatus
    checkpoint_state: ArtifactCollectionState
    artifacts: tuple[CollectedArtifact, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


@dataclass(frozen=True)
class ArtifactCollectionResult:
    job_id: str
    snapshot_root: str
    status: CollectionStatus
    workers: tuple[WorkerArtifactCollection, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


DEFAULT_ARTIFACT_RULES = (
    ArtifactRule("stdout", "logs/stdout.log", "log"),
    ArtifactRule("stderr", "logs/stderr.log", "log"),
    ArtifactRule("runtime", "diagnostics/runtime.json", "diagnostic", optional=True),
    ArtifactRule("failure", "diagnostics/failure.json", "diagnostic", optional=True),
    ArtifactRule(
        "checkpoint-metadata",
        "checkpoint/checkpoint-metadata.json",
        "checkpoint_metadata",
    ),
    ArtifactRule("checkpoint", "checkpoint/model.pt", "checkpoint_file", optional=True),
)


class ArtifactCollector:
    def __init__(
        self,
        *,
        transport: ArtifactTransport,
        ssh_factory: Callable[[WorkerArtifactSource], Any] | None = None,
        runtime_factory: Callable[[WorkerArtifactSource, Any], Any] | None = None,
    ) -> None:
        self.transport = transport
        self._ssh_factory = ssh_factory
        self._runtime_factory = runtime_factory

    def collect(
        self,
        snapshot: JobSnapshot,
        *,
        sources: Sequence[WorkerArtifactSource],
        secrets: Sequence[str] = (),
        artifact_paths: Sequence[str] | None = None,
    ) -> ArtifactCollectionResult:
        snapshot_root = Path(snapshot.root_path).resolve(strict=False)
        validate_job_id(snapshot.job_id)
        workers = tuple(
            self._collect_worker(
                snapshot,
                snapshot_root=snapshot_root,
                source=source,
                rules=self._rules_for_source(source, artifact_paths=artifact_paths),
                secrets=secrets,
            )
            for source in sources
        )
        return ArtifactCollectionResult(
            job_id=str(snapshot.job_id),
            snapshot_root=str(snapshot_root),
            status=_overall_worker_status(workers),
            workers=workers,
        )

    def _collect_worker(
        self,
        snapshot: JobSnapshot,
        *,
        snapshot_root: Path,
        source: WorkerArtifactSource,
        rules: Sequence[ArtifactRule],
        secrets: Sequence[str],
    ) -> WorkerArtifactCollection:
        if _uses_windows_wsl_staging(source):
            return self._collect_windows_wsl_worker(
                snapshot,
                snapshot_root=snapshot_root,
                source=source,
                rules=rules,
                secrets=secrets,
            )
        artifacts = tuple(
            self._collect_artifact(
                snapshot,
                snapshot_root=snapshot_root,
                source=source,
                rule=rule,
                secrets=secrets,
            )
            for rule in rules
        )
        checkpoint_state = _checkpoint_state(artifacts)
        status = _worker_status(artifacts)
        if (
            checkpoint_state is ArtifactCollectionState.PARTIAL
            and status is CollectionStatus.SUCCESS
        ):
            status = CollectionStatus.PARTIAL
        return WorkerArtifactCollection(
            worker_id=source.worker_id,
            host=source.host,
            rank=source.rank,
            stage=source.stage,
            status=status,
            checkpoint_state=checkpoint_state,
            artifacts=artifacts,
        )

    def _rules_for_source(
        self,
        source: WorkerArtifactSource,
        *,
        artifact_paths: Sequence[str] | None,
    ) -> tuple[ArtifactRule, ...]:
        if artifact_paths is not None:
            return tuple(
                ArtifactRule(path, path, "custom", optional=False)
                for path in artifact_paths
            )
        rules: list[ArtifactRule] = []
        if source.log_path:
            rules.append(ArtifactRule("combined", source.log_path, "log"))
        for index, checkpoint_path in enumerate(source.checkpoint_paths):
            label = "checkpoint" if index == 0 else f"checkpoint-{index}"
            rules.append(ArtifactRule(label, checkpoint_path, "checkpoint_file"))
        return tuple(rules) or DEFAULT_ARTIFACT_RULES

    def _collect_artifact(
        self,
        snapshot: JobSnapshot,
        *,
        snapshot_root: Path,
        source: WorkerArtifactSource,
        rule: ArtifactRule,
        secrets: Sequence[str],
    ) -> CollectedArtifact:
        relative = _validate_relative_path(rule.relative_path)
        remote_path = str(PurePosixPath(source.remote_root, relative))
        transfer_source = remote_path
        staged = self._stage_for_windows_wsl_pull(
            snapshot=snapshot,
            source=source,
            rule=rule,
            relative=relative,
            remote_path=remote_path,
            secrets=secrets,
        )
        if isinstance(staged, CollectedArtifact):
            return staged
        if staged is not None:
            transfer_source = staged["scp_source"]
        local_path = _artifact_destination(snapshot, source, rule)
        _ensure_contained(local_path, snapshot_root)
        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return self._synthetic_failed_artifact(
                source=source,
                rule=rule,
                remote_path=remote_path,
                local_path=local_path,
                failure_class="ARTIFACT_DESTINATION_FAILED",
                reason=f"local artifact destination preparation failed: {exc}",
                command="local artifact destination preparation",
                exit_code=None,
                timed_out=False,
                stdout="",
                stderr=str(exc),
                secrets=secrets,
            )
        incoming_path = local_path.parent / f".incoming-{local_path.name}"
        if incoming_path.exists():
            incoming_path.unlink()
        result = self.transport.transfer(
            [
                ArtifactTransferSpec(
                    label=_artifact_label(source, rule),
                    source=transfer_source,
                    destination=str(incoming_path),
                    direction="pull",
                    local_root=str(snapshot_root),
                )
            ],
            remote=RemoteArtifactLocation(
                host=source.host,
                user=source.ssh_user,
                port=source.ssh_port,
                path=source.remote_root,
                private_key_path=source.private_key_path,
                connect_timeout_seconds=source.connect_timeout_seconds,
                command_timeout_seconds=source.command_timeout_seconds,
                known_host_policy=source.known_host_policy,
                known_hosts_path=source.known_hosts_path,
            ),
            secrets=secrets,
        ).items[0]
        if result.status is not ArtifactTransferStatus.SUCCESS:
            return self._failed_artifact(
                source=source,
                rule=rule,
                remote_path=remote_path,
                local_path=local_path,
                result=result,
                reason=result.stderr or "artifact transfer failed",
                failure_class=_transfer_failure_class(result),
                staged=staged,
            )
        if not incoming_path.exists():
            return self._failed_artifact(
                source=source,
                rule=rule,
                remote_path=remote_path,
                local_path=local_path,
                result=result,
                reason="artifact transfer completed without a local file",
                failure_class="ARTIFACT_DESTINATION_FAILED",
                staged=staged,
            )
        size = incoming_path.stat().st_size
        if size == 0:
            incoming_path.unlink(missing_ok=True)
            status = (
                ArtifactCollectionState.MISSING
                if rule.optional
                else ArtifactCollectionState.PARTIAL
            )
            return self._preserve_existing(
                source=source,
                rule=rule,
                remote_path=remote_path,
                local_path=local_path,
                result=result,
                status=status,
                reason="artifact is empty",
                failure_class="ARTIFACT_SIZE_MISMATCH",
                staged=staged,
            )
        checksum = _file_checksum(incoming_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if local_path.exists():
            existing_checksum = _file_checksum(local_path)
            if existing_checksum == checksum:
                incoming_path.unlink(missing_ok=True)
                self._cleanup_staged_artifact(staged=staged, secrets=secrets)
                return CollectedArtifact(
                    worker_id=source.worker_id,
                    rank=source.rank,
                    stage=source.stage,
                    artifact_type=rule.artifact_type,
                    relative_path=relative,
                    remote_path=remote_path,
                    local_path=str(local_path),
                    status=ArtifactCollectionState.SKIPPED,
                    optional=rule.optional,
                    size_bytes=local_path.stat().st_size,
                    checksum=existing_checksum,
                    recorded_command=result.recorded_command,
                    transport=result.transport,
                    wsl_source=remote_path if staged else None,
                    windows_staging_path=staged["windows_staging_path"] if staged else None,
                    control_destination=str(local_path),
                    exit_code=result.exit_code,
                    timed_out=result.timed_out,
                    stdout_summary=_summary(result.stdout),
                    stderr_summary=_summary(result.stderr),
                    bytes_received=local_path.stat().st_size,
                    checksum_actual=existing_checksum,
                )
        incoming_path.replace(local_path)
        self._cleanup_staged_artifact(staged=staged, secrets=secrets)
        state = ArtifactCollectionState.COMPLETE
        if rule.artifact_type == "checkpoint_file" and _metadata_declares_partial(snapshot, source):
            state = ArtifactCollectionState.PARTIAL
        return CollectedArtifact(
            worker_id=source.worker_id,
            rank=source.rank,
            stage=source.stage,
            artifact_type=rule.artifact_type,
            relative_path=relative,
            remote_path=remote_path,
            local_path=str(local_path),
            status=state,
            optional=rule.optional,
            size_bytes=size,
            checksum=checksum,
            recorded_command=result.recorded_command,
            transport=result.transport,
            wsl_source=remote_path if staged else None,
            windows_staging_path=staged["windows_staging_path"] if staged else None,
            control_destination=str(local_path),
            exit_code=result.exit_code,
            timed_out=result.timed_out,
            stdout_summary=_summary(result.stdout),
            stderr_summary=_summary(result.stderr),
            bytes_received=size,
            checksum_actual=checksum,
        )

    def _collect_windows_wsl_worker(
        self,
        snapshot: JobSnapshot,
        *,
        snapshot_root: Path,
        source: WorkerArtifactSource,
        rules: Sequence[ArtifactRule],
        secrets: Sequence[str],
    ) -> WorkerArtifactCollection:
        relative_rules = tuple(
            (_validate_relative_path(rule.relative_path), rule) for rule in rules
        )
        staged = self._stage_windows_wsl_worker(
            snapshot=snapshot,
            snapshot_root=snapshot_root,
            source=source,
            relative_rules=relative_rules,
            secrets=secrets,
        )
        if isinstance(staged, tuple):
            return _worker_collection(source, staged)

        incoming_root = snapshot_root / ".incoming-artifacts" / source.worker_id / _identity(source)
        _ensure_contained(incoming_root, snapshot_root)
        if incoming_root.exists():
            shutil.rmtree(incoming_root)
        result = self.transport.transfer(
            [
                ArtifactTransferSpec(
                    label=f"{source.worker_id}-{_identity(source)}-artifact-batch",
                    source=staged["scp_source"],
                    destination=str(incoming_root),
                    direction="pull",
                    local_root=str(snapshot_root),
                    recursive=True,
                )
            ],
            remote=RemoteArtifactLocation(
                host=source.host,
                user=source.ssh_user,
                port=source.ssh_port,
                path=source.remote_root,
                private_key_path=source.private_key_path,
                connect_timeout_seconds=source.connect_timeout_seconds,
                command_timeout_seconds=source.command_timeout_seconds,
                known_host_policy=source.known_host_policy,
                known_hosts_path=source.known_hosts_path,
            ),
            secrets=secrets,
        ).items[0]
        artifacts = tuple(
            self._complete_staged_artifact(
                snapshot=snapshot,
                snapshot_root=snapshot_root,
                source=source,
                rule=rule,
                relative=relative,
                staged=staged,
                incoming_root=incoming_root,
                result=result,
                secrets=secrets,
            )
            for relative, rule in relative_rules
        )
        if all(not _artifact_is_problem(artifact) for artifact in artifacts):
            shutil.rmtree(incoming_root, ignore_errors=True)
            self._cleanup_staged_worker(staged=staged, secrets=secrets)
        return _worker_collection(source, artifacts)

    def _complete_staged_artifact(
        self,
        *,
        snapshot: JobSnapshot,
        snapshot_root: Path,
        source: WorkerArtifactSource,
        rule: ArtifactRule,
        relative: str,
        staged: dict[str, Any],
        incoming_root: Path,
        result: ArtifactTransferItemResult,
        secrets: Sequence[str],
    ) -> CollectedArtifact:
        remote_path = str(PurePosixPath(source.remote_root, relative))
        local_path = _artifact_destination(snapshot, source, rule)
        _ensure_contained(local_path, snapshot_root)
        if result.status is not ArtifactTransferStatus.SUCCESS:
            return self._failed_artifact(
                source=source,
                rule=rule,
                remote_path=remote_path,
                local_path=local_path,
                result=result,
                reason=result.stderr or "artifact batch transfer failed",
                failure_class=_transfer_failure_class(result),
                staged=staged,
            )
        incoming_path = incoming_root / relative
        if not incoming_path.exists():
            staged_item = staged.get("stage_artifacts", {}).get(relative, {})
            if staged_item.get("status") == "missing" and rule.optional:
                return self._preserve_existing(
                    source=source,
                    rule=rule,
                    remote_path=remote_path,
                    local_path=local_path,
                    result=result,
                    status=ArtifactCollectionState.MISSING,
                    reason="optional source artifact is missing in the WSL runtime",
                    failure_class="ARTIFACT_SOURCE_MISSING",
                    staged=staged,
                )
            return self._failed_artifact(
                source=source,
                rule=rule,
                remote_path=remote_path,
                local_path=local_path,
                result=result,
                reason="artifact batch transfer completed without a local file",
                failure_class="ARTIFACT_DESTINATION_FAILED",
                staged=staged,
            )
        size = incoming_path.stat().st_size
        if size == 0:
            incoming_path.unlink(missing_ok=True)
            return self._preserve_existing(
                source=source,
                rule=rule,
                remote_path=remote_path,
                local_path=local_path,
                result=result,
                status=ArtifactCollectionState.MISSING
                if rule.optional
                else ArtifactCollectionState.PARTIAL,
                reason="artifact is empty",
                failure_class="ARTIFACT_SIZE_MISMATCH",
                staged=staged,
            )
        checksum = _file_checksum(incoming_path)
        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return self._synthetic_failed_artifact(
                source=source,
                rule=rule,
                remote_path=remote_path,
                local_path=local_path,
                failure_class="ARTIFACT_DESTINATION_FAILED",
                reason=f"local artifact destination preparation failed: {exc}",
                command="local artifact destination preparation",
                exit_code=None,
                timed_out=False,
                stdout="",
                stderr=str(exc),
                secrets=secrets,
                windows_staging_path=staged.get("windows_staging_path"),
            )
        if local_path.exists() and _file_checksum(local_path) == checksum:
            return CollectedArtifact(
                worker_id=source.worker_id,
                rank=source.rank,
                stage=source.stage,
                artifact_type=rule.artifact_type,
                relative_path=relative,
                remote_path=remote_path,
                local_path=str(local_path),
                status=ArtifactCollectionState.SKIPPED,
                optional=rule.optional,
                size_bytes=local_path.stat().st_size,
                checksum=checksum,
                recorded_command=result.recorded_command,
                transport=result.transport,
                wsl_source=remote_path,
                windows_staging_path=staged["windows_staging_path"],
                control_destination=str(local_path),
                exit_code=result.exit_code,
                timed_out=result.timed_out,
                stdout_summary=_summary(result.stdout),
                stderr_summary=_summary(result.stderr),
                bytes_received=local_path.stat().st_size,
                checksum_actual=checksum,
            )
        incoming_path.replace(local_path)
        state = ArtifactCollectionState.COMPLETE
        if rule.artifact_type == "checkpoint_file" and _metadata_declares_partial(snapshot, source):
            state = ArtifactCollectionState.PARTIAL
        return CollectedArtifact(
            worker_id=source.worker_id,
            rank=source.rank,
            stage=source.stage,
            artifact_type=rule.artifact_type,
            relative_path=relative,
            remote_path=remote_path,
            local_path=str(local_path),
            status=state,
            optional=rule.optional,
            size_bytes=size,
            checksum=checksum,
            recorded_command=result.recorded_command,
            transport=result.transport,
            wsl_source=remote_path,
            windows_staging_path=staged["windows_staging_path"],
            control_destination=str(local_path),
            exit_code=result.exit_code,
            timed_out=result.timed_out,
            stdout_summary=_summary(result.stdout),
            stderr_summary=_summary(result.stderr),
            bytes_received=size,
            checksum_actual=checksum,
        )

    def _stage_windows_wsl_worker(
        self,
        *,
        snapshot: JobSnapshot,
        snapshot_root: Path,
        source: WorkerArtifactSource,
        relative_rules: Sequence[tuple[str, ArtifactRule]],
        secrets: Sequence[str],
    ) -> dict[str, Any] | tuple[CollectedArtifact, ...]:
        del snapshot_root
        if self._ssh_factory is None or self._runtime_factory is None:
            return tuple(
                self._synthetic_failed_artifact(
                    source=source,
                    rule=rule,
                    remote_path=str(PurePosixPath(source.remote_root, relative)),
                    local_path=_artifact_destination(snapshot, source, rule),
                    failure_class="ARTIFACT_STAGE_FAILED",
                    reason="Windows+WSL artifact pull requires a staging runtime",
                    command=None,
                    exit_code=None,
                    timed_out=False,
                    stdout="",
                    stderr="missing ssh/runtime staging factory",
                    secrets=secrets,
                )
                for relative, rule in relative_rules
            )
        try:
            ssh = self._ssh_factory(source)
            runtime = self._runtime_factory(source, ssh)
            windows_profile = _read_windows_userprofile(ssh)
            staging_relative = str(
                PurePosixPath(
                    ".shardgrid",
                    "artifacts",
                    str(snapshot.job_id),
                    source.worker_id,
                    _identity(source),
                )
            )
            staging_windows = str(PureWindowsPath(windows_profile, staging_relative))
            _prepare_windows_staging_dir(ssh, staging_windows)
            staging_wsl = _windows_to_wsl_path(staging_windows)
        except Exception as exc:
            return tuple(
                self._synthetic_failed_artifact(
                    source=source,
                    rule=rule,
                    remote_path=str(PurePosixPath(source.remote_root, relative)),
                    local_path=_artifact_destination(snapshot, source, rule),
                    failure_class="ARTIFACT_STAGE_FAILED",
                    reason=f"artifact staging setup failed: {exc}",
                    command="windows+wsl artifact staging setup",
                    exit_code=None,
                    timed_out=False,
                    stdout="",
                    stderr=f"{type(exc).__name__}: {exc}",
                    secrets=secrets,
                )
                for relative, rule in relative_rules
            )
        artifacts = [
            {
                "relative_path": relative,
                "source": str(PurePosixPath(source.remote_root, relative)),
                "optional": rule.optional,
            }
            for relative, rule in relative_rules
        ]
        stage_result = runtime.run_script(
            _stage_artifacts_script(artifacts, staging_wsl),
            timeout=source.command_timeout_seconds,
            secrets=secrets,
        )
        payload = _load_stage_payload(stage_result.stdout)
        if not stage_result.ok:
            by_relative = {
                str(item.get("relative_path")): item for item in payload.get("artifacts", [])
            }
            failures: list[CollectedArtifact] = []
            for relative, rule in relative_rules:
                artifact_result = by_relative.get(relative, {})
                failure_class = (
                    "ARTIFACT_SOURCE_MISSING"
                    if artifact_result.get("status") == "missing"
                    else "ARTIFACT_STAGE_FAILED"
                )
                failures.append(
                    self._synthetic_failed_artifact(
                        source=source,
                        rule=rule,
                        remote_path=str(PurePosixPath(source.remote_root, relative)),
                        local_path=_artifact_destination(snapshot, source, rule),
                        failure_class=failure_class,
                        reason=_stage_failure_message(failure_class, stage_result),
                        command=stage_result.recorded_command,
                        exit_code=stage_result.exit_code,
                        timed_out=stage_result.timed_out,
                        stdout=stage_result.stdout,
                        stderr=stage_result.stderr,
                        secrets=secrets,
                        windows_staging_path=staging_windows,
                    )
                )
            return tuple(failures)
        return {
            "runtime": runtime,
            "scp_source": staging_relative,
            "windows_staging_path": staging_windows,
            "stage_command": stage_result.recorded_command,
            "stage_stdout": stage_result.stdout,
            "stage_exit_code": stage_result.exit_code,
            "stage_artifacts": {
                str(item.get("relative_path")): item
                for item in payload.get("artifacts", [])
                if isinstance(item, dict)
            },
        }

    def _stage_for_windows_wsl_pull(
        self,
        *,
        snapshot: JobSnapshot,
        source: WorkerArtifactSource,
        rule: ArtifactRule,
        relative: str,
        remote_path: str,
        secrets: Sequence[str],
    ) -> dict[str, Any] | CollectedArtifact | None:
        if not (
            source.physical_os is PhysicalOS.WINDOWS
            and source.runtime_os is RuntimeOS.WSL2_LINUX
        ):
            return None
        if self._ssh_factory is None or self._runtime_factory is None:
            return self._synthetic_failed_artifact(
                source=source,
                rule=rule,
                remote_path=remote_path,
                local_path=_artifact_destination(snapshot, source, rule),
                failure_class="ARTIFACT_STAGE_FAILED",
                reason="Windows+WSL artifact pull requires a staging runtime",
                command=None,
                exit_code=None,
                timed_out=False,
                stdout="",
                stderr="missing ssh/runtime staging factory",
                secrets=secrets,
            )
        try:
            ssh = self._ssh_factory(source)
            runtime = self._runtime_factory(source, ssh)
            windows_profile = _read_windows_userprofile(ssh)
            staging_relative = str(
                PurePosixPath(
                    ".shardgrid",
                    "artifacts",
                    str(snapshot.job_id),
                    source.worker_id,
                    _identity(source),
                    relative,
                )
            )
            staging_windows = str(PureWindowsPath(windows_profile, staging_relative))
            _prepare_windows_staging_dir(
                ssh,
                str(PureWindowsPath(staging_windows).parent),
            )
            staging_wsl = _windows_to_wsl_path(staging_windows)
        except Exception as exc:
            return self._synthetic_failed_artifact(
                source=source,
                rule=rule,
                remote_path=remote_path,
                local_path=_artifact_destination(snapshot, source, rule),
                failure_class="ARTIFACT_STAGE_FAILED",
                reason=f"artifact staging setup failed: {exc}",
                command="windows+wsl artifact staging setup",
                exit_code=None,
                timed_out=False,
                stdout="",
                stderr=f"{type(exc).__name__}: {exc}",
                secrets=secrets,
            )
        stage_result = runtime.run_script(
            _stage_artifact_script(remote_path, staging_wsl),
            timeout=source.command_timeout_seconds,
            secrets=secrets,
        )
        if not stage_result.ok:
            failure_class = (
                "ARTIFACT_SOURCE_MISSING"
                if stage_result.exit_code == 10
                else "ARTIFACT_STAGE_FAILED"
            )
            return self._synthetic_failed_artifact(
                source=source,
                rule=rule,
                remote_path=remote_path,
                local_path=_artifact_destination(snapshot, source, rule),
                failure_class=failure_class,
                reason=_stage_failure_message(failure_class, stage_result),
                command=stage_result.recorded_command,
                exit_code=stage_result.exit_code,
                timed_out=stage_result.timed_out,
                stdout=stage_result.stdout,
                stderr=stage_result.stderr,
                windows_staging_path=staging_windows,
                secrets=secrets,
            )
        return {
            "runtime": runtime,
            "scp_source": staging_relative,
            "windows_staging_path": staging_windows,
            "stage_command": stage_result.recorded_command,
            "stage_stdout": stage_result.stdout,
            "stage_exit_code": stage_result.exit_code,
        }

    def _cleanup_staged_artifact(
        self,
        *,
        staged: dict[str, Any] | None,
        secrets: Sequence[str],
    ) -> None:
        if staged is None:
            return
        runtime = staged.get("runtime")
        path = staged.get("windows_staging_path")
        if runtime is None or not isinstance(path, str):
            return
        try:
            runtime.run_script(
                _remove_artifact_script(_windows_to_wsl_path(path)),
                timeout=None,
                secrets=secrets,
            )
        except Exception:
            pass

    def _cleanup_staged_worker(
        self,
        *,
        staged: dict[str, Any],
        secrets: Sequence[str],
    ) -> None:
        runtime = staged.get("runtime")
        path = staged.get("windows_staging_path")
        if runtime is None or not isinstance(path, str):
            return
        try:
            runtime.run_script(
                _remove_artifact_tree_script(_windows_to_wsl_path(path)),
                timeout=None,
                secrets=secrets,
            )
        except Exception:
            pass

    def _failed_artifact(
        self,
        *,
        source: WorkerArtifactSource,
        rule: ArtifactRule,
        remote_path: str,
        local_path: Path,
        result,
        reason: str,
        failure_class: str,
        staged: dict[str, Any] | None = None,
    ) -> CollectedArtifact:
        missing = "no such file" in reason.lower()
        status = (
            ArtifactCollectionState.MISSING
            if missing and rule.optional
            else ArtifactCollectionState.PARTIAL
            if missing
            else ArtifactCollectionState.FAILED
        )
        return self._preserve_existing(
            source=source,
            rule=rule,
            remote_path=remote_path,
            local_path=local_path,
            result=result,
            status=status,
            reason=reason,
            failure_class=failure_class,
            staged=staged,
        )

    def _preserve_existing(
        self,
        *,
        source: WorkerArtifactSource,
        rule: ArtifactRule,
        remote_path: str,
        local_path: Path,
        result,
        status: ArtifactCollectionState,
        reason: str,
        failure_class: str,
        staged: dict[str, Any] | None = None,
    ) -> CollectedArtifact:
        size = checksum = None
        if local_path.exists() and local_path.stat().st_size > 0:
            size = local_path.stat().st_size
            checksum = _file_checksum(local_path)
        failure = make_failure_record(
            stage=FailureStage.CHECKPOINT,
            host=source.host,
            worker_id=source.worker_id,
            command=result.recorded_command,
            exit_code=result.exit_code,
            message=reason,
            recommended_action="inspect the remote artifact and retry collection",
            retryable=True,
        )
        stderr = getattr(result, "stderr", "")
        stdout = getattr(result, "stdout", "")
        return CollectedArtifact(
            worker_id=source.worker_id,
            rank=source.rank,
            stage=source.stage,
            artifact_type=rule.artifact_type,
            relative_path=rule.relative_path,
            remote_path=remote_path,
            local_path=str(local_path),
            status=status,
            optional=rule.optional,
            size_bytes=size,
            checksum=checksum,
            recorded_command=result.recorded_command,
            transport=getattr(result, "transport", None),
            failure_class=failure_class,
            wsl_source=remote_path if staged else None,
            windows_staging_path=staged["windows_staging_path"] if staged else None,
            control_destination=str(local_path),
            exit_code=result.exit_code,
            timed_out=getattr(result, "timed_out", False),
            stdout_summary=_summary(stdout),
            stderr_summary=_summary(stderr),
            bytes_received=size,
            checksum_actual=checksum,
            failure=failure,
        )

    def _synthetic_failed_artifact(
        self,
        *,
        source: WorkerArtifactSource,
        rule: ArtifactRule,
        remote_path: str,
        local_path: Path,
        failure_class: str,
        reason: str,
        command: str | None,
        exit_code: int | None,
        timed_out: bool,
        stdout: str,
        stderr: str,
        secrets: Sequence[str],
        windows_staging_path: str | None = None,
    ) -> CollectedArtifact:
        result = ArtifactTransferItemResult(
            label=_artifact_label(source, rule),
            transport=self.transport.name.value,
            status=ArtifactTransferStatus.FAILED,
            source=remote_path,
            destination=str(local_path),
            recorded_command=command,
            exit_code=exit_code,
            stdout=redact_text(stdout, secrets) or "",
            stderr=redact_text(stderr, secrets) or "",
            timed_out=timed_out,
        )
        return self._preserve_existing(
            source=source,
            rule=rule,
            remote_path=remote_path,
            local_path=local_path,
            result=result,
            status=ArtifactCollectionState.FAILED,
            reason=reason,
            failure_class=failure_class,
            staged={"windows_staging_path": windows_staging_path}
            if windows_staging_path
            else None,
        )


def _artifact_destination(
    snapshot: JobSnapshot,
    source: WorkerArtifactSource,
    rule: ArtifactRule,
) -> Path:
    identity = _identity(source)
    filename = Path(rule.relative_path).name
    if rule.artifact_type == "log":
        return Path(snapshot.logs_path) / source.worker_id / identity / filename
    if rule.artifact_type == "diagnostic":
        return Path(snapshot.diagnostics_path) / source.worker_id / identity / filename
    if rule.artifact_type == "checkpoint_metadata":
        return Path(snapshot.checkpoint_path) / "metadata" / source.worker_id / identity / filename
    return Path(snapshot.checkpoint_path) / "files" / source.worker_id / identity / filename


def _validate_relative_path(path: str) -> str:
    pure = PurePosixPath(path)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError("artifact path escaped snapshot root")
    return str(pure)


def _ensure_contained(path: Path, root: Path) -> None:
    candidate = path.resolve(strict=False)
    if root not in candidate.parents and candidate != root:
        raise ValueError("artifact path escaped snapshot root")


def _uses_windows_wsl_staging(source: WorkerArtifactSource) -> bool:
    return (
        source.physical_os is PhysicalOS.WINDOWS
        and source.runtime_os is RuntimeOS.WSL2_LINUX
    )


def _worker_collection(
    source: WorkerArtifactSource,
    artifacts: Sequence[CollectedArtifact],
) -> WorkerArtifactCollection:
    artifact_tuple = tuple(artifacts)
    checkpoint_state = _checkpoint_state(artifact_tuple)
    status = _worker_status(artifact_tuple)
    if (
        checkpoint_state is ArtifactCollectionState.PARTIAL
        and status is CollectionStatus.SUCCESS
    ):
        status = CollectionStatus.PARTIAL
    return WorkerArtifactCollection(
        worker_id=source.worker_id,
        host=source.host,
        rank=source.rank,
        stage=source.stage,
        status=status,
        checkpoint_state=checkpoint_state,
        artifacts=artifact_tuple,
    )


def _transfer_failure_class(result: ArtifactTransferItemResult) -> str:
    stderr = result.stderr.lower()
    if "local destination preparation failed" in stderr:
        return "ARTIFACT_DESTINATION_FAILED"
    if result.timed_out or "connection timed out" in stderr or "connect to host" in stderr:
        return "ARTIFACT_SCP_CONNECT_TIMEOUT"
    return "ARTIFACT_SCP_FAILED"


def _artifact_label(source: WorkerArtifactSource, rule: ArtifactRule) -> str:
    stage = "" if source.stage is None else f"-{source.stage}"
    return f"{source.worker_id}-rank{source.rank}{stage}-{rule.label}"


def _identity(source: WorkerArtifactSource) -> str:
    return f"rank{source.rank}" if source.stage is None else f"rank{source.rank}-{source.stage}"


def _file_checksum(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _checkpoint_state(artifacts: Sequence[CollectedArtifact]) -> ArtifactCollectionState:
    metadata = next(
        (item for item in artifacts if item.artifact_type == "checkpoint_metadata"),
        None,
    )
    checkpoints = [item for item in artifacts if item.artifact_type == "checkpoint_file"]
    if metadata is None and not checkpoints:
        return ArtifactCollectionState.MISSING
    if metadata is not None:
        if metadata.status in {
            ArtifactCollectionState.MISSING,
            ArtifactCollectionState.FAILED,
        }:
            return ArtifactCollectionState.MISSING
        if metadata.status is ArtifactCollectionState.PARTIAL:
            return ArtifactCollectionState.PARTIAL
    if not checkpoints:
        return ArtifactCollectionState.MISSING
    if any(
        item.status in {ArtifactCollectionState.FAILED, ArtifactCollectionState.PARTIAL}
        for item in checkpoints
    ):
        return ArtifactCollectionState.PARTIAL
    if any(
        item.status is ArtifactCollectionState.MISSING
        for item in checkpoints
        if not item.optional
    ):
        return ArtifactCollectionState.PARTIAL
    if metadata is None:
        return ArtifactCollectionState.COMPLETE
    if all(
        item.status in {ArtifactCollectionState.COMPLETE, ArtifactCollectionState.SKIPPED}
        for item in checkpoints
    ) and not _metadata_file_says_partial(metadata.local_path):
        return ArtifactCollectionState.COMPLETE
    return ArtifactCollectionState.PARTIAL


def _metadata_declares_partial(snapshot: JobSnapshot, source: WorkerArtifactSource) -> bool:
    path = (
        Path(snapshot.checkpoint_path)
        / "metadata"
        / source.worker_id
        / (f"rank{source.rank}" if source.stage is None else f"rank{source.rank}-{source.stage}")
        / "checkpoint-metadata.json"
    )
    return _metadata_file_says_partial(path)


def _metadata_file_says_partial(path: str | Path) -> bool:
    metadata_path = Path(path)
    if not metadata_path.exists():
        return False
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return True
    return str(payload.get("status", "")).lower() != "complete"


def _relative_remote_path(remote_root: str, path: str | None) -> str | None:
    if path is None or not str(path).strip():
        return None
    remote = PurePosixPath(str(path))
    if not remote.is_absolute():
        return _validate_relative_path(str(remote))
    root = PurePosixPath(remote_root)
    if remote == root or root not in remote.parents:
        raise ValueError("artifact path escaped remote root")
    return _validate_relative_path(str(remote.relative_to(root)))


def _worker_status(artifacts: Sequence[CollectedArtifact]) -> CollectionStatus:
    if all(not _artifact_is_problem(artifact) for artifact in artifacts):
        return CollectionStatus.SUCCESS
    if any(
        artifact.status in {ArtifactCollectionState.COMPLETE, ArtifactCollectionState.SKIPPED}
        for artifact in artifacts
    ):
        return CollectionStatus.PARTIAL
    return CollectionStatus.PARTIAL


def _overall_worker_status(workers: Sequence[WorkerArtifactCollection]) -> CollectionStatus:
    statuses = {worker.status for worker in workers}
    if statuses == {CollectionStatus.SUCCESS}:
        return CollectionStatus.SUCCESS
    if CollectionStatus.PARTIAL in statuses or CollectionStatus.SUCCESS in statuses:
        return CollectionStatus.PARTIAL
    return CollectionStatus.FAILED


def _artifact_is_problem(artifact: CollectedArtifact) -> bool:
    if artifact.status in {ArtifactCollectionState.COMPLETE, ArtifactCollectionState.SKIPPED}:
        return False
    if artifact.status is ArtifactCollectionState.MISSING and artifact.optional:
        return False
    return True


def _summary(text: str | None, limit: int = 500) -> str | None:
    if not text:
        return None
    stripped = text.strip()
    if len(stripped) <= limit:
        return stripped
    return stripped[:limit] + "...[truncated]"


def _stage_failure_message(failure_class: str, result: ProcessResult) -> str:
    if failure_class == "ARTIFACT_SOURCE_MISSING":
        return "source artifact is missing in the WSL runtime"
    if result.timed_out:
        return "artifact staging from WSL to Windows timed out"
    return "artifact staging from WSL to Windows failed"


def _stage_artifact_script(source: str, destination: str) -> str:
    return f"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

source = Path({source!r})
destination = Path({destination!r})
payload = {{
    "source": str(source),
    "destination": str(destination),
    "exists": source.is_file(),
    "size_bytes": None,
    "checksum": None,
}}
if not source.is_file():
    print(json.dumps(payload, sort_keys=True))
    sys.exit(10)
destination.parent.mkdir(parents=True, exist_ok=True)
incoming = destination.parent / (".incoming-" + destination.name)
if incoming.exists():
    incoming.unlink()
shutil.copy2(source, incoming)
size = incoming.stat().st_size
if size <= 0:
    incoming.unlink(missing_ok=True)
    payload["size_bytes"] = size
    print(json.dumps(payload, sort_keys=True))
    sys.exit(11)
digest = hashlib.sha256()
digest.update(incoming.read_bytes())
payload["size_bytes"] = size
payload["checksum"] = digest.hexdigest()
incoming.replace(destination)
print(json.dumps(payload, sort_keys=True))
"""


def _stage_artifacts_script(artifacts: Sequence[dict[str, object]], staging_root: str) -> str:
    return f"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

staging_root = Path({staging_root!r})
artifacts = {json.dumps(list(artifacts), sort_keys=True)!r}
payload = {{"artifacts": []}}
failed = False
for artifact in json.loads(artifacts):
    relative = artifact["relative_path"]
    source = Path(artifact["source"])
    destination = staging_root / relative
    item = {{
        "relative_path": relative,
        "source": str(source),
        "destination": str(destination),
        "status": "missing",
        "size_bytes": None,
        "checksum": None,
    }}
    if not source.is_file():
        payload["artifacts"].append(item)
        if not artifact.get("optional", False):
            failed = True
        continue
    destination.parent.mkdir(parents=True, exist_ok=True)
    incoming = destination.parent / (".incoming-" + destination.name)
    if incoming.exists():
        incoming.unlink()
    shutil.copy2(source, incoming)
    size = incoming.stat().st_size
    if size <= 0:
        incoming.unlink(missing_ok=True)
        item["status"] = "empty"
        item["size_bytes"] = size
        payload["artifacts"].append(item)
        if not artifact.get("optional", False):
            failed = True
        continue
    digest = hashlib.sha256()
    digest.update(incoming.read_bytes())
    item["status"] = "staged"
    item["size_bytes"] = size
    item["checksum"] = digest.hexdigest()
    incoming.replace(destination)
    payload["artifacts"].append(item)
print(json.dumps(payload, sort_keys=True))
sys.exit(10 if failed else 0)
"""


def _remove_artifact_script(path: str) -> str:
    return f"""
from pathlib import Path

path = Path({path!r})
if path.exists() and path.is_file():
    path.unlink()
"""


def _remove_artifact_tree_script(path: str) -> str:
    return f"""
import shutil
from pathlib import Path

path = Path({path!r})
if path.exists():
    shutil.rmtree(path, ignore_errors=True)
"""


def _load_stage_payload(stdout: str) -> dict[str, Any]:
    try:
        payload = json.loads(stdout or "{}")
    except json.JSONDecodeError:
        return {"artifacts": []}
    return payload if isinstance(payload, dict) else {"artifacts": []}


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
