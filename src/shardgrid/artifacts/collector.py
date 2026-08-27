"""Collect job artifacts from Workers back into the control snapshot."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Sequence, cast

from shardgrid.artifacts.store import validate_job_id
from shardgrid.artifacts.transport import (
    ArtifactTransferSpec,
    ArtifactTransferStatus,
    ArtifactTransport,
    RemoteArtifactLocation,
)
from shardgrid.common.enums import FailureStage, SerializableStrEnum
from shardgrid.common.errors import make_failure_record
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
    private_key_path: str | None
    rank: int
    stage: str | None
    remote_root: str

    @classmethod
    def from_worker_assignment(
        cls,
        *,
        worker,
        assignment: WorkerAssignment,
        remote_root: str,
    ) -> "WorkerArtifactSource":
        return cls(
            worker_id=str(assignment.worker_id),
            host=str(worker.host),
            ssh_user=worker.ssh_user,
            ssh_port=worker.ssh_port,
            private_key_path=None,
            rank=assignment.rank,
            stage=assignment.stage,
            remote_root=remote_root,
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
    def __init__(self, *, transport: ArtifactTransport) -> None:
        self.transport = transport

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
        rules = (
            tuple(
                ArtifactRule(path, path, "custom", optional=False)
                for path in artifact_paths or ()
            )
            if artifact_paths is not None
            else DEFAULT_ARTIFACT_RULES
        )
        workers = tuple(
            self._collect_worker(
                snapshot,
                snapshot_root=snapshot_root,
                source=source,
                rules=rules,
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
        local_path = _artifact_destination(snapshot, source, rule)
        _ensure_contained(local_path, snapshot_root)
        incoming_path = local_path.parent / f".incoming-{local_path.name}"
        if incoming_path.exists():
            incoming_path.unlink()
        result = self.transport.transfer(
            [
                ArtifactTransferSpec(
                    label=_artifact_label(source, rule),
                    source=remote_path,
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
            )
        if not incoming_path.exists():
            return self._failed_artifact(
                source=source,
                rule=rule,
                remote_path=remote_path,
                local_path=local_path,
                result=result,
                reason="artifact transfer completed without a local file",
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
            )
        checksum = _file_checksum(incoming_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if local_path.exists():
            existing_checksum = _file_checksum(local_path)
            if existing_checksum == checksum:
                incoming_path.unlink(missing_ok=True)
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
        )

    def _failed_artifact(
        self,
        *,
        source: WorkerArtifactSource,
        rule: ArtifactRule,
        remote_path: str,
        local_path: Path,
        result,
        reason: str,
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
            failure=failure,
        )


def _artifact_destination(
    snapshot: JobSnapshot,
    source: WorkerArtifactSource,
    rule: ArtifactRule,
) -> Path:
    identity = f"rank{source.rank}" if source.stage is None else f"rank{source.rank}-{source.stage}"
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


def _artifact_label(source: WorkerArtifactSource, rule: ArtifactRule) -> str:
    stage = "" if source.stage is None else f"-{source.stage}"
    return f"{source.worker_id}-rank{source.rank}{stage}-{rule.label}"


def _file_checksum(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _checkpoint_state(artifacts: Sequence[CollectedArtifact]) -> ArtifactCollectionState:
    metadata = next(
        (item for item in artifacts if item.artifact_type == "checkpoint_metadata"),
        None,
    )
    checkpoint = next(
        (item for item in artifacts if item.artifact_type == "checkpoint_file"),
        None,
    )
    if metadata is None or metadata.status in {
        ArtifactCollectionState.MISSING,
        ArtifactCollectionState.FAILED,
    }:
        return ArtifactCollectionState.MISSING
    if metadata.status is ArtifactCollectionState.PARTIAL:
        return ArtifactCollectionState.PARTIAL
    if checkpoint is None:
        return ArtifactCollectionState.MISSING
    if checkpoint.status is ArtifactCollectionState.COMPLETE and not _metadata_file_says_partial(
        metadata.local_path
    ):
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
        return True
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return True
    return str(payload.get("status", "")).lower() != "complete"


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
