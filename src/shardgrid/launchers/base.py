"""Backend-neutral launcher lifecycle contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, cast

from shardgrid.common.enums import BackendStatus, JobState, SerializableStrEnum
from shardgrid.control.resource_manager import ClusterState
from shardgrid.jobs.models import FailureRecord, JobSnapshot, JobStatus, TrainingJob
from shardgrid.planner.models import ExecutionPlan


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


class LauncherOperation(SerializableStrEnum):
    PREPARE = "prepare"
    DISTRIBUTE = "distribute"
    LAUNCH = "launch"
    MONITOR = "monitor"
    LOGS = "logs"
    STOP = "stop"
    CLEANUP = "cleanup"


class LauncherResultStatus(SerializableStrEnum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    NOOP = "noop"
    BLOCKED = "blocked"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class LauncherContext:
    job: TrainingJob
    execution_plan: ExecutionPlan
    cluster_state: ClusterState
    snapshot: JobSnapshot | None = None
    job_status: JobStatus | None = None
    runtime_environment_refs: dict[str, str] = field(default_factory=dict)
    backend_config: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


@dataclass(frozen=True)
class RankResult:
    rank: int
    worker_id: str
    stage: str | None = None
    status: LauncherResultStatus = LauncherResultStatus.SUCCESS
    pid: int | None = None
    log_ref: str | None = None
    evidence_ref: str | None = None
    failure: FailureRecord | None = None
    idempotent: bool = True
    duplicate_safe: bool = True
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


@dataclass(frozen=True)
class WorkerResult:
    worker_id: str
    status: LauncherResultStatus
    rank_results: tuple[RankResult, ...] = ()
    failure: FailureRecord | None = None
    evidence_ref: str | None = None
    idempotent: bool = True
    duplicate_safe: bool = True
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


@dataclass(frozen=True)
class LogResult:
    worker_id: str
    rank: int | None = None
    stage: str | None = None
    job_id: str | None = None
    stream: str | None = None
    source: str | None = None
    location: str | None = None
    source_path: str | None = None
    tail: str = ""
    content: str = ""
    status: LauncherResultStatus = LauncherResultStatus.SUCCESS
    failure: FailureRecord | None = None
    message: str = ""

    def __post_init__(self) -> None:
        if self.content and not self.tail:
            object.__setattr__(self, "tail", self.content)
        elif self.tail and not self.content:
            object.__setattr__(self, "content", self.tail)

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


@dataclass(frozen=True)
class LauncherResult:
    operation: LauncherOperation
    status: LauncherResultStatus
    backend: str
    job_id: str
    worker_results: tuple[WorkerResult, ...] = ()
    log_results: tuple[LogResult, ...] = ()
    failure: FailureRecord | None = None
    blocker: str | None = None
    evidence_refs: tuple[str, ...] = ()
    idempotent: bool = True
    duplicate_safe: bool = True
    next_job_state: JobState | None = None
    message: str = ""

    @property
    def ok(self) -> bool:
        return self.status in {LauncherResultStatus.SUCCESS, LauncherResultStatus.NOOP}

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)

    @classmethod
    def from_worker_results(
        cls,
        *,
        operation: LauncherOperation,
        backend: str,
        job_id: str,
        worker_results: list[WorkerResult],
        next_job_state: JobState | None = None,
        blocker: str | None = None,
        failure: FailureRecord | None = None,
        message: str = "",
    ) -> "LauncherResult":
        statuses = {result.status for result in worker_results}
        if statuses <= {LauncherResultStatus.SUCCESS, LauncherResultStatus.NOOP}:
            status = (
                LauncherResultStatus.NOOP
                if statuses == {LauncherResultStatus.NOOP}
                else LauncherResultStatus.SUCCESS
            )
        elif (
            any(
                value in statuses
                for value in (LauncherResultStatus.SUCCESS, LauncherResultStatus.NOOP)
            )
            and any(
                value in statuses
                for value in (
                    LauncherResultStatus.FAILED,
                    LauncherResultStatus.BLOCKED,
                    LauncherResultStatus.UNSUPPORTED,
                    LauncherResultStatus.PARTIAL,
                )
            )
        ):
            status = LauncherResultStatus.PARTIAL
        elif LauncherResultStatus.UNSUPPORTED in statuses:
            status = LauncherResultStatus.UNSUPPORTED
        elif LauncherResultStatus.BLOCKED in statuses:
            status = LauncherResultStatus.BLOCKED
        elif LauncherResultStatus.FAILED in statuses and (
            LauncherResultStatus.SUCCESS in statuses or LauncherResultStatus.NOOP in statuses
        ):
            status = LauncherResultStatus.PARTIAL
        elif LauncherResultStatus.PARTIAL in statuses:
            status = LauncherResultStatus.PARTIAL
        else:
            status = LauncherResultStatus.FAILED
        if failure is None:
            failure = next(
                (result.failure for result in worker_results if result.failure is not None),
                None,
            )
        evidence_refs = tuple(
            ref
            for ref in dict.fromkeys(
                result.evidence_ref for result in worker_results if result.evidence_ref
            )
        )
        return cls(
            operation=operation,
            status=status,
            backend=backend,
            job_id=job_id,
            worker_results=tuple(worker_results),
            failure=failure,
            blocker=blocker,
            evidence_refs=evidence_refs,
            idempotent=all(result.idempotent for result in worker_results),
            duplicate_safe=all(result.duplicate_safe for result in worker_results),
            next_job_state=next_job_state,
            message=message,
        )


@dataclass(frozen=True)
class LauncherCapabilities:
    backend: str
    status: BackendStatus
    supported_operations: tuple[LauncherOperation, ...]
    limitations: tuple[str, ...] = ()

    def supports(self, operation: LauncherOperation) -> bool:
        return operation in self.supported_operations

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


class Launcher(ABC):
    capabilities: LauncherCapabilities

    @abstractmethod
    def prepare(self, context: LauncherContext) -> LauncherResult: ...

    @abstractmethod
    def distribute(self, context: LauncherContext) -> LauncherResult: ...

    @abstractmethod
    def launch(self, context: LauncherContext) -> LauncherResult: ...

    @abstractmethod
    def monitor(self, context: LauncherContext) -> LauncherResult: ...

    @abstractmethod
    def logs(self, context: LauncherContext) -> LauncherResult: ...

    @abstractmethod
    def stop(self, context: LauncherContext) -> LauncherResult: ...

    @abstractmethod
    def cleanup(self, context: LauncherContext) -> LauncherResult: ...
