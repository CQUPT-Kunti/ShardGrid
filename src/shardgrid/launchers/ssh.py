"""OpenSSH-backed launcher implementation."""

from __future__ import annotations

import json
import shlex
import tempfile
from dataclasses import asdict, dataclass, is_dataclass, replace
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Sequence, cast

from shardgrid.artifacts.ssh_transport import (
    DistributionStatus,
    WorkerDistributionResult,
    _build_snapshot_archive,
    _is_prepare_only_layout,
    _load_snapshot_transport_preference,
    _probe_remote_snapshot,
    _read_windows_userprofile,
    _windows_to_wsl_path,
    distribute_job_snapshot_to_worker,
    snapshot_checksum,
)
from shardgrid.artifacts.transport import (
    ArtifactTransferItemResult,
    ArtifactTransport,
    build_transport_config,
    select_artifact_transport,
)
from shardgrid.common.config import ClusterConfig, WorkerConfig
from shardgrid.common.enums import BackendStatus, FailureStage, JobState
from shardgrid.common.errors import make_failure_record
from shardgrid.common.process import ProcessResult, redact_text
from shardgrid.control.status_store import StatusStore
from shardgrid.jobs.models import FailureRecord, JobStatus
from shardgrid.launchers.base import (
    Launcher,
    LauncherCapabilities,
    LauncherContext,
    LauncherOperation,
    LauncherResult,
    LauncherResultStatus,
    LogResult,
    RankResult,
    WorkerResult,
)
from shardgrid.transport.runtime import WSLRuntimeConfig, WSLRuntimeWrapper
from shardgrid.transport.ssh import SSHOptions, SSHTransport

RuntimeFactory = Callable[[WorkerConfig], WSLRuntimeWrapper]
SSHFactory = Callable[[WorkerConfig], SSHTransport]
TransportFactory = Callable[[WorkerConfig], ArtifactTransport]
_PREPARE_OUTPUT_DIRS = ("logs", "diagnostics", "checkpoint")
_EVENT_MARKER = "STAGE_PLACEMENT_EVIDENCE "
_FORWARD_MARKER = "T072_FORWARD_EVIDENCE "
_BACKWARD_MARKER = "T073_BACKWARD_EVIDENCE "
_TRAIN_MARKER = "T074_TRAIN_EVIDENCE "
_LAUNCHER_OWNS_LOG_ENV = "SHARDGRID_LAUNCHER_OWNS_LOG_SINK"
_PLAIN_TRAIN_MARKERS = (
    "TRAIN_STEP_BEGIN",
    "TRAIN_STEP_END",
    "LOSS_READY",
    "STAGE0_FORWARD_BEGIN",
    "STAGE0_FORWARD_END",
    "STAGE1_FORWARD_BEGIN",
    "STAGE1_FORWARD_END",
    "STAGE0_BACKWARD_BEGIN",
    "STAGE0_BACKWARD_END",
    "STAGE1_BACKWARD_BEGIN",
    "STAGE1_BACKWARD_END",
    "ACTIVATION_SEND_BEGIN",
    "ACTIVATION_SEND_END",
    "ACTIVATION_RECV_BEGIN",
    "ACTIVATION_RECV_END",
    "GRADIENT_RECV_BEGIN",
    "GRADIENT_RECV_END",
    "GRADIENT_SEND_BEGIN",
    "GRADIENT_SEND_END",
    "OPTIMIZER_STEP_END",
)


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


def _now() -> str:
    return datetime.now(tz=UTC).isoformat()


def _clip_marker_payload(text: str, *, limit: int = 160) -> tuple[str, str]:
    if len(text) <= limit:
        return text, text
    return text[:limit], text[-limit:]


@dataclass(frozen=True)
class SSHProcessRecord:
    job_id: str
    worker_id: str
    rank: int
    local_rank: int
    stage: str | None
    pid: int
    command_argv: tuple[str, ...]
    log_path: str
    launched_at: str
    remote_host: str
    remote_root: str
    status: str = "running"

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


@dataclass(frozen=True)
class ProcessLivenessProbe:
    state: str
    detail: str
    transport_status: str
    recorded_command: str
    exit_code: int | None
    timed_out: bool
    stdout: str
    stderr: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "detail": self.detail,
            "transport_status": self.transport_status,
            "recorded_command": self.recorded_command,
            "exit_code": self.exit_code,
            "timed_out": self.timed_out,
            "stdout": self.stdout,
            "stderr": self.stderr,
        }


class SSHLauncher(Launcher):
    def __init__(
        self,
        cluster_config: ClusterConfig,
        *,
        artifact_transport: ArtifactTransport | None = None,
        ssh_factory: SSHFactory | None = None,
        runtime_factory: RuntimeFactory | None = None,
        transport_factory: TransportFactory | None = None,
        secrets: Sequence[str] = (),
    ) -> None:
        self.cluster_config = cluster_config
        self._artifact_transport = artifact_transport
        self._ssh_factory = ssh_factory or self._default_ssh_factory
        self._runtime_factory = runtime_factory or self._default_runtime_factory
        self._transport_factory = transport_factory or self._default_transport_factory
        self._secrets = tuple(secret for secret in secrets if secret)
        self._process_records: dict[tuple[str, str, int], SSHProcessRecord] = {}
        self._distribution_records: dict[tuple[str, str], WorkerDistributionResult] = {}
        self.capabilities = LauncherCapabilities(
            backend="ssh",
            status=BackendStatus.AVAILABLE,
            supported_operations=tuple(LauncherOperation),
            limitations=(
                "T093 foundation backend: full prepare/distribute/monitor/stop/cleanup "
                "workflow lands in later tasks",
            ),
        )

    def prepare(self, context: LauncherContext) -> LauncherResult:
        workers = self._selected_workers(context)
        try:
            local_entrypoints = self._validate_local_entrypoints(context)
        except ValueError as exc:
            return LauncherResult.from_worker_results(
                operation=LauncherOperation.PREPARE,
                backend=self.capabilities.backend,
                job_id=str(context.job.job_id),
                worker_results=[
                    WorkerResult(
                        worker_id=str(worker.worker_id),
                        status=LauncherResultStatus.BLOCKED,
                        failure=self._launcher_failure(
                            stage=FailureStage.LAUNCH,
                            worker=worker,
                            message=str(exc),
                            recommended_action=(
                                "repair the immutable snapshot inputs before rerunning prepare"
                            ),
                        ),
                        evidence_ref=self._evidence_ref(context, "prepare", worker),
                    )
                    for worker in workers
                ],
                next_job_state=JobState.PROBING,
            )
        results: list[WorkerResult] = []
        for worker in workers:
            runtime_refs = self._runtime_refs_for_worker(context, str(worker.worker_id))
            if not runtime_refs:
                results.append(
                    WorkerResult(
                        worker_id=str(worker.worker_id),
                        status=LauncherResultStatus.BLOCKED,
                        failure=self._launcher_failure(
                            stage=FailureStage.LAUNCH,
                            worker=worker,
                            message="runtime environment reference is missing for selected worker",
                            recommended_action=(
                                "persist runtime_environment_refs for every selected rank, "
                                "then rerun prepare"
                            ),
                        ),
                        evidence_ref=self._evidence_ref(context, "prepare", worker),
                    )
                )
                continue
            runtime = self._runtime_factory(worker)
            remote_root = self._remote_job_root(context)
            script = self._prepare_probe_script(
                remote_root=remote_root,
                job_id=str(context.job.job_id),
                entrypoints=local_entrypoints[str(worker.worker_id)],
                expected_conda_prefix=worker.conda_prefix,
            )
            try:
                probe = runtime.run_script(
                    script,
                    timeout=self._remote_command_timeout_seconds(),
                )
            except Exception as exc:
                results.append(
                    WorkerResult(
                        worker_id=str(worker.worker_id),
                        status=LauncherResultStatus.FAILED,
                        failure=self._launcher_failure(
                            stage=FailureStage.LAUNCH,
                            worker=worker,
                            message=f"runtime wrapper failure during prepare probe: {exc}",
                            recommended_action=(
                                "repair the SSH -> WSL runtime chain and rerun prepare"
                            ),
                        ),
                        evidence_ref=self._evidence_ref(context, "prepare", worker),
                    )
                )
                continue
            if not probe.ok:
                results.append(
                    WorkerResult(
                        worker_id=str(worker.worker_id),
                        status=self._failure_status(probe),
                        failure=self._failure_from_result(
                            stage=FailureStage.LAUNCH,
                            worker=worker,
                            result=probe,
                            message="remote prepare path probe failed",
                            recommended_action=(
                                "inspect remote snapshot/output path permissions and rerun prepare"
                            ),
                        ),
                        evidence_ref=self._evidence_ref(context, "prepare", worker),
                    )
                )
                continue
            parsed = self._parse_prepare_probe(probe, worker)
            if parsed is None:
                results.append(
                    WorkerResult(
                        worker_id=str(worker.worker_id),
                        status=LauncherResultStatus.FAILED,
                        failure=self._launcher_failure(
                            stage=FailureStage.LAUNCH,
                            worker=worker,
                            message="prepare probe did not return valid JSON evidence",
                            recommended_action=(
                                "repair the runtime probe output formatting and rerun prepare"
                            ),
                        ),
                        evidence_ref=self._evidence_ref(context, "prepare", worker),
                    )
                )
                continue
            mismatch = self._prepare_probe_failure(context, worker, parsed)
            if mismatch is not None:
                results.append(
                    WorkerResult(
                        worker_id=str(worker.worker_id),
                        status=LauncherResultStatus.FAILED,
                        failure=mismatch,
                        evidence_ref=self._evidence_ref(context, "prepare", worker),
                    )
                )
                continue
            results.append(
                WorkerResult(
                    worker_id=str(worker.worker_id),
                    status=LauncherResultStatus.SUCCESS,
                    evidence_ref=self._evidence_ref(context, "prepare", worker),
                    message=self._prepare_success_message(parsed),
                )
            )
        return LauncherResult.from_worker_results(
            operation=LauncherOperation.PREPARE,
            backend=self.capabilities.backend,
            job_id=str(context.job.job_id),
            worker_results=results,
            next_job_state=JobState.PROBING,
        )

    def distribute(self, context: LauncherContext) -> LauncherResult:
        if context.snapshot is None:
            return LauncherResult(
                operation=LauncherOperation.DISTRIBUTE,
                status=LauncherResultStatus.BLOCKED,
                backend=self.capabilities.backend,
                job_id=str(context.job.job_id),
                blocker="snapshot is required before SSH distribution",
                next_job_state=JobState.DISTRIBUTING,
            )
        snapshot = context.snapshot
        preferred_transport = _load_snapshot_transport_preference(snapshot)
        control_checksum = snapshot_checksum(Path(snapshot.root_path))
        results: list[WorkerResult] = []
        with tempfile.TemporaryDirectory(
            prefix=f"shardgrid-launcher-{context.job.job_id}-"
        ) as tmp_dir:
            archive_path = _build_snapshot_archive(snapshot, Path(tmp_dir))
            for worker in self._selected_workers(context):
                worker_result, distribution = self._distribute_worker(
                    context=context,
                    worker=worker,
                    snapshot=snapshot,
                    archive_path=archive_path,
                    control_checksum=control_checksum,
                    preferred_transport=preferred_transport,
                )
                self._distribution_records[(str(context.job.job_id), str(worker.worker_id))] = (
                    distribution
                )
                self._write_evidence(worker_result.evidence_ref, distribution.to_dict())
                results.append(worker_result)
        return LauncherResult.from_worker_results(
            operation=LauncherOperation.DISTRIBUTE,
            backend=self.capabilities.backend,
            job_id=str(context.job.job_id),
            worker_results=results,
            next_job_state=JobState.DISTRIBUTING,
        )

    def launch(self, context: LauncherContext) -> LauncherResult:
        launch_gate = self._distribution_gate(context)
        if launch_gate is not None:
            return launch_gate
        results: list[WorkerResult] = []
        for assignment, worker in self._selected_assignments(context):
            key = self._process_key(context, str(worker.worker_id), assignment.rank)
            existing = self._process_records.get(key)
            if existing is not None:
                results.append(
                    WorkerResult(
                        worker_id=str(worker.worker_id),
                        status=LauncherResultStatus.NOOP,
                        rank_results=(
                            RankResult(
                                rank=assignment.rank,
                                worker_id=str(worker.worker_id),
                                stage=assignment.stage,
                                pid=existing.pid,
                                log_ref=existing.log_path,
                                evidence_ref=self._evidence_ref(context, "launch", worker),
                                status=LauncherResultStatus.NOOP,
                                message="launch already recorded for job/worker/rank",
                            ),
                        ),
                        evidence_ref=self._evidence_ref(context, "launch", worker),
                        message="launch already recorded for job/worker/rank",
                    )
                )
                continue
            if not assignment.launch_command:
                failure = self._launcher_failure(
                    stage=FailureStage.LAUNCH,
                    worker=worker,
                    worker_id=str(worker.worker_id),
                    command="launch",
                    message="launch command is missing from the execution plan",
                    recommended_action=(
                        "populate assignment.launch_command before invoking SSH launch"
                    ),
                )
                results.append(
                    WorkerResult(
                        worker_id=str(worker.worker_id),
                        status=LauncherResultStatus.BLOCKED,
                        failure=failure,
                        evidence_ref=self._evidence_ref(context, "launch", worker),
                    )
                )
                continue
            failure = self._validate_launch_assignment(context, worker, assignment)
            if failure is not None:
                results.append(
                    WorkerResult(
                        worker_id=str(worker.worker_id),
                        status=LauncherResultStatus.BLOCKED,
                        failure=failure,
                        evidence_ref=self._evidence_ref(context, "launch", worker),
                    )
                )
                continue
            runtime = self._runtime_factory(worker)
            remote_root = self._distribution_records[
                (str(context.job.job_id), str(worker.worker_id))
            ].remote_snapshot_root
            remote_code_root = self._remote_code_root(remote_root)
            log_path = self._launch_log_path(context, worker, assignment, remote_root)
            argv = self._launch_argv(context, worker, assignment, remote_code_root)
            env = self._launch_env(context, assignment, log_path, remote_code_root)
            script = self._spawn_script(
                argv=argv,
                log_path=log_path,
                cwd=remote_code_root,
                env=env,
            )
            try:
                result = runtime.run_script(script, timeout=30)
            except Exception as exc:
                failure = self._launcher_failure(
                    stage=FailureStage.LAUNCH,
                    worker=worker,
                    worker_id=str(worker.worker_id),
                    command=assignment.launch_command,
                    message=f"runtime wrapper failure during launch: {exc}",
                    recommended_action=("repair the SSH -> WSL runtime chain and retry launch"),
                )
                results.append(
                    WorkerResult(
                        worker_id=str(worker.worker_id),
                        status=LauncherResultStatus.FAILED,
                        failure=failure,
                        evidence_ref=self._evidence_ref(context, "launch", worker),
                    )
                )
                continue
            if not result.ok:
                failure = self._failure_from_result(
                    stage=FailureStage.LAUNCH,
                    worker=worker,
                    result=result,
                    message="remote launch command failed",
                    recommended_action=(
                        "inspect SSH stderr and the WSL runtime logs, then retry launch"
                    ),
                )
                results.append(
                    WorkerResult(
                        worker_id=str(worker.worker_id),
                        status=self._failure_status(result),
                        failure=failure,
                        evidence_ref=self._evidence_ref(context, "launch", worker),
                        rank_results=(
                            RankResult(
                                rank=assignment.rank,
                                worker_id=str(worker.worker_id),
                                stage=assignment.stage,
                                status=self._failure_status(result),
                                failure=failure,
                            ),
                        ),
                    )
                )
                continue
            pid = self._parse_pid(result, worker, assignment.rank, assignment.launch_command)
            if pid is None:
                failure = self._launcher_failure(
                    stage=FailureStage.LAUNCH,
                    worker=worker,
                    worker_id=str(worker.worker_id),
                    command=assignment.launch_command,
                    exit_code=result.exit_code,
                    message="remote launch did not return a valid PID",
                    recommended_action=(
                        "ensure the launcher shim prints exactly one numeric PID and retry"
                    ),
                )
                results.append(
                    WorkerResult(
                        worker_id=str(worker.worker_id),
                        status=LauncherResultStatus.FAILED,
                        failure=failure,
                        evidence_ref=self._evidence_ref(context, "launch", worker),
                    )
                )
                continue
            record = SSHProcessRecord(
                job_id=str(context.job.job_id),
                worker_id=str(worker.worker_id),
                rank=assignment.rank,
                pid=pid,
                command_argv=argv,
                log_path=log_path,
                launched_at=_now(),
                remote_host=str(worker.host),
                local_rank=assignment.local_rank,
                stage=assignment.stage,
                remote_root=remote_root,
            )
            self._process_records[key] = record
            results.append(
                WorkerResult(
                    worker_id=str(worker.worker_id),
                    status=LauncherResultStatus.SUCCESS,
                    evidence_ref=self._evidence_ref(context, "launch", worker),
                    rank_results=(
                        RankResult(
                            rank=assignment.rank,
                            worker_id=str(worker.worker_id),
                            stage=assignment.stage,
                            pid=pid,
                            log_ref=log_path,
                            evidence_ref=self._evidence_ref(context, "launch", worker),
                            message="remote process launched",
                        ),
                    ),
                )
            )
        return LauncherResult.from_worker_results(
            operation=LauncherOperation.LAUNCH,
            backend=self.capabilities.backend,
            job_id=str(context.job.job_id),
            worker_results=results,
            next_job_state=JobState.LAUNCHING,
        )

    def monitor(self, context: LauncherContext) -> LauncherResult:
        results: list[WorkerResult] = []
        observed_at = _now()
        statuses: list[dict[str, Any]] = []
        for assignment, worker in self._selected_assignments(context):
            key = self._process_key(context, str(worker.worker_id), assignment.rank)
            record = self._process_records.get(key)
            if record is None:
                evidence_ref = self._monitor_evidence_ref(context, worker, assignment.rank)
                payload = {
                    "observed_at": observed_at,
                    "worker_id": str(worker.worker_id),
                    "rank": assignment.rank,
                    "stage": assignment.stage,
                    "status": "missing",
                    "launcher_status": LauncherResultStatus.NOOP.value,
                    "running": False,
                    "terminal_success": False,
                    "timeout_stage": None,
                    "rendezvous_ready": False,
                    "training_started": False,
                    "last_progress": "missing",
                    "loss_history": [],
                    "message": "no tracked process for this job/worker/rank",
                }
                self._write_evidence(evidence_ref, payload)
                statuses.append(payload)
                results.append(
                    WorkerResult(
                        worker_id=str(worker.worker_id),
                        status=LauncherResultStatus.NOOP,
                        rank_results=(
                            RankResult(
                                rank=assignment.rank,
                                worker_id=str(worker.worker_id),
                                stage=assignment.stage,
                                evidence_ref=evidence_ref,
                                status=LauncherResultStatus.NOOP,
                                message="no tracked process for this job/worker/rank",
                            ),
                        ),
                        evidence_ref=evidence_ref,
                    )
                )
                continue
            runtime = self._runtime_factory(worker)
            evidence_ref = self._monitor_evidence_ref(context, worker, assignment.rank)
            previous_payload = self._read_evidence(evidence_ref)
            try:
                process_probe = self._probe_process_liveness(runtime, record.pid)
                log_result = runtime.run(
                    ["tail", "-n", "200", record.log_path],
                    timeout=self._remote_command_timeout_seconds(),
                )
            except Exception as exc:
                failure = self._launcher_failure(
                    stage=FailureStage.LAUNCH,
                    worker=worker,
                    worker_id=str(worker.worker_id),
                    command=f"monitor rank={assignment.rank} pid={record.pid}",
                    message=f"monitor lost connection to remote runtime: {exc}",
                    recommended_action="restore SSH/WSL runtime access, then retry monitor",
                )
                payload = {
                    "observed_at": observed_at,
                    "worker_id": str(worker.worker_id),
                    "rank": assignment.rank,
                    "stage": assignment.stage,
                    "pid": record.pid,
                    "status": "connection_lost",
                    "launcher_status": LauncherResultStatus.BLOCKED.value,
                    "running": False,
                    "terminal_success": False,
                    "timeout_stage": None,
                    "rendezvous_ready": False,
                    "training_started": False,
                    "last_progress": "connection_lost",
                    "loss_history": [],
                    "message": failure.message,
                    "failure": failure.to_dict(),
                }
                self._write_evidence(evidence_ref, payload)
                statuses.append(payload)
                results.append(
                    WorkerResult(
                        worker_id=str(worker.worker_id),
                        status=LauncherResultStatus.BLOCKED,
                        failure=failure,
                        evidence_ref=evidence_ref,
                        rank_results=(
                            RankResult(
                                rank=assignment.rank,
                                worker_id=str(worker.worker_id),
                                stage=assignment.stage,
                                pid=record.pid,
                                log_ref=record.log_path,
                                evidence_ref=evidence_ref,
                                status=LauncherResultStatus.BLOCKED,
                                failure=failure,
                                message="monitor lost connection",
                            ),
                        ),
                    )
                )
                continue
            if not log_result.ok:
                process_state = process_probe.state
                if process_state != "exited":
                    payload = self._transient_monitor_payload(
                        context,
                        worker=worker,
                        assignment=assignment,
                        record=record,
                        process_probe=process_probe,
                        log_result=log_result,
                        previous_payload=previous_payload,
                        observed_at=observed_at,
                    )
                    self._write_evidence(evidence_ref, payload)
                    statuses.append(payload)
                    rank_status = self._rank_launcher_status(payload)
                    payload["launcher_status"] = rank_status.value
                    results.append(
                        WorkerResult(
                            worker_id=str(worker.worker_id),
                            status=rank_status,
                            evidence_ref=evidence_ref,
                            rank_results=(
                                RankResult(
                                    rank=assignment.rank,
                                    worker_id=str(worker.worker_id),
                                    stage=assignment.stage,
                                    pid=record.pid,
                                    log_ref=record.log_path,
                                    evidence_ref=evidence_ref,
                                    status=rank_status,
                                    message=str(payload["message"]),
                                ),
                            ),
                        )
                    )
                    continue
                failure = self._failure_from_result(
                    stage=FailureStage.TRAIN,
                    worker=worker,
                    result=log_result,
                    message="remote rank log read failed during monitor",
                    recommended_action="inspect remote log path permissions and retry monitor",
                )
                payload = {
                    "observed_at": observed_at,
                    "worker_id": str(worker.worker_id),
                    "rank": assignment.rank,
                    "stage": assignment.stage,
                    "pid": record.pid,
                    "status": "log_unavailable",
                    "launcher_status": self._failure_status(log_result).value,
                    "running": False,
                    "terminal_success": False,
                    "timeout_stage": None,
                    "rendezvous_ready": False,
                    "training_started": False,
                    "last_progress": "log_unavailable",
                    "loss_history": [],
                    "message": failure.message,
                    "failure": failure.to_dict(),
                }
                self._write_evidence(evidence_ref, payload)
                statuses.append(payload)
                results.append(
                    WorkerResult(
                        worker_id=str(worker.worker_id),
                        status=self._failure_status(log_result),
                        failure=failure,
                        evidence_ref=evidence_ref,
                        rank_results=(
                            RankResult(
                                rank=assignment.rank,
                                worker_id=str(worker.worker_id),
                                stage=assignment.stage,
                                pid=record.pid,
                                log_ref=record.log_path,
                                evidence_ref=evidence_ref,
                                status=self._failure_status(log_result),
                                failure=failure,
                                message="log unavailable",
                            ),
                        ),
                    )
                )
                continue
            payload = self._rank_monitor_payload(
                context,
                worker=worker,
                assignment=assignment,
                record=record,
                process_probe=process_probe,
                log_result=log_result,
                previous_payload=previous_payload,
                observed_at=observed_at,
            )
            self._write_evidence(evidence_ref, payload)
            statuses.append(payload)
            rank_status = self._rank_launcher_status(payload)
            payload["launcher_status"] = rank_status.value
            failure = self._rank_failure(context, worker, assignment, record, payload)
            results.append(
                WorkerResult(
                    worker_id=str(worker.worker_id),
                    status=rank_status,
                    rank_results=(
                        RankResult(
                            rank=assignment.rank,
                            worker_id=str(worker.worker_id),
                            stage=assignment.stage,
                            pid=record.pid,
                            log_ref=record.log_path,
                            evidence_ref=evidence_ref,
                            status=rank_status,
                            failure=failure,
                            message=str(payload["message"]),
                        ),
                    ),
                    evidence_ref=evidence_ref,
                    failure=failure,
                )
            )
        persisted = self._persist_monitored_status(context, statuses)
        return LauncherResult.from_worker_results(
            operation=LauncherOperation.MONITOR,
            backend=self.capabilities.backend,
            job_id=str(context.job.job_id),
            worker_results=results,
            next_job_state=persisted.state,
        )

    def logs(self, context: LauncherContext) -> LauncherResult:
        worker_selector = (context.backend_config.get("worker_id") or "").strip()
        rank_selector = self._parse_rank_selector(context.backend_config.get("rank"))
        tail_lines = self._parse_tail_lines(context.backend_config.get("tail"))
        selected = self._selected_log_assignments(
            context,
            worker_selector=worker_selector,
            rank_selector=rank_selector,
        )
        if not selected:
            selector = []
            if worker_selector:
                selector.append(f"worker={worker_selector}")
            if rank_selector is not None:
                selector.append(f"rank={rank_selector}")
            message = (
                f"log selector not found for job {context.job.job_id}: {', '.join(selector)}"
                if selector
                else f"no log assignments recorded for job {context.job.job_id}"
            )
            return LauncherResult(
                operation=LauncherOperation.LOGS,
                status=LauncherResultStatus.FAILED,
                backend=self.capabilities.backend,
                job_id=str(context.job.job_id),
                blocker=message,
                message=message,
                next_job_state=context.job_status.state if context.job_status else None,
            )
        log_results: list[LogResult] = []
        for assignment, worker in selected:
            local_results = self._local_log_results(context, assignment, tail_lines)
            if local_results:
                log_results.extend(local_results)
                continue
            remote_path = self._remote_log_path(context, worker, assignment)
            if remote_path is None:
                log_results.append(
                    LogResult(
                        job_id=str(context.job.job_id),
                        worker_id=str(worker.worker_id),
                        rank=assignment.rank,
                        stage=assignment.stage,
                        stream="combined",
                        source="MISSING",
                        status=LauncherResultStatus.NOOP,
                        message="local log unavailable and no remote log reference was recorded",
                    )
                )
                continue
            runtime = self._runtime_factory(worker)
            try:
                result = runtime.run(
                    ["tail", "-n", str(tail_lines), remote_path],
                    timeout=self._remote_command_timeout_seconds(),
                )
            except Exception as exc:
                failure = self._launcher_failure(
                    stage=FailureStage.TRAIN,
                    worker=worker,
                    worker_id=str(worker.worker_id),
                    command=f"tail -n {tail_lines} {remote_path}",
                    message=f"remote log retrieval failed: {exc}",
                    recommended_action="restore SSH/WSL access and retry log retrieval",
                )
                log_results.append(
                    LogResult(
                        job_id=str(context.job.job_id),
                        worker_id=str(worker.worker_id),
                        rank=assignment.rank,
                        stage=assignment.stage,
                        stream="combined",
                        source="FAILED",
                        location="remote",
                        source_path=redact_text(remote_path, self._secrets) or remote_path,
                        status=LauncherResultStatus.FAILED,
                        failure=failure,
                        message=failure.message,
                    )
                )
                continue
            if not result.ok:
                if self._is_missing_log_error(result.stderr or result.stdout):
                    log_results.append(
                        LogResult(
                            job_id=str(context.job.job_id),
                            worker_id=str(worker.worker_id),
                            rank=assignment.rank,
                            stage=assignment.stage,
                            stream="combined",
                            source="MISSING",
                            location="remote",
                            source_path=redact_text(remote_path, self._secrets) or remote_path,
                            status=LauncherResultStatus.NOOP,
                            message="local and remote logs are unavailable for this rank",
                        )
                    )
                    continue
                failure = self._failure_from_result(
                    stage=FailureStage.TRAIN,
                    worker=worker,
                    result=result,
                    message="remote log read failed",
                    recommended_action="inspect the remote log path and retry logs",
                )
                log_results.append(
                    LogResult(
                        job_id=str(context.job.job_id),
                        worker_id=str(worker.worker_id),
                        rank=assignment.rank,
                        stage=assignment.stage,
                        stream="combined",
                        source="FAILED",
                        location="remote",
                        source_path=redact_text(remote_path, self._secrets) or remote_path,
                        status=self._failure_status(result),
                        failure=failure,
                        message=failure.message,
                    )
                )
                continue
            log_results.append(
                LogResult(
                    job_id=str(context.job.job_id),
                    worker_id=str(worker.worker_id),
                    rank=assignment.rank,
                    stage=assignment.stage,
                    stream="combined",
                    source="REMOTE_FALLBACK",
                    location="remote",
                    source_path=redact_text(remote_path, self._secrets) or remote_path,
                    tail=redact_text(result.stdout, self._secrets) or "",
                    status=LauncherResultStatus.SUCCESS,
                )
            )
        return LauncherResult(
            operation=LauncherOperation.LOGS,
            status=self._log_result_status(log_results),
            backend=self.capabilities.backend,
            job_id=str(context.job.job_id),
            log_results=tuple(log_results),
            next_job_state=context.job_status.state if context.job_status else None,
        )

    def stop(self, context: LauncherContext) -> LauncherResult:
        current = self._load_job_status(context)
        results: list[WorkerResult] = []
        stop_payloads: list[dict[str, Any]] = []
        signal_sent = False
        for assignment, worker in self._selected_assignments(context):
            record = self._tracked_process_record(
                context,
                current=current,
                worker=worker,
                assignment=assignment,
            )
            evidence_ref = self._stop_evidence_ref(context, worker, assignment.rank)
            if record is None:
                payload = self._missing_stop_payload(
                    worker_id=str(worker.worker_id),
                    rank=assignment.rank,
                    stage=assignment.stage,
                )
                self._write_evidence(evidence_ref, payload)
                stop_payloads.append(payload)
                results.append(
                    WorkerResult(
                        worker_id=str(worker.worker_id),
                        status=LauncherResultStatus.NOOP,
                        evidence_ref=evidence_ref,
                        rank_results=(
                            RankResult(
                                rank=assignment.rank,
                                worker_id=str(worker.worker_id),
                                stage=assignment.stage,
                                evidence_ref=evidence_ref,
                                status=LauncherResultStatus.NOOP,
                                message="no tracked process for this job/worker/rank",
                            ),
                        ),
                    )
                )
                continue
            worker_result, payload, changed = self._stop_rank(
                context,
                current=current,
                worker=worker,
                assignment=assignment,
                record=record,
                evidence_ref=evidence_ref,
            )
            signal_sent = signal_sent or changed
            stop_payloads.append(payload)
            results.append(worker_result)
        persisted = self._persist_stopped_status(
            context,
            current=current,
            payloads=stop_payloads,
            signal_sent=signal_sent,
        )
        return LauncherResult.from_worker_results(
            operation=LauncherOperation.STOP,
            backend=self.capabilities.backend,
            job_id=str(context.job.job_id),
            worker_results=results,
            next_job_state=persisted.state,
        )

    def cleanup(self, context: LauncherContext) -> LauncherResult:
        current = self._load_job_status(context)
        worker_results: list[WorkerResult] = []
        for worker in self._selected_workers(context):
            worker_results.append(self._cleanup_worker(context, current=current, worker=worker))
        self._save_job_status(
            context,
            replace(
                current,
                assignments=[
                    self._assignment_with_runtime_pid(context, assignment)
                    for assignment in context.execution_plan.workers
                ],
            ),
        )
        return LauncherResult.from_worker_results(
            operation=LauncherOperation.CLEANUP,
            backend=self.capabilities.backend,
            job_id=str(context.job.job_id),
            worker_results=worker_results,
            next_job_state=current.state,
        )

    def process_records(self) -> tuple[SSHProcessRecord, ...]:
        return tuple(
            sorted(
                self._process_records.values(),
                key=lambda item: (item.job_id, item.rank, item.worker_id),
            )
        )

    def _selected_workers(self, context: LauncherContext) -> list[WorkerConfig]:
        worker_ids = {str(assignment.worker_id) for assignment in context.execution_plan.workers}
        workers = [
            worker for worker in self.cluster_config.workers if str(worker.worker_id) in worker_ids
        ]
        if len(workers) != len(worker_ids):
            missing = sorted(worker_ids - {str(worker.worker_id) for worker in workers})
            raise KeyError(f"unknown worker config(s): {', '.join(missing)}")
        return sorted(workers, key=lambda worker: str(worker.worker_id))

    def _selected_assignments(self, context: LauncherContext) -> list[tuple[Any, WorkerConfig]]:
        workers = {str(worker.worker_id): worker for worker in self._selected_workers(context)}
        return [
            (assignment, workers[str(assignment.worker_id)])
            for assignment in sorted(
                context.execution_plan.workers,
                key=lambda item: (item.rank, str(item.worker_id)),
            )
        ]

    def _selected_log_assignments(
        self,
        context: LauncherContext,
        *,
        worker_selector: str,
        rank_selector: int | None,
    ) -> list[tuple[Any, WorkerConfig]]:
        assignments = (
            context.job_status.assignments
            if context.job_status is not None and context.job_status.assignments
            else context.execution_plan.workers
        )
        workers = {str(worker.worker_id): worker for worker in self.cluster_config.workers}
        selected: list[tuple[Any, WorkerConfig]] = []
        for assignment in sorted(assignments, key=lambda item: (item.rank, str(item.worker_id))):
            worker_id = str(assignment.worker_id)
            if worker_selector and worker_id != worker_selector:
                continue
            if rank_selector is not None and assignment.rank != rank_selector:
                continue
            worker = workers.get(worker_id)
            if worker is None:
                raise KeyError(f"unknown worker config: {worker_id}")
            selected.append((assignment, worker))
        return selected

    def _parse_rank_selector(self, value: str | None) -> int | None:
        if value is None or not value.strip():
            return None
        return int(value)

    def _parse_tail_lines(self, value: str | None) -> int:
        if value is None or not value.strip():
            return 50
        parsed = int(value)
        if parsed <= 0:
            raise ValueError("tail must be > 0")
        return parsed

    def _local_log_results(
        self,
        context: LauncherContext,
        assignment,
        tail_lines: int,
    ) -> list[LogResult]:
        if context.snapshot is None:
            return []
        log_dir = self._local_log_dir(context, assignment)
        if not log_dir.is_dir():
            return []
        preferred = [
            ("stdout", log_dir / "stdout.log"),
            ("stderr", log_dir / "stderr.log"),
            ("combined", log_dir / f"rank{assignment.rank}.log"),
            ("combined", log_dir / "combined.log"),
        ]
        seen: set[Path] = set()
        files: list[tuple[str, Path]] = []
        for stream, path in preferred:
            if path.is_file() and path not in seen:
                files.append((stream, path))
                seen.add(path)
        if not files:
            for path in sorted(log_dir.glob("*.log")):
                if path in seen or not path.is_file():
                    continue
                stream = path.stem if path.stem in {"stdout", "stderr"} else "combined"
                files.append((stream, path))
                seen.add(path)
        results: list[LogResult] = []
        for stream, path in files:
            text = path.read_text(encoding="utf-8")
            content = redact_text(self._tail_text(text, tail_lines), self._secrets) or ""
            source_path = redact_text(str(path), self._secrets) or str(path)
            results.append(
                LogResult(
                    job_id=str(context.job.job_id),
                    worker_id=str(assignment.worker_id),
                    rank=assignment.rank,
                    stage=assignment.stage,
                    stream=stream,
                    source="LOCAL",
                    location="local",
                    source_path=source_path,
                    tail=content,
                    status=LauncherResultStatus.SUCCESS,
                )
            )
        return results

    def _remote_log_path(
        self,
        context: LauncherContext,
        worker: WorkerConfig,
        assignment,
    ) -> str | None:
        key = self._process_key(context, str(worker.worker_id), assignment.rank)
        record = self._process_records.get(key)
        if record is not None:
            return record.log_path
        if assignment.log_path:
            return self._launch_log_path(
                context,
                worker,
                assignment,
                self._remote_job_root(context),
            )
        return None

    def _local_log_dir(self, context: LauncherContext, assignment) -> Path:
        assert context.snapshot is not None
        identity = (
            f"rank{assignment.rank}"
            if assignment.stage is None
            else f"rank{assignment.rank}-{assignment.stage}"
        )
        return Path(context.snapshot.logs_path) / str(assignment.worker_id) / identity

    def _tail_text(self, text: str, tail_lines: int) -> str:
        lines = text.splitlines()
        if not lines:
            return ""
        return "\n".join(lines[-tail_lines:])

    def _is_missing_log_error(self, text: str) -> bool:
        lowered = text.lower()
        return any(marker in lowered for marker in ("no such file", "not found", "cannot open"))

    def _log_result_status(self, log_results: list[LogResult]) -> LauncherResultStatus:
        if not log_results:
            return LauncherResultStatus.NOOP
        statuses = {item.status for item in log_results}
        if statuses == {LauncherResultStatus.SUCCESS}:
            return LauncherResultStatus.SUCCESS
        if statuses == {LauncherResultStatus.NOOP}:
            return LauncherResultStatus.NOOP
        if any(
            status in statuses
            for status in (LauncherResultStatus.SUCCESS, LauncherResultStatus.NOOP)
        ) and any(
            status in statuses
            for status in (
                LauncherResultStatus.FAILED,
                LauncherResultStatus.BLOCKED,
                LauncherResultStatus.UNSUPPORTED,
                LauncherResultStatus.PARTIAL,
            )
        ):
            return LauncherResultStatus.PARTIAL
        if statuses <= {LauncherResultStatus.SUCCESS, LauncherResultStatus.NOOP}:
            return LauncherResultStatus.PARTIAL
        if LauncherResultStatus.BLOCKED in statuses:
            return LauncherResultStatus.BLOCKED
        if LauncherResultStatus.FAILED in statuses:
            return LauncherResultStatus.FAILED
        if LauncherResultStatus.UNSUPPORTED in statuses:
            return LauncherResultStatus.UNSUPPORTED
        return LauncherResultStatus.PARTIAL

    def _default_ssh_factory(self, worker: WorkerConfig) -> SSHTransport:
        return SSHTransport(
            SSHOptions.from_ssh_config(
                self.cluster_config.ssh,
                host=str(worker.host),
                user=worker.ssh_user,
                port=worker.ssh_port,
            )
        )

    def _default_runtime_factory(self, worker: WorkerConfig) -> WSLRuntimeWrapper:
        return WSLRuntimeWrapper(
            WSLRuntimeConfig.from_worker_and_runtime(worker, self.cluster_config.runtime),
            self._ssh_factory(worker),
        )

    def _default_transport_factory(self, worker: WorkerConfig) -> ArtifactTransport:
        return select_artifact_transport(build_transport_config("auto"))

    def _python_executable(self, worker: WorkerConfig) -> str:
        if worker.conda_prefix:
            return worker.conda_prefix + "/bin/python"
        return self.cluster_config.runtime.python_executable

    def _remote_job_root(self, context: LauncherContext) -> str:
        return str(PurePosixPath(str(self.cluster_config.jobs_root), str(context.job.job_id)))

    def _runtime_refs_for_worker(
        self,
        context: LauncherContext,
        worker_id: str,
    ) -> tuple[str, ...]:
        refs: list[str] = []
        for assignment in context.execution_plan.workers:
            if str(assignment.worker_id) != worker_id:
                continue
            ref = context.runtime_environment_refs.get(str(assignment.rank))
            if ref:
                refs.append(ref)
        return tuple(refs)

    def _validate_local_entrypoints(
        self,
        context: LauncherContext,
    ) -> dict[str, tuple[str, ...]]:
        if context.snapshot is None:
            raise ValueError("snapshot is required before SSH prepare")
        root = Path(context.snapshot.root_path)
        code_root = Path(context.snapshot.code_path)
        entrypoints: dict[str, list[str]] = {}
        for assignment in context.execution_plan.workers:
            entrypoint = self._entrypoint_from_assignment(assignment.launch_command)
            if entrypoint is None:
                raise ValueError(
                    "launch command is missing an executable entry point "
                    f"for rank {assignment.rank}"
                )
            candidates = (code_root / entrypoint, root / entrypoint)
            if not any(path.is_file() for path in candidates):
                raise ValueError(
                    f"snapshot entry point is missing for rank {assignment.rank}: {entrypoint}"
                )
            entrypoints.setdefault(str(assignment.worker_id), []).append(entrypoint)
        return {worker_id: tuple(dict.fromkeys(items)) for worker_id, items in entrypoints.items()}

    def _entrypoint_from_assignment(self, launch_command: str | None) -> str | None:
        if not launch_command:
            return None
        argv = shlex.split(launch_command)
        if not argv:
            return None
        if "python" in Path(argv[0]).name.lower():
            for item in argv[1:]:
                if item.startswith("-"):
                    continue
                return item
            return None
        return argv[0]

    def _prepare_probe_script(
        self,
        *,
        remote_root: str,
        job_id: str,
        entrypoints: tuple[str, ...],
        expected_conda_prefix: str | None,
    ) -> str:
        payload = {
            "remote_root": remote_root,
            "job_id": job_id,
            "entrypoints": list(entrypoints),
            "output_dirs": list(_PREPARE_OUTPUT_DIRS),
            "expected_conda_prefix": expected_conda_prefix,
        }
        return (
            "import json, platform, sys\n"
            "from pathlib import Path\n"
            f"payload = json.loads({json.dumps(json.dumps(payload))})\n"
            "root = Path(payload['remote_root'])\n"
            "root.mkdir(parents=True, exist_ok=True)\n"
            "python_executable = sys.executable\n"
            "expected_prefix = payload.get('expected_conda_prefix')\n"
            "python_under_expected_prefix = True\n"
            "if expected_prefix:\n"
            "    prefix = str(Path(expected_prefix).resolve())\n"
            "    actual = str(Path(python_executable).resolve())\n"
            "    python_under_expected_prefix = actual.startswith(prefix.rstrip('/') + '/')\n"
            "created = []\n"
            "for name in payload['output_dirs']:\n"
            "    path = root / name\n"
            "    before = path.exists()\n"
            "    path.mkdir(parents=True, exist_ok=True)\n"
            "    if not before:\n"
            "        created.append(name)\n"
            "metadata_path = root / 'diagnostics' / 'snapshot-metadata.json'\n"
            "snapshot_present = metadata_path.exists()\n"
            "metadata_job_id = None\n"
            "entrypoint_exists = False\n"
            "if snapshot_present:\n"
            "    metadata = json.loads(metadata_path.read_text())\n"
            "    metadata_job_id = metadata.get('job_id')\n"
            "    entrypoint_exists = all(\n"
            "        (root / 'code' / item).exists()\n"
            "        for item in payload['entrypoints']\n"
            "    )\n"
            "print(json.dumps({\n"
            "    'python_executable': python_executable,\n"
            "    'python_version': 'Python ' + platform.python_version(),\n"
            "    'python_under_expected_prefix': python_under_expected_prefix,\n"
            "    'remote_root': str(root),\n"
            "    'snapshot_present': snapshot_present,\n"
            "    'metadata_job_id': metadata_job_id,\n"
            "    'entrypoint_exists': entrypoint_exists,\n"
            "    'created_dirs': created,\n"
            "    'output_dirs_ready': all(\n"
            "        (root / item).is_dir()\n"
            "        for item in payload['output_dirs']\n"
            "    ),\n"
            "}))\n"
        )

    def _parse_prepare_probe(
        self,
        result: ProcessResult,
        worker: WorkerConfig,
    ) -> dict[str, Any] | None:
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        return payload

    def _prepare_probe_failure(
        self,
        context: LauncherContext,
        worker: WorkerConfig,
        payload: dict[str, Any],
    ) -> FailureRecord | None:
        if payload.get("python_under_expected_prefix") is not True:
            return self._launcher_failure(
                stage=FailureStage.LAUNCH,
                worker=worker,
                command="ssh prepare runtime_prepare_script",
                message="selected runtime Python mismatch during SSH prepare",
                recommended_action=(
                    "fix the selected WSL Conda prefix so prepare uses the same "
                    "runtime proven by formal probe"
                ),
            )
        if not payload.get("output_dirs_ready"):
            return self._launcher_failure(
                stage=FailureStage.LAUNCH,
                worker=worker,
                command="ssh prepare output_dirs",
                message="prepare probe could not create required remote output directories",
                recommended_action=(
                    "repair remote write permissions for the job output paths and rerun prepare"
                ),
            )
        if payload.get("snapshot_present"):
            if payload.get("metadata_job_id") != str(context.job.job_id):
                return self._launcher_failure(
                    stage=FailureStage.LAUNCH,
                    worker=worker,
                    command="ssh prepare snapshot_identity",
                    message="remote snapshot metadata belongs to a different job identity",
                    recommended_action=(
                        "remove the conflicting remote snapshot or use a new "
                        "job_id, then rerun prepare"
                    ),
                )
            if not payload.get("entrypoint_exists"):
                return self._launcher_failure(
                    stage=FailureStage.LAUNCH,
                    worker=worker,
                    command="ssh prepare entrypoint",
                    message="remote snapshot is missing the training entry point",
                    recommended_action=(
                        "repair or redistribute the immutable snapshot before launch"
                    ),
                )
        return None

    def _prepare_success_message(self, payload: dict[str, Any]) -> str:
        if payload.get("snapshot_present"):
            return "runtime ready; remote snapshot identity verified"
        created = payload.get("created_dirs") or []
        if created:
            return "runtime ready; remote output paths prepared; snapshot pending distribution"
        return "runtime ready; output paths already prepared; snapshot pending distribution"

    def _process_key(
        self,
        context: LauncherContext,
        worker_id: str,
        rank: int,
    ) -> tuple[str, str, int]:
        return (str(context.job.job_id), worker_id, rank)

    def _spawn_script(
        self,
        *,
        argv: Sequence[str],
        log_path: str,
        cwd: str,
        env: dict[str, str],
    ) -> str:
        payload = {
            "argv": list(argv),
            "log_path": log_path,
            "cwd": cwd,
            "env": env,
        }
        return (
            "import json, os, pathlib, subprocess, sys\n"
            f"payload = json.loads({json.dumps(json.dumps(payload))})\n"
            "log_path = pathlib.Path(payload['log_path'])\n"
            "log_path.parent.mkdir(parents=True, exist_ok=True)\n"
            "cwd = payload['cwd']\n"
            "env = os.environ.copy()\n"
            "env.update(payload['env'])\n"
            "with log_path.open('ab', buffering=0) as handle:\n"
            "    proc = subprocess.Popen(\n"
            "        payload['argv'],\n"
            "        cwd=cwd,\n"
            "        env=env,\n"
            "        stdout=handle,\n"
            "        stderr=handle,\n"
            "        stdin=subprocess.DEVNULL,\n"
            "        start_new_session=True,\n"
            "    )\n"
            "    sys.stdout.write(str(proc.pid))\n"
        )

    def _parse_pid(
        self,
        result: ProcessResult,
        worker: WorkerConfig,
        rank: int,
        command: str,
    ) -> int | None:
        try:
            return int((result.stdout or "").strip())
        except (TypeError, ValueError):
            return None

    def _failure_from_result(
        self,
        *,
        stage: FailureStage,
        worker: WorkerConfig,
        result: ProcessResult,
        message: str,
        recommended_action: str,
    ) -> FailureRecord:
        return make_failure_record(
            stage=stage,
            host=str(worker.host),
            worker_id=str(worker.worker_id),
            command=result.recorded_command,
            exit_code=result.exit_code,
            runtime_environment=dict(result.runtime_environment),
            python_executable=self._python_executable(worker),
            conda_environment=worker.conda_environment,
            conda_prefix=worker.conda_prefix,
            message=message,
            recommended_action=recommended_action,
            retryable=not self._is_auth_problem(result.stderr),
            manual_action_required=self._is_auth_problem(result.stderr),
            secrets=self._secrets,
        )

    def _launcher_failure(
        self,
        *,
        stage: FailureStage,
        worker: WorkerConfig,
        message: str,
        recommended_action: str,
        worker_id: str | None = None,
        command: str | None = None,
        exit_code: int | None = None,
    ) -> FailureRecord:
        return make_failure_record(
            stage=stage,
            host=str(worker.host),
            worker_id=worker_id or str(worker.worker_id),
            command=command,
            exit_code=exit_code,
            python_executable=self._python_executable(worker),
            conda_environment=worker.conda_environment,
            conda_prefix=worker.conda_prefix,
            message=message,
            recommended_action=recommended_action,
            secrets=self._secrets,
        )

    def _transport_failure(
        self,
        worker: WorkerConfig,
        item: ArtifactTransferItemResult,
    ) -> FailureRecord:
        return make_failure_record(
            stage=FailureStage.DISTRIBUTE,
            host=str(worker.host),
            worker_id=str(worker.worker_id),
            command=item.recorded_command,
            exit_code=item.exit_code,
            conda_environment=worker.conda_environment,
            conda_prefix=worker.conda_prefix,
            python_executable=self._python_executable(worker),
            message="artifact transport failed before launch preparation completed",
            recommended_action="inspect transport stderr and remote write permissions, then retry",
            retryable=item.retryable,
            manual_action_required=self._is_auth_problem(item.stderr),
            secrets=self._secrets,
        )

    def _distribute_worker(
        self,
        *,
        context: LauncherContext,
        worker: WorkerConfig,
        snapshot,
        archive_path: Path,
        control_checksum: str,
        preferred_transport: str,
    ) -> tuple[WorkerResult, WorkerDistributionResult]:
        attempts: list[WorkerDistributionResult] = []
        max_attempts = 2
        for attempt in range(1, max_attempts + 1):
            distribution = distribute_job_snapshot_to_worker(
                snapshot,
                archive_path=archive_path,
                control_checksum=control_checksum,
                cluster_config=self.cluster_config,
                worker=worker,
                preferred_transport=preferred_transport,
                secrets=self._secrets,
                transport=self._artifact_transport,
                ssh=self._ssh_factory(worker),
                runtime=self._runtime_factory(worker),
            )
            attempts.append(distribution)
            if (
                distribution.status is DistributionStatus.PASS
                or distribution.failure is None
                or not distribution.failure.retryable
                or attempt == max_attempts
            ):
                break
        final = attempts[-1]
        status = self._distribution_status(final)
        message = self._distribution_message(final, len(attempts))
        worker_result = WorkerResult(
            worker_id=str(worker.worker_id),
            status=status,
            failure=final.failure,
            evidence_ref=self._evidence_ref(context, "distribute", worker),
            message=message,
        )
        return worker_result, final

    def _distribution_status(
        self,
        result: WorkerDistributionResult,
    ) -> LauncherResultStatus:
        if result.status is DistributionStatus.BLOCKED:
            return LauncherResultStatus.BLOCKED
        if result.status is DistributionStatus.PASS and result.skipped:
            return LauncherResultStatus.NOOP
        if result.status is DistributionStatus.PASS:
            return LauncherResultStatus.SUCCESS
        return LauncherResultStatus.FAILED

    def _distribution_message(
        self,
        result: WorkerDistributionResult,
        attempts: int,
    ) -> str:
        if result.status is DistributionStatus.PASS and result.skipped:
            return "remote snapshot already verified"
        if result.status is DistributionStatus.PASS and attempts > 1:
            return f"remote snapshot verified after {attempts} attempts"
        if result.status is DistributionStatus.PASS:
            return "remote snapshot transferred and verified"
        return ""

    def _distribution_gate(self, context: LauncherContext) -> LauncherResult | None:
        results: list[WorkerResult] = []
        missing = False
        for worker in self._selected_workers(context):
            distribution = self._distribution_records.get(
                (str(context.job.job_id), str(worker.worker_id))
            )
            if distribution and distribution.status is DistributionStatus.PASS:
                continue
            missing = True
            if distribution is not None and distribution.failure is not None:
                failure = distribution.failure
            else:
                failure = self._launcher_failure(
                    stage=FailureStage.DISTRIBUTE,
                    worker=worker,
                    message="remote snapshot has not been verified for launch",
                    recommended_action=(
                        "run launcher distribute and confirm checksum verification "
                        "for every selected worker before launch"
                    ),
                )
            results.append(
                WorkerResult(
                    worker_id=str(worker.worker_id),
                    status=LauncherResultStatus.BLOCKED,
                    failure=failure,
                    evidence_ref=self._evidence_ref(context, "distribute", worker),
                )
            )
        if not missing:
            return None
        return LauncherResult.from_worker_results(
            operation=LauncherOperation.LAUNCH,
            backend=self.capabilities.backend,
            job_id=str(context.job.job_id),
            worker_results=results,
            next_job_state=JobState.LAUNCHING,
        )

    def _monitor_evidence_ref(
        self,
        context: LauncherContext,
        worker: WorkerConfig,
        rank: int,
    ) -> str:
        base = context.snapshot.diagnostics_path if context.snapshot else "diagnostics"
        return str(PurePosixPath(base) / f"monitor-{worker.worker_id}-rank{rank}.json")

    def _stop_evidence_ref(
        self,
        context: LauncherContext,
        worker: WorkerConfig,
        rank: int,
    ) -> str:
        base = context.snapshot.diagnostics_path if context.snapshot else "diagnostics"
        return str(PurePosixPath(base) / f"stop-{worker.worker_id}-rank{rank}.json")

    def _missing_stop_payload(
        self,
        *,
        worker_id: str,
        rank: int,
        stage: str | None,
    ) -> dict[str, Any]:
        return {
            "observed_at": _now(),
            "worker_id": worker_id,
            "rank": rank,
            "stage": stage,
            "pid": None,
            "initial_state": "missing",
            "action": "noop",
            "escalation_level": 0,
            "final_state": "missing",
            "exit_evidence": "no tracked process",
            "elapsed_seconds": 0.0,
            "launcher_status": LauncherResultStatus.NOOP.value,
            "message": "no tracked process for this job/worker/rank",
        }

    def _stop_rank(
        self,
        context: LauncherContext,
        *,
        current: JobStatus,
        worker: WorkerConfig,
        assignment,
        record: SSHProcessRecord,
        evidence_ref: str,
    ) -> tuple[WorkerResult, dict[str, Any], bool]:
        runtime = self._runtime_factory(worker)
        observed_at = _now()
        try:
            process_probe = self._probe_process_liveness(runtime, record.pid)
        except Exception as exc:
            failure = self._launcher_failure(
                stage=FailureStage.STOP,
                worker=worker,
                worker_id=str(worker.worker_id),
                command=f"stop rank={assignment.rank} pid={record.pid}",
                message=f"stop lost connection to remote runtime: {exc}",
                recommended_action="restore SSH/WSL runtime access, then retry stop",
            )
            payload = self._stop_failure_payload(
                worker_id=str(worker.worker_id),
                rank=assignment.rank,
                stage=assignment.stage,
                pid=record.pid,
                observed_at=observed_at,
                initial_state=record.status,
                final_state="unknown",
                action="probe_failed",
                exit_evidence="initial liveness probe unavailable",
                failure=failure,
            )
            self._write_evidence(evidence_ref, payload)
            return (
                WorkerResult(
                    worker_id=str(worker.worker_id),
                    status=LauncherResultStatus.BLOCKED,
                    failure=failure,
                    evidence_ref=evidence_ref,
                    rank_results=(
                        RankResult(
                            rank=assignment.rank,
                            worker_id=str(worker.worker_id),
                            stage=assignment.stage,
                            pid=record.pid,
                            log_ref=record.log_path,
                            evidence_ref=evidence_ref,
                            status=LauncherResultStatus.BLOCKED,
                            failure=failure,
                            message=failure.message,
                        ),
                    ),
                ),
                payload,
                False,
            )
        log_result = self._safe_tail_log(runtime, record.log_path)
        monitor_payload = self._rank_monitor_payload(
            context,
            worker=worker,
            assignment=assignment,
            record=record,
            process_probe=process_probe,
            log_result=log_result,
            previous_payload=self._read_evidence(
                self._monitor_evidence_ref(context, worker, assignment.rank)
            ),
            observed_at=observed_at,
        )
        initial_state = self._stop_initial_state(current, monitor_payload, record)
        if process_probe.state == "unknown":
            failure = self._launcher_failure(
                stage=FailureStage.STOP,
                worker=worker,
                worker_id=str(worker.worker_id),
                command=f"stop rank={assignment.rank} pid={record.pid}",
                message=(
                    f"stop could not confirm remote pid {record.pid} liveness: "
                    f"{process_probe.detail}"
                ),
                recommended_action="restore SSH/WSL runtime access, then retry stop",
            )
            payload = self._stop_failure_payload(
                worker_id=str(worker.worker_id),
                rank=assignment.rank,
                stage=assignment.stage,
                pid=record.pid,
                observed_at=observed_at,
                initial_state=record.status,
                final_state="unknown",
                action="probe_failed",
                exit_evidence=process_probe.detail,
                failure=failure,
            )
            payload["process_probe"] = process_probe.to_dict()
            self._write_evidence(evidence_ref, payload)
            return (
                WorkerResult(
                    worker_id=str(worker.worker_id),
                    status=LauncherResultStatus.BLOCKED,
                    failure=failure,
                    evidence_ref=evidence_ref,
                    rank_results=(
                        RankResult(
                            rank=assignment.rank,
                            worker_id=str(worker.worker_id),
                            stage=assignment.stage,
                            pid=record.pid,
                            log_ref=record.log_path,
                            evidence_ref=evidence_ref,
                            status=LauncherResultStatus.BLOCKED,
                            failure=failure,
                            message=failure.message,
                        ),
                    ),
                ),
                payload,
                False,
            )
        if process_probe.state == "exited":
            return self._stop_dead_rank(
                context,
                current=current,
                worker=worker,
                assignment=assignment,
                record=record,
                evidence_ref=evidence_ref,
                monitor_payload=monitor_payload,
                initial_state=initial_state,
            )
        stop_result = runtime.run_script(
            self._stop_script(
                pid=record.pid,
                grace_seconds=self._stop_grace_seconds(context),
                kill_seconds=self._stop_kill_seconds(context),
                poll_interval_seconds=self._stop_poll_interval_seconds(context),
            ),
            timeout=max(
                30,
                self._stop_grace_seconds(context) + self._stop_kill_seconds(context) + 10,
            ),
        )
        payload = self._parse_stop_script_payload(
            stop_result,
            worker=worker,
            assignment=assignment,
            record=record,
            initial_state=initial_state,
            observed_at=observed_at,
        )
        payload["log_path"] = record.log_path
        self._write_evidence(evidence_ref, payload)
        failure = None
        status = LauncherResultStatus.SUCCESS
        next_record_status = "stopped"
        if payload["final_state"] == "unknown":
            failure = self._launcher_failure(
                stage=FailureStage.STOP,
                worker=worker,
                worker_id=str(worker.worker_id),
                command=f"stop rank={assignment.rank} pid={record.pid}",
                message=f"rank {assignment.rank} stop evidence was invalid",
                recommended_action="inspect remote process state and retry stop if needed",
            )
            payload["failure"] = failure.to_dict()
            payload["launcher_status"] = LauncherResultStatus.FAILED.value
            status = LauncherResultStatus.FAILED
            next_record_status = record.status
        elif payload["final_state"] == "running":
            failure = self._launcher_failure(
                stage=FailureStage.STOP,
                worker=worker,
                worker_id=str(worker.worker_id),
                command=f"stop rank={assignment.rank} pid={record.pid}",
                message=(
                    f"rank {assignment.rank} did not exit within "
                    f"{payload['elapsed_seconds']:.2f}s after bounded escalation"
                ),
                recommended_action="inspect the remote process tree and terminate it manually",
            )
            payload["failure"] = failure.to_dict()
            payload["launcher_status"] = LauncherResultStatus.FAILED.value
            status = LauncherResultStatus.FAILED
            next_record_status = "running"
        self._write_evidence(evidence_ref, payload)
        key = self._process_key(context, str(worker.worker_id), assignment.rank)
        self._process_records[key] = replace(record, status=next_record_status)
        return (
            WorkerResult(
                worker_id=str(worker.worker_id),
                status=status,
                failure=failure,
                evidence_ref=evidence_ref,
                rank_results=(
                    RankResult(
                        rank=assignment.rank,
                        worker_id=str(worker.worker_id),
                        stage=assignment.stage,
                        pid=record.pid,
                        log_ref=record.log_path,
                        evidence_ref=evidence_ref,
                        status=status,
                        failure=failure,
                        message=str(payload["message"]),
                    ),
                ),
            ),
            payload,
            True,
        )

    def _tracked_process_record(
        self,
        context: LauncherContext,
        *,
        current: JobStatus,
        worker: WorkerConfig,
        assignment,
    ) -> SSHProcessRecord | None:
        key = self._process_key(context, str(worker.worker_id), assignment.rank)
        record = self._process_records.get(key)
        if record is not None:
            return record
        persisted = next(
            (
                item
                for item in current.assignments
                if str(item.worker_id) == str(worker.worker_id) and item.rank == assignment.rank
            ),
            None,
        )
        if persisted is None or persisted.pid is None or not persisted.log_path:
            return None
        record = SSHProcessRecord(
            job_id=str(context.job.job_id),
            worker_id=str(worker.worker_id),
            rank=assignment.rank,
            local_rank=assignment.local_rank,
            stage=assignment.stage,
            pid=persisted.pid,
            command_argv=tuple(shlex.split(persisted.launch_command or "")),
            log_path=persisted.log_path,
            launched_at=current.started_at or _now(),
            remote_host=str(worker.host),
            remote_root=self._remote_root_for_worker(context, str(worker.worker_id)),
            status=persisted.status or "running",
        )
        self._process_records[key] = record
        return record

    def _stop_dead_rank(
        self,
        context: LauncherContext,
        *,
        current: JobStatus,
        worker: WorkerConfig,
        assignment,
        record: SSHProcessRecord,
        evidence_ref: str,
        monitor_payload: dict[str, Any],
        initial_state: str,
    ) -> tuple[WorkerResult, dict[str, Any], bool]:
        if monitor_payload.get("terminal_success") and current.state is JobState.COMPLETED:
            payload = {
                "observed_at": monitor_payload["observed_at"],
                "worker_id": str(worker.worker_id),
                "rank": assignment.rank,
                "stage": assignment.stage,
                "pid": record.pid,
                "log_path": record.log_path,
                "initial_state": initial_state,
                "action": "preserved",
                "escalation_level": 0,
                "final_state": "completed",
                "exit_evidence": "terminal training evidence already recorded",
                "elapsed_seconds": 0.0,
                "launcher_status": LauncherResultStatus.NOOP.value,
                "message": "rank already completed before stop",
                "checkpoint_ref": monitor_payload.get("checkpoint_ref"),
            }
            self._write_evidence(evidence_ref, payload)
            key = self._process_key(context, str(worker.worker_id), assignment.rank)
            self._process_records[key] = replace(record, status="completed")
            return (
                WorkerResult(
                    worker_id=str(worker.worker_id),
                    status=LauncherResultStatus.NOOP,
                    evidence_ref=evidence_ref,
                    rank_results=(
                        RankResult(
                            rank=assignment.rank,
                            worker_id=str(worker.worker_id),
                            stage=assignment.stage,
                            pid=record.pid,
                            log_ref=record.log_path,
                            evidence_ref=evidence_ref,
                            status=LauncherResultStatus.NOOP,
                            message="rank already completed before stop",
                        ),
                    ),
                ),
                payload,
                False,
            )
        existing_failure = current.failure if current.state is JobState.FAILED else None
        failure = existing_failure or self._rank_failure(
            context,
            worker,
            assignment,
            record,
            monitor_payload,
        )
        payload = {
            "observed_at": monitor_payload["observed_at"],
            "worker_id": str(worker.worker_id),
            "rank": assignment.rank,
            "stage": assignment.stage,
            "pid": record.pid,
            "log_path": record.log_path,
            "initial_state": initial_state,
            "action": "preserved",
            "escalation_level": 0,
            "final_state": "failed" if failure is not None else "stopped",
            "exit_evidence": "process already exited before stop",
            "elapsed_seconds": 0.0,
            "launcher_status": (
                LauncherResultStatus.NOOP.value
                if current.state is JobState.FAILED
                else LauncherResultStatus.FAILED.value
            ),
            "message": (
                "rank already failed before stop"
                if failure is not None
                else "rank already stopped before stop"
            ),
            "last_progress": monitor_payload.get("last_progress"),
        }
        if failure is not None:
            payload["failure"] = failure.to_dict()
        self._write_evidence(evidence_ref, payload)
        key = self._process_key(context, str(worker.worker_id), assignment.rank)
        self._process_records[key] = replace(
            record,
            status="failed" if failure is not None else "stopped",
        )
        status = (
            LauncherResultStatus.NOOP
            if current.state in {JobState.FAILED, JobState.STOPPED}
            else LauncherResultStatus.FAILED
            if failure is not None
            else LauncherResultStatus.NOOP
        )
        return (
            WorkerResult(
                worker_id=str(worker.worker_id),
                status=status,
                failure=None if status is LauncherResultStatus.NOOP else failure,
                evidence_ref=evidence_ref,
                rank_results=(
                    RankResult(
                        rank=assignment.rank,
                        worker_id=str(worker.worker_id),
                        stage=assignment.stage,
                        pid=record.pid,
                        log_ref=record.log_path,
                        evidence_ref=evidence_ref,
                        status=status,
                        failure=None if status is LauncherResultStatus.NOOP else failure,
                        message=str(payload["message"]),
                    ),
                ),
            ),
            payload,
            False,
        )

    def _stop_initial_state(
        self,
        current: JobStatus,
        payload: dict[str, Any],
        record: SSHProcessRecord,
    ) -> str:
        if current.state is JobState.COMPLETED and payload.get("terminal_success"):
            return "completed"
        if current.state is JobState.FAILED:
            return "failed"
        if current.state is JobState.STOPPED:
            return "stopped"
        return record.status

    def _safe_tail_log(
        self,
        runtime: WSLRuntimeWrapper,
        log_path: str,
    ) -> ProcessResult:
        result = runtime.run(
            ["tail", "-n", "200", log_path],
            timeout=self._remote_command_timeout_seconds(),
        )
        if result.ok:
            return result
        return ProcessResult(
            args=result.args,
            recorded_command=result.recorded_command,
            shell=result.shell,
            cwd=result.cwd,
            exit_code=result.exit_code,
            stdout="",
            stderr=result.stderr,
            timed_out=result.timed_out,
            runtime_environment=result.runtime_environment,
        )

    def _stop_script(
        self,
        *,
        pid: int,
        grace_seconds: int,
        kill_seconds: int,
        poll_interval_seconds: float,
    ) -> str:
        payload = {
            "pid": pid,
            "grace_seconds": grace_seconds,
            "kill_seconds": kill_seconds,
            "poll_interval_seconds": poll_interval_seconds,
        }
        return (
            "import json, os, signal, time\n"
            f"payload = json.loads({json.dumps(json.dumps(payload))})\n"
            "pid = int(payload['pid'])\n"
            "grace = max(float(payload['grace_seconds']), 0.0)\n"
            "kill_wait = max(float(payload['kill_seconds']), 0.0)\n"
            "poll = max(float(payload['poll_interval_seconds']), 0.05)\n"
            "started = time.monotonic()\n"
            "def alive() -> bool:\n"
            "    try:\n"
            "        os.kill(pid, 0)\n"
            "    except ProcessLookupError:\n"
            "        return False\n"
            "    except PermissionError:\n"
            "        return True\n"
            "    return True\n"
            "def signal_target(sig: int) -> None:\n"
            "    try:\n"
            "        os.killpg(pid, sig)\n"
            "    except ProcessLookupError:\n"
            "        return\n"
            "    except OSError:\n"
            "        os.kill(pid, sig)\n"
            "signals = []\n"
            "initial = 'running' if alive() else 'exited'\n"
            "if initial == 'running':\n"
            "    signal_target(signal.SIGTERM)\n"
            "    signals.append('SIGTERM')\n"
            "deadline = time.monotonic() + grace\n"
            "while signals and time.monotonic() < deadline and alive():\n"
            "    time.sleep(poll)\n"
            "if alive():\n"
            "    signal_target(signal.SIGKILL)\n"
            "    signals.append('SIGKILL')\n"
            "    deadline = time.monotonic() + kill_wait\n"
            "    while time.monotonic() < deadline and alive():\n"
            "        time.sleep(poll)\n"
            "running = alive()\n"
            "print(json.dumps({\n"
            "    'pid': pid,\n"
            "    'initial': initial,\n"
            "    'signals': signals,\n"
            "    'final': 'running' if running else 'stopped',\n"
            "    'elapsed_seconds': round(time.monotonic() - started, 3),\n"
            "}, sort_keys=True))\n"
        )

    def _parse_stop_script_payload(
        self,
        result: ProcessResult,
        *,
        worker: WorkerConfig,
        assignment,
        record: SSHProcessRecord,
        initial_state: str,
        observed_at: str,
    ) -> dict[str, Any]:
        if not result.ok:
            failure = self._failure_from_result(
                stage=FailureStage.STOP,
                worker=worker,
                result=result,
                message="remote stop command failed",
                recommended_action="inspect SSH stderr and remote process permissions, then retry",
            )
            return self._stop_failure_payload(
                worker_id=str(worker.worker_id),
                rank=assignment.rank,
                stage=assignment.stage,
                pid=record.pid,
                observed_at=observed_at,
                initial_state=initial_state,
                final_state="unknown",
                action="stop_failed",
                exit_evidence="remote stop command failed",
                failure=failure,
            )
        try:
            parsed = json.loads(result.stdout)
        except json.JSONDecodeError:
            parsed = None
        if not isinstance(parsed, dict):
            return {
                "observed_at": observed_at,
                "worker_id": str(worker.worker_id),
                "rank": assignment.rank,
                "stage": assignment.stage,
                "pid": record.pid,
                "initial_state": initial_state,
                "action": "stop_failed",
                "escalation_level": 0,
                "final_state": "unknown",
                "exit_evidence": "remote stop output was not valid JSON",
                "elapsed_seconds": 0.0,
                "launcher_status": LauncherResultStatus.FAILED.value,
                "message": "remote stop output was not valid JSON",
            }
        signals = [str(item) for item in parsed.get("signals", []) if str(item)]
        final_state = "stopped" if parsed.get("final") == "stopped" else "running"
        return {
            "observed_at": observed_at,
            "worker_id": str(worker.worker_id),
            "rank": assignment.rank,
            "stage": assignment.stage,
            "pid": record.pid,
            "initial_state": initial_state,
            "action": "signal" if signals else "noop",
            "escalation_level": 2 if "SIGKILL" in signals else 1 if signals else 0,
            "final_state": final_state,
            "exit_evidence": (
                "process exited after " + " -> ".join(signals)
                if final_state == "stopped" and signals
                else "process remained alive after " + " -> ".join(signals)
                if signals
                else "process already exited"
            ),
            "elapsed_seconds": float(parsed.get("elapsed_seconds", 0.0) or 0.0),
            "launcher_status": (
                LauncherResultStatus.SUCCESS.value
                if final_state == "stopped"
                else LauncherResultStatus.FAILED.value
            ),
            "message": (
                "rank stopped successfully"
                if final_state == "stopped"
                else "rank did not exit within bounded stop timeout"
            ),
        }

    def _stop_failure_payload(
        self,
        *,
        worker_id: str,
        rank: int,
        stage: str | None,
        pid: int,
        observed_at: str,
        initial_state: str,
        final_state: str,
        action: str,
        exit_evidence: str,
        failure: FailureRecord,
    ) -> dict[str, Any]:
        return {
            "observed_at": observed_at,
            "worker_id": worker_id,
            "rank": rank,
            "stage": stage,
            "pid": pid,
            "initial_state": initial_state,
            "action": action,
            "escalation_level": 0,
            "final_state": final_state,
            "exit_evidence": exit_evidence,
            "elapsed_seconds": 0.0,
            "launcher_status": LauncherResultStatus.BLOCKED.value,
            "message": failure.message,
            "failure": failure.to_dict(),
        }

    def _persist_stopped_status(
        self,
        context: LauncherContext,
        *,
        current: JobStatus,
        payloads: list[dict[str, Any]],
        signal_sent: bool,
    ) -> JobStatus:
        failure = next(
            (
                FailureRecord.from_dict(payload["failure"])
                for payload in payloads
                if isinstance(payload.get("failure"), dict)
            ),
            None,
        )
        if current.state is JobState.COMPLETED and not signal_sent and failure is None:
            state = JobState.COMPLETED
            phase = current.phase
        elif any(payload.get("final_state") == "running" for payload in payloads):
            state = JobState.FAILED
            phase = "stopping"
        elif current.state is JobState.FAILED:
            state = JobState.FAILED
            phase = current.phase
        elif failure is not None and current.state is not JobState.STOPPED:
            state = JobState.FAILED
            phase = "stopping"
        elif signal_sent or any(payload.get("final_state") == "stopped" for payload in payloads):
            state = JobState.STOPPED
            phase = "stopped"
        elif current.state in {JobState.FAILED, JobState.STOPPED, JobState.COMPLETED}:
            state = current.state
            phase = current.phase
        else:
            state = JobState.STOPPED
            phase = "stopped"
        persisted = replace(
            current,
            state=state,
            phase=phase,
            assignments=[
                self._assignment_with_runtime_pid(context, assignment)
                for assignment in context.execution_plan.workers
            ],
            failure=(
                current.failure
                if current.state is JobState.FAILED
                else failure
                if state is JobState.FAILED
                else None
            ),
            finished_at=current.finished_at if state is current.state else None,
        )
        self._save_job_status(context, persisted)
        return persisted

    def _stop_grace_seconds(self, context: LauncherContext) -> int:
        value = context.backend_config.get("stop_grace_seconds")
        return max(int(value), 5) if value else 10

    def _stop_kill_seconds(self, context: LauncherContext) -> int:
        value = context.backend_config.get("stop_kill_seconds")
        return max(int(value), 1) if value else 5

    def _stop_poll_interval_seconds(self, context: LauncherContext) -> float:
        value = context.backend_config.get("stop_poll_interval_seconds")
        return max(float(value), 0.05) if value else 0.2

    def _cleanup_worker(
        self,
        context: LauncherContext,
        *,
        current: JobStatus,
        worker: WorkerConfig,
    ) -> WorkerResult:
        runtime = self._runtime_factory(worker)
        ssh = self._ssh_factory(worker)
        assignments = [
            assignment
            for assignment in context.execution_plan.workers
            if str(assignment.worker_id) == str(worker.worker_id)
        ]
        evidence_ref = self._evidence_ref(context, "cleanup", worker)
        removed_process_refs: list[dict[str, Any]] = []
        removed_temp_paths: list[str] = []
        preserved_paths: list[str] = []
        skipped_items: list[str] = []
        failures: list[FailureRecord] = []
        rank_results: list[RankResult] = []

        for assignment in assignments:
            record = self._process_records.get(
                self._process_key(context, str(worker.worker_id), assignment.rank)
            )
            rank_result, process_payload, failure = self._cleanup_process_record(
                context,
                worker=worker,
                assignment=assignment,
                record=record,
            )
            rank_results.append(rank_result)
            if process_payload is not None:
                if process_payload.get("removed"):
                    removed_process_refs.append(process_payload)
                else:
                    skipped_items.append(str(process_payload["message"]))
            if failure is not None:
                failures.append(failure)

        remote_root = self._cleanup_remote_root(context, worker=worker, runtime=runtime)
        if remote_root.failure is not None:
            failures.append(remote_root.failure)
        removed_temp_paths.extend(remote_root.removed_paths)
        preserved_paths.extend(remote_root.preserved_paths)
        skipped_items.extend(remote_root.skipped_items)
        if remote_root.failure is not None:
            status = self._cleanup_status(failures, removed_process_refs, removed_temp_paths)
            payload = {
                "job_id": str(context.job.job_id),
                "worker_id": str(worker.worker_id),
                "cleanup_status": status.value,
                "removed_process_refs": removed_process_refs,
                "removed_temp_paths": removed_temp_paths,
                "preserved_paths": list(dict.fromkeys(preserved_paths)),
                "skipped_items": skipped_items,
                "failures": [failure.to_dict() for failure in failures],
                "recommended_action": failures[0].recommended_action,
            }
            self._write_evidence(evidence_ref, payload)
            return WorkerResult(
                worker_id=str(worker.worker_id),
                status=status,
                rank_results=tuple(rank_results),
                failure=failures[0],
                evidence_ref=evidence_ref,
                message=failures[0].message,
            )

        staging = self._cleanup_windows_staging(context, worker=worker, runtime=runtime, ssh=ssh)
        if staging.failure is not None:
            failures.append(staging.failure)
        removed_temp_paths.extend(staging.removed_paths)
        skipped_items.extend(staging.skipped_items)

        status = self._cleanup_status(failures, removed_process_refs, removed_temp_paths)
        payload = {
            "job_id": str(context.job.job_id),
            "worker_id": str(worker.worker_id),
            "cleanup_status": status.value,
            "removed_process_refs": removed_process_refs,
            "removed_temp_paths": removed_temp_paths,
            "preserved_paths": list(dict.fromkeys(preserved_paths)),
            "skipped_items": skipped_items,
            "failures": [failure.to_dict() for failure in failures],
            "recommended_action": (
                failures[0].recommended_action if failures else "cleanup already complete"
            ),
        }
        self._write_evidence(evidence_ref, payload)
        return WorkerResult(
            worker_id=str(worker.worker_id),
            status=status,
            rank_results=tuple(rank_results),
            failure=failures[0] if failures else None,
            evidence_ref=evidence_ref,
            message=(
                "cleanup completed"
                if status is LauncherResultStatus.SUCCESS
                else "cleanup already complete"
                if status is LauncherResultStatus.NOOP
                else failures[0].message
            ),
        )

    def _cleanup_process_record(
        self,
        context: LauncherContext,
        *,
        worker: WorkerConfig,
        assignment,
        record: SSHProcessRecord | None,
    ) -> tuple[RankResult, dict[str, Any] | None, FailureRecord | None]:
        evidence_ref = self._cleanup_rank_evidence_ref(context, worker, assignment.rank)
        if record is None:
            return (
                RankResult(
                    rank=assignment.rank,
                    worker_id=str(worker.worker_id),
                    stage=assignment.stage,
                    evidence_ref=evidence_ref,
                    status=LauncherResultStatus.NOOP,
                    message="no managed process record for cleanup",
                ),
                {
                    "rank": assignment.rank,
                    "worker_id": str(worker.worker_id),
                    "message": "no managed process record for cleanup",
                    "removed": False,
                },
                None,
            )
        runtime = self._runtime_factory(worker)
        try:
            probe = self._probe_process_liveness(runtime, record.pid)
        except Exception as exc:
            failure = self._launcher_failure(
                stage=FailureStage.CLEANUP,
                worker=worker,
                worker_id=str(worker.worker_id),
                command=f"cleanup rank={assignment.rank} pid={record.pid}",
                message=f"cleanup lost connection while probing pid {record.pid}: {exc}",
                recommended_action="restore SSH/WSL access, then rerun cleanup",
            )
            return (
                RankResult(
                    rank=assignment.rank,
                    worker_id=str(worker.worker_id),
                    stage=assignment.stage,
                    pid=record.pid,
                    log_ref=record.log_path,
                    evidence_ref=evidence_ref,
                    status=LauncherResultStatus.FAILED,
                    failure=failure,
                    message=failure.message,
                ),
                None,
                failure,
            )
        if probe.state == "unknown":
            status = (
                LauncherResultStatus.BLOCKED
                if self._is_auth_problem(probe.stderr)
                else LauncherResultStatus.FAILED
            )
            failure = self._launcher_failure(
                stage=FailureStage.CLEANUP,
                worker=worker,
                worker_id=str(worker.worker_id),
                command=f"cleanup rank={assignment.rank} pid={record.pid}",
                message=(
                    f"cleanup could not confirm remote pid {record.pid} liveness: "
                    f"{probe.detail}"
                ),
                recommended_action="restore SSH/WSL access, then rerun cleanup",
            )
            return (
                RankResult(
                    rank=assignment.rank,
                    worker_id=str(worker.worker_id),
                    stage=assignment.stage,
                    pid=record.pid,
                    log_ref=record.log_path,
                    evidence_ref=evidence_ref,
                    status=status,
                    failure=failure,
                    message=failure.message,
                ),
                None,
                failure,
            )
        if probe.state == "exited":
            self._process_records[
                self._process_key(context, str(worker.worker_id), assignment.rank)
            ] = replace(
                record,
                status="stopped" if record.status == "running" else record.status,
            )
            return (
                RankResult(
                    rank=assignment.rank,
                    worker_id=str(worker.worker_id),
                    stage=assignment.stage,
                    pid=record.pid,
                    log_ref=record.log_path,
                    evidence_ref=evidence_ref,
                    status=LauncherResultStatus.NOOP,
                    message="managed process already exited before cleanup",
                ),
                {
                    "rank": assignment.rank,
                    "worker_id": str(worker.worker_id),
                    "pid": record.pid,
                    "message": "managed process already exited before cleanup",
                    "removed": False,
                },
                None,
            )
        result = runtime.run_script(
            self._stop_script(
                pid=record.pid,
                grace_seconds=self._stop_grace_seconds(context),
                kill_seconds=self._stop_kill_seconds(context),
                poll_interval_seconds=self._stop_poll_interval_seconds(context),
            ),
            timeout=max(
                30, self._stop_grace_seconds(context) + self._stop_kill_seconds(context) + 10
            ),
        )
        payload = self._parse_stop_script_payload(
            result,
            worker=worker,
            assignment=assignment,
            record=record,
            initial_state=record.status,
            observed_at=_now(),
        )
        failure = None
        status = LauncherResultStatus.SUCCESS
        next_record_status = "stopped"
        if payload["final_state"] != "stopped":
            failure = self._launcher_failure(
                stage=FailureStage.CLEANUP,
                worker=worker,
                worker_id=str(worker.worker_id),
                command=f"cleanup rank={assignment.rank} pid={record.pid}",
                message=f"managed process {record.pid} survived cleanup escalation",
                recommended_action="inspect the remote process tree and retry cleanup",
            )
            status = LauncherResultStatus.FAILED
            next_record_status = record.status
        self._process_records[
            self._process_key(context, str(worker.worker_id), assignment.rank)
        ] = replace(
            record,
            status=next_record_status,
        )
        return (
            RankResult(
                rank=assignment.rank,
                worker_id=str(worker.worker_id),
                stage=assignment.stage,
                pid=record.pid,
                log_ref=record.log_path,
                evidence_ref=evidence_ref,
                status=status,
                failure=failure,
                message=(
                    "managed process removed during cleanup" if failure is None else failure.message
                ),
            ),
            {
                "rank": assignment.rank,
                "worker_id": str(worker.worker_id),
                "pid": record.pid,
                "action": "signal",
                "escalation_level": payload["escalation_level"],
                "removed": failure is None,
            },
            failure,
        )

    @dataclass(frozen=True)
    class _CleanupPathOutcome:
        removed_paths: tuple[str, ...] = ()
        preserved_paths: tuple[str, ...] = ()
        skipped_items: tuple[str, ...] = ()
        failure: FailureRecord | None = None

    def _cleanup_remote_root(
        self,
        context: LauncherContext,
        *,
        worker: WorkerConfig,
        runtime: WSLRuntimeWrapper,
    ) -> _CleanupPathOutcome:
        remote_root = self._remote_root_for_worker(context, str(worker.worker_id))
        try:
            self._validate_cleanup_target(
                target=remote_root,
                allowed_root=str(self.cluster_config.jobs_root),
                label="remote job root",
            )
        except ValueError as exc:
            return self._CleanupPathOutcome(
                failure=self._launcher_failure(
                    stage=FailureStage.CLEANUP,
                    worker=worker,
                    message=str(exc),
                    recommended_action="repair the recorded remote path before rerunning cleanup",
                )
            )
        probe = _probe_remote_snapshot(runtime, remote_root)
        if not probe.exists:
            return self._CleanupPathOutcome(skipped_items=("remote snapshot already absent",))
        if _is_prepare_only_layout(probe):
            return self._cleanup_path(
                runtime,
                worker=worker,
                target=remote_root,
                allowed_root=str(self.cluster_config.jobs_root),
                label="prepare-only remote root",
            )
        return self._CleanupPathOutcome(preserved_paths=(remote_root,))

    def _cleanup_windows_staging(
        self,
        context: LauncherContext,
        *,
        worker: WorkerConfig,
        runtime: WSLRuntimeWrapper,
        ssh: SSHTransport,
    ) -> _CleanupPathOutcome:
        try:
            windows_profile = _read_windows_userprofile(ssh)
        except Exception as exc:
            return self._CleanupPathOutcome(
                failure=self._launcher_failure(
                    stage=FailureStage.CLEANUP,
                    worker=worker,
                    message=f"failed to resolve remote staging root: {exc}",
                    recommended_action="inspect Windows home directory access and rerun cleanup",
                )
            )
        staging_windows = str(
            PurePosixPath(
                _windows_to_wsl_path(windows_profile),
                ".shardgrid",
                "snapshots",
                str(context.job.job_id),
            )
        )
        allowed_root = str(
            PurePosixPath(_windows_to_wsl_path(windows_profile), ".shardgrid", "snapshots")
        )
        return self._cleanup_path(
            runtime,
            worker=worker,
            target=staging_windows,
            allowed_root=allowed_root,
            label="windows staging root",
        )

    def _cleanup_path(
        self,
        runtime: WSLRuntimeWrapper,
        *,
        worker: WorkerConfig,
        target: str,
        allowed_root: str,
        label: str,
    ) -> _CleanupPathOutcome:
        try:
            self._validate_cleanup_target(
                target=target,
                allowed_root=allowed_root,
                label=label,
            )
        except ValueError as exc:
            return self._CleanupPathOutcome(
                failure=self._launcher_failure(
                    stage=FailureStage.CLEANUP,
                    worker=worker,
                    message=str(exc),
                    recommended_action="repair the recorded cleanup path before rerunning cleanup",
                )
            )
        result = runtime.run_script(
            self._cleanup_tree_script(target=target, allowed_root=allowed_root), timeout=30
        )
        if not result.ok:
            return self._CleanupPathOutcome(
                failure=self._failure_from_result(
                    stage=FailureStage.CLEANUP,
                    worker=worker,
                    result=result,
                    message=f"{label} cleanup command failed",
                    recommended_action="inspect remote path permissions and retry cleanup",
                )
            )
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            return self._CleanupPathOutcome(
                failure=self._launcher_failure(
                    stage=FailureStage.CLEANUP,
                    worker=worker,
                    message=f"{label} cleanup returned invalid JSON: {exc}",
                    recommended_action="repair the cleanup helper output and rerun cleanup",
                )
            )
        removed_paths = tuple(str(item) for item in payload.get("removed_paths", []))
        skipped_items = tuple(str(item) for item in payload.get("skipped_items", []))
        if payload.get("final_state") == "missing":
            return self._CleanupPathOutcome(
                skipped_items=skipped_items or (f"{label} already clean",)
            )
        return self._CleanupPathOutcome(removed_paths=removed_paths, skipped_items=skipped_items)

    def _remote_root_for_worker(self, context: LauncherContext, worker_id: str) -> str:
        distribution = self._distribution_records.get((str(context.job.job_id), worker_id))
        if distribution is not None:
            return distribution.remote_snapshot_root
        return self._remote_job_root(context)

    def _validate_cleanup_target(
        self,
        *,
        target: str,
        allowed_root: str,
        label: str,
    ) -> None:
        candidate = PurePosixPath(target)
        base = PurePosixPath(allowed_root)
        if not candidate.is_absolute() or not base.is_absolute():
            raise ValueError(f"{label} escaped allowed root")
        if candidate == PurePosixPath("/") or candidate == base:
            raise ValueError(f"{label} escaped allowed root")
        if ".." in candidate.parts:
            raise ValueError(f"{label} escaped allowed root")
        if base not in candidate.parents:
            raise ValueError(f"{label} escaped allowed root")

    def _cleanup_tree_script(self, *, target: str, allowed_root: str) -> str:
        payload = {"target": target, "allowed_root": allowed_root}
        return (
            "import json\n"
            "import shutil\n"
            "from pathlib import Path\n"
            f"payload = json.loads({json.dumps(json.dumps(payload))})\n"
            "target = Path(payload['target']).resolve(strict=False)\n"
            "allowed = Path(payload['allowed_root']).resolve(strict=False)\n"
            "if allowed not in target.parents:\n"
            "    raise SystemExit('escaped allowed root')\n"
            "if not target.exists():\n"
            "    print(json.dumps({\n"
            "        'final_state': 'missing',\n"
            "        'removed_paths': [],\n"
            "        'skipped_items': ['missing'],\n"
            "    }, sort_keys=True))\n"
            "    raise SystemExit(0)\n"
            "for path in [target, *target.rglob('*')]:\n"
            "    if path.is_symlink():\n"
            "        raise SystemExit('symlink escape detected')\n"
            "shutil.rmtree(target)\n"
            "print(json.dumps({\n"
            "    'final_state': 'removed',\n"
            "    'removed_paths': [str(target)],\n"
            "    'skipped_items': [],\n"
            "}, sort_keys=True))\n"
        )

    def _cleanup_rank_evidence_ref(
        self,
        context: LauncherContext,
        worker: WorkerConfig,
        rank: int,
    ) -> str:
        base = context.snapshot.diagnostics_path if context.snapshot else "diagnostics"
        return str(PurePosixPath(base) / f"cleanup-{worker.worker_id}-rank{rank}.json")

    def _cleanup_status(
        self,
        failures: list[FailureRecord],
        removed_process_refs: list[dict[str, Any]],
        removed_temp_paths: list[str],
    ) -> LauncherResultStatus:
        if failures:
            return (
                LauncherResultStatus.PARTIAL
                if (removed_process_refs or removed_temp_paths)
                else LauncherResultStatus.FAILED
            )
        if removed_process_refs or removed_temp_paths:
            return LauncherResultStatus.SUCCESS
        return LauncherResultStatus.NOOP

    def _rank_monitor_payload(
        self,
        context: LauncherContext,
        *,
        worker: WorkerConfig,
        assignment,
        record: SSHProcessRecord,
        process_probe: ProcessLivenessProbe,
        log_result: ProcessResult,
        previous_payload: dict[str, Any] | None,
        observed_at: str,
    ) -> dict[str, Any]:
        baseline = self._monitor_baseline(context, previous_payload)
        log_tail = redact_text(log_result.stdout, self._secrets) or ""
        placement, placement_error = self._parse_marker_payload(log_tail, _EVENT_MARKER)
        forward, forward_error = self._parse_marker_payload(log_tail, _FORWARD_MARKER)
        backward, backward_error = self._parse_marker_payload(log_tail, _BACKWARD_MARKER)
        train, train_error = self._parse_marker_payload(log_tail, _TRAIN_MARKER)
        plain_training_marker = self._last_plain_marker(log_tail, _PLAIN_TRAIN_MARKERS)
        marker_parse_errors = [
            {
                "worker_id": str(worker.worker_id),
                "rank": assignment.rank,
                "stage": assignment.stage,
                **item,
            }
            for item in (placement_error, forward_error, backward_error, train_error)
            if item is not None
        ]
        training_started = any(item is not None for item in (forward, backward, train)) or (
            plain_training_marker is not None
        )
        process_state = process_probe.state
        training_started = baseline["training_started"] or training_started
        rendezvous_ready = baseline["rendezvous_ready"] or placement is not None
        phase = str(baseline["phase"])
        if rendezvous_ready:
            phase = "rendezvous"
        if training_started:
            phase = "training"
        if train is not None and process_state == "exited":
            phase = "checkpoint"
        checkpoint_ref = self._checkpoint_ref(context, train) or baseline["checkpoint_ref"]
        final_loss = self._finite_float(train, "final_loss")
        if final_loss is None:
            final_loss = baseline["final_loss"]
        loss_history = self._finite_loss_history(train) or list(baseline["loss_history"])
        checkpoint_roundtrip_ok = bool(
            isinstance(train, dict) and train.get("checkpoint_roundtrip_ok") is True
        )
        terminal_success = (
            train is not None
            and checkpoint_ref is not None
            and checkpoint_roundtrip_ok
            and process_state == "exited"
        )
        terminal_state = (
            "success"
            if terminal_success
            else "invalid"
            if process_state == "exited"
            else "none"
        )
        last_progress = (
            "T074_TRAIN_EVIDENCE"
            if train is not None
            else "T073_BACKWARD_EVIDENCE"
            if backward is not None
            else "T072_FORWARD_EVIDENCE"
            if forward is not None
            else plain_training_marker
            if plain_training_marker is not None
            else "STAGE_PLACEMENT_EVIDENCE"
            if placement is not None
            else str(baseline["last_progress"])
        )
        progress_changed = self._progress_changed(
            baseline=baseline,
            phase=phase,
            last_progress=last_progress,
            checkpoint_ref=checkpoint_ref,
            final_loss=final_loss,
            loss_history=loss_history,
            terminal_success=terminal_success,
            log_tail=log_tail,
        )
        last_update_timestamp = (
            observed_at
            if progress_changed
            else str(baseline["last_update_timestamp"] or observed_at)
        )
        timeout_stage = self._rank_timeout_stage(
            context,
            record,
            phase,
            observed_at,
            last_update_timestamp=last_update_timestamp,
        )
        return {
            "observed_at": observed_at,
            "last_observed_at": observed_at,
            "worker_id": str(worker.worker_id),
            "rank": assignment.rank,
            "stage": assignment.stage,
            "pid": record.pid,
            "log_path": record.log_path,
            "status": (
                "running"
                if process_state == "alive"
                else "exited"
                if process_state == "exited"
                else "unknown"
            ),
            "process_state": process_state,
            "process_probe": process_probe.to_dict(),
            "log_state": "available",
            "terminal_state": terminal_state,
            "running": process_state == "alive",
            "process_exit_known": process_state == "exited",
            "process_exit_code": None,
            "phase": phase,
            "message": self._rank_progress_message(
                process_state=process_state,
                phase=phase,
                terminal_success=terminal_success,
                timeout_stage=timeout_stage,
                log_state="available",
            ),
            "rendezvous_ready": rendezvous_ready,
            "training_started": training_started,
            "terminal_success": terminal_success,
            "progress_changed": progress_changed,
            "timeout_stage": timeout_stage.value if timeout_stage is not None else None,
            "last_progress": last_progress,
            "last_update_timestamp": last_update_timestamp,
            "placement": placement,
            "forward": forward,
            "backward": backward,
            "train": train,
            "loss_history": loss_history,
            "latest_loss": loss_history[-1] if loss_history else final_loss,
            "final_loss": final_loss,
            "checkpoint_ref": checkpoint_ref,
            "marker_parse_errors": marker_parse_errors,
            "log_tail": log_tail[-4000:],
        }

    def _rank_launcher_status(self, payload: dict[str, Any]) -> LauncherResultStatus:
        if payload.get("launcher_status"):
            return LauncherResultStatus(str(payload["launcher_status"]))
        if payload.get("terminal_success"):
            return LauncherResultStatus.SUCCESS
        if payload.get("timeout_stage") is not None:
            return LauncherResultStatus.FAILED
        if payload.get("process_state") == "unknown":
            return LauncherResultStatus.SUCCESS
        if payload.get("running"):
            return LauncherResultStatus.SUCCESS
        return LauncherResultStatus.FAILED

    def _rank_failure(
        self,
        context: LauncherContext,
        worker: WorkerConfig,
        assignment,
        record: SSHProcessRecord,
        payload: dict[str, Any],
    ) -> FailureRecord | None:
        if payload.get("terminal_success"):
            return None
        timeout_stage = payload.get("timeout_stage")
        if timeout_stage is not None:
            stage = FailureStage(timeout_stage)
            return make_failure_record(
                stage=stage,
                host=str(worker.host),
                worker_id=str(worker.worker_id),
                command=f"monitor rank={assignment.rank} pid={record.pid}",
                python_executable=self._python_executable(worker),
                conda_environment=worker.conda_environment,
                conda_prefix=worker.conda_prefix,
                message=f"rank {assignment.rank} timed out during {stage.value.lower()}",
                recommended_action=(
                    "inspect rank logs, rendezvous configuration, and progress markers"
                ),
                runtime_environment={
                    "rank": str(assignment.rank),
                    "stage": assignment.stage or "",
                    "pid": str(record.pid),
                },
                stdout_path=record.log_path,
                secrets=self._secrets,
            )
        if payload.get("process_state") == "unknown":
            return None
        if payload.get("running"):
            return None
        stage = self._rank_failure_stage(payload)
        return make_failure_record(
            stage=stage,
            host=str(worker.host),
            worker_id=str(worker.worker_id),
            command=f"monitor rank={assignment.rank} pid={record.pid}",
            python_executable=self._python_executable(worker),
            conda_environment=worker.conda_environment,
            conda_prefix=worker.conda_prefix,
            message=(
                f"rank {assignment.rank} exited without a valid terminal result; "
                f"last progress={payload['last_progress']}"
            ),
            recommended_action="inspect surviving rank logs and remote runtime state",
            runtime_environment={
                "rank": str(assignment.rank),
                "stage": assignment.stage or "",
                "pid": str(record.pid),
                "last_progress": str(payload["last_progress"]),
            },
            stdout_path=record.log_path,
            secrets=self._secrets,
        )

    def _rank_failure_stage(self, payload: dict[str, Any]) -> FailureStage:
        if payload.get("training_started"):
            return FailureStage.TRAIN
        if payload.get("rendezvous_ready"):
            return FailureStage.RENDEZVOUS
        return FailureStage.LAUNCH

    def _rank_timeout_stage(
        self,
        context: LauncherContext,
        record: SSHProcessRecord,
        phase: str,
        observed_at: str,
        *,
        last_update_timestamp: str | None,
    ) -> FailureStage | None:
        del context
        observed = self._parse_timestamp(observed_at)
        launched = self._parse_timestamp(record.launched_at)
        if observed is None or launched is None:
            return None
        if phase == "launch" and (
            observed - launched
        ).total_seconds() > self._rendezvous_timeout_seconds():
            return FailureStage.RENDEZVOUS
        anchor = self._parse_timestamp(last_update_timestamp) or launched
        if phase in {"rendezvous", "training"} and (
            observed - anchor
        ).total_seconds() > self._progress_timeout_seconds():
            return FailureStage.RENDEZVOUS if phase == "rendezvous" else FailureStage.TRAIN
        return None

    def _rank_progress_message(
        self,
        *,
        process_state: str,
        phase: str,
        terminal_success: bool,
        timeout_stage: FailureStage | None,
        log_state: str,
    ) -> str:
        if terminal_success:
            return "process exited after valid terminal training evidence"
        if timeout_stage is not None:
            return f"timed out during {timeout_stage.value.lower()}"
        if process_state == "unknown":
            if log_state == "transient_timeout":
                return (
                    "process liveness is unknown and remote log read timed out; "
                    "keeping last known progress"
                )
            return "process liveness is unknown; keeping last known progress"
        if process_state == "alive" and log_state == "transient_timeout":
            return "process is alive but remote log read timed out; keeping last known progress"
        if process_state == "alive" and log_state == "missing":
            return (
                "process is alive but remote log is not available yet; "
                "keeping last known progress"
            )
        if process_state == "alive" and log_state == "failed":
            return "process is alive but remote log read failed; keeping last known progress"
        if process_state == "alive":
            return f"running in {phase}"
        return f"process exited during {phase}"

    def _persist_monitored_status(
        self,
        context: LauncherContext,
        statuses: list[dict[str, Any]],
    ):
        current = self._load_job_status(context)
        final_train = next(
            (
                payload["train"]
                for payload in reversed(statuses)
                if payload.get("train") is not None
            ),
            None,
        )
        authoritative_loss = next(
            (
                list(payload.get("loss_history") or [])
                for payload in reversed(statuses)
                if payload.get("train") is not None and payload.get("loss_history")
            ),
            next(
                (
                    list(payload.get("loss_history") or [])
                    for payload in reversed(statuses)
                    if payload.get("loss_history")
                ),
                list(current.loss_history),
            ),
        )
        checkpoint_ref = current.checkpoint_ref
        final_metrics = dict(current.final_metrics)
        final_loss = self._finite_float(final_train, "final_loss")
        if final_loss is not None:
            final_metrics["final_loss"] = final_loss
        phase = self._job_phase_from_rank_statuses(statuses)
        state = self._job_state_from_rank_statuses(statuses)
        failure = next(
            (
                self._rank_failure(
                    context,
                    next(
                        worker
                        for worker in self.cluster_config.workers
                        if str(worker.worker_id) == payload["worker_id"]
                    ),
                    next(
                        assignment
                        for assignment in context.execution_plan.workers
                        if assignment.rank == payload["rank"]
                    ),
                    self._process_records[
                        (str(context.job.job_id), payload["worker_id"], payload["rank"])
                    ],
                    payload,
                )
                for payload in statuses
                if self._rank_launcher_status(payload)
                in {LauncherResultStatus.FAILED, LauncherResultStatus.BLOCKED}
            ),
            None,
        )
        persisted = type(current)(
            job_id=current.job_id,
            state=state,
            phase=phase,
            workers=[assignment.worker_id for assignment in context.execution_plan.workers],
            assignments=[
                self._assignment_with_runtime_pid(context, assignment)
                for assignment in context.execution_plan.workers
            ],
            runtime_environment_refs=dict(context.runtime_environment_refs),
            latest_loss=authoritative_loss[-1] if authoritative_loss else current.latest_loss,
            loss_history=authoritative_loss,
            final_metrics=final_metrics,
            backend=context.execution_plan.backend,
            fallback_used=current.fallback_used,
            started_at=current.started_at or self._process_started_at(context),
            finished_at=current.finished_at,
            failure=failure if state is JobState.FAILED else None,
            checkpoint_ref=checkpoint_ref,
        )
        self._save_job_status(context, persisted)
        return persisted

    def _job_phase_from_rank_statuses(self, statuses: list[dict[str, Any]]) -> str:
        if statuses and all(payload.get("terminal_success") for payload in statuses):
            return "checkpoint"
        if any(payload.get("training_started") for payload in statuses):
            return "training"
        if any(payload.get("rendezvous_ready") for payload in statuses):
            return "rendezvous"
        return "launch"

    def _job_state_from_rank_statuses(self, statuses: list[dict[str, Any]]) -> JobState:
        if statuses and all(payload.get("terminal_success") for payload in statuses):
            return JobState.CHECKPOINTING
        if any(
            self._rank_launcher_status(payload) is LauncherResultStatus.BLOCKED
            for payload in statuses
        ):
            current = JobState.LAUNCHING
            if any(payload.get("rendezvous_ready") for payload in statuses):
                current = JobState.RENDEZVOUS
            if any(payload.get("training_started") for payload in statuses):
                current = JobState.TRAINING
            return current
        if any(
            self._rank_launcher_status(payload) is LauncherResultStatus.FAILED
            for payload in statuses
        ):
            return JobState.FAILED
        if any(payload.get("training_started") for payload in statuses):
            return JobState.TRAINING
        if any(payload.get("rendezvous_ready") for payload in statuses):
            return JobState.RENDEZVOUS
        return JobState.LAUNCHING

    def _load_job_status(self, context: LauncherContext):
        if context.job_status is not None:
            return context.job_status
        path = self._job_status_path(context)
        if path is not None and path.exists():
            return StatusStore(path.parent.parent).load_path(path)
        return self._initial_job_status(context)

    def _save_job_status(self, context: LauncherContext, status) -> None:
        path = self._job_status_path(context)
        if path is None:
            return
        StatusStore(path.parent.parent).save_path(path, status)

    def _job_status_path(self, context: LauncherContext) -> Path | None:
        if context.snapshot is None:
            return None
        return Path(context.snapshot.diagnostics_path) / "job-status.json"

    def _initial_job_status(self, context: LauncherContext):
        return JobStatus(
            job_id=context.job.job_id,
            state=JobState.LAUNCHING,
            phase="launch",
            workers=[assignment.worker_id for assignment in context.execution_plan.workers],
            assignments=list(context.execution_plan.workers),
            runtime_environment_refs=dict(context.runtime_environment_refs),
            backend=context.execution_plan.backend,
            started_at=self._process_started_at(context),
        )

    def _process_started_at(self, context: LauncherContext) -> str | None:
        stamps = [
            record.launched_at
            for key, record in self._process_records.items()
            if key[0] == str(context.job.job_id)
        ]
        return min(stamps) if stamps else None

    def _assignment_with_runtime_pid(self, context: LauncherContext, assignment):
        record = self._process_records.get(
            (str(context.job.job_id), str(assignment.worker_id), assignment.rank)
        )
        if record is None:
            return assignment
        return type(assignment)(
            worker_id=assignment.worker_id,
            rank=assignment.rank,
            local_rank=assignment.local_rank,
            stage=assignment.stage,
            gpu_index=assignment.gpu_index,
            conda_environment=assignment.conda_environment,
            conda_prefix=assignment.conda_prefix,
            python_executable=assignment.python_executable,
            launch_command=assignment.launch_command,
            environment=dict(assignment.environment),
            status=record.status,
            pid=record.pid,
            log_path=record.log_path,
        )

    def _parse_marker_payload(
        self,
        text: str,
        marker: str,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        decoder = json.JSONDecoder()
        for line in reversed(text.splitlines()):
            if not line.startswith(marker):
                continue
            payload_text = line[len(marker) :]
            try:
                payload, end = decoder.raw_decode(payload_text)
            except json.JSONDecodeError as exc:
                return None, self._marker_parse_error(
                    marker=marker,
                    payload_text=payload_text,
                    error=f"{exc.msg} at char {exc.pos}",
                )
            if payload_text[end:].strip():
                return None, self._marker_parse_error(
                    marker=marker,
                    payload_text=payload_text,
                    error="extra trailing data after JSON payload",
                )
            if isinstance(payload, dict):
                return payload, None
            return None, self._marker_parse_error(
                marker=marker,
                payload_text=payload_text,
                error=f"expected JSON object, got {type(payload).__name__}",
            )
        return None, None

    def _marker_parse_error(
        self,
        *,
        marker: str,
        payload_text: str,
        error: str,
    ) -> dict[str, Any]:
        prefix, suffix = _clip_marker_payload(payload_text)
        return {
            "marker": marker.strip(),
            "error": error,
            "payload_length": len(payload_text),
            "payload_prefix": prefix,
            "payload_suffix": suffix,
        }

    def _last_plain_marker(self, text: str, markers: Sequence[str]) -> str | None:
        for line in reversed(text.splitlines()):
            stripped = line.strip()
            for marker in markers:
                if stripped == marker or f"[MARKER={marker}]" in stripped:
                    return marker
        return None

    def _finite_float(self, payload: dict[str, Any] | None, key: str) -> float | None:
        if not isinstance(payload, dict):
            return None
        value = payload.get(key)
        if not isinstance(value, (int, float)):
            return None
        number = float(value)
        if number != number or number in {float("inf"), float("-inf")}:
            return None
        return number

    def _finite_loss_history(self, payload: dict[str, Any] | None) -> list[float]:
        if not isinstance(payload, dict):
            return []
        history = payload.get("loss_history")
        if not isinstance(history, list):
            return []
        values: list[float] = []
        for item in history:
            if not isinstance(item, (int, float)):
                continue
            number = float(item)
            if number != number or number in {float("inf"), float("-inf")}:
                continue
            values.append(number)
        return values

    def _checkpoint_ref(
        self,
        context: LauncherContext,
        payload: dict[str, Any] | None,
    ) -> str | None:
        if not isinstance(payload, dict):
            return None
        path = payload.get("checkpoint_path")
        if not isinstance(path, str) or not path.strip():
            return None
        snapshot_root = Path(context.snapshot.root_path).resolve() if context.snapshot else None
        checkpoint = Path(path).resolve()
        if snapshot_root is not None:
            try:
                return checkpoint.relative_to(snapshot_root).as_posix()
            except ValueError:
                return str(checkpoint)
        return str(checkpoint)

    def _parse_timestamp(self, value: str | None) -> datetime | None:
        if not value:
            return None
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None

    def _rendezvous_timeout_seconds(self) -> int:
        return max(self.cluster_config.ssh.connect_timeout_seconds * 4, 60)

    def _progress_timeout_seconds(self) -> int:
        return max(self.cluster_config.ssh.connect_timeout_seconds * 8, 120)

    def _remote_command_timeout_seconds(self) -> float:
        return float(self.cluster_config.ssh.command_timeout_seconds)

    def _read_evidence(self, path: str | None) -> dict[str, Any] | None:
        if not path:
            return None
        target = Path(path)
        if not target.exists():
            return None
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    def _write_evidence(self, path: str | None, payload: dict[str, Any]) -> None:
        if not path:
            return
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def _process_liveness_command(self, pid: int) -> tuple[str, str, str]:
        return (
            "/bin/sh",
            "-lc",
            (
                f"kill -0 {int(pid)} >/dev/null 2>&1; code=$?; "
                "if [ \"$code\" -eq 0 ]; then "
                "printf '%s\\n' "
                "'{\"state\":\"alive\",\"detail\":\"signal 0 succeeded\"}'; "
                "elif [ \"$code\" -eq 1 ]; then "
                "printf '%s\\n' "
                "'{\"state\":\"exited\",\"detail\":\"signal 0 reported missing pid\"}'; "
                "else "
                "printf "
                "'{\"state\":\"unknown\",\"detail\":\"kill -0 returned exit code %s\"}\\n' "
                "\"$code\"; "
                "fi"
            ),
        )

    def _probe_process_liveness(
        self,
        runtime: WSLRuntimeWrapper,
        pid: int,
    ) -> ProcessLivenessProbe:
        result = runtime.run(
            self._process_liveness_command(pid),
            timeout=self._remote_command_timeout_seconds(),
        )
        stdout = redact_text(result.stdout, self._secrets) or ""
        stderr = redact_text(result.stderr, self._secrets) or ""
        legacy_state = self._legacy_probe_state(result)
        if legacy_state is not None:
            return ProcessLivenessProbe(
                state=legacy_state,
                detail="legacy exit-code probe result",
                transport_status="legacy",
                recorded_command=result.recorded_command,
                exit_code=result.exit_code,
                timed_out=result.timed_out,
                stdout=stdout[-1000:],
                stderr=stderr[-1000:],
            )
        if not result.ok:
            detail = (
                "remote liveness probe timed out"
                if result.timed_out
                else "remote liveness probe failed before returning structured state"
            )
            return ProcessLivenessProbe(
                state="unknown",
                detail=detail,
                transport_status="failed",
                recorded_command=result.recorded_command,
                exit_code=result.exit_code,
                timed_out=result.timed_out,
                stdout=stdout[-1000:],
                stderr=stderr[-1000:],
            )
        payload = self._parse_probe_json(stdout)
        if payload is None:
            return ProcessLivenessProbe(
                state="unknown",
                detail="remote liveness probe returned invalid JSON evidence",
                transport_status="ok",
                recorded_command=result.recorded_command,
                exit_code=result.exit_code,
                timed_out=result.timed_out,
                stdout=stdout[-1000:],
                stderr=stderr[-1000:],
            )
        state = payload.get("state")
        detail = payload.get("detail")
        if state not in {"alive", "exited", "unknown"}:
            state = "unknown"
        if not isinstance(detail, str) or not detail.strip():
            detail = "remote liveness probe returned incomplete evidence"
        return ProcessLivenessProbe(
            state=state,
            detail=detail,
            transport_status="ok",
            recorded_command=result.recorded_command,
            exit_code=result.exit_code,
            timed_out=result.timed_out,
            stdout=stdout[-1000:],
            stderr=stderr[-1000:],
        )

    def _parse_probe_json(self, stdout: str) -> dict[str, Any] | None:
        lines = [line.strip() for line in stdout.splitlines() if line.strip()]
        if not lines:
            return None
        try:
            payload = json.loads(lines[-1])
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None

    def _legacy_probe_state(self, result: ProcessResult) -> str | None:
        command = result.recorded_command.strip()
        if not command.startswith("kill -0 "):
            return None
        if result.exit_code == 0 and not result.timed_out:
            return "alive"
        if result.exit_code == 1 and not result.timed_out:
            return "exited"
        return None

    def _transient_monitor_payload(
        self,
        context: LauncherContext,
        *,
        worker: WorkerConfig,
        assignment,
        record: SSHProcessRecord,
        process_probe: ProcessLivenessProbe,
        log_result: ProcessResult,
        previous_payload: dict[str, Any] | None,
        observed_at: str,
    ) -> dict[str, Any]:
        baseline = self._monitor_baseline(context, previous_payload)
        process_state = process_probe.state
        log_state = self._log_state(log_result)
        phase = str(baseline["phase"])
        last_update_timestamp = str(baseline["last_update_timestamp"] or observed_at)
        timeout_stage = self._rank_timeout_stage(
            context,
            record,
            phase,
            observed_at,
            last_update_timestamp=last_update_timestamp,
        )
        return {
            "observed_at": observed_at,
            "last_observed_at": observed_at,
            "worker_id": str(worker.worker_id),
            "rank": assignment.rank,
            "stage": assignment.stage,
            "pid": record.pid,
            "log_path": record.log_path,
            "status": "log_unavailable" if log_state != "failed" else "log_failed",
            "process_state": process_state,
            "process_probe": process_probe.to_dict(),
            "log_state": log_state,
            "terminal_state": "none",
            "running": process_state == "alive",
            "process_exit_known": process_state == "exited",
            "process_exit_code": None,
            "phase": phase,
            "message": self._rank_progress_message(
                process_state=process_state,
                phase=phase,
                terminal_success=False,
                timeout_stage=timeout_stage,
                log_state=log_state,
            ),
            "rendezvous_ready": bool(baseline["rendezvous_ready"]),
            "training_started": bool(baseline["training_started"]),
            "terminal_success": False,
            "progress_changed": False,
            "timeout_stage": timeout_stage.value if timeout_stage is not None else None,
            "last_progress": str(baseline["last_progress"]),
            "last_update_timestamp": last_update_timestamp,
            "placement": None,
            "forward": None,
            "backward": None,
            "train": None,
            "loss_history": list(baseline["loss_history"]),
            "latest_loss": baseline["latest_loss"],
            "final_loss": baseline["final_loss"],
            "checkpoint_ref": baseline["checkpoint_ref"],
            "marker_parse_errors": [],
            "log_tail": "",
        }

    def _monitor_baseline(
        self,
        context: LauncherContext,
        previous_payload: dict[str, Any] | None,
    ) -> dict[str, Any]:
        current = context.job_status
        previous = previous_payload or {}
        previous_phase = previous.get("phase")
        previous_progress = previous.get("last_progress")
        previous_rendezvous = previous.get("rendezvous_ready") is True
        previous_training = previous.get("training_started") is True
        rendezvous_ready = previous_rendezvous or (
            current is not None and current.phase in {"rendezvous", "training", "checkpoint"}
        )
        training_started = previous_training or (
            current is not None and current.phase in {"training", "checkpoint"}
        )
        phase = "training" if training_started else "rendezvous" if rendezvous_ready else "launch"
        if (
            phase == "launch"
            and isinstance(previous_phase, str)
            and previous_phase in {"launch", "rendezvous", "training", "checkpoint"}
        ):
            phase = previous_phase
        last_progress = (
            str(previous_progress)
            if isinstance(previous_progress, str) and previous_progress.strip()
            else phase
        )
        loss_history = self._coerce_loss_history(previous.get("loss_history"))
        if not loss_history and current is not None:
            loss_history = list(current.loss_history)
        latest_loss = self._coerce_finite_float(previous.get("latest_loss"))
        if latest_loss is None and current is not None:
            latest_loss = current.latest_loss
        final_loss = self._coerce_finite_float(previous.get("final_loss"))
        if final_loss is None and current is not None:
            final_loss = self._coerce_finite_float(current.final_metrics.get("final_loss"))
        checkpoint_ref = previous.get("checkpoint_ref")
        if (
            not isinstance(checkpoint_ref, str) or not checkpoint_ref.strip()
        ) and current is not None:
            checkpoint_ref = current.checkpoint_ref
        last_update_timestamp = previous.get("last_update_timestamp")
        if not isinstance(last_update_timestamp, str) or not last_update_timestamp.strip():
            last_update_timestamp = previous.get("observed_at")
        log_tail = previous.get("log_tail")
        return {
            "phase": phase,
            "last_progress": last_progress,
            "rendezvous_ready": rendezvous_ready,
            "training_started": training_started,
            "loss_history": loss_history,
            "latest_loss": latest_loss,
            "final_loss": final_loss,
            "checkpoint_ref": checkpoint_ref if isinstance(checkpoint_ref, str) else None,
            "last_update_timestamp": (
                last_update_timestamp if isinstance(last_update_timestamp, str) else None
            ),
            "log_tail": log_tail if isinstance(log_tail, str) else "",
        }

    def _progress_changed(
        self,
        *,
        baseline: dict[str, Any],
        phase: str,
        last_progress: str,
        checkpoint_ref: str | None,
        final_loss: float | None,
        loss_history: list[float],
        terminal_success: bool,
        log_tail: str,
    ) -> bool:
        return any(
            (
                phase != str(baseline["phase"]),
                last_progress != str(baseline["last_progress"]),
                checkpoint_ref != baseline["checkpoint_ref"],
                final_loss != baseline["final_loss"],
                loss_history != list(baseline["loss_history"]),
                terminal_success is True and baseline.get("terminal_success") is not True,
                log_tail[-4000:] != str(baseline["log_tail"]),
            )
        )

    def _coerce_loss_history(self, value: Any) -> list[float]:
        if not isinstance(value, list):
            return []
        values: list[float] = []
        for item in value:
            number = self._coerce_finite_float(item)
            if number is not None:
                values.append(number)
        return values

    def _coerce_finite_float(self, value: Any) -> float | None:
        if not isinstance(value, (int, float)):
            return None
        number = float(value)
        if number != number or number in {float("inf"), float("-inf")}:
            return None
        return number

    def _log_state(self, result: ProcessResult) -> str:
        if result.ok:
            return "available"
        if result.timed_out:
            return "transient_timeout"
        if self._is_missing_log_error(result.stderr or result.stdout):
            return "missing"
        return "failed"

    def _validate_launch_assignment(
        self,
        context: LauncherContext,
        worker: WorkerConfig,
        assignment,
    ) -> FailureRecord | None:
        master = context.execution_plan.master
        if not master.address.strip():
            return self._launcher_failure(
                stage=FailureStage.LAUNCH,
                worker=worker,
                command=assignment.launch_command,
                message="launch master address is missing from the execution plan",
                recommended_action="populate execution_plan.master.address before launch",
            )
        if master.port <= 0:
            return self._launcher_failure(
                stage=FailureStage.LAUNCH,
                worker=worker,
                command=assignment.launch_command,
                message="launch master port is invalid in the execution plan",
                recommended_action="populate a valid execution_plan.master.port before launch",
            )
        if context.execution_plan.world_size <= 0:
            return self._launcher_failure(
                stage=FailureStage.LAUNCH,
                worker=worker,
                command=assignment.launch_command,
                message="launch world_size is invalid in the execution plan",
                recommended_action="populate a valid execution_plan.world_size before launch",
            )
        return None

    def _remote_code_root(self, remote_root: str) -> str:
        return str(PurePosixPath(remote_root) / "code")

    def _launch_log_path(
        self,
        context: LauncherContext,
        worker: WorkerConfig,
        assignment,
        remote_root: str,
    ) -> str:
        if assignment.log_path:
            path = PurePosixPath(assignment.log_path)
            if path.is_absolute():
                return str(path)
            return str(PurePosixPath(remote_root) / path)
        return str(
            PurePosixPath(remote_root) / str(worker.worker_id) / f"rank{assignment.rank}.log"
        )

    def _launch_argv(
        self,
        context: LauncherContext,
        worker: WorkerConfig,
        assignment,
        remote_code_root: str,
    ) -> tuple[str, ...]:
        argv = shlex.split(assignment.launch_command or "")
        if not argv:
            return ()
        entrypoint = self._entrypoint_from_assignment(assignment.launch_command)
        if entrypoint is None:
            return tuple(argv)
        remote_entrypoint = str(PurePosixPath(remote_code_root, entrypoint))
        if "python" in Path(argv[0]).name.lower():
            resolved = [self._python_executable(worker), remote_entrypoint, *argv[2:]]
        else:
            resolved = [remote_entrypoint, *argv[1:]]
        return tuple(resolved)

    def _launch_env(
        self,
        context: LauncherContext,
        assignment,
        log_path: str,
        remote_code_root: str,
    ) -> dict[str, str]:
        env = dict(context.execution_plan.environment)
        env.update(assignment.environment)
        network_env = self._network_env(context, assignment)
        env.update(
            {
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONUNBUFFERED": "1",
                "PYTHONPATH": remote_code_root,
                "RANK": str(assignment.rank),
                "WORLD_SIZE": str(context.execution_plan.world_size),
                "LOCAL_RANK": str(assignment.local_rank),
                "MASTER_ADDR": context.execution_plan.master.address,
                "MASTER_PORT": str(context.execution_plan.master.port),
                "SHARDGRID_JOB_ID": str(context.job.job_id),
                "SHARDGRID_STAGE": assignment.stage or "",
                "SHARDGRID_LOG_PATH": log_path,
                _LAUNCHER_OWNS_LOG_ENV: "1",
                "CUDA_VISIBLE_DEVICES": str(assignment.gpu_index),
                **network_env,
            }
        )
        backend = str(context.execution_plan.backend)
        env.setdefault("SHARDGRID_BACKEND", backend)
        env.setdefault("GLOO_SOCKET_IFNAME", env.get("GLOO_SOCKET_IFNAME", ""))
        env.setdefault("NCCL_SOCKET_IFNAME", env.get("NCCL_SOCKET_IFNAME", ""))
        return env

    def _network_env(self, context: LauncherContext, assignment) -> dict[str, str]:
        state = context.cluster_state.network_state
        if state is None:
            return {}
        worker_id = str(assignment.worker_id)
        interface = state.selected_interfaces.get(worker_id)
        source_ip = ""
        peer_ip = ""
        for link in state.links:
            if str(link.source_worker_id) != worker_id:
                continue
            interface = interface or link.interface
            source_ip = link.source_ip or source_ip
            peer_ip = link.target_ip or peer_ip
            break
        env: dict[str, str] = {}
        if interface:
            env["NCCL_SOCKET_IFNAME"] = interface
            env["GLOO_SOCKET_IFNAME"] = interface
            env["SHARDGRID_NETWORK_INTERFACE"] = interface
        env["NCCL_SOCKET_FAMILY"] = "AF_INET"
        env["NCCL_IB_DISABLE"] = "1"
        if source_ip:
            env["SHARDGRID_NETWORK_SOURCE_IP"] = source_ip
        if peer_ip:
            env["SHARDGRID_NETWORK_PEER_IP"] = peer_ip
        return env

    def _failure_status(self, result: ProcessResult) -> LauncherResultStatus:
        if self._is_auth_problem(result.stderr):
            return LauncherResultStatus.BLOCKED
        return LauncherResultStatus.FAILED

    def _is_auth_problem(self, text: str | None) -> bool:
        lowered = (text or "").lower()
        return any(
            marker in lowered
            for marker in (
                "permission denied",
                "authentication failed",
                "publickey",
                "host key verification failed",
            )
        )

    def _evidence_ref(self, context: LauncherContext, phase: str, worker: WorkerConfig) -> str:
        base = context.snapshot.diagnostics_path if context.snapshot else "diagnostics"
        return str(PurePosixPath(base) / f"{phase}-{worker.worker_id}.json")
