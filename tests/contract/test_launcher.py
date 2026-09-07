from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

import pytest

from shardgrid.common.enums import (
    BackendStatus,
    FailureStage,
    Health,
    JobState,
    PhysicalOS,
    RuntimeOS,
)
from shardgrid.common.models import (
    as_backend_name,
    as_engine_name,
    as_hostname,
    as_job_id,
    as_worker_id,
)
from shardgrid.control.resource_manager import ResourceManager
from shardgrid.jobs.models import FailureRecord, JobSnapshot, JobStatus, TrainingJob
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
from shardgrid.planner.models import ExecutionPlan, MasterMetadata, WorkerAssignment
from shardgrid.resources.models import NetworkLink, NetworkState, WorkerResource


def _job() -> TrainingJob:
    return TrainingJob(
        job_id=as_job_id("job-0092"),
        config_path="examples/train-minimal.yaml",
        model="tiny-sequential",
        requested_world_size=2,
        backend_preference=as_backend_name("ssh"),
        runtime_environment_ref="env:cluster/shardgrid",
        created_at="2026-08-27T11:00:00+00:00",
        updated_at="2026-08-27T11:00:00+00:00",
    )


def _snapshot() -> JobSnapshot:
    return JobSnapshot(
        job_id=as_job_id("job-0092"),
        root_path="jobs/job-0092",
        code_path="jobs/job-0092/code",
        config_path="jobs/job-0092/config",
        plan_path="jobs/job-0092/plan",
        logs_path="jobs/job-0092/logs",
        environment_path="jobs/job-0092/environment",
        checkpoint_path="jobs/job-0092/checkpoint",
        diagnostics_path="jobs/job-0092/diagnostics",
        created_at="2026-08-27T11:00:00+00:00",
    )


def _workers() -> list[WorkerResource]:
    return [
        WorkerResource(
            worker_id=as_worker_id("gpu4060"),
            hostname=as_hostname("ldj"),
            physical_os=PhysicalOS.WINDOWS,
            runtime_os=RuntimeOS.WSL2_LINUX,
            conda_environment="shardgrid",
            conda_prefix="/home/shardgrid/miniconda3/envs/shardgrid",
            python_executable="/home/shardgrid/miniconda3/envs/shardgrid/bin/python",
            ip="10.87.5.155",
            gpu_name="NVIDIA GeForce RTX 4060 Laptop GPU",
            gpu_total_memory=8188,
            gpu_free_memory=7000,
            torch_version="2.7.1+cu118",
            torch_cuda_version="11.8",
            cuda_version="11.8",
            nccl_available=True,
            gloo_available=True,
            health=Health.HEALTHY,
            last_probe_at="2026-08-27T11:00:00+00:00",
        ),
        WorkerResource(
            worker_id=as_worker_id("gpu1060"),
            hostname=as_hostname("laptop-5g3quogm"),
            physical_os=PhysicalOS.WINDOWS,
            runtime_os=RuntimeOS.WSL2_LINUX,
            conda_environment="shardgrid",
            conda_prefix="/home/shardgrid/miniconda3/envs/shardgrid",
            python_executable="/home/shardgrid/miniconda3/envs/shardgrid/bin/python",
            ip="10.87.5.15",
            gpu_name="NVIDIA GeForce GTX 1650",
            gpu_total_memory=4096,
            gpu_free_memory=3500,
            torch_version="2.7.1+cu118",
            torch_cuda_version="11.8",
            cuda_version="11.8",
            nccl_available=True,
            gloo_available=True,
            health=Health.HEALTHY,
            last_probe_at="2026-08-27T11:00:00+00:00",
        ),
    ]


def _network() -> NetworkState:
    return NetworkState(
        network_id="mvp-pair",
        workers=[as_worker_id("gpu4060"), as_worker_id("gpu1060")],
        links=[
            NetworkLink(
                source_worker_id=as_worker_id("gpu4060"),
                target_worker_id=as_worker_id("gpu1060"),
                source_ip="10.87.5.155",
                target_ip="10.87.5.15",
                interface="eth3",
                tcp_reachable=True,
                measured_at="2026-08-27T11:00:00+00:00",
            ),
            NetworkLink(
                source_worker_id=as_worker_id("gpu1060"),
                target_worker_id=as_worker_id("gpu4060"),
                source_ip="10.87.5.15",
                target_ip="10.87.5.155",
                interface="eth0",
                tcp_reachable=True,
                measured_at="2026-08-27T11:00:00+00:00",
            ),
        ],
        created_at="2026-08-27T11:00:00+00:00",
    )


def _execution_plan() -> ExecutionPlan:
    return ExecutionPlan(
        job_id=as_job_id("job-0092"),
        engine=as_engine_name("torchrun"),
        backend=as_backend_name("ssh"),
        world_size=2,
        master=MasterMetadata(address="10.87.5.155", port=29500),
        workers=[
            WorkerAssignment(
                worker_id=as_worker_id("gpu4060"),
                rank=0,
                stage="0",
                conda_environment="shardgrid",
                conda_prefix="/home/shardgrid/miniconda3/envs/shardgrid",
                python_executable="/home/shardgrid/miniconda3/envs/shardgrid/bin/python",
            ),
            WorkerAssignment(
                worker_id=as_worker_id("gpu1060"),
                rank=1,
                stage="1",
                conda_environment="shardgrid",
                conda_prefix="/home/shardgrid/miniconda3/envs/shardgrid",
                python_executable="/home/shardgrid/miniconda3/envs/shardgrid/bin/python",
            ),
        ],
        snapshot_ref="jobs/job-0092",
    )


def _context() -> LauncherContext:
    cluster_state = ResourceManager().build_cluster_state(
        _workers(),
        network_state=_network(),
        require_network=True,
        now=datetime(2026, 8, 27, 11, 0, tzinfo=UTC),
    )
    return LauncherContext(
        job=_job(),
        execution_plan=_execution_plan(),
        cluster_state=cluster_state,
        snapshot=_snapshot(),
        job_status=JobStatus(job_id=as_job_id("job-0092"), state=JobState.CREATED, phase="created"),
        runtime_environment_refs={"0": "env:gpu4060/shardgrid", "1": "env:gpu1060/shardgrid"},
    )


def _failure(stage: FailureStage, worker_id: str, message: str) -> FailureRecord:
    return FailureRecord(
        stage=stage,
        host=worker_id,
        worker_id=as_worker_id(worker_id),
        command=f"{stage.value.lower()} {worker_id}",
        exit_code=1,
        runtime_environment={"worker": worker_id, "runtime": "wsl2"},
        python_executable="/home/shardgrid/miniconda3/envs/shardgrid/bin/python",
        conda_environment="shardgrid",
        conda_prefix="/home/shardgrid/miniconda3/envs/shardgrid",
        message=message,
        recommended_action=f"inspect {worker_id} logs",
    )


class FakeLauncher(Launcher):
    def __init__(
        self,
        backend: str,
        *,
        partial_prepare: bool = False,
        partial_launch: bool = False,
        unsupported_logs: bool = False,
    ) -> None:
        supported = tuple(LauncherOperation)
        if unsupported_logs:
            supported = tuple(op for op in LauncherOperation if op is not LauncherOperation.LOGS)
        self.capabilities = LauncherCapabilities(
            backend=backend,
            status=BackendStatus.AVAILABLE if not unsupported_logs else BackendStatus.EXPERIMENTAL,
            supported_operations=supported,
            limitations=() if not unsupported_logs else ("logs streaming unavailable",),
        )
        self.partial_prepare = partial_prepare
        self.partial_launch = partial_launch
        self.unsupported_logs = unsupported_logs
        self._prepared: set[str] = set()
        self._distributed: set[str] = set()
        self._launched: set[str] = set()
        self._stopped: set[str] = set()
        self._cleaned: set[str] = set()

    def prepare(self, context: LauncherContext) -> LauncherResult:
        if context.job.job_id in self._prepared:
            return self._noop(LauncherOperation.PREPARE, context.job.job_id, JobState.PROBING)
        worker_results = [
            WorkerResult(
                worker_id="gpu4060",
                status=LauncherResultStatus.SUCCESS,
                evidence_ref="jobs/job-0092/diagnostics/prepare-gpu4060.json",
                message="runtime ready",
            ),
            WorkerResult(
                worker_id="gpu1060",
                status=(
                    LauncherResultStatus.FAILED
                    if self.partial_prepare
                    else LauncherResultStatus.SUCCESS
                ),
                evidence_ref="jobs/job-0092/diagnostics/prepare-gpu1060.json",
                failure=(
                    _failure(FailureStage.LAUNCH, "gpu1060", "prepare failed")
                    if self.partial_prepare
                    else None
                ),
                message="runtime ready" if not self.partial_prepare else "conda missing",
            ),
        ]
        if not self.partial_prepare:
            self._prepared.add(str(context.job.job_id))
        return LauncherResult.from_worker_results(
            operation=LauncherOperation.PREPARE,
            backend=self.capabilities.backend,
            job_id=str(context.job.job_id),
            worker_results=worker_results,
            next_job_state=JobState.PROBING,
        )

    def distribute(self, context: LauncherContext) -> LauncherResult:
        if context.job.job_id in self._distributed:
            return self._noop(
                LauncherOperation.DISTRIBUTE,
                context.job.job_id,
                JobState.DISTRIBUTING,
            )
        self._distributed.add(str(context.job.job_id))
        return LauncherResult.from_worker_results(
            operation=LauncherOperation.DISTRIBUTE,
            backend=self.capabilities.backend,
            job_id=str(context.job.job_id),
            worker_results=[
                WorkerResult(
                    worker_id="gpu4060",
                    status=LauncherResultStatus.SUCCESS,
                    evidence_ref="jobs/job-0092/plan/distribution-gpu4060.json",
                    message="snapshot present",
                ),
                WorkerResult(
                    worker_id="gpu1060",
                    status=LauncherResultStatus.SUCCESS,
                    evidence_ref="jobs/job-0092/plan/distribution-gpu1060.json",
                    message="snapshot present",
                ),
            ],
            next_job_state=JobState.DISTRIBUTING,
        )

    def launch(self, context: LauncherContext) -> LauncherResult:
        if context.job.job_id in self._launched:
            return LauncherResult.from_worker_results(
                operation=LauncherOperation.LAUNCH,
                backend=self.capabilities.backend,
                job_id=str(context.job.job_id),
                worker_results=[
                    WorkerResult(
                        worker_id="gpu4060",
                        status=LauncherResultStatus.NOOP,
                        rank_results=(
                            RankResult(rank=0, worker_id="gpu4060", stage="0", pid=4100),
                        ),
                        message="already launched",
                    ),
                    WorkerResult(
                        worker_id="gpu1060",
                        status=LauncherResultStatus.NOOP,
                        rank_results=(
                            RankResult(rank=1, worker_id="gpu1060", stage="1", pid=4200),
                        ),
                        message="already launched",
                    ),
                ],
                next_job_state=JobState.LAUNCHING,
            )
        worker_results = [
            WorkerResult(
                worker_id="gpu4060",
                status=LauncherResultStatus.SUCCESS,
                rank_results=(
                    RankResult(
                        rank=0,
                        worker_id="gpu4060",
                        stage="0",
                        pid=4100,
                        log_ref="jobs/job-0092/logs/rank0.log",
                    ),
                ),
                evidence_ref="jobs/job-0092/diagnostics/launch-gpu4060.json",
            ),
            WorkerResult(
                worker_id="gpu1060",
                status=(
                    LauncherResultStatus.FAILED
                    if self.partial_launch
                    else LauncherResultStatus.SUCCESS
                ),
                rank_results=(
                    RankResult(
                        rank=1,
                        worker_id="gpu1060",
                        stage="1",
                        pid=None if self.partial_launch else 4200,
                        log_ref="jobs/job-0092/logs/rank1.log",
                        status=(
                            LauncherResultStatus.FAILED
                            if self.partial_launch
                            else LauncherResultStatus.SUCCESS
                        ),
                        failure=(
                            _failure(FailureStage.LAUNCH, "gpu1060", "rank launch failed")
                            if self.partial_launch
                            else None
                        ),
                    ),
                ),
                evidence_ref="jobs/job-0092/diagnostics/launch-gpu1060.json",
                failure=(
                    _failure(FailureStage.LAUNCH, "gpu1060", "rank launch failed")
                    if self.partial_launch
                    else None
                ),
            ),
        ]
        if not self.partial_launch:
            self._launched.add(str(context.job.job_id))
        return LauncherResult.from_worker_results(
            operation=LauncherOperation.LAUNCH,
            backend=self.capabilities.backend,
            job_id=str(context.job.job_id),
            worker_results=worker_results,
            next_job_state=JobState.LAUNCHING,
        )

    def monitor(self, context: LauncherContext) -> LauncherResult:
        return LauncherResult.from_worker_results(
            operation=LauncherOperation.MONITOR,
            backend=self.capabilities.backend,
            job_id=str(context.job.job_id),
            worker_results=[
                WorkerResult(
                    worker_id="gpu4060",
                    status=LauncherResultStatus.SUCCESS,
                    rank_results=(
                        RankResult(
                            rank=0,
                            worker_id="gpu4060",
                            stage="0",
                            pid=4100,
                            message="running",
                        ),
                    ),
                ),
                WorkerResult(
                    worker_id="gpu1060",
                    status=LauncherResultStatus.SUCCESS,
                    rank_results=(
                        RankResult(
                            rank=1,
                            worker_id="gpu1060",
                            stage="1",
                            pid=4200,
                            message="running",
                        ),
                    ),
                ),
            ],
            next_job_state=JobState.TRAINING,
        )

    def logs(self, context: LauncherContext) -> LauncherResult:
        if self.unsupported_logs:
            return LauncherResult(
                operation=LauncherOperation.LOGS,
                status=LauncherResultStatus.UNSUPPORTED,
                backend=self.capabilities.backend,
                job_id=str(context.job.job_id),
                blocker="logs streaming unavailable",
                duplicate_safe=True,
                idempotent=True,
                message="backend does not support logs",
            )
        return LauncherResult(
            operation=LauncherOperation.LOGS,
            status=LauncherResultStatus.SUCCESS,
            backend=self.capabilities.backend,
            job_id=str(context.job.job_id),
            log_results=(
                LogResult(
                    worker_id="gpu4060",
                    rank=0,
                    stage="0",
                    source="remote",
                    location="jobs/job-0092/logs/rank0.log",
                    tail="rank0 ok",
                ),
                LogResult(
                    worker_id="gpu1060",
                    rank=1,
                    stage="1",
                    source="remote",
                    location="jobs/job-0092/logs/rank1.log",
                    tail="rank1 ok",
                ),
            ),
            next_job_state=context.job_status.state if context.job_status else None,
        )

    def stop(self, context: LauncherContext) -> LauncherResult:
        if context.job.job_id in self._stopped:
            return self._noop(LauncherOperation.STOP, context.job.job_id, JobState.STOPPED)
        self._stopped.add(str(context.job.job_id))
        return LauncherResult.from_worker_results(
            operation=LauncherOperation.STOP,
            backend=self.capabilities.backend,
            job_id=str(context.job.job_id),
            worker_results=[
                WorkerResult(
                    worker_id="gpu4060",
                    status=LauncherResultStatus.SUCCESS,
                    message="stopped",
                ),
                WorkerResult(
                    worker_id="gpu1060",
                    status=LauncherResultStatus.SUCCESS,
                    message="stopped",
                ),
            ],
            next_job_state=JobState.STOPPED,
        )

    def cleanup(self, context: LauncherContext) -> LauncherResult:
        if context.job.job_id in self._cleaned:
            return self._noop(LauncherOperation.CLEANUP, context.job.job_id, None)
        self._cleaned.add(str(context.job.job_id))
        return LauncherResult.from_worker_results(
            operation=LauncherOperation.CLEANUP,
            backend=self.capabilities.backend,
            job_id=str(context.job.job_id),
            worker_results=[
                WorkerResult(
                    worker_id="gpu4060",
                    status=LauncherResultStatus.SUCCESS,
                    message="job-scoped temp files removed",
                ),
                WorkerResult(
                    worker_id="gpu1060",
                    status=LauncherResultStatus.SUCCESS,
                    message="job-scoped temp files removed",
                ),
            ],
        )

    def _noop(
        self,
        operation: LauncherOperation,
        job_id: object,
        next_job_state: JobState | None,
    ) -> LauncherResult:
        return LauncherResult(
            operation=operation,
            status=LauncherResultStatus.NOOP,
            backend=self.capabilities.backend,
            job_id=str(job_id),
            idempotent=True,
            duplicate_safe=True,
            next_job_state=next_job_state,
            message="already satisfied",
        )


@pytest.mark.parametrize("backend", ["ssh", "kubernetes", "volcano"])
def test_fake_backends_implement_same_contract(backend: str) -> None:
    launcher = FakeLauncher(backend)
    context = _context()

    prepare = launcher.prepare(context)
    distribute = launcher.distribute(context)
    launch = launcher.launch(context)
    monitor = launcher.monitor(context)
    logs = launcher.logs(context)
    stop = launcher.stop(context)
    cleanup = launcher.cleanup(context)

    assert prepare.status is LauncherResultStatus.SUCCESS
    assert distribute.status is LauncherResultStatus.SUCCESS
    assert launch.status is LauncherResultStatus.SUCCESS
    assert monitor.status is LauncherResultStatus.SUCCESS
    assert logs.status is LauncherResultStatus.SUCCESS
    assert stop.status is LauncherResultStatus.SUCCESS
    assert cleanup.status is LauncherResultStatus.SUCCESS
    assert launch.next_job_state is JobState.LAUNCHING
    assert monitor.next_job_state is JobState.TRAINING
    assert stop.next_job_state is JobState.STOPPED


def test_prepare_partial_failure_preserves_success_evidence_and_failure_record() -> None:
    result = FakeLauncher("ssh", partial_prepare=True).prepare(_context())

    assert result.status is LauncherResultStatus.PARTIAL
    assert result.worker_results[0].status is LauncherResultStatus.SUCCESS
    assert result.worker_results[0].evidence_ref is not None
    assert result.worker_results[1].status is LauncherResultStatus.FAILED
    assert result.worker_results[1].failure is not None
    assert result.worker_results[1].failure.stage is FailureStage.LAUNCH
    assert result.failure is not None
    assert result.failure.recommended_action == "inspect gpu1060 logs"


def test_launch_partial_failure_keeps_per_rank_evidence() -> None:
    result = FakeLauncher("ssh", partial_launch=True).launch(_context())

    assert result.status is LauncherResultStatus.PARTIAL
    assert result.worker_results[0].rank_results[0].pid == 4100
    assert result.worker_results[1].rank_results[0].status is LauncherResultStatus.FAILED
    assert result.worker_results[1].rank_results[0].failure is not None


def test_duplicate_prepare_distribute_launch_stop_and_cleanup_are_idempotent() -> None:
    launcher = FakeLauncher("ssh")
    context = _context()

    assert launcher.prepare(context).status is LauncherResultStatus.SUCCESS
    assert launcher.prepare(context).status is LauncherResultStatus.NOOP
    assert launcher.distribute(context).status is LauncherResultStatus.SUCCESS
    assert launcher.distribute(context).status is LauncherResultStatus.NOOP
    first_launch = launcher.launch(context)
    second_launch = launcher.launch(context)
    assert first_launch.status is LauncherResultStatus.SUCCESS
    assert second_launch.status is LauncherResultStatus.NOOP
    assert [
        rank.pid
        for worker in second_launch.worker_results
        for rank in worker.rank_results
    ] == [4100, 4200]
    assert launcher.stop(context).status is LauncherResultStatus.SUCCESS
    assert launcher.stop(context).status is LauncherResultStatus.NOOP
    assert launcher.cleanup(context).status is LauncherResultStatus.SUCCESS
    assert launcher.cleanup(context).status is LauncherResultStatus.NOOP


def test_logs_and_monitor_are_read_only_and_structured() -> None:
    launcher = FakeLauncher("ssh")
    context = _context()
    original_status = context.job_status

    monitor = launcher.monitor(context)
    logs = launcher.logs(context)

    assert context.job_status == original_status
    assert monitor.worker_results[0].rank_results[0].message == "running"
    assert logs.log_results[0].worker_id == "gpu4060"
    assert logs.log_results[0].rank == 0
    assert logs.log_results[0].tail == "rank0 ok"


def test_unsupported_operation_is_not_reported_as_success() -> None:
    result = FakeLauncher("kubernetes", unsupported_logs=True).logs(_context())

    assert result.status is LauncherResultStatus.UNSUPPORTED
    assert result.ok is False
    assert result.blocker == "logs streaming unavailable"


def test_launcher_result_serializes_backend_neutral_fields() -> None:
    result = FakeLauncher("volcano").launch(_context())
    payload = result.to_dict()

    assert payload["backend"] == "volcano"
    assert "ssh_host" not in payload
    assert payload["worker_results"][0]["rank_results"][0]["rank"] == 0
    assert payload["next_job_state"] == "launching"
    assert payload["worker_results"][0]["rank_results"][0]["log_ref"].endswith("rank0.log")


def test_launcher_context_reuses_existing_models() -> None:
    context = _context()
    copied = replace(context, backend_config={"mode": "test"})

    assert copied.execution_plan.job_id == copied.job.job_id
    assert copied.cluster_state.eligible_workers
    assert copied.snapshot is not None
    assert copied.runtime_environment_refs["0"] == "env:gpu4060/shardgrid"
