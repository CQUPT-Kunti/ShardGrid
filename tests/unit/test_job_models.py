from __future__ import annotations

import json

import pytest

from shardgrid.common.enums import BackendStatus, FailureStage, JobState
from shardgrid.common.models import as_backend_name, as_engine_name, as_job_id, as_worker_id
from shardgrid.engines.models import (
    CompatibilitySpikeReport,
    GPUShare,
    ParallelEngineCandidate,
    ParallelPlan,
)
from shardgrid.jobs.models import (
    EnvironmentSnapshot,
    FailureRecord,
    JobSnapshot,
    JobStatus,
    TrainingJob,
    TrainingResult,
)
from shardgrid.planner.models import (
    ExecutionPlan,
    MasterMetadata,
    PlatformAdapterState,
    WorkerAssignment,
)


def test_execution_plan_round_trip_and_constraints() -> None:
    plan = ExecutionPlan(
        job_id=as_job_id("job-0001"),
        engine=as_engine_name("galvatron"),
        backend=as_backend_name("ssh"),
        world_size=2,
        master=MasterMetadata(address="10.0.0.13", port=29500),
        workers=[
            WorkerAssignment(
                worker_id=as_worker_id("gpu4060"),
                rank=0,
                stage="0",
                conda_environment="shardgrid-worker",
                python_executable="python",
            ),
            WorkerAssignment(
                worker_id=as_worker_id("gpu1060"),
                rank=1,
                stage="1",
                conda_environment="shardgrid-worker",
                python_executable="python",
            ),
        ],
        conda_environment="shardgrid-worker",
        python_executable="python",
        labels={"backend_result": "gloo_fallback"},
    )

    restored = ExecutionPlan.from_dict(json.loads(json.dumps(plan.to_dict())))

    assert restored == plan
    assert restored.world_size == 2
    assert [item.rank for item in restored.workers] == [0, 1]
    assert all(item.local_rank == 0 for item in restored.workers)
    assert restored.conda_environment == "shardgrid-worker"


def test_execution_plan_rejects_duplicate_rank_and_nonzero_local_rank() -> None:
    with pytest.raises(ValueError):
        ExecutionPlan(
            job_id=as_job_id("job-0001"),
            engine=as_engine_name("galvatron"),
            backend=as_backend_name("ssh"),
            world_size=2,
            master=MasterMetadata(address="10.0.0.13", port=29500),
            workers=[
                WorkerAssignment(worker_id=as_worker_id("gpu4060"), rank=0),
                WorkerAssignment(worker_id=as_worker_id("gpu1060"), rank=0),
            ],
        )

    with pytest.raises(ValueError):
        ExecutionPlan(
            job_id=as_job_id("job-0001"),
            engine=as_engine_name("galvatron"),
            backend=as_backend_name("ssh"),
            world_size=1,
            master=MasterMetadata(address="10.0.0.13", port=29500),
            workers=[
                WorkerAssignment(worker_id=as_worker_id("gpu4060"), rank=0, local_rank=1)
            ],
        )


def test_training_job_and_status_lifecycle_transitions() -> None:
    job = TrainingJob(
        job_id=as_job_id("job-0001"),
        config_path="examples/train-minimal.yaml",
        model="tiny-sequential",
        requested_world_size=2,
        backend_preference=as_backend_name("ssh"),
        runtime_environment_ref="env:worker-pair/shardgrid",
    )
    job = job.transition_to(JobState.PROBING)
    job = job.transition_to(JobState.PLANNING)
    assert job.state is JobState.PLANNING
    assert job.runtime_environment_ref == "env:worker-pair/shardgrid"
    assert job.updated_at is not None

    assignments = [
        WorkerAssignment(
            worker_id=as_worker_id("gpu4060"),
            rank=0,
            stage="0",
            conda_environment="shardgrid-worker",
        ),
        WorkerAssignment(
            worker_id=as_worker_id("gpu1060"),
            rank=1,
            stage="1",
            conda_environment="shardgrid-worker",
        ),
    ]
    status = JobStatus(
        job_id=as_job_id("job-0001"),
        state=JobState.CREATED,
        phase="created",
        workers=[as_worker_id("gpu4060"), as_worker_id("gpu1060")],
        assignments=assignments,
        runtime_environment_refs={"0": "env:gpu4060/shardgrid", "1": "env:gpu1060/shardgrid"},
    )
    status = status.transition_to(JobState.PROBING, phase="probe")
    status = status.transition_to(JobState.PLANNING, phase="plan")
    assert status.state is JobState.PLANNING
    assert status.phase == "plan"
    assert status.assignments == assignments


def test_invalid_lifecycle_transitions_are_rejected() -> None:
    job = TrainingJob(
        job_id=as_job_id("job-0001"),
        config_path="examples/train-minimal.yaml",
        model="tiny-sequential",
        requested_world_size=2,
        backend_preference=as_backend_name("ssh"),
        runtime_environment_ref="env:worker-pair/shardgrid",
    )
    with pytest.raises(ValueError):
        job.transition_to(JobState.TRAINING)

    status = JobStatus(job_id=as_job_id("job-0001"), state=JobState.CREATED, phase="created")
    with pytest.raises(ValueError):
        status.transition_to(JobState.COMPLETED, checkpoint_ref="/tmp/ckpt")


def test_failure_and_completion_requirements_are_enforced() -> None:
    failure = FailureRecord(
        stage=FailureStage.LAUNCH,
        host="machine-c.local",
        worker_id=as_worker_id("gpu4060"),
        message="ssh launch failed",
        recommended_action="inspect remote logs and ssh access",
        conda_environment="shardgrid-worker",
        python_executable="python",
        retryable=True,
    )
    failed_status = JobStatus(
        job_id=as_job_id("job-0001"),
        state=JobState.FAILED,
        phase="launch",
        failure=failure,
    )
    completed_status = JobStatus(
        job_id=as_job_id("job-0001"),
        state=JobState.COMPLETED,
        phase="checkpoint",
        checkpoint_ref="jobs/job-0001/checkpoint",
        final_metrics={"final_loss": 0.42},
    )

    assert failed_status.failure == failure
    assert completed_status.checkpoint_ref is not None

    with pytest.raises(ValueError):
        JobStatus(job_id=as_job_id("job-0001"), state=JobState.FAILED, phase="launch")

    with pytest.raises(ValueError):
        JobStatus(job_id=as_job_id("job-0001"), state=JobState.COMPLETED, phase="done")


def test_job_status_rejects_invalid_assignment_runtime_and_metric_data() -> None:
    assignment = WorkerAssignment(worker_id=as_worker_id("gpu4060"), rank=0)

    with pytest.raises(ValueError, match="duplicate ranks"):
        JobStatus(
            job_id=as_job_id("job-0001"),
            state=JobState.CREATED,
            phase="created",
            assignments=[assignment, assignment],
        )

    with pytest.raises(ValueError, match="assignment ranks"):
        JobStatus(
            job_id=as_job_id("job-0001"),
            state=JobState.CREATED,
            phase="created",
            assignments=[assignment],
            runtime_environment_refs={"worker": "env:gpu4060/shardgrid"},
        )

    with pytest.raises(ValueError, match="finite"):
        JobStatus(
            job_id=as_job_id("job-0001"),
            state=JobState.CREATED,
            phase="created",
            loss_history=[0.5, float("nan")],
        )


def test_engine_and_platform_models_cover_experimental_fallback_and_blocked() -> None:
    candidate = ParallelEngineCandidate(
        engine_id="engine-1",
        name=as_engine_name("galvatron"),
        status=BackendStatus.EXPERIMENTAL,
        capabilities=["pipeline_parallel"],
    )
    report = CompatibilitySpikeReport(
        report_id="report-1",
        component="kubernetes",
        stage=FailureStage.SCHEDULE,
        status=BackendStatus.BLOCKED,
        blockers=["cluster not ready"],
        decision="defer to ssh backend",
        recommended_next_action="use ssh MVP until platform gate passes",
    )
    result = TrainingResult(
        job_id=as_job_id("job-0001"),
        forward_success=True,
        activation_transfer_success=True,
        loss_success=True,
        backward_success=True,
        gradient_transfer_success=True,
        optimizer_step_success=True,
        parameters_changed=True,
        initial_loss=2.0,
        final_loss=1.5,
        loss_decrease_percent=25.0,
        checkpoint_path="jobs/job-0001/checkpoint/model.pt",
        backend_label="gloo_fallback",
        status="success",
    )
    platform = PlatformAdapterState(
        adapter_id="wsl",
        platform="wsl2_linux",
        shell="/bin/bash",
        path_rules="posix",
        conda_environment="shardgrid-worker",
        supports_bootstrap=True,
        supports_probe=True,
        manual_action_rules=["admin_required", "reboot_required"],
    )
    share = GPUShare(
        worker_id="gpu4060",
        gpu_index=0,
        total_memory_mb=8192,
        allocated_memory_mb=0,
        free_memory_mb=8192,
        isolation_status="not_enabled",
    )
    snapshot = JobSnapshot(
        job_id=as_job_id("job-0001"),
        root_path="jobs/job-0001",
        code_path="jobs/job-0001/code",
        config_path="jobs/job-0001/config",
        plan_path="jobs/job-0001/plan",
        logs_path="jobs/job-0001/logs",
        environment_path="jobs/job-0001/environment",
        checkpoint_path="jobs/job-0001/checkpoint",
        diagnostics_path="jobs/job-0001/diagnostics",
    )
    parallel_plan = ParallelPlan(
        parallel_plan_id="plan-1",
        engine=as_engine_name("galvatron"),
        engine_plan_path="plans/plan-1.yaml",
        model_name="tiny-sequential",
        world_size=2,
        stages=["0", "1"],
        limitations=["static validation only"],
    )
    environment_snapshot = EnvironmentSnapshot(
        snapshot_id="env-1",
        scope="worker:gpu4060",
        conda_executable="/opt/conda/bin/conda",
        conda_environment="shardgrid-worker",
        conda_prefix="/opt/conda/envs/shardgrid-worker",
        python_executable="python",
        python_version="3.13.5",
        torch_version="2.5.1",
        torch_cuda_version="12.4",
        cuda_version="12.4",
    )

    assert candidate.status is BackendStatus.EXPERIMENTAL
    assert report.status is BackendStatus.BLOCKED
    assert result.backend_label == "gloo_fallback"
    assert platform.supports_probe is True
    assert platform.conda_environment == "shardgrid-worker"
    assert share.isolation_status == "not_enabled"
    assert snapshot.root_path.endswith("job-0001")
    assert parallel_plan.world_size == 2
    assert EnvironmentSnapshot.from_dict(environment_snapshot.to_dict()) == environment_snapshot
