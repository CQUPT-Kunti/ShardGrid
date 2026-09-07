"""Unit tests for Memory Probe live-preflight, result classification, and fallback."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from shardgrid.common.config import ClusterConfig, TrainingConfig, load_training_config
from shardgrid.common.enums import Health, PhysicalOS, RuntimeOS
from shardgrid.common.models import as_backend_name, as_engine_name, as_hostname, as_job_id
from shardgrid.control.job_manager import (
    PROBE_INFRA_FAILURE,
    PROBE_MEMORY_REJECT,
    PROBE_PASS,
    JobManager,
    MemoryProbeInfraFailure,
    MemoryProbeNoFeasiblePlan,
    MemoryProbeResourceChanged,
    MemoryProbeResult,
    MemoryProbeRuntimeFailure,
    create_training_job,
)
from shardgrid.control.resource_manager import ResourceManager
from shardgrid.launchers.base import (
    LauncherOperation,
    LauncherResult,
    LauncherResultStatus,
)
from shardgrid.planner.models import ExecutionPlan, MasterMetadata, WorkerAssignment
from shardgrid.resources.models import NetworkLink, NetworkState, WorkerResource


def _cluster_config(tmp_path: Path) -> ClusterConfig:
    return ClusterConfig.from_dict(
        {
            "control": {"machine_id": "machine-a", "hostname": "control-a.local"},
            "jobs_root": str((tmp_path / "jobs").resolve()),
            "ssh": {},
            "runtime": {
                "conda_environment": "shardgrid",
                "conda_prefix": "/opt/conda/envs/shardgrid",
            },
            "network": {"rendezvous_port": 29500},
            "backend_preference": {},
            "manual_override": {},
            "workers": [
                {
                    "id": "gpu4060",
                    "machine_id": "machine-c",
                    "physical_os": "windows",
                    "runtime_os": "wsl2_linux",
                    "runtime": "wsl2",
                    "host": "10.87.5.155",
                    "ssh_user": "shardgrid",
                    "runtime_distro": "Ubuntu-22.04",
                    "conda_environment": "shardgrid",
                    "conda_prefix": "/opt/conda/envs/shardgrid",
                },
                {
                    "id": "gpu1060",
                    "machine_id": "machine-d",
                    "physical_os": "windows",
                    "runtime_os": "wsl2_linux",
                    "runtime": "wsl2",
                    "host": "10.87.5.15",
                    "ssh_user": "shardgrid",
                    "runtime_distro": "Ubuntu-22.04",
                    "conda_environment": "shardgrid",
                    "conda_prefix": "/opt/conda/envs/shardgrid",
                },
            ],
        }
    )


def _training_config(tmp_path: Path) -> TrainingConfig:
    path = tmp_path / "train-probe.yaml"
    path.write_text(
        """
job:
  name: probe-test
  backend: ssh
  communication_backend: nccl
model:
  name: residual-mlp-dag
  type: generic_dag
  parameters:
    zoo_model: residual_mlp_dag
    width: 4096
resources: {}
planning:
  mode: automatic
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return load_training_config(path)


def _job(tmp_path: Path) -> object:
    config = _training_config(tmp_path)
    return create_training_job(
        config_path=str(tmp_path / "train-probe.yaml"),
        model=config.model.name,
        requested_world_size=2,
        backend_preference=as_backend_name("nccl"),
        runtime_environment_ref="env:cluster/shardgrid",
        job_id=as_job_id("job-probe-launch"),
    )


def _plan(candidate_id: str) -> object:
    from shardgrid.engines.models import (
        ParallelPlan,
        ParallelPlanPlacement,
        ParallelPlanProvenance,
        ParallelPlanStage,
        TrainingMemoryEstimate,
    )

    return ParallelPlan(
        parallel_plan_id=f"plan-{candidate_id}",
        engine=as_engine_name("galvatron"),
        model_name="residual-mlp-dag",
        world_size=2,
        stages=["stage0", "stage1"],
        partition_source="automatic",
        selected_candidate_id=candidate_id,
        stage_metadata=[
            ParallelPlanStage(
                stage_id="stage0",
                rank=0,
                module_ids=("m0",),
                module_paths=("input_proj",),
                start_index=0,
                stop_index=2,
                estimated_peak_training_memory=TrainingMemoryEstimate(
                    estimated_peak_bytes=512 * 1024 * 1024,
                    planner_required_bytes=1024 * 1024 * 1024,
                ),
                placement=ParallelPlanPlacement(
                    worker_id="gpu4060",
                    rank=0,
                    gpu_index=0,
                ),
            ),
            ParallelPlanStage(
                stage_id="stage1",
                rank=1,
                module_ids=("m1",),
                module_paths=("output_head",),
                start_index=2,
                stop_index=4,
                estimated_peak_training_memory=TrainingMemoryEstimate(
                    estimated_peak_bytes=512 * 1024 * 1024,
                    planner_required_bytes=1024 * 1024 * 1024,
                ),
                placement=ParallelPlanPlacement(
                    worker_id="gpu1060",
                    rank=1,
                    gpu_index=0,
                ),
            ),
        ],
        planning_provenance=ParallelPlanProvenance(
            partition_source="automatic",
            selected_candidate_id=candidate_id,
            selected_worker_count=2,
        ),
        requirements={"plan_mode": "automatic", "generic_dag_runtime": "true"},
    )


def _captured_plan(candidate_id: str) -> object:
    plan = _plan(candidate_id)
    return replace(
        plan,
        requirements={
            **plan.requirements,
            "workload_source": "captured_context",
        },
    )


def _network_state() -> NetworkState:
    return NetworkState(
        network_id="pair",
        workers=["gpu4060", "gpu1060"],
        links=[
            NetworkLink(
                source_worker_id="gpu4060",
                target_worker_id="gpu1060",
                source_ip="10.87.5.155",
                target_ip="10.87.5.15",
                interface="eth3",
                tcp_reachable=True,
            ),
            NetworkLink(
                source_worker_id="gpu1060",
                target_worker_id="gpu4060",
                source_ip="10.87.5.15",
                target_ip="10.87.5.155",
                interface="eth0",
                tcp_reachable=True,
            ),
        ],
    )


def _cluster_state() -> object:
    return ResourceManager().build_cluster_state(
        [
            WorkerResource(
                worker_id="gpu4060",
                hostname=as_hostname("10.87.5.155"),
                physical_os=PhysicalOS.WINDOWS,
                runtime_os=RuntimeOS.WSL2_LINUX,
                ip="10.87.5.155",
                gpu_total_memory=8192,
                gpu_free_memory=6144,
                health=Health.HEALTHY,
            ),
            WorkerResource(
                worker_id="gpu1060",
                hostname=as_hostname("10.87.5.15"),
                physical_os=PhysicalOS.WINDOWS,
                runtime_os=RuntimeOS.WSL2_LINUX,
                ip="10.87.5.15",
                gpu_total_memory=4096,
                gpu_free_memory=3072,
                health=Health.HEALTHY,
            ),
        ],
        network_state=_network_state(),
        require_network=True,
    )


def _fake_preflight(fresh_port: int = 31001):
    def fake(*, training_config, execution_plan, cluster_state, network_state):
        del training_config
        updated = replace(
            execution_plan,
            master=MasterMetadata(
                address=execution_plan.master.address,
                port=fresh_port,
            ),
        )
        return updated, cluster_state, network_state

    return fake


def test_captured_workload_launch_uses_internal_generic_bootstrap(tmp_path: Path) -> None:
    manager = JobManager(_cluster_config(tmp_path))
    job = _job(tmp_path)
    snapshot = manager._artifact_store.create_snapshot(job)
    plan = _captured_plan("candidate-C")

    command = manager._launch_command_for_assignment(
        plan,
        0,
        job=job,
        snapshot=snapshot,
    )
    probe_command = manager._launch_command_for_assignment(
        replace(plan, requirements={**plan.requirements, "memory_probe": "true"}),
        0,
        job=job,
        snapshot=snapshot,
    )

    assert "examples/models/train_generic_dag.py" not in command
    assert "examples/models/train_automatic_plan.py" not in command
    assert "python -m shardgrid.runtime.generic_bootstrap" in command
    assert "--plan-artifact" in command
    assert str(Path(snapshot.plan_path) / "original-parallel-plan.json") in command
    assert "--context-artifact" in command
    assert str(Path(snapshot.plan_path) / "captured-context.json") in command
    assert "--job-id job-probe-launch" in command
    assert "--selected-candidate-id candidate-C" in command
    assert probe_command.startswith("python -m shardgrid.runtime.generic_bootstrap")
    assert "--memory-probe" in probe_command


def test_probe_launch_uses_live_plan_fresh_port_not_default(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manager = JobManager(_cluster_config(tmp_path))
    training_config = _training_config(tmp_path)
    job = create_training_job(
        config_path=str(tmp_path / "train-probe.yaml"),
        model=training_config.model.name,
        requested_world_size=2,
        backend_preference=as_backend_name("nccl"),
        runtime_environment_ref="env:cluster/shardgrid",
        job_id=as_job_id("job-live-port"),
    )
    plan = _plan("candidate-B")
    captured: dict[str, object] = {}

    def memory_probe(probe_plan, execution: ExecutionPlan) -> MemoryProbeResult:
        captured["port"] = execution.master.port
        captured["candidate_id"] = execution.labels.get("selected_candidate_id")
        captured["worker_ids"] = [
            str(assignment.worker_id) for assignment in execution.workers
        ]
        captured["launch_commands"] = [
            assignment.launch_command for assignment in execution.workers
        ]
        return MemoryProbeResult(
            PROBE_PASS,
            probe_plan.selected_candidate_id,
            "probe passed",
        )

    manager._memory_probe = memory_probe
    monkeypatch.setattr(manager, "_prepare_live_execution_plan", _fake_preflight(31001))

    selected, selection = manager._select_memory_probe_candidate(
        job=job,
        training_config=training_config,
        candidate_plans=(plan,),
        selected_workers=manager.cluster_config.workers,
        cluster_state=_cluster_state(),
        network_state=_network_state(),
        rejected_engine_ids=(),
    )

    assert selected.selected_candidate_id == "candidate-B"
    assert captured["port"] == 31001
    assert captured["port"] != 29500
    assert captured["candidate_id"] == "candidate-B"
    assert captured["worker_ids"] == ["gpu4060", "gpu1060"]
    assert all("--memory-probe" in command for command in captured["launch_commands"])
    infra = selection["probe_infra_summary"]
    assert infra["fresh_port_per_attempt"] == [31001]
    assert infra["stale_default_port_reuse"] == 0


def test_candidate_identity_preserved_across_live_preflight(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manager = JobManager(_cluster_config(tmp_path))
    training_config = _training_config(tmp_path)
    job = create_training_job(
        config_path=str(tmp_path / "train-probe.yaml"),
        model=training_config.model.name,
        requested_world_size=2,
        backend_preference=as_backend_name("nccl"),
        runtime_environment_ref="env:cluster/shardgrid",
        job_id=as_job_id("job-identity"),
    )
    plan = _plan("candidate-B")
    captured: dict[str, object] = {}

    def memory_probe(probe_plan, execution: ExecutionPlan) -> MemoryProbeResult:
        captured["before"] = {
            "candidate_id": probe_plan.selected_candidate_id,
            "workers": [
                (str(assignment.worker_id), assignment.rank, assignment.stage)
                for assignment in execution.workers
            ],
        }
        return MemoryProbeResult(PROBE_PASS, probe_plan.selected_candidate_id, "ok")

    manager._memory_probe = memory_probe
    monkeypatch.setattr(manager, "_prepare_live_execution_plan", _fake_preflight(31002))

    selected, _selection = manager._select_memory_probe_candidate(
        job=job,
        training_config=training_config,
        candidate_plans=(plan,),
        selected_workers=manager.cluster_config.workers,
        cluster_state=_cluster_state(),
        network_state=_network_state(),
        rejected_engine_ids=(),
    )

    assert selected.selected_candidate_id == "candidate-B"
    assert captured["before"]["candidate_id"] == "candidate-B"
    assert captured["before"]["workers"] == [
        ("gpu4060", 0, "stage0"),
        ("gpu1060", 1, "stage1"),
    ]


def test_candidate_identity_mismatch_rejected(tmp_path: Path) -> None:
    manager = JobManager(_cluster_config(tmp_path))
    plan = _plan("candidate-B")
    live = ExecutionPlan(
        job_id=as_job_id("job-identity-mismatch"),
        engine=as_engine_name("galvatron"),
        backend=as_backend_name("nccl"),
        world_size=2,
        master=MasterMetadata(address="10.87.5.155", port=31000),
        workers=[
            WorkerAssignment(worker_id="gpu4060", rank=0, stage="stage0"),
            WorkerAssignment(worker_id="gpu1060", rank=1, stage="stage1"),
        ],
        labels={"selected_candidate_id": "candidate-B"},
    )
    assert manager._candidate_identity_preserved(plan, live) is None

    tampered = replace(
        live,
        workers=[
            WorkerAssignment(worker_id="gpu4060", rank=0, stage="stage0"),
            WorkerAssignment(worker_id="gpu4060", rank=1, stage="stage1"),
        ],
    )
    assert manager._candidate_identity_preserved(plan, tampered) is not None

    relabeled = replace(live, labels={"selected_candidate_id": "candidate-C"})
    assert manager._candidate_identity_preserved(plan, relabeled) is not None


class FakeProbeLauncher:
    def __init__(
        self,
        *,
        rendezvous_ready: bool,
        training_started: bool,
        log_tail: str,
        timeout_stage: str | None = None,
        terminal_success: bool = False,
    ) -> None:
        self.rendezvous_ready = rendezvous_ready
        self.training_started = training_started
        self.log_tail = log_tail
        self.timeout_stage = timeout_stage
        self.terminal_success = terminal_success

    def _result(self, operation, context, status=LauncherResultStatus.SUCCESS):
        return LauncherResult(
            operation=operation,
            status=status,
            backend="ssh",
            job_id=str(context.job.job_id),
        )

    def prepare(self, context) -> LauncherResult:
        return self._result(LauncherOperation.PREPARE, context)

    def distribute(self, context) -> LauncherResult:
        return self._result(LauncherOperation.DISTRIBUTE, context)

    def launch(self, context) -> LauncherResult:
        return self._result(LauncherOperation.LAUNCH, context)

    def monitor(self, context) -> LauncherResult:
        diag = Path(context.snapshot.diagnostics_path)
        diag.mkdir(parents=True, exist_ok=True)
        for assignment in context.execution_plan.workers:
            payload = {
                "worker_id": str(assignment.worker_id),
                "rank": assignment.rank,
                "stage": assignment.stage,
                "phase": (
                    "training"
                    if self.training_started
                    else "rendezvous"
                    if self.rendezvous_ready
                    else "launch"
                ),
                "rendezvous_ready": self.rendezvous_ready,
                "training_started": self.training_started,
                "terminal_success": self.terminal_success,
                "timeout_stage": self.timeout_stage,
                "log_tail": self.log_tail,
                "message": "probe failed",
            }
            path = diag / f"monitor-{assignment.worker_id}-rank{assignment.rank}.json"
            path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return self._result(LauncherOperation.MONITOR, context, LauncherResultStatus.FAILED)

    def stop(self, context) -> LauncherResult:
        return self._result(LauncherOperation.STOP, context, LauncherResultStatus.NOOP)

    def cleanup(self, context) -> LauncherResult:
        return self._result(LauncherOperation.CLEANUP, context, LauncherResultStatus.NOOP)


def _run_probe_with_launcher(
    manager: JobManager,
    tmp_path: Path,
    launcher: FakeProbeLauncher,
    *,
    monkeypatch,
) -> MemoryProbeResult:
    training_config = _training_config(tmp_path)
    probe_job = create_training_job(
        config_path=str(tmp_path / "train-probe.yaml"),
        model=training_config.model.name,
        requested_world_size=2,
        backend_preference=as_backend_name("nccl"),
        runtime_environment_ref="env:cluster/shardgrid",
        job_id=as_job_id("job-probe-run"),
    )
    probe_snapshot = manager._artifact_store.create_snapshot(probe_job)
    plan = _plan("candidate-B")
    probe_plan = replace(
        plan,
        requirements={**plan.requirements, "memory_probe": "true"},
    )
    probe_execution = manager._build_execution_plan(
        job=probe_job,
        training_config=training_config,
        parallel_plan=probe_plan,
        workers=manager.cluster_config.workers,
        snapshot=probe_snapshot,
    )
    current = manager._status_store.create_initial_status(probe_job)
    monkeypatch.setattr(manager, "_launcher_factory", lambda backend: launcher)
    return manager._run_memory_probe(
        training_config=training_config,
        probe_job=probe_job,
        probe_snapshot=probe_snapshot,
        probe_plan=probe_plan,
        probe_execution=probe_execution,
        cluster_state=_cluster_state(),
        current=current,
    )


def test_probe_rendezvous_timeout_classified_infra_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manager = JobManager(_cluster_config(tmp_path))
    launcher = FakeProbeLauncher(
        rendezvous_ready=False,
        training_started=False,
        log_tail="",
        timeout_stage="rendezvous",
    )
    result = _run_probe_with_launcher(manager, tmp_path, launcher, monkeypatch=monkeypatch)

    assert result.status == PROBE_INFRA_FAILURE
    assert result.subtype == "RENDEZVOUS_TIMEOUT"
    assert result.rendezvous_ready is False
    assert manager._status_store.active_reservations() == []


def test_probe_cuda_oom_classified_memory_reject(tmp_path: Path, monkeypatch) -> None:
    manager = JobManager(_cluster_config(tmp_path))
    launcher = FakeProbeLauncher(
        rendezvous_ready=True,
        training_started=True,
        log_tail=(
            "forward step failed\n"
            "torch.cuda.OutOfMemoryError: CUDA out of memory. "
            "Tried to allocate 512.00 MiB\n"
        ),
    )
    result = _run_probe_with_launcher(manager, tmp_path, launcher, monkeypatch=monkeypatch)

    assert result.status == PROBE_MEMORY_REJECT
    assert result.subtype == "CUDA_OOM"
    assert result.oom_evidence is True
    assert manager._status_store.active_reservations() == []


def test_probe_infra_failure_does_not_fallback_to_next_plan(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manager = JobManager(_cluster_config(tmp_path))
    training_config = _training_config(tmp_path)
    job = create_training_job(
        config_path=str(tmp_path / "train-probe.yaml"),
        model=training_config.model.name,
        requested_world_size=2,
        backend_preference=as_backend_name("nccl"),
        runtime_environment_ref="env:cluster/shardgrid",
        job_id=as_job_id("job-infra-nofallback"),
    )
    plan_a = _plan("candidate-A")
    plan_b = _plan("candidate-B")
    calls: list[str | None] = []

    def memory_probe(probe_plan, execution) -> MemoryProbeResult:
        del execution
        calls.append(probe_plan.selected_candidate_id)
        return MemoryProbeResult(
            PROBE_INFRA_FAILURE,
            probe_plan.selected_candidate_id,
            subtype="RENDEZVOUS_TIMEOUT",
            message="rendezvous timeout",
            rendezvous_ready=False,
        )

    manager._memory_probe = memory_probe
    monkeypatch.setattr(manager, "_prepare_live_execution_plan", _fake_preflight(31001))

    with pytest.raises(MemoryProbeInfraFailure) as exc:
        manager._select_memory_probe_candidate(
            job=job,
            training_config=training_config,
            candidate_plans=(plan_a, plan_b),
            selected_workers=manager.cluster_config.workers,
            cluster_state=_cluster_state(),
            network_state=_network_state(),
            rejected_engine_ids=(),
        )

    assert exc.value.result.status == PROBE_INFRA_FAILURE
    assert exc.value.result.subtype == "RENDEZVOUS_TIMEOUT"
    assert calls and set(calls) == {"candidate-A"}
    assert "candidate-B" not in calls


def test_memory_reject_falls_back_to_next_probe_passing_plan(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manager = JobManager(_cluster_config(tmp_path))
    training_config = _training_config(tmp_path)
    job = create_training_job(
        config_path=str(tmp_path / "train-probe.yaml"),
        model=training_config.model.name,
        requested_world_size=2,
        backend_preference=as_backend_name("nccl"),
        runtime_environment_ref="env:cluster/shardgrid",
        job_id=as_job_id("job-mem-fallback"),
    )
    plan_a = _plan("candidate-A")
    plan_b = _plan("candidate-B")
    calls: list[str | None] = []

    def memory_probe(probe_plan, execution) -> MemoryProbeResult:
        del execution
        calls.append(probe_plan.selected_candidate_id)
        if probe_plan.selected_candidate_id == "candidate-A":
            return MemoryProbeResult(
                PROBE_MEMORY_REJECT,
                probe_plan.selected_candidate_id,
                subtype="CUDA_OOM",
                message="probe oom",
                oom_evidence=True,
                actual_peak_reserved_bytes=6 * 1024 * 1024 * 1024,
            )
        return MemoryProbeResult(
            PROBE_PASS,
            probe_plan.selected_candidate_id,
            "probe passed",
            actual_peak_allocated_bytes=123,
            actual_peak_reserved_bytes=456,
        )

    manager._memory_probe = memory_probe
    monkeypatch.setattr(manager, "_prepare_live_execution_plan", _fake_preflight(31001))

    selected, selection = manager._select_memory_probe_candidate(
        job=job,
        training_config=training_config,
        candidate_plans=(plan_a, plan_b),
        selected_workers=manager.cluster_config.workers,
        cluster_state=_cluster_state(),
        network_state=_network_state(),
        rejected_engine_ids=(),
    )

    assert selected.selected_candidate_id == "candidate-B"
    assert calls == ["candidate-A", "candidate-B"]
    assert [item["result"] for item in selection["candidates"]] == [
        PROBE_MEMORY_REJECT,
        PROBE_PASS,
    ]
    assert selection["selected_candidate_id"] == "candidate-B"
    assert manager._status_store.active_reservations() == []


def test_all_memory_reject_raises_no_feasible_plan(tmp_path: Path, monkeypatch) -> None:
    manager = JobManager(_cluster_config(tmp_path))
    training_config = _training_config(tmp_path)
    job = create_training_job(
        config_path=str(tmp_path / "train-probe.yaml"),
        model=training_config.model.name,
        requested_world_size=2,
        backend_preference=as_backend_name("nccl"),
        runtime_environment_ref="env:cluster/shardgrid",
        job_id=as_job_id("job-all-reject"),
    )
    plans = [_plan("candidate-A"), _plan("candidate-B"), _plan("candidate-C")]

    def memory_probe(probe_plan, execution) -> MemoryProbeResult:
        del execution
        return MemoryProbeResult(
            PROBE_MEMORY_REJECT,
            probe_plan.selected_candidate_id,
            subtype="CUDA_OOM",
            message="probe oom",
            oom_evidence=True,
        )

    manager._memory_probe = memory_probe
    monkeypatch.setattr(manager, "_prepare_live_execution_plan", _fake_preflight(31001))

    with pytest.raises(MemoryProbeNoFeasiblePlan) as exc:
        manager._select_memory_probe_candidate(
            job=job,
            training_config=training_config,
            candidate_plans=tuple(plans),
            selected_workers=manager.cluster_config.workers,
            cluster_state=_cluster_state(),
            network_state=_network_state(),
            rejected_engine_ids=(),
        )

    payload = json.loads(str(exc.value))
    assert payload["final_result"] == "NO_FEASIBLE_PLAN"
    assert [item["result"] for item in payload["candidates"]] == [
        PROBE_MEMORY_REJECT,
        PROBE_MEMORY_REJECT,
        PROBE_MEMORY_REJECT,
    ]
    assert manager._status_store.active_reservations() == []


def test_runtime_start_failure_after_rendezvous_is_not_memory_reject(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manager = JobManager(_cluster_config(tmp_path))
    launcher = FakeProbeLauncher(
        rendezvous_ready=True,
        training_started=False,
        log_tail="model build failed\n",
        timeout_stage="rendezvous",
    )
    result = _run_probe_with_launcher(manager, tmp_path, launcher, monkeypatch=monkeypatch)

    assert result.status == "RUNTIME_FAILURE"
    assert result.subtype == "RUNTIME_START_FAILURE"
    assert result.rendezvous_ready is True


def test_resource_changed_raises_replan_signal(tmp_path: Path, monkeypatch) -> None:
    manager = JobManager(_cluster_config(tmp_path))
    training_config = _training_config(tmp_path)
    job = create_training_job(
        config_path=str(tmp_path / "train-probe.yaml"),
        model=training_config.model.name,
        requested_world_size=2,
        backend_preference=as_backend_name("nccl"),
        runtime_environment_ref="env:cluster/shardgrid",
        job_id=as_job_id("job-resource-changed"),
    )

    def failing_preflight(*, training_config, execution_plan, cluster_state, network_state):
        del training_config, execution_plan, cluster_state, network_state
        raise ValueError(
            "RESOURCE_CHANGED: worker gpu1060 usable memory 1073741824 is below "
            "required peak 2147483648"
        )

    monkeypatch.setattr(manager, "_prepare_live_execution_plan", failing_preflight)

    with pytest.raises(MemoryProbeResourceChanged) as exc:
        manager._select_memory_probe_candidate(
            job=job,
            training_config=training_config,
            candidate_plans=(_plan("candidate-A"),),
            selected_workers=manager.cluster_config.workers,
            cluster_state=_cluster_state(),
            network_state=_network_state(),
            rejected_engine_ids=(),
        )

    assert exc.value.result.status == "RESOURCE_CHANGED"
    assert "RESOURCE_CHANGED" in exc.value.result.message
    assert manager._status_store.active_reservations() == []


def test_candidate_identity_mismatch_raises_runtime_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manager = JobManager(_cluster_config(tmp_path))
    training_config = _training_config(tmp_path)
    job = create_training_job(
        config_path=str(tmp_path / "train-probe.yaml"),
        model=training_config.model.name,
        requested_world_size=2,
        backend_preference=as_backend_name("nccl"),
        runtime_environment_ref="env:cluster/shardgrid",
        job_id=as_job_id("job-identity-mismatch"),
    )

    def relabeling_preflight(
        *,
        training_config,
        execution_plan,
        cluster_state,
        network_state,
    ):
        del training_config, cluster_state, network_state
        relabeled = replace(
            execution_plan,
            labels={**execution_plan.labels, "selected_candidate_id": "candidate-X"},
        )
        return relabeled, None, _network_state()

    monkeypatch.setattr(manager, "_prepare_live_execution_plan", relabeling_preflight)

    with pytest.raises(MemoryProbeRuntimeFailure) as exc:
        manager._select_memory_probe_candidate(
            job=job,
            training_config=training_config,
            candidate_plans=(_plan("candidate-A"),),
            selected_workers=manager.cluster_config.workers,
            cluster_state=_cluster_state(),
            network_state=_network_state(),
            rejected_engine_ids=(),
        )

    assert exc.value.result.status == "RUNTIME_FAILURE"
    assert exc.value.result.subtype == "CANDIDATE_IDENTITY_CHANGED"
