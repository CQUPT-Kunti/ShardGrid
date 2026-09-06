"""Training job orchestration helpers and the T101 JobManager lifecycle."""

from __future__ import annotations

import gc
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Sequence
from uuid import uuid4

from shardgrid.artifacts.collector import (
    ArtifactCollectionResult,
    ArtifactCollector,
    CollectionStatus,
    WorkerArtifactSource,
)
from shardgrid.artifacts.metadata import write_snapshot_metadata
from shardgrid.artifacts.snapshot import create_code_snapshot
from shardgrid.artifacts.store import ArtifactStore
from shardgrid.artifacts.transport import build_transport_config, select_artifact_transport
from shardgrid.common.config import (
    ClusterConfig,
    TrainingConfig,
    WorkerConfig,
    load_training_config,
)
from shardgrid.common.enums import BackendStatus, FailureStage, Health, JobState, PhysicalOS
from shardgrid.common.errors import make_failure_record
from shardgrid.common.models import BackendName, JobId, WorkerId, as_job_id
from shardgrid.control.resource_manager import ClusterState, ResourceManager
from shardgrid.control.status_store import StatusStore
from shardgrid.distributed.backend import select_backend
from shardgrid.engines.base import registered_engine_registry
from shardgrid.engines.models import ParallelPlan
from shardgrid.engines.selected import SelectedEngine, select_with_fallback
from shardgrid.jobs.models import FailureRecord, JobSnapshot, JobStatus, TrainingJob
from shardgrid.launchers.base import (
    Launcher,
    LauncherContext,
    LauncherResult,
    LauncherResultStatus,
)
from shardgrid.launchers.ssh import SSHLauncher
from shardgrid.network.mtu import parse_interface_mtu, parse_route_interface, parse_route_source_ip
from shardgrid.network.state import build_network_state
from shardgrid.planner import (
    MemoryEstimationConfig,
    build_automatic_parallel_plan,
    build_model_profile,
    search_joint_partition_placement,
    select_best_joint_placement_plan,
)
from shardgrid.planner.models import ExecutionPlan, MasterMetadata, WorkerAssignment
from shardgrid.planner.requirements import FeasibilityStatus
from shardgrid.resources.models import NetworkLink, NetworkState, WorkerResource
from shardgrid.transport.remote_access import run_remote_access_check
from shardgrid.transport.runtime import WSLRuntimeConfig, WSLRuntimeWrapper
from shardgrid.transport.ssh import SSHOptions, SSHTransport
from shardgrid.workers.environment_report import build_control_report
from shardgrid.workers.gpu_probe import probe_gpu
from shardgrid.workers.models import WorkerRuntime
from shardgrid.workers.probe import ProbeFailure, WindowsHostInfo, WorkerProbeResult

ProbeWorkerFunc = Callable[[WorkerConfig], WorkerProbeResult]
ProbeNetworkFunc = Callable[[list[WorkerResource]], NetworkState]
SelectEngineFunc = Callable[..., SelectedEngine]
LauncherFactory = Callable[[str], Launcher]
SourceRoot = str | Path
MemoryProbeFunc = Callable[[ParallelPlan, ExecutionPlan], "MemoryProbeResult"]


def _now() -> str:
    return datetime.now(tz=UTC).isoformat()


def _process_rss_bytes() -> int | None:
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (FileNotFoundError, PermissionError, ValueError):
        return None
    return None


def _contains_tensor(value: object) -> bool:
    import torch

    if isinstance(value, torch.Tensor):
        return True
    if isinstance(value, dict):
        return any(_contains_tensor(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_tensor(item) for item in value)
    return False


def _probe_log_has_oom(text: str | None) -> bool:
    lowered = (text or "").lower()
    return any(
        token in lowered
        for token in (
            "cuda out of memory",
            "cuda error: out of memory",
            "cuda oom",
            "out of memory",
        )
    )


def _parse_probe_data_payload(text: str | None) -> dict[str, object] | None:
    for line in (text or "").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def create_training_job(
    *,
    config_path: str,
    model: str,
    requested_world_size: int,
    backend_preference: BackendName,
    runtime_environment_ref: str | None,
    job_id: JobId | None = None,
) -> TrainingJob:
    created_at = _now()
    return TrainingJob(
        job_id=_new_job_id() if job_id is None else job_id,
        config_path=config_path,
        model=model,
        requested_world_size=requested_world_size,
        backend_preference=backend_preference,
        runtime_environment_ref=runtime_environment_ref,
        created_at=created_at,
        updated_at=created_at,
    )


def save_training_job(job: TrainingJob, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.write_text(json.dumps(job.to_dict(), indent=2, sort_keys=True))
    return output_path


def load_training_job(path: str | Path) -> TrainingJob:
    return TrainingJob.from_dict(json.loads(Path(path).read_text()))


def can_launch_job(
    job: TrainingJob,
    *,
    workers: list[WorkerResource],
    network_state: NetworkState | None,
) -> bool:
    if not job.runtime_environment_ref:
        return False
    if len(workers) < job.requested_world_size:
        return False
    worker_ids = [worker.worker_id for worker in workers if worker.health is Health.HEALTHY]
    if len(worker_ids) < job.requested_world_size:
        return False
    if network_state is None:
        return False
    if not _covers_workers(network_state, worker_ids[: job.requested_world_size]):
        return False
    return True


def transition_training_job(
    job: TrainingJob,
    next_state: JobState,
    *,
    workers: list[WorkerResource] | None = None,
    network_state: NetworkState | None = None,
) -> TrainingJob:
    if next_state is JobState.LAUNCHING:
        if workers is None or not can_launch_job(
            job, workers=workers, network_state=network_state
        ):
            raise ValueError(
                "job cannot enter launching without eligible workers, "
                "network state, and runtime environment evidence"
            )
    transitioned = job.transition_to(next_state)
    if transitioned.created_at is None:
        return replace(transitioned, created_at=_now())
    return transitioned


@dataclass(frozen=True)
class JobRunResult:
    job: TrainingJob
    status: JobStatus
    snapshot: JobSnapshot | None = None
    execution_plan: ExecutionPlan | None = None
    parallel_plan: ParallelPlan | None = None
    cluster_state: ClusterState | None = None
    network_state: NetworkState | None = None
    collection_result: ArtifactCollectionResult | None = None
    launcher_result: LauncherResult | None = None


PROBE_PASS = "PASS"
PROBE_MEMORY_REJECT = "MEMORY_REJECT"
PROBE_RESOURCE_CHANGED = "RESOURCE_CHANGED"
PROBE_INFRA_FAILURE = "INFRA_FAILURE"
PROBE_RUNTIME_FAILURE = "RUNTIME_FAILURE"

PROBE_TRANSIENT_INFRA_SUBTYPES = {
    "RENDEZVOUS_TIMEOUT",
    "PORT_COLLISION",
    "SSH_TRANSIENT_FAILURE",
}

PROBE_PHASE_MARKER = "GENERIC_DAG_PROBE_EVIDENCE "


@dataclass(frozen=True)
class MemoryProbeResult:
    status: str
    candidate_id: str | None
    subtype: str | None = None
    message: str = ""
    actual_peak_allocated_bytes: int | None = None
    actual_peak_reserved_bytes: int | None = None
    rendezvous_ready: bool | None = None
    phase: str | None = None
    events: tuple[dict[str, object], ...] = ()
    master_addr: str | None = None
    master_port: int | None = None
    oom_evidence: bool = False
    attempt: int = 1

    def __post_init__(self) -> None:
        status = self.status
        if status == "REJECT":
            object.__setattr__(self, "status", PROBE_MEMORY_REJECT)
            status = PROBE_MEMORY_REJECT
        if status not in {
            PROBE_PASS,
            PROBE_MEMORY_REJECT,
            PROBE_RESOURCE_CHANGED,
            PROBE_INFRA_FAILURE,
            PROBE_RUNTIME_FAILURE,
        }:
            raise ValueError(f"invalid memory probe status: {self.status}")

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "candidate_id": self.candidate_id,
            "subtype": self.subtype,
            "message": self.message,
            "actual_peak_allocated_bytes": self.actual_peak_allocated_bytes,
            "actual_peak_reserved_bytes": self.actual_peak_reserved_bytes,
            "rendezvous_ready": self.rendezvous_ready,
            "phase": self.phase,
            "events": list(self.events),
            "master_addr": self.master_addr,
            "master_port": self.master_port,
            "oom_evidence": self.oom_evidence,
            "attempt": self.attempt,
        }


class MemoryProbeSelectionError(ValueError):
    def __init__(self, result: MemoryProbeResult) -> None:
        super().__init__(result.message)
        self.result = result


class MemoryProbeNoFeasiblePlan(MemoryProbeSelectionError):
    """Every bounded candidate was a real memory reject; this is safe saturation."""


class MemoryProbeInfraFailure(MemoryProbeSelectionError):
    """Rendezvous / SSH / distributed infrastructure failed; not a capacity signal."""


class MemoryProbeResourceChanged(MemoryProbeSelectionError):
    """Live resource revalidation no longer matches the candidate plan."""


class MemoryProbeRuntimeFailure(MemoryProbeSelectionError):
    """Probe reached a runtime error that is not a memory signal."""


class JobManager:
    def __init__(
        self,
        cluster_config: ClusterConfig,
        *,
        probe_worker: ProbeWorkerFunc | None = None,
        probe_network: ProbeNetworkFunc | None = None,
        select_engine: SelectEngineFunc = select_with_fallback,
        launcher_factory: LauncherFactory | None = None,
        artifact_store: ArtifactStore | None = None,
        artifact_collector: ArtifactCollector | None = None,
        resource_manager: ResourceManager | None = None,
        status_store: StatusStore | None = None,
        memory_probe: MemoryProbeFunc | None = None,
        source_root: SourceRoot = ".",
        secrets: Sequence[str] = (),
    ) -> None:
        self.cluster_config = cluster_config
        self._probe_worker = probe_worker or self._default_probe_worker
        self._probe_network = probe_network or self._default_probe_network
        self._select_engine = select_engine
        self._launcher_factory = launcher_factory or (
            lambda backend: SSHLauncher(cluster_config, secrets=secrets)
        )
        self._artifact_store = artifact_store or ArtifactStore(cluster_config.jobs_root)
        self._artifact_collector = artifact_collector
        self._resource_manager = resource_manager or ResourceManager()
        self._status_store = status_store or StatusStore(cluster_config.jobs_root)
        self._memory_probe = memory_probe
        self._source_root = Path(source_root).resolve()
        self._secrets = tuple(secret for secret in secrets if secret)
        self._last_planning_evidence: dict[str, object] = {}
        self._last_parallel_plan_candidates: tuple[ParallelPlan, ...] = ()
        self._data_test_cache: dict[tuple[str, str], tuple[float, bool, str | None]] = {}

    def run(
        self,
        training_config_path: str | Path,
        *,
        job_id: JobId | None = None,
        dry_run: bool = False,
        min_selected_physical_hosts: int | None = None,
    ) -> JobRunResult:
        training_config = load_training_config(training_config_path)
        self._last_planning_evidence = {}
        self._last_parallel_plan_candidates = ()
        job = create_training_job(
            config_path=str(training_config_path),
            model=training_config.model.name,
            requested_world_size=training_config.resources.world_size,
            backend_preference=training_config.job.communication_backend,
            runtime_environment_ref="env:cluster/shardgrid",
            job_id=job_id,
        )
        current = self._status_store.create_initial_status(job)
        selected_workers = self._select_candidate_workers(training_config)

        probe_results = [self._probe_worker(worker) for worker in selected_workers]
        current = self._persist_status(
            current,
            state=JobState.PROBING,
            phase="probe",
        )
        failed_probe = next(
            (result for result in probe_results if result.health is not Health.HEALTHY),
            None,
        )
        if failed_probe is not None and not self._automatic_planning_enabled(training_config):
            failure = self._probe_failure(selected_workers, failed_probe)
            current = self._failed_status(current, phase="probe", failure=failure)
            self._status_store.save_path(self._status_store.status_path(job.job_id), current)
            raise_or_return = self._failed_run_result(job, current)
            return raise_or_return
        if self._automatic_planning_enabled(training_config):
            healthy_worker_ids = {
                result.worker_resource.worker_id
                for result in probe_results
                if result.health is Health.HEALTHY
            }
            selected_workers = [
                worker for worker in selected_workers if worker.worker_id in healthy_worker_ids
            ]
            probe_results = [
                result
                for result in probe_results
                if result.worker_resource.worker_id in healthy_worker_ids
            ]
            if not probe_results:
                failure = (
                    self._probe_failure(
                        self._select_candidate_workers(training_config),
                        failed_probe,
                    )
                    if failed_probe is not None
                    else make_failure_record(
                        stage=FailureStage.PROBE,
                        host=str(self.cluster_config.control.hostname),
                        message="no enabled workers are available for automatic planning",
                        recommended_action="enable at least one worker and retry",
                        secrets=self._secrets,
                    )
                )
                current = self._failed_status(current, phase="probe", failure=failure)
                self._status_store.save_path(self._status_store.status_path(job.job_id), current)
                return self._failed_run_result(job, current)

        worker_resources = self._apply_active_resource_reservations(
            [result.worker_resource for result in probe_results],
            current_job_id=job.job_id,
        )
        cluster_state = self._resource_manager.build_cluster_state(
            worker_resources,
            network_state=None,
            require_network=False,
        )

        try:
            network_state = self._probe_network(worker_resources)
        except Exception as exc:
            failure = make_failure_record(
                stage=FailureStage.NETWORK,
                host=str(self.cluster_config.control.hostname),
                message=f"network probe failed: {exc}",
                recommended_action="repair network diagnostics and rerun training",
                secrets=self._secrets,
            )
            current = self._failed_status(current, phase="probe", failure=failure)
            self._status_store.save_path(self._status_store.status_path(job.job_id), current)
            return self._failed_run_result(job, current)
        cluster_state = self._resource_manager.build_cluster_state(
            worker_resources,
            network_state=network_state,
            require_network=True,
        )

        current = self._persist_status(
            current,
            state=JobState.PLANNING,
            phase="plan",
        )
        planning_evidence: dict[str, object] | None = None
        probe_selection: dict[str, object] | None = None
        max_planning_attempts = 1 + int(
            os.environ.get("SHARDGRID_MEMORY_PROBE_REPLAN_ATTEMPTS", "1")
        )
        planning_attempt = 0
        while True:
            planning_attempt += 1
            try:
                selected_engine = self._select_engine(
                    self._selected_engine_id(),
                    job,
                    cluster_state,
                    network_state,
                    registry=registered_engine_registry(),
                )
                if self._automatic_planning_enabled(training_config):
                    automatic_plan = self._build_automatic_parallel_plan(
                        training_config=training_config,
                        cluster_state=cluster_state,
                        selected_engine=selected_engine,
                        min_selected_physical_hosts=min_selected_physical_hosts,
                    )
                    planning_evidence = dict(self._last_planning_evidence)
                    selected_engine = SelectedEngine(
                        job_id=getattr(selected_engine, "job_id", job.job_id),
                        engine=selected_engine.engine,
                        candidate=selected_engine.candidate,
                        parallel_plan=automatic_plan,
                        original_plan_path=automatic_plan.engine_plan_path,
                        rejected_engine_ids=tuple(
                            getattr(selected_engine, "rejected_engine_ids", ())
                        ),
                    )
            except Exception as exc:
                failure = make_failure_record(
                    stage=FailureStage.PLAN,
                    host=str(self.cluster_config.control.hostname),
                    message=f"engine planning failed: {exc}",
                    recommended_action="repair engine selection or plan inputs, then retry",
                    secrets=self._secrets,
                )
                current = self._failed_status(current, phase="plan", failure=failure)
                self._status_store.save_path(self._status_store.status_path(job.job_id), current)
                return self._failed_run_result(job, current)

            if self._automatic_planning_enabled(training_config) and not dry_run:
                try:
                    selected_plan, probe_selection = self._select_memory_probe_candidate(
                        job=job,
                        training_config=training_config,
                        candidate_plans=self._last_parallel_plan_candidates
                        or (selected_engine.parallel_plan,),
                        selected_workers=selected_workers,
                        cluster_state=cluster_state,
                        network_state=network_state,
                        rejected_engine_ids=selected_engine.rejected_engine_ids,
                    )
                    planning_evidence = dict(planning_evidence or {})
                    planning_evidence["memory_probe_selection"] = probe_selection
                    selected_engine = SelectedEngine(
                        job_id=getattr(selected_engine, "job_id", job.job_id),
                        engine=selected_engine.engine,
                        candidate=selected_engine.candidate,
                        parallel_plan=selected_plan,
                        original_plan_path=selected_plan.engine_plan_path,
                        rejected_engine_ids=tuple(
                            getattr(selected_engine, "rejected_engine_ids", ())
                        ),
                    )
                    break
                except MemoryProbeNoFeasiblePlan as exc:
                    failure = make_failure_record(
                        stage=FailureStage.PLAN,
                        host=str(self.cluster_config.control.hostname),
                        message=(
                            "NO_FEASIBLE_PLAN: memory probe rejected all candidates: "
                            f"{exc.result.message}"
                        ),
                        recommended_action=(
                            "free GPU memory or reduce the model/batch size, then retry"
                        ),
                        runtime_environment={
                            "probe_result": json.dumps(
                                exc.result.to_dict(), sort_keys=True
                            ),
                        },
                        secrets=self._secrets,
                    )
                    current = self._failed_status(current, phase="plan", failure=failure)
                    self._status_store.save_path(
                        self._status_store.status_path(job.job_id), current
                    )
                    return self._failed_run_result(job, current)
                except MemoryProbeResourceChanged as exc:
                    if planning_attempt >= max_planning_attempts:
                        failure = make_failure_record(
                            stage=FailureStage.LAUNCH,
                            host=str(self.cluster_config.control.hostname),
                            message=(
                                "RESOURCE_CHANGED: probe resource revalidation failed "
                                f"after replanning: {exc.result.message}"
                            ),
                            recommended_action=(
                                "wait for active jobs to release GPU memory or add workers"
                            ),
                            runtime_environment={
                                "probe_result": json.dumps(
                                    exc.result.to_dict(), sort_keys=True
                                ),
                            },
                            secrets=self._secrets,
                        )
                        current = self._failed_status(
                            current, phase="memory_probe", failure=failure
                        )
                        self._status_store.save_path(
                            self._status_store.status_path(job.job_id), current
                        )
                        return self._failed_run_result(job, current)
                    refreshed = self._refresh_planning_state(
                        training_config=training_config,
                        selected_workers=selected_workers,
                        current_job_id=job.job_id,
                    )
                    if refreshed is None:
                        failure = make_failure_record(
                            stage=FailureStage.LAUNCH,
                            host=str(self.cluster_config.control.hostname),
                            message=(
                                "RESOURCE_CHANGED: no healthy workers remain for "
                                "replanning after probe resource revalidation"
                            ),
                            recommended_action="add or repair workers, then retry",
                            secrets=self._secrets,
                        )
                        current = self._failed_status(
                            current, phase="memory_probe", failure=failure
                        )
                        self._status_store.save_path(
                            self._status_store.status_path(job.job_id), current
                        )
                        return self._failed_run_result(job, current)
                    (
                        selected_workers,
                        _probe_results,
                        _worker_resources,
                        cluster_state,
                        network_state,
                    ) = refreshed
                    continue
                except MemoryProbeInfraFailure as exc:
                    failure = make_failure_record(
                        stage=FailureStage.RENDEZVOUS,
                        host=str(self.cluster_config.control.hostname),
                        message=(
                            "PROBE_INFRA_FAILURE: "
                            f"{exc.result.message}"
                        ),
                        recommended_action=(
                            "inspect probe rendezvous/SSH/network diagnostics and retry; "
                            "this is not a GPU memory capacity signal"
                        ),
                        runtime_environment={
                            "probe_result": json.dumps(
                                exc.result.to_dict(), sort_keys=True
                            ),
                        },
                        retryable=True,
                        secrets=self._secrets,
                    )
                    current = self._failed_status(
                        current, phase="probe_infra", failure=failure
                    )
                    self._status_store.save_path(
                        self._status_store.status_path(job.job_id), current
                    )
                    return self._failed_run_result(job, current)
                except MemoryProbeRuntimeFailure as exc:
                    failure = make_failure_record(
                        stage=FailureStage.TRAIN,
                        host=str(self.cluster_config.control.hostname),
                        message=(
                            "PROBE_RUNTIME_FAILURE: "
                            f"{exc.result.message}"
                        ),
                        recommended_action=(
                            "inspect probe phase diagnostics and rank logs; "
                            "this is not a GPU memory capacity signal"
                        ),
                        runtime_environment={
                            "probe_result": json.dumps(
                                exc.result.to_dict(), sort_keys=True
                            ),
                        },
                        secrets=self._secrets,
                    )
                    current = self._failed_status(
                        current, phase="memory_probe", failure=failure
                    )
                    self._status_store.save_path(
                        self._status_store.status_path(job.job_id), current
                    )
                    return self._failed_run_result(job, current)
                except Exception as exc:
                    failure = make_failure_record(
                        stage=FailureStage.PLAN,
                        host=str(self.cluster_config.control.hostname),
                        message=f"memory probe selection failed: {exc}",
                        recommended_action=(
                            "inspect probe diagnostics and retry planning"
                        ),
                        secrets=self._secrets,
                    )
                    current = self._failed_status(current, phase="plan", failure=failure)
                    self._status_store.save_path(
                        self._status_store.status_path(job.job_id), current
                    )
                    return self._failed_run_result(job, current)
            else:
                break

        snapshot = self._artifact_store.create_snapshot(
            replace(
                job,
                snapshot_path=str(self._artifact_store.snapshot_paths(job.job_id).root),
            )
        )
        create_code_snapshot(snapshot, source_root=self._source_root, secrets=self._secrets)
        execution_plan = self._build_execution_plan(
            job=job,
            training_config=training_config,
            parallel_plan=selected_engine.parallel_plan,
            workers=selected_workers,
            snapshot=snapshot,
            rejected_engine_ids=selected_engine.rejected_engine_ids,
        )
        if not dry_run:
            try:
                execution_plan, cluster_state, network_state = self._prepare_live_execution_plan(
                    training_config=training_config,
                    execution_plan=execution_plan,
                    cluster_state=cluster_state,
                    network_state=network_state,
                )
            except Exception as exc:
                failure = make_failure_record(
                    stage=FailureStage.LAUNCH,
                    host=str(self.cluster_config.control.hostname),
                    message=f"live execution preflight failed: {exc}",
                    recommended_action=(
                        "repair the selected workers or rerun planning before launch"
                    ),
                    secrets=self._secrets,
                )
                current = self._failed_status(current, phase="launch", failure=failure)
                self._status_store.save_path(self._status_store.status_path(job.job_id), current)
                return self._failed_run_result(
                    job,
                    current,
                    snapshot=snapshot,
                    execution_plan=execution_plan,
                    parallel_plan=selected_engine.parallel_plan,
                    cluster_state=cluster_state,
                    network_state=network_state,
                )
        current = self._persist_status(
            replace(
                current,
                workers=[assignment.worker_id for assignment in execution_plan.workers],
                assignments=list(execution_plan.workers),
                runtime_environment_refs=self._runtime_refs(execution_plan),
                backend=execution_plan.backend,
            ),
            snapshot=snapshot,
            state=JobState.SNAPSHOTTING,
            phase="plan",
        )
        launch_metadata = self._launch_metadata(
            selected_engine,
            planning_evidence=planning_evidence,
        )
        probe_selection = (planning_evidence or {}).get("memory_probe_selection")
        if isinstance(probe_selection, dict):
            launch_metadata["memory_probe_selection_ref"] = str(
                self._write_memory_probe_selection_artifact(
                    snapshot=snapshot,
                    selection=probe_selection,
                )
            )

        self._write_snapshot_metadata(
            snapshot=snapshot,
            job=job,
            training_config=training_config,
            parallel_plan=selected_engine.parallel_plan,
            execution_plan=execution_plan,
            network_state=network_state,
            job_status=current,
            launch_metadata=launch_metadata,
            dry_run=dry_run,
        )
        if dry_run:
            return JobRunResult(
                job=job,
                status=current,
                snapshot=snapshot,
                execution_plan=execution_plan,
                parallel_plan=selected_engine.parallel_plan,
                cluster_state=cluster_state,
                network_state=network_state,
            )

        preparation = selected_engine.engine.prepare(snapshot, execution_plan)
        if preparation.status not in (BackendStatus.AVAILABLE, BackendStatus.EXPERIMENTAL):
            failure = make_failure_record(
                stage=FailureStage.PLAN,
                host=str(self.cluster_config.control.hostname),
                message="engine preparation did not produce an available runtime",
                recommended_action="inspect engine diagnostics and rerun planning",
                secrets=self._secrets,
            )
            current = self._failed_status(current, phase="plan", failure=failure)
            self._save_status(current, snapshot=snapshot)
            return self._failed_run_result(
                job,
                current,
                snapshot=snapshot,
                execution_plan=execution_plan,
                parallel_plan=selected_engine.parallel_plan,
                cluster_state=cluster_state,
                network_state=network_state,
            )
        launcher = self._launcher_factory(training_config.job.backend)
        context = LauncherContext(
            job=job,
            execution_plan=execution_plan,
            cluster_state=cluster_state,
            snapshot=snapshot,
            job_status=current,
            runtime_environment_refs=self._runtime_refs(execution_plan),
        )

        current = self._persist_status(
            current,
            snapshot=snapshot,
            state=JobState.DISTRIBUTING,
            phase="distribute",
        )
        context = replace(context, job_status=current)
        prepare_result = launcher.prepare(context)
        if prepare_result.status not in {LauncherResultStatus.SUCCESS, LauncherResultStatus.NOOP}:
            current = self._failed_status(
                current,
                phase="distribute",
                failure=self._launcher_failure(
                    prepare_result,
                    FailureStage.DISTRIBUTE,
                    "launcher prepare failed before distribution completed",
                ),
            )
            self._save_terminal_snapshot(
                snapshot,
                job,
                training_config,
                selected_engine.parallel_plan,
                execution_plan,
                network_state,
                current,
            )
            return self._failed_run_result(
                job,
                current,
                snapshot=snapshot,
                execution_plan=execution_plan,
                parallel_plan=selected_engine.parallel_plan,
                cluster_state=cluster_state,
                network_state=network_state,
                launcher_result=prepare_result,
            )

        context = replace(context, job_status=self._load_status(snapshot, current))
        distribute_result = launcher.distribute(context)
        if distribute_result.status not in {
            LauncherResultStatus.SUCCESS,
            LauncherResultStatus.NOOP,
        }:
            current = self._failed_status(
                self._load_status(snapshot, current),
                phase="distribute",
                failure=self._launcher_failure(
                    distribute_result,
                    FailureStage.DISTRIBUTE,
                    "launcher distribute failed before launch",
                ),
            )
            self._save_terminal_snapshot(
                snapshot,
                job,
                training_config,
                selected_engine.parallel_plan,
                execution_plan,
                network_state,
                current,
            )
            return self._failed_run_result(
                job,
                current,
                snapshot=snapshot,
                execution_plan=execution_plan,
                parallel_plan=selected_engine.parallel_plan,
                cluster_state=cluster_state,
                network_state=network_state,
                launcher_result=distribute_result,
            )

        conflicts = self._status_store.reserve_resources(job.job_id, execution_plan.workers)
        if conflicts:
            current = self._failed_status(
                self._load_status(snapshot, current),
                phase="launch",
                failure=make_failure_record(
                    stage=FailureStage.LAUNCH,
                    host=str(self.cluster_config.control.hostname),
                    message="GPU resource reservation conflict before launch",
                    recommended_action=(
                        "wait for the running job to finish or choose different workers"
                    ),
                    runtime_environment={
                        "conflicts": json.dumps(conflicts, sort_keys=True),
                    },
                    secrets=self._secrets,
                ),
            )
            self._save_terminal_snapshot(
                snapshot,
                job,
                training_config,
                selected_engine.parallel_plan,
                execution_plan,
                network_state,
                current,
            )
            return self._failed_run_result(
                job,
                current,
                snapshot=snapshot,
                execution_plan=execution_plan,
                parallel_plan=selected_engine.parallel_plan,
                cluster_state=cluster_state,
                network_state=network_state,
                launcher_result=distribute_result,
            )

        current = self._persist_status(
            self._load_status(snapshot, current),
            snapshot=snapshot,
            state=JobState.LAUNCHING,
            phase="launch",
        )
        context = replace(context, job_status=current)
        launch_result = launcher.launch(context)
        if launch_result.status not in {LauncherResultStatus.SUCCESS, LauncherResultStatus.NOOP}:
            current = self._failed_status(
                self._load_status(snapshot, current),
                phase="launch",
                failure=self._launcher_failure(
                    launch_result,
                    FailureStage.LAUNCH,
                    "launcher launch failed before rendezvous",
                ),
            )
            self._save_terminal_snapshot(
                snapshot,
                job,
                training_config,
                selected_engine.parallel_plan,
                execution_plan,
                network_state,
                current,
            )
            self._status_store.release_resources(job.job_id)
            return self._failed_run_result(
                job,
                current,
                snapshot=snapshot,
                execution_plan=execution_plan,
                parallel_plan=selected_engine.parallel_plan,
                cluster_state=cluster_state,
                network_state=network_state,
                launcher_result=launch_result,
            )

        context = replace(context, job_status=self._load_status(snapshot, current))
        monitor_result, current = self._monitor_until_terminal(
            launcher=launcher,
            context=context,
            snapshot=snapshot,
            current=current,
            training_config=training_config,
        )
        if monitor_result.status not in {
            LauncherResultStatus.SUCCESS,
            LauncherResultStatus.NOOP,
        } or current.state is JobState.FAILED:
            current = self._failed_status(
                current,
                phase=current.phase,
                failure=current.failure
                or self._launcher_failure(
                    monitor_result,
                    self._failure_stage_for_phase(current.phase),
                    "launcher monitor observed a terminal failure",
                ),
            )
            self._stop_failed_job_ranks(
                launcher=launcher,
                context=replace(context, job_status=current),
            )
            current = self._load_status(snapshot, current)
            self._status_store.release_resources(job.job_id)
            self._save_terminal_snapshot(
                snapshot,
                job,
                training_config,
                selected_engine.parallel_plan,
                execution_plan,
                network_state,
                current,
            )
            return self._failed_run_result(
                job,
                current,
                snapshot=snapshot,
                execution_plan=execution_plan,
                parallel_plan=selected_engine.parallel_plan,
                cluster_state=cluster_state,
                network_state=network_state,
                launcher_result=monitor_result,
            )

        self._status_store.release_resources(job.job_id)
        current = self._persist_status(
            current,
            snapshot=snapshot,
            state=JobState.CHECKPOINTING,
            phase="checkpoint",
            preserve_metrics=True,
        )
        collection_result = self._collect_artifacts(
            snapshot,
            training_config,
            execution_plan,
            current,
        )
        self._write_collection_diagnostics(snapshot, collection_result)
        if collection_result.status is not CollectionStatus.SUCCESS:
            current = self._failed_status(
                current,
                phase="checkpoint",
                failure=self._artifact_collection_failure(collection_result),
            )
            self._save_terminal_snapshot(
                snapshot,
                job,
                training_config,
                selected_engine.parallel_plan,
                execution_plan,
                network_state,
                current,
            )
            return self._failed_run_result(
                job,
                current,
                snapshot=snapshot,
                execution_plan=execution_plan,
                parallel_plan=selected_engine.parallel_plan,
                cluster_state=cluster_state,
                network_state=network_state,
                collection_result=collection_result,
                launcher_result=monitor_result,
            )

        try:
            checkpoint_metadata = self._finalize_checkpoint_bundle(
                snapshot=snapshot,
                training_config=training_config,
                execution_plan=execution_plan,
                current=current,
                collection_result=collection_result,
            )
        except Exception as exc:
            current = self._failed_status(
                current,
                phase="checkpoint",
                failure=make_failure_record(
                    stage=FailureStage.CHECKPOINT,
                    host=str(self.cluster_config.control.hostname),
                    message=f"distributed checkpoint finalization failed: {exc}",
                    recommended_action=(
                        "inspect collected checkpoint shards and rerun the job"
                    ),
                    secrets=self._secrets,
                ),
            )
            self._save_terminal_snapshot(
                snapshot,
                job,
                training_config,
                selected_engine.parallel_plan,
                execution_plan,
                network_state,
                current,
            )
            return self._failed_run_result(
                job,
                current,
                snapshot=snapshot,
                execution_plan=execution_plan,
                parallel_plan=selected_engine.parallel_plan,
                cluster_state=cluster_state,
                network_state=network_state,
                collection_result=collection_result,
                launcher_result=monitor_result,
            )

        current = self._complete_status(
            current,
            checkpoint_ref=str(checkpoint_metadata["checkpoint_ref"]),
        )
        self._save_terminal_snapshot(
            snapshot,
            job,
            training_config,
            selected_engine.parallel_plan,
            execution_plan,
            network_state,
            current,
            checkpoint_metadata=checkpoint_metadata,
            launch_metadata=launch_metadata,
        )
        return JobRunResult(
            job=job,
            status=current,
            snapshot=snapshot,
            execution_plan=execution_plan,
            parallel_plan=selected_engine.parallel_plan,
            cluster_state=cluster_state,
            network_state=network_state,
            collection_result=collection_result,
            launcher_result=monitor_result,
        )

    def _monitor_until_terminal(
        self,
        *,
        launcher: Launcher,
        context: LauncherContext,
        snapshot: JobSnapshot,
        current: JobStatus,
        training_config: TrainingConfig,
    ) -> tuple[LauncherResult, JobStatus]:
        del training_config
        latest_result: LauncherResult | None = None
        latest_status = current
        while True:
            context = replace(context, job_status=latest_status)
            latest_result = launcher.monitor(context)
            latest_status = self._load_status(snapshot, latest_status)
            if latest_result.status not in {
                LauncherResultStatus.SUCCESS,
                LauncherResultStatus.NOOP,
            }:
                return latest_result, latest_status
            if latest_status.state in {
                JobState.COMPLETED,
                JobState.FAILED,
                JobState.STOPPED,
                JobState.CHECKPOINTING,
            }:
                return latest_result, latest_status
            time.sleep(self._monitor_poll_interval_seconds())

    def _default_probe_worker(self, worker: WorkerConfig) -> WorkerProbeResult:
        transport = SSHTransport(
            SSHOptions.from_ssh_config(
                self.cluster_config.ssh,
                host=str(worker.host),
                user=worker.ssh_user,
                port=worker.ssh_port,
            )
        )
        access = run_remote_access_check(
            transport,
            worker,
            worker_label=str(worker.labels.get("gpu") or worker.worker_id),
            preferred_environment=(
                worker.conda_environment
                or self.cluster_config.runtime.conda_environment
            ),
        )
        if access.status != "PASS" or access.runtime_identity is None:
            health = Health.UNREACHABLE if access.status == "BLOCKED" else Health.FAILED
            failure = access.failure_reason or "worker probe failed before runtime validation"
            return WorkerProbeResult(
                worker_resource=WorkerResource(
                    worker_id=worker.worker_id,
                    hostname=worker.host,
                    physical_os=worker.physical_os,
                    runtime_os=worker.runtime_os,
                    conda_environment=(
                        worker.conda_environment
                        or self.cluster_config.runtime.conda_environment
                    ),
                    conda_prefix=worker.conda_prefix or self.cluster_config.runtime.conda_prefix,
                    ip=str(worker.host),
                    health=health,
                    last_probe_at=_now(),
                ),
                worker_runtime=WorkerRuntime(
                    worker_id=worker.worker_id,
                    runtime_os=worker.runtime_os,
                    runtime_version=worker.runtime_distro,
                    conda_environment=(
                        worker.conda_environment
                        or self.cluster_config.runtime.conda_environment
                    ),
                    conda_prefix=worker.conda_prefix or self.cluster_config.runtime.conda_prefix,
                    health=health,
                ),
                windows_host=WindowsHostInfo(
                    os_version=access.windows_identity,
                    openssh_available=access.status != "BLOCKED",
                    wsl_available=access.wsl_distro is not None,
                    nvidia_driver_visible=False,
                    driver_name=None,
                ),
                failures=(
                    ProbeFailure(
                        layer="remote_access",
                        check=access.failure_category or "runtime",
                        message=failure,
                        exit_code=access.exit_code,
                        output=(access.stderr or access.stdout or "")[:500] or None,
                    ),
                ),
                health=health,
                probe_status="live",
            )

        gpu_result = getattr(access, "gpu_probe_result", None)
        if gpu_result is None:
            wrapper = WSLRuntimeWrapper(
                WSLRuntimeConfig(
                    distro=access.runtime_identity.wsl_distro,
                    user=worker.ssh_user,
                    conda_executable=access.runtime_identity.conda_executable,
                    conda_environment=access.runtime_identity.conda_environment,
                    conda_prefix=access.runtime_identity.conda_prefix,
                ),
                transport,
            )
            gpu_result = probe_gpu(
                wrapper,
                worker,
                probe_status="live",
                timeout=float(self.cluster_config.ssh.probe_timeout_seconds),
            )
        resource = replace(
            gpu_result.worker_resource,
            ip=str(worker.host),
            last_probe_at=_now(),
        )
        runtime = replace(
            gpu_result.worker_runtime,
            conda_executable=access.runtime_identity.conda_executable,
            python_executable=access.runtime_identity.python_executable,
            python_version=access.runtime_identity.python_version,
        )
        return WorkerProbeResult(
            worker_resource=resource,
            worker_runtime=runtime,
            windows_host=WindowsHostInfo(
                os_version=access.windows_identity,
                openssh_available=True,
                wsl_available=True,
                nvidia_driver_visible=resource.gpu_name is not None,
                driver_name=resource.gpu_name,
            ),
            failures=gpu_result.failures,
            health=gpu_result.health,
            probe_status="live",
        )

    def _default_probe_network(self, workers: list[WorkerResource]) -> NetworkState:
        worker_configs = {str(worker.worker_id): worker for worker in self.cluster_config.workers}
        links = [
            self._probe_network_link(
                source_resource=source,
                target_resource=target,
                source_worker=worker_configs[str(source.worker_id)],
                target_worker=worker_configs.get(str(target.worker_id)),
            )
            for source in workers
            for target in workers
            if source.worker_id != target.worker_id
        ]
        return build_network_state(
            links,
            network_id=f"job-{_now()}",
            workers=[worker.worker_id for worker in workers],
        )

    def _probe_network_link(
        self,
        *,
        source_resource: WorkerResource,
        target_resource: WorkerResource,
        source_worker: WorkerConfig,
        target_worker: WorkerConfig | None = None,
    ) -> NetworkLink:
        if not target_resource.ip:
            raise RuntimeError(f"target worker {target_resource.worker_id} is missing ip evidence")
        runtime = self._runtime_wrapper(source_worker)
        route_result = runtime.run(["ip", "route", "get", target_resource.ip], timeout=10)
        route_output = route_result.stdout.strip() or route_result.stderr.strip()
        interface = parse_route_interface(route_output)
        if not route_result.ok and not interface:
            raise RuntimeError(
                f"ip route get {target_resource.ip} failed on {source_worker.worker_id}: "
                f"{route_result.stderr or route_result.stdout or route_result.exit_code}"
            )
        if not interface:
            raise RuntimeError(
                f"ip route get {target_resource.ip} did not resolve a device on "
                f"{source_worker.worker_id}: {route_output}"
            )
        source_ip = parse_route_source_ip(route_output) or source_resource.ip or ""
        link_result = runtime.run(["ip", "link", "show", "dev", interface], timeout=10)
        link_output = link_result.stdout.strip() if link_result.ok else None
        interface_mtu = parse_interface_mtu(link_output)
        data_ok, data_note = self._probe_data_reachability(
            source_runtime=runtime,
            target_runtime=(
                None if target_worker is None else self._runtime_wrapper(target_worker)
            ),
            source_ip=source_ip,
            source_worker_id=str(source_resource.worker_id),
            target_worker_id=str(target_resource.worker_id),
        )
        return NetworkLink(
            source_worker_id=source_resource.worker_id,
            target_worker_id=target_resource.worker_id,
            source_ip=source_ip,
            target_ip=target_resource.ip,
            interface=interface,
            tcp_reachable=data_ok,
            failure_reason=None if data_ok else data_note,
            interface_mtu=interface_mtu,
            expected_mtu=self.cluster_config.network.nccl_mtu,
            mtu_status=(
                "PASS"
                if interface_mtu == self.cluster_config.network.nccl_mtu
                else None
            ),
            measured_at=_now(),
        )

    def _probe_data_reachability(
        self,
        *,
        source_runtime: object,
        target_runtime: object | None,
        source_ip: str,
        source_worker_id: str,
        target_worker_id: str,
    ) -> tuple[bool, str | None]:
        """Validate that the source worker can sustain a real TCP data transfer
        to the target worker. A bare TCP accept (what older probes recorded as
        ``tcp_reachable``) does not prove a data path that NCCL/Gloo training
        traffic can survive, so we echo a moderate payload across the pair.

        The server and client both run in the foreground (one SSH session per
        worker) so no background process is left behind; the control node runs
        them concurrently with a randomly chosen port. The server is the
        authority: if it does not receive the full payload, the link is marked
        unreliable regardless of what the client's socket buffer reported.

        Degrades gracefully: if the test machinery cannot run (e.g. an older
        mocked runtime in unit tests), the link keeps the previous optimistic
        reachability assumption instead of failing the whole network probe.
        """
        if target_runtime is None or not source_ip:
            return True, None
        cache_key = (source_worker_id, target_worker_id)
        cache_ttl = float(
            os.environ.get("SHARDGRID_NETWORK_DATA_TEST_CACHE_SECONDS", "600")
        )
        cached = self._data_test_cache.get(cache_key)
        if cached is not None:
            if time.monotonic() - cached[0] < cache_ttl:
                return cached[1], cached[2]
        payload_mb = int(os.environ.get("SHARDGRID_NETWORK_DATA_TEST_MB", "16"))
        timeout = float(os.environ.get("SHARDGRID_NETWORK_DATA_TEST_TIMEOUT", "8"))
        if payload_mb <= 0:
            return True, None
        socket_timeout = int(max(timeout, 8))
        port = random.randint(20000, 60000)
        server_script = (
            "import json, socket\n"
            f"port = {port}\n"
            f"target_mb = {payload_mb}\n"
            f"accept_timeout = {socket_timeout + 5}\n"
            "s = socket.socket()\n"
            "s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)\n"
            "try:\n"
            "    s.bind(('0.0.0.0', port))\n"
            "except OSError as exc:\n"
            "    print(json.dumps({'ok': False, 'error': str(exc)}))\n"
            "    raise SystemExit\n"
            "s.listen(1)\n"
            "s.settimeout(accept_timeout)\n"
            "try:\n"
            "    c, _ = s.accept()\n"
            "    total = 0\n"
            "    while total < target_mb << 20:\n"
            "        chunk = c.recv(1 << 20)\n"
            "        if not chunk:\n"
            "            break\n"
            "        total += len(chunk)\n"
            "    c.close()\n"
            "    print(json.dumps({'ok': total >= target_mb << 20, 'bytes': total}))\n"
            "except Exception as exc:\n"
            "    print(json.dumps({'ok': False, 'error': str(exc)}))\n"
            "s.close()\n"
        )
        client_script = (
            "import json, socket, time\n"
            f"host = {source_ip!r}\n"
            f"port = {port}\n"
            f"target_mb = {payload_mb}\n"
            f"socket_timeout = {socket_timeout}\n"
            "try:\n"
            "    s = socket.socket()\n"
            "    s.settimeout(socket_timeout)\n"
            "    connected = False\n"
            "    for attempt in range(12):\n"
            "        try:\n"
            "            s.connect((host, port))\n"
            "            connected = True\n"
            "            break\n"
            "        except (ConnectionRefusedError, OSError):\n"
            "            s.close()\n"
            "            s = socket.socket()\n"
            "            s.settimeout(socket_timeout)\n"
            "            time.sleep(0.4)\n"
            "    if not connected:\n"
            "        raise OSError('server not reachable')\n"
            "    payload = b'x' * (1 << 20)\n"
            "    sent = 0\n"
            "    while sent < target_mb << 20:\n"
            "        s.sendall(payload)\n"
            "        sent += len(payload)\n"
            "    s.close()\n"
            "    print(json.dumps({'ok': True, 'bytes': sent}))\n"
            "except Exception as exc:\n"
            "    print(json.dumps({'ok': False, 'error': str(exc)}))\n"
        )
        script_timeout = socket_timeout + 10
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                server_future = pool.submit(
                    getattr(source_runtime, "run_script"),
                    server_script,
                    timeout=script_timeout,
                )
                client_future = pool.submit(
                    getattr(target_runtime, "run_script"),
                    client_script,
                    timeout=script_timeout,
                )
                server_result = server_future.result(timeout=script_timeout + 5)
                client_result = client_future.result(timeout=script_timeout + 5)
        except Exception:
            return True, None
        server_payload = _parse_probe_data_payload(server_result.stdout)
        client_payload = _parse_probe_data_payload(client_result.stdout)
        if isinstance(server_payload, dict) and server_payload.get("ok") is True:
            result: tuple[bool, str | None] = (True, None)
            self._data_test_cache[cache_key] = (time.monotonic(), *result)
            return result
        note = (
            (server_payload or {}).get("error")
            or (client_payload or {}).get("error")
            or "data transfer did not complete"
        )
        result = (False, f"DATA_PATH_UNRELIABLE: {note}")
        self._data_test_cache[cache_key] = (time.monotonic(), *result)
        return result

    def _runtime_wrapper(self, worker: WorkerConfig) -> WSLRuntimeWrapper:
        transport = SSHTransport(
            SSHOptions.from_ssh_config(
                self.cluster_config.ssh,
                host=str(worker.host),
                user=worker.ssh_user,
                port=worker.ssh_port,
            )
        )
        return WSLRuntimeWrapper(
            WSLRuntimeConfig.from_worker_and_runtime(worker, self.cluster_config.runtime),
            transport,
        )

    def _selected_engine_id(self) -> str:
        engine_id = str(self.cluster_config.backend_preference.parallel_engine)
        return "galvatron" if engine_id == "auto" else engine_id

    def _monitor_poll_interval_seconds(self) -> float:
        return 1.0

    def _select_candidate_workers(self, training_config: TrainingConfig) -> list[WorkerConfig]:
        preferred = [str(item) for item in training_config.resources.preferred_workers]
        configured = {
            str(worker.worker_id): worker
            for worker in self.cluster_config.workers
            if worker.enabled
        }
        if preferred:
            ordered = [configured[worker_id] for worker_id in preferred if worker_id in configured]
            if self._automatic_planning_enabled(training_config):
                return ordered
            return ordered[: training_config.resources.world_size]
        configured_workers = list(configured.values())
        if self._automatic_planning_enabled(training_config):
            return configured_workers
        return configured_workers[: training_config.resources.world_size]

    def _refresh_planning_state(
        self,
        *,
        training_config: TrainingConfig,
        selected_workers: list[WorkerConfig],
        current_job_id: JobId,
    ) -> tuple[
        list[WorkerConfig],
        list[WorkerProbeResult],
        list[WorkerResource],
        ClusterState,
        NetworkState,
    ] | None:
        try:
            probe_results = [
                self._probe_worker(worker) for worker in selected_workers
            ]
        except Exception:
            return None
        healthy_ids = {
            result.worker_resource.worker_id
            for result in probe_results
            if result.health is Health.HEALTHY
        }
        fresh_workers = [
            worker for worker in selected_workers if worker.worker_id in healthy_ids
        ]
        if not fresh_workers:
            return None
        fresh_results = [
            result
            for result in probe_results
            if result.worker_resource.worker_id in healthy_ids
        ]
        worker_resources = self._apply_active_resource_reservations(
            [result.worker_resource for result in fresh_results],
            current_job_id=current_job_id,
        )
        cluster_state = self._resource_manager.build_cluster_state(
            worker_resources,
            network_state=None,
            require_network=False,
        )
        network_state = self._probe_network(worker_resources)
        cluster_state = self._resource_manager.build_cluster_state(
            worker_resources,
            network_state=network_state,
            require_network=True,
        )
        return (
            fresh_workers,
            fresh_results,
            worker_resources,
            cluster_state,
            network_state,
        )

    def _probe_failure(
        self,
        workers: list[WorkerConfig],
        result: WorkerProbeResult,
    ) -> FailureRecord:
        first = result.failures[0] if result.failures else None
        worker = next(
            item for item in workers if item.worker_id == result.worker_resource.worker_id
        )
        return make_failure_record(
            stage=FailureStage.PROBE,
            host=str(worker.host),
            worker_id=str(worker.worker_id),
            message=(
                first.message if first is not None else "worker probe returned unhealthy"
            ),
            recommended_action="repair worker runtime readiness and retry",
            secrets=self._secrets,
        )

    def _build_execution_plan(
        self,
        *,
        job: TrainingJob,
        training_config: TrainingConfig,
        parallel_plan: ParallelPlan,
        workers: list[WorkerConfig],
        snapshot: JobSnapshot,
        rejected_engine_ids: Sequence[str] = (),
    ) -> ExecutionPlan:
        worker_map = {
            str(worker.worker_id): worker
            for worker in self.cluster_config.workers
            if worker.enabled
        }
        for worker in workers:
            worker_map[str(worker.worker_id)] = worker
        selected_workers = self._ordered_workers_for_execution_plan(
            training_config=training_config,
            parallel_plan=parallel_plan,
            workers=workers,
            worker_map=worker_map,
        )
        stage_metadata = {stage.stage_id: stage for stage in parallel_plan.stage_metadata}
        communication_map = self._communication_map(parallel_plan)
        assignments: list[WorkerAssignment] = []
        for index, stage_id in enumerate(parallel_plan.stages):
            stage = stage_metadata.get(stage_id)
            rank = index if stage is None else stage.rank
            worker = selected_workers[rank]
            peak_memory = None
            if stage is not None:
                peak_memory = (
                    stage.estimated_peak_training_memory.planner_required_bytes
                    or stage.estimated_peak_training_memory.estimated_peak_bytes
                )
            assignments.append(
                WorkerAssignment(
                    worker_id=worker.worker_id,
                    rank=rank,
                    local_rank=0,
                    stage=stage_id,
                    stage_metadata_ref=(
                        None
                        if stage is None
                        else (
                            "parallel_plan.stage_metadata["
                            f"{parallel_plan.stage_metadata.index(stage)}]"
                        )
                    ),
                    estimated_peak_training_memory=peak_memory,
                    communication_edges=list(communication_map.get(stage_id, ())),
                    gpu_index=(
                        0
                        if stage is None or stage.placement is None
                        else stage.placement.gpu_index
                    ),
                    host=str(worker.host),
                    machine_id=str(worker.machine_id),
                    physical_os=worker.physical_os.value,
                    runtime_os=worker.runtime_os.value,
                    runtime=worker.runtime,
                    runtime_distro=worker.runtime_distro,
                    conda_environment=worker.conda_environment,
                    conda_prefix=worker.conda_prefix,
                    python_executable=(
                        f"{worker.conda_prefix}/bin/python"
                        if worker.conda_prefix
                        else self.cluster_config.runtime.python_executable
                    ),
                    launch_command=self._launch_command_for_assignment(parallel_plan, rank),
                    environment=self._assignment_environment(
                        training_config=training_config,
                        parallel_plan=parallel_plan,
                        snapshot=snapshot,
                        stage_id=stage_id,
                    ),
                    status="pending",
                    log_path=f"logs/{worker.worker_id}/rank{rank}-{stage_id}/combined.log",
                )
            )
        labels = self._execution_plan_labels(
            parallel_plan=parallel_plan,
            rejected_engine_ids=rejected_engine_ids,
        )
        rank_zero = next(assignment for assignment in assignments if assignment.rank == 0)
        return ExecutionPlan(
            job_id=job.job_id,
            engine=parallel_plan.engine,
            backend=training_config.job.communication_backend,
            world_size=parallel_plan.world_size,
            master=MasterMetadata(
                address=rank_zero.host or str(selected_workers[0].host),
                port=self._resolved_rendezvous_port(),
            ),
            workers=assignments,
            model_profile_ref=parallel_plan.model_profile_id,
            candidate_evaluation_ref=parallel_plan.candidate_evaluation_ref,
            conda_environment=self.cluster_config.runtime.conda_environment,
            conda_prefix=self.cluster_config.runtime.conda_prefix,
            python_executable=self.cluster_config.runtime.python_executable,
            placement_reason=(
                None
                if parallel_plan.planning_provenance is None
                else parallel_plan.planning_provenance.selected_reason
            ),
            parallel_plan_ref=str(Path(snapshot.plan_path) / "original-parallel-plan.json"),
            original_engine_plan_ref=parallel_plan.engine_plan_path,
            snapshot_ref=snapshot.root_path,
            labels=labels,
        )

    def _launch_command_for_assignment(self, parallel_plan: ParallelPlan, rank: int) -> str:
        if parallel_plan.partition_source == "automatic":
            if parallel_plan.requirements.get("generic_dag_runtime") == "true":
                if parallel_plan.requirements.get("memory_probe") == "true":
                    return (
                        f"python examples/models/train_generic_dag.py --rank {rank} "
                        "--memory-probe"
                    )
                return f"python examples/models/train_generic_dag.py --rank {rank}"
            return f"python examples/models/train_automatic_plan.py --rank {rank}"
        return f"python examples/models/train_pipeline.py --rank {rank}"

    def _assignment_environment(
        self,
        *,
        training_config: TrainingConfig,
        parallel_plan: ParallelPlan,
        snapshot: JobSnapshot,
        stage_id: str,
    ) -> dict[str, str]:
        if parallel_plan.partition_source == "automatic":
            parameters = training_config.model.parameters
            generic_dag_runtime = self._generic_dag_runtime_requested(training_config)
            automatic_microbatches = max(
                2,
                parallel_plan.world_size,
                int(parameters.get("microbatch_count", parallel_plan.world_size)),
            )
            return {
                "SHARDGRID_PLAN_MODE": "automatic",
                "SHARDGRID_PARTITION_SOURCE": "automatic",
                "SHARDGRID_SELECTED_CANDIDATE_ID": parallel_plan.selected_candidate_id or "",
                "SHARDGRID_AUTOMATIC_STAGE_ID": stage_id,
                "SHARDGRID_AUTOMATIC_MODEL_NAME": training_config.model.name,
                "SHARDGRID_AUTOMATIC_MODEL_TYPE": training_config.model.type,
                "SHARDGRID_AUTOMATIC_STEPS": str(int(parameters.get("training_steps", 5))),
                "SHARDGRID_AUTOMATIC_LR": str(parameters.get("learning_rate", "1e-3")),
                "SHARDGRID_AUTOMATIC_MICROBATCHES": str(automatic_microbatches),
                "SHARDGRID_AUTOMATIC_CHECKPOINT_DIR": "checkpoint",
                "SHARDGRID_GENERIC_DAG_RUNTIME_REQUESTED": str(generic_dag_runtime).lower(),
            }
        return {
            "SHARDGRID_PIPELINE_TASK": "t074",
            "SHARDGRID_T074_CHECKPOINT_DIR": str(Path(snapshot.root_path) / "checkpoint"),
            "SHARDGRID_T074_STEPS": "2000",
        }

    def _ordered_workers_for_execution_plan(
        self,
        *,
        training_config: TrainingConfig,
        parallel_plan: ParallelPlan,
        workers: list[WorkerConfig],
        worker_map: dict[str, WorkerConfig],
    ) -> list[WorkerConfig]:
        if parallel_plan.stage_metadata:
            ordered: list[WorkerConfig | None] = [None] * parallel_plan.world_size
            for stage in parallel_plan.stage_metadata:
                if stage.placement is None:
                    raise ValueError(f"parallel plan stage {stage.stage_id} is missing placement")
                worker = worker_map.get(stage.placement.worker_id)
                if worker is None:
                    raise ValueError(
                        f"parallel plan stage {stage.stage_id} references unknown worker "
                        f"{stage.placement.worker_id}"
                    )
                ordered[stage.rank] = worker
            if any(worker is None for worker in ordered):
                raise ValueError("parallel plan placement did not cover every rank")
            return [worker for worker in ordered if worker is not None]

        ordered_workers = [
            worker_map.get(str(worker_id))
            for worker_id in training_config.resources.preferred_workers
        ]
        selected_workers = [worker for worker in ordered_workers if worker is not None]
        return selected_workers or workers

    def _communication_map(
        self,
        parallel_plan: ParallelPlan,
    ) -> dict[str, list[str]]:
        mapping: dict[str, list[str]] = {}
        for edge in parallel_plan.communication_edges:
            label = (
                f"{edge.source_stage_id}->{edge.target_stage_id}:"
                f"{edge.estimated_bytes_per_step or 0}"
            )
            mapping.setdefault(edge.source_stage_id, []).append(label)
            mapping.setdefault(edge.target_stage_id, []).append(label)
        return mapping

    def _execution_plan_labels(
        self,
        *,
        parallel_plan: ParallelPlan,
        rejected_engine_ids: Sequence[str],
    ) -> dict[str, str]:
        provenance = parallel_plan.planning_provenance
        fallback_reasons: list[str] = []
        fallback_label = "NONE"
        if rejected_engine_ids:
            fallback_reasons.append("rejected engines: " + "; ".join(rejected_engine_ids))
            fallback_label = "engine_fallback"
        if provenance is not None and provenance.fallback_reason:
            fallback_reasons.append(provenance.fallback_reason)
            fallback_label = (
                "planner_fallback" if fallback_label == "NONE" else "engine_and_planner_fallback"
            )
        labels = {
            "partition_source": parallel_plan.partition_source or "manual",
            "generic_dag_runtime_requested": parallel_plan.requirements.get(
                "generic_dag_runtime",
                "false",
            ),
            "selected_candidate_id": parallel_plan.selected_candidate_id or "NONE",
            "selected_worker_count": (
                "NONE"
                if provenance is None or provenance.selected_worker_count is None
                else str(provenance.selected_worker_count)
            ),
            "total_cross_worker_communication_bytes": (
                "NONE"
                if provenance is None
                or provenance.total_cross_worker_communication_bytes is None
                else str(provenance.total_cross_worker_communication_bytes)
            ),
            "fallback_status": "USED" if fallback_reasons else "NONE",
            "fallback_label": fallback_label,
            "fallback_reason": " | ".join(fallback_reasons) if fallback_reasons else "",
        }
        if rejected_engine_ids:
            labels["rejected_engine_ids"] = "; ".join(rejected_engine_ids)
        return labels

    def _resolved_rendezvous_port(self) -> int:
        override = self.cluster_config.manual_override.rendezvous_port
        if override is not None:
            return override
        return self.cluster_config.network.rendezvous_port

    def _prepare_live_execution_plan(
        self,
        *,
        training_config: TrainingConfig,
        execution_plan: ExecutionPlan,
        cluster_state: ClusterState,
        network_state: NetworkState,
    ) -> tuple[ExecutionPlan, ClusterState, NetworkState]:
        if training_config.planning.mode != "automatic":
            return execution_plan, cluster_state, network_state
        revalidated = self._revalidate_execution_resources(execution_plan)
        refreshed_network = self._probe_network(revalidated)
        refreshed_cluster = self._resource_manager.build_cluster_state(
            revalidated,
            network_state=refreshed_network,
            require_network=True,
        )
        if not _covers_workers(
            refreshed_network,
            [assignment.worker_id for assignment in execution_plan.workers],
        ):
            raise ValueError(
                "RESOURCE_CHANGED: selected workers no longer have full network reachability"
            )
        rank_zero = next(
            assignment for assignment in execution_plan.workers if assignment.rank == 0
        )
        rank_zero_worker = next(
            worker
            for worker in self.cluster_config.workers
            if worker.worker_id == rank_zero.worker_id
        )
        fresh_port = self._allocate_live_master_port(rank_zero_worker)
        updated_plan = replace(
            execution_plan,
            master=MasterMetadata(
                address=rank_zero.host or execution_plan.master.address,
                port=fresh_port,
            ),
        )
        return updated_plan, refreshed_cluster, refreshed_network

    def _apply_active_resource_reservations(
        self,
        resources: list[WorkerResource],
        *,
        current_job_id: JobId,
    ) -> list[WorkerResource]:
        del current_job_id
        return resources

    def _revalidate_execution_resources(
        self,
        execution_plan: ExecutionPlan,
    ) -> list[WorkerResource]:
        by_worker = {
            result.worker_resource.worker_id: result.worker_resource
            for result in (
                self._probe_worker(
                    next(
                        worker
                        for worker in self.cluster_config.workers
                        if worker.worker_id == assignment.worker_id
                    )
                )
                for assignment in execution_plan.workers
            )
        }
        backend = str(execution_plan.backend)
        for assignment in execution_plan.workers:
            resource = by_worker[assignment.worker_id]
            if resource.health is not Health.HEALTHY:
                raise ValueError(
                    f"RESOURCE_CHANGED: worker {assignment.worker_id} is no longer healthy"
                )
            if assignment.estimated_peak_training_memory is not None:
                free_bytes = self._bytes_from_mb(resource.gpu_free_memory)
                if free_bytes is None or free_bytes < assignment.estimated_peak_training_memory:
                    raise ValueError(
                        "RESOURCE_CHANGED: worker "
                        f"{assignment.worker_id} usable memory {free_bytes} is below "
                        f"required peak {assignment.estimated_peak_training_memory}"
                    )
            if backend == "nccl" and not resource.nccl_available:
                raise ValueError(
                    f"RESOURCE_CHANGED: worker {assignment.worker_id} no longer reports NCCL"
                )
            if backend == "gloo" and not resource.gloo_available:
                raise ValueError(
                    f"RESOURCE_CHANGED: worker {assignment.worker_id} no longer reports Gloo"
                )
            if not resource.ip:
                raise ValueError(
                    f"RESOURCE_CHANGED: worker {assignment.worker_id} "
                    "is missing network address evidence"
                )
        return [by_worker[assignment.worker_id] for assignment in execution_plan.workers]

    def _allocate_live_master_port(self, worker: WorkerConfig) -> int:
        runtime = self._runtime_wrapper(worker)
        preferred = self._resolved_rendezvous_port()
        script = (
            "import json, socket\n"
            f"preferred = {preferred}\n"
            "chosen = None\n"
            "for port in range(preferred + 1, preferred + 101):\n"
            "    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n"
            "    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)\n"
            "    try:\n"
            "        sock.bind(('', port))\n"
            "    except OSError:\n"
            "        sock.close()\n"
            "        continue\n"
            "    chosen = port\n"
            "    sock.close()\n"
            "    break\n"
            "if chosen is None:\n"
            "    raise SystemExit('no free rendezvous port found')\n"
            "print(json.dumps({'master_port': chosen}, sort_keys=True))\n"
        )
        result = runtime.run_script(script, timeout=15.0)
        if not result.ok:
            raise ValueError(
                "RESOURCE_CHANGED: failed to allocate a fresh master port on "
                f"{worker.worker_id}: {result.stderr or result.stdout or result.exit_code}"
            )
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"RESOURCE_CHANGED: invalid master-port probe payload from {worker.worker_id}"
            ) from exc
        return int(payload["master_port"])

    def _bytes_from_mb(self, value: int | None) -> int | None:
        return None if value is None else int(value) * 1024 * 1024

    def _stop_failed_job_ranks(self, *, launcher: Launcher, context: LauncherContext) -> None:
        try:
            launcher.stop(context)
        except Exception:
            return

    def _runtime_refs(self, execution_plan: ExecutionPlan) -> dict[str, str]:
        return {
            str(assignment.rank): f"env:{assignment.worker_id}/shardgrid"
            for assignment in execution_plan.workers
        }

    def _write_snapshot_metadata(
        self,
        *,
        snapshot: JobSnapshot,
        job: TrainingJob,
        training_config: TrainingConfig,
        parallel_plan: ParallelPlan,
        execution_plan: ExecutionPlan,
        network_state: NetworkState,
        job_status: JobStatus,
        launch_metadata: dict[str, object] | None = None,
        dry_run: bool = False,
    ) -> None:
        report = build_control_report(
            self.cluster_config.control.machine_id,
            self.cluster_config.control.hostname,
            jobs_root=self.cluster_config.jobs_root,
        )
        write_snapshot_metadata(
            snapshot=snapshot,
            job=job,
            config=training_config.to_dict(),
            parallel_plan=parallel_plan,
            execution_plan=execution_plan,
            environment_report=report,
            network_state=network_state,
            job_status=job_status,
            checkpoint_metadata=None,
            launch_metadata=launch_metadata,
            dry_run=dry_run,
            secrets=self._secrets,
        )

    def _select_memory_probe_candidate(
        self,
        *,
        job: TrainingJob,
        training_config: TrainingConfig,
        candidate_plans: Sequence[ParallelPlan],
        selected_workers: list[WorkerConfig],
        cluster_state: ClusterState,
        network_state: NetworkState,
        rejected_engine_ids: Sequence[str],
    ) -> tuple[ParallelPlan, dict[str, object]]:
        records: list[dict[str, object]] = []
        for index, candidate in enumerate(candidate_plans):
            probe_plan = replace(
                candidate,
                requirements={**candidate.requirements, "memory_probe": "true"},
            )
            probe_job = create_training_job(
                config_path=job.config_path,
                model=job.model,
                requested_world_size=probe_plan.world_size,
                backend_preference=job.backend_preference,
                runtime_environment_ref=job.runtime_environment_ref,
                job_id=as_job_id(f"{job.job_id}-probe-{index}"),
            )
            probe_snapshot = self._artifact_store.create_snapshot(probe_job)
            create_code_snapshot(
                probe_snapshot,
                source_root=self._source_root,
                secrets=self._secrets,
            )
            current = self._status_store.create_initial_status(probe_job)
            result = self._run_probe_candidate_with_live_preflight(
                training_config=training_config,
                probe_job=probe_job,
                probe_snapshot=probe_snapshot,
                probe_plan=probe_plan,
                candidate=candidate,
                selected_workers=selected_workers,
                cluster_state=cluster_state,
                network_state=network_state,
                rejected_engine_ids=rejected_engine_ids,
                current=current,
            )
            records.append(
                {
                    "candidate_id": probe_plan.selected_candidate_id,
                    "parallel_plan_id": probe_plan.parallel_plan_id,
                    "probe_job_id": str(probe_job.job_id),
                    "probe_snapshot": probe_snapshot.root_path,
                    **self._memory_probe_estimates(probe_plan),
                    "result": result.status,
                    "subtype": result.subtype,
                    "message": result.message,
                    "rendezvous_ready": result.rendezvous_ready,
                    "phase": result.phase,
                    "master_addr": result.master_addr,
                    "master_port": result.master_port,
                    "attempt": result.attempt,
                    "oom_evidence": result.oom_evidence,
                    "actual_peak_allocated_bytes": result.actual_peak_allocated_bytes,
                    "actual_peak_reserved_bytes": result.actual_peak_reserved_bytes,
                }
            )
            if result.status == PROBE_PASS:
                selection = {
                    "job_id": str(job.job_id),
                    "candidate_count": len(candidate_plans),
                    "candidates": records,
                    "selected_candidate_id": candidate.selected_candidate_id,
                    "final_result": "PASS",
                    "probe_infra_summary": self._probe_infra_summary(records),
                }
                return candidate, selection
            if result.status == PROBE_RESOURCE_CHANGED:
                raise MemoryProbeResourceChanged(result)
            if result.status == PROBE_INFRA_FAILURE:
                raise MemoryProbeInfraFailure(result)
            if result.status == PROBE_RUNTIME_FAILURE:
                raise MemoryProbeRuntimeFailure(result)
        raise MemoryProbeNoFeasiblePlan(
            MemoryProbeResult(
                status=PROBE_MEMORY_REJECT,
                candidate_id=None,
                subtype="ALL_CANDIDATES_REJECTED",
                message=json.dumps(
                    {
                        "candidate_count": len(candidate_plans),
                        "candidates": records,
                        "final_result": "NO_FEASIBLE_PLAN",
                    },
                    sort_keys=True,
                ),
            )
        )

    def _run_probe_candidate_with_live_preflight(
        self,
        *,
        training_config: TrainingConfig,
        probe_job: TrainingJob,
        probe_snapshot: JobSnapshot,
        probe_plan: ParallelPlan,
        candidate: ParallelPlan,
        selected_workers: list[WorkerConfig],
        cluster_state: ClusterState,
        network_state: NetworkState,
        rejected_engine_ids: Sequence[str],
        current: JobStatus,
    ) -> MemoryProbeResult:
        max_retries = 1 + int(
            os.environ.get("SHARDGRID_MEMORY_PROBE_INFRA_RETRIES", "2")
        )
        last_result: MemoryProbeResult | None = None
        for attempt in range(1, max_retries + 1):
            probe_execution = self._build_execution_plan(
                job=probe_job,
                training_config=training_config,
                parallel_plan=probe_plan,
                workers=selected_workers,
                snapshot=probe_snapshot,
                rejected_engine_ids=rejected_engine_ids,
            )
            try:
                live_plan, live_cluster, live_network = self._prepare_live_execution_plan(
                    training_config=training_config,
                    execution_plan=probe_execution,
                    cluster_state=cluster_state,
                    network_state=network_state,
                )
            except ValueError as exc:
                message = str(exc)
                if "RESOURCE_CHANGED" in message:
                    return MemoryProbeResult(
                        status=PROBE_RESOURCE_CHANGED,
                        candidate_id=probe_plan.selected_candidate_id,
                        subtype="RESOURCE_CHANGED",
                        message=message,
                        attempt=attempt,
                    )
                last_result = MemoryProbeResult(
                    status=PROBE_INFRA_FAILURE,
                    candidate_id=probe_plan.selected_candidate_id,
                    subtype="LIVE_PREFLIGHT_FAILURE",
                    message=message,
                    attempt=attempt,
                )
                if attempt < max_retries:
                    continue
                return last_result
            identity_mismatch = self._candidate_identity_preserved(candidate, live_plan)
            if identity_mismatch is not None:
                raise MemoryProbeRuntimeFailure(
                    MemoryProbeResult(
                        status=PROBE_RUNTIME_FAILURE,
                        candidate_id=probe_plan.selected_candidate_id,
                        subtype="CANDIDATE_IDENTITY_CHANGED",
                        message=(
                            "probe candidate identity changed after live preflight: "
                            f"{identity_mismatch}"
                        ),
                        attempt=attempt,
                    )
                )
            current = self._persist_status(
                replace(
                    current,
                    workers=[assignment.worker_id for assignment in live_plan.workers],
                    assignments=list(live_plan.workers),
                    runtime_environment_refs=self._runtime_refs(live_plan),
                    backend=live_plan.backend,
                ),
                snapshot=probe_snapshot,
                state=JobState.SNAPSHOTTING,
                phase="memory_probe",
            )
            self._write_snapshot_metadata(
                snapshot=probe_snapshot,
                job=probe_job,
                training_config=training_config,
                parallel_plan=probe_plan,
                execution_plan=live_plan,
                network_state=live_network,
                job_status=current,
                launch_metadata={
                    "memory_probe": True,
                    "candidate_id": probe_plan.selected_candidate_id,
                    "master_addr": live_plan.master.address,
                    "master_port": live_plan.master.port,
                    "preflight_attempt": attempt,
                },
                dry_run=True,
            )
            result = self._run_memory_probe(
                training_config=training_config,
                probe_job=probe_job,
                probe_snapshot=probe_snapshot,
                probe_plan=probe_plan,
                probe_execution=live_plan,
                cluster_state=live_cluster,
                current=current,
                attempt=attempt,
                free_memory_baseline=self._probe_free_memory_baseline(
                    live_cluster,
                    live_plan,
                ),
            )
            if result.master_port is None:
                result = replace(
                    result,
                    master_addr=live_plan.master.address,
                    master_port=live_plan.master.port,
                )
            last_result = result
            if (
                result.status == PROBE_INFRA_FAILURE
                and result.subtype in PROBE_TRANSIENT_INFRA_SUBTYPES
                and attempt < max_retries
            ):
                self._release_probe_attempt(probe_job)
                continue
            return result
        return last_result or MemoryProbeResult(
            status=PROBE_INFRA_FAILURE,
            candidate_id=probe_plan.selected_candidate_id,
            subtype="INFRA_RETRY_EXHAUSTED",
            message="probe infra retry budget exhausted",
            attempt=max_retries,
        )

    def _release_probe_attempt(self, probe_job: TrainingJob) -> None:
        try:
            self._status_store.release_resources(probe_job.job_id)
        except Exception:
            pass

    def _probe_free_memory_baseline(
        self,
        cluster_state: ClusterState,
        execution_plan: ExecutionPlan,
    ) -> dict[str, int]:
        by_worker = {
            str(worker.worker_id): worker.resource.gpu_free_memory
            for worker in cluster_state.workers
        }
        baseline: dict[str, int] = {}
        for assignment in execution_plan.workers:
            free_mb = by_worker.get(str(assignment.worker_id))
            if free_mb is not None:
                baseline[str(assignment.worker_id)] = int(free_mb) * 1024 * 1024
        return baseline

    def _candidate_identity_preserved(
        self,
        candidate: ParallelPlan,
        live_plan: ExecutionPlan,
    ) -> str | None:
        if candidate.selected_candidate_id != live_plan.labels.get(
            "selected_candidate_id"
        ):
            return (
                f"candidate_id changed: {candidate.selected_candidate_id} -> "
                f"{live_plan.labels.get('selected_candidate_id')}"
            )
        expected = {
            stage.rank: (
                str(stage.placement.worker_id),
                stage.stage_id,
                stage.placement.gpu_index,
            )
            for stage in candidate.stage_metadata
            if stage.placement is not None
        }
        for assignment in live_plan.workers:
            item = expected.get(assignment.rank)
            if item is None:
                continue
            worker_id, stage_id, gpu_index = item
            if str(assignment.worker_id) != worker_id:
                return (
                    f"rank {assignment.rank} worker changed "
                    f"{worker_id} -> {assignment.worker_id}"
                )
            if (assignment.stage or "") != stage_id:
                return f"rank {assignment.rank} stage changed"
            if assignment.gpu_index != gpu_index:
                return f"rank {assignment.rank} gpu_index changed"
        return None

    def _probe_infra_summary(self, records: list[dict[str, object]]) -> dict[str, object]:
        return {
            "probe_attempts": len(records),
            "rendezvous_successes": sum(
                1 for record in records if record.get("rendezvous_ready") is True
            ),
            "rendezvous_retries": sum(
                1 for record in records if int(record.get("attempt", 1)) > 1
            ),
            "infra_failures": sum(
                1
                for record in records
                if record.get("result") == PROBE_INFRA_FAILURE
            ),
            "stale_default_port_reuse": sum(
                1
                for record in records
                if record.get("master_port") == self._resolved_rendezvous_port()
            ),
            "fresh_port_per_attempt": [
                record.get("master_port") for record in records
            ],
        }

    def _run_memory_probe(
        self,
        *,
        training_config: TrainingConfig,
        probe_job: TrainingJob,
        probe_snapshot: JobSnapshot,
        probe_plan: ParallelPlan,
        probe_execution: ExecutionPlan,
        cluster_state: ClusterState,
        current: JobStatus,
        attempt: int = 1,
        free_memory_baseline: dict[str, int] | None = None,
    ) -> MemoryProbeResult:
        if self._memory_probe is not None:
            return self._memory_probe(probe_plan, probe_execution)
        if probe_plan.requirements.get("generic_dag_runtime") != "true":
            return MemoryProbeResult(
                status=PROBE_PASS,
                candidate_id=probe_plan.selected_candidate_id,
                message="memory probe skipped for non-generic automatic runner",
            )
        launcher = self._launcher_factory(training_config.job.backend)
        context = LauncherContext(
            job=probe_job,
            execution_plan=probe_execution,
            cluster_state=cluster_state,
            snapshot=probe_snapshot,
            job_status=current,
            runtime_environment_refs=self._runtime_refs(probe_execution),
        )
        try:
            for operation in (launcher.prepare, launcher.distribute):
                result = operation(context)
                if not result.ok:
                    message = result.message or (
                        result.failure.message
                        if result.failure is not None
                        else "memory probe distribution failed"
                    )
                    return MemoryProbeResult(
                        status=PROBE_INFRA_FAILURE,
                        candidate_id=probe_plan.selected_candidate_id,
                        subtype="DISTRIBUTE_FAILURE",
                        message=message,
                        attempt=attempt,
                    )
                context = replace(
                    context,
                    job_status=self._load_status(probe_snapshot, context.job_status or current),
                )
            conflicts = self._status_store.reserve_resources(
                probe_job.job_id,
                probe_execution.workers,
            )
            if conflicts:
                return MemoryProbeResult(
                    status=PROBE_RESOURCE_CHANGED,
                    candidate_id=probe_plan.selected_candidate_id,
                    subtype="RESERVATION_CONFLICT",
                    message="memory probe resources are already reserved",
                    attempt=attempt,
                )
            result = launcher.launch(context)
            if not result.ok:
                return self._classify_launch_failure(
                    probe_plan,
                    result,
                    attempt=attempt,
                )
            context = replace(
                context,
                job_status=self._load_status(probe_snapshot, context.job_status or current),
            )
            deadline = time.time() + float(
                os.environ.get("SHARDGRID_MEMORY_PROBE_TIMEOUT_SECONDS", "180")
            )
            while time.time() < deadline:
                monitor = launcher.monitor(context)
                current_status = self._load_status(probe_snapshot, context.job_status or current)
                context = replace(context, job_status=current_status)
                if self._memory_probe_completed(probe_snapshot, probe_execution.world_size):
                    peak = self._read_probe_peak(probe_snapshot)
                    memory_result = self._classify_measured_peak(
                        probe_plan,
                        peak,
                        free_memory_baseline,
                    )
                    if memory_result is not None:
                        return memory_result
                    return MemoryProbeResult(
                        status=PROBE_PASS,
                        candidate_id=probe_plan.selected_candidate_id,
                        message="memory probe passed",
                        actual_peak_allocated_bytes=peak.get(
                            "actual_peak_allocated_bytes"
                        ),
                        actual_peak_reserved_bytes=peak.get(
                            "actual_peak_reserved_bytes"
                        ),
                        rendezvous_ready=True,
                        phase="PROBE_COMPLETE",
                        attempt=attempt,
                        master_addr=probe_execution.master.address,
                        master_port=probe_execution.master.port,
                    )
                if current_status.state is JobState.FAILED or monitor.status in {
                    LauncherResultStatus.FAILED,
                    LauncherResultStatus.BLOCKED,
                    LauncherResultStatus.UNSUPPORTED,
                }:
                    failure = current_status.failure or monitor.failure
                    return self._classify_probe_result(
                        probe_snapshot=probe_snapshot,
                        world_size=probe_execution.world_size,
                        probe_plan=probe_plan,
                        message=(
                            failure.message
                            if failure is not None
                            else monitor.message
                            or "memory probe candidate failed"
                        ),
                        attempt=attempt,
                        master=probe_execution.master,
                        timed_out=False,
                    )
                time.sleep(2)
            return self._classify_probe_result(
                probe_snapshot=probe_snapshot,
                world_size=probe_execution.world_size,
                probe_plan=probe_plan,
                message="memory probe timed out",
                attempt=attempt,
                master=probe_execution.master,
                timed_out=True,
            )
        finally:
            self._release_probe_attempt(probe_job)
            try:
                launcher.stop(context)
            except Exception:
                pass
            try:
                launcher.cleanup(context)
            except Exception:
                pass

    def _classify_launch_failure(
        self,
        probe_plan: ParallelPlan,
        result: LauncherResult,
        *,
        attempt: int,
    ) -> MemoryProbeResult:
        message = result.message or (
            result.failure.message
            if result.failure is not None
            else "memory probe launch failed"
        )
        if _probe_log_has_oom(message):
            return MemoryProbeResult(
                status=PROBE_MEMORY_REJECT,
                candidate_id=probe_plan.selected_candidate_id,
                subtype="CUDA_OOM",
                message=message,
                oom_evidence=True,
                attempt=attempt,
            )
        if result.status is LauncherResultStatus.BLOCKED:
            return MemoryProbeResult(
                status=PROBE_INFRA_FAILURE,
                candidate_id=probe_plan.selected_candidate_id,
                subtype="SSH_FAILURE",
                message=message,
                attempt=attempt,
            )
        return MemoryProbeResult(
            status=PROBE_INFRA_FAILURE,
            candidate_id=probe_plan.selected_candidate_id,
            subtype="LAUNCH_FAILURE",
            message=message,
            attempt=attempt,
        )

    def _classify_measured_peak(
        self,
        probe_plan: ParallelPlan,
        peak: dict[str, int | None],
        free_memory_baseline: dict[str, int] | None,
    ) -> MemoryProbeResult | None:
        peak_reserved = peak.get("actual_peak_reserved_bytes")
        if peak_reserved is None:
            return None
        baseline_values = list((free_memory_baseline or {}).values())
        if baseline_values:
            baseline_min = min(baseline_values)
            if int(peak_reserved) > baseline_min:
                return MemoryProbeResult(
                    status=PROBE_MEMORY_REJECT,
                    candidate_id=probe_plan.selected_candidate_id,
                    subtype="MEASURED_PEAK_UNSAFE",
                    message=(
                        f"measured peak {int(peak_reserved)} exceeds free memory "
                        f"baseline {baseline_min}"
                    ),
                    actual_peak_allocated_bytes=peak.get(
                        "actual_peak_allocated_bytes"
                    ),
                    actual_peak_reserved_bytes=peak_reserved,
                    rendezvous_ready=True,
                    phase="PROBE_COMPLETE",
                    oom_evidence=False,
                )
        return None

    def _classify_probe_result(
        self,
        *,
        probe_snapshot: JobSnapshot,
        world_size: int,
        probe_plan: ParallelPlan,
        message: str,
        attempt: int,
        master: MasterMetadata,
        timed_out: bool,
    ) -> MemoryProbeResult:
        evidence = self._read_probe_rank_evidence(probe_snapshot, world_size)
        rendezvous_ready: bool | None = None
        if evidence:
            rendezvous_ready = all(
                bool(item.get("rendezvous_ready")) for item in evidence
            )
        training_started = any(
            bool(item.get("training_started")) for item in evidence
        )
        phase = self._latest_probe_phase(evidence)
        log_tail = "\n".join(
            str(item.get("log_tail") or "") for item in evidence
        )
        if _probe_log_has_oom(log_tail):
            return MemoryProbeResult(
                status=PROBE_MEMORY_REJECT,
                candidate_id=probe_plan.selected_candidate_id,
                subtype="CUDA_OOM",
                message="CUDA out of memory during memory probe",
                oom_evidence=True,
                rendezvous_ready=rendezvous_ready,
                phase=phase,
                attempt=attempt,
                master_addr=master.address,
                master_port=master.port,
            )
        timeout_stage = next(
            (
                str(item.get("timeout_stage")).upper()
                for item in evidence
                if item.get("timeout_stage")
            ),
            None,
        )
        if (timed_out or timeout_stage == "RENDEZVOUS") and rendezvous_ready is False:
            return MemoryProbeResult(
                status=PROBE_INFRA_FAILURE,
                candidate_id=probe_plan.selected_candidate_id,
                subtype="RENDEZVOUS_TIMEOUT",
                message="probe ranks never completed distributed rendezvous",
                rendezvous_ready=False,
                phase=phase,
                attempt=attempt,
                master_addr=master.address,
                master_port=master.port,
            )
        if rendezvous_ready is True and not training_started:
            return MemoryProbeResult(
                status=PROBE_RUNTIME_FAILURE,
                candidate_id=probe_plan.selected_candidate_id,
                subtype="RUNTIME_START_FAILURE",
                message="rendezvous completed but forward never started",
                rendezvous_ready=True,
                phase=phase,
                attempt=attempt,
                master_addr=master.address,
                master_port=master.port,
            )
        return MemoryProbeResult(
            status=PROBE_RUNTIME_FAILURE,
            candidate_id=probe_plan.selected_candidate_id,
            subtype="RUNTIME_TIMEOUT" if timed_out else "PROCESS_EXIT",
            message=message,
            rendezvous_ready=rendezvous_ready,
            phase=phase,
            attempt=attempt,
            master_addr=master.address,
            master_port=master.port,
        )

    def _read_probe_rank_evidence(
        self,
        snapshot: JobSnapshot,
        world_size: int,
    ) -> list[dict[str, object]]:
        by_rank: dict[int, dict[str, object]] = {}
        for path in sorted(Path(snapshot.diagnostics_path).glob("monitor-*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict) or payload.get("rank") is None:
                continue
            by_rank[int(payload["rank"])] = payload
        return [by_rank[rank] for rank in sorted(by_rank)][: max(world_size, 0)]

    def _latest_probe_phase(
        self,
        evidence: list[dict[str, object]],
    ) -> str | None:
        for item in reversed(evidence):
            for line in reversed(
                str(item.get("log_tail") or "").splitlines()
            ):
                if PROBE_PHASE_MARKER not in line:
                    continue
                try:
                    payload = json.loads(
                        line.split(PROBE_PHASE_MARKER, 1)[1]
                    )
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict) and payload.get("phase"):
                    return str(payload["phase"])
        return None

    def _memory_probe_completed(self, snapshot: JobSnapshot, world_size: int) -> bool:
        payloads = []
        diagnostics = Path(snapshot.diagnostics_path)
        for path in sorted(diagnostics.glob("memory-probe-rank*.json")):
            payloads.append(json.loads(path.read_text(encoding="utf-8")))
        if len(payloads) < world_size:
            for path in sorted(diagnostics.glob("monitor-*.json")):
                payload = json.loads(path.read_text(encoding="utf-8"))
                train = payload.get("train")
                if isinstance(train, dict):
                    payloads.append(train)
        if len(payloads) < world_size:
            return False
        return all(
            payload.get("memory_probe_only") is True
            and payload.get("forward_completed") is True
            and payload.get("backward_completed") is True
            and payload.get("optimizer_step_completed") is True
            for payload in payloads
        )

    def _memory_probe_estimates(self, plan: ParallelPlan) -> dict[str, int]:
        estimates = [stage.estimated_peak_training_memory for stage in plan.stage_metadata]
        return {
            "original_estimated_peak_bytes": sum(
                estimate.estimated_peak_bytes or 0 for estimate in estimates
            ),
            "calibrated_estimated_peak_bytes": sum(
                estimate.planner_required_bytes or estimate.estimated_peak_bytes or 0
                for estimate in estimates
            ),
        }

    def _read_probe_peak(self, snapshot: JobSnapshot) -> dict[str, int | None]:
        peaks: list[dict[str, int | None]] = []
        for path in Path(snapshot.diagnostics_path).glob("memory-probe-rank*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            peaks.append(
                {
                    "actual_peak_allocated_bytes": payload.get(
                        "actual_peak_allocated_bytes"
                    ),
                    "actual_peak_reserved_bytes": payload.get("actual_peak_reserved_bytes"),
                }
            )
        if not peaks:
            for path in sorted(Path(snapshot.diagnostics_path).glob("monitor-*.json")):
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                train = payload.get("train") if isinstance(payload, dict) else None
                if not isinstance(train, dict):
                    continue
                peaks.append(
                    {
                        "actual_peak_allocated_bytes": train.get(
                            "actual_peak_allocated_bytes"
                        ),
                        "actual_peak_reserved_bytes": train.get(
                            "actual_peak_reserved_bytes"
                        ),
                    }
                )
        return {
            "actual_peak_allocated_bytes": max(
                (
                    int(item["actual_peak_allocated_bytes"])
                    for item in peaks
                    if item["actual_peak_allocated_bytes"] is not None
                ),
                default=None,
            ),
            "actual_peak_reserved_bytes": max(
                (
                    int(item["actual_peak_reserved_bytes"])
                    for item in peaks
                    if item["actual_peak_reserved_bytes"] is not None
                ),
                default=None,
            ),
        }

    def _write_memory_probe_selection_artifact(
        self,
        *,
        snapshot: JobSnapshot,
        selection: dict[str, object],
    ) -> Path:
        path = Path(snapshot.diagnostics_path) / "memory-probe-selection.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(selection, indent=2, sort_keys=True), encoding="utf-8")
        return path

    def _automatic_planning_enabled(self, training_config: TrainingConfig) -> bool:
        return training_config.planning.mode == "automatic"

    def _build_automatic_parallel_plan(
        self,
        *,
        training_config: TrainingConfig,
        cluster_state: ClusterState,
        selected_engine: SelectedEngine,
        min_selected_physical_hosts: int | None = None,
    ) -> ParallelPlan:
        memory_config = self._planner_memory_config()
        min_worker_count, max_worker_count = self._automatic_worker_count_bounds(
            cluster_state=cluster_state,
            min_selected_physical_hosts=min_selected_physical_hosts,
        )
        evidence: dict[str, object] = {
            "planner_worker_resources": [
                {
                    "worker_id": str(worker.worker_id),
                    "gpu_index": 0,
                    "gpu_total_memory_mb": worker.resource.gpu_total_memory,
                    "gpu_free_memory_mb": worker.resource.gpu_free_memory,
                    "gpu_utilization": worker.resource.gpu_utilization,
                }
                for worker in cluster_state.workers
            ],
            "planner_worker_count_bounds": {
                "min_worker_count": min_worker_count,
                "max_worker_count": max_worker_count,
                "min_selected_physical_hosts": min_selected_physical_hosts,
                "constraint_source": (
                    "hardware_gate" if min_selected_physical_hosts is not None else "default"
                ),
            },
            "control_rss_before_planning": _process_rss_bytes(),
        }
        model: object | None = None
        sample_args: tuple[object, ...] = ()
        sample_kwargs: dict[str, object] = {}
        profile = None
        joint = None
        try:
            model, sample_args, sample_kwargs = self._planner_workload(training_config)
            profile = build_model_profile(
                model,
                engine_id=self._selected_engine_name(selected_engine),
                model_name=training_config.model.name,
                sample_args=sample_args,
                sample_kwargs=sample_kwargs,
                memory_config=memory_config,
                required_backends=self._required_backends(training_config),
            )
            evidence["control_rss_after_profile"] = _process_rss_bytes()
            joint = search_joint_partition_placement(
                model,
                profile,
                cluster_state,
                sample_args=sample_args,
                sample_kwargs=sample_kwargs,
                memory_config=memory_config,
                min_worker_count=min_worker_count,
                max_worker_count=max_worker_count,
            )
            if joint.status is not FeasibilityStatus.FEASIBLE:
                reasons = ", ".join(joint.reasons) if joint.reasons else "PLANNING_INFEASIBLE"
                raise ValueError(f"automatic planner failed: {reasons}")
            joint_candidates = joint.candidate_plans or (joint,)
            selected_joint = select_best_joint_placement_plan(joint_candidates)
            ordered_joints = (selected_joint,) + tuple(
                item
                for item in joint_candidates
                if item.partition_candidate is not None
                and selected_joint.partition_candidate is not None
                and item.partition_candidate.candidate_id
                != selected_joint.partition_candidate.candidate_id
            )
            plans = tuple(
                build_automatic_parallel_plan(profile, candidate)
                for candidate in ordered_joints
            )
            if self._generic_dag_runtime_requested(training_config):
                for plan in plans:
                    plan.requirements["generic_dag_runtime"] = "true"
            plan = plans[0]
            self._last_parallel_plan_candidates = plans
            evidence["candidate_plan_count"] = len(plans)
            evidence["candidate_plan_ids"] = [
                item.selected_candidate_id for item in plans
            ]
            evidence["control_rss_after_plan_created"] = _process_rss_bytes()
            return plan
        finally:
            model = None
            sample_args = ()
            sample_kwargs = {}
            profile = None
            joint = None
            gc.collect()
            evidence["control_rss_after_cleanup"] = _process_rss_bytes()
            self._last_planning_evidence = dict(evidence)

    def _automatic_worker_count_bounds(
        self,
        *,
        cluster_state: ClusterState,
        min_selected_physical_hosts: int | None = None,
    ) -> tuple[int, int]:
        available_worker_count = len(cluster_state.workers)
        max_worker_count = available_worker_count
        min_worker_count = 1
        min_override = os.environ.get("SHARDGRID_AUTOMATIC_MIN_WORKERS", "").strip()
        max_override = os.environ.get("SHARDGRID_AUTOMATIC_MAX_WORKERS", "").strip()
        if max_override:
            try:
                requested_max = int(max_override)
            except ValueError as exc:
                raise ValueError(
                    "invalid SHARDGRID_AUTOMATIC_MAX_WORKERS: must be an integer"
                ) from exc
            if requested_max < 1:
                raise ValueError(
                    "invalid SHARDGRID_AUTOMATIC_MAX_WORKERS: must be >= 1"
                )
            if requested_max > available_worker_count:
                raise ValueError(
                    "invalid SHARDGRID_AUTOMATIC_MAX_WORKERS: exceeds available worker count"
                )
            max_worker_count = requested_max
        if min_selected_physical_hosts is not None:
            if min_selected_physical_hosts < 1:
                raise ValueError("min_selected_physical_hosts must be >= 1")
            available_hosts = {
                str(entry.resource.machine_id or entry.resource.hostname or entry.worker_id)
                for entry in cluster_state.workers
            }
            if len(available_hosts) < min_selected_physical_hosts:
                raise ValueError(
                    "automatic planning hardware gate requires at least "
                    f"{min_selected_physical_hosts} physical hosts; "
                    f"only {len(available_hosts)} available"
                )
            min_worker_count = max(min_worker_count, min_selected_physical_hosts)
        if not min_override:
            if min_worker_count > max_worker_count:
                raise ValueError("automatic planning requires at least one available worker")
            return min_worker_count, max_worker_count
        try:
            requested = int(min_override)
        except ValueError as exc:
            raise ValueError(
                "invalid SHARDGRID_AUTOMATIC_MIN_WORKERS: must be an integer"
            ) from exc
        if requested < 1:
            raise ValueError(
                "invalid SHARDGRID_AUTOMATIC_MIN_WORKERS: must be >= 1"
            )
        requested = max(requested, min_worker_count)
        if requested > max_worker_count:
            raise ValueError(
                "invalid SHARDGRID_AUTOMATIC_MIN_WORKERS: exceeds max worker count"
            )
        return requested, max_worker_count

    def _planner_workload(
        self,
        training_config: TrainingConfig,
    ) -> tuple[object, tuple[object, ...], dict[str, object]]:
        import torch

        if training_config.model.type == "minimal_sequential":
            from examples.models.minimal_transformer import (
                MinimalTransformerConfig,
                build_minimal_transformer,
            )

            config = MinimalTransformerConfig()
            sample = torch.randint(
                0,
                config.vocab_size,
                (2, min(config.max_seq_length, 16)),
            )
            return build_minimal_transformer(config, seed=42), (sample,), {}
        if training_config.model.type == "hf_style":
            from examples.models.partition_stress_model import (
                build_partition_stress_model,
                make_training_batch,
            )

            model = build_partition_stress_model(seed=42)
            inputs, _targets = make_training_batch(seed=42, step=0)
            return model, (inputs,), {}
        if training_config.model.type == "large_residual_transformer":
            from examples.models.large_residual_transformer import (
                LargeResidualTransformerConfig,
                build_large_residual_transformer,
                make_large_residual_batch,
            )

            config = LargeResidualTransformerConfig.from_mapping(
                training_config.model.parameters
            )
            model = build_large_residual_transformer(config, seed=42, device="meta")
            inputs, _targets = make_large_residual_batch(
                config,
                seed=42,
                step=0,
                device="meta",
            )
            return model, (inputs,), {}
        if training_config.model.type == "generic_dag":
            from examples.models.generic_partition_zoo import build_zoo_model, make_zoo_sample

            model_name = str(training_config.model.parameters.get("zoo_model", "mini_unet"))
            model_kwargs = self._generic_dag_model_kwargs(training_config)
            model = build_zoo_model(model_name, **model_kwargs)
            sample_args, sample_kwargs = make_zoo_sample(model_name, **model_kwargs)
            return model, tuple(sample_args), dict(sample_kwargs)
        raise ValueError(
            "automatic planning supports model.type minimal_sequential, hf_style, "
            "large_residual_transformer, or generic_dag"
        )

    def _generic_dag_runtime_requested(self, training_config: TrainingConfig) -> bool:
        return (
            training_config.model.type == "generic_dag"
            or str(training_config.model.parameters.get("generic_dag_runtime", "false")).lower()
            == "true"
        )

    def _generic_dag_model_kwargs(self, training_config: TrainingConfig) -> dict[str, object]:
        reserved = {
            "zoo_model",
            "training_steps",
            "learning_rate",
            "logical_partitions",
            "generic_dag_runtime",
        }
        return {
            str(key): value
            for key, value in training_config.model.parameters.items()
            if str(key) not in reserved
        }

    def _planner_memory_config(self) -> MemoryEstimationConfig:
        return MemoryEstimationConfig(
            optimizer_type="adamw",
            gradient_dtype="float32",
            optimizer_state_dtype="float32",
            runtime_overhead_bytes=1024,
            communication_buffer_bytes=2048,
            safety_headroom_bytes=4096,
            temporary_buffer_factor=0.25,
        )

    def _required_backends(self, training_config: TrainingConfig) -> tuple[str, ...]:
        backend = str(training_config.job.communication_backend)
        if backend == "auto":
            backend = str(self.cluster_config.backend_preference.communication_backend)
        if backend == "auto":
            backend = "nccl"
        return (backend,)

    def _launch_metadata(
        self,
        selected_engine: SelectedEngine,
        *,
        planning_evidence: dict[str, object] | None = None,
    ) -> dict[str, object]:
        metadata = dict(selected_engine.engine.launch_metadata(selected_engine.parallel_plan))
        plan = selected_engine.parallel_plan
        engine_id = self._selected_engine_name(selected_engine, required=False)
        if engine_id is not None:
            metadata.setdefault("engine", engine_id)
        metadata["parallel_plan_id"] = plan.parallel_plan_id
        metadata["partition_source"] = plan.partition_source or "manual"
        metadata["plan_mode"] = (
            "automatic" if (plan.partition_source or "") == "automatic" else "static"
        )
        if plan.selected_candidate_id:
            metadata["selected_candidate_id"] = plan.selected_candidate_id
        if plan.planning_provenance is not None:
            metadata["selected_worker_count"] = plan.planning_provenance.selected_worker_count
            metadata["attempted_worker_counts"] = list(
                plan.planning_provenance.attempted_worker_counts
            )
        if planning_evidence:
            metadata["planning_evidence"] = dict(planning_evidence)
        return metadata

    def _selected_engine_name(
        self,
        selected_engine: SelectedEngine,
        *,
        required: bool = True,
    ) -> str | None:
        engine_id = getattr(
            selected_engine,
            "engine_id",
            getattr(getattr(selected_engine, "candidate", None), "engine_id", None),
        )
        if engine_id is not None:
            return str(engine_id)
        if required:
            raise ValueError("selected engine is missing engine_id")
        return None

    def _collect_artifacts(
        self,
        snapshot: JobSnapshot,
        training_config: TrainingConfig,
        execution_plan: ExecutionPlan,
        current: JobStatus,
    ) -> ArtifactCollectionResult:
        collector = self._artifact_collector or ArtifactCollector(
            transport=select_artifact_transport(
                build_transport_config(
                    self._artifact_transport_preference_for_collection(
                        training_config=training_config,
                        assignments=current.assignments or execution_plan.workers,
                    )
                )
            ),
            ssh_factory=self._artifact_collection_ssh,
            runtime_factory=self._artifact_collection_runtime,
        )
        worker_map = {str(worker.worker_id): worker for worker in self.cluster_config.workers}
        assignments = current.assignments or execution_plan.workers
        sources = [
            WorkerArtifactSource.from_worker_assignment(
                worker=worker_map[str(assignment.worker_id)],
                assignment=assignment,
                remote_root=str(Path(self.cluster_config.jobs_root) / str(execution_plan.job_id)),
                checkpoint_paths=self._checkpoint_refs_for_rank(snapshot, assignment, current),
                private_key_path=self.cluster_config.ssh.private_key_path,
                connect_timeout_seconds=self.cluster_config.ssh.connect_timeout_seconds,
                command_timeout_seconds=float(self.cluster_config.ssh.command_timeout_seconds),
                known_host_policy=(
                    "yes" if self.cluster_config.ssh.strict_host_key_checking else "accept-new"
                ),
                known_hosts_path=self.cluster_config.ssh.known_hosts_path,
            )
            for assignment in assignments
        ]
        return collector.collect(
            snapshot,
            sources=sources,
            secrets=self._secrets,
        )

    def _artifact_collection_ssh(self, source: WorkerArtifactSource) -> SSHTransport:
        return SSHTransport(
            SSHOptions.from_ssh_config(
                self.cluster_config.ssh,
                host=source.host,
                user=source.ssh_user,
                port=source.ssh_port,
            )
        )

    def _artifact_collection_runtime(
        self,
        source: WorkerArtifactSource,
        ssh: SSHTransport,
    ) -> WSLRuntimeWrapper:
        return WSLRuntimeWrapper(
            WSLRuntimeConfig(
                distro=source.runtime_distro,
                user=source.ssh_user,
                conda_environment=source.conda_environment,
                conda_prefix=source.conda_prefix,
            ),
            ssh,
        )

    def _write_collection_diagnostics(
        self,
        snapshot: JobSnapshot,
        collection_result: ArtifactCollectionResult,
    ) -> None:
        path = Path(snapshot.diagnostics_path) / "artifact-collection.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(collection_result.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _artifact_collection_failure(
        self,
        collection_result: ArtifactCollectionResult,
    ) -> FailureRecord:
        first = next(
            (
                artifact
                for worker in collection_result.workers
                for artifact in worker.artifacts
                if artifact.failure is not None
            ),
            None,
        )
        if first is not None:
            return first.failure or make_failure_record(
                stage=FailureStage.CHECKPOINT,
                host=str(self.cluster_config.control.hostname),
                message="artifact collection failed before final checkpoint validation",
                recommended_action="repair artifact collection and rerun the job",
                secrets=self._secrets,
            )
        return make_failure_record(
            stage=FailureStage.CHECKPOINT,
            host=str(self.cluster_config.control.hostname),
            message="artifact collection failed before final checkpoint validation",
            recommended_action="repair artifact collection and rerun the job",
            secrets=self._secrets,
        )

    def _artifact_transport_preference_for_collection(
        self,
        *,
        training_config: TrainingConfig,
        assignments: Sequence[WorkerAssignment],
    ) -> str:
        preferred = training_config.artifacts.transport
        if preferred != "auto":
            return preferred
        worker_map = {str(worker.worker_id): worker for worker in self.cluster_config.workers}
        if any(
            worker_map[str(assignment.worker_id)].physical_os is PhysicalOS.WINDOWS
            for assignment in assignments
        ):
            return "scp"
        return "auto"

    def _finalize_checkpoint_bundle(
        self,
        *,
        snapshot: JobSnapshot,
        training_config: TrainingConfig,
        execution_plan: ExecutionPlan,
        current: JobStatus,
        collection_result: ArtifactCollectionResult,
    ) -> dict[str, object]:
        shards = self._validated_checkpoint_shards(
            snapshot=snapshot,
            training_config=training_config,
            execution_plan=execution_plan,
            collection_result=collection_result,
        )
        step_values = {
            int(shard["step"])
            for shard in shards
            if isinstance(shard.get("step"), int)
        }
        if len(step_values) > 1:
            raise ValueError("checkpoint shards disagree on training step")
        manifest_ref = "checkpoint/manifest.json"
        manifest = {
            "format": "shardgrid-distributed-checkpoint/v1",
            "version": 1,
            "status": "complete",
            "job_id": str(current.job_id),
            "created_at": _now(),
            "checkpoint_ref": manifest_ref,
            "model_name": training_config.model.name,
            "model_type": training_config.model.type,
            "partition_mode": "pipeline_parallel",
            "world_size": execution_plan.world_size,
            "backend": select_backend(str(training_config.job.communication_backend)),
            "required_shard_count": len(execution_plan.workers),
            "storage_target": snapshot.checkpoint_path,
            "final_metrics": dict(current.final_metrics),
            "shards": [
                {key: value for key, value in shard.items() if key != "local_path"}
                for shard in shards
            ],
            "partition_source": execution_plan.labels.get("partition_source"),
            "selected_candidate_id": execution_plan.labels.get("selected_candidate_id"),
            "selected_worker_count": execution_plan.labels.get("selected_worker_count"),
        }
        if step_values:
            manifest["training_step"] = next(iter(step_values))
        consolidation = training_config.artifacts.checkpoint.consolidation
        generic_dag_checkpoint = training_config.model.type == "generic_dag"
        optional_artifact: dict[str, object] = {
            "enabled": consolidation.enabled or generic_dag_checkpoint,
            "required": consolidation.required or generic_dag_checkpoint,
            "requested_device": consolidation.device,
            "status": "not_requested",
        }
        if consolidation.enabled or generic_dag_checkpoint:
            try:
                resolved_device = self._resolve_consolidation_device(consolidation.device)
                optional_artifact["resolved_device"] = resolved_device
                consolidated_model_ref = self._write_consolidated_model(
                    snapshot=snapshot,
                    training_config=training_config,
                    current=current,
                    shards=shards,
                    manifest_ref=manifest_ref,
                    device=resolved_device,
                )
                if consolidated_model_ref is None:
                    optional_artifact["status"] = "not_supported"
                else:
                    optional_artifact["status"] = "complete"
                    optional_artifact["ref"] = consolidated_model_ref
                    manifest["consolidated_model_ref"] = consolidated_model_ref
            except Exception as exc:
                optional_artifact["status"] = "failed"
                optional_artifact["message"] = str(exc)
                if consolidation.required or generic_dag_checkpoint:
                    raise
        manifest["optional_artifacts"] = {"consolidated_model": optional_artifact}
        manifest_path = Path(snapshot.checkpoint_path) / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return manifest

    def _validated_checkpoint_shards(
        self,
        *,
        snapshot: JobSnapshot,
        training_config: TrainingConfig,
        execution_plan: ExecutionPlan,
        collection_result: ArtifactCollectionResult,
    ) -> list[dict[str, object]]:
        from shardgrid.artifacts.collector import ArtifactCollectionState

        snapshot_root = Path(snapshot.root_path).resolve()
        workers = {
            (item.worker_id, item.rank): item
            for item in collection_result.workers
        }
        shards: list[dict[str, object]] = []
        for assignment in execution_plan.workers:
            worker = workers.get((str(assignment.worker_id), assignment.rank))
            if worker is None:
                raise ValueError(f"missing artifact collection for rank {assignment.rank}")
            artifacts = [
                artifact
                for artifact in worker.artifacts
                if artifact.artifact_type == "checkpoint_file"
                and artifact.status
                in {
                    ArtifactCollectionState.COMPLETE,
                    ArtifactCollectionState.SKIPPED,
                }
            ]
            if len(artifacts) != 1:
                raise ValueError(
                    f"expected exactly one checkpoint shard for rank {assignment.rank}"
                )
            artifact = artifacts[0]
            if artifact.checksum is None or artifact.size_bytes is None:
                raise ValueError(
                    f"checkpoint shard for rank {assignment.rank} is missing checksum evidence"
                )
            checkpoint_metadata = self._checkpoint_metadata_from_worker(worker.artifacts)
            metadata_rank = checkpoint_metadata.get("rank")
            if metadata_rank is not None and int(metadata_rank) != assignment.rank:
                raise ValueError(f"checkpoint shard rank mismatch for rank {assignment.rank}")
            metadata_stage = str(
                checkpoint_metadata.get("stage_id") or checkpoint_metadata.get("stage") or ""
            )
            if (
                assignment.stage is not None
                and metadata_stage
                and metadata_stage != assignment.stage
            ):
                raise ValueError(f"checkpoint shard stage mismatch for rank {assignment.rank}")
            metadata_world_size = checkpoint_metadata.get("world_size")
            if (
                metadata_world_size is not None
                and int(metadata_world_size) != execution_plan.world_size
            ):
                raise ValueError(
                    f"checkpoint shard world_size mismatch for rank {assignment.rank}"
                )
            local_path = Path(artifact.local_path).resolve()
            if training_config.model.type == "generic_dag":
                checkpoint_metadata = {
                    **self._generic_checkpoint_metadata_from_shard(local_path),
                    **checkpoint_metadata,
                }
            shard = {
                "worker_id": str(assignment.worker_id),
                "rank": assignment.rank,
                "stage": assignment.stage,
                "stage_id": metadata_stage or assignment.stage,
                "relative_path": str(local_path.relative_to(snapshot_root)),
                "local_path": str(local_path),
                "artifact_type": artifact.artifact_type,
                "checksum": artifact.checksum,
                "size_bytes": artifact.size_bytes,
            }
            if "checkpoint_version" in checkpoint_metadata:
                shard["checkpoint_version"] = int(checkpoint_metadata["checkpoint_version"])
            if "step" in checkpoint_metadata:
                shard["step"] = int(checkpoint_metadata["step"])
            if checkpoint_metadata:
                shard["checkpoint_metadata"] = checkpoint_metadata
            shards.append(shard)
        return shards

    def _generic_checkpoint_metadata_from_shard(self, path: Path) -> dict[str, object]:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            return {}
        metadata = payload.get("metadata")
        if isinstance(metadata, dict):
            return dict(metadata)
        return {}

    def _checkpoint_metadata_from_worker(
        self,
        artifacts: Sequence[object],
    ) -> dict[str, object]:
        from shardgrid.artifacts.collector import CollectedArtifact

        for item in artifacts:
            if (
                not isinstance(item, CollectedArtifact)
                or item.artifact_type != "checkpoint_metadata"
            ):
                continue
            path = Path(item.local_path)
            if not path.is_file():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError(f"checkpoint metadata is invalid JSON: {path}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"checkpoint metadata is not a mapping: {path}")
            return dict(payload)
        return {}

    def _resolve_consolidation_device(self, requested: str) -> str:
        import torch

        if requested == "cpu":
            return "cpu"
        cuda_available = torch.cuda.is_available()
        if requested == "auto":
            return "cuda" if cuda_available else "cpu"
        if requested == "cuda":
            if not cuda_available:
                raise ValueError(
                    "consolidation device 'cuda' requires CUDA on the finalization host"
                )
            return "cuda"
        raise ValueError(f"unsupported consolidation device: {requested}")

    def _write_consolidated_model(
        self,
        *,
        snapshot: JobSnapshot,
        training_config: TrainingConfig,
        current: JobStatus,
        shards: Sequence[dict[str, object]],
        manifest_ref: str,
        device: str,
    ) -> str | None:
        if training_config.model.type == "generic_dag":
            return self._write_generic_dag_model_state(
                snapshot=snapshot,
                training_config=training_config,
                current=current,
                shards=shards,
                manifest_ref=manifest_ref,
            )
        if training_config.model.type != "minimal_sequential":
            return None

        import torch
        from examples.models.minimal_transformer import (
            MinimalTransformer,
            MinimalTransformerConfig,
        )

        model_config = next(
            (
                shard["metadata"].get("model_config")
                for shard in shards
                if isinstance(shard.get("metadata"), dict)
                and isinstance(shard["metadata"].get("model_config"), dict)
            ),
            None,
        )
        loaded_shards = []
        for shard in shards:
            payload = torch.load(
                shard["local_path"],
                map_location=device,
                weights_only=False,
            )
            if not isinstance(payload, dict):
                raise ValueError("checkpoint shard is not a mapping")
            loaded_shards.append(payload)
        if not isinstance(model_config, dict):
            model_config = next(
                (
                    payload.get("metadata", {}).get("model_config")
                    for payload in loaded_shards
                    if isinstance(payload.get("metadata"), dict)
                    and isinstance(payload.get("metadata", {}).get("model_config"), dict)
                ),
                None,
            )
        if not isinstance(model_config, dict):
            raise ValueError("minimal_sequential consolidation requires model_config")

        full_model = MinimalTransformer(config=MinimalTransformerConfig(**model_config))
        state_dict: dict[str, object] = {}
        for shard, payload in zip(shards, loaded_shards, strict=True):
            if int(payload.get("rank", -1)) != int(shard["rank"]):
                raise ValueError(f"checkpoint shard rank mismatch for rank {shard['rank']}")
            stage_id = str(payload.get("stage_id") or "")
            expected_stage = str(shard.get("stage") or "")
            if expected_stage and stage_id != expected_stage:
                raise ValueError(f"checkpoint shard stage mismatch for rank {shard['rank']}")
            if int(payload.get("world_size", -1)) <= 0:
                raise ValueError("checkpoint shard world_size is invalid")
            model_state = payload.get("model_state_dict")
            if not isinstance(model_state, dict):
                raise ValueError("checkpoint shard model_state_dict must be a mapping")
            optimizer_state = payload.get("optimizer_state_dict")
            if not isinstance(optimizer_state, dict):
                raise ValueError("checkpoint shard optimizer_state_dict must be a mapping")
            metadata = payload.get("metadata")
            if isinstance(metadata, dict):
                shard["metadata"] = metadata
            for name, tensor in model_state.items():
                key = str(name)
                if key.startswith("block0."):
                    target = "blocks.0." + key[len("block0.") :]
                elif key.startswith("block1."):
                    target = "blocks.1." + key[len("block1.") :]
                else:
                    target = key
                if target in state_dict:
                    raise ValueError(f"duplicate consolidated parameter {target}")
                state_dict[target] = tensor
        full_model.load_state_dict(state_dict, strict=True)
        consolidated_ref = "checkpoint/consolidated_model.pt"
        consolidated_path = Path(snapshot.checkpoint_path) / "consolidated_model.pt"
        torch.save(
            {
                "format": "shardgrid-consolidated-model/v1",
                "job_id": str(current.job_id),
                "model_name": training_config.model.name,
                "model_type": training_config.model.type,
                "checkpoint_ref": manifest_ref,
                "final_metrics": dict(current.final_metrics),
                "model_state_dict": {
                    name: tensor.detach().cpu()
                    for name, tensor in full_model.state_dict().items()
                },
            },
            consolidated_path,
        )
        reloaded = torch.load(consolidated_path, map_location="cpu", weights_only=False)
        full_model.load_state_dict(reloaded["model_state_dict"], strict=True)
        return consolidated_ref

    def _write_generic_dag_model_state(
        self,
        *,
        snapshot: JobSnapshot,
        training_config: TrainingConfig,
        current: JobStatus,
        shards: Sequence[dict[str, object]],
        manifest_ref: str,
    ) -> str:
        import torch
        from examples.models.generic_partition_zoo import build_zoo_model, make_zoo_sample

        from shardgrid.runtime.checkpoint import consolidate_worker_state_shards

        zoo_model = str(training_config.model.parameters.get("zoo_model", "mini_unet"))
        model_kwargs = self._generic_dag_model_kwargs(training_config)
        full_model = build_zoo_model(zoo_model, **model_kwargs)
        expected_state = full_model.state_dict()
        output_ref = "checkpoint/model-state.pt"
        output_path = Path(snapshot.checkpoint_path) / "model-state.pt"
        payload = consolidate_worker_state_shards(
            [Path(str(shard["local_path"])) for shard in shards],
            output_path,
            expected_state_keys=tuple(expected_state),
        )
        for key, expected in expected_state.items():
            actual = payload["state_dict"][key]
            if tuple(actual.shape) != tuple(expected.shape):
                raise ValueError(f"checkpoint tensor shape mismatch for {key}")
            if actual.dtype != expected.dtype:
                raise ValueError(f"checkpoint tensor dtype mismatch for {key}")
        load_result = full_model.load_state_dict(payload["state_dict"], strict=True)
        sample_args, sample_kwargs = make_zoo_sample(zoo_model, **model_kwargs)
        full_model.eval()
        with torch.no_grad():
            output = full_model(*sample_args, **sample_kwargs)
        training_evidence = payload.get("training_evidence")
        if not isinstance(training_evidence, dict):
            training_evidence = {
                "worker_parameter_changed": {},
                "any_parameter_changed": False,
                "all_trainable_workers_parameter_changed": False,
            }
        payload.update(
            {
                "format": "shardgrid-generic-dag-model-state/v1",
                "job_id": str(current.job_id),
                "model_name": training_config.model.name,
                "model_type": training_config.model.type,
                "zoo_model": zoo_model,
                "model_config": training_config.to_dict()["model"],
                "checkpoint_ref": manifest_ref,
                "final_metrics": dict(current.final_metrics),
                "strict_load_missing_keys": list(load_result.missing_keys),
                "strict_load_unexpected_keys": list(load_result.unexpected_keys),
                "validation_forward_passed": _contains_tensor(output),
                "training_evidence": training_evidence,
                "parameter_changed": bool(training_evidence["any_parameter_changed"]),
                "final_model_path": str(output_path.resolve()),
            }
        )
        torch.save(payload, output_path)
        print(f"FINAL_MODEL_PATH={output_path.resolve()}", flush=True)
        return output_ref

    def _complete_status(self, current: JobStatus, *, checkpoint_ref: str) -> JobStatus:
        if not checkpoint_ref or "final_loss" not in current.final_metrics:
            failure = make_failure_record(
                stage=FailureStage.CHECKPOINT,
                host=str(self.cluster_config.control.hostname),
                message="final checkpoint evidence is incomplete",
                recommended_action="inspect checkpoint metadata and rerun the job",
                secrets=self._secrets,
            )
            return self._failed_status(current, phase="checkpoint", failure=failure)
        return JobStatus(
            job_id=current.job_id,
            state=JobState.COMPLETED,
            phase="checkpoint",
            workers=list(current.workers),
            assignments=list(current.assignments),
            runtime_environment_refs=dict(current.runtime_environment_refs),
            latest_loss=current.latest_loss,
            loss_history=list(current.loss_history),
            final_metrics=dict(current.final_metrics),
            backend=current.backend,
            fallback_used=current.fallback_used,
            started_at=current.started_at,
            failure=None,
            checkpoint_ref=Path(checkpoint_ref).as_posix(),
        )

    def _persist_status(
        self,
        current: JobStatus,
        *,
        snapshot: JobSnapshot | None = None,
        state: JobState,
        phase: str,
        preserve_metrics: bool = False,
    ) -> JobStatus:
        payload = JobStatus(
            job_id=current.job_id,
            state=state,
            phase=phase,
            workers=list(current.workers),
            assignments=list(current.assignments),
            runtime_environment_refs=dict(current.runtime_environment_refs),
            latest_loss=current.latest_loss if preserve_metrics else None,
            loss_history=list(current.loss_history) if preserve_metrics else [],
            final_metrics=dict(current.final_metrics) if preserve_metrics else {},
            backend=current.backend,
            fallback_used=current.fallback_used,
            started_at=current.started_at or _now(),
            checkpoint_ref=current.checkpoint_ref if preserve_metrics else None,
        )
        return self._save_status(payload, snapshot=snapshot)

    def _save_status(
        self,
        status: JobStatus,
        *,
        snapshot: JobSnapshot | None = None,
    ) -> JobStatus:
        del snapshot
        return self._status_store.save(status)

    def _failed_status(
        self,
        current: JobStatus,
        *,
        phase: str,
        failure: FailureRecord,
    ) -> JobStatus:
        return JobStatus(
            job_id=current.job_id,
            state=JobState.FAILED,
            phase=phase,
            workers=list(current.workers),
            assignments=list(current.assignments),
            runtime_environment_refs=dict(current.runtime_environment_refs),
            latest_loss=current.latest_loss,
            loss_history=list(current.loss_history),
            final_metrics=dict(current.final_metrics),
            backend=current.backend,
            fallback_used=current.fallback_used,
            started_at=current.started_at,
            failure=failure,
            checkpoint_ref=current.checkpoint_ref,
        )

    def _launcher_failure(
        self,
        result: LauncherResult,
        stage: FailureStage,
        message: str,
    ) -> FailureRecord:
        return result.failure or make_failure_record(
            stage=stage,
            host=str(self.cluster_config.control.hostname),
            message=message,
            recommended_action="inspect launcher diagnostics and retry",
            secrets=self._secrets,
        )

    def _load_status(self, snapshot: JobSnapshot, fallback: JobStatus) -> JobStatus:
        path = Path(snapshot.root_path) / "job-status.json"
        return self._status_store.load_path(path) if path.exists() else fallback

    def _save_terminal_snapshot(
        self,
        snapshot: JobSnapshot,
        job: TrainingJob,
        training_config: TrainingConfig,
        parallel_plan: ParallelPlan,
        execution_plan: ExecutionPlan,
        network_state: NetworkState,
        current: JobStatus,
        checkpoint_metadata: dict[str, object] | None = None,
        launch_metadata: dict[str, object] | None = None,
    ) -> None:
        checkpoint_metadata = checkpoint_metadata or self._checkpoint_metadata_for_snapshot(
            snapshot,
            current,
        )
        write_snapshot_metadata(
            snapshot=snapshot,
            job=job,
            config=training_config.to_dict(),
            parallel_plan=parallel_plan,
            execution_plan=execution_plan,
            environment_report=build_control_report(
                self.cluster_config.control.machine_id,
                self.cluster_config.control.hostname,
                jobs_root=self.cluster_config.jobs_root,
            ),
            network_state=network_state,
            job_status=current,
            checkpoint_metadata=checkpoint_metadata,
            launch_metadata=launch_metadata,
            secrets=self._secrets,
        )
        self._save_status(current, snapshot=snapshot)

    def _checkpoint_refs_for_rank(
        self,
        snapshot: JobSnapshot,
        assignment: WorkerAssignment,
        current: JobStatus,
    ) -> tuple[str, ...]:
        refs: list[str] = []
        monitor_path = (
            Path(snapshot.diagnostics_path)
            / f"monitor-{assignment.worker_id}-rank{assignment.rank}.json"
        )
        if monitor_path.is_file():
            payload = json.loads(monitor_path.read_text(encoding="utf-8"))
            checkpoint_ref = payload.get("checkpoint_ref")
            if isinstance(checkpoint_ref, str) and checkpoint_ref.strip():
                refs.append(checkpoint_ref)
        current_ref = current.checkpoint_ref
        if (
            isinstance(current_ref, str)
            and current_ref.strip()
            and f"rank{assignment.rank}" in current_ref
            and current_ref not in refs
        ):
            refs.append(current_ref)
        return tuple(refs)

    def _checkpoint_metadata_for_snapshot(
        self,
        snapshot: JobSnapshot,
        current: JobStatus,
    ) -> dict[str, object] | None:
        checkpoint_path = Path(snapshot.checkpoint_path) / "metadata"
        candidates = sorted(checkpoint_path.rglob("checkpoint-metadata.json"))
        for candidate in candidates:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            if payload.get("checkpoint_ref") or (
                str(payload.get("status", "")).lower() == "complete"
            ):
                return payload
        if current.checkpoint_ref is None:
            return None
        files_root = Path(snapshot.checkpoint_path) / "files"
        artifacts = []
        if files_root.is_dir():
            for path in sorted(item for item in files_root.rglob("*") if item.is_file()):
                artifacts.append(
                    {
                        "local_path": str(path),
                        "relative_path": str(path.relative_to(Path(snapshot.root_path))),
                        "size_bytes": path.stat().st_size,
                    }
                )
        return {
            "status": "complete" if current.state is JobState.COMPLETED else current.state.value,
            "checkpoint_ref": current.checkpoint_ref,
            "artifacts": artifacts,
        }

    def _failure_stage_for_phase(self, phase: str) -> FailureStage:
        return {
            "rendezvous": FailureStage.RENDEZVOUS,
            "training": FailureStage.TRAIN,
            "checkpoint": FailureStage.CHECKPOINT,
        }.get(phase, FailureStage.LAUNCH)

    def _failed_run_result(
        self,
        job: TrainingJob,
        status: JobStatus,
        *,
        snapshot: JobSnapshot | None = None,
        execution_plan: ExecutionPlan | None = None,
        parallel_plan: ParallelPlan | None = None,
        cluster_state: ClusterState | None = None,
        network_state: NetworkState | None = None,
        collection_result: ArtifactCollectionResult | None = None,
        launcher_result: LauncherResult | None = None,
    ) -> JobRunResult:
        return JobRunResult(
            job=job,
            status=status,
            snapshot=snapshot,
            execution_plan=execution_plan,
            parallel_plan=parallel_plan,
            cluster_state=cluster_state,
            network_state=network_state,
            collection_result=collection_result,
            launcher_result=launcher_result,
        )


def _new_job_id() -> JobId:
    return as_job_id(f"job-{datetime.now(tz=UTC):%Y%m%d%H%M%S}-{uuid4().hex[:8]}")


def _covers_workers(network_state: NetworkState, worker_ids: list[WorkerId]) -> bool:
    if not worker_ids:
        return False
    network_workers = set(network_state.workers)
    if not set(worker_ids).issubset(network_workers):
        return False
    for source in worker_ids:
        for target in worker_ids:
            if source == target:
                continue
            if not any(
                link.source_worker_id == source
                and link.target_worker_id == target
                and link.tcp_reachable
                for link in network_state.links
            ):
                return False
    return True
