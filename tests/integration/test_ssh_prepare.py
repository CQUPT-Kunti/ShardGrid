from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from shardgrid.common.config import (
    BackendPreferenceConfig,
    ClusterConfig,
    ControlNodeConfig,
    ManualOverrideConfig,
    NetworkConfig,
    RuntimeConfig,
    SSHConfig,
    WorkerConfig,
)
from shardgrid.common.enums import Health, JobState, PhysicalOS, RuntimeOS
from shardgrid.common.models import (
    as_backend_name,
    as_engine_name,
    as_hostname,
    as_job_id,
    as_machine_id,
    as_worker_id,
)
from shardgrid.common.process import ProcessResult
from shardgrid.control.resource_manager import ResourceManager
from shardgrid.jobs.models import JobSnapshot, JobStatus, TrainingJob
from shardgrid.launchers.base import LauncherContext, LauncherResultStatus
from shardgrid.launchers.ssh import SSHLauncher
from shardgrid.planner.models import ExecutionPlan, MasterMetadata, WorkerAssignment
from shardgrid.resources.models import NetworkLink, NetworkState, WorkerResource


class FakeRuntime:
    def __init__(
        self,
        *,
        run_results: list[ProcessResult] | None = None,
        script_results: list[ProcessResult] | None = None,
        run_error: Exception | None = None,
        script_error: Exception | None = None,
    ) -> None:
        self.run_results = list(run_results or [])
        self.script_results = list(script_results or [])
        self.run_error = run_error
        self.script_error = script_error
        self.run_calls: list[object] = []
        self.script_calls: list[str] = []
        self.script_kwargs: list[dict[str, object]] = []

    def run(self, command, **kwargs) -> ProcessResult:
        self.run_calls.append(command)
        if self.run_error is not None:
            raise self.run_error
        if not self.run_results:
            raise AssertionError("unexpected runtime.run call")
        return self.run_results.pop(0)

    def run_script(self, script: str, **kwargs) -> ProcessResult:
        self.script_calls.append(script)
        self.script_kwargs.append(dict(kwargs))
        if self.script_error is not None:
            raise self.script_error
        if not self.script_results:
            raise AssertionError("unexpected runtime.run_script call")
        return self.script_results.pop(0)


def _result(
    command: str,
    *,
    stdout: str = "",
    stderr: str = "",
    exit_code: int = 0,
) -> ProcessResult:
    return ProcessResult(
        args=command,
        recorded_command=command,
        shell=False,
        cwd=None,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        timed_out=False,
        runtime_environment={"via": "ssh-wsl"},
    )


def _cluster_config(worker_count: int = 3) -> ClusterConfig:
    workers = [
        WorkerConfig(
            worker_id=as_worker_id("gpu4060"),
            machine_id=as_machine_id("machine-c"),
            physical_os=PhysicalOS.WINDOWS,
            runtime_os=RuntimeOS.WSL2_LINUX,
            runtime="wsl2",
            host=as_hostname("10.87.5.155"),
            ssh_user="shardgrid",
            runtime_distro="Ubuntu-22.04",
            conda_environment="shardgrid",
            conda_prefix="/home/shardgrid/miniconda3/envs/shardgrid",
        ),
        WorkerConfig(
            worker_id=as_worker_id("gpu1060"),
            machine_id=as_machine_id("machine-d"),
            physical_os=PhysicalOS.WINDOWS,
            runtime_os=RuntimeOS.WSL2_LINUX,
            runtime="wsl2",
            host=as_hostname("10.87.5.15"),
            ssh_user="shardgrid",
            runtime_distro="Ubuntu-22.04",
            conda_environment="shardgrid",
            conda_prefix="/home/shardgrid/miniconda3/envs/shardgrid",
        ),
        WorkerConfig(
            worker_id=as_worker_id("gpu3090"),
            machine_id=as_machine_id("machine-e"),
            physical_os=PhysicalOS.WINDOWS,
            runtime_os=RuntimeOS.WSL2_LINUX,
            runtime="wsl2",
            host=as_hostname("10.87.5.99"),
            ssh_user="shardgrid",
            runtime_distro="Ubuntu-22.04",
            conda_environment="shardgrid",
            conda_prefix="/home/shardgrid/miniconda3/envs/shardgrid",
        ),
    ]
    return ClusterConfig(
        control=ControlNodeConfig(
            machine_id=as_machine_id("machine-a"),
            hostname=as_hostname("control"),
        ),
        jobs_root=Path("/var/tmp/shardgrid/jobs"),
        ssh=SSHConfig(private_key_path="/home/test/.ssh/id_ed25519"),
        runtime=RuntimeConfig(
            python_executable="python3",
            conda_environment="shardgrid",
            conda_prefix="/home/shardgrid/miniconda3/envs/shardgrid",
            default_wsl_distro="Ubuntu-22.04",
        ),
        network=NetworkConfig(),
        backend_preference=BackendPreferenceConfig(
            launcher=as_backend_name("ssh"),
            communication_backend=as_backend_name("nccl"),
            parallel_engine=as_engine_name("torchrun"),
        ),
        manual_override=ManualOverrideConfig(),
        workers=workers[:worker_count],
    )


def _context(
    tmp_path: Path,
    worker_ids: tuple[str, ...] = ("gpu4060", "gpu1060"),
) -> LauncherContext:
    snapshot_root = tmp_path / "job-0094"
    for rel in (
        "code",
        "config",
        "plan",
        "logs",
        "checkpoint",
        "environment",
        "diagnostics",
    ):
        (snapshot_root / rel).mkdir(parents=True, exist_ok=True)
    (snapshot_root / "code" / "train.py").write_text("print('ok')\n")
    config = _cluster_config(3)
    selected_configs = [worker for worker in config.workers if str(worker.worker_id) in worker_ids]
    resources = [
        WorkerResource(
            worker_id=worker.worker_id,
            hostname=as_hostname(str(worker.host)),
            physical_os=worker.physical_os,
            runtime_os=worker.runtime_os,
            conda_environment=worker.conda_environment,
            conda_prefix=worker.conda_prefix,
            python_executable=f"{worker.conda_prefix}/bin/python",
            ip=str(worker.host),
            gpu_name="gpu",
            torch_version="2.7.1+cu118",
            health=Health.HEALTHY,
            last_probe_at="2026-08-27T11:00:00+00:00",
        )
        for worker in selected_configs
    ]
    network = NetworkState(
        network_id="lan",
        workers=[resource.worker_id for resource in resources],
        links=[
            NetworkLink(
                source_worker_id=source.worker_id,
                target_worker_id=target.worker_id,
                source_ip=source.ip or "",
                target_ip=target.ip or "",
                interface="eth0",
                tcp_reachable=True,
                measured_at="2026-08-27T11:00:00+00:00",
            )
            for source in resources
            for target in resources
            if source.worker_id != target.worker_id
        ],
        created_at="2026-08-27T11:00:00+00:00",
    )
    cluster_state = ResourceManager().build_cluster_state(
        resources,
        network_state=network,
        require_network=len(resources) > 1,
        now=datetime(2026, 8, 27, 11, 0, tzinfo=UTC),
    )
    job = TrainingJob(
        job_id=as_job_id("job-0094"),
        config_path="examples/train-minimal.yaml",
        model="tiny",
        requested_world_size=len(worker_ids),
        backend_preference=as_backend_name("ssh"),
        runtime_environment_ref="env:cluster/shardgrid",
    )
    plan = ExecutionPlan(
        job_id=job.job_id,
        engine=as_engine_name("torchrun"),
        backend=as_backend_name("ssh"),
        world_size=len(worker_ids),
        master=MasterMetadata(address="10.87.5.155", port=29500),
        workers=[
            WorkerAssignment(
                worker_id=as_worker_id(worker_id),
                rank=index,
                stage=str(index),
                launch_command=f"python train.py --rank {index}",
                log_path=f"jobs/job-0094/logs/rank{index}.log",
            )
            for index, worker_id in enumerate(worker_ids)
        ],
        snapshot_ref=str(snapshot_root),
    )
    snapshot = JobSnapshot(
        job_id=job.job_id,
        root_path=str(snapshot_root),
        code_path=str(snapshot_root / "code"),
        config_path=str(snapshot_root / "config"),
        plan_path=str(snapshot_root / "plan"),
        logs_path=str(snapshot_root / "logs"),
        environment_path=str(snapshot_root / "environment"),
        checkpoint_path=str(snapshot_root / "checkpoint"),
        diagnostics_path=str(snapshot_root / "diagnostics"),
    )
    return LauncherContext(
        job=job,
        execution_plan=plan,
        cluster_state=cluster_state,
        snapshot=snapshot,
        job_status=JobStatus(job_id=job.job_id, state=JobState.CREATED, phase="created"),
        runtime_environment_refs={
            str(index): f"env:{worker_id}/shardgrid"
            for index, worker_id in enumerate(worker_ids)
        },
    )


def _probe_payload(
    *,
    python_executable: str = "/home/shardgrid/miniconda3/envs/shardgrid/bin/python",
    python_under_expected_prefix: bool = True,
    snapshot_present: bool = False,
    metadata_job_id: str | None = None,
    entrypoint_exists: bool = True,
    created_dirs: list[str] | None = None,
    output_dirs_ready: bool = True,
) -> str:
    import json

    return json.dumps(
        {
            "python_executable": python_executable,
            "python_version": "Python 3.12.13",
            "python_under_expected_prefix": python_under_expected_prefix,
            "remote_root": "/var/tmp/shardgrid/jobs/job-0094",
            "snapshot_present": snapshot_present,
            "metadata_job_id": metadata_job_id,
            "entrypoint_exists": entrypoint_exists,
            "created_dirs": created_dirs or [],
            "output_dirs_ready": output_dirs_ready,
        }
    )


def test_all_workers_prepare_pass_and_rerun_is_idempotent(tmp_path: Path) -> None:
    context = _context(tmp_path, ("gpu4060", "gpu1060", "gpu3090"))
    runtimes = {
        worker_id: FakeRuntime(
            script_results=[
                _result("probe", stdout=_probe_payload(created_dirs=["logs"])),
                _result("probe", stdout=_probe_payload(created_dirs=[])),
            ],
        )
        for worker_id in ("gpu4060", "gpu1060", "gpu3090")
    }
    launcher = SSHLauncher(
        _cluster_config(3),
        runtime_factory=lambda worker: runtimes[str(worker.worker_id)],
    )

    first = launcher.prepare(context)
    second = launcher.prepare(context)

    assert first.status is LauncherResultStatus.SUCCESS
    assert second.status is LauncherResultStatus.SUCCESS
    assert "snapshot pending distribution" in first.worker_results[0].message
    assert (
        "already prepared" in second.worker_results[0].message
        or "output paths already prepared" in second.worker_results[0].message
    )
    for runtime in runtimes.values():
        assert runtime.run_calls == []
        assert len(runtime.script_calls) == 2


def test_prepare_partial_failure_keeps_successful_worker_evidence(tmp_path: Path) -> None:
    context = _context(tmp_path)
    runtimes = {
        "gpu4060": FakeRuntime(
            script_results=[_result("probe", stdout=_probe_payload(created_dirs=["logs"]))],
        ),
        "gpu1060": FakeRuntime(
            script_results=[_result("probe", stderr="Permission denied", exit_code=1)],
        ),
    }
    launcher = SSHLauncher(
        _cluster_config(2),
        runtime_factory=lambda worker: runtimes[str(worker.worker_id)],
    )

    result = launcher.prepare(context)

    assert result.status is LauncherResultStatus.PARTIAL
    results = {item.worker_id: item for item in result.worker_results}
    assert results["gpu4060"].status is LauncherResultStatus.SUCCESS
    assert results["gpu1060"].failure is not None
    assert runtimes["gpu4060"].run_calls == []
    assert runtimes["gpu1060"].run_calls == []


@pytest.mark.parametrize(
    ("script_error", "script_result", "expected", "needle"),
    [
        (
            RuntimeError("worker unreachable"),
            None,
            LauncherResultStatus.FAILED,
            "runtime wrapper failure during prepare probe",
        ),
        (
            None,
            _result("probe", stderr="WSL not installed", exit_code=1),
            LauncherResultStatus.FAILED,
            "remote prepare path probe failed",
        ),
        (
            None,
            _result("probe", stderr="conda missing", exit_code=127),
            LauncherResultStatus.FAILED,
            "remote prepare path probe failed",
        ),
        (
            None,
            _result("probe", stderr="Permission denied", exit_code=1),
            LauncherResultStatus.BLOCKED,
            "remote prepare path probe failed",
        ),
        (
            None,
            _result("probe", stdout=_probe_payload(python_under_expected_prefix=False)),
            LauncherResultStatus.FAILED,
            "selected runtime Python mismatch",
        ),
        (
            None,
            _result(
                "probe",
                stdout=_probe_payload(
                    snapshot_present=True,
                    metadata_job_id="wrong-job",
                ),
            ),
            LauncherResultStatus.FAILED,
            "different job identity",
        ),
        (
            None,
            _result(
                "probe",
                stdout=_probe_payload(
                    snapshot_present=True,
                    metadata_job_id="job-0094",
                    entrypoint_exists=False,
                ),
            ),
            LauncherResultStatus.FAILED,
            "missing the training entry point",
        ),
        (
            None,
            _result("probe", stdout=_probe_payload(output_dirs_ready=False)),
            LauncherResultStatus.FAILED,
            "output directories",
        ),
    ],
)
def test_prepare_failure_modes_are_structured(
    tmp_path: Path,
    script_error: Exception | None,
    script_result: ProcessResult | None,
    expected: LauncherResultStatus,
    needle: str,
) -> None:
    context = _context(tmp_path, ("gpu4060",))
    launcher = SSHLauncher(
        _cluster_config(1),
        runtime_factory=lambda worker: FakeRuntime(
            script_results=[] if script_result is None else [script_result],
            script_error=script_error,
        ),
    )

    result = launcher.prepare(context)

    assert result.status is expected
    assert result.failure is not None
    assert needle in result.failure.message


def test_prepare_uses_one_script_and_configured_command_timeout(tmp_path: Path) -> None:
    context = _context(tmp_path, ("gpu4060",))
    runtime = FakeRuntime(
        script_results=[_result("probe", stdout=_probe_payload(created_dirs=["logs"]))]
    )
    config = _cluster_config(1)
    launcher = SSHLauncher(
        replace(config, ssh=replace(config.ssh, command_timeout_seconds=123)),
        runtime_factory=lambda worker: runtime,
    )

    result = launcher.prepare(context)

    assert result.status is LauncherResultStatus.SUCCESS
    assert runtime.run_calls == []
    assert len(runtime.script_calls) == 1
    assert runtime.script_kwargs == [{"timeout": 123.0}]
    assert "sys.executable" in runtime.script_calls[0]
    assert "expected_conda_prefix" in runtime.script_calls[0]


def test_prepare_blocks_when_entrypoint_missing_in_snapshot(tmp_path: Path) -> None:
    context = _context(tmp_path, ("gpu4060",))
    (Path(context.snapshot.code_path) / "train.py").unlink()
    launcher = SSHLauncher(
        _cluster_config(1),
        runtime_factory=lambda worker: FakeRuntime(),
    )

    result = launcher.prepare(context)

    assert result.status is LauncherResultStatus.BLOCKED
    assert result.failure is not None
    assert "snapshot entry point is missing" in result.failure.message


def test_prepare_requires_runtime_refs_for_selected_worker(tmp_path: Path) -> None:
    context = replace(_context(tmp_path, ("gpu4060",)), runtime_environment_refs={})
    launcher = SSHLauncher(
        _cluster_config(1),
        runtime_factory=lambda worker: FakeRuntime(),
    )

    result = launcher.prepare(context)

    assert result.status is LauncherResultStatus.BLOCKED
    assert result.failure is not None
    assert "runtime environment reference is missing" in result.failure.message
