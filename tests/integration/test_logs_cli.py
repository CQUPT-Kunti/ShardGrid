from __future__ import annotations

import json
from pathlib import Path

from shardgrid.cli.app import main
from shardgrid.cli.commands import logs as logs_command
from shardgrid.common.enums import FailureStage, JobState
from shardgrid.common.errors import make_failure_record
from shardgrid.common.models import as_backend_name, as_engine_name, as_job_id
from shardgrid.common.process import ProcessResult
from shardgrid.jobs.models import JobStatus
from shardgrid.launchers.base import LauncherResultStatus
from shardgrid.launchers.ssh import SSHLauncher
from shardgrid.planner.models import ExecutionPlan, MasterMetadata, WorkerAssignment


class FakeRuntime:
    def __init__(
        self,
        *,
        run_results: list[ProcessResult] | None = None,
        run_error: Exception | None = None,
    ) -> None:
        self.run_results = list(run_results or [])
        self.run_error = run_error

    def run(self, command, **kwargs) -> ProcessResult:
        if self.run_error is not None:
            raise self.run_error
        if not self.run_results:
            raise AssertionError(f"unexpected runtime.run call: {command!r}")
        return self.run_results.pop(0)

    def run_script(self, script: str, **kwargs) -> ProcessResult:
        raise AssertionError("run_script should not be called by shardgrid logs")


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


def _config_path(tmp_path: Path) -> Path:
    path = tmp_path / "workers.yaml"
    path.write_text(
        f"""
control:
  machine_id: machine-a
  hostname: control-a.local
jobs_root: {tmp_path / "jobs"}
ssh: {{}}
runtime:
  conda_environment: shardgrid
  conda_prefix: /home/shardgrid/miniconda3/envs/shardgrid
  default_wsl_distro: Ubuntu-22.04
network: {{}}
backend_preference: {{}}
manual_override:
  preferred_workers: []
  disabled_workers: []
  worker_address_overrides: {{}}
  rendezvous_port: null
workers:
  - id: gpu4060
    machine_id: machine-c
    physical_os: windows
    runtime_os: wsl2_linux
    runtime: wsl2
    host: 10.87.5.155
    ssh_user: shardgrid
    runtime_distro: Ubuntu-22.04
    conda_environment: shardgrid
    conda_prefix: /home/shardgrid/miniconda3/envs/shardgrid
  - id: gpu1060
    machine_id: machine-d
    physical_os: windows
    runtime_os: wsl2_linux
    runtime: wsl2
    host: 10.87.5.15
    ssh_user: shardgrid
    runtime_distro: Ubuntu-22.04
    conda_environment: shardgrid
    conda_prefix: /home/shardgrid/miniconda3/envs/shardgrid
  - id: gpu3090
    machine_id: machine-e
    physical_os: windows
    runtime_os: wsl2_linux
    runtime: wsl2
    host: 10.87.5.99
    ssh_user: shardgrid
    runtime_distro: Ubuntu-22.04
    conda_environment: shardgrid
    conda_prefix: /home/shardgrid/miniconda3/envs/shardgrid
""".strip(),
        encoding="utf-8",
    )
    return path


def _job_paths(tmp_path: Path, job_id: str = "job-0098") -> Path:
    root = tmp_path / "jobs" / job_id
    for rel in ("code", "config", "plan", "logs", "checkpoint", "environment", "diagnostics"):
        (root / rel).mkdir(parents=True, exist_ok=True)
    return root


def _assignment(worker_id: str, rank: int, stage: str) -> WorkerAssignment:
    return WorkerAssignment(
        worker_id=worker_id,
        rank=rank,
        local_rank=0,
        stage=stage,
        gpu_index=0,
        launch_command=f"python examples/models/train_pipeline.py --rank {rank}",
        log_path=f"/var/tmp/shardgrid/jobs/job-0098/logs/{worker_id}/rank{rank}.log",
        status="running",
        pid=4100 + rank,
    )


def _write_snapshot_metadata(
    tmp_path: Path,
    *,
    job_id: str = "job-0098",
    state: JobState = JobState.TRAINING,
    assignments: list[WorkerAssignment] | None = None,
) -> Path:
    root = _job_paths(tmp_path, job_id)
    entries = assignments or [
        _assignment("gpu4060", 0, "stage0"),
        _assignment("gpu1060", 1, "stage1"),
        _assignment("gpu3090", 2, "stage2"),
    ]
    plan = ExecutionPlan(
        job_id=as_job_id(job_id),
        engine=as_engine_name("torchrun"),
        backend=as_backend_name("ssh"),
        world_size=len(entries),
        master=MasterMetadata(address="10.87.5.155", port=29500),
        workers=entries,
        snapshot_ref=str(root),
    )
    status = JobStatus(
        job_id=plan.job_id,
        state=state,
        phase=(
            "checkpoint"
            if state is JobState.COMPLETED
            else "training"
            if state in {JobState.TRAINING, JobState.FAILED}
            else "launch"
        ),
        workers=[item.worker_id for item in entries],
        assignments=entries,
        runtime_environment_refs={
            str(item.rank): f"env:{item.worker_id}/shardgrid" for item in entries
        },
        backend=as_backend_name("ssh"),
        loss_history=[1.0, 0.5] if state in {JobState.TRAINING, JobState.COMPLETED} else [],
        latest_loss=0.5 if state in {JobState.TRAINING, JobState.COMPLETED} else None,
        final_metrics={"final_loss": 0.25} if state is JobState.COMPLETED else {},
        checkpoint_ref="checkpoint/rank1.pt" if state is JobState.COMPLETED else None,
        failure=(
            make_failure_record(
                stage=FailureStage.TRAIN,
                host="10.87.5.15",
                worker_id="gpu1060",
                message="rank 1 exited",
                recommended_action="inspect rank logs",
            )
            if state is JobState.FAILED
            else None
        ),
    )
    (root / "plan" / "execution-plan.json").write_text(
        json.dumps(plan.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (root / "diagnostics" / "job-status.json").write_text(
        json.dumps(status.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return root


def _local_log(root: Path, worker_id: str, rank: int, stage: str, filename: str, text: str) -> None:
    path = root / "logs" / worker_id / f"rank{rank}-{stage}" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _install_launcher(monkeypatch, config_path: Path, runtimes: dict[str, FakeRuntime]) -> None:
    monkeypatch.setattr(
        logs_command,
        "_build_launcher",
        lambda cluster_config: SSHLauncher(
            cluster_config,
            runtime_factory=lambda worker: runtimes[str(worker.worker_id)],
            secrets=("secret-token", "/home/test/.ssh/id_secret"),
        ),
    )


def test_logs_is_registered_as_real_command(capsys) -> None:
    try:
        main(["--help"])
    except SystemExit as error:
        assert error.code == 0

    captured = capsys.readouterr()
    assert "logs" in captured.out
    assert "placeholder command" not in captured.out


def test_logs_requires_config(capsys) -> None:
    exit_code = main(["logs", "job-0098"])

    captured = capsys.readouterr()
    assert exit_code == 3
    assert "requires a cluster config" in captured.out


def test_logs_invalid_tail_is_usage_error(tmp_path: Path, capsys) -> None:
    config = _config_path(tmp_path)
    _write_snapshot_metadata(tmp_path)

    exit_code = main(["--config", str(config), "logs", "job-0098", "--tail", "0"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "--tail must be > 0" in captured.err


def test_logs_reads_local_collected_logs_and_keeps_stream_identity(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    config = _config_path(tmp_path)
    root = _write_snapshot_metadata(tmp_path, state=JobState.COMPLETED)
    _local_log(root, "gpu4060", 0, "stage0", "stdout.log", "a\nb\nc\n")
    _local_log(root, "gpu4060", 0, "stage0", "stderr.log", "err1\nsecret-token\n")
    _install_launcher(monkeypatch, config, {})

    exit_code = main(
        [
            "--config",
            str(config),
            "logs",
            "job-0098",
            "--worker",
            "gpu4060",
            "--rank",
            "0",
            "--tail",
            "2",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Worker: gpu4060" in captured.out
    assert "Rank: 0" in captured.out
    assert "Stage: stage0" in captured.out
    assert "Stream: stdout" in captured.out
    assert "Stream: stderr" in captured.out
    assert "Source: LOCAL" in captured.out
    assert "Source Location: local" in captured.out
    assert "Source Path:" in captured.out
    assert "b\nc" in captured.out
    assert "err1\n***" in captured.out
    assert "secret-token" not in captured.out


def test_logs_remote_fallback_for_running_job_and_tail_rank_selector(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    config = _config_path(tmp_path)
    _write_snapshot_metadata(tmp_path, state=JobState.TRAINING)
    _install_launcher(
        monkeypatch,
        config,
        {
            "gpu4060": FakeRuntime(),
            "gpu1060": FakeRuntime(
                run_results=[_result("tail", stdout="0\n1\n2\n3\n")]
            ),
            "gpu3090": FakeRuntime(),
        },
    )

    exit_code = main(["--config", str(config), "logs", "job-0098", "--rank", "1", "--tail", "2"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Worker: gpu1060" in captured.out
    assert "Rank: 1" in captured.out
    assert "Source: REMOTE_FALLBACK" in captured.out
    assert "Source Location: remote" in captured.out
    assert "2\n3" in captured.out


def test_logs_whole_job_human_output_keeps_stable_identity_per_block(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    config = _config_path(tmp_path)
    root = _write_snapshot_metadata(tmp_path, job_id="job-0103-whole", state=JobState.TRAINING)
    _local_log(root, "gpu4060", 0, "stage0", "stdout.log", "rank0-local\n")
    _install_launcher(
        monkeypatch,
        config,
        {
            "gpu4060": FakeRuntime(),
            "gpu1060": FakeRuntime(run_results=[_result("tail", stdout="rank1-remote\n")]),
            "gpu3090": FakeRuntime(
                run_results=[_result("tail", stderr="No such file or directory", exit_code=1)]
            ),
        },
    )

    exit_code = main(["--config", str(config), "logs", "job-0103-whole"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Job: job-0103-whole" in captured.out
    assert "Entries: 3" in captured.out
    assert "Overall Status: PARTIAL" in captured.out
    assert "Selector: whole-job" in captured.out
    assert "[worker=gpu4060 rank=0 stage=stage0 stream=stdout source=LOCAL]" in captured.out
    assert (
        "[worker=gpu1060 rank=1 stage=stage1 stream=combined "
        "source=REMOTE_FALLBACK]" in captured.out
    )
    assert "[worker=gpu3090 rank=2 stage=stage2 stream=combined source=MISSING]" in captured.out
    assert captured.out.index("worker=gpu4060") < captured.out.index("worker=gpu1060")
    assert captured.out.index("worker=gpu1060") < captured.out.index("worker=gpu3090")


def test_logs_local_is_preferred_when_local_and_remote_both_exist(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    config = _config_path(tmp_path)
    root = _write_snapshot_metadata(tmp_path, state=JobState.TRAINING)
    _local_log(root, "gpu1060", 1, "stage1", "stdout.log", "local-line\n")
    runtimes = {
        "gpu4060": FakeRuntime(),
        "gpu1060": FakeRuntime(run_results=[_result("tail", stdout="remote-line\n")]),
        "gpu3090": FakeRuntime(),
    }
    _install_launcher(monkeypatch, config, runtimes)

    exit_code = main(["--config", str(config), "logs", "job-0098", "--worker", "gpu1060"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Source: LOCAL" in captured.out
    assert "local-line" in captured.out
    assert runtimes["gpu1060"].run_results == [_result("tail", stdout="remote-line\n")]


def test_logs_json_supports_worker_rank_intersection_and_multiple_ranks(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    config = _config_path(tmp_path)
    root = _write_snapshot_metadata(tmp_path)
    _local_log(root, "gpu3090", 2, "stage2", "stdout.log", "x\ny\n")
    _install_launcher(monkeypatch, config, {})

    exit_code = main(
        [
            "--json",
            "--config",
            str(config),
            "logs",
            "job-0098",
            "--worker",
            "gpu3090",
            "--rank",
            "2",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["job_id"] == "job-0098"
    assert payload["selector"] == {"worker": "gpu3090", "rank": 2, "tail": 50}
    assert payload["status"] == "success"
    assert len(payload["logs"]) == 1
    item = payload["logs"][0]
    assert item["job_id"] == "job-0098"
    assert item["worker_id"] == "gpu3090"
    assert item["rank"] == 2
    assert item["stage"] == "stage2"
    assert item["stream"] == "stdout"
    assert item["source"] == "LOCAL"
    assert item["location"] == "local"
    assert item["source_path"].endswith("logs/gpu3090/rank2-stage2/stdout.log")


def test_logs_handles_completed_and_failed_jobs_without_worker_dependency(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    config = _config_path(tmp_path)
    root = _write_snapshot_metadata(tmp_path, state=JobState.FAILED)
    _local_log(root, "gpu1060", 1, "stage1", "stdout.log", "surviving\nlog\n")
    _install_launcher(monkeypatch, config, {})

    exit_code = main(["--config", str(config), "logs", "job-0098", "--rank", "1"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "surviving\nlog" in captured.out
    assert "Source: LOCAL" in captured.out


def test_logs_reads_stopped_job_history_from_local_snapshot(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    config = _config_path(tmp_path)
    root = _write_snapshot_metadata(tmp_path, job_id="job-0103-stopped", state=JobState.STOPPED)
    _local_log(root, "gpu1060", 1, "stage1", "stdout.log", "stopped-history\n")
    _install_launcher(monkeypatch, config, {})

    exit_code = main(["--config", str(config), "logs", "job-0103-stopped", "--rank", "1"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Job State: STOPPED" in captured.out
    assert "stopped-history" in captured.out
    assert "Source: LOCAL" in captured.out


def test_logs_partial_across_ranks_reports_missing_and_success(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    config = _config_path(tmp_path)
    root = _write_snapshot_metadata(tmp_path)
    _local_log(root, "gpu4060", 0, "stage0", "stdout.log", "rank0\n")
    _install_launcher(
        monkeypatch,
        config,
        {
            "gpu1060": FakeRuntime(run_results=[_result("tail", stdout="remote-r1\n")]),
            "gpu4060": FakeRuntime(),
            "gpu3090": FakeRuntime(
                run_results=[_result("tail", stderr="No such file or directory", exit_code=1)]
            ),
        },
    )

    exit_code = main(["--json", "--config", str(config), "logs", "job-0098"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["status"] == "partial"
    sources = {(item["worker_id"], item["source"]) for item in payload["logs"]}
    assert ("gpu4060", "LOCAL") in sources
    assert ("gpu1060", "REMOTE_FALLBACK") in sources
    assert ("gpu3090", "MISSING") in sources


def test_logs_nonexistent_worker_or_rank_returns_explicit_error(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    config = _config_path(tmp_path)
    _write_snapshot_metadata(tmp_path)
    _install_launcher(monkeypatch, config, {})

    exit_code = main(["--config", str(config), "logs", "job-0098", "--worker", "nope"])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "log selector not found" in captured.out

    exit_code = main(["--config", str(config), "logs", "job-0098", "--rank", "9"])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "log selector not found" in captured.out


def test_logs_missing_local_and_remote_failure_are_structured(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    config = _config_path(tmp_path)
    _write_snapshot_metadata(tmp_path, assignments=[_assignment("gpu1060", 1, "stage1")])
    _install_launcher(
        monkeypatch,
        config,
        {
            "gpu1060": FakeRuntime(
                run_results=[_result("tail", stderr="Permission denied", exit_code=1)]
            )
        },
    )

    exit_code = main(["--json", "--config", str(config), "logs", "job-0098"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert payload["status"] == "blocked"
    item = payload["logs"][0]
    assert item["source"] == "FAILED"
    assert item["location"] == "remote"
    assert item["failure"]["message"] == "remote log read failed"


def test_logs_remote_runtime_exception_is_redacted_and_failed(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    config = _config_path(tmp_path)
    _write_snapshot_metadata(tmp_path, assignments=[_assignment("gpu1060", 1, "stage1")])
    _install_launcher(
        monkeypatch,
        config,
        {"gpu1060": FakeRuntime(run_error=RuntimeError("secret-token /home/test/.ssh/id_secret"))},
    )

    exit_code = main(["--json", "--config", str(config), "logs", "job-0098"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    item = payload["logs"][0]
    assert item["status"] == LauncherResultStatus.FAILED.value
    assert "secret-token" not in json.dumps(item)
    assert "/home/test/.ssh/id_secret" not in json.dumps(item)
