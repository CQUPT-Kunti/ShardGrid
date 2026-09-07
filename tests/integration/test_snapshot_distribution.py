from __future__ import annotations

import json
import os
import shutil
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from shardgrid.artifacts.metadata import write_snapshot_metadata
from shardgrid.artifacts.snapshot import create_code_snapshot
from shardgrid.artifacts.ssh_transport import (
    DistributionStatus,
    RemoteSnapshotProbe,
    RemoteSnapshotProbeError,
    distribute_job_snapshot,
    snapshot_checksum,
)
from shardgrid.artifacts.store import ArtifactStore
from shardgrid.artifacts.transport import (
    ArtifactTransferItemResult,
    ArtifactTransferResult,
    ArtifactTransferStatus,
    ArtifactTransportName,
)
from shardgrid.common.config import ClusterConfig, WorkerConfig, load_cluster_config
from shardgrid.common.enums import Health, JobState, PhysicalOS, RuntimeOS
from shardgrid.common.models import (
    as_backend_name,
    as_engine_name,
    as_hostname,
    as_job_id,
    as_machine_id,
    as_worker_id,
)
from shardgrid.control.job_manager import create_training_job
from shardgrid.engines.models import ParallelPlan
from shardgrid.jobs.models import JobStatus
from shardgrid.network.state import build_network_state
from shardgrid.planner.models import ExecutionPlan, MasterMetadata, WorkerAssignment
from shardgrid.resources.models import NetworkLink
from shardgrid.workers.environment_report import EnvironmentReport, ReportScope

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "examples" / "workers.yaml"
ADDRESS_PATH = ROOT / "tests" / "address.json"


def _cluster_config(tmp_path: Path) -> ClusterConfig:
    config = load_cluster_config(CONFIG_PATH)
    payload = config.to_dict()
    payload["jobs_root"] = str((tmp_path / "jobs-root").resolve())
    payload["ssh"]["strict_host_key_checking"] = False
    payload["ssh"]["known_hosts_path"] = None
    return ClusterConfig.from_dict(payload)


def _workers(config: ClusterConfig) -> list[WorkerConfig]:
    return [
        next(worker for worker in config.workers if str(worker.worker_id) == "gpu4060"),
        next(worker for worker in config.workers if str(worker.worker_id) == "gpu1060"),
    ]


def _job_snapshot(tmp_path: Path):
    config = _cluster_config(tmp_path)
    store = ArtifactStore(config.jobs_root)
    job = create_training_job(
        job_id=as_job_id("job-001"),
        config_path="examples/train-minimal.yaml",
        model="tiny-sequential",
        requested_world_size=2,
        backend_preference=as_backend_name("ssh"),
        runtime_environment_ref="env:worker-pair/shardgrid",
    )
    snapshot = store.create_snapshot(job)
    create_code_snapshot(snapshot, source_root=ROOT)
    write_snapshot_metadata(
        snapshot=snapshot,
        job=job,
        config={
            "job": {"name": "train-minimal", "backend": "ssh"},
            "artifacts": {"transport": "auto"},
        },
        parallel_plan=ParallelPlan(
            parallel_plan_id="plan-1",
            engine=as_engine_name("galvatron"),
            engine_plan_path="/var/tmp/original-plan.json",
            model_name="tiny-sequential",
            world_size=2,
            stages=["stage0", "stage1"],
        ),
        execution_plan=ExecutionPlan(
            job_id=job.job_id,
            engine=as_engine_name("galvatron"),
            backend=as_backend_name("ssh"),
            world_size=2,
            master=MasterMetadata(address="10.0.0.1", port=29500),
            workers=[
                WorkerAssignment(worker_id=as_worker_id("gpu4060"), rank=0, stage="0"),
                WorkerAssignment(worker_id=as_worker_id("gpu1060"), rank=1, stage="1"),
            ],
            conda_environment="shardgrid",
            conda_prefix="/home/shardgrid/miniconda3/envs/shardgrid",
            python_executable="/home/shardgrid/miniconda3/envs/shardgrid/bin/python",
            snapshot_ref=snapshot.root_path,
        ),
        environment_report=EnvironmentReport(
            report_id="worker-gpu4060",
            scope=ReportScope.WORKER,
            target="gpu4060",
            machine_id=as_machine_id("machine-c"),
            hostname=as_hostname("machine-c.local"),
            physical_os=PhysicalOS.WINDOWS,
            runtime_os=RuntimeOS.WSL2_LINUX,
            timestamp="2026-08-27T00:00:00+00:00",
            health=Health.HEALTHY,
            conda_executable="/home/shardgrid/miniconda3/bin/conda",
            conda_environment="shardgrid",
            conda_prefix="/home/shardgrid/miniconda3/envs/shardgrid",
            python_executable="/home/shardgrid/miniconda3/envs/shardgrid/bin/python",
            python_version="3.12.13",
            torch_version="2.7.1+cu118",
            torch_cuda_version="11.8",
            cuda_version="11.8",
            gpu_name="NVIDIA GeForce RTX 4060 Laptop GPU",
            components={"nccl": "2.21.5", "backend": "nccl"},
            evidence_status="live",
        ),
        network_state=build_network_state(
            [
                NetworkLink(
                    source_worker_id=as_worker_id("gpu4060"),
                    target_worker_id=as_worker_id("gpu1060"),
                    source_ip="10.0.0.1",
                    target_ip="10.0.0.2",
                    interface="eth0",
                    tcp_reachable=True,
                ),
                NetworkLink(
                    source_worker_id=as_worker_id("gpu1060"),
                    target_worker_id=as_worker_id("gpu4060"),
                    source_ip="10.0.0.2",
                    target_ip="10.0.0.1",
                    interface="eth0",
                    tcp_reachable=True,
                ),
            ],
            network_id="lan-a",
            diagnostics_path="/var/tmp/shardgrid/network/latest.json",
        ),
        job_status=JobStatus(
            job_id=job.job_id,
            state=JobState.COMPLETED,
            phase="checkpoint",
            workers=[as_worker_id("gpu4060"), as_worker_id("gpu1060")],
            backend=as_backend_name("ssh"),
            checkpoint_ref="jobs/job-001/checkpoint/model.pt",
            final_metrics={"final_loss": 0.5},
            finished_at="2026-08-27T00:10:00+00:00",
        ),
        checkpoint_metadata={"step": 1, "status": "complete"},
    )
    return config, snapshot


class _FakeTransport:
    name = ArtifactTransportName.SCP

    def __init__(self) -> None:
        self.calls = 0

    def transfer(self, items, *, remote, secrets=()):
        self.calls += 1
        return ArtifactTransferResult(
            transport="scp",
            status=ArtifactTransferStatus.SUCCESS,
            items=[
                ArtifactTransferItemResult(
                    label=items[0].label,
                    transport="scp",
                    status=ArtifactTransferStatus.SUCCESS,
                    source=items[0].source,
                    destination=items[0].destination,
                    recorded_command="scp ***",
                    exit_code=0,
                )
            ],
        )


def test_distribution_verifies_both_workers_and_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, snapshot = _job_snapshot(tmp_path)
    fake = _FakeTransport()
    workers = _workers(config)
    state: dict[str, list[RemoteSnapshotProbe]] = {
        str(workers[0].host): [
            RemoteSnapshotProbe(exists=False, empty=False),
            RemoteSnapshotProbe(
                exists=True,
                empty=False,
                checksum="same",
                job_id="job-001",
            ),
            RemoteSnapshotProbe(
                exists=True,
                empty=False,
                checksum="same",
                job_id="job-001",
            ),
        ],
        str(workers[1].host): [
            RemoteSnapshotProbe(exists=False, empty=False),
            RemoteSnapshotProbe(
                exists=True,
                empty=False,
                checksum="same",
                job_id="job-001",
            ),
            RemoteSnapshotProbe(
                exists=True,
                empty=False,
                checksum="same",
                job_id="job-001",
            ),
        ],
    }

    control_checksum = None
    expected_checksum = snapshot_checksum(Path(snapshot.root_path))

    def fake_probe(runtime, remote_root: str) -> RemoteSnapshotProbe:
        host = str(runtime.executor.options.host)
        probe = state[host].pop(0)
        if probe.checksum == "same":
            return replace(probe, checksum=expected_checksum)
        return probe

    monkeypatch.setattr(
        "shardgrid.artifacts.ssh_transport._select_worker_transport",
        lambda **_: fake,
    )
    monkeypatch.setattr(
        "shardgrid.artifacts.ssh_transport._read_windows_userprofile",
        lambda ssh: r"C:\Users\shardgrid",
    )
    monkeypatch.setattr(
        "shardgrid.artifacts.ssh_transport._prepare_windows_staging_dir",
        lambda ssh, staging_dir: None,
    )
    monkeypatch.setattr(
        "shardgrid.artifacts.ssh_transport._extract_remote_snapshot",
        lambda runtime, archive_wsl_path, remote_root: type("R", (), {"ok": True})(),
    )
    monkeypatch.setattr("shardgrid.artifacts.ssh_transport._probe_remote_snapshot", fake_probe)

    first = distribute_job_snapshot(snapshot, cluster_config=config, workers=workers)
    control_checksum = first.control_checksum
    assert first.status is DistributionStatus.PASS
    assert [worker.status for worker in first.workers] == [
        DistributionStatus.PASS,
        DistributionStatus.PASS,
    ]
    assert fake.calls == 2

    second = distribute_job_snapshot(snapshot, cluster_config=config, workers=workers)
    assert second.status is DistributionStatus.PASS
    assert all(worker.skipped for worker in second.workers)
    assert control_checksum == expected_checksum


def test_snapshot_checksum_ignores_mutable_launcher_outputs(tmp_path: Path) -> None:
    _, snapshot = _job_snapshot(tmp_path)
    root = Path(snapshot.root_path)
    baseline = snapshot_checksum(root)

    (root / "logs" / "gpu4060").mkdir(parents=True, exist_ok=True)
    (root / "logs" / "gpu4060" / "rank0.log").write_text("rank output\n", encoding="utf-8")
    (root / "diagnostics" / "launch-gpu4060.json").write_text("{}", encoding="utf-8")
    (root / "checkpoint" / "model.pt").write_text("checkpoint\n", encoding="utf-8")
    (root / "code" / "examples" / "__pycache__").mkdir(parents=True, exist_ok=True)
    (root / "code" / "examples" / "__pycache__" / "train.cpython-312.pyc").write_bytes(b"pyc")

    assert snapshot_checksum(root) == baseline


def test_distribution_fails_on_checksum_conflict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, snapshot = _job_snapshot(tmp_path)
    fake = _FakeTransport()
    workers = _workers(config)
    monkeypatch.setattr(
        "shardgrid.artifacts.ssh_transport._select_worker_transport",
        lambda **_: fake,
    )
    monkeypatch.setattr(
        "shardgrid.artifacts.ssh_transport._probe_remote_snapshot",
        lambda runtime, remote_root: RemoteSnapshotProbe(
            exists=True,
            empty=False,
            checksum="wrong",
            job_id="job-001",
        ),
    )

    result = distribute_job_snapshot(snapshot, cluster_config=config, workers=workers[:1])

    assert result.status is DistributionStatus.FAIL
    assert result.workers[0].failure is not None


def test_distribution_allows_prepare_only_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, snapshot = _job_snapshot(tmp_path)
    fake = _FakeTransport()
    workers = _workers(config)
    expected_checksum = snapshot_checksum(Path(snapshot.root_path))
    state: dict[str, list[RemoteSnapshotProbe]] = {
        str(worker.host): [
            RemoteSnapshotProbe(
                exists=True,
                empty=False,
                top_level_entries=("checkpoint", "diagnostics", "logs"),
            ),
            RemoteSnapshotProbe(
                exists=True,
                empty=False,
                checksum=expected_checksum,
                job_id="job-001",
            ),
        ]
        for worker in workers
    }

    def fake_probe(runtime, remote_root: str) -> RemoteSnapshotProbe:
        host = str(runtime.executor.options.host)
        return state[host].pop(0)

    monkeypatch.setattr(
        "shardgrid.artifacts.ssh_transport._select_worker_transport",
        lambda **_: fake,
    )
    monkeypatch.setattr(
        "shardgrid.artifacts.ssh_transport._read_windows_userprofile",
        lambda ssh: r"C:\Users\shardgrid",
    )
    monkeypatch.setattr(
        "shardgrid.artifacts.ssh_transport._prepare_windows_staging_dir",
        lambda ssh, staging_dir: None,
    )
    monkeypatch.setattr(
        "shardgrid.artifacts.ssh_transport._extract_remote_snapshot",
        lambda runtime, archive_wsl_path, remote_root: type("R", (), {"ok": True})(),
    )
    monkeypatch.setattr("shardgrid.artifacts.ssh_transport._probe_remote_snapshot", fake_probe)

    result = distribute_job_snapshot(snapshot, cluster_config=config, workers=workers)

    assert result.status is DistributionStatus.PASS
    assert all(worker.status is DistributionStatus.PASS for worker in result.workers)


def test_distribution_rejects_unknown_nonempty_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, snapshot = _job_snapshot(tmp_path)
    fake = _FakeTransport()
    worker = _workers(config)[0]
    monkeypatch.setattr(
        "shardgrid.artifacts.ssh_transport._select_worker_transport",
        lambda **_: fake,
    )
    monkeypatch.setattr(
        "shardgrid.artifacts.ssh_transport._probe_remote_snapshot",
        lambda runtime, remote_root: RemoteSnapshotProbe(
            exists=True,
            empty=False,
            top_level_entries=("checkpoint", "diagnostics", "logs", "mystery"),
        ),
    )

    result = distribute_job_snapshot(snapshot, cluster_config=config, workers=[worker])

    assert result.status is DistributionStatus.FAIL
    assert result.workers[0].failure is not None
    assert result.workers[0].details is not None
    assert result.workers[0].details["substep"] == "preflight_identity_conflict"
    assert result.workers[0].details["probe_state"] == "PRESENT"


def test_distribution_preserves_partial_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, snapshot = _job_snapshot(tmp_path)
    workers = _workers(config)

    class _PartialTransport(_FakeTransport):
        def transfer(self, items, *, remote, secrets=()):
            self.calls += 1
            status = (
                ArtifactTransferStatus.SUCCESS
                if remote.host == str(workers[0].host)
                else ArtifactTransferStatus.FAILED
            )
            return ArtifactTransferResult(
                transport="scp",
                status=status,
                items=[
                    ArtifactTransferItemResult(
                        label=items[0].label,
                        transport="scp",
                        status=status,
                        source=items[0].source,
                        destination=items[0].destination,
                        recorded_command="scp ***",
                        exit_code=0 if status is ArtifactTransferStatus.SUCCESS else 1,
                        stderr=(
                            ""
                            if status is ArtifactTransferStatus.SUCCESS
                            else "Permission denied"
                        ),
                        retryable=False,
                    )
                ],
            )

    fake = _PartialTransport()
    expected_checksum = snapshot_checksum(Path(snapshot.root_path))

    def fake_probe(runtime, remote_root: str) -> RemoteSnapshotProbe:
        host = str(runtime.executor.options.host)
        if host == str(workers[0].host):
            return RemoteSnapshotProbe(
                exists=True,
                empty=False,
                checksum=expected_checksum,
                job_id="job-001",
            )
        return RemoteSnapshotProbe(exists=False, empty=False)

    monkeypatch.setattr(
        "shardgrid.artifacts.ssh_transport._select_worker_transport",
        lambda **_: fake,
    )
    monkeypatch.setattr(
        "shardgrid.artifacts.ssh_transport._read_windows_userprofile",
        lambda ssh: r"C:\Users\shardgrid",
    )
    monkeypatch.setattr(
        "shardgrid.artifacts.ssh_transport._prepare_windows_staging_dir",
        lambda ssh, staging_dir: None,
    )
    monkeypatch.setattr(
        "shardgrid.artifacts.ssh_transport._extract_remote_snapshot",
        lambda runtime, archive_wsl_path, remote_root: type("R", (), {"ok": True})(),
    )
    monkeypatch.setattr("shardgrid.artifacts.ssh_transport._probe_remote_snapshot", fake_probe)

    result = distribute_job_snapshot(snapshot, cluster_config=config, workers=workers)

    assert result.status is DistributionStatus.BLOCKED
    assert result.workers[0].status is DistributionStatus.PASS
    assert result.workers[1].status is DistributionStatus.BLOCKED


def test_distribution_reports_first_failed_probe_substep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, snapshot = _job_snapshot(tmp_path)
    fake = _FakeTransport()
    worker = _workers(config)[0]
    monkeypatch.setattr(
        "shardgrid.artifacts.ssh_transport._select_worker_transport",
        lambda **_: fake,
    )
    monkeypatch.setattr(
        "shardgrid.artifacts.ssh_transport._probe_remote_snapshot",
        lambda runtime, remote_root: (_ for _ in ()).throw(
            RemoteSnapshotProbeError(
                "failed to execute remote snapshot probe",
                {
                    "worker_id": "gpu4060",
                    "stage": "DISTRIBUTE",
                    "action": "remote_snapshot",
                    "substep": "probe_command",
                    "remote_path": remote_root,
                    "command_summary": "python -",
                    "exit_code": 1,
                    "stdout_summary": None,
                    "stderr_summary": "permission denied",
                    "parse_error": None,
                    "expected_job_id": "job-001",
                    "detected_job_id": None,
                    "expected_checksum": "expected",
                    "detected_checksum": None,
                    "metadata_ready": False,
                    "recommended_action": "inspect SSH/WSL runtime command execution and retry",
                },
            )
        ),
    )

    result = distribute_job_snapshot(snapshot, cluster_config=config, workers=[worker])

    assert result.status is DistributionStatus.BLOCKED
    details = result.workers[0].details
    assert details is not None
    assert details["substep"] == "probe_command"
    assert details["remote_path"].endswith("/job-001")
    assert details["stderr_summary"] == "permission denied"


def test_distribution_reports_partial_snapshot_probe_details(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, snapshot = _job_snapshot(tmp_path)
    fake = _FakeTransport()
    worker = _workers(config)[0]
    monkeypatch.setattr(
        "shardgrid.artifacts.ssh_transport._select_worker_transport",
        lambda **_: fake,
    )
    monkeypatch.setattr(
        "shardgrid.artifacts.ssh_transport._read_windows_userprofile",
        lambda ssh: r"C:\Users\shardgrid",
    )
    monkeypatch.setattr(
        "shardgrid.artifacts.ssh_transport._prepare_windows_staging_dir",
        lambda ssh, staging_dir: None,
    )
    monkeypatch.setattr(
        "shardgrid.artifacts.ssh_transport._extract_remote_snapshot",
        lambda runtime, archive_wsl_path, remote_root: type("R", (), {"ok": True})(),
    )
    expected_checksum = snapshot_checksum(Path(snapshot.root_path))
    probes = iter(
        (
            RemoteSnapshotProbe(exists=False, empty=False),
            RemoteSnapshotProbe(
                exists=True,
                empty=False,
                checksum="partial-checksum",
                job_id=None,
                missing_paths=("diagnostics/snapshot-metadata.json",),
                metadata_ready=False,
            ),
        )
    )
    monkeypatch.setattr(
        "shardgrid.artifacts.ssh_transport._probe_remote_snapshot",
        lambda runtime, remote_root: next(probes),
    )

    result = distribute_job_snapshot(snapshot, cluster_config=config, workers=[worker])

    assert result.status is DistributionStatus.FAIL
    details = result.workers[0].details
    assert details is not None
    assert details["substep"] == "verify_post_transfer"
    assert details["expected_checksum"] == expected_checksum
    assert details["detected_checksum"] == "partial-checksum"
    assert details["probe_state"] == "PARTIAL"
    assert details["missing_paths"] == ["diagnostics/snapshot-metadata.json"]


def _require_password() -> str:
    password = os.environ.get("SHARDGRID_TEST_SSH_PASSWORD")
    if not password:
        pytest.skip("set SHARDGRID_TEST_SSH_PASSWORD to run live snapshot distribution")
    return password


def _live_workers(config: ClusterConfig) -> list[WorkerConfig]:
    address_book = json.loads(ADDRESS_PATH.read_text(encoding="utf-8"))
    mapping = {
        "gpu4060": next(
            entry for entry in address_book if "RTX 4060" in str(entry.get("gpu_model") or "")
        ),
        "gpu1060": next(
            entry for entry in address_book if "GTX 1650" in str(entry.get("gpu_model") or "")
        ),
    }
    return [
        replace(
            worker,
            host=mapping[str(worker.worker_id)]["ip"],
            ssh_user=mapping[str(worker.worker_id)]["username"],
        )
        for worker in _workers(config)
    ]


def _write_wrapper(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


@pytest.mark.multi_host
def test_live_snapshot_distribution_to_two_workers(tmp_path: Path) -> None:
    password = _require_password()
    if not shutil.which("sshpass"):
        pytest.skip("sshpass is required for live snapshot distribution")

    config, snapshot = _job_snapshot(tmp_path)
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["jobs_root"] = str(config.jobs_root)
    payload["ssh"]["strict_host_key_checking"] = False
    payload["ssh"]["known_hosts_path"] = None
    live_workers = _live_workers(config)
    worker_map = {str(worker.worker_id): worker for worker in live_workers}
    for worker_payload in payload["workers"]:
        worker = worker_map.get(str(worker_payload["id"]))
        if worker is None:
            continue
        worker_payload["host"] = str(worker.host)
        worker_payload["ssh_user"] = worker.ssh_user
    live_config = ClusterConfig.from_dict(payload)

    wrapper_dir = tmp_path / "bin"
    wrapper_dir.mkdir()
    _write_wrapper(
        wrapper_dir / "ssh",
        """#!/usr/bin/env bash
set -euo pipefail
args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -o)
      if [[ $# -ge 2 && "$2" == "BatchMode=yes" ]]; then
        shift 2
        continue
      fi
      args+=("$1" "$2")
      shift 2
      ;;
    -oBatchMode=yes)
      shift
      ;;
    *)
      args+=("$1")
      shift
      ;;
  esac
done
exec /usr/bin/sshpass -p "$SHARDGRID_TEST_SSH_PASSWORD" /usr/bin/ssh "${args[@]}"
""",
    )
    _write_wrapper(
        wrapper_dir / "scp",
        """#!/usr/bin/env bash
set -euo pipefail
exec /usr/bin/sshpass -p "$SHARDGRID_TEST_SSH_PASSWORD" /usr/bin/scp "$@"
""",
    )

    env_path = os.environ["PATH"]
    os.environ["PATH"] = f"{wrapper_dir}:{env_path}"
    os.environ["SHARDGRID_TEST_SSH_PASSWORD"] = password
    try:
        result = distribute_job_snapshot(
            snapshot,
            cluster_config=live_config,
            workers=_live_workers(live_config),
            secrets=[password],
        )
        repeated = distribute_job_snapshot(
            snapshot,
            cluster_config=live_config,
            workers=_live_workers(live_config),
            secrets=[password],
        )
    finally:
        os.environ["PATH"] = env_path

    assert result.status is DistributionStatus.PASS
    assert all(worker.status is DistributionStatus.PASS for worker in result.workers)
    assert all(worker.metadata_ready is True for worker in result.workers)
    assert all(worker.remote_job_id == "job-001" for worker in result.workers)
    assert all(worker.remote_checksum == result.control_checksum for worker in result.workers)
    assert repeated.status is DistributionStatus.PASS
    assert all(worker.skipped is True for worker in repeated.workers)
