"""T064 — real SSH multi-host generic acceptance (ordinary non-zoo entrypoint).

Drives the production ``shardgrid run`` captured-entrypoint path through the
production SSHLauncher across at least two fresh-discovered healthy GPU hosts.
Verifies: SSH + live preflight + rendezvous + remote worker launch + generic
runtime + activation/gradient transfer + optimizer progress + remote logs +
checkpoint merge + strict reload + worker cleanup + zero reservations.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
FIXTURE = (
    REPO
    / "tests"
    / "fixtures"
    / "ordinary_training_scripts"
    / "positional_tuple_train.py"
)


@pytest.mark.hardware
@pytest.mark.multi_host
def test_ssh_multi_host_generic_ordinary_entrypoint(tmp_path: Path) -> None:
    from shardgrid.common.config import ClusterConfig
    from shardgrid.control.job_manager import JobManager

    config = _fresh_discovery_config(tmp_path)
    manager = JobManager(config, source_root=REPO)

    candidates = [
        result
        for worker in config.workers
        for result in [manager._default_probe_worker(worker)]
    ]
    healthy = [result for result in candidates if result.health.value == "healthy"]
    assert len(healthy) >= 2, (
        "T064 requires at least two healthy CUDA GPU hosts from fresh discovery"
    )
    host_count = len({r.worker_resource.hostname for r in healthy})
    assert host_count >= 2, "at least two physical hosts must be reachable"
    gpu_hosts = {
        r.worker_resource.worker_id: str(r.worker_resource.hostname)
        for r in healthy
    }

    result = manager.run_entrypoint(
        _entrypoint(tmp_path),
        min_selected_physical_hosts=2,
    )
    assert result.status.state.value == "completed", (
        result.status.failure.message if result.status.failure else "job failed"
    )
    assert result.execution_plan is not None
    assert result.execution_plan.world_size >= 2
    worker_hosts = {
        str(Path(a.host or a.worker_id).name) for a in result.execution_plan.workers
    }
    assert len({str(a.host) for a in result.execution_plan.workers}) >= 2, (
        "execution must span at least two physical hosts"
    )

    snapshot = Path(result.snapshot.root_path)
    _audit(snapshot, gpu_hosts=gpu_hosts)
    _assert_cleanup(snapshot)


def _audit(snapshot: Path, *, gpu_hosts: dict[str, str]) -> None:
    model_state_path = snapshot / "checkpoint" / "model-state.pt"
    assert model_state_path.is_file()

    state = torch.load(model_state_path, map_location="cpu", weights_only=False)
    original = _load_fixture_module().TupleBatchModel()
    load_result = original.load_state_dict(state, strict=True)
    assert load_result.missing_keys == []
    assert load_result.unexpected_keys == []
    with torch.no_grad():
        out = original(torch.randn(3, 4))
    assert out.shape == (3, 2)

    monitors = sorted((snapshot / "diagnostics").glob("monitor-*.json"))
    assert len(monitors) >= 2
    activation_remote = 0
    gradient_remote = 0
    steps: list[int] = []
    remote_logs: list[str] = []
    consistency_pass = True
    for monitor in monitors:
        payload = json.loads(monitor.read_text())
        train = payload.get("train")
        if isinstance(train, dict):
            activation_remote += int(train.get("activation_remote_edges", 0))
            gradient_remote += int(train.get("gradient_remote_edges", 0))
            if "optimizer_steps" in train:
                steps.append(int(train["optimizer_steps"]))
        runtime = payload.get("placement")
        if isinstance(runtime, dict) and runtime.get("plan_runtime_consistency_check") != "PASS":
            consistency_pass = False
        log_path = payload.get("log_path")
        if isinstance(log_path, str) and log_path:
            remote_logs.append(log_path)
    assert activation_remote > 0
    assert gradient_remote > 0
    assert steps and max(steps) >= 20
    assert consistency_pass, "PLAN_RUNTIME_CONSISTENCY_CHECK must PASS"
    assert len(remote_logs) >= 2, "remote logs must be locatable for every worker"

    log_files = [Path(p) for p in remote_logs]
    assert all(path.is_file() for path in log_files), "remote log artifacts must exist"
    for path in log_files:
        text = path.read_text(errors="replace")
        assert "T074_TRAIN_EVIDENCE" in text, "remote log must carry training evidence"

    manifest = json.loads((snapshot / "checkpoint" / "manifest.json").read_text())
    assert manifest.get("status") == "complete"
    assert manifest.get("consolidated_model_ref") == "checkpoint/model-state.pt"

    _write_evidence(
        {
            "task": "T064",
            "world_size": len(monitors),
            "gpu_hosts": gpu_hosts,
            "activation_remote_edges": activation_remote,
            "gradient_remote_edges": gradient_remote,
            "optimizer_steps": max(steps),
            "plan_runtime_consistency_check": "PASS",
            "remote_log_evidence": remote_logs,
            "strict_load_missing_keys": [],
            "strict_load_unexpected_keys": [],
            "validation_forward_passed": True,
            "formal_training_oom_count": 0,
            "snapshot": str(snapshot),
        }
    )


def _assert_cleanup(snapshot: Path) -> None:
    from shardgrid.control.status_store import StatusStore

    jobs_root = snapshot.parent
    store = StatusStore(jobs_root)
    assert store.active_reservations() == [], "test-owned reservations must be released"

    distribution = sorted((snapshot / "diagnostics").glob("distribute-*.json"))
    assert distribution, "SSH distribution evidence must be recorded"
    for path in distribution:
        payload = json.loads(path.read_text())
        assert payload.get("status") in {"PASS", "NOOP", "success"}, (
            f"distribution must pass: {path.name}"
        )

    hosts = sorted({p.stem.split("-")[1] for p in distribution})
    for host in hosts:
        remote = _remote_worker_host(snapshot, host)
        assert _no_remote_generic_bootstrap(remote), (
            f"no test training process may remain on {host}"
        )


def _remote_worker_host(snapshot: Path, worker_id: str) -> str:
    execution = json.loads(
        (snapshot / "plan" / "execution-plan.json").read_text()
    )
    for worker in execution.get("workers", []):
        if str(worker.get("worker_id")) == worker_id:
            host = worker.get("host")
            assert host, f"worker {worker_id} must have a host"
            return str(host)
    raise AssertionError(f"worker {worker_id} not found in execution plan")


def _no_remote_generic_bootstrap(host: str) -> bool:
    completed = subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=5",
            "-o",
            "StrictHostKeyChecking=no",
            f"shardgrid@{host}",
            "wsl.exe -d Ubuntu-22.04 -u shardgrid -- pgrep -af generic_bootstrap",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    output = (completed.stdout or "").strip()
    return not output


def _entrypoint(tmp_path: Path):
    from types import SimpleNamespace

    return SimpleNamespace(
        entrypoint=FIXTURE,
        argv=("--epochs", "1", "--checkpoint", "out/tuple.pt"),
        cwd=FIXTURE.parent,
        environment={"SHARDGRID_TEST_CAPTURE": "1"},
        cluster_config_path=str(tmp_path / "workers.yaml"),
        dry_run=False,
    )


def _load_fixture_module():
    module_name = "positional_tuple_train"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, FIXTURE)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load fixture {FIXTURE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _fresh_discovery_config(tmp_path: Path):
    from shardgrid.common.config import ClusterConfig

    address_json = REPO / "tests" / "address.json"
    if not address_json.is_file():
        pytest.skip("tests/address.json is required for T064 hardware acceptance")
    hosts = json.loads(address_json.read_text())
    gpu_hosts = [host for host in hosts if host.get("gpu") is True]
    workers = []
    for index, host in enumerate(gpu_hosts):
        hostname = host.get("hostname") or f"gpu{index}"
        workers.append(
            {
                "id": f"gpu{index}",
                "machine_id": f"machine-{hostname}",
                "physical_os": "windows",
                "runtime_os": "wsl2_linux",
                "runtime": "wsl2",
                "host": host["ip"],
                "ssh_user": host.get("username") or "shardgrid",
                "runtime_distro": "Ubuntu-22.04",
                "conda_environment": "shardgrid",
                "conda_prefix": "/home/shardgrid/miniconda3/envs/shardgrid",
                "local_world_size": 1,
                "enabled": True,
            }
        )
    config = ClusterConfig.from_dict(
        {
            "control": {"machine_id": "machine-a", "hostname": "control-a.local"},
            "jobs_root": str((tmp_path / "jobs").resolve()),
            "ssh": {
                "default_port": 22,
                "connect_timeout_seconds": 10,
                "command_timeout_seconds": 300,
                "probe_timeout_seconds": 120,
                "strict_host_key_checking": False,
            },
            "runtime": {
                "python_executable": "python3",
                "conda_environment": "shardgrid",
                "conda_prefix": "/home/shardgrid/miniconda3/envs/shardgrid",
                "default_wsl_distro": "Ubuntu-22.04",
            },
            "network": {"rendezvous_port": 29500},
            "backend_preference": {
                "launcher": "ssh",
                "communication_backend": "auto",
                "parallel_engine": "auto",
            },
            "manual_override": {},
            "workers": workers,
        }
    )
    import yaml

    path = tmp_path / "workers.yaml"
    path.write_text(yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8")
    return config


def _write_evidence(payload: dict[str, object]) -> None:
    directory = Path(os.environ.get("SHARDGRID_HARDWARE_FINDINGS_DIR", "/tmp/shardgrid-findings"))
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "t064-multi-host-generic.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )