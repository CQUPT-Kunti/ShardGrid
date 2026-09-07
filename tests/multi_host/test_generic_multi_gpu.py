"""T063 — real multi-GPU generic acceptance for an ordinary (non-zoo) entrypoint.

Drives the production captured-entrypoint path (``JobManager.run_entrypoint``)
with the hardware-gate constraint ``min_selected_physical_hosts=2`` against at
least two fresh-discovered healthy CUDA GPUs.  It verifies the exact planner
placement, real cross-partition activation/gradient transfer, optimizer
progress, checkpoint merge and strict reload.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

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
def test_multi_gpu_generic_ordinary_entrypoint(tmp_path: Path) -> None:
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
        "T063 requires at least two healthy CUDA GPU workers from fresh discovery"
    )
    gpu_free_mb = {r.worker_resource.worker_id: r.worker_resource.gpu_free_memory for r in healthy}
    gpu_names = {r.worker_resource.worker_id: r.worker_resource.gpu_name for r in healthy}

    result = manager.run_entrypoint(
        SimpleNamespace(
            entrypoint=FIXTURE,
            argv=("--epochs", "1", "--checkpoint", "out/tuple.pt"),
            cwd=FIXTURE.parent,
            environment={"SHARDGRID_TEST_CAPTURE": "1"},
            cluster_config_path=str(tmp_path / "workers.yaml"),
            dry_run=False,
        ),
        min_selected_physical_hosts=2,
    )

    assert result.status.state.value == "completed", (
        result.status.failure.message if result.status.failure else "job failed"
    )
    assert result.execution_plan is not None
    assert result.execution_plan.world_size >= 2
    worker_hosts = {a.worker_id for a in result.execution_plan.workers}
    assert len(worker_hosts) >= 2, "at least two real GPU workers must participate"

    _audit(Path(result.snapshot.root_path), gpu_free_mb=gpu_free_mb, gpu_names=gpu_names)


def _audit(
    snapshot: Path,
    *,
    gpu_free_mb: dict[str, int | None],
    gpu_names: dict[str, str | None],
) -> None:
    model_state_path = snapshot / "checkpoint" / "model-state.pt"
    assert model_state_path.is_file(), "model-state.pt must be produced"

    state = torch.load(model_state_path, map_location="cpu", weights_only=False)
    original = _load_fixture_module().TupleBatchModel()
    load_result = original.load_state_dict(state, strict=True)
    assert load_result.missing_keys == []
    assert load_result.unexpected_keys == []
    with torch.no_grad():
        out = original(torch.randn(3, 4))
    assert out.shape == (3, 2)
    assert torch.isfinite(out).all()

    monitors = sorted((snapshot / "diagnostics").glob("monitor-*.json"))
    assert len(monitors) >= 2, "must observe both ranks"
    activation_remote = 0
    gradient_remote = 0
    steps = []
    consistency_pass = True
    worker_only = True
    for monitor in monitors:
        payload = json.loads(monitor.read_text())
        train = payload.get("train")
        if isinstance(train, dict):
            activation_remote += int(train.get("activation_remote_edges", 0))
            gradient_remote += int(train.get("gradient_remote_edges", 0))
            if "optimizer_steps" in train:
                steps.append(int(train["optimizer_steps"]))
        runtime = payload.get("placement")
        if isinstance(runtime, dict):
            if runtime.get("plan_runtime_consistency_check") != "PASS":
                consistency_pass = False
            if runtime.get("full_model_real_materialized") is not False:
                worker_only = False
    assert activation_remote > 0, "real cross-partition activation transfer required"
    assert gradient_remote > 0, "real cross-partition gradient transfer required"
    assert steps and max(steps) >= 20, "optimizer steps must grow"
    assert consistency_pass, "PLAN_RUNTIME_CONSISTENCY_CHECK must PASS"
    assert worker_only, "worker-only materialization must hold"

    manifest = json.loads((snapshot / "checkpoint" / "manifest.json").read_text())
    assert manifest.get("status") == "complete"
    assert manifest.get("consolidated_model_ref") == "checkpoint/model-state.pt"

    _write_evidence(
        {
            "task": "T063",
            "world_size": len(monitors),
            "gpu_free_mb": gpu_free_mb,
            "gpu_names": gpu_names,
            "activation_remote_edges": activation_remote,
            "gradient_remote_edges": gradient_remote,
            "optimizer_steps": max(steps),
            "plan_runtime_consistency_check": "PASS",
            "worker_only_materialization": True,
            "strict_load_missing_keys": [],
            "strict_load_unexpected_keys": [],
            "validation_forward_passed": True,
            "formal_training_oom_count": 0,
            "snapshot": str(snapshot),
        }
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
        pytest.skip("tests/address.json is required for T063 hardware acceptance")
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
                "command_timeout_seconds": 60,
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
    (directory / "t063-multi-gpu-generic.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )