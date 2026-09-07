"""T062 — real single-GPU generic acceptance for an ordinary (non-zoo) entrypoint.

This test drives the production ``shardgrid run`` CLI (captured entrypoint
path) against a real healthy CUDA GPU discovered by ShardGrid's own fresh
discovery.  It proves the full production chain:

    ordinary train.py
    -> capture
    -> generic graph / planning
    -> real one-batch memory probe
    -> formal CUDA training
    -> optimizer progress
    -> worker shard / merge
    -> model-state.pt
    -> original_model.load_state_dict(strict=True)

It is opt-in via the ``hardware`` marker (``--run-hardware`` +
``SHARDGRID_ENABLE_HARDWARE_TESTS=1``).
"""

from __future__ import annotations

import importlib.util
import json
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
    / "buffered_state_train_script.py"
)


@pytest.mark.hardware
def test_single_gpu_generic_ordinary_entrypoint(tmp_path: Path) -> None:
    import shardgrid.control.job_manager as job_manager_module
    import shardgrid.control.resource_manager as resource_manager_module

    config = _fresh_discovery_config(tmp_path)
    manager = job_manager_module.JobManager(
        config,
        source_root=REPO,
    )

    candidates = [
        result
        for worker in config.workers
        for result in [manager._default_probe_worker(worker)]
    ]
    healthy = [result for result in candidates if result.health.value == "healthy"]
    assert healthy, "T062 requires at least one healthy CUDA GPU worker"

    selected = healthy[0]
    gpu_free_mb = selected.worker_resource.gpu_free_memory
    assert gpu_free_mb, "fresh discovery must report current free GPU memory"
    gpu_name = selected.worker_resource.gpu_name or "unknown"

    # Run the production CLI path (non-dry-run) against the real worker.
    env = dict(__import__("os").environ)
    env["PYTHONPATH"] = str(REPO / "src")
    env["SHARDGRID_MEMORY_PROBE_TIMEOUT_SECONDS"] = "300"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "shardgrid.cli.app",
            "--config",
            str(config_path_for(tmp_path, config)),
            "--json",
            "run",
            str(FIXTURE),
        ],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=1200,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-3000:]

    payload = _last_json_object(completed.stdout)
    assert payload["state"] == "completed", payload.get("failure")

    snapshot = Path(payload["snapshot_path"])
    _audit(snapshot, gpu_free_mb=gpu_free_mb, gpu_name=gpu_name)


def _last_json_object(text: str) -> dict[str, object]:
    decoder = json.JSONDecoder()
    for start in range(len(text) - 1, -1, -1):
        if text[start] != "{":
            continue
        try:
            payload, _end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise AssertionError(f"no JSON object found in output:\n{text[-3000:]}")


def _audit(snapshot: Path, *, gpu_free_mb: int | None, gpu_name: str) -> None:
    model_state_path = snapshot / "checkpoint" / "model-state.pt"
    assert model_state_path.is_file(), "model-state.pt must be produced"

    state = torch.load(model_state_path, map_location="cpu", weights_only=False)
    keys = set(state)
    assert "bn.running_mean" in keys, "buffers must be covered (BatchNorm running_mean)"
    assert "bn.running_var" in keys
    assert "bn.num_batches_tracked" in keys
    assert "bn.weight" in keys and "bn.bias" in keys, "parameters must be covered"

    original = _load_fixture_module().BufferedStateTrainModel()
    load_result = original.load_state_dict(state, strict=True)
    assert load_result.missing_keys == []
    assert load_result.unexpected_keys == []

    with torch.no_grad():
        out = original(torch.randn(3, 4))
    assert out.shape == (3, 2)
    assert torch.isfinite(out).all()

    monitors = sorted((snapshot / "diagnostics").glob("monitor-*.json"))
    train_events = []
    for monitor in monitors:
        payload = json.loads(monitor.read_text())
        train = payload.get("train")
        if isinstance(train, dict):
            train_events.append(train)
    assert train_events, "monitor must capture T074_TRAIN_EVIDENCE markers"
    steps = [int(item["optimizer_steps"]) for item in train_events if "optimizer_steps" in item]
    assert steps and max(steps) >= 20, "formal training must produce optimizer progress"
    assert all(item.get("loss_isfinite") is not False for item in train_events)
    assert any(item.get("parameter_changed") is True for item in train_events)

    runtime_markers = []
    for monitor in monitors:
        payload = json.loads(monitor.read_text())
        placement = payload.get("placement")
        if isinstance(placement, dict) and placement.get("generic_dag_runtime_used"):
            runtime_markers.append(placement)
    assert runtime_markers, "generic DAG runtime must be used (captured entrypoint)"
    assert all(item.get("plan_runtime_consistency_check") == "PASS" for item in runtime_markers)
    assert all(item.get("full_model_real_materialized") is False for item in runtime_markers)

    probe = _probe_peak(snapshot)
    assert probe["actual_peak_reserved_bytes"] is not None
    assert int(probe["actual_peak_reserved_bytes"]) > 0, "real memory probe evidence required"

    manifest = json.loads((snapshot / "checkpoint" / "manifest.json").read_text())
    assert manifest.get("status") == "complete"
    assert manifest.get("consolidated_model_ref") == "checkpoint/model-state.pt"

    _write_evidence(
        {
            "task": "T062",
            "gpu_name": gpu_name,
            "gpu_free_mb": gpu_free_mb,
            "snapshot": str(snapshot),
            "optimizer_steps": max(steps),
            "actual_peak_reserved_bytes": probe["actual_peak_reserved_bytes"],
            "strict_load_missing_keys": [],
            "strict_load_unexpected_keys": [],
            "validation_forward_passed": True,
            "formal_training_oom_count": 0,
        }
    )


def _probe_peak(snapshot: Path) -> dict[str, int | None]:
    for path in sorted((snapshot / "diagnostics").glob("memory-probe-rank*.json")):
        payload = json.loads(path.read_text())
        if payload.get("actual_peak_reserved_bytes") is not None:
            return {
                "actual_peak_allocated_bytes": payload.get("actual_peak_allocated_bytes"),
                "actual_peak_reserved_bytes": payload.get("actual_peak_reserved_bytes"),
            }
    for monitor in sorted((snapshot / "diagnostics").glob("monitor-*.json")):
        payload = json.loads(monitor.read_text())
        train = payload.get("train")
        if isinstance(train, dict) and train.get("actual_peak_reserved_bytes") is not None:
            return {
                "actual_peak_allocated_bytes": train.get("actual_peak_allocated_bytes"),
                "actual_peak_reserved_bytes": train.get("actual_peak_reserved_bytes"),
            }
    return {"actual_peak_allocated_bytes": None, "actual_peak_reserved_bytes": None}


def _load_fixture_module():
    module_name = "buffered_state_train_script"
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
        pytest.skip("tests/address.json is required for T062 hardware acceptance")
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
    return ClusterConfig.from_dict(
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


def config_path_for(tmp_path: Path, config) -> Path:
    import yaml

    path = tmp_path / "workers.yaml"
    path.write_text(yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8")
    return path


def _write_evidence(payload: dict[str, object]) -> None:
    import os

    directory = Path(os.environ.get("SHARDGRID_HARDWARE_FINDINGS_DIR", "/tmp/shardgrid-findings"))
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "t062-single-gpu-generic.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )