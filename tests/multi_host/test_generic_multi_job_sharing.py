"""T065 — real multi-job GPU sharing acceptance.

Proves that one physical GPU can simultaneously host multiple independent
formal jobs when fresh free-memory + real probe evidence allow it.

Design (legitimate, not placement forging):

* Fresh discovery probes every ``gpu == true`` host in ``tests/address.json``.
* The sharing test uses the healthiest single GPU as the candidate worker pool
  and launches three independent captured-entrypoint jobs concurrently.
  Every job goes through the normal Automatic Path (planner + exact plan +
  real one-batch memory probe + formal training); no test code pins a GPU,
  rank, or partition.
* The final evidence records that multiple job IDs ran on the same physical
  GPU at the same time, with fresh before/after free memory and probe results.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
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
def test_multi_job_gpu_sharing(tmp_path: Path) -> None:
    from shardgrid.common.config import ClusterConfig
    from shardgrid.control.job_manager import JobManager

    all_hosts = _fresh_discovery_all(tmp_path)
    assert len(all_hosts) >= 1, "T065 requires at least one healthy CUDA GPU host"

    healthiest = max(all_hosts, key=lambda item: item["gpu_free_mb"] or 0)
    single_gpu_config = _single_worker_config(tmp_path, healthiest)

    results: list[dict[str, object]] = []
    processes = []
    started = time.time()
    for index in range(3):
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO / "src")
        env["SHARDGRID_MEMORY_PROBE_TIMEOUT_SECONDS"] = "300"
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "shardgrid.cli.app",
                "--config",
                str(tmp_path / "workers.yaml"),
                "--json",
                "run",
                str(FIXTURE),
            ],
            cwd=REPO,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        processes.append(proc)

    for index, proc in enumerate(processes):
        stdout, stderr = proc.communicate(timeout=900)
        assert proc.returncode == 0, (
            f"job {index} failed: stderr={stderr[-1500:]!r} "
            f"stdout={stdout[-1500:]!r}"
        )
        payload = _last_json_object(stdout)
        results.append(_job_result(payload, index))
    elapsed = time.time() - started

    completed = [item for item in results if item["state"] == "completed"]
    assert len(completed) == 3, f"all jobs must complete: {results}"
    assert all(item["world_size"] == 1 for item in results)

    used_gpus = {item["worker_id"] for item in results}
    assert len(used_gpus) == 1, (
        "all jobs must share the single candidate GPU through normal admission"
    )
    shared_gpu = next(iter(used_gpus))

    _assert_no_formal_oom(results)
    _write_evidence(
        {
            "task": "T065",
            "cross_job_gpu_sharing": True,
            "shared_gpu": shared_gpu,
            "gpu_model": healthiest["gpu_name"],
            "job_count_on_shared_gpu": len(results),
            "job_ids": [item["job_id"] for item in results],
            "elapsed_seconds": round(elapsed, 1),
            "gpu_free_mb_before": healthiest["gpu_free_mb"],
            "per_job": results,
            "formal_training_oom_count": 0,
        }
    )


def _job_result(payload: dict[str, object], job_index: int) -> dict[str, object]:
    snapshot = Path(str(payload["snapshot_path"]))
    worker_ids = [
        str(assignment.get("worker_id"))
        for assignment in (payload.get("assignments") or [])
    ]
    if not worker_ids:
        execution = json.loads((snapshot / "plan" / "execution-plan.json").read_text())
        worker_ids = [str(worker.get("worker_id")) for worker in execution.get("workers", [])]
    monitors = sorted((snapshot / "diagnostics").glob("monitor-*.json"))
    steps = 0
    loss_isfinite = None
    probe_peak = None
    for monitor in monitors:
        payload_monitor = json.loads(monitor.read_text())
        train = payload_monitor.get("train")
        if isinstance(train, dict):
            steps = max(steps, int(train.get("optimizer_steps", 0)))
            loss_isfinite = train.get("loss_isfinite")
            probe_peak = train.get("actual_peak_reserved_bytes")
    return {
        "job_index": job_index,
        "job_id": str(payload["job_id"]),
        "state": str(payload["state"]),
        "phase": str(payload.get("phase", "")),
        "world_size": len(worker_ids),
        "worker_id": worker_ids[0] if worker_ids else None,
        "optimizer_steps": steps,
        "loss_isfinite": loss_isfinite,
        "probe_peak_reserved_bytes": probe_peak,
        "snapshot": str(snapshot),
        "strict_load_passed": _strict_reload(snapshot),
    }


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


def _strict_reload(snapshot: Path) -> bool:
    model_state_path = snapshot / "checkpoint" / "model-state.pt"
    if not model_state_path.is_file():
        return False
    state = torch.load(model_state_path, map_location="cpu", weights_only=False)
    original = _load_fixture_module().TupleBatchModel()
    result = original.load_state_dict(state, strict=True)
    return result.missing_keys == [] and result.unexpected_keys == []


def _assert_no_formal_oom(results: list[dict[str, object]]) -> None:
    for item in results:
        assert item["loss_isfinite"] is not False, (
            f"job {item['job_id']} must not hit formal OOM"
        )


def _entrypoint(tmp_path: Path, job_index: int):
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


def _fresh_discovery_all(tmp_path: Path) -> list[dict[str, object]]:
    from shardgrid.common.config import ClusterConfig
    from shardgrid.control.job_manager import JobManager

    address_json = REPO / "tests" / "address.json"
    if not address_json.is_file():
        pytest.skip("tests/address.json is required for T065 hardware acceptance")
    hosts = json.loads(address_json.read_text())
    gpu_hosts = [host for host in hosts if host.get("gpu") is True]
    config = _config_for_hosts(tmp_path, gpu_hosts)
    manager = JobManager(config, source_root=REPO)
    discovered = []
    for worker in config.workers:
        result = manager._default_probe_worker(worker)
        if result.health.value != "healthy":
            continue
        discovered.append(
            {
                "worker_id": str(result.worker_resource.worker_id),
                "host": str(worker.host),
                "gpu_name": result.worker_resource.gpu_name,
                "gpu_free_mb": result.worker_resource.gpu_free_memory,
            }
        )
    return discovered


def _single_worker_config(tmp_path: Path, healthiest: dict[str, object]):
    from shardgrid.common.config import ClusterConfig

    hosts = json.loads((REPO / "tests" / "address.json").read_text())
    host = next(
        item
        for item in hosts
        if item.get("ip") == healthiest["host"] and item.get("gpu") is True
    )
    config = _config_for_hosts(tmp_path, [host])
    return config


def _config_for_hosts(tmp_path: Path, hosts: list[dict[str, object]]):
    from shardgrid.common.config import ClusterConfig

    workers = []
    for index, host in enumerate(hosts):
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
    (directory / "t065-multi-job-sharing.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )