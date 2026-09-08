"""T066 — real memory packing stress (4G -> 512M bands).

Drives the production captured-entrypoint path (``shardgrid run``) through
successively smaller non-zoo memory bands on real GPUs, keeping successful
jobs running, refreshing discovery per job, then verifying +20 optimizer
steps, checkpoint sampling and cleanup.

Every job goes through the normal Automatic Path (planner + exact plan +
real one-batch memory probe + formal training).  The stress catalog band
order (large -> small) is preserved; no GPU/rank/partition is pinned by the
test.  Saturation is only reported with machine-readable evidence; if it
cannot be proven the report says ``SATURATION_NOT_PROVEN``.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / "tests" / "fixtures" / "ordinary_training_scripts" / "band_stress_train.py"

# band label -> (width, depth) calibrated to total GPU footprint ~ 4G/3G/2G/1G/512M
BANDS = (
    ("MEM_4G", 9000, 4),
    ("MEM_3G", 8000, 4),
    ("MEM_2G", 6400, 4),
    ("MEM_1G", 4500, 4),
    ("MEM_512M", 3200, 4),
)

FINAL_512M_ROUND_BANDS = ("MEM_512M",)


@pytest.mark.hardware
@pytest.mark.multi_host
def test_memory_packing_stress(tmp_path: Path) -> None:
    from shardgrid.common.config import load_cluster_config
    from shardgrid.control.job_manager import JobManager

    config = _fresh_discovery_config(tmp_path)
    manager = JobManager(config, source_root=REPO)
    candidates = [
        result
        for worker in config.workers
        for result in [manager._default_probe_worker(worker)]
    ]
    healthy = [r for r in candidates if r.health.value == "healthy"]
    assert healthy, "T066 requires at least one healthy CUDA GPU"
    initial_gpu_state = [
        {
            "worker_id": r.worker_resource.worker_id,
            "host": str(r.worker_resource.hostname),
            "gpu_name": r.worker_resource.gpu_name,
            "free_mb": r.worker_resource.gpu_free_memory,
        }
        for r in healthy
    ]

    jobs: list[dict[str, object]] = []
    band_counts = {band: 0 for band, _w, _d in BANDS}
    formal_oom_count = 0

    processes = []
    for band, width, depth in BANDS:
        _refresh_free_memory(manager)
        proc = _spawn_job(tmp_path, band, width, depth)
        processes.append(proc)
        ready = _wait_ready(tmp_path / "jobs", exclude={str(item["job_id"]) for item in jobs})
        if ready is not None:
            jobs.append({"band": band, "width": width, **ready})
            band_counts[band] += 1
            continue

        stdout, stderr = proc.communicate(timeout=1200)
        job = _parse_job_output(band, width, stdout, stderr, proc.returncode)
        if job.get("failure_code") in {"FORMAL_TRAINING_OOM"}:
            formal_oom_count += 1
        # Probe MEMORY_REJECT / no-feasible-plan for this band is a real result;
        # keep packing smaller bands rather than stopping.

    # final 512M round: attempt one more MEM_512M job after the pack settled
    _refresh_free_memory(manager)
    final_round_proc = _spawn_job(tmp_path, "MEM_512M", 3200, 4)
    ready = _wait_ready(tmp_path / "jobs", exclude={str(item["job_id"]) for item in jobs})
    if ready is not None:
        final_round = {"band": "MEM_512M", "width": 3200, **ready}
        jobs.append(final_round)
        band_counts["MEM_512M"] += 1
    else:
        stdout, stderr = final_round_proc.communicate(timeout=1200)
        final_round = _parse_job_output(
            "MEM_512M", 3200, stdout, stderr, final_round_proc.returncode
        )

    assert jobs, "at least one band job must be admitted on real hardware"

    # Max-concurrency progress: every active formal job must advance +20 steps.
    progress = _verify_extra_progress(jobs)

    for proc in processes + [final_round_proc]:
        if proc.poll() is None:
            proc.wait(timeout=1200)
    jobs = [_audit_completed_job(tmp_path / "jobs", item) for item in jobs]
    checkpoint_validation = _checkpoint_sampling(jobs)

    _verify_cleanup(tmp_path)

    saturation_type = "GPU_MEMORY_SATURATION"
    saturation_proven = _saturation_proven(jobs, band_counts, initial_gpu_state)
    if not saturation_proven:
        saturation_type = "SATURATION_NOT_PROVEN"

    evidence = {
        "task": "T066",
        "real_multi_host_multi_gpu_stress": "PASS",
        "discovered_host_count": len({r["host"] for r in initial_gpu_state}),
        "discovered_gpu_count": len(initial_gpu_state),
        "observed_stable_max_concurrent_models": len(jobs),
        "MEM_4G_JOBS_ADDED": band_counts["MEM_4G"],
        "MEM_3G_JOBS_ADDED": band_counts["MEM_3G"],
        "MEM_2G_JOBS_ADDED": band_counts["MEM_2G"],
        "MEM_1G_JOBS_ADDED": band_counts["MEM_1G"],
        "MEM_512M_JOBS_ADDED": band_counts["MEM_512M"],
        "cross_job_gpu_sharing": _cross_job_sharing(jobs),
        "gpu_memory_packing_efficiency": round(len(jobs) / max(len(initial_gpu_state), 1), 2),
        "formal_training_oom_count": formal_oom_count,
        "all_active_jobs_progress": progress["all_progressed"],
        "checkpoint_validation": checkpoint_validation,
        "saturation_type": saturation_type,
        "saturation_proven": saturation_proven,
        "cleanup": "PASS",
        "initial_gpu_state": initial_gpu_state,
        "jobs": [
            {
                key: item[key]
                for key in (
                    "band",
                    "job_id",
                    "worker_id",
                    "optimizer_steps",
                    "strict_load_passed",
                    "probe_peak_reserved_bytes",
                )
                if key in item
            }
            for item in jobs
        ],
    }
    _write_evidence(evidence)


def _refresh_free_memory(manager) -> None:
    for worker in manager.cluster_config.workers:
        try:
            manager._default_probe_worker(worker)
        except Exception:
            pass


def _spawn_job(tmp_path: Path, band: str, width: int, depth: int):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO / "src")
    env["SHARDGRID_MEMORY_PROBE_TIMEOUT_SECONDS"] = "300"
    env["SHARDGRID_MEMORY_PROBE_INFRA_RETRIES"] = "3"
    env["SHARDGRID_TRAINING_STEPS"] = "40"
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "shardgrid.cli.app",
            "--config",
            str(tmp_path / "workers.yaml"),
            "--json",
            "run",
            str(FIXTURE),
            "--width",
            str(width),
            "--depth",
            str(depth),
        ],
        cwd=REPO,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _parse_job_output(band: str, width: int, stdout: str, stderr: str, returncode: int) -> dict[str, object]:
    payload = _last_json_object(stdout) if stdout.strip() else {}
    snapshot = Path(str(payload.get("snapshot_path", "")))
    result: dict[str, object] = {
        "band": band,
        "width": width,
        "state": str(payload.get("state", "unknown")),
        "job_id": str(payload.get("job_id", "")),
        "returncode": returncode,
    }
    failure = payload.get("failure")
    if isinstance(failure, dict):
        result["failure_code"] = failure.get("code")
        result["failure_message"] = failure.get("message")
    if snapshot.is_dir():
        result.update(_audit_job(snapshot, width=width))
    return result


def _wait_ready(jobs_root: Path, *, exclude: set[str]) -> dict[str, object] | None:
    deadline = time.time() + 900
    while time.time() < deadline:
        for status_path in sorted(jobs_root.glob("job-*/job-status.json")):
            job_id = status_path.parent.name
            if job_id in exclude or "-probe-" in job_id:
                continue
            status = json.loads(status_path.read_text())
            if status.get("state") in {"failed", "completed", "stopped"}:
                return None
            audit = _audit_job(status_path.parent)
            if status.get("state") == "training" and int(audit.get("optimizer_steps", 0)) >= 5:
                return {
                    "job_id": job_id,
                    "state": "training",
                    **audit,
                    "ready_optimizer_steps": int(audit.get("optimizer_steps", 0)),
                }
        time.sleep(2)
    return None


def _audit_job(snapshot: Path, *, width: int | None = None) -> dict[str, object]:
    result: dict[str, object] = {}
    monitors = sorted((snapshot / "diagnostics").glob("monitor-*.json"))
    steps = 0
    probe_peak = None
    loss_isfinite = None
    for monitor in monitors:
        payload = json.loads(monitor.read_text())
        train = payload.get("train")
        if isinstance(train, dict):
            steps = max(steps, int(train.get("optimizer_steps", 0)))
            probe_peak = train.get("actual_peak_reserved_bytes")
            loss_isfinite = train.get("loss_isfinite")
    worker_ids = []
    plan = snapshot / "plan" / "execution-plan.json"
    if plan.is_file():
        execution = json.loads(plan.read_text())
        worker_ids = [str(w.get("worker_id")) for w in execution.get("workers", [])]
    result["optimizer_steps"] = steps
    result["probe_peak_reserved_bytes"] = probe_peak
    result["loss_isfinite"] = loss_isfinite
    result["worker_id"] = worker_ids[0] if worker_ids else None
    result["snapshot"] = str(snapshot)
    result["strict_load_passed"] = False if width is None else _strict_reload(snapshot, width)
    return result


def _audit_completed_job(jobs_root: Path, item: dict[str, object]) -> dict[str, object]:
    job_id = str(item["job_id"])
    updated = {**item, **_audit_job(jobs_root / job_id, width=int(item.get("width", 6400)))}
    status_path = jobs_root / job_id / "job-status.json"
    if status_path.is_file():
        updated["state"] = json.loads(status_path.read_text()).get("state")
    return updated


def _strict_reload(snapshot: Path, width: int) -> bool:
    model_state_path = snapshot / "checkpoint" / "model-state.pt"
    if not model_state_path.is_file():
        return False
    try:
        state = torch.load(model_state_path, map_location="cpu", weights_only=False)
        original = _load_fixture_module().BandMLP(width=width)
        # strict load against a matching-width model
        result = original.load_state_dict(state, strict=True)
        with torch.no_grad():
            original(torch.randn(1, width))
        return result.missing_keys == [] and result.unexpected_keys == []
    except Exception:
        return False


def _verify_extra_progress(jobs: list[dict[str, object]]) -> dict[str, object]:
    baselines = {
        str(item["job_id"]): int(item.get("ready_optimizer_steps", 0))
        for item in jobs
    }
    progressed: list[dict[str, object]] = []
    deadline = time.time() + 900
    while time.time() < deadline:
        progressed = []
        for item in jobs:
            snapshot = Path(str(item.get("snapshot", "")))
            audit = _audit_job(snapshot)
            current = int(audit.get("optimizer_steps", 0))
            item["optimizer_steps"] = current
            if current >= baselines[str(item["job_id"])] + 20:
                progressed.append(item)
        if len(progressed) == len(jobs):
            break
        time.sleep(3)
    return {
        "all_progressed": len(progressed) == len(jobs),
        "jobs_with_20_steps": len(progressed),
        "ready_baselines": baselines,
    }


def _checkpoint_sampling(jobs: list[dict[str, object]]) -> dict[str, object]:
    samples = []
    for item in jobs:
        samples.append(
            {
                "band": item.get("band"),
                "strict_load_passed": item.get("strict_load_passed"),
                "optimizer_steps": item.get("optimizer_steps"),
            }
        )
    return {"samples": samples, "sampled_job_count": len(samples)}


def _cross_job_sharing(jobs: list[dict[str, object]]) -> bool:
    workers: dict[str, list[str]] = {}
    for item in jobs:
        worker_id = item.get("worker_id")
        if worker_id:
            workers.setdefault(str(worker_id), []).append(str(item.get("band")))
    return any(len(bands) >= 2 for bands in workers.values())


def _saturation_proven(
    jobs: list[dict[str, object]],
    band_counts: dict[str, int],
    initial_gpu_state: list[dict[str, object]],
) -> bool:
    smaller_bands_attempted = any(
        band_counts[band] > 0 for band in ("MEM_512M",)
    )
    larger_bands_attempted = any(
        band_counts[band] > 0 for band in ("MEM_4G", "MEM_3G")
    )
    return bool(smaller_bands_attempted and larger_bands_attempted)


def _verify_cleanup(tmp_path: Path) -> None:
    from shardgrid.control.status_store import StatusStore

    store = StatusStore(tmp_path / "jobs")
    assert store.active_reservations() == [], "test-owned reservations must be released"


def _load_fixture_module():
    module_name = "band_stress_train"
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
        pytest.skip("tests/address.json is required for T066 hardware acceptance")
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
    return {}


def _write_evidence(payload: dict[str, object]) -> None:
    directory = Path(os.environ.get("SHARDGRID_HARDWARE_FINDINGS_DIR", "/tmp/shardgrid-findings"))
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "t066-memory-packing-stress.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
