#!/usr/bin/env python3
"""Dynamic GPU multi-model stress runner for real ShardGrid workers."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from shardgrid.common.config import load_cluster_config
from shardgrid.common.enums import Health
from shardgrid.control.job_manager import JobManager
from shardgrid.control.status_store import StatusStore

REPO = Path(__file__).resolve().parents[1]
MODEL_SPECS: tuple[dict[str, Any], ...] = (
    {
        "label": "MiniUNet",
        "zoo_model": "mini_unet",
        "candidates": (
            {"base_channels": 1024},
            {"base_channels": 768},
            {"base_channels": 512},
            {"base_channels": 384},
        ),
    },
    {
        "label": "MiniDenseNet",
        "zoo_model": "mini_densenet",
        "candidates": (
            {"width": 8192, "growth_rate": 4096, "layers": 4},
            {"width": 6144, "growth_rate": 3072, "layers": 4},
            {"width": 4096, "growth_rate": 2048, "layers": 4},
        ),
    },
    {
        "label": "ResidualMLPDAG",
        "zoo_model": "residual_mlp_dag",
        "candidates": (
            {"width": 8192},
            {"width": 6144},
            {"width": 4096},
        ),
    },
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="examples/workers.yaml")
    parser.add_argument("--ready-steps", type=int, default=5)
    parser.add_argument("--steady-extra-steps", type=int, default=10)
    parser.add_argument("--steady-samples", type=int, default=5)
    parser.add_argument("--sample-interval", type=int, default=5)
    parser.add_argument("--training-steps", type=int, default=100000)
    parser.add_argument("--max-attempts", type=int, default=100)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    run_id = datetime.now(UTC).strftime("dynamic-stress-%Y%m%d-%H%M%S")
    output_dir = Path(args.output_dir or REPO / "artifacts" / run_id).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "run_id": run_id,
        "started_at": datetime.now(UTC).isoformat(),
        "stress_model_types": [item["label"] for item in MODEL_SPECS],
        "jobs": [],
        "attempts": [],
        "saturation": {},
        "steady_state_samples": [],
        "probe_infra_summary": {},
        "multi_plan_summary": [],
        "cleanup": {},
    }
    procs: list[subprocess.Popen[str]] = []
    active_jobs: list[dict[str, Any]] = []
    known_ids: set[str] = set()
    status = "FAIL"
    exit_code = 1

    try:
        discovery = discover(args.config)
        report["initial_discovery"] = discovery
        report["discovered_gpu_count"] = len(discovery["eligible_gpus"])
        print(f"DISCOVERED_GPU_COUNT={report['discovered_gpu_count']}", flush=True)
        if not discovery["eligible_gpus"]:
            report["saturation"] = {"reason": "NO_ELIGIBLE_GPU"}
            return exit_code

        cluster_path = write_stress_cluster_config(args.config, discovery, output_dir)
        jobs_root = Path(load_cluster_config(cluster_path).jobs_root)
        known_ids = existing_job_ids(jobs_root)
        failed_round: dict[str, Any] = {}
        model_index = 0

        for attempt_index in range(1, args.max_attempts + 1):
            exited = active_job_exited(jobs_root, active_jobs)
            if exited:
                report["saturation"] = {
                    "stress_safety_failure": True,
                    "reason": "ACTIVE_JOB_EXITED_BEFORE_SATURATION",
                    "job": exited,
                }
                break
            spec = MODEL_SPECS[model_index % len(MODEL_SPECS)]
            model_index += 1
            before = discover(cluster_path)
            attempt = {
                "attempt": attempt_index,
                "model": spec["label"],
                "before_gpu_state": before["eligible_gpus"],
            }
            report["attempts"].append(attempt)
            selected = select_feasible_candidate(
                spec,
                args,
                cluster_path,
                output_dir,
                known_ids,
                attempt_index,
            )
            if selected is None:
                attempt["result"] = "NO_FEASIBLE_PLAN"
                failed_round[spec["label"]] = attempt
                if len(failed_round) == len(MODEL_SPECS):
                    evidence = build_saturation_evidence(
                        args=args,
                        report=report,
                        failed_round=failed_round,
                        active_jobs=active_jobs,
                        cluster_path=cluster_path,
                        jobs_root=jobs_root,
                    )
                    saturation = classify_saturation_evidence(evidence)
                    report["saturation_evidence"] = evidence
                    report["saturation"] = {
                        **saturation,
                        "active_jobs": len(active_jobs),
                        "failed_round": list(failed_round.values()),
                    }
                    break
                continue

            attempt.update(selected["dry_run_summary"])
            before_probe = active_job_step_counts(jobs_root, active_jobs)
            job = launch_job(
                selected["training_path"],
                cluster_path,
                jobs_root,
                known_ids,
                procs,
                ready_steps=args.ready_steps,
            )
            after_probe = active_job_step_counts(jobs_root, active_jobs)
            probe_isolation = probe_isolation_check(
                active_jobs,
                before_probe,
                after_probe,
            )
            if probe_isolation:
                attempt["result"] = "PROBE_ISOLATION_FAILURE"
                attempt["probe_isolation"] = probe_isolation
                report["saturation"] = {
                    "stress_safety_failure": True,
                    "reason": "PROBE_ISOLATION_FAILURE",
                    "attempt": attempt,
                }
                break
            if job.get("ready") is not True:
                attempt["result"] = "LAUNCH_REJECTED_OR_FAILED"
                attempt["status"] = job.get("status")
                if is_infra_failure(job):
                    attempt["infra_failure"] = True
                    report["saturation"] = {
                        "stress_infra_failure": True,
                        "reason": "PROBE_INFRA_FAILURE_OR_RENDEZVOUS",
                        "attempt": attempt,
                    }
                    break
                failed_round[spec["label"]] = attempt
                if is_safety_failure(job):
                    report["saturation"] = {"stress_safety_failure": True, "attempt": attempt}
                    break
                if len(failed_round) == len(MODEL_SPECS):
                    evidence = build_saturation_evidence(
                        args=args,
                        report=report,
                        failed_round=failed_round,
                        active_jobs=active_jobs,
                        cluster_path=cluster_path,
                        jobs_root=jobs_root,
                    )
                    saturation = classify_saturation_evidence(evidence)
                    report["saturation_evidence"] = evidence
                    report["saturation"] = {
                        **saturation,
                        "active_jobs": len(active_jobs),
                        "failed_round": list(failed_round.values()),
                    }
                    break
                continue

            failed_round = {}
            after = discover(cluster_path)
            job.update(
                {
                    "instance": f"{spec['label']}-{count_model(active_jobs, spec['label']) + 1}",
                    "model": spec["label"],
                    "zoo_model": spec["zoo_model"],
                    "model_parameters": selected["parameters"],
                    "before_gpu_state": before["eligible_gpus"],
                    "after_ready_gpu_state": after["eligible_gpus"],
                    "placement": placement_for_job(jobs_root, str(job["job_id"])),
                    "probe_summary": probe_summary_for_job(jobs_root, str(job["job_id"])),
                }
            )
            active_jobs.append(job)
            report["jobs"] = active_jobs
            report["model_instance_counts"] = dict(Counter(job["model"] for job in active_jobs))
            report["observed_stable_max_concurrent_models"] = len(active_jobs)
            report["probe_infra_summary"] = aggregate_probe_summaries(active_jobs)
            report["multi_plan_summary"] = [
                {
                    "job_id": item["job_id"],
                    "instance": item.get("instance"),
                    "model": item.get("model"),
                    "candidates": [
                        {
                            "candidate_id": candidate.get("candidate_id"),
                            "result": candidate.get("result"),
                            "subtype": candidate.get("subtype"),
                            "rendezvous_ready": candidate.get("rendezvous_ready"),
                            "phase": candidate.get("phase"),
                            "master_port": candidate.get("master_port"),
                            "actual_peak_reserved_bytes": candidate.get(
                                "actual_peak_reserved_bytes"
                            ),
                        }
                        for candidate in (item.get("probe_summary") or {}).get(
                            "candidates", []
                        )
                    ],
                    "selected_candidate_id": (item.get("probe_summary") or {}).get(
                        "selected_candidate_id"
                    ),
                }
                for item in active_jobs
            ]
            write_json(output_dir / "stress-result.json", report)
            print(f"JOB_READY {job['instance']} {job['job_id']}", flush=True)
        else:
            report["saturation"] = {
                "reached": False,
                "saturation_not_proven": True,
                "reason": "MAX_ATTEMPTS_REACHED",
                "max_attempts": args.max_attempts,
            }

        if report["saturation"].get("stress_safety_failure") or report["saturation"].get(
            "stress_infra_failure"
        ):
            return exit_code

        if not active_jobs:
            return exit_code

        baseline = {str(job["job_id"]): int(job["min_steps"]) for job in active_jobs}
        wait_for_extra_steps(jobs_root, active_jobs, baseline, args.steady_extra_steps)
        report["steady_state_samples"] = sample_steady_state(
            cluster_path,
            jobs_root,
            active_jobs,
            samples=args.steady_samples,
            interval=args.sample_interval,
        )
        steady_progress = steady_state_progress_check(
            report["steady_state_samples"],
            active_jobs,
        )
        if steady_progress:
            report["saturation"] = {
                "stress_safety_failure": True,
                "reason": "STEADY_STATE_NO_PROGRESS_OR_REGRESSION",
                "steady_state_progress": steady_progress,
            }
            return exit_code
        report["gpu_sharing_matrix"] = sharing_matrix(jobs_root, cluster_path, active_jobs)
        report["model_instance_counts"] = dict(Counter(job["model"] for job in active_jobs))
        report["observed_stable_max_concurrent_models"] = len(active_jobs)
        report["cross_job_gpu_sharing"] = any(
            item["active_job_count"] >= 2
            for item in report["gpu_sharing_matrix"].values()
        )
        status = "PASS" if report["saturation"].get("reached") else "FAIL"
    finally:
        cleanup = cleanup_jobs(args.config, [str(job["job_id"]) for job in active_jobs], procs)
        report["cleanup"] = cleanup
        report["finished_at"] = datetime.now(UTC).isoformat()
        if cleanup.get("active_reservations"):
            status = "FAIL"
            report["cleanup_failure"] = "ACTIVE_RESERVATIONS_REMAIN"
        exit_code = finish(output_dir, report, status)
    return exit_code


def discover(config_path: str | Path) -> dict[str, Any]:
    config = load_cluster_config(config_path)
    manager = JobManager(config, source_root=REPO)
    eligible: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    hosts: set[str] = set()
    for worker in config.workers:
        result = manager._probe_worker(worker)
        resource = result.worker_resource
        total = mb_to_bytes(resource.gpu_total_memory)
        free = mb_to_bytes(resource.gpu_free_memory)
        item = {
            "worker_id": str(resource.worker_id),
            "machine_id": None if resource.machine_id is None else str(resource.machine_id),
            "hostname": str(resource.hostname),
            "host": str(worker.host),
            "gpu_index": 0,
            "gpu_model": resource.gpu_name,
            "total_memory_bytes": total,
            "free_memory_bytes": free,
            "used_memory_bytes": None if total is None or free is None else max(total - free, 0),
            "utilization": resource.gpu_utilization,
            "health": resource.health.value,
        }
        if item["machine_id"]:
            hosts.add(str(item["machine_id"]))
        if resource.health is Health.HEALTHY and free:
            eligible.append(item)
        else:
            excluded.append({**item, "reason": result.probe_status})
    return {
        "physical_hosts": sorted(hosts),
        "healthy_workers": [item["worker_id"] for item in eligible],
        "eligible_gpus": eligible,
        "excluded_gpus": excluded,
    }


def write_stress_cluster_config(
    source_config: str | Path,
    discovery: dict[str, Any],
    output_dir: Path,
) -> Path:
    config = load_cluster_config(source_config)
    allowed = {item["worker_id"] for item in discovery["eligible_gpus"]}
    filtered = replace(
        config,
        workers=[worker for worker in config.workers if str(worker.worker_id) in allowed],
    )
    path = output_dir / "workers-stress.yaml"
    path.write_text(yaml.safe_dump(filtered.to_dict(), sort_keys=False), encoding="utf-8")
    return path


def select_feasible_candidate(
    spec: dict[str, Any],
    args: argparse.Namespace,
    cluster_path: Path,
    output_dir: Path,
    known_ids: set[str],
    attempt_index: int,
) -> dict[str, Any] | None:
    for candidate_index, candidate in enumerate(spec["candidates"]):
        parameters = {
            "zoo_model": spec["zoo_model"],
            "training_steps": args.training_steps,
            "learning_rate": "1e-3",
            **candidate,
        }
        training_path = write_training_config(
            output_dir,
            f"stress-{spec['zoo_model']}-{attempt_index}-{candidate_index}",
            parameters,
        )
        result = run_train(cluster_path, training_path, dry_run=True, timeout=300)
        payload = parse_json(result.stdout)
        if payload.get("job_id"):
            known_ids.add(str(payload["job_id"]))
        if result.returncode != 0:
            continue
        return {
            "training_path": training_path,
            "parameters": parameters,
            "dry_run_summary": {
                "dry_run_job_id": payload.get("job_id"),
                "selected_candidate_id": payload.get("planning", {}).get("selected_candidate_id"),
                "selected_worker_count": payload.get("planning", {}).get("selected_worker_count"),
                "dry_run_assignments": payload.get("assignments", []),
            },
        }
    return None


def write_training_config(output_dir: Path, name: str, parameters: dict[str, Any]) -> Path:
    payload = {
        "job": {"name": name, "backend": "ssh", "communication_backend": "nccl"},
        "model": {
            "name": name,
            "type": "generic_dag",
            "min_loss_decrease_percent": 0.0,
            "parameters": parameters,
        },
        "resources": {},
        "planning": {"mode": "automatic"},
        "artifacts": {"snapshot_name": name, "keep_failed_snapshots": True, "transport": "auto"},
    }
    path = output_dir / "configs" / f"{name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def launch_job(
    training_path: Path,
    cluster_path: Path,
    jobs_root: Path,
    known_ids: set[str],
    procs: list[subprocess.Popen[str]],
    *,
    ready_steps: int,
) -> dict[str, Any]:
    proc = subprocess.Popen(
        train_command(cluster_path, training_path),
        cwd=REPO,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
        env=env(),
    )
    procs.append(proc)
    job_id = wait_new_job_id(jobs_root, known_ids)
    if job_id is None:
        return {"ready": False, "reason": "JOB_ID_NOT_CREATED"}
    known_ids.add(job_id)
    ready = wait_ready(jobs_root, job_id, ready_steps=ready_steps)
    return {"job_id": job_id, **ready}


def wait_ready(jobs_root: Path, job_id: str, *, ready_steps: int) -> dict[str, Any]:
    deadline = time.time() + 900
    while time.time() < deadline:
        status = read_json(jobs_root / job_id / "job-status.json")
        monitors = monitor_payloads(jobs_root, job_id)
        steps = [
            int(monitor.get("train", {}).get("optimizer_steps") or 0)
            for monitor in monitors
            if isinstance(monitor.get("train"), dict)
        ]
        changed = [
            bool(monitor.get("train", {}).get("parameter_changed"))
            for monitor in monitors
            if isinstance(monitor.get("train"), dict)
        ]
        ready_state = status.get("state") == "training"
        enough_steps = bool(steps) and min(steps) >= ready_steps
        if ready_state and enough_steps and all(changed):
            return {
                "ready": True,
                "min_steps": min(steps),
                "rank_steps": steps,
                "loss": first_loss(monitors),
                "monitors": monitors,
            }
        if status.get("state") in {"failed", "completed", "stopped"}:
            return {"ready": False, "status": status}
        time.sleep(2)
    return {"ready": False, "reason": "READY_TIMEOUT"}


def placement_for_job(jobs_root: Path, job_id: str) -> list[dict[str, Any]]:
    root = jobs_root / job_id
    plan = read_json(root / "plan" / "execution-plan.json")
    metadata = read_json(root / "diagnostics" / "snapshot-metadata.json")
    audit = metadata["execution_plan_audit"]
    return [
        {
            "partition": worker.get("stage") or f"rank{worker['rank']}",
            "worker_id": worker["worker_id"],
            "rank": worker["rank"],
            "gpu_index": worker.get("gpu_index", 0),
            "host": audit["assignments"][index].get("host"),
            "machine_id": audit["assignments"][index].get("machine_id"),
            "estimated_peak_training_memory": worker.get("estimated_peak_training_memory"),
        }
        for index, worker in enumerate(plan["workers"])
    ]


def wait_for_extra_steps(
    jobs_root: Path,
    jobs: list[dict[str, Any]],
    baseline: dict[str, int],
    extra_steps: int,
) -> None:
    deadline = time.time() + 600
    while time.time() < deadline:
        current = {}
        for job in jobs:
            monitors = monitor_payloads(jobs_root, str(job["job_id"]))
            steps = [
                int(monitor.get("train", {}).get("optimizer_steps") or 0)
                for monitor in monitors
                if isinstance(monitor.get("train"), dict)
            ]
            if steps:
                current[str(job["job_id"])] = min(steps)
        advanced = all(
            current.get(job_id, 0) >= value + extra_steps
            for job_id, value in baseline.items()
        )
        if current and advanced:
            return
        time.sleep(3)
    raise RuntimeError("steady-state jobs did not advance by the required extra steps")


def active_job_exited(jobs_root: Path, jobs: list[dict[str, Any]]) -> dict[str, Any] | None:
    for job in jobs:
        status = read_json(jobs_root / str(job["job_id"]) / "job-status.json")
        if status.get("state") != "training":
            return {
                "job_id": job["job_id"],
                "instance": job["instance"],
                "state": status.get("state"),
                "phase": status.get("phase"),
            }
    return None


def sample_steady_state(
    cluster_path: Path,
    jobs_root: Path,
    jobs: list[dict[str, Any]],
    *,
    samples: int,
    interval: int,
) -> list[dict[str, Any]]:
    result = []
    last_gpu_used: dict[str, list[int]] = defaultdict(list)
    for index in range(samples):
        if index:
            time.sleep(interval)
        gpu_state = discover(cluster_path)["eligible_gpus"]
        for item in gpu_state:
            used = item.get("used_memory_bytes")
            if used is not None:
                last_gpu_used[str(item["worker_id"])].append(int(used))
        result.append(
            {
                "sample": index + 1,
                "gpu_state": gpu_state,
                "jobs": {
                    str(job["job_id"]): {
                        "state": read_json(
                            jobs_root / str(job["job_id"]) / "job-status.json"
                        ).get("state"),
                        "min_steps": min_steps(jobs_root, str(job["job_id"])),
                        "loss": first_loss(monitor_payloads(jobs_root, str(job["job_id"]))),
                    }
                    for job in jobs
                },
            }
        )
    for item in result:
        item["possible_memory_leak"] = {
            worker_id: values == sorted(values) and len(set(values)) > 1
            for worker_id, values in last_gpu_used.items()
        }
    return result


def sharing_matrix(
    jobs_root: Path,
    cluster_path: Path,
    jobs: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    active_ids = {str(job["job_id"]) for job in jobs}
    reservations = [
        item
        for item in StatusStore(jobs_root).active_reservations()
        if str(item.get("job_id")) in active_ids
    ]
    gpu_state = {str(item["worker_id"]): item for item in discover(cluster_path)["eligible_gpus"]}
    matrix: dict[str, dict[str, Any]] = {}
    for job in jobs:
        for placement in job["placement"]:
            worker_id = str(placement["worker_id"])
            gpu_key = (
                f"{placement.get('machine_id') or placement.get('host')}/"
                f"GPU{placement.get('gpu_index', 0)}"
            )
            entry = matrix.setdefault(
                gpu_key,
                {
                    "worker_id": worker_id,
                    "gpu_index": placement.get("gpu_index", 0),
                    "gpu_state": gpu_state.get(worker_id),
                    "items": [],
                },
            )
            entry["items"].append(
                {
                    "instance": job["instance"],
                    "job_id": job["job_id"],
                    "partition": placement["partition"],
                }
            )
    for entry in matrix.values():
        entry["active_job_count"] = len({item["job_id"] for item in entry["items"]})
        entry["active_partition_count"] = len(entry["items"])
        entry["reservation_count"] = sum(
            1 for item in reservations if str(item.get("worker_id")) == entry["worker_id"]
        )
    return matrix


def cleanup_jobs(
    config_path: str | Path,
    job_ids: list[str],
    procs: list[subprocess.Popen[str]],
) -> dict[str, Any]:
    stopped = {}
    for job_id in job_ids:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "shardgrid.cli.app",
                "--config",
                str(config_path),
                "stop",
                job_id,
                "--yes",
                "--json",
            ],
            cwd=REPO,
            env=env(),
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        stopped[job_id] = (
            parse_json(result.stdout)
            if result.stdout.strip()
            else {"returncode": result.returncode}
        )
    for proc in procs:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=30)
    root = Path(load_cluster_config(config_path).jobs_root)
    return {
        "stopped": stopped,
        "active_reservations": StatusStore(root).active_reservations(),
        "final_gpu_state": discover(config_path),
    }


def finish(output_dir: Path, report: dict[str, Any], status: str) -> int:
    report["status"] = status
    probe_infra = report.get("probe_infra_summary") or {}
    report["gate_markers"] = {
        "MEMORY_PROBE_LIVE_PREFLIGHT": (
            "PASS"
            if bool(probe_infra.get("no_stale_default_port_reuse", True))
            and bool(report.get("jobs"))
            else "FAIL"
        ),
        "MEMORY_PROBE_RESULT_CLASSIFICATION": (
            "PASS"
            if not report["saturation"].get("stress_infra_failure")
            and int(probe_infra.get("infra_failures", 0)) == 0
            else "FAIL"
        ),
        "MULTI_PLAN_MEMORY_PROBE": (
            "PASS"
            if all(
                job.get("probe_summary") is not None for job in report.get("jobs", [])
            )
            else "FAIL"
        ),
    }
    write_json(output_dir / "stress-result.json", report)
    write_text_report(output_dir, report, status)
    print(f"DYNAMIC_GPU_MULTI_MODEL_STRESS={status}", flush=True)
    for name, value in report["gate_markers"].items():
        print(f"{name}={value}", flush=True)
    observed = report.get("observed_stable_max_concurrent_models", 0)
    print(f"OBSERVED_STABLE_MAX_CONCURRENT_MODELS={observed}", flush=True)
    print(f"STRESS_ARTIFACT_DIR={output_dir}", flush=True)
    return 0 if status == "PASS" else 1


def write_text_report(output_dir: Path, report: dict[str, Any], status: str) -> None:
    lines = [
        f"DYNAMIC_GPU_MULTI_MODEL_STRESS = {status}",
        f"DISCOVERED_GPU_COUNT = {report.get('discovered_gpu_count', 0)}",
        "OBSERVED_STABLE_MAX_CONCURRENT_MODELS = "
        f"{report.get('observed_stable_max_concurrent_models', 0)}",
        f"STRESS_MODEL_TYPES = {report.get('stress_model_types', [])}",
    ]
    for name, value in (report.get("gate_markers") or {}).items():
        lines.append(f"{name} = {value}")
    lines += [
        "",
        "MODEL_INSTANCE_COUNTS",
        json.dumps(report.get("model_instance_counts", {}), indent=2, sort_keys=True),
        "",
        "JOB_PLACEMENT_TABLE",
    ]
    for job in report.get("jobs", []):
        lines.append(
            f"{job.get('instance')} | {job.get('job_id')} | {job.get('model')} | "
            f"steps={job.get('min_steps')} | loss={job.get('loss')}"
        )
    lines += [
        "",
        "GPU_SHARING_MATRIX",
        json.dumps(report.get("gpu_sharing_matrix", {}), indent=2, sort_keys=True),
    ]
    lines += [
        "",
        "PROBE_INFRA_SUMMARY",
        json.dumps(report.get("probe_infra_summary", {}), indent=2, sort_keys=True),
    ]
    lines += [
        "",
        "MULTI_PLAN_SUMMARY",
        json.dumps(report.get("multi_plan_summary", []), indent=2, sort_keys=True),
    ]
    lines += [
        "",
        "SATURATION_REASON",
        json.dumps(report.get("saturation", {}), indent=2, sort_keys=True),
    ]
    output_dir.joinpath("TEST_RESULTS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_train(
    cluster_path: Path,
    training_path: Path,
    *,
    dry_run: bool,
    timeout: int,
) -> subprocess.CompletedProcess[str]:
    command = train_command(cluster_path, training_path)
    if dry_run:
        command.append("--dry-run")
    return subprocess.run(
        command,
        cwd=REPO,
        env=env(),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def train_command(cluster_path: Path, training_path: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "shardgrid.cli.app",
        "--config",
        str(cluster_path),
        "--json",
        "train",
        str(training_path),
    ]


def env() -> dict[str, str]:
    return {**os.environ, "PYTHONPATH": str(REPO / "src")}


def existing_job_ids(root: Path) -> set[str]:
    return {path.name for path in root.glob("job-*") if (path / "job-status.json").exists()}


def wait_new_job_id(root: Path, known_ids: set[str]) -> str | None:
    deadline = time.time() + 180
    while time.time() < deadline:
        ids = sorted(
            job_id
            for job_id in existing_job_ids(root) - known_ids
            if "-probe-" not in job_id
        )
        if ids:
            return ids[-1]
        time.sleep(1)
    return None


def monitor_payloads(root: Path, job_id: str) -> list[dict[str, Any]]:
    return [
        read_json(path)
        for path in sorted((root / job_id / "diagnostics").glob("monitor-*.json"))
    ]


def min_steps(root: Path, job_id: str) -> int:
    steps = [
        int(monitor.get("train", {}).get("optimizer_steps") or 0)
        for monitor in monitor_payloads(root, job_id)
        if isinstance(monitor.get("train"), dict)
    ]
    return min(steps) if steps else 0


def first_loss(monitors: list[dict[str, Any]]) -> float | None:
    for monitor in monitors:
        loss = monitor.get("train", {}).get("final_loss")
        if loss is not None and math.isfinite(float(loss)):
            return float(loss)
    return None


def classify_job_failure(job: dict[str, Any]) -> dict[str, Any]:
    """Classify a stress job failure from structured evidence.

    Primary classifier is the structured failure ``code`` (and ``stage``)
    recorded by JobManager/SSHLauncher; message substrings are only a
    compatibility fallback for records produced before the taxonomy existed.
    """
    status = job.get("status") or {}
    failure = status.get("failure") or {}
    code = failure.get("code")
    stage = failure.get("stage")
    producer = failure.get("producer")
    retryable = failure.get("retryable")
    log_refs = failure.get("log_refs") or []
    artifact_refs = failure.get("artifact_refs") or []
    message = failure.get("message") or ""
    evidence = {
        "code": code,
        "stage": stage,
        "producer": producer,
        "retryable": retryable,
        "log_refs": list(log_refs),
        "artifact_refs": list(artifact_refs),
    }
    if code is not None:
        classification = _classify_by_code(code)
        return {**evidence, "classified": classification, "classifier": "code"}
    legacy = _classify_legacy_message(job)
    if legacy != "other":
        return {**evidence, "classified": legacy, "classifier": "message"}
    if stage is not None:
        classification = _classify_by_stage(stage)
        if classification is not None:
            return {**evidence, "classified": classification, "classifier": "stage"}
    return {**evidence, "classified": legacy, "classifier": "message"}


def _classify_by_code(code: str) -> str:
    infra_codes = {
        "NETWORK_FAILURE",
        "RENDEZVOUS_FAILURE",
        "PROCESS_LAUNCH_FAILURE",
        "INFRA_FAILURE",
    }
    if code == "FORMAL_TRAINING_OOM":
        return "safety_failure"
    if code in infra_codes:
        return "infra_failure"
    if code == "RUNTIME_FAILURE":
        return "runtime_failure"
    if code == "MEMORY_REJECT":
        return "memory_reject"
    if code == "SEARCH_BUDGET_LIMIT":
        return "search_budget_limit"
    if code == "CPU_PROCESS_SATURATION":
        return "cpu_saturation"
    if code == "GPU_MEMORY_SATURATION":
        return "gpu_memory_saturation"
    if code == "TEST_LIMIT_REACHED":
        return "test_limit_reached"
    if code == "SATURATION_NOT_PROVEN":
        return "saturation_not_proven"
    return "other"


def _classify_by_stage(stage: str) -> str | None:
    if stage == "RENDEZVOUS":
        return "infra_failure"
    if stage == "LAUNCH":
        return "infra_failure"
    if stage == "NETWORK":
        return "infra_failure"
    if stage == "TRAIN":
        return "runtime_failure"
    if stage == "PROBE":
        return "memory_reject"
    return None


def is_safety_failure(job: dict[str, Any]) -> bool:
    classification = classify_job_failure(job)
    if classification["classifier"] in {"code", "stage"}:
        return classification["classified"] == "safety_failure"
    return _legacy_safety_by_message(job)


def _legacy_safety_by_message(job: dict[str, Any]) -> bool:
    status = job.get("status") or {}
    failure = status.get("failure") or {}
    message = failure.get("message") or ""
    if any(
        token in message
        for token in (
            "NO_FEASIBLE_PLAN",
            "PROBE_INFRA_FAILURE",
            "PROBE_RUNTIME_FAILURE",
            "RESOURCE_CHANGED",
        )
    ):
        return False
    text = json.dumps(job, sort_keys=True).upper()
    return any(token in text for token in ("CUDA OOM", "OUT OF MEMORY", "DEADLOCK", "TRACEBACK"))


def is_infra_failure(job: dict[str, Any]) -> bool:
    classification = classify_job_failure(job)
    if classification["classifier"] in {"code", "stage"}:
        return classification["classified"] == "infra_failure"
    return _legacy_infra_by_message(job)


def _legacy_infra_by_message(job: dict[str, Any]) -> bool:
    status = job.get("status") or {}
    failure = status.get("failure") or {}
    message = failure.get("message") or ""
    return any(
        token in message
        for token in (
            "PROBE_INFRA_FAILURE",
            "PROBE_RUNTIME_FAILURE",
            "RESOURCE_CHANGED",
            "live execution preflight failed",
            "launcher launch failed",
            "launcher prepare failed",
            "launcher distribute failed",
            "launcher stop failed",
            "launcher cleanup failed",
        )
    )


def _classify_legacy_message(job: dict[str, Any]) -> str:
    if _legacy_infra_by_message(job):
        return "infra_failure"
    if _legacy_safety_by_message(job):
        return "safety_failure"
    return "other"


def active_job_step_counts(
    jobs_root: Path,
    jobs: list[dict[str, Any]],
) -> dict[str, int]:
    return {
        str(job["job_id"]): min_steps(jobs_root, str(job["job_id"]))
        for job in jobs
    }


def classify_step_progress(before: int, after: int) -> str:
    """Classify optimizer-step progress between two observations.

    after > before  -> progress
    after == before -> no progress / stall
    after < before  -> regression / failure
    """
    if after > before:
        return "progress"
    if after == before:
        return "no_progress"
    return "regression"


def band_attempts_from_report(report: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Map per-family candidate attempts for evidence reporting."""
    per_family: dict[str, list[dict[str, Any]]] = {}
    for attempt in report.get("attempts", []):
        label = str(attempt.get("model") or "")
        per_family.setdefault(label, []).append(
            {
                "attempt": int(attempt.get("attempt", 0)),
                "result": attempt.get("result"),
                "selected_candidate_id": attempt.get("dry_run_summary", {}).get(
                    "selected_candidate_id"
                ),
            }
        )
    return per_family


def build_saturation_evidence(
    *,
    args: argparse.Namespace,
    report: dict[str, Any],
    failed_round: dict[str, dict[str, Any]],
    active_jobs: list[dict[str, Any]],
    cluster_path: Path,
    jobs_root: Path,
) -> dict[str, Any]:
    """Collect machine-readable evidence required before declaring GPU saturation."""
    fresh = discover(cluster_path)
    fresh_free_memory = [
        {
            "worker_id": str(item.get("worker_id")),
            "gpu_total_memory_bytes": item.get("gpu_total_memory_bytes"),
            "gpu_free_memory_bytes": item.get("gpu_free_memory_bytes"),
            "used_memory_bytes": item.get("used_memory_bytes"),
        }
        for item in fresh.get("eligible_gpus", [])
    ]
    per_family = band_attempts_from_report(report)
    family_coverage: dict[str, dict[str, Any]] = {}
    family_attempted_bands: dict[str, list[str]] = {}
    for spec in MODEL_SPECS:
        label = spec["label"]
        attempts = per_family.get(label, [])
        failed = label in failed_round
        smallest_candidate_tried = False
        for candidate in spec["candidates"]:
            for attempt in attempts:
                if attempt.get("selected_candidate_id") == str(
                    candidate.get("candidate_id")
                ):
                    if (
                        int(candidate.get("base_channels") or candidate.get("width") or 0)
                        <= 512
                    ):
                        smallest_candidate_tried = True
        family_attempted_bands[label] = [
            f"{int(c.get('base_channels') or c.get('width') or 0)}"
            for c in spec["candidates"]
            if any(
                a.get("selected_candidate_id") == str(c.get("candidate_id"))
                for a in attempts
            )
        ]
        family_coverage[label] = {
            "candidate_count": len(spec["candidates"]),
            "attempted_candidate_count": len(attempts),
            "failed_in_last_round": failed,
            "smallest_band_attempted": smallest_candidate_tried,
        }
    probe_reject_reasons: list[dict[str, Any]] = []
    for job in active_jobs:
        for candidate in (job.get("probe_summary") or {}).get("candidates", []):
            if candidate.get("result") == "MEMORY_REJECT":
                probe_reject_reasons.append(
                    {
                        "job_id": str(job.get("job_id")),
                        "candidate_id": candidate.get("candidate_id"),
                        "subtype": candidate.get("subtype"),
                        "message": candidate.get("message"),
                        "actual_peak_reserved_bytes": candidate.get(
                            "actual_peak_reserved_bytes"
                        ),
                    }
                )
    non_gpu_bottlenecks: list[str] = []
    probe_infra = report.get("probe_infra_summary") or {}
    if int(probe_infra.get("infra_failures", 0)) > 0:
        non_gpu_bottlenecks.append("probe_infra_failure")
    if report.get("saturation", {}).get("stress_infra_failure"):
        non_gpu_bottlenecks.append("stress_infra_failure")
    test_limit_hit = len(report.get("attempts", [])) >= int(args.max_attempts)
    if test_limit_hit:
        non_gpu_bottlenecks.append("test_limit_reached")
    all_smallest_band_tried = all(
        coverage["smallest_band_attempted"]
        for coverage in family_coverage.values()
    )
    all_families_failed_last_round = all(
        coverage["failed_in_last_round"]
        for coverage in family_coverage.values()
    )
    fresh_free_total_mb = sum(
        int(item.get("gpu_free_memory_bytes") or 0) for item in fresh_free_memory
    ) // (1024 * 1024)
    return {
        "fresh_gpu_state": fresh.get("eligible_gpus", []),
        "fresh_free_memory_mb": fresh_free_total_mb,
        "family_coverage": family_coverage,
        "attempted_bands_per_family": family_attempted_bands,
        "all_smallest_band_attempted": all_smallest_band_tried,
        "all_families_failed_last_round": all_families_failed_last_round,
        "probe_reject_reasons": probe_reject_reasons,
        "candidate_search": {
            "attempt_count": len(report.get("attempts", [])),
            "max_attempts": int(args.max_attempts),
        },
        "non_gpu_bottlenecks": non_gpu_bottlenecks,
        "active_job_count": len(active_jobs),
        "active_jobs": [
            {
                "job_id": str(job.get("job_id")),
                "instance": job.get("instance"),
                "min_steps": job.get("min_steps"),
            }
            for job in active_jobs
        ],
        "remaining_plausible_candidates": sum(
            int(coverage["candidate_count"]) - int(coverage["attempted_candidate_count"])
            for coverage in family_coverage.values()
        ),
        "checkpoint_evidence": checkpoint_sampling_evidence(jobs_root, active_jobs),
        "cleanup_evidence": report.get("cleanup") or {},
    }


def classify_saturation_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    """Only declare GPU memory saturation when evidence is sufficient."""
    if evidence.get("non_gpu_bottlenecks"):
        return {
            "reached": False,
            "code": "SATURATION_NOT_PROVEN",
            "reason": "NON_GPU_BOTTLENECK_FIRST",
            "non_gpu_bottlenecks": evidence.get("non_gpu_bottlenecks"),
        }
    if not evidence.get("all_families_failed_last_round"):
        return {
            "reached": False,
            "code": "SATURATION_NOT_PROVEN",
            "reason": "FAMILY_COVERAGE_INCOMPLETE",
            "family_coverage": evidence.get("family_coverage"),
        }
    if not evidence.get("all_smallest_band_attempted"):
        return {
            "reached": False,
            "code": "SEARCH_BUDGET_LIMIT",
            "reason": "SMALLEST_BAND_NOT_ATTEMPTED",
            "attempted_bands_per_family": evidence.get("attempted_bands_per_family"),
        }
    if int(evidence.get("remaining_plausible_candidates", 0)) > 0:
        return {
            "reached": False,
            "code": "SEARCH_BUDGET_LIMIT",
            "reason": "REMAINING_PLAUSIBLE_CANDIDATES",
            "remaining_plausible_candidates": evidence.get(
                "remaining_plausible_candidates"
            ),
        }
    return {
        "reached": True,
        "code": "GPU_MEMORY_SATURATION",
        "reason": "EVIDENCE_COMPLETE",
    }


def checkpoint_sampling_evidence(
    jobs_root: Path,
    jobs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Sample representative checkpoint evidence for active stress jobs."""
    samples: list[dict[str, Any]] = []
    for job in jobs:
        job_id = str(job.get("job_id"))
        job_root = jobs_root / job_id
        model_state_path = job_root / "checkpoint" / "model-state.pt"
        metadata_path = job_root / "checkpoint" / "model-state-metadata.json"
        sample: dict[str, Any] = {
            "job_id": job_id,
            "instance": job.get("instance"),
            "model": job.get("model"),
            "zoo_model": job.get("zoo_model"),
            "model_state_present": model_state_path.is_file(),
            "metadata_present": metadata_path.is_file(),
        }
        if model_state_path.is_file():
            try:
                state = torch_load_plain(model_state_path)
                sample["state_dict_key_count"] = len(state) if isinstance(state, dict) else 0
                sample["state_dict_is_mapping"] = isinstance(state, dict)
                sample["merge_evidence"] = "STATE_DICT_PLAIN_MAPPING"
                strict = _checkpoint_strict_load(job, state)
                sample.update(strict)
            except Exception as exc:
                sample["merge_evidence"] = f"LOAD_FAILED: {type(exc).__name__}"
        if metadata_path.is_file():
            try:
                metadata = read_json(metadata_path)
                sample["checkpoint_metadata_format"] = metadata.get("format")
                sample["parameter_changed"] = metadata.get("parameter_changed")
            except (OSError, json.JSONDecodeError):
                sample["checkpoint_metadata_format"] = "UNREADABLE"
        samples.append(sample)
    return {"samples": samples, "sampled_job_count": len(samples)}


def _checkpoint_strict_load(
    job: dict[str, Any],
    state: Any,
) -> dict[str, Any]:
    """Attempt strict load and validation forward using the known zoo model.

    The stress runner only records strict-load / validation-forward evidence
    when the model can be reconstructed from the catalog; otherwise it records
    that the evidence is unavailable rather than fabricating a pass.
    """
    zoo_model = job.get("zoo_model")
    parameters = job.get("model_parameters") or {}
    if not zoo_model:
        return {
            "strict_load_evidence": "NOT_ATTEMPTED_NO_MODEL",
            "validation_forward_evidence": "NOT_ATTEMPTED_NO_MODEL",
        }
    try:
        model, sample_args, sample_kwargs = build_zoo_model_and_sample_for_evidence(
            zoo_model, parameters
        )
        load_result = model.load_state_dict(state, strict=True)
        result: dict[str, Any] = {
            "strict_load_evidence": "PASS",
            "missing_keys": list(load_result.missing_keys),
            "unexpected_keys": list(load_result.unexpected_keys),
        }
        try:
            import torch

            with torch.no_grad():
                model(*sample_args, **(sample_kwargs or {}))
            result["validation_forward_evidence"] = "PASS"
        except Exception as exc:
            result["validation_forward_evidence"] = f"FAILED: {type(exc).__name__}"
        return result
    except Exception as exc:
        return {
            "strict_load_evidence": f"NOT_ATTEMPTED: {type(exc).__name__}",
            "validation_forward_evidence": "NOT_ATTEMPTED",
        }


def torch_load_plain(path: Path) -> Any:
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def build_zoo_model_and_sample_for_evidence(zoo_model: str, parameters: dict[str, Any]):
    from examples.models.generic_partition_zoo.models import (
        build_zoo_model,
        make_zoo_sample,
    )

    model = build_zoo_model(zoo_model, **parameters).eval()
    sample_args, sample_kwargs = make_zoo_sample(zoo_model, **parameters)
    return model, sample_args, sample_kwargs


def probe_isolation_check(
    jobs: list[dict[str, Any]],
    before: dict[str, int],
    after: dict[str, int],
) -> dict[str, Any] | None:
    no_progress = [
        {
            "job_id": str(job["job_id"]),
            "instance": job.get("instance"),
            "before_steps": before.get(str(job["job_id"]), 0),
            "after_steps": after.get(str(job["job_id"]), 0),
        }
        for job in jobs
        if classify_step_progress(
            int(before.get(str(job["job_id"]), 0)),
            int(after.get(str(job["job_id"]), 0)),
        )
        == "no_progress"
    ]
    regression = [
        {
            "job_id": str(job["job_id"]),
            "instance": job.get("instance"),
            "before_steps": before.get(str(job["job_id"]), 0),
            "after_steps": after.get(str(job["job_id"]), 0),
        }
        for job in jobs
        if classify_step_progress(
            int(before.get(str(job["job_id"]), 0)),
            int(after.get(str(job["job_id"]), 0)),
        )
        == "regression"
    ]
    if no_progress:
        return {
            "reason": "existing active job optimizer steps stalled (no progress) during probe/launch",
            "no_progress": no_progress,
            "stalled": no_progress,
        }
    if regression:
        return {
            "reason": "existing active job optimizer steps regressed during probe/launch",
            "regression": regression,
        }
    return None


def steady_state_progress_check(
    samples: list[dict[str, Any]],
    jobs: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Compare optimizer steps across steady-state samples.

    A job whose optimizer step count does not increase between consecutive
    samples is stalled; equal step counts must never be treated as healthy.
    """
    stalled: list[dict[str, Any]] = []
    for job in jobs:
        job_id = str(job["job_id"])
        steps = [
            int((sample.get("jobs") or {}).get(job_id, {}).get("min_steps") or 0)
            for sample in samples
        ]
        if len(steps) < 2:
            continue
        if all(step == steps[0] for step in steps):
            stalled.append(
                {
                    "job_id": job_id,
                    "instance": job.get("instance"),
                    "sample_steps": steps,
                    "progress": "no_progress",
                }
            )
        elif any(
            step < steps[index]
            for index, step in enumerate(steps[1:], start=0)
        ):
            stalled.append(
                {
                    "job_id": job_id,
                    "instance": job.get("instance"),
                    "sample_steps": steps,
                    "progress": "regression",
                }
            )
    if not stalled:
        return None
    return {
        "reason": "steady-state optimizer step evidence shows stall or regression",
        "stalled": stalled,
    }


def probe_summary_for_job(jobs_root: Path, job_id: str) -> dict[str, Any] | None:
    path = jobs_root / job_id / "diagnostics" / "memory-probe-selection.json"
    if not path.exists():
        return None
    try:
        payload = read_json(path)
    except (OSError, json.JSONDecodeError):
        return None
    return {
        "job_id": job_id,
        "candidates": [
            {
                "candidate_id": candidate.get("candidate_id"),
                "result": candidate.get("result"),
                "subtype": candidate.get("subtype"),
                "rendezvous_ready": candidate.get("rendezvous_ready"),
                "phase": candidate.get("phase"),
                "master_addr": candidate.get("master_addr"),
                "master_port": candidate.get("master_port"),
                "attempt": candidate.get("attempt"),
                "oom_evidence": candidate.get("oom_evidence"),
                "actual_peak_allocated_bytes": candidate.get(
                    "actual_peak_allocated_bytes"
                ),
                "actual_peak_reserved_bytes": candidate.get(
                    "actual_peak_reserved_bytes"
                ),
            }
            for candidate in payload.get("candidates", [])
        ],
        "selected_candidate_id": payload.get("selected_candidate_id"),
        "final_result": payload.get("final_result"),
        "probe_infra_summary": payload.get("probe_infra_summary"),
    }


def aggregate_probe_summaries(
    jobs: list[dict[str, Any]],
) -> dict[str, Any]:
    summaries = [
        item["probe_summary"]
        for item in jobs
        if isinstance(item.get("probe_summary"), dict)
    ]
    attempts = 0
    rendezvous_successes = 0
    rendezvous_retries = 0
    infra_failures = 0
    stale_default_port_reuse = 0
    fresh_ports: list[int] = []
    for summary in summaries:
        infra = summary.get("probe_infra_summary") or {}
        attempts += int(infra.get("probe_attempts", 0))
        rendezvous_successes += int(infra.get("rendezvous_successes", 0))
        rendezvous_retries += int(infra.get("rendezvous_retries", 0))
        infra_failures += int(infra.get("infra_failures", 0))
        stale_default_port_reuse += int(infra.get("stale_default_port_reuse", 0))
        for candidate in summary.get("candidates", []):
            port = candidate.get("master_port")
            if isinstance(port, int):
                fresh_ports.append(port)
    return {
        "probe_attempts": attempts,
        "rendezvous_successes": rendezvous_successes,
        "rendezvous_retries": rendezvous_retries,
        "infra_failures": infra_failures,
        "stale_default_port_reuse": stale_default_port_reuse,
        "fresh_port_per_attempt": fresh_ports,
        "no_stale_default_port_reuse": stale_default_port_reuse == 0,
    }


def count_model(jobs: list[dict[str, Any]], model: str) -> int:
    return sum(1 for job in jobs if job["model"] == model)


def mb_to_bytes(value: int | None) -> int | None:
    return None if value is None else int(value) * 1024 * 1024


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_json(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return {"raw": text[-2000:]}
    return value if isinstance(value, dict) else {"raw": value}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
