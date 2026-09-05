"""Two-worker multi-job shared-GPU gate for automatic medium/small and large/small jobs."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

import pytest

from tests.multi_host.large_model_gate import (
    active_reservations,
    emit_gate_marker,
    execution_plan_payload,
    find_feasible_plan,
    jobs_root,
    job_status_payload,
    large_b_parameters,
    live_worker_inventory,
    live_worker_memory,
    medium_model_candidates,
    monitor_payloads,
    pytorch_pipeline_cluster_config,
    run_train,
    small_model_candidates,
    snapshot_metadata_payload,
    start_train,
    stop_job,
    stop_payloads,
    three_worker_medium_packing_candidates,
    three_worker_small_packing_candidates,
    wait_for_job_ids,
    wait_for_job_terminal,
    wait_for_job_training_steps,
    write_training_config,
)

BYTES_PER_MB = 1024 * 1024


def _job_env() -> dict[str, str]:
    return {
        "SHARDGRID_AUTOMATIC_MIN_WORKERS": "2",
        "SHARDGRID_AUTOMATIC_MAX_WORKERS": "2",
    }


def _packing_job_env(worker_count: int) -> dict[str, str]:
    return {
        "SHARDGRID_AUTOMATIC_MIN_WORKERS": str(worker_count),
        "SHARDGRID_AUTOMATIC_MAX_WORKERS": str(worker_count),
    }


def _top_two_worker_ids(inventory: list[dict[str, Any]]) -> list[str]:
    ranked = sorted(
        (
            item
            for item in inventory
            if item.get("gpu_free_memory_bytes") is not None
        ),
        key=lambda item: int(item["gpu_free_memory_bytes"]),
        reverse=True,
    )
    assert len(ranked) >= 2, "multi-job sharing requires at least two live GPU workers"
    return [str(item["worker_id"]) for item in ranked[:2]]


def _memory_subset(memory: dict[str, int], worker_ids: list[str]) -> dict[str, int]:
    subset = {worker_id: int(memory[worker_id]) for worker_id in worker_ids if worker_id in memory}
    assert len(subset) == len(worker_ids), {"worker_ids": worker_ids, "memory": memory}
    return subset


def _three_worker_ids(inventory: list[dict[str, Any]]) -> list[str]:
    live_ids = sorted(
        str(item["worker_id"])
        for item in inventory
        if item.get("gpu_free_memory_bytes") is not None
    )
    assert len(live_ids) >= 3, {"discovered_gpu_count": len(live_ids), "live_ids": live_ids}
    return live_ids


def _remember_scan_job_ids(known_job_ids: set[str], attempts: list[dict[str, Any]]) -> None:
    known_job_ids.update(
        str(item["job_id"])
        for item in attempts
        if item.get("job_id")
    )


def _start_and_wait_job(
    *,
    cluster_path: Path,
    root: Path,
    known_job_ids: set[str],
    procs: list[subprocess.Popen[str]],
    training_path: Path,
    extra_env: dict[str, str],
    submit_marker: str,
    wait_marker: str,
    timeout: int = 1200,
) -> tuple[str, list[dict[str, Any]]]:
    emit_gate_marker("D1_JOB_SUBMIT_START", training_path=str(training_path))
    proc = start_train(cluster_path, training_path, extra_env=extra_env)
    procs.append(proc)
    discovered = wait_for_job_ids(root, expected=1, known=known_job_ids, timeout=300)
    assert discovered, "new job_id did not appear in jobs_root before timeout"
    job_id = discovered[0]
    emit_gate_marker("D1_JOB_SUBMITTED", job_id=job_id, training_path=str(training_path))
    emit_gate_marker(submit_marker, job_id=job_id, training_path=str(training_path))
    known_job_ids.add(job_id)
    emit_gate_marker(wait_marker, job_id=job_id)
    monitors = wait_for_job_training_steps(root, job_id, min_steps=3, timeout=timeout)
    assert monitors, f"{job_id} did not reach >=3 real training steps while RUNNING"
    emit_gate_marker(
        "D1_JOB_READY",
        job_id=job_id,
        steps={
            int(payload["rank"]): int(payload["train"]["steps"])
            for payload in monitors
            if isinstance(payload.get("train"), dict)
        },
    )
    return job_id, monitors


def _planning_memory_mb(root: Path, job_id: str) -> dict[str, int]:
    metadata = snapshot_metadata_payload(root, job_id)
    planner_workers = metadata["execution_plan_audit"]["launch_metadata"]["planning_evidence"][
        "planner_worker_resources"
    ]
    return {
        str(item["worker_id"]): int(item["gpu_free_memory_mb"])
        for item in planner_workers
        if item.get("gpu_free_memory_mb") is not None
    }


def _planning_rss(root: Path, job_id: str) -> dict[str, int | None]:
    planning_evidence = snapshot_metadata_payload(root, job_id)["execution_plan_audit"][
        "launch_metadata"
    ]["planning_evidence"]
    return {
        key: planning_evidence.get(key)
        for key in (
            "control_rss_before_planning",
            "control_rss_after_profile",
            "control_rss_after_plan_created",
            "control_rss_after_cleanup",
        )
    }


def _job_step_floor(monitors: list[dict[str, Any]]) -> int:
    steps = sorted(
        int(payload["train"]["steps"])
        for payload in monitors
        if isinstance(payload.get("train"), dict)
    )
    assert steps, monitors
    return steps[0]


def _capacity_reason(payload: dict[str, Any]) -> str | None:
    failure = payload.get("failure")
    parts: list[str] = []
    if isinstance(failure, dict):
        parts.extend(
            str(value)
            for value in (
                failure.get("stage"),
                failure.get("message"),
                failure.get("recommended_action"),
            )
            if value
        )
    parts.extend(str(value) for value in (payload.get("message"), payload.get("state")) if value)
    text = " ".join(parts).upper()
    if any(
        token in text
        for token in (
            "NO_FEASIBLE",
            "RESOURCE_CHANGED",
            "INSUFFICIENT",
            "USABLE-MEMORY",
            "USABLE MEMORY",
            "REQUIRED PEAK",
        )
    ):
        return text
    return None


def _assert_real_training(monitors: list[dict[str, Any]]) -> None:
    assert monitors
    assert all(
        isinstance(payload.get("train"), dict)
        and int(payload["train"]["steps"]) >= 3
        and payload["train"]["loss_isfinite"] is True
        and payload["train"]["distributed_initialized"] is True
        and payload["train"]["stage_materialized"] is True
        and payload["train"]["optimizer_step_completed"] is True
        and payload["train"]["full_model_materialized"] is False
        and (
            not isinstance(payload.get("placement"), dict)
            or payload["placement"].get("full_model_materialized") is False
        )
        for payload in monitors
    )


def _sharing_matrix(
    *,
    reservations: list[dict[str, Any]],
    inventory: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    current_free = {str(item["worker_id"]): item["gpu_free_memory_bytes"] for item in inventory}
    matrix: dict[str, dict[str, Any]] = {}
    for item in reservations:
        worker_id = str(item["worker_id"])
        entry = matrix.setdefault(
            worker_id,
            {
                "worker_id": worker_id,
                "gpu_index": int(item.get("gpu_index", 0)),
                "current_free_memory_bytes": current_free.get(worker_id),
                "entries": [],
            },
        )
        entry["entries"].append(
            {
                "job_id": str(item["job_id"]),
                "stage": str(item["stage"]),
                "estimated_peak_training_memory": item.get("estimated_peak_training_memory"),
            }
        )
    return matrix


def _run_packing_dry_run(
    *,
    tmp_path: Path,
    cluster_path: Path,
    root: Path,
    known_job_ids: set[str],
    name: str,
    parameters: dict[str, object],
    worker_ids: list[str],
    extra_env: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    training_path = write_training_config(
        tmp_path,
        parameters,
        name=name,
        stage_count=len(worker_ids),
        world_size=len(worker_ids),
        preferred_workers=worker_ids,
    )
    result = run_train(
        cluster_path,
        training_path,
        timeout=1200,
        dry_run=True,
        extra_env=extra_env,
    )
    payload = json.loads(result.stdout)
    payload["_returncode"] = result.returncode
    if payload.get("job_id"):
        known_job_ids.add(str(payload["job_id"]))
    execution = None
    if result.returncode == 0 and payload.get("job_id"):
        execution = execution_plan_payload(root, str(payload["job_id"]))
    return payload, execution


def _select_fixed_packing_parameters(
    *,
    tmp_path: Path,
    cluster_path: Path,
    root: Path,
    known_job_ids: set[str],
    name_prefix: str,
    worker_ids: list[str],
    base_memory: dict[str, int],
    candidates: list[dict[str, object]],
    peak_limit_ratio: float,
) -> dict[str, object]:
    peak_limit = int(min(base_memory.values()) * peak_limit_ratio)
    attempts: list[dict[str, Any]] = []
    for index, parameters in enumerate(candidates):
        payload, execution = _run_packing_dry_run(
            tmp_path=tmp_path,
            cluster_path=cluster_path,
            root=root,
            known_job_ids=known_job_ids,
            name=f"{name_prefix}-seed-{index}",
            parameters=parameters,
            worker_ids=worker_ids,
            extra_env=_packing_job_env(len(worker_ids)),
        )
        planning = payload.get("planning", {})
        peaks = (
            [
                int(item["estimated_peak_training_memory"])
                for item in execution["workers"]
                if item.get("estimated_peak_training_memory") is not None
            ]
            if execution is not None
            else []
        )
        attempts.append(
            {
                "candidate_index": index,
                "job_id": payload.get("job_id"),
                "returncode": payload.get("_returncode"),
                "selected_worker_count": planning.get("selected_worker_count"),
                "selected_candidate_id": planning.get("selected_candidate_id"),
                "max_estimated_peak_training_memory": max(peaks) if peaks else None,
                "peak_limit": peak_limit,
            }
        )
        if payload.get("_returncode") != 0:
            continue
        if (
            planning.get("selected_worker_count") != str(len(worker_ids))
            or execution is None
            or len(peaks) != len(worker_ids)
        ):
            continue
        if max(peaks) > peak_limit:
            continue
        emit_gate_marker(
            "PACKING_CONFIG_SELECTED",
            name=name_prefix,
            job_id=str(payload["job_id"]),
            selected_candidate_id=planning.get("selected_candidate_id"),
            peak_limit=peak_limit,
            peaks=peaks,
            parameters=parameters,
        )
        return parameters
    raise RuntimeError(json.dumps({"name": name_prefix, "attempts": attempts}, indent=2, sort_keys=True))


def _job_assignment_summary(root: Path, job_id: str) -> list[dict[str, Any]]:
    return [
        {
            "worker_id": str(item["worker_id"]),
            "rank": int(item["rank"]),
            "stage": str(item["stage"]),
            "estimated_peak_training_memory": item.get("estimated_peak_training_memory"),
        }
        for item in execution_plan_payload(root, job_id)["workers"]
    ]


def _verify_jobs_are_still_training(
    *,
    root: Path,
    jobs: list[dict[str, Any]],
    previous_floors: dict[str, int],
) -> dict[str, int]:
    current: dict[str, int] = {}
    for job in jobs:
        job_id = str(job["job_id"])
        status = job_status_payload(root, job_id)
        assert status["state"] == "training", {"job_id": job_id, "status": status}
        monitors = monitor_payloads(root, job_id)
        _assert_real_training(monitors)
        assert all(payload.get("process_state") == "alive" for payload in monitors), {
            "job_id": job_id,
            "monitors": monitors,
        }
        floor = _job_step_floor(monitors)
        previous = previous_floors.get(job_id)
        if previous is not None:
            assert floor > previous, {"job_id": job_id, "previous": previous, "current": floor}
        current[job_id] = floor
        job["last_min_steps"] = floor
        job["last_loss"] = next(
            (
                payload["train"].get("final_loss")
                for payload in monitors
                if isinstance(payload.get("train"), dict) and payload["train"].get("final_loss") is not None
            ),
            job.get("last_loss"),
        )
    return current


def _observe_jobs(
    *,
    root: Path,
    jobs: list[dict[str, Any]],
    marker: str,
    polls: int = 3,
    interval_seconds: int = 10,
) -> None:
    floors: dict[str, int] = {}
    for index in range(1, polls + 1):
        if index > 1:
            time.sleep(interval_seconds)
        floors = _verify_jobs_are_still_training(root=root, jobs=jobs, previous_floors=floors)
        emit_gate_marker(
            f"{marker}_{index}",
            jobs={
                job["job_id"]: {
                    "model_class": job["model_class"],
                    "min_steps": floors[job["job_id"]],
                    "last_loss": job.get("last_loss"),
                }
                for job in jobs
            },
        )


def _launch_packing_job(
    *,
    cluster_path: Path,
    root: Path,
    known_job_ids: set[str],
    procs: list[subprocess.Popen[str]],
    training_path: Path,
    submit_marker: str,
    wait_marker: str,
    worker_count: int,
) -> tuple[str | None, list[dict[str, Any]], dict[str, Any] | None]:
    proc = start_train(cluster_path, training_path, extra_env=_packing_job_env(worker_count))
    procs.append(proc)
    discovered = wait_for_job_ids(root, expected=1, known=known_job_ids, timeout=300)
    assert discovered, "new packing job_id did not appear before timeout"
    job_id = discovered[0]
    known_job_ids.add(job_id)
    emit_gate_marker(submit_marker, job_id=job_id, training_path=str(training_path))
    emit_gate_marker(wait_marker, job_id=job_id)
    monitors = wait_for_job_training_steps(root, job_id, min_steps=3, timeout=1200)
    if monitors:
        emit_gate_marker(
            "PACKING_JOB_READY",
            job_id=job_id,
            steps={
                int(payload["rank"]): int(payload["train"]["steps"])
                for payload in monitors
                if isinstance(payload.get("train"), dict)
            },
        )
        return job_id, monitors, None
    status = job_status_payload(root, job_id)
    if _capacity_reason(status) is not None:
        return None, [], status
    raise AssertionError(json.dumps(status, indent=2, sort_keys=True))


def _cleanup_jobs(
    *,
    cluster_path: Path,
    root: Path,
    job_ids: list[str],
    procs: list[subprocess.Popen[str]],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    stop_results: dict[str, dict[str, Any]] = {}
    for job_id in job_ids:
        result = stop_job(cluster_path, job_id)
        if result.returncode == 0 and result.stdout.strip():
            stop_results[job_id] = json.loads(result.stdout)
    for job_id in job_ids:
        status = wait_for_job_terminal(root, job_id, timeout=300)
        assert status is not None, f"{job_id} did not reach a terminal state after stop"
        assert status["state"] in {"stopped", "completed", "failed"}, status
    for proc in procs:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.communicate(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate(timeout=30)
    cleanup_probe = live_worker_inventory(cluster_path)
    return stop_results, cleanup_probe


def _assert_cleanup(
    *,
    root: Path,
    job_ids: list[str],
    stop_results: dict[str, dict[str, Any]],
    cleanup_probe: list[dict[str, Any]],
) -> None:
    assert set(stop_results) == set(job_ids)
    assert active_reservations(root) == []
    remaining_test_pids = [
        {"job_id": job_id, "pid": payload.get("pid"), "final_state": payload.get("final_state")}
        for job_id in job_ids
        for payload in stop_payloads(root, job_id)
        if payload.get("final_state") not in {"stopped", "missing", "exited", "completed", "failed"}
    ]
    assert not remaining_test_pids, json.dumps(
        {
            "remaining_test_pids": remaining_test_pids,
            "cleanup_probe": cleanup_probe,
        },
        indent=2,
        sort_keys=True,
    )


def _run_shared_pair_case(
    *,
    tmp_path: Path,
    first_name: str,
    first_candidates: Callable[[dict[str, int]], list[dict[str, object]]],
    second_name: str,
    second_candidates: Callable[[dict[str, int]], list[dict[str, object]]],
    first_steps_env: str,
    second_steps_env: str,
) -> None:
    cluster_path = pytorch_pipeline_cluster_config(tmp_path)
    root = jobs_root(cluster_path)
    if active_reservations(root):
        pytest.skip("active ShardGrid GPU reservations exist; rerun after current jobs finish")

    before_first = live_worker_inventory(cluster_path)
    worker_ids = _top_two_worker_ids(before_first)
    base_memory = _memory_subset(live_worker_memory(cluster_path), worker_ids)
    known_job_ids = {
        path.name for path in root.iterdir() if path.is_dir() and (path / "job-status.json").exists()
    }
    procs: list[subprocess.Popen[str]] = []
    job_ids: list[str] = []
    stop_results: dict[str, dict[str, Any]] = {}
    cleanup_probe: list[dict[str, Any]] = []
    env = _job_env()

    try:
        emit_gate_marker("D1_MEDIUM_SCAN_START", name=first_name)
        first_parameters, first_dry_run, first_scan = find_feasible_plan(
            tmp_path,
            config_path=cluster_path,
            name_prefix=first_name,
            candidates=first_candidates(base_memory),
            expected_worker_count="2",
            extra_env=env,
            stage_count=2,
            world_size=2,
            preferred_workers=worker_ids,
        )
        emit_gate_marker("D1_MEDIUM_SCAN_DONE", job_id=str(first_dry_run["job_id"]))
        _remember_scan_job_ids(known_job_ids, first_scan)
        assert first_dry_run["planning"]["attempted_worker_counts"] == [2], json.dumps(
            {"before_first": before_first, "first_scan": first_scan},
            indent=2,
            sort_keys=True,
        )
        emit_gate_marker("D1_MEDIUM_CONFIG_START", name=first_name)
        first_training_path = write_training_config(
            tmp_path,
            {**first_parameters, "training_steps": int(os.environ.get(first_steps_env, "500"))},
            name=first_name,
            stage_count=2,
            world_size=2,
            preferred_workers=worker_ids,
        )
        emit_gate_marker("D1_MEDIUM_SUBMIT_START", training_path=str(first_training_path))
        first_job_id, first_monitors = _start_and_wait_job(
            cluster_path=cluster_path,
            root=root,
            known_job_ids=known_job_ids,
            procs=procs,
            training_path=first_training_path,
            extra_env=env,
            submit_marker="D1_MEDIUM_SUBMITTED",
            wait_marker="D1_WAIT_MEDIUM_READY_START",
        )
        job_ids.append(first_job_id)
        _assert_real_training(first_monitors)
        assert job_status_payload(root, first_job_id)["state"] == "training"
        emit_gate_marker("D1_MEDIUM_READY", job_id=first_job_id)

        emit_gate_marker("D1_LIVE_PROBE_START", job_id=first_job_id)
        after_first = live_worker_inventory(cluster_path)
        after_first_memory = _memory_subset(live_worker_memory(cluster_path), worker_ids)
        emit_gate_marker(
            "D1_LIVE_PROBE_DONE",
            job_id=first_job_id,
            inventory=after_first,
            free_memory=after_first_memory,
        )
        emit_gate_marker("D1_SMALL_CONFIG_START", name=second_name)
        second_parameters, second_dry_run, second_scan = find_feasible_plan(
            tmp_path,
            config_path=cluster_path,
            name_prefix=second_name,
            candidates=second_candidates(after_first_memory),
            expected_worker_count="2",
            extra_env=env,
            stage_count=2,
            world_size=2,
            preferred_workers=worker_ids,
        )
        _remember_scan_job_ids(known_job_ids, second_scan)
        emit_gate_marker("D1_SMALL_CONFIG_WRITTEN", dry_run_job_id=str(second_dry_run["job_id"]))
        second_training_path = write_training_config(
            tmp_path,
            {**second_parameters, "training_steps": int(os.environ.get(second_steps_env, "300"))},
            name=second_name,
            stage_count=2,
            world_size=2,
            preferred_workers=worker_ids,
        )
        emit_gate_marker("D1_SMALL_SUBMIT_START", training_path=str(second_training_path))
        second_job_id, second_monitors = _start_and_wait_job(
            cluster_path=cluster_path,
            root=root,
            known_job_ids=known_job_ids,
            procs=procs,
            training_path=second_training_path,
            extra_env=env,
            submit_marker="D1_SMALL_SUBMITTED",
            wait_marker="D1_WAIT_SMALL_READY_START",
        )
        job_ids.append(second_job_id)
        _assert_real_training(second_monitors)
        emit_gate_marker("D1_SMALL_READY", job_id=second_job_id)

        running = {job_id: job_status_payload(root, job_id)["state"] for job_id in job_ids}
        assert all(state == "training" for state in running.values()), json.dumps(
            {"running": running, "second_scan": second_scan},
            indent=2,
            sort_keys=True,
        )

        shared_probe = live_worker_inventory(cluster_path)
        reservations = [
            item for item in active_reservations(root) if str(item.get("job_id")) in set(job_ids)
        ]
        emit_gate_marker("D1_SHARING_CHECK_START", job_ids=job_ids)
        assert reservations, "shared reservation audit records were not created"
        sharing_matrix = _sharing_matrix(reservations=reservations, inventory=shared_probe)
        assert any(
            len({entry["job_id"] for entry in worker["entries"]}) >= 2
            for worker in sharing_matrix.values()
        ), json.dumps(sharing_matrix, indent=2, sort_keys=True)
        emit_gate_marker("D1_SHARING_CONFIRMED", sharing_matrix=sharing_matrix)

        planning_memories = {job_id: _planning_memory_mb(root, job_id) for job_id in job_ids}
        assert any(
            planning_memories[second_job_id].get(worker_id, 0) * BYTES_PER_MB
            < base_memory.get(worker_id, 0)
            for worker_id in worker_ids
        ), json.dumps(
            {
                "base_memory": base_memory,
                "after_first": after_first,
                "planning_memories": planning_memories,
            },
            indent=2,
            sort_keys=True,
        )

        rss_evidence = {job_id: _planning_rss(root, job_id) for job_id in job_ids}
        assert all(
            all(value is not None for value in payload.values())
            for payload in rss_evidence.values()
        ), json.dumps(rss_evidence, indent=2, sort_keys=True)
        cleanup_values = [int(rss_evidence[job_id]["control_rss_after_cleanup"]) for job_id in job_ids]
        assert max(cleanup_values) - min(cleanup_values) < 512 * BYTES_PER_MB, json.dumps(
            rss_evidence,
            indent=2,
            sort_keys=True,
        )

        for job_id in job_ids:
            plan = execution_plan_payload(root, job_id)
            assert len(plan["workers"]) == 2
            assert {item["worker_id"] for item in plan["workers"]} == set(worker_ids)
            _assert_real_training(monitor_payloads(root, job_id))
    finally:
        emit_gate_marker("D1_CLEANUP_START", job_ids=job_ids)
        stop_results, cleanup_probe = _cleanup_jobs(
            cluster_path=cluster_path,
            root=root,
            job_ids=job_ids,
            procs=procs,
        )
        emit_gate_marker("D1_CLEANUP_DONE", job_ids=job_ids, stop_results=stop_results)

    _assert_cleanup(
        root=root,
        job_ids=job_ids,
        stop_results=stop_results,
        cleanup_probe=cleanup_probe,
    )


def _large_two_worker_candidates(memory: dict[str, int]) -> list[dict[str, object]]:
    primary = large_b_parameters(memory)
    smaller = dict(primary)
    smaller["memory_bank_rows"] = max(16000, int(int(primary["memory_bank_rows"]) * 0.82))
    return [primary, smaller]


@pytest.mark.hardware
@pytest.mark.multi_host
def test_d1_medium_small_share_real_gpu_free_memory_on_two_workers(tmp_path: Path) -> None:
    if os.environ.get("SHARDGRID_RUN_MULTI_JOB_HW") != "1":
        pytest.skip("set SHARDGRID_RUN_MULTI_JOB_HW=1 to run the two-worker multi-job gate")

    _run_shared_pair_case(
        tmp_path=tmp_path,
        first_name="scheme-d-medium",
        first_candidates=medium_model_candidates,
        second_name="scheme-d-small",
        second_candidates=small_model_candidates,
        first_steps_env="SHARDGRID_SCHEME_D_MEDIUM_STEPS",
        second_steps_env="SHARDGRID_SCHEME_D_SMALL_STEPS",
    )


@pytest.mark.hardware
@pytest.mark.multi_host
def test_d2_large_small_share_real_gpu_free_memory_on_two_workers(tmp_path: Path) -> None:
    if os.environ.get("SHARDGRID_RUN_MULTI_JOB_HW") != "1":
        pytest.skip("set SHARDGRID_RUN_MULTI_JOB_HW=1 to run the two-worker multi-job gate")

    _run_shared_pair_case(
        tmp_path=tmp_path,
        first_name="scheme-d-large",
        first_candidates=_large_two_worker_candidates,
        second_name="scheme-d-small",
        second_candidates=small_model_candidates,
        first_steps_env="SHARDGRID_SCHEME_D_LARGE_STEPS",
        second_steps_env="SHARDGRID_SCHEME_D_SMALL_STEPS",
    )


@pytest.mark.hardware
@pytest.mark.multi_host
def test_d3_three_gpu_packing_prefers_medium_then_small(tmp_path: Path) -> None:
    if os.environ.get("SHARDGRID_RUN_PACKING_HW") != "1":
        pytest.skip("set SHARDGRID_RUN_PACKING_HW=1 to run the three-GPU packing gate")

    cluster_path = pytorch_pipeline_cluster_config(tmp_path)
    root = jobs_root(cluster_path)
    if active_reservations(root):
        pytest.skip("active ShardGrid GPU reservations exist; rerun after current jobs finish")

    before = live_worker_inventory(cluster_path)
    worker_ids = _three_worker_ids(before)
    base_memory = _memory_subset(live_worker_memory(cluster_path), worker_ids)
    known_job_ids = {
        path.name for path in root.iterdir() if path.is_dir() and (path / "job-status.json").exists()
    }
    procs: list[subprocess.Popen[str]] = []
    stop_results: dict[str, dict[str, Any]] = {}
    cleanup_probe: list[dict[str, Any]] = []
    successful_medium_jobs: list[dict[str, Any]] = []
    successful_small_jobs: list[dict[str, Any]] = []
    failed_medium_attempts: list[dict[str, Any]] = []
    failed_small_attempts: list[dict[str, Any]] = []
    test_controller_pids: list[int] = []
    test_worker_pids: set[int] = set()
    packing_test_run_id = f"packing-{os.getpid()}-{int(time.time())}"

    emit_gate_marker(
        "PACKING_INITIAL_GPU_STATE",
        packing_test_run_id=packing_test_run_id,
        inventory=before,
        worker_ids=worker_ids,
    )

    medium_parameters = _select_fixed_packing_parameters(
        tmp_path=tmp_path,
        cluster_path=cluster_path,
        root=root,
        known_job_ids=known_job_ids,
        name_prefix="packing-medium",
        worker_ids=worker_ids,
        base_memory=base_memory,
        candidates=three_worker_medium_packing_candidates(base_memory),
        peak_limit_ratio=0.50,
    )
    small_parameters = _select_fixed_packing_parameters(
        tmp_path=tmp_path,
        cluster_path=cluster_path,
        root=root,
        known_job_ids=known_job_ids,
        name_prefix="packing-small",
        worker_ids=worker_ids,
        base_memory=base_memory,
        candidates=three_worker_small_packing_candidates(base_memory),
        peak_limit_ratio=0.25,
    )

    try:
        for index in range(1, 16):
            if successful_medium_jobs:
                _verify_jobs_are_still_training(
                    root=root,
                    jobs=successful_medium_jobs + successful_small_jobs,
                    previous_floors={
                        job["job_id"]: int(job["last_min_steps"])
                        for job in successful_medium_jobs + successful_small_jobs
                    },
                )
            inventory = live_worker_inventory(cluster_path)
            current_memory = _memory_subset(live_worker_memory(cluster_path), worker_ids)
            emit_gate_marker(f"MEDIUM_{index}_BEFORE_PLAN", inventory=inventory, free_memory=current_memory)
            dry_run, execution = _run_packing_dry_run(
                tmp_path=tmp_path,
                cluster_path=cluster_path,
                root=root,
                known_job_ids=known_job_ids,
                name=f"packing-medium-{index}-dry-run",
                parameters=medium_parameters,
                worker_ids=worker_ids,
                extra_env=_packing_job_env(len(worker_ids)),
            )
            if dry_run["_returncode"] != 0:
                reason = _capacity_reason(dry_run)
                assert reason is not None, dry_run
                failed_medium_attempts.append(
                    {"attempt": index, "job_id": dry_run.get("job_id"), "reason": reason}
                )
                emit_gate_marker("MEDIUM_CAPACITY_REACHED", attempt=index, reason=reason)
                break
            assert execution is not None
            planner_memory = _planning_memory_mb(root, str(dry_run["job_id"]))
            if successful_medium_jobs:
                assert any(
                    planner_memory.get(worker_id, 0) * BYTES_PER_MB < base_memory[worker_id]
                    for worker_id in worker_ids
                ), {"planner_memory": planner_memory, "base_memory": base_memory}
            training_path = write_training_config(
                tmp_path,
                medium_parameters,
                name=f"packing-medium-{index}",
                stage_count=len(worker_ids),
                world_size=len(worker_ids),
                preferred_workers=worker_ids,
            )
            job_id, monitors, capacity_status = _launch_packing_job(
                cluster_path=cluster_path,
                root=root,
                known_job_ids=known_job_ids,
                procs=procs,
                training_path=training_path,
                submit_marker="PACKING_MEDIUM_SUBMITTED",
                wait_marker="PACKING_WAIT_MEDIUM_READY_START",
                worker_count=len(worker_ids),
            )
            if job_id is None:
                reason = _capacity_reason(capacity_status or {}) or "RESOURCE_CHANGED"
                failed_medium_attempts.append({"attempt": index, "reason": reason})
                emit_gate_marker("MEDIUM_CAPACITY_REACHED", attempt=index, reason=reason)
                break
            _assert_real_training(monitors)
            test_controller_pids.extend(proc.pid for proc in procs if proc.pid)
            test_worker_pids.update(
                int(payload["train"]["pid"])
                for payload in monitors
                if isinstance(payload.get("train"), dict) and payload["train"].get("pid") is not None
            )
            successful_medium_jobs.append(
                {
                    "job_id": job_id,
                    "model_class": "medium",
                    "selected_worker_count": str(len(worker_ids)),
                    "assignments": _job_assignment_summary(root, job_id),
                    "planner_memory_mb": _planning_memory_mb(root, job_id),
                    "rss": _planning_rss(root, job_id),
                    "last_min_steps": _job_step_floor(monitors),
                }
            )
            emit_gate_marker(
                f"AFTER_MEDIUM_{index}",
                job_id=job_id,
                assignments=successful_medium_jobs[-1]["assignments"],
                inventory=live_worker_inventory(cluster_path),
            )
        else:
            raise AssertionError("medium packing loop hit safety limit before capacity was reached")

        assert successful_medium_jobs, "packing gate did not start any medium job"

        for index in range(1, 16):
            _verify_jobs_are_still_training(
                root=root,
                jobs=successful_medium_jobs + successful_small_jobs,
                previous_floors={
                    job["job_id"]: int(job["last_min_steps"])
                    for job in successful_medium_jobs + successful_small_jobs
                },
            )
            inventory = live_worker_inventory(cluster_path)
            current_memory = _memory_subset(live_worker_memory(cluster_path), worker_ids)
            emit_gate_marker(f"SMALL_{index}_BEFORE_PLAN", inventory=inventory, free_memory=current_memory)
            dry_run, execution = _run_packing_dry_run(
                tmp_path=tmp_path,
                cluster_path=cluster_path,
                root=root,
                known_job_ids=known_job_ids,
                name=f"packing-small-{index}-dry-run",
                parameters=small_parameters,
                worker_ids=worker_ids,
                extra_env=_packing_job_env(len(worker_ids)),
            )
            if dry_run["_returncode"] != 0:
                reason = _capacity_reason(dry_run)
                assert reason is not None, dry_run
                failed_small_attempts.append(
                    {"attempt": index, "job_id": dry_run.get("job_id"), "reason": reason}
                )
                emit_gate_marker("SMALL_CAPACITY_REACHED", attempt=index, reason=reason)
                break
            assert execution is not None
            planner_memory = _planning_memory_mb(root, str(dry_run["job_id"]))
            assert any(
                planner_memory.get(worker_id, 0) * BYTES_PER_MB < base_memory[worker_id]
                for worker_id in worker_ids
            ), {"planner_memory": planner_memory, "base_memory": base_memory}
            training_path = write_training_config(
                tmp_path,
                small_parameters,
                name=f"packing-small-{index}",
                stage_count=len(worker_ids),
                world_size=len(worker_ids),
                preferred_workers=worker_ids,
            )
            job_id, monitors, capacity_status = _launch_packing_job(
                cluster_path=cluster_path,
                root=root,
                known_job_ids=known_job_ids,
                procs=procs,
                training_path=training_path,
                submit_marker="PACKING_SMALL_SUBMITTED",
                wait_marker="PACKING_WAIT_SMALL_READY_START",
                worker_count=len(worker_ids),
            )
            if job_id is None:
                reason = _capacity_reason(capacity_status or {}) or "RESOURCE_CHANGED"
                failed_small_attempts.append({"attempt": index, "reason": reason})
                emit_gate_marker("SMALL_CAPACITY_REACHED", attempt=index, reason=reason)
                break
            _assert_real_training(monitors)
            test_worker_pids.update(
                int(payload["train"]["pid"])
                for payload in monitors
                if isinstance(payload.get("train"), dict) and payload["train"].get("pid") is not None
            )
            successful_small_jobs.append(
                {
                    "job_id": job_id,
                    "model_class": "small",
                    "selected_worker_count": str(len(worker_ids)),
                    "assignments": _job_assignment_summary(root, job_id),
                    "planner_memory_mb": _planning_memory_mb(root, job_id),
                    "rss": _planning_rss(root, job_id),
                    "last_min_steps": _job_step_floor(monitors),
                }
            )
            emit_gate_marker(
                f"AFTER_SMALL_{index}",
                job_id=job_id,
                assignments=successful_small_jobs[-1]["assignments"],
                inventory=live_worker_inventory(cluster_path),
            )
        else:
            raise AssertionError("small packing loop hit safety limit before capacity was reached")

        all_jobs = successful_medium_jobs + successful_small_jobs
        assert all_jobs, "packing gate launched no jobs"
        _observe_jobs(root=root, jobs=all_jobs, marker="PACKING_OBSERVE")

        shared_probe = live_worker_inventory(cluster_path)
        reservations = [
            item
            for item in active_reservations(root)
            if str(item.get("job_id")) in {job["job_id"] for job in all_jobs}
        ]
        sharing_matrix = _sharing_matrix(reservations=reservations, inventory=shared_probe)
        emit_gate_marker("PACKING_STEADY_STATE", sharing_matrix=sharing_matrix, inventory=shared_probe)
        assert any(
            len({entry["job_id"] for entry in worker["entries"]}) >= 2
            for worker in sharing_matrix.values()
        ), json.dumps(sharing_matrix, indent=2, sort_keys=True)
        rss_evidence = {job["job_id"]: job["rss"] for job in all_jobs}
        cleanup_values = [int(payload["control_rss_after_cleanup"]) for payload in rss_evidence.values()]
        assert max(cleanup_values) - min(cleanup_values) < 1024 * BYTES_PER_MB, json.dumps(
            rss_evidence,
            indent=2,
            sort_keys=True,
        )
        emit_gate_marker(
            "PACKING_RESULT",
            packing_test_run_id=packing_test_run_id,
            discovered_gpu_count=len(worker_ids),
            successful_medium_count=len(successful_medium_jobs),
            successful_small_count=len(successful_small_jobs),
            total_concurrent_jobs=len(all_jobs),
            all_discovered_gpus_active=all(
                worker_id in sharing_matrix and sharing_matrix[worker_id]["entries"]
                for worker_id in worker_ids
            ),
            test_controller_pids=test_controller_pids,
            test_worker_pids=sorted(test_worker_pids),
            successful_medium_jobs=successful_medium_jobs,
            successful_small_jobs=successful_small_jobs,
            failed_medium_attempts=failed_medium_attempts,
            failed_small_attempts=failed_small_attempts,
        )
    finally:
        job_ids = [job["job_id"] for job in successful_medium_jobs + successful_small_jobs]
        emit_gate_marker("PACKING_CLEANUP_START", job_ids=job_ids)
        stop_results, cleanup_probe = _cleanup_jobs(
            cluster_path=cluster_path,
            root=root,
            job_ids=job_ids,
            procs=procs,
        )
        emit_gate_marker("PACKING_CLEANUP_DONE", job_ids=job_ids, stop_results=stop_results)

    _assert_cleanup(
        root=root,
        job_ids=[job["job_id"] for job in successful_medium_jobs + successful_small_jobs],
        stop_results=stop_results,
        cleanup_probe=cleanup_probe,
    )
