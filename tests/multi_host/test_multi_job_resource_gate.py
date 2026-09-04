"""Multi-job GPU ownership gate for large automatic jobs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from shardgrid.control.status_store import StatusStore
from tests.multi_host.large_model_gate import (
    jobs_root,
    large_b_parameters,
    live_worker_memory,
    pytorch_pipeline_cluster_config,
    run_train,
    start_train,
    train_env,
    wait_for_reservation,
    write_training_config,
)


@pytest.mark.hardware
@pytest.mark.multi_host
def test_second_large_job_is_rejected_while_gpus_are_reserved(tmp_path: Path) -> None:
    if os.environ.get("SHARDGRID_RUN_MULTI_JOB_HW") != "1":
        pytest.skip("set SHARDGRID_RUN_MULTI_JOB_HW=1 to run the multi-job hardware gate")

    cluster_path = pytorch_pipeline_cluster_config(tmp_path)
    root = jobs_root(cluster_path)
    if wait_for_reservation(root, timeout=1):
        pytest.skip("active ShardGrid GPU reservations exist; rerun after current jobs finish")

    memory = live_worker_memory(cluster_path)
    parameters = large_b_parameters(memory)
    parameters["training_steps"] = int(os.environ.get("SHARDGRID_MULTI_JOB_A_STEPS", "20"))
    training_path = write_training_config(tmp_path, parameters, name="large-b-multijob")

    first = start_train(cluster_path, training_path)
    reservations: list[dict[str, object]] = []
    try:
        reservations = wait_for_reservation(root, timeout=300)
        assert reservations, "first job did not reserve GPUs before timeout"

        second = run_train(cluster_path, training_path, timeout=300)
        assert second.returncode != 0, second.stderr or second.stdout
        second_payload = json.loads(second.stdout)
        failure = second_payload["failure"]
        combined = (second.stdout + "\n" + second.stderr).lower()
        assert "cuda out of memory" not in combined
        assert second_payload["state"] == "failed"
        assert failure["stage"] == "PLAN"
        assert "automatic planner failed" in failure["message"]
        assert "usable GPU memory" in failure["message"] or "NO_ELIGIBLE_WORKERS" in failure["message"]

        stdout, stderr = first.communicate(timeout=1500)
        assert first.returncode == 0, stderr or stdout
        first_payload = json.loads(stdout)
        assert first_payload["state"] == "completed"
        assert first_payload["planning"]["selected_worker_count"] == "2"
        assert first_payload["planning"]["attempted_worker_counts"] == [2]

        assert not wait_for_reservation(root, timeout=5)
        after_release = run_train(cluster_path, training_path, timeout=300, dry_run=True)
        assert after_release.returncode == 0, after_release.stderr or after_release.stdout
    finally:
        if first.poll() is None:
            for job_id in {str(item["job_id"]) for item in reservations if item.get("job_id")}:
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "shardgrid.cli.app",
                        "--config",
                        str(cluster_path),
                        "stop",
                        job_id,
                        "--yes",
                        "--json",
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=120,
                    env=train_env(),
                )
        if first.poll() is None:
            first.terminate()
            try:
                first.communicate(timeout=30)
            except subprocess.TimeoutExpired:
                first.kill()
                first.communicate(timeout=30)
        store = StatusStore(root)
        for item in reservations:
            if item.get("job_id"):
                store.release_resources(str(item["job_id"]))
