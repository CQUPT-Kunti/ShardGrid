"""Automatic plan real-training hardware gate."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.hardware
@pytest.mark.multi_host
def test_live_automatic_partition_gate() -> None:
    if os.environ.get("SHARDGRID_RUN_AUTOMATIC_HW") != "1":
        pytest.skip("set SHARDGRID_RUN_AUTOMATIC_HW=1 to run the automatic hardware gate")

    repo = Path(__file__).resolve().parents[2]
    command = [
        sys.executable,
        "-m",
        "shardgrid.cli.app",
        "--config",
        "examples/workers.yaml",
        "train",
        "examples/train-automatic-hf.yaml",
        "--json",
    ]
    result = subprocess.run(
        command,
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(repo / "src")},
    )
    assert result.returncode == 0, result.stderr or result.stdout
    payload = json.loads(result.stdout)
    snapshot_root = Path(payload["snapshot_path"])
    metadata = json.loads((snapshot_root / "diagnostics" / "snapshot-metadata.json").read_text())
    manifest = json.loads((snapshot_root / "checkpoint" / "manifest.json").read_text())
    checkpoint_metadata = json.loads(
        (snapshot_root / "checkpoint" / "checkpoint-metadata.json").read_text()
    )

    assert payload["plan_mode"] == "automatic"
    assert payload["planning"]["partition_source"] == "automatic"
    assert payload["state"] == "completed"
    assert payload["failure"] is None
    assert payload["world_size"] == len(payload["assignments"])
    assert manifest["world_size"] == payload["world_size"]
    assert checkpoint_metadata["partition_source"] == "automatic"
    assert checkpoint_metadata["selected_candidate_id"] == payload["planning"]["selected_candidate_id"]
    assert metadata["execution_plan_audit"]["planning"]["selected_workers"] == [
        item["worker_id"] for item in payload["placement"]
    ]
