from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from shardgrid.common.config import ClusterConfig, ConfigValidationError
from shardgrid.workers.inventory import WorkerInventory, load_inventory, merge_inventory


def _cluster_config(tmp_path: Path) -> ClusterConfig:
    config_path = tmp_path / "workers.yaml"
    config_path.write_text(
        """
control:
  machine_id: machine-a
  hostname: control-a.local
jobs_root: /tmp/shardgrid/jobs
ssh: {}
runtime: {}
network: {}
backend_preference: {}
manual_override:
  preferred_workers: [gpu4060, gpu1060]
  disabled_workers: [gpu4060e]
  worker_address_overrides:
    gpu1060: machine-d-override.local
workers:
  - id: gpu4060
    machine_id: machine-c
    physical_os: windows
    runtime_os: wsl2_linux
    runtime: wsl2
    host: machine-c.local
    ssh_user: shardgrid
    local_world_size: 1
    enabled: true
  - id: gpu1060
    machine_id: machine-d
    physical_os: windows
    runtime_os: wsl2_linux
    runtime: wsl2
    host: machine-d.local
    ssh_user: shardgrid
    local_world_size: 1
    enabled: true
  - id: gpu4060e
    machine_id: machine-e
    physical_os: windows
    runtime_os: wsl2_linux
    runtime: wsl2
    host: machine-e.local
    ssh_user: shardgrid
    local_world_size: 1
    enabled: true
""".strip(),
        encoding="utf-8",
    )
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return ClusterConfig.from_dict(payload)


def test_load_inventory_supports_required_and_optional_workers(tmp_path: Path) -> None:
    config = _cluster_config(tmp_path)

    inventory = load_inventory(config)

    assert [str(worker.worker_id) for worker in inventory.workers] == [
        "gpu4060",
        "gpu1060",
        "gpu4060e",
    ]
    assert [str(worker.worker_id) for worker in inventory.enabled_workers()] == [
        "gpu4060",
        "gpu1060",
    ]
    assert [str(worker_id) for worker_id in inventory.preferred_workers] == ["gpu4060", "gpu1060"]
    assert inventory.identify_worker("gpu1060").host == "machine-d-override.local"


def test_inventory_rejects_duplicate_worker_ids(tmp_path: Path) -> None:
    config = _cluster_config(tmp_path)
    with pytest.raises(ConfigValidationError, match="duplicate worker_id"):
        loaded = load_inventory(config)
        WorkerInventory(
            workers=(
                loaded.workers[0],
                loaded.workers[1],
                loaded.workers[2],
                loaded.workers[0],
            )
        )


def test_inventory_enable_disable_and_merge(tmp_path: Path) -> None:
    config = _cluster_config(tmp_path)
    inventory = load_inventory(config)

    reenabled = inventory.enable_worker("gpu4060e")
    disabled = reenabled.disable_worker("gpu1060")
    merged = merge_inventory(disabled, config)

    assert reenabled.identify_worker("gpu4060e").enabled is True
    assert disabled.identify_worker("gpu1060").enabled is False
    assert merged.identify_worker("gpu1060").enabled is False
    assert merged.identify_worker("gpu4060e").enabled is False


def test_inventory_rejects_invalid_local_world_size(tmp_path: Path) -> None:
    config = _cluster_config(tmp_path)
    payload = config.to_dict()
    payload["workers"][0]["local_world_size"] = 2

    with pytest.raises(ConfigValidationError, match="local_world_size must be 1"):
        load_inventory(ClusterConfig.from_dict(payload))


def test_inventory_identify_unknown_worker_raises(tmp_path: Path) -> None:
    inventory = load_inventory(_cluster_config(tmp_path))

    with pytest.raises(KeyError, match="unknown worker"):
        inventory.identify_worker("missing-worker")
