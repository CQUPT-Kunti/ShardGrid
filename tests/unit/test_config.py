from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from shardgrid.common.config import (
    ConfigValidationError,
    load_cluster_config,
    load_config_data,
    load_training_config,
)


def test_load_cluster_config_from_example() -> None:
    config = load_cluster_config(Path("examples/workers.yaml"))

    assert config.control.machine_id == "machine-a"
    assert str(config.jobs_root) == "/var/tmp/shardgrid/jobs"
    assert config.runtime.environment_manager == "conda"
    assert config.runtime.conda_environment is None
    assert config.ssh.command_timeout_seconds == 60
    assert config.ssh.probe_timeout_seconds == 120
    assert config.network.nccl_mtu == 1500
    assert len(config.workers) == 3
    assert [worker.worker_id for worker in config.workers[:2]] == ["gpu4060", "gpu1060"]
    assert config.workers[0].conda_environment == "shardgrid"
    assert config.workers[0].conda_prefix == "/home/shardgrid/miniconda3/envs/shardgrid"
    assert config.workers[0].runtime_distro == "Ubuntu-22.04"
    assert config.workers[2].enabled is False


def test_load_training_config_from_example() -> None:
    config = load_training_config(Path("examples/train-minimal.yaml"))

    assert config.job.name == "train-minimal"
    assert config.model.type == "minimal_sequential"
    assert config.resources.preferred_workers == ["gpu4060", "gpu1060"]
    assert config.artifacts.keep_failed_snapshots is True
    assert config.artifacts.transport == "auto"
    assert config.artifacts.checkpoint.consolidation.enabled is False
    assert config.artifacts.checkpoint.consolidation.device == "auto"
    assert config.artifacts.checkpoint.consolidation.required is False


def test_missing_worker_identity_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "workers-missing-id.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "control": {"machine_id": "machine-a", "hostname": "control-a.local"},
                "jobs_root": "/var/tmp/shardgrid/jobs",
                "workers": [
                    {
                        "machine_id": "machine-c",
                        "physical_os": "windows",
                        "runtime_os": "wsl2_linux",
                        "runtime": "wsl2",
                        "host": "machine-c.local",
                        "ssh_user": "shardgrid",
                    }
                ],
            }
        )
    )

    with pytest.raises(ConfigValidationError):
        load_cluster_config(path)


def test_dangerous_jobs_root_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "workers-dangerous-path.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "control": {"machine_id": "machine-a", "hostname": "control-a.local"},
                "jobs_root": "/",
                "workers": [
                    {
                        "id": "gpu4060",
                        "machine_id": "machine-c",
                        "physical_os": "windows",
                        "runtime_os": "wsl2_linux",
                        "runtime": "wsl2",
                        "host": "machine-c.local",
                        "ssh_user": "shardgrid",
                    }
                ],
            }
        )
    )

    with pytest.raises(ConfigValidationError):
        load_cluster_config(path)


def test_cluster_config_serialization_is_stable() -> None:
    config = load_cluster_config(Path("examples/workers.yaml"))

    assert config.to_dict() == load_config_data(Path("examples/workers.yaml"))


def test_training_config_serialization_is_stable() -> None:
    config = load_training_config(Path("examples/train-minimal.yaml"))

    assert config.to_dict() == load_config_data(Path("examples/train-minimal.yaml"))


def test_invalid_consolidation_device_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "train-invalid-consolidation.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "job": {"name": "train-minimal", "backend": "ssh"},
                "model": {"name": "tiny-sequential", "type": "minimal_sequential"},
                "resources": {"world_size": 2},
                "artifacts": {
                    "checkpoint": {
                        "consolidation": {
                            "enabled": True,
                            "device": "tpu",
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ConfigValidationError,
        match="artifacts.checkpoint.consolidation.device must be one of: auto, cpu, cuda",
    ):
        load_training_config(path)
