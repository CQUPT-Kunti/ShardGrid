from __future__ import annotations

import json
import multiprocessing as mp
import os
import socket
import subprocess
import sys
from pathlib import Path

import torch
import torch.distributed as dist

from shardgrid.runtime.transport import recv_tensor, send_tensor, tensor_tag


def test_tensor_tag_is_stable() -> None:
    assert tensor_tag(step=3, value_id="v0017", direction="FORWARD") == tensor_tag(
        step=3,
        value_id="v0017",
        direction="FORWARD",
    )
    assert tensor_tag(step=3, value_id="v0017", direction="FORWARD") != tensor_tag(
        step=3,
        value_id="v0017",
        direction="BACKWARD",
    )


def test_gloo_forward_activation_and_backward_gradient_transport(tmp_path: Path) -> None:
    init_file = tmp_path / "dist-init"
    queue: mp.Queue = mp.Queue()
    processes = [
        mp.Process(target=_transport_worker, args=(rank, str(init_file), queue))
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    results = [queue.get(timeout=15) for _ in processes]
    for process in processes:
        process.join(timeout=15)
        assert process.exitcode == 0

    by_rank = {item["rank"]: item for item in results}
    assert by_rank[0]["forward_send_complete"] is True
    assert by_rank[0]["backward_recv_complete"] is True
    assert by_rank[1]["forward_recv_complete"] is True
    assert by_rank[1]["backward_send_complete"] is True
    assert by_rank[0]["gradient"] == [2.0, 2.0, 2.0, 2.0]


def test_generic_dag_runner_trains_two_local_ranks(tmp_path: Path) -> None:
    port = _free_port()
    roots = [tmp_path / f"rank{rank}" for rank in range(2)]
    for root in roots:
        _write_generic_dag_snapshot(root)

    processes = []
    for rank, root in enumerate(roots):
        env = {
            **os.environ,
            "PYTHONPATH": str(Path.cwd() / "src"),
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
            "RANK": str(rank),
            "WORLD_SIZE": "2",
            "LOCAL_RANK": "0",
            "SHARDGRID_BACKEND": "gloo",
            "SHARDGRID_WORKER_ID": f"worker{rank}",
            "SHARDGRID_REMOTE_SNAPSHOT_ROOT": str(root),
        }
        processes.append(
            subprocess.Popen(
                [sys.executable, "examples/models/train_generic_dag.py", "--rank", str(rank)],
                cwd=Path.cwd(),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        )

    outputs = []
    try:
        for process in processes:
            stdout, stderr = process.communicate(timeout=40)
            outputs.append((process.returncode, stdout, stderr))
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()

    assert all(returncode == 0 for returncode, _stdout, _stderr in outputs), outputs
    combined = "\n".join(stdout for _returncode, stdout, _stderr in outputs)
    assert "GENERIC_DAG_RUNTIME_EVIDENCE" in combined
    assert "T074_TRAIN_EVIDENCE" in combined
    for rank, root in enumerate(roots):
        metadata = json.loads(
            (root / "checkpoint" / "checkpoint-metadata.json").read_text()
        )
        assert metadata["rank"] == rank
        assert metadata["generic_dag_runtime_used"] is True
        assert metadata["full_model_real_materialized"] is False
        assert metadata["parameter_changed"] is True
        shard = torch.load(
            root / "checkpoint" / "model.pt",
            map_location="cpu",
            weights_only=False,
        )
        assert shard["metadata"]["parameter_changed"] is True
        assert shard["metadata"]["checked_parameter_count"] > 0


def _transport_worker(rank: int, init_file: str, queue: mp.Queue) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
    )
    device = torch.device("cpu")
    if rank == 0:
        activation = torch.arange(4, dtype=torch.float32)
        forward = send_tensor(
            activation,
            dst=1,
            step=0,
            value_id="v0",
            direction="FORWARD",
        )
        gradient, backward = recv_tensor(
            shape=(4,),
            dtype=torch.float32,
            src=1,
            device=device,
            step=0,
            value_id="v0",
            direction="BACKWARD",
        )
        queue.put(
            {
                "rank": rank,
                "forward_send_complete": forward.complete,
                "backward_recv_complete": backward.complete,
                "gradient": gradient.tolist(),
            }
        )
    else:
        activation, forward = recv_tensor(
            shape=(4,),
            dtype=torch.float32,
            src=0,
            device=device,
            step=0,
            value_id="v0",
            direction="FORWARD",
        )
        boundary = activation.detach().requires_grad_(True)
        loss = (boundary * 2).sum()
        loss.backward()
        backward = send_tensor(
            boundary.grad,
            dst=0,
            step=0,
            value_id="v0",
            direction="BACKWARD",
        )
        queue.put(
            {
                "rank": rank,
                "forward_recv_complete": forward.complete,
                "backward_send_complete": backward.complete,
            }
        )
    dist.destroy_process_group()


def _write_generic_dag_snapshot(root: Path) -> None:
    (root / "config").mkdir(parents=True)
    (root / "plan").mkdir(parents=True)
    training_config = {
        "job": {
            "name": "generic-dag-smoke",
            "backend": "ssh",
            "communication_backend": "gloo",
        },
        "model": {
            "name": "generic-dag-smoke",
            "type": "generic_dag",
            "parameters": {
                "generic_dag_runtime": "true",
                "zoo_model": "mini_unet",
                "logical_partitions": 4,
                "training_steps": 2,
            },
        },
        "resources": {"world_size": 2},
        "artifacts": {
            "checkpoint": {
                "consolidation": {
                    "enabled": True,
                    "device": "cpu",
                    "required": True,
                }
            }
        },
        "planning": {"mode": "automatic"},
    }
    execution_plan = {
        "job_id": "job-generic-dag-smoke",
        "engine": "pytorch_pipeline",
        "backend": "gloo",
        "world_size": 2,
        "master": {"address": "127.0.0.1", "port": 29500},
        "workers": [
            {"worker_id": "worker0", "rank": 0, "stage": "stage0", "gpu_index": 0},
            {"worker_id": "worker1", "rank": 1, "stage": "stage1", "gpu_index": 0},
        ],
        "labels": {"selected_candidate_id": "generic-dag-smoke"},
    }
    (root / "config" / "training-config.json").write_text(
        json.dumps(training_config),
        encoding="utf-8",
    )
    (root / "plan" / "execution-plan.json").write_text(
        json.dumps(execution_plan),
        encoding="utf-8",
    )


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])
