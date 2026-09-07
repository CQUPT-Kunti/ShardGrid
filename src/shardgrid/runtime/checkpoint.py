"""Generic DAG checkpoint shard helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from shardgrid.planner.generic_graph import CanonicalGraphIR
from shardgrid.runtime.dag import RuntimePlan

CHECKPOINT_SCHEMA_VERSION = "shardgrid.generic_dag_checkpoint.v1"


@dataclass(frozen=True)
class CheckpointShardResult:
    path: str
    bytes: int
    parameter_count: int
    buffer_count: int


def save_worker_state_shard(
    path: Path,
    *,
    graph: CanonicalGraphIR,
    runtime_plan: RuntimePlan,
    worker_id: str,
    gpu_index: int,
    state_dict: Mapping[str, Any],
    job_id: str,
    plan_id: str,
    training_step: int,
    metadata: Mapping[str, Any] | None = None,
) -> CheckpointShardResult:
    owner = next(
        worker
        for worker in runtime_plan.ownership.workers
        if worker.worker_id == worker_id and worker.gpu_index == gpu_index
    )
    parameter_keys = _parameter_keys(graph)
    buffer_keys = _buffer_keys(graph)
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "job_id": job_id,
        "graph_fingerprint": graph.graph_fingerprint,
        "plan_id": plan_id,
        "training_step": training_step,
        "worker_id": worker_id,
        "gpu_index": gpu_index,
        "owned_partition_ids": owner.owned_partitions,
        "metadata": dict(metadata or {}),
        "parameters": _entries(owner.local_parameter_ids, parameter_keys, state_dict),
        "buffers": _entries(owner.local_buffer_ids, buffer_keys, state_dict),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    _fsync_file(path)
    return CheckpointShardResult(
        path=str(path),
        bytes=path.stat().st_size,
        parameter_count=len(payload["parameters"]),
        buffer_count=len(payload["buffers"]),
    )


def consolidate_worker_state_shards(
    shard_paths: Sequence[Path],
    output_path: Path,
    *,
    expected_state_keys: Sequence[str] | None = None,
) -> dict[str, Any]:
    state_dict: dict[str, Any] = {}
    seen_ids: set[str] = set()
    shards = []
    expected_metadata: dict[str, Any] | None = None
    for shard_path in shard_paths:
        shard = torch.load(shard_path, map_location="cpu", weights_only=False)
        if shard.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(f"{shard_path} has unsupported checkpoint schema")
        metadata = {
            "job_id": shard.get("job_id"),
            "graph_fingerprint": shard.get("graph_fingerprint"),
            "plan_id": shard.get("plan_id"),
            "training_step": shard.get("training_step"),
        }
        if expected_metadata is None:
            expected_metadata = metadata
        elif metadata != expected_metadata:
            raise ValueError(f"{shard_path} checkpoint metadata disagrees")
        shards.append(
            {
                "path": str(shard_path),
                "worker_id": shard["worker_id"],
                "gpu_index": shard["gpu_index"],
                "owned_partition_ids": tuple(shard["owned_partition_ids"]),
                "metadata": dict(shard.get("metadata") or {}),
            }
        )
        for section in ("parameters", "buffers"):
            for item in shard[section]:
                item_id = item["canonical_id"]
                if item_id in seen_ids:
                    raise ValueError(f"duplicate checkpoint state id {item_id!r}")
                seen_ids.add(item_id)
                state_dict[item["state_dict_key"]] = item["tensor"]
    missing = set(expected_state_keys or ()) - set(state_dict)
    if missing:
        raise ValueError(f"consolidated checkpoint missing keys: {sorted(missing)!r}")
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        **(expected_metadata or {}),
        "state_dict": state_dict,
        "shards": shards,
        "training_evidence": _training_evidence(shards),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    _fsync_file(output_path)
    return payload


def _entries(
    ids: Sequence[str],
    keys: Mapping[str, str],
    state_dict: Mapping[str, Any],
) -> list[dict[str, Any]]:
    entries = []
    for item_id in ids:
        key = keys[item_id]
        tensor = state_dict[key].detach().cpu()
        entries.append(
            {
                "canonical_id": item_id,
                "state_dict_key": key,
                "shape": tuple(tensor.shape),
                "dtype": str(tensor.dtype).replace("torch.", ""),
                "tensor": tensor,
            }
        )
    return entries


def _training_evidence(shards: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    worker_changed: dict[str, bool] = {}
    trainable_workers: list[str] = []
    for shard in shards:
        worker_id = str(shard["worker_id"])
        metadata = shard.get("metadata")
        if not isinstance(metadata, Mapping):
            metadata = {}
        checked = int(metadata.get("checked_parameter_count") or 0)
        changed = bool(metadata.get("parameter_changed", False))
        worker_changed[worker_id] = changed
        if checked > 0:
            trainable_workers.append(worker_id)
    return {
        "worker_parameter_changed": worker_changed,
        "any_parameter_changed": any(worker_changed.values()),
        "all_trainable_workers_parameter_changed": bool(trainable_workers)
        and all(worker_changed[worker_id] for worker_id in trainable_workers),
    }


def _parameter_keys(graph: CanonicalGraphIR) -> dict[str, str]:
    return {item.parameter_id: item.canonical_path for item in graph.parameter_uses}


def _buffer_keys(graph: CanonicalGraphIR) -> dict[str, str]:
    keys: dict[str, str] = {}
    for node in graph.nodes:
        for buffer_id, path in zip(node.buffer_ids, node.buffer_paths, strict=True):
            keys.setdefault(buffer_id, path)
    return keys


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())
