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


class CheckpointContractError(ValueError):
    stage = "checkpoint"


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
    rank: int | None = None,
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
        "rank": rank,
        "gpu_index": gpu_index,
        "gpu_id": owner.gpu_id,
        "owned_partition_ids": owner.owned_partitions,
        "ownership": {
            "owned_partitions": owner.owned_partitions,
            "local_parameter_ids": owner.local_parameter_ids,
            "local_buffer_ids": owner.local_buffer_ids,
            "read_only_state_ids": owner.read_only_state_ids,
        },
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
    expected_graph_fingerprint: str | None = None,
    expected_plan_id: str | None = None,
    expected_training_step: int | None = None,
    runtime_plan: RuntimePlan | None = None,
) -> dict[str, Any]:
    state_dict: dict[str, Any] = {}
    seen_ids: set[str] = set()
    shards = []
    expected_metadata: dict[str, Any] | None = None
    expected_workers = _expected_workers(runtime_plan)
    seen_workers: set[tuple[str, int]] = set()
    for shard_path in shard_paths:
        shard = torch.load(shard_path, map_location="cpu", weights_only=False)
        if shard.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise _checkpoint_failure(f"{shard_path} has unsupported checkpoint schema")
        metadata = {
            "job_id": shard.get("job_id"),
            "graph_fingerprint": shard.get("graph_fingerprint"),
            "plan_id": shard.get("plan_id"),
            "training_step": shard.get("training_step"),
        }
        if expected_metadata is None:
            expected_metadata = metadata
        elif metadata != expected_metadata:
            raise _checkpoint_failure(f"{shard_path} checkpoint metadata disagrees")
        _validate_expected_metadata(
            shard_path,
            metadata,
            graph_fingerprint=expected_graph_fingerprint,
            plan_id=expected_plan_id,
            training_step=expected_training_step,
        )
        worker_key = (str(shard["worker_id"]), int(shard["gpu_index"]))
        if expected_workers and worker_key not in expected_workers:
            raise _checkpoint_failure(f"{shard_path} checkpoint ownership mismatch")
        seen_workers.add(worker_key)
        _validate_ownership(shard_path, shard, expected_workers.get(worker_key))
        shards.append(
            {
                "path": str(shard_path),
                "worker_id": shard["worker_id"],
                "rank": shard.get("rank"),
                "gpu_index": shard["gpu_index"],
                "gpu_id": shard.get("gpu_id"),
                "owned_partition_ids": tuple(shard["owned_partition_ids"]),
                "ownership": dict(shard.get("ownership") or {}),
                "metadata": dict(shard.get("metadata") or {}),
            }
        )
        seen_keys: set[str] = set(state_dict)
        for section in ("parameters", "buffers"):
            for item in shard[section]:
                item_id = item["canonical_id"]
                if item_id in seen_ids:
                    raise _checkpoint_failure(
                        f"duplicate checkpoint state id {item_id!r}"
                    )
                key = item["state_dict_key"]
                if key in seen_keys:
                    raise _checkpoint_failure(f"duplicate checkpoint state key {key!r}")
                seen_ids.add(item_id)
                seen_keys.add(key)
                tensor = item["tensor"]
                _validate_entry_tensor(shard_path, item, tensor)
                state_dict[key] = tensor
    missing = set(expected_state_keys or ()) - set(state_dict)
    if missing:
        raise _checkpoint_failure(
            f"consolidated checkpoint missing keys: {sorted(missing)!r}"
        )
    missing_workers = set(expected_workers) - seen_workers
    if missing_workers:
        raise _checkpoint_failure(
            f"consolidated checkpoint missing worker shards: {sorted(missing_workers)!r}"
        )
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        **(expected_metadata or {}),
        "state_dict": state_dict,
        "shards": shards,
        "training_evidence": _training_evidence(shards),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, output_path)
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


def _validate_entry_tensor(path: Path, item: Mapping[str, Any], tensor: Any) -> None:
    actual_shape = tuple(tensor.shape)
    if tuple(item["shape"]) != actual_shape:
        raise _checkpoint_failure(
            f"{path} checkpoint shape mismatch for {item['state_dict_key']!r}"
        )
    actual_dtype = str(tensor.dtype).replace("torch.", "")
    if item["dtype"] != actual_dtype:
        raise _checkpoint_failure(
            f"{path} checkpoint dtype mismatch for {item['state_dict_key']!r}"
        )


def _validate_expected_metadata(
    path: Path,
    metadata: Mapping[str, Any],
    *,
    graph_fingerprint: str | None,
    plan_id: str | None,
    training_step: int | None,
) -> None:
    expected = {
        "graph_fingerprint": graph_fingerprint,
        "plan_id": plan_id,
        "training_step": training_step,
    }
    for key, value in expected.items():
        if value is not None and metadata.get(key) != value:
            raise _checkpoint_failure(f"{path} checkpoint {key} mismatch")


def _expected_workers(
    runtime_plan: RuntimePlan | None,
) -> dict[tuple[str, int], dict[str, Any]]:
    if runtime_plan is None:
        return {}
    return {
        (worker.worker_id, worker.gpu_index): {
            "gpu_id": worker.gpu_id,
            "owned_partitions": worker.owned_partitions,
            "local_parameter_ids": worker.local_parameter_ids,
            "local_buffer_ids": worker.local_buffer_ids,
            "read_only_state_ids": worker.read_only_state_ids,
        }
        for worker in runtime_plan.ownership.workers
    }


def _validate_ownership(
    path: Path,
    shard: Mapping[str, Any],
    expected: Mapping[str, Any] | None,
) -> None:
    if expected is None:
        return
    actual = dict(shard.get("ownership") or {})
    actual["gpu_id"] = shard.get("gpu_id")
    for key, value in expected.items():
        if isinstance(value, tuple):
            mismatch = tuple(actual.get(key, ())) != value
        else:
            mismatch = actual.get(key) != value
        if mismatch:
            raise _checkpoint_failure(f"{path} checkpoint ownership mismatch")


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


def _checkpoint_failure(message: str) -> CheckpointContractError:
    return CheckpointContractError(f"CHECKPOINT_CONTRACT_FAILURE: {message}")
