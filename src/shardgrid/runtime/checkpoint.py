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
    load_state_dict: bool = False,
) -> dict[str, Any]:
    """Consolidate worker state shards into a standard model-state artifact.

    Memory-safe for large models: shard payload tensors are never all resident
    at the same time.  Validation runs per shard (bounded), and finalization
    streams each tensor's storage directly into the standard PyTorch zipfile
    format, so the control plane holds at most one shard's payload at once.

    The returned ``state_dict`` is a key manifest by default.  Set
    ``load_state_dict=True`` only when the caller must hold the full tensors
    in memory (small models / tests); production finalization leaves it off.
    """
    seen_ids: set[str] = set()
    seen_original_keys: set[str] = set()
    expected_metadata: dict[str, Any] | None = None
    expected_workers = _expected_workers(runtime_plan)
    seen_workers: set[tuple[str, int]] = set()
    shards = []

    # Pass 1: bounded validation.  Each shard is loaded, validated, then its
    # payload is released before the next shard is read.
    for shard_path in shard_paths:
        shard = torch.load(shard_path, map_location="cpu", weights_only=False)
        try:
            _validate_shard_contract(
                shard_path,
                shard,
                expected_metadata=expected_metadata,
                graph_fingerprint=expected_graph_fingerprint,
                plan_id=expected_plan_id,
                training_step=expected_training_step,
                expected_workers=expected_workers,
                seen_workers=seen_workers,
                seen_ids=seen_ids,
                seen_original_keys=seen_original_keys,
            )
            if expected_metadata is None:
                expected_metadata = {
                    "job_id": shard.get("job_id"),
                    "graph_fingerprint": shard.get("graph_fingerprint"),
                    "plan_id": shard.get("plan_id"),
                    "training_step": shard.get("training_step"),
                }
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
        finally:
            del shard
    missing_workers = set(expected_workers) - seen_workers
    if missing_workers:
        raise _checkpoint_failure(
            f"consolidated checkpoint missing worker shards: {sorted(missing_workers)!r}"
        )
    missing = set(expected_state_keys or ()) - seen_original_keys
    if missing:
        raise _checkpoint_failure(
            f"consolidated checkpoint missing keys: {sorted(missing)!r}"
        )

    # Pass 2: bounded finalization.  Each shard is re-read one at a time and
    # its tensor payload is streamed into the standard zip artifact.
    state_dict = _stream_state_dict_to_zip(
        shard_paths,
        output_path,
        expected_state_keys=expected_state_keys,
    )
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        **(expected_metadata or {}),
        "state_dict": state_dict,
        "shards": shards,
        "training_evidence": _training_evidence(shards),
    }
    _fsync_file(output_path)
    if load_state_dict:
        payload["state_dict"] = torch.load(
            output_path, map_location="cpu", weights_only=False
        )
    return payload


def _validate_shard_contract(
    path: Path,
    shard: Mapping[str, Any],
    *,
    expected_metadata: dict[str, Any] | None,
    graph_fingerprint: str | None,
    plan_id: str | None,
    training_step: int | None,
    expected_workers: set[tuple[str, int]],
    seen_workers: set[tuple[str, int]],
    seen_ids: set[str],
    seen_original_keys: set[str] | None = None,
) -> None:
    """Validate one worker shard without retaining its tensor payload."""
    if shard.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise _checkpoint_failure(f"{path} has unsupported checkpoint schema")
    metadata = {
        "job_id": shard.get("job_id"),
        "graph_fingerprint": shard.get("graph_fingerprint"),
        "plan_id": shard.get("plan_id"),
        "training_step": shard.get("training_step"),
    }
    if expected_metadata is not None and metadata != expected_metadata:
        raise _checkpoint_failure(f"{path} checkpoint metadata disagrees")
    _validate_expected_metadata(
        path,
        metadata,
        graph_fingerprint=graph_fingerprint,
        plan_id=plan_id,
        training_step=training_step,
    )
    worker_key = (str(shard["worker_id"]), int(shard["gpu_index"]))
    if expected_workers and worker_key not in expected_workers:
        raise _checkpoint_failure(f"{path} checkpoint ownership mismatch")
    seen_workers.add(worker_key)
    _validate_ownership(path, shard, expected_workers.get(worker_key))
    _validate_claimed_entries(path, shard)
    for section in ("parameters", "buffers"):
        for item in shard[section]:
            item_id = item["canonical_id"]
            if item_id in seen_ids:
                raise _checkpoint_failure(
                    f"duplicate checkpoint state id {item_id!r}"
                )
            seen_ids.add(item_id)
            key = item["state_dict_key"]
            if seen_original_keys is not None:
                if key in seen_original_keys:
                    raise _checkpoint_failure(f"duplicate checkpoint state key {key!r}")
                seen_original_keys.add(key)
            _validate_entry_tensor(path, item, item["tensor"])


def _stream_state_dict_to_zip(
    shard_paths: Sequence[Path],
    output_path: Path,
    *,
    expected_state_keys: Sequence[str] | None,
) -> dict[str, Any]:
    """Stream worker shard tensors into a standard PyTorch model-state zip.

    The produced artifact is a normal ``torch.load``-able ``state_dict``.  Only
    one shard's tensor payload is resident at a time: each tensor storage is
    written directly to its own zip record, and the structure pickle references
    storages by key via the standard persistent-id protocol.
    """
    import io
    import pickle
    import sys
    import zipfile

    import torch._utils

    expected = set(expected_state_keys or ())
    written: set[str] = set()
    seen_original_keys: set[str] = set()
    entries = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(output_path.name + ".streaming.tmp")
    try:
        with zipfile.ZipFile(temp_path, "w", zipfile.ZIP_STORED) as zip_file:
            for shard_path in shard_paths:
                shard = torch.load(shard_path, map_location="cpu", weights_only=False)
                try:
                    for section in ("parameters", "buffers"):
                        for item in shard[section]:
                            key = item["state_dict_key"]
                            if key in seen_original_keys:
                                raise _checkpoint_failure(
                                    f"duplicate checkpoint state key {key!r}"
                                )
                            seen_original_keys.add(key)
                            tensor = item["tensor"]
                            storage_key = str(len(entries))
                            _write_storage_record(zip_file, storage_key, tensor)
                            entries.append(
                                {
                                    "key": key,
                                    "storage_key": storage_key,
                                    "shape": tuple(tensor.shape),
                                    "stride": tuple(tensor.stride()),
                                    "offset": 0,
                                    "dtype": tensor.dtype,
                                    "requires_grad": tensor.requires_grad,
                                    "numel": tensor.numel(),
                                }
                            )
                            written.add(key)
                finally:
                    del shard
            _write_structure_pickle(zip_file, entries)
            _write_archive_metadata_records(zip_file)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise
    temp_path.replace(output_path)
    missing = expected - written
    if missing:
        raise _checkpoint_failure(
            f"consolidated checkpoint missing keys: {sorted(missing)!r}"
        )
    return dict.fromkeys([entry["key"] for entry in entries])


def _write_storage_record(zip_file: Any, storage_key: str, tensor: Any) -> None:
    if tensor.device.type != "cpu":
        tensor = tensor.detach().cpu()
    else:
        tensor = tensor.detach()
    data = tensor.contiguous().numpy().tobytes()
    zip_file.writestr(f"archive/data/{storage_key}", data)


def _write_structure_pickle(zip_file: Any, entries: Sequence[Mapping[str, Any]]) -> None:
    import io
    import pickle

    import torch._utils

    class StorageRef:
        def __init__(self, key: str, dtype: Any, numel: int) -> None:
            self.key = key
            self.dtype = dtype
            self.numel = numel

    class _Pickler(pickle.Pickler):
        def persistent_id(self, obj):
            if isinstance(obj, StorageRef):
                storage_type = _storage_type_for(obj.dtype)
                return ("storage", storage_type, obj.key, "cpu", obj.numel)
            return None

    class TensorProxy:
        def __init__(
            self,
            ref: Any,
            size: tuple[int, ...],
            stride: tuple[int, ...],
            offset: int,
            requires_grad: bool,
        ) -> None:
            self.ref = ref
            self.size = size
            self.stride = stride
            self.offset = offset
            self.requires_grad = requires_grad

        def __reduce__(self):
            return (
                torch._utils._rebuild_tensor_v2,
                (
                    self.ref,
                    self.offset,
                    self.size,
                    self.stride,
                    self.requires_grad,
                    (),
                ),
            )

    structure = {}
    for entry in entries:
        ref = StorageRef(entry["storage_key"], entry["dtype"], entry["numel"])
        structure[entry["key"]] = TensorProxy(
            ref,
            entry["shape"],
            entry["stride"],
            entry["offset"],
            entry["requires_grad"],
        )
    buffer = io.BytesIO()
    pickler = _Pickler(buffer, protocol=2)
    pickler.dump(structure)
    zip_file.writestr("archive/data.pkl", buffer.getvalue())


def _storage_type_for(dtype: Any) -> Any:
    import torch

    mapping = {
        torch.float32: torch.FloatStorage,
        torch.float64: torch.DoubleStorage,
        torch.float16: torch.HalfStorage,
        torch.bfloat16: torch.BFloat16Storage,
        torch.int64: torch.LongStorage,
        torch.int32: torch.IntStorage,
        torch.int16: torch.ShortStorage,
        torch.int8: torch.CharStorage,
        torch.uint8: torch.ByteStorage,
        torch.bool: torch.BoolStorage,
    }
    return mapping.get(dtype, torch.FloatStorage)


def _write_archive_metadata_records(zip_file: Any) -> None:
    import sys

    zip_file.writestr("archive/.format_version", "1")
    zip_file.writestr("archive/.storage_alignment", "64")
    zip_file.writestr("archive/byteorder", sys.byteorder)
    zip_file.writestr("archive/version", "3\n")
    zip_file.writestr("archive/.data/serialization_id", "0")


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


def _validate_claimed_entries(path: Path, shard: Mapping[str, Any]) -> None:
    ownership = shard.get("ownership")
    if not isinstance(ownership, Mapping):
        raise _checkpoint_failure(f"{path} checkpoint ownership missing")
    claimed_parameters = set(ownership.get("local_parameter_ids") or ())
    claimed_buffers = set(ownership.get("local_buffer_ids") or ())
    entry_parameters = {item["canonical_id"] for item in shard.get("parameters", ())}
    entry_buffers = {item["canonical_id"] for item in shard.get("buffers", ())}
    if entry_parameters != claimed_parameters:
        raise _checkpoint_failure(
            f"{path} checkpoint parameter entries disagree with claimed ownership"
        )
    if entry_buffers != claimed_buffers:
        raise _checkpoint_failure(
            f"{path} checkpoint buffer entries disagree with claimed ownership"
        )


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
