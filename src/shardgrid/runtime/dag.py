"""Generic local DAG runtime contracts.

This module is intentionally independent from SSH/GPipe. It proves the runtime
model can own multiple logical partitions per worker/GPU before wiring remote
transport.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable, Mapping

from shardgrid.planner.generic_graph import CanonicalGraphIR
from shardgrid.planner.planning_contract import (
    LogicalPartitionPlan,
    LogicalPartitionSpec,
    PlacementPlan,
)

TensorMap = dict[str, Any]
PartitionCallable = Callable[[Mapping[str, Any]], Mapping[str, Any]]


class EdgeKind(StrEnum):
    LOCAL = "local"
    REMOTE = "remote"


@dataclass(frozen=True)
class WorkerOwnershipSpec:
    worker_id: str
    gpu_index: int
    gpu_id: str
    owned_partitions: tuple[str, ...]
    local_parameter_ids: tuple[str, ...]
    local_buffer_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__


@dataclass(frozen=True)
class WorkerOwnershipPlan:
    workers: tuple[WorkerOwnershipSpec, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"workers": [worker.to_dict() for worker in self.workers]}


@dataclass(frozen=True)
class RuntimeEdgeSpec:
    producer_partition: str
    consumer_partition: str
    value_id: str
    edge_kind: EdgeKind
    producer_worker_id: str
    consumer_worker_id: str
    producer_gpu_id: str
    consumer_gpu_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.__dict__,
            "edge_kind": self.edge_kind.value,
        }


@dataclass(frozen=True)
class RuntimePlan:
    graph_fingerprint: str
    ownership: WorkerOwnershipPlan
    edges: tuple[RuntimeEdgeSpec, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph_fingerprint": self.graph_fingerprint,
            "ownership": self.ownership.to_dict(),
            "edges": [edge.to_dict() for edge in self.edges],
        }


@dataclass(frozen=True)
class RuntimePartition:
    spec: LogicalPartitionSpec
    executable: PartitionCallable


@dataclass(frozen=True)
class RuntimeExecutionEvidence:
    executed_partition_sequence: tuple[str, ...]
    local_edges_used: int
    remote_edges_used: int
    peak_value_store_bytes: int
    optimizer_step_completed: bool

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__


class ValueStore:
    def __init__(self, consumer_counts: Mapping[str, int]) -> None:
        self._values: TensorMap = {}
        self._remaining_consumers = dict(consumer_counts)
        self.peak_bytes = 0

    def put(self, value_id: str, value: Any) -> None:
        self._values[value_id] = value
        self.peak_bytes = max(self.peak_bytes, self.current_bytes)

    def get(self, value_id: str) -> Any:
        return self._values[value_id]

    def consume(self, value_id: str) -> None:
        if value_id not in self._remaining_consumers:
            return
        self._remaining_consumers[value_id] -= 1
        if self._remaining_consumers[value_id] <= 0:
            self._values.pop(value_id, None)

    @property
    def current_bytes(self) -> int:
        return sum(_tensor_bytes(value) for value in self._values.values())


class LocalDAGRuntime:
    def __init__(
        self,
        runtime_plan: RuntimePlan,
        partitions: Mapping[str, RuntimePartition],
    ) -> None:
        self.runtime_plan = runtime_plan
        self.partitions = dict(partitions)
        self.value_store = ValueStore(_consumer_counts(tuple(partitions.values())))

    def forward(self, inputs: Mapping[str, Any]) -> tuple[TensorMap, RuntimeExecutionEvidence]:
        for value_id, value in inputs.items():
            self.value_store.put(value_id, value)
        completed: set[str] = set()
        sequence: list[str] = []
        queue = deque(sorted(self.partitions))

        while queue:
            partition_id = queue.popleft()
            if partition_id in completed:
                continue
            partition = self.partitions[partition_id]
            if not all(
                value_id in self.value_store._values
                for value_id in partition.spec.input_value_ids
            ):
                queue.append(partition_id)
                if not any(
                    other not in completed
                    and all(
                        value_id in self.value_store._values
                        for value_id in self.partitions[other].spec.input_value_ids
                    )
                    for other in queue
                ):
                    raise RuntimeError("DAG runtime has no ready partition")
                continue
            outputs = partition.executable(
                {
                    value_id: self.value_store.get(value_id)
                    for value_id in partition.spec.input_value_ids
                }
            )
            for value_id, value in outputs.items():
                self.value_store.put(value_id, value)
            for value_id in partition.spec.input_value_ids:
                self.value_store.consume(value_id)
            completed.add(partition_id)
            sequence.append(partition_id)

        final_outputs = {
            value_id: self.value_store.get(value_id)
            for partition in self.partitions.values()
            for value_id in partition.spec.output_value_ids
            if value_id in self.value_store._values
        }
        return final_outputs, RuntimeExecutionEvidence(
            executed_partition_sequence=tuple(sequence),
            local_edges_used=sum(
                1 for edge in self.runtime_plan.edges if edge.edge_kind is EdgeKind.LOCAL
            ),
            remote_edges_used=sum(
                1 for edge in self.runtime_plan.edges if edge.edge_kind is EdgeKind.REMOTE
            ),
            peak_value_store_bytes=self.value_store.peak_bytes,
            optimizer_step_completed=False,
        )

    def optimizer_step(
        self,
        optimizers: Mapping[str, Any],
        evidence: RuntimeExecutionEvidence,
    ) -> RuntimeExecutionEvidence:
        seen: set[int] = set()
        for optimizer in optimizers.values():
            params = [
                parameter
                for group in optimizer.param_groups
                for parameter in group["params"]
                if id(parameter) not in seen
            ]
            seen.update(id(parameter) for parameter in params)
            if params:
                optimizer.step()
        return RuntimeExecutionEvidence(
            evidence.executed_partition_sequence,
            evidence.local_edges_used,
            evidence.remote_edges_used,
            evidence.peak_value_store_bytes,
            optimizer_step_completed=True,
        )


def compile_runtime_plan(
    graph: CanonicalGraphIR,
    logical: LogicalPartitionPlan,
    placement: PlacementPlan,
) -> RuntimePlan:
    partition_by_id = {partition.partition_id: partition for partition in logical.partitions}
    partition_by_node = {
        node_id: partition.partition_id
        for partition in logical.partitions
        for node_id in partition.node_ids
    }
    placement_by_partition = {
        placed.partition_id: placed for placed in placement.placements
    }
    worker_groups: dict[tuple[str, int, str], list[LogicalPartitionSpec]] = {}
    for placed in placement.placements:
        partition = partition_by_id[placed.partition_id]
        worker_groups.setdefault(
            (placed.worker_id, placed.gpu_index, placed.gpu_id),
            [],
        ).append(partition)

    ownership = WorkerOwnershipPlan(
        tuple(
            WorkerOwnershipSpec(
                worker_id=worker_id,
                gpu_index=gpu_index,
                gpu_id=gpu_id,
                owned_partitions=tuple(partition.partition_id for partition in partitions),
                local_parameter_ids=tuple(
                    sorted({pid for partition in partitions for pid in partition.parameter_ids})
                ),
                local_buffer_ids=tuple(
                    sorted({bid for partition in partitions for bid in partition.buffer_ids})
                ),
            )
            for (worker_id, gpu_index, gpu_id), partitions in sorted(worker_groups.items())
        )
    )
    edges: list[RuntimeEdgeSpec] = []
    for edge in graph.edges:
        source_partition = partition_by_node.get(edge.source_node_id)
        target_partition = partition_by_node.get(edge.target_node_id)
        if (
            source_partition is None
            or target_partition is None
            or source_partition == target_partition
        ):
            continue
        source = placement_by_partition[source_partition]
        target = placement_by_partition[target_partition]
        same_device = source.worker_id == target.worker_id and source.gpu_index == target.gpu_index
        edges.append(
            RuntimeEdgeSpec(
                producer_partition=source_partition,
                consumer_partition=target_partition,
                value_id=edge.value_id,
                edge_kind=EdgeKind.LOCAL if same_device else EdgeKind.REMOTE,
                producer_worker_id=source.worker_id,
                consumer_worker_id=target.worker_id,
                producer_gpu_id=source.gpu_id,
                consumer_gpu_id=target.gpu_id,
            )
        )
    return RuntimePlan(
        graph_fingerprint=graph.graph_fingerprint,
        ownership=ownership,
        edges=tuple(edges),
    )


def _consumer_counts(partitions: tuple[RuntimePartition, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for partition in partitions:
        for value_id in partition.spec.input_value_ids:
            counts[value_id] = counts.get(value_id, 0) + 1
    for partition in partitions:
        for value_id in partition.spec.output_value_ids:
            counts.setdefault(value_id, 0)
    return counts


def _tensor_bytes(value: Any) -> int:
    numel = getattr(value, "numel", None)
    element_size = getattr(value, "element_size", None)
    if callable(numel) and callable(element_size):
        return int(numel() * element_size())
    return 0
