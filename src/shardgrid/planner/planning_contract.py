"""Stable DAG planning contract decoupled from model/runtime code."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from shardgrid.planner.generic_graph import CanonicalGraphIR

CONTRACT_SCHEMA_VERSION = "shardgrid.dag_planning.v1"


@dataclass(frozen=True)
class RuntimeCapabilities:
    supports_linear_pipeline: bool = True
    supports_dag_execution: bool = False
    supports_multiple_partitions_per_device: bool = False
    supports_multi_consumer_value: bool = True
    supports_pytree_io: bool = True
    supports_shared_parameter: bool = False
    supports_non_contiguous_placement: bool = False


@dataclass(frozen=True)
class PlanningConstraints:
    max_logical_partitions: int | None = None
    max_partition_candidates: int = 32
    max_placement_candidates_per_partition: int = 16
    max_total_plan_candidates: int = 128
    top_k_plans: int = 8
    beam_width: int = 8
    safety_margin_bytes: int = 0


@dataclass(frozen=True)
class GPUResourceSpec:
    gpu_id: str
    worker_id: str
    gpu_index: int
    total_memory_bytes: int
    free_memory_bytes: int
    utilization: float = 0.0
    health: str = "healthy"
    gpu_model: str | None = None


@dataclass(frozen=True)
class ResourceSnapshot:
    gpus: tuple[GPUResourceSpec, ...]
    schema_version: str = CONTRACT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "gpus": [gpu.__dict__ for gpu in self.gpus],
        }


@dataclass(frozen=True)
class LogicalPartitionSpec:
    partition_id: str
    node_ids: tuple[str, ...]
    input_value_ids: tuple[str, ...]
    output_value_ids: tuple[str, ...]
    parameter_ids: tuple[str, ...]
    buffer_ids: tuple[str, ...]
    estimated_compute: int
    estimated_memory: int
    boundary_edges: tuple[str, ...]
    owned_state_ids: tuple[str, ...] = ()
    read_only_state_ids: tuple[str, ...] = ()
    estimated_transfer_bytes: int = 0
    validation_evidence: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "partition_id": self.partition_id,
            "node_ids": list(self.node_ids),
            "input_value_ids": list(self.input_value_ids),
            "output_value_ids": list(self.output_value_ids),
            "parameter_ids": list(self.parameter_ids),
            "buffer_ids": list(self.buffer_ids),
            "estimated_compute": self.estimated_compute,
            "estimated_memory": self.estimated_memory,
            "boundary_edges": list(self.boundary_edges),
            "owned_state_ids": list(self.owned_state_ids),
            "read_only_state_ids": list(self.read_only_state_ids),
            "estimated_transfer_bytes": self.estimated_transfer_bytes,
            "validation_evidence": dict(self.validation_evidence or {}),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LogicalPartitionSpec":
        return cls(
            partition_id=str(data["partition_id"]),
            node_ids=tuple(str(item) for item in data.get("node_ids", ())),
            input_value_ids=tuple(str(item) for item in data.get("input_value_ids", ())),
            output_value_ids=tuple(str(item) for item in data.get("output_value_ids", ())),
            parameter_ids=tuple(str(item) for item in data.get("parameter_ids", ())),
            buffer_ids=tuple(str(item) for item in data.get("buffer_ids", ())),
            estimated_compute=int(data.get("estimated_compute", 0)),
            estimated_memory=int(data.get("estimated_memory", 0)),
            boundary_edges=tuple(str(item) for item in data.get("boundary_edges", ())),
            owned_state_ids=tuple(str(item) for item in data.get("owned_state_ids", ())),
            read_only_state_ids=tuple(
                str(item) for item in data.get("read_only_state_ids", ())
            ),
            estimated_transfer_bytes=int(data.get("estimated_transfer_bytes", 0)),
            validation_evidence=(
                None
                if data.get("validation_evidence") is None
                else dict(data.get("validation_evidence", {}))
            ),
        )


@dataclass(frozen=True)
class LogicalPartitionPlan:
    graph_fingerprint: str
    partitions: tuple[LogicalPartitionSpec, ...]
    schema_version: str = CONTRACT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "graph_fingerprint": self.graph_fingerprint,
            "partitions": [partition.to_dict() for partition in self.partitions],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LogicalPartitionPlan":
        return cls(
            graph_fingerprint=str(data["graph_fingerprint"]),
            partitions=tuple(
                LogicalPartitionSpec.from_dict(item)
                for item in data.get("partitions", ())
            ),
            schema_version=str(data.get("schema_version", CONTRACT_SCHEMA_VERSION)),
        )


@dataclass(frozen=True)
class PlacementSpec:
    partition_id: str
    gpu_id: str
    worker_id: str
    gpu_index: int


@dataclass(frozen=True)
class PlacementPlan:
    graph_fingerprint: str
    selected_gpu_count: int
    placements: tuple[PlacementSpec, ...]
    schema_version: str = CONTRACT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "graph_fingerprint": self.graph_fingerprint,
            "selected_gpu_count": self.selected_gpu_count,
            "placements": [placement.__dict__ for placement in self.placements],
        }


@dataclass(frozen=True)
class PlanningResult:
    logical_partition_plan: LogicalPartitionPlan | None
    placement_plan: PlacementPlan | None
    estimated_cost: Mapping[str, int]
    diagnostics: tuple[str, ...] = ()
    plan_candidates: tuple["PlanCandidate", ...] = ()
    search_diagnostics: Mapping[str, int] | None = None
    schema_version: str = CONTRACT_SCHEMA_VERSION


@dataclass(frozen=True)
class PlanCandidate:
    logical_partition_plan: LogicalPartitionPlan
    placement_plan: PlacementPlan
    estimated_cost: Mapping[str, int]
    score: int


@dataclass(frozen=True)
class PlanValidationResult:
    valid: bool
    diagnostics: tuple[str, ...] = ()
    fingerprint_inputs: Mapping[str, Any] | None = None


def plan(
    graph: CanonicalGraphIR,
    resources: ResourceSnapshot,
    constraints: PlanningConstraints,
    capabilities: RuntimeCapabilities,
) -> PlanningResult:
    if graph.shared_parameter_ids and not capabilities.supports_shared_parameter:
        return PlanningResult(
            None,
            None,
            {},
            diagnostics=("AUTO_PARTITION_UNSUPPORTED_SHARED_PARAMETER",),
        )
    if not resources.gpus:
        return PlanningResult(None, None, {}, diagnostics=("NO_AVAILABLE_GPUS",))
    top: list[PlanCandidate] = []
    partition_candidates = generate_logical_partition_candidates(
        graph,
        resources,
        constraints,
    )
    validation_failures: list[str] = []
    evaluated_placements = 0
    evaluated_plans = 0
    for logical in partition_candidates:
        placements = generate_placement_candidates(
            graph,
            logical,
            resources,
            constraints,
            capabilities,
        )
        evaluated_placements += len(placements)
        for placement in placements:
            validation = validate_final_plan(graph, logical, placement)
            if not validation.valid:
                validation_failures.extend(validation.diagnostics[1:])
                continue
            cost = _estimate_cost(graph, logical, placement)
            candidate = PlanCandidate(
                logical_partition_plan=logical,
                placement_plan=placement,
                estimated_cost=cost,
                score=_score_plan(logical, placement, cost, resources),
            )
            evaluated_plans += 1
            top.append(candidate)
            top.sort(key=lambda item: item.score)
            del candidate
            if len(top) > constraints.top_k_plans:
                top.pop()
            if evaluated_plans >= constraints.max_total_plan_candidates:
                break
        del placements
        if evaluated_plans >= constraints.max_total_plan_candidates:
            break
    if not top:
        if validation_failures:
            return PlanningResult(
                partition_candidates[0] if partition_candidates else None,
                None,
                {},
                diagnostics=(
                    "PLAN_VALIDATION_FAILURE",
                    *tuple(sorted(set(validation_failures))),
                ),
                search_diagnostics={
                    "partition_candidates": len(partition_candidates),
                    "placement_candidates": evaluated_placements,
                    "evaluated_plans": evaluated_plans,
                    "search_budget": constraints.max_total_plan_candidates,
                },
            )
        return PlanningResult(
            partition_candidates[0] if partition_candidates else None,
            None,
            {},
            diagnostics=("PLACEMENT_INFEASIBLE",),
            search_diagnostics={
                "partition_candidates": len(partition_candidates),
                "placement_candidates": evaluated_placements,
                "evaluated_plans": evaluated_plans,
                "search_budget": constraints.max_total_plan_candidates,
            },
        )
    selected = top[0]
    return PlanningResult(
        selected.logical_partition_plan,
        selected.placement_plan,
        estimated_cost=selected.estimated_cost,
        plan_candidates=tuple(top),
        search_diagnostics={
            "partition_candidates": len(partition_candidates),
            "placement_candidates": evaluated_placements,
            "evaluated_plans": evaluated_plans,
            "search_budget": constraints.max_total_plan_candidates,
        },
    )


def validate_final_plan(
    graph: CanonicalGraphIR,
    logical: LogicalPartitionPlan,
    placement: PlacementPlan,
) -> PlanValidationResult:
    diagnostics: list[str] = []
    if graph.graph_fingerprint != logical.graph_fingerprint:
        diagnostics.append("graph_logical_fingerprint_mismatch")
    if graph.graph_fingerprint != placement.graph_fingerprint:
        diagnostics.append("graph_placement_fingerprint_mismatch")

    diagnostics.extend(_logical_node_violations(graph, logical))
    diagnostics.extend(_logical_state_violations(graph, logical))
    diagnostics.extend(_logical_boundary_violations(graph, logical))
    diagnostics.extend(_placement_reference_violations(logical, placement))

    fingerprint_inputs = final_plan_fingerprint_inputs(graph, logical, placement)
    try:
        json.dumps(fingerprint_inputs, sort_keys=True)
    except TypeError:
        diagnostics.append("plan_fingerprint_inputs_not_serializable")

    if diagnostics:
        return PlanValidationResult(
            False,
            diagnostics=("PLAN_VALIDATION_FAILURE", *tuple(sorted(set(diagnostics)))),
            fingerprint_inputs=fingerprint_inputs,
        )
    return PlanValidationResult(True, fingerprint_inputs=fingerprint_inputs)


def final_plan_fingerprint_inputs(
    graph: CanonicalGraphIR,
    logical: LogicalPartitionPlan,
    placement: PlacementPlan,
) -> dict[str, Any]:
    return {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "graph": {
            "fingerprint": graph.graph_fingerprint,
            "nodes": [
                {
                    "node_id": node.node_id,
                    "input_value_ids": list(node.input_value_ids),
                    "output_value_ids": list(node.output_value_ids),
                    "parameter_ids": list(node.parameter_ids),
                    "buffer_ids": list(node.buffer_ids),
                }
                for node in graph.nodes
            ],
            "states": [asdict(state) for state in graph.states],
        },
        "logical": logical.to_dict(),
        "placement": placement.to_dict(),
    }


def _logical_node_violations(
    graph: CanonicalGraphIR,
    logical: LogicalPartitionPlan,
) -> tuple[str, ...]:
    expected = {node.node_id for node in graph.nodes if node.op_kind != "output"}
    counts = Counter(
        node_id for partition in logical.partitions for node_id in partition.node_ids
    )
    diagnostics: list[str] = []
    if any(count > 1 for count in counts.values()):
        diagnostics.append("duplicate_partition_node")
    if expected - set(counts):
        diagnostics.append("missing_partition_node")
    if set(counts) - expected:
        diagnostics.append("unknown_partition_node")
    return tuple(diagnostics)


def _logical_state_violations(
    graph: CanonicalGraphIR,
    logical: LogicalPartitionPlan,
) -> tuple[str, ...]:
    required = (
        {state.canonical_state_id for state in graph.states}
        if graph.states
        else {
            state_id
            for node in graph.nodes
            for state_id in node.parameter_ids + node.buffer_ids
        }
    )
    owner_counts = Counter(
        state_id
        for partition in logical.partitions
        for state_id in partition.owned_state_ids
    )
    diagnostics: list[str] = []
    if any(count > 1 for state_id, count in owner_counts.items() if state_id in required):
        diagnostics.append("duplicate_state_owner")
    if required - set(owner_counts):
        diagnostics.append("missing_state_owner")
    if set(owner_counts) - required:
        diagnostics.append("unknown_state_owner")
    if any(
        set(partition.owned_state_ids) & set(partition.read_only_state_ids)
        for partition in logical.partitions
    ):
        diagnostics.append("state_read_owner_conflict")
    return tuple(diagnostics)


def _logical_boundary_violations(
    graph: CanonicalGraphIR,
    logical: LogicalPartitionPlan,
) -> tuple[str, ...]:
    partition_by_node = {
        node_id: partition.partition_id
        for partition in logical.partitions
        for node_id in partition.node_ids
    }
    partition_by_id = {partition.partition_id: partition for partition in logical.partitions}
    for edge in graph.edges:
        source_partition_id = partition_by_node.get(edge.source_node_id)
        target_partition_id = partition_by_node.get(edge.target_node_id)
        if (
            source_partition_id is None
            or target_partition_id is None
            or source_partition_id == target_partition_id
        ):
            continue
        source = partition_by_id[source_partition_id]
        target = partition_by_id[target_partition_id]
        edge_id = edge.edge_id or f"{edge.source_node_id}->{edge.target_node_id}:{edge.value_id}"
        if (
            edge.value_id not in source.output_value_ids
            or edge.value_id not in target.input_value_ids
            or edge_id not in source.boundary_edges
        ):
            return ("missing_boundary_value",)
    return ()


def _placement_reference_violations(
    logical: LogicalPartitionPlan,
    placement: PlacementPlan,
) -> tuple[str, ...]:
    expected = {partition.partition_id for partition in logical.partitions}
    counts = Counter(placed.partition_id for placed in placement.placements)
    diagnostics: list[str] = []
    if any(count > 1 for count in counts.values()):
        diagnostics.append("duplicate_placement_partition")
    if expected - set(counts):
        diagnostics.append("missing_placement_partition")
    if set(counts) - expected:
        diagnostics.append("unknown_placement_partition")
    if placement.selected_gpu_count != len({placed.gpu_id for placed in placement.placements}):
        diagnostics.append("selected_gpu_count_mismatch")
    return tuple(diagnostics)


def generate_logical_partition_candidates(
    graph: CanonicalGraphIR,
    resources: ResourceSnapshot,
    constraints: PlanningConstraints,
) -> tuple[LogicalPartitionPlan, ...]:
    executable = tuple(node for node in graph.nodes if node.op_kind != "output")
    if not executable:
        return ()
    free_memory = tuple(
        sorted(
            (
                gpu.free_memory_bytes
                for gpu in resources.gpus
                if gpu.health == "healthy" and gpu.free_memory_bytes > 0
            ),
            reverse=True,
        )
    )
    if not free_memory:
        return ()
    total_memory = sum(_node_memory(node) for node in executable)
    minimum = 1 if total_memory <= free_memory[0] else _ceil_div(total_memory, free_memory[0])
    upper = min(len(executable), constraints.max_logical_partitions or len(executable))
    counts = range(max(1, minimum), upper + 1)
    plans: list[LogicalPartitionPlan] = []
    seen: set[tuple[tuple[str, ...], ...]] = set()
    for count in counts:
        targets = _partition_targets(total_memory, count, free_memory)
        logical = build_logical_partition_plan(
            graph,
            max_partitions=count,
            target_partition_memory=targets,
        )
        key = tuple(partition.node_ids for partition in logical.partitions)
        if key not in seen:
            seen.add(key)
            plans.append(logical)
        if len(plans) >= constraints.max_partition_candidates:
            break
    return tuple(plans)


def build_logical_partition_plan(
    graph: CanonicalGraphIR,
    *,
    max_partitions: int,
    target_partition_memory: Sequence[int] | None = None,
) -> LogicalPartitionPlan:
    executable = tuple(node for node in graph.nodes if node.op_kind != "output")
    partition_count = max(1, min(max_partitions, len(executable)))
    chunks = (
        _capacity_chunks(executable, target_partition_memory)
        if target_partition_memory
        else _chunks(executable, partition_count)
    )
    value_producer = {
        value.value_id: value.producer_node_id
        for value in graph.values
        if value.producer_node_id is not None
    }
    partitions: list[LogicalPartitionSpec] = []
    for index, chunk in enumerate(chunks):
        node_ids = tuple(node.node_id for node in chunk)
        node_set = set(node_ids)
        input_values = sorted(
            {
                value_id
                for node in chunk
                for value_id in node.input_value_ids
                if value_producer.get(value_id) not in node_set
            }
        )
        output_values = sorted(
            {
                value_id
                for node in chunk
                for value_id in node.output_value_ids
                if any(
                    edge.value_id == value_id and edge.target_node_id not in node_set
                    for edge in graph.edges
                )
                or value_id in graph.output_value_ids
            }
        )
        boundary_edges = sorted(
            edge.edge_id or f"{edge.source_node_id}->{edge.target_node_id}:{edge.value_id}"
            for edge in graph.edges
            if edge.source_node_id in node_set and edge.target_node_id not in node_set
        )
        owned_state_ids, read_only_state_ids = _partition_state_ids(graph, node_set, chunk)
        partitions.append(
            LogicalPartitionSpec(
                partition_id=f"P{index}",
                node_ids=node_ids,
                input_value_ids=tuple(input_values),
                output_value_ids=tuple(output_values),
                parameter_ids=tuple(sorted({pid for node in chunk for pid in node.parameter_ids})),
                buffer_ids=tuple(sorted({bid for node in chunk for bid in node.buffer_ids})),
                estimated_compute=sum(node.estimated_compute_cost for node in chunk),
                estimated_memory=sum(_node_memory(node) for node in chunk),
                boundary_edges=tuple(boundary_edges),
                owned_state_ids=owned_state_ids,
                read_only_state_ids=read_only_state_ids,
                estimated_transfer_bytes=sum(
                    edge.communication_weight
                    for edge in graph.edges
                    if edge.source_node_id in node_set
                    and edge.target_node_id not in node_set
                ),
                validation_evidence={
                    "node_count": len(node_ids),
                    "input_value_count": len(input_values),
                    "output_value_count": len(output_values),
                    "boundary_edge_count": len(boundary_edges),
                },
            )
        )
    return LogicalPartitionPlan(
        graph_fingerprint=graph.graph_fingerprint,
        partitions=tuple(partitions),
    )


def _partition_state_ids(
    graph: CanonicalGraphIR,
    node_ids: set[str],
    nodes: Sequence[Any],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if not graph.states:
        legacy = sorted(
            {state_id for node in nodes for state_id in node.parameter_ids + node.buffer_ids}
        )
        return tuple(legacy), ()

    owned: set[str] = set()
    read_only: set[str] = set()
    for state in graph.states:
        use_nodes = set(state.use_node_ids)
        if not use_nodes & node_ids:
            continue
        owner_nodes = set(state.owner_node_ids)
        if owner_nodes & node_ids:
            owned.add(state.canonical_state_id)
        else:
            read_only.add(state.canonical_state_id)
    return tuple(sorted(owned)), tuple(sorted(read_only - owned))


def build_placement_plan(
    graph: CanonicalGraphIR,
    logical: LogicalPartitionPlan,
    resources: ResourceSnapshot,
    constraints: PlanningConstraints,
    capabilities: RuntimeCapabilities,
) -> PlacementPlan | None:
    candidates = generate_placement_candidates(
        graph,
        logical,
        resources,
        constraints,
        capabilities,
    )
    return candidates[0] if candidates else None


def generate_placement_candidates(
    graph: CanonicalGraphIR,
    logical: LogicalPartitionPlan,
    resources: ResourceSnapshot,
    constraints: PlanningConstraints,
    capabilities: RuntimeCapabilities,
) -> tuple[PlacementPlan, ...]:
    healthy = tuple(
        sorted(
            (gpu for gpu in resources.gpus if gpu.health == "healthy"),
            key=lambda gpu: (-gpu.free_memory_bytes, gpu.worker_id, gpu.gpu_index),
        )
    )
    if not healthy:
        return ()
    if (
        not capabilities.supports_multiple_partitions_per_device
        and len(healthy) < len(logical.partitions)
    ):
        return ()

    states: list[tuple[int, dict[str, int], tuple[PlacementSpec, ...], frozenset[str]]] = [
        (0, {gpu.gpu_id: gpu.free_memory_bytes for gpu in healthy}, (), frozenset())
    ]
    partitions = tuple(
        sorted(logical.partitions, key=lambda item: (-item.estimated_memory, item.partition_id))
    )
    for partition in partitions:
        next_states: list[
            tuple[int, dict[str, int], tuple[PlacementSpec, ...], frozenset[str]]
        ] = []
        required = partition.estimated_memory + constraints.safety_margin_bytes
        for score, remaining, placed, used in states:
            for gpu in _placement_gpu_choices(required, healthy, remaining):
                if (
                    not capabilities.supports_multiple_partitions_per_device
                    and gpu.gpu_id in used
                ):
                    continue
                next_remaining = dict(remaining)
                next_remaining[gpu.gpu_id] -= required
                next_used = used | {gpu.gpu_id}
                next_placed = placed + (
                    PlacementSpec(
                        partition_id=partition.partition_id,
                        gpu_id=gpu.gpu_id,
                        worker_id=gpu.worker_id,
                        gpu_index=gpu.gpu_index,
                    ),
                )
                slack = next_remaining[gpu.gpu_id]
                next_states.append((score + slack, next_remaining, next_placed, next_used))
        next_states.sort(key=lambda item: (item[0], tuple(sorted(item[3]))))
        states = next_states[: constraints.beam_width]
        del next_states
        if not states:
            return ()
    plans = [
        PlacementPlan(
            graph_fingerprint=graph.graph_fingerprint,
            selected_gpu_count=len(used),
            placements=tuple(sorted(placed, key=lambda item: item.partition_id)),
        )
        for _score, _remaining, placed, used in states
    ]
    return tuple(plans[: constraints.max_placement_candidates_per_partition])


def _chunks(items: Sequence[Any], count: int) -> tuple[tuple[Any, ...], ...]:
    size = max(1, (len(items) + count - 1) // count)
    return tuple(
        tuple(items[start : start + size])
        for start in range(0, len(items), size)
    )


def _capacity_chunks(
    items: Sequence[Any],
    targets: Sequence[int],
) -> tuple[tuple[Any, ...], ...]:
    chunks: list[tuple[Any, ...]] = []
    current: list[Any] = []
    current_memory = 0
    target_index = 0
    for index, item in enumerate(items):
        current.append(item)
        current_memory += _node_memory(item)
        remaining_items = len(items) - index - 1
        remaining_targets = len(targets) - target_index - 1
        if (
            target_index < len(targets) - 1
            and current_memory >= targets[target_index]
            and remaining_items >= remaining_targets
        ):
            chunks.append(tuple(current))
            current = []
            current_memory = 0
            target_index += 1
    if current:
        chunks.append(tuple(current))
    return tuple(chunk for chunk in chunks if chunk)


def _best_fit_gpu(
    required: int,
    gpus: Sequence[GPUResourceSpec],
    remaining: Mapping[str, int],
) -> GPUResourceSpec | None:
    candidates = [gpu for gpu in gpus if remaining[gpu.gpu_id] >= required]
    if not candidates:
        return None
    return min(candidates, key=lambda gpu: (remaining[gpu.gpu_id] - required, gpu.gpu_id))


def _placement_gpu_choices(
    required: int,
    gpus: Sequence[GPUResourceSpec],
    remaining: Mapping[str, int],
) -> tuple[GPUResourceSpec, ...]:
    candidates = [gpu for gpu in gpus if remaining[gpu.gpu_id] >= required]
    return tuple(
        sorted(
            candidates,
            key=lambda gpu: (remaining[gpu.gpu_id] - required, -remaining[gpu.gpu_id], gpu.gpu_id),
        )
    )


def _partition_targets(
    total_memory: int,
    count: int,
    free_memory: Sequence[int],
) -> tuple[int, ...]:
    capacities = tuple(free_memory[index % len(free_memory)] for index in range(count))
    total_capacity = sum(capacities) or count
    return tuple(max(1, total_memory * capacity // total_capacity) for capacity in capacities)


def _node_memory(node: Any) -> int:
    return max(
        1,
        int(
            getattr(node, "estimated_peak_memory_contribution", 0)
            or getattr(node, "parameter_bytes", 0)
            or getattr(node, "activation_bytes", 0)
            or 1
        ),
    )


def _ceil_div(value: int, divisor: int) -> int:
    return max(1, (value + divisor - 1) // divisor)


def _score_plan(
    logical: LogicalPartitionPlan,
    placement: PlacementPlan,
    cost: Mapping[str, int],
    resources: ResourceSnapshot,
) -> int:
    used = {placed.gpu_id for placed in placement.placements}
    placed_memory = {
        partition.partition_id: partition.estimated_memory
        for partition in logical.partitions
    }
    used_memory: dict[str, int] = {}
    for placed in placement.placements:
        used_memory[placed.gpu_id] = (
            used_memory.get(placed.gpu_id, 0)
            + placed_memory[placed.partition_id]
        )
    slack = 0
    for gpu in resources.gpus:
        if gpu.gpu_id in used:
            slack += max(0, gpu.free_memory_bytes - used_memory.get(gpu.gpu_id, 0))
    return (
        slack
        + len(logical.partitions) * 1_000
        + int(cost.get("estimated_communication_cost", 0))
    )


def _estimate_cost(
    graph: CanonicalGraphIR,
    logical: LogicalPartitionPlan,
    placement: PlacementPlan,
) -> dict[str, int]:
    partition_by_node = {
        node_id: partition.partition_id
        for partition in logical.partitions
        for node_id in partition.node_ids
    }
    gpu_by_partition = {
        placed.partition_id: placed.gpu_id for placed in placement.placements
    }
    cross_partition = 0
    cross_gpu_forward = 0
    cross_gpu_backward = 0
    for edge in graph.edges:
        source_partition = partition_by_node.get(edge.source_node_id)
        target_partition = partition_by_node.get(edge.target_node_id)
        if (
            source_partition is None
            or target_partition is None
            or source_partition == target_partition
        ):
            continue
        cross_partition += edge.forward_transfer_bytes
        if gpu_by_partition.get(source_partition) == gpu_by_partition.get(target_partition):
            continue
        cross_gpu_forward += edge.forward_transfer_bytes
        cross_gpu_backward += edge.backward_transfer_bytes
    return {
        "total_graph_edge_bytes": sum(edge.forward_transfer_bytes for edge in graph.edges),
        "cross_partition_bytes": cross_partition,
        "cross_gpu_forward_bytes": cross_gpu_forward,
        "cross_gpu_backward_bytes": cross_gpu_backward,
        "estimated_communication_cost": cross_gpu_forward + cross_gpu_backward,
    }
