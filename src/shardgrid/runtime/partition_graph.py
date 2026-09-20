"""FX partition extraction for generic DAG runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from shardgrid.planner.generic_graph import CanonicalGraphIR
from shardgrid.planner.planning_contract import LogicalPartitionSpec


@dataclass(frozen=True)
class ExtractedPartitionGraph:
    graph_module: Any
    input_value_ids: tuple[str, ...]
    output_value_ids: tuple[str, ...]

    def __call__(self, values: Mapping[str, Any]) -> dict[str, Any]:
        result = self.graph_module(*(values[value_id] for value_id in self.input_value_ids))
        outputs = result if isinstance(result, tuple) else (result,)
        return dict(zip(self.output_value_ids, outputs, strict=True))


def extract_partition_graph(
    graph: CanonicalGraphIR,
    backend_graph: Any,
    partition: LogicalPartitionSpec,
) -> ExtractedPartitionGraph:
    """Extract an executable FX GraphModule for one logical partition."""
    from torch.fx import Graph, GraphModule, Node

    fx_nodes = tuple(backend_graph.graph.nodes)
    fx_by_node_id = _map_fx_nodes_by_canonical_id(graph, fx_nodes)
    value_by_fx = {}
    producer_by_value = {}
    for node in graph.nodes:
        fx_node = fx_by_node_id[node.node_id]
        for value_id in node.output_value_ids:
            value_by_fx[fx_node] = value_id
            producer_by_value[value_id] = fx_node
    owned = set(partition.node_ids)
    missing = owned - set(fx_by_node_id)
    if missing:
        raise _plan_validation_failure(
            f"partition references unknown node ids: {sorted(missing)!r}"
        )

    known_values = {
        value_id for node in graph.nodes for value_id in node.output_value_ids
    } | set(graph.input_value_ids)
    unknown_values = (
        set(partition.input_value_ids) | set(partition.output_value_ids)
    ) - known_values
    if unknown_values:
        raise _plan_validation_failure(
            f"partition references unknown value ids: {sorted(unknown_values)!r}"
        )

    new_graph = Graph()
    env: dict[Node, Node] = {}
    placeholders: dict[str, Node] = {}

    def placeholder(value_id: str) -> Node:
        if value_id not in placeholders:
            placeholders[value_id] = new_graph.placeholder(value_id)
        return placeholders[value_id]

    for value_id in partition.input_value_ids:
        placeholder(value_id)

    def lookup(node: Node) -> Node:
        if node in env:
            return env[node]
        value_id = value_by_fx.get(node)
        if value_id is None:
            raise _plan_validation_failure(
                f"cannot map external FX node {node.name!r} to value id"
            )
        return placeholder(value_id)

    for node_spec in graph.nodes:
        if node_spec.node_id not in owned:
            continue
        fx_node = fx_by_node_id[node_spec.node_id]
        if fx_node.op == "placeholder":
            value_id = node_spec.output_value_ids[0]
            env[fx_node] = placeholder(value_id)
            continue
        env[fx_node] = new_graph.node_copy(fx_node, lookup)

    outputs: list[Node] = []
    for value_id in partition.output_value_ids:
        producer = producer_by_value.get(value_id)
        if producer is None:
            raise _plan_validation_failure(f"partition output {value_id!r} has no producer")
        outputs.append(env[producer] if producer in env else placeholder(value_id))
    if not outputs:
        raise _plan_validation_failure(f"partition {partition.partition_id!r} has no outputs")
    new_graph.output(outputs[0] if len(outputs) == 1 else tuple(outputs))
    new_graph.lint()
    return ExtractedPartitionGraph(
        GraphModule(backend_graph, new_graph),
        input_value_ids=tuple(placeholders),
        output_value_ids=tuple(partition.output_value_ids),
    )


def _map_fx_nodes_by_canonical_id(
    graph: CanonicalGraphIR,
    fx_nodes: tuple[Any, ...],
) -> dict[str, Any]:
    from torch.fx import Node

    remaining = list(fx_nodes)
    mapped: dict[str, Node] = {}
    producer_by_value = {
        value.value_id: value.producer_node_id
        for value in graph.values
        if value.producer_node_id is not None
    }
    for node in graph.nodes:
        expected_inputs = {
            mapped[producer]
            for value_id in node.input_value_ids
            for producer in (producer_by_value.get(value_id),)
            if producer in mapped
        }
        matches = [
            fx_node
            for fx_node in remaining
            if _fx_node_matches(node, fx_node)
            and _fx_inputs_match(
                node.input_value_ids,
                expected_inputs,
                _walk_fx_nodes((fx_node.args, fx_node.kwargs)),
            )
        ]
        if len(matches) != 1:
            raise _plan_validation_failure(
                f"backend graph mismatch for canonical node {node.node_id}: "
                f"matched {len(matches)} FX nodes"
            )
        fx_node = matches[0]
        mapped[node.node_id] = fx_node
        remaining.remove(fx_node)
    if remaining:
        raise _plan_validation_failure("backend graph has extra FX nodes")
    return mapped


def _fx_inputs_match(
    input_value_ids: tuple[str, ...],
    expected_inputs: set[Any],
    fx_inputs: tuple[Any, ...],
) -> bool:
    return len(fx_inputs) == len(input_value_ids) and expected_inputs.issubset(set(fx_inputs))


def _fx_node_matches(node_spec: Any, fx_node: Any) -> bool:
    if str(fx_node.op) != node_spec.op_kind or str(fx_node.target) != node_spec.target:
        return False
    if fx_node.op != "call_module":
        return True
    return _fx_module_path(fx_node) == node_spec.module_path


def _fx_module_path(fx_node: Any) -> str | None:
    if fx_node.op == "call_module":
        return str(fx_node.target)
    stack = fx_node.meta.get("nn_module_stack", {})
    if isinstance(stack, Mapping) and stack:
        value = tuple(stack.values())[-1]
        if isinstance(value, tuple) and value:
            return str(value[0])
    return None


def _walk_fx_nodes(value: Any) -> tuple[Any, ...]:
    from torch.fx import Node

    if isinstance(value, Node):
        return (value,)
    if isinstance(value, Mapping):
        return tuple(node for item in value.values() for node in _walk_fx_nodes(item))
    if isinstance(value, (tuple, list)):
        return tuple(node for item in value for node in _walk_fx_nodes(item))
    return ()


def _plan_validation_failure(reason: str) -> ValueError:
    return ValueError(f"PLAN_VALIDATION_FAILURE: {reason}")
