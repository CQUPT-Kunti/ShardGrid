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
    if len(fx_nodes) != len(graph.nodes):
        raise ValueError("canonical graph and backend graph node counts differ")

    fx_by_node_id = {
        node.node_id: fx_node for node, fx_node in zip(graph.nodes, fx_nodes, strict=True)
    }
    value_by_fx = {
        fx_node: node.output_value_ids[0]
        for node, fx_node in zip(graph.nodes, fx_nodes, strict=True)
        if node.output_value_ids
    }
    producer_by_value = {
        node.output_value_ids[0]: fx_node
        for node, fx_node in zip(graph.nodes, fx_nodes, strict=True)
        if node.output_value_ids
    }
    owned = set(partition.node_ids)
    missing = owned - set(fx_by_node_id)
    if missing:
        raise ValueError(f"partition references unknown node ids: {sorted(missing)!r}")

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
            raise ValueError(f"cannot map external FX node {node.name!r} to value id")
        return placeholder(value_id)

    for node_spec, fx_node in zip(graph.nodes, fx_nodes, strict=True):
        if node_spec.node_id not in owned:
            continue
        if fx_node.op == "placeholder":
            value_id = node_spec.output_value_ids[0]
            env[fx_node] = placeholder(value_id)
            continue
        env[fx_node] = new_graph.node_copy(fx_node, lookup)

    outputs: list[Node] = []
    for value_id in partition.output_value_ids:
        producer = producer_by_value.get(value_id)
        if producer is None:
            raise ValueError(f"partition output {value_id!r} has no producer")
        outputs.append(env[producer] if producer in env else placeholder(value_id))
    if not outputs:
        raise ValueError(f"partition {partition.partition_id!r} has no outputs")
    new_graph.output(outputs[0] if len(outputs) == 1 else tuple(outputs))
    new_graph.lint()
    return ExtractedPartitionGraph(
        GraphModule(backend_graph, new_graph),
        input_value_ids=tuple(placeholders),
        output_value_ids=tuple(partition.output_value_ids),
    )
