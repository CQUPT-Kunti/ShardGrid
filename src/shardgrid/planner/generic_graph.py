"""Generic graph capture data used by automatic partition planning."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from shardgrid.common.enums import FailureCode

GRAPH_IR_SCHEMA_VERSION = "shardgrid.canonical_graph.v1"


@dataclass(frozen=True)
class GraphValueSpec:
    value_id: str
    producer_node_id: str | None
    consumer_node_ids: tuple[str, ...] = ()
    shape: tuple[int | str, ...] = ()
    dtype: str | None = None
    requires_grad: bool | None = None
    estimated_bytes: int | None = None
    pytree_path: str | None = None


@dataclass(frozen=True)
class GraphNodeSpec:
    node_id: str
    op_kind: str
    target: str
    module_path: str | None
    input_value_ids: tuple[str, ...] = ()
    output_value_ids: tuple[str, ...] = ()
    parameter_paths: tuple[str, ...] = ()
    buffer_paths: tuple[str, ...] = ()
    canonical_target: str | None = None
    parameter_ids: tuple[str, ...] = ()
    buffer_ids: tuple[str, ...] = ()
    estimated_compute_cost: int = 0
    parameter_bytes: int = 0
    activation_bytes: int = 0
    gradient_bytes: int = 0
    optimizer_bytes: int = 0
    temporary_bytes: int = 0
    estimated_peak_memory_contribution: int = 0


@dataclass(frozen=True)
class GraphEdgeSpec:
    source_node_id: str
    target_node_id: str
    value_id: str
    edge_id: str = ""
    forward_transfer_bytes: int = 0
    backward_transfer_bytes: int = 0
    communication_weight: int = 0


@dataclass(frozen=True)
class ParameterUseSpec:
    parameter_id: str
    canonical_path: str
    consumer_nodes: tuple[str, ...]


@dataclass(frozen=True)
class StateObjectSpec:
    canonical_state_id: str
    kind: str
    state_dict_key: str
    shape: tuple[int | str, ...] = ()
    dtype: str | None = None
    requires_grad: bool | None = None
    storage_id: str | None = None
    shared_group_id: str | None = None
    checkpoint_owner_key: str | None = None
    owner_node_ids: tuple[str, ...] = ()
    use_node_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class BoundaryValueSpec:
    value_id: str
    producer_stage: str
    consumer_stages: tuple[str, ...]
    shape: tuple[int | str, ...] = ()
    dtype: str | None = None
    requires_grad: bool | None = None
    estimated_bytes: int | None = None


@dataclass(frozen=True)
class GenericGraphIR:
    nodes: tuple[GraphNodeSpec, ...]
    values: tuple[GraphValueSpec, ...]
    edges: tuple[GraphEdgeSpec, ...]
    input_value_ids: tuple[str, ...]
    output_value_ids: tuple[str, ...]
    parameter_owners: Mapping[str, str]
    capture_backend: str
    schema_version: str = GRAPH_IR_SCHEMA_VERSION
    graph_fingerprint: str = ""
    input_pytree_spec: str = "tuple"
    output_pytree_spec: str = "unknown"
    parameter_uses: tuple[ParameterUseSpec, ...] = ()
    shared_parameter_ids: tuple[str, ...] = ()
    states: tuple[StateObjectSpec, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "capture_backend": self.capture_backend,
            "graph_fingerprint": self.graph_fingerprint,
            "input_pytree_spec": self.input_pytree_spec,
            "output_pytree_spec": self.output_pytree_spec,
            "nodes": [node.__dict__ for node in self.nodes],
            "values": [value.__dict__ for value in self.values],
            "edges": [edge.__dict__ for edge in self.edges],
            "input_value_ids": list(self.input_value_ids),
            "output_value_ids": list(self.output_value_ids),
            "parameter_owners": dict(self.parameter_owners),
            "parameter_uses": [use.__dict__ for use in self.parameter_uses],
            "shared_parameter_ids": list(self.shared_parameter_ids),
            "states": [state.__dict__ for state in self.states],
        }


CanonicalGraphIR = GenericGraphIR


@dataclass(frozen=True)
class GraphCaptureResult:
    canonical_graph: CanonicalGraphIR
    backend_graph: Any
    input_pytree_spec: str
    output_pytree_spec: str
    metadata: Mapping[str, Any]
    diagnostics: tuple[str, ...]
    graph_fingerprint: str


class GraphCaptureUnsupported(ValueError):
    def __init__(
        self,
        code: FailureCode | str,
        message: str,
        *,
        diagnostics: Sequence[str] = (),
        backend: str | None = None,
        fallback_allowed: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = FailureCode.from_value(str(code))
        self.diagnostics = tuple(diagnostics)
        self.backend = backend
        self.fallback_allowed = fallback_allowed


@dataclass(frozen=True)
class ModelFactorySpec:
    model_factory: Any
    sample_input_builder: Any
    config: Mapping[str, Any] | None = None
    use_meta: bool = True


class GraphCaptureAdapter(Protocol):
    backend_name: str

    def capture(
        self,
        model: Any,
        *,
        sample_args: Sequence[Any] = (),
        sample_kwargs: Mapping[str, Any] | None = None,
    ) -> GraphCaptureResult:
        ...

    def capture_factory(self, spec: ModelFactorySpec) -> GraphCaptureResult:
        ...


class FXGraphCaptureAdapter:
    backend_name = "torch.fx.symbolic_trace"

    def capture(
        self,
        model: Any,
        *,
        sample_args: Sequence[Any] = (),
        sample_kwargs: Mapping[str, Any] | None = None,
    ) -> GraphCaptureResult:
        from torch.fx import GraphModule, symbolic_trace
        from torch.fx.passes.shape_prop import ShapeProp

        sample_kwargs = dict(sample_kwargs or {})
        graph_module: GraphModule = symbolic_trace(model)
        if sample_args or sample_kwargs:
            ShapeProp(graph_module).propagate(*sample_args, **sample_kwargs)
        graph = _graph_from_fx(
            model,
            graph_module,
            capture_backend=self.backend_name,
            input_pytree_spec=_pytree_name(sample_args),
            output_pytree_spec="unknown",
        )
        diagnostics = _custom_op_diagnostics(graph_module)
        return GraphCaptureResult(
            canonical_graph=graph,
            backend_graph=graph_module,
            input_pytree_spec=graph.input_pytree_spec,
            output_pytree_spec=graph.output_pytree_spec,
            metadata={
                "control_plane_parameter_real_storage_bytes": _real_parameter_storage_bytes(
                    model
                ),
                "control_plane_full_real_model_materialized": (
                    _real_parameter_storage_bytes(model) > 0
                ),
                "node_count": len(graph.nodes),
                "edge_count": len(graph.edges),
            },
            diagnostics=diagnostics,
            graph_fingerprint=graph.graph_fingerprint,
        )

    def capture_factory(self, spec: ModelFactorySpec) -> GraphCaptureResult:
        import torch

        config = dict(spec.config or {})
        if spec.use_meta:
            with torch.device("meta"):
                model = spec.model_factory(**config)
            sample_args, sample_kwargs = _build_factory_sample(spec, device="meta")
        else:
            model = spec.model_factory(**config)
            sample_args, sample_kwargs = _build_factory_sample(spec, device="cpu")
        result = self.capture(
            model,
            sample_args=sample_args,
            sample_kwargs=sample_kwargs,
        )
        real_bytes = _real_parameter_storage_bytes(model)
        return GraphCaptureResult(
            canonical_graph=result.canonical_graph,
            backend_graph=result.backend_graph,
            input_pytree_spec=result.input_pytree_spec,
            output_pytree_spec=result.output_pytree_spec,
            metadata={
                **dict(result.metadata),
                "control_plane_parameter_real_storage_bytes": real_bytes,
                "control_plane_full_real_model_materialized": real_bytes > 0,
            },
            diagnostics=result.diagnostics,
            graph_fingerprint=result.graph_fingerprint,
        )


class TorchExportGraphCaptureAdapter:
    backend_name = "torch.export"

    def capture(
        self,
        model: Any,
        *,
        sample_args: Sequence[Any] = (),
        sample_kwargs: Mapping[str, Any] | None = None,
    ) -> GraphCaptureResult:
        import torch

        sample_kwargs = dict(sample_kwargs or {})
        try:
            exported = torch.export.export(
                model,
                args=tuple(sample_args),
                kwargs=sample_kwargs,
                strict=False,
            )
        except Exception as exc:
            code = _classify_capture_exception(exc)
            raise GraphCaptureUnsupported(
                code,
                str(exc),
                diagnostics=(f"{code}: {exc}",),
                backend=self.backend_name,
                fallback_allowed=code == "MODEL_CAPTURE_UNSUPPORTED",
            ) from exc
        graph_module = exported.module()
        graph = _graph_from_fx(
            model,
            graph_module,
            capture_backend=self.backend_name,
            input_pytree_spec=_pytree_name(sample_args),
            output_pytree_spec="unknown",
        )
        diagnostics = _custom_op_diagnostics(graph_module)
        if diagnostics:
            raise GraphCaptureUnsupported(
                "CUSTOM_OP_UNSUPPORTED",
                "; ".join(diagnostics),
                diagnostics=diagnostics,
                backend=self.backend_name,
            )
        return GraphCaptureResult(
            canonical_graph=graph,
            backend_graph=exported,
            input_pytree_spec=graph.input_pytree_spec,
            output_pytree_spec=graph.output_pytree_spec,
            metadata={
                "backend": self.backend_name,
                "node_count": len(graph.nodes),
                "edge_count": len(graph.edges),
                "control_plane_parameter_real_storage_bytes": _real_parameter_storage_bytes(
                    model
                ),
                "control_plane_full_real_model_materialized": (
                    _real_parameter_storage_bytes(model) > 0
                ),
            },
            diagnostics=diagnostics,
            graph_fingerprint=graph.graph_fingerprint,
        )

    def capture_factory(self, spec: ModelFactorySpec) -> GraphCaptureResult:
        return FXGraphCaptureAdapter().capture_factory(spec)


def capture_generic_graph_with_backend_fallback(
    model: Any,
    *,
    sample_args: Sequence[Any] = (),
    sample_kwargs: Mapping[str, Any] | None = None,
) -> GraphCaptureResult:
    try:
        return TorchExportGraphCaptureAdapter().capture(
            model,
            sample_args=sample_args,
            sample_kwargs=sample_kwargs,
        )
    except GraphCaptureUnsupported as export_failure:
        if not export_failure.fallback_allowed:
            raise
        try:
            fx_result = FXGraphCaptureAdapter().capture(
                model,
                sample_args=sample_args,
                sample_kwargs=sample_kwargs,
            )
        except Exception as exc:
            code = _classify_capture_exception(exc)
            raise GraphCaptureUnsupported(
                code,
                str(exc),
                diagnostics=(*export_failure.diagnostics, f"{code}: {exc}"),
                backend=FXGraphCaptureAdapter.backend_name,
            ) from exc
        if fx_result.diagnostics:
            raise GraphCaptureUnsupported(
                "CUSTOM_OP_UNSUPPORTED",
                "; ".join(fx_result.diagnostics),
                diagnostics=(*export_failure.diagnostics, *fx_result.diagnostics),
                backend=fx_result.canonical_graph.capture_backend,
            )
        return GraphCaptureResult(
            canonical_graph=fx_result.canonical_graph,
            backend_graph=fx_result.backend_graph,
            input_pytree_spec=fx_result.input_pytree_spec,
            output_pytree_spec=fx_result.output_pytree_spec,
            metadata={
                **dict(fx_result.metadata),
                "export_failure_code": export_failure.code,
                "fallback_from": TorchExportGraphCaptureAdapter.backend_name,
            },
            diagnostics=(
                f"FX_FALLBACK_FROM:{export_failure.code}",
                *export_failure.diagnostics,
                *fx_result.diagnostics,
            ),
            graph_fingerprint=fx_result.graph_fingerprint,
        )


def capture_generic_graph(
    model: Any,
    *,
    sample_args: Sequence[Any] = (),
    sample_kwargs: Mapping[str, Any] | None = None,
) -> GenericGraphIR:
    """Capture a model into value-id based FX graph IR."""
    return FXGraphCaptureAdapter().capture(
        model,
        sample_args=sample_args,
        sample_kwargs=sample_kwargs,
    ).canonical_graph


def _graph_from_fx(
    model: Any,
    graph_module: Any,
    *,
    capture_backend: str,
    input_pytree_spec: str,
    output_pytree_spec: str,
) -> CanonicalGraphIR:
    from torch.fx import Node

    value_by_node: dict[Node, str] = {}
    values: list[GraphValueSpec] = []
    nodes: list[GraphNodeSpec] = []
    edges: list[GraphEdgeSpec] = []
    module_by_path = dict(model.named_modules())
    parameter_ids_by_path, parameter_bytes_by_path = _parameter_schema(model)
    buffer_ids_by_path = _buffer_schema(model)
    value_index = 0
    node_index = 0

    def new_value(producer: str | None, meta: Any) -> str:
        nonlocal value_index
        value_id = f"v{value_index:04d}"
        value_index += 1
        values.append(_value_spec(value_id, producer, meta))
        return value_id

    def walk_fx_nodes(value: Any) -> list[Node]:
        if isinstance(value, Node):
            return [value]
        if isinstance(value, Mapping):
            return [node for item in value.values() for node in walk_fx_nodes(item)]
        if isinstance(value, (tuple, list)):
            return [node for item in value for node in walk_fx_nodes(item)]
        return []

    input_value_ids: list[str] = []
    output_value_ids: list[str] = []
    for fx_node in graph_module.graph.nodes:
        node_id = f"n{node_index:04d}"
        node_index += 1
        input_nodes = walk_fx_nodes((fx_node.args, fx_node.kwargs))
        input_ids = tuple(
            value_by_node[parent] for parent in input_nodes if parent in value_by_node
        )
        module_path = _node_module_path(fx_node)
        parameter_paths = _node_parameter_paths(fx_node, module_by_path)
        buffer_paths = _node_buffer_paths(fx_node, module_by_path)
        if fx_node.op in {"call_function", "call_method"}:
            parameter_paths = _unique(
                parameter_paths
                + tuple(
                    path
                    for parent in input_nodes
                    for path in _get_attr_state_paths(parent, parameter_ids_by_path)
                )
            )
            buffer_paths = _unique(
                buffer_paths
                + tuple(
                    path
                    for parent in input_nodes
                    for path in _get_attr_state_paths(parent, buffer_ids_by_path)
                )
            )
        output_ids: tuple[str, ...]
        if fx_node.op == "output":
            output_ids = ()
            output_value_ids = [
                value_by_node[parent]
                for parent in walk_fx_nodes(fx_node.args)
                if parent in value_by_node
            ]
            output_pytree_spec = _pytree_name(fx_node.args)
        else:
            output_id = new_value(node_id, fx_node.meta.get("tensor_meta"))
            value_by_node[fx_node] = output_id
            output_ids = (output_id,)
            if fx_node.op == "placeholder":
                input_value_ids.append(output_id)
        nodes.append(
            GraphNodeSpec(
                node_id=node_id,
                op_kind=str(fx_node.op),
                target=str(fx_node.target),
                module_path=module_path,
                input_value_ids=input_ids,
                output_value_ids=output_ids,
                parameter_paths=parameter_paths,
                buffer_paths=buffer_paths,
                canonical_target=_canonical_target(fx_node, module_by_path),
                parameter_ids=tuple(parameter_ids_by_path[path] for path in parameter_paths),
                buffer_ids=tuple(buffer_ids_by_path[path] for path in buffer_paths),
                parameter_bytes=sum(
                    parameter_bytes_by_path.get(path, 0) for path in parameter_paths
                ),
                activation_bytes=sum(
                    _value_estimated_bytes(values, value_id) or 0 for value_id in output_ids
                ),
                gradient_bytes=sum(
                    _value_estimated_bytes(values, value_id) or 0 for value_id in output_ids
                ),
                estimated_peak_memory_contribution=(
                    sum(parameter_bytes_by_path.get(path, 0) for path in parameter_paths)
                    + sum(_value_estimated_bytes(values, value_id) or 0 for value_id in output_ids)
                ),
            )
        )
        for value_id in input_ids:
            producer = _producer_node_id(values, value_id)
            if producer is not None:
                transfer = _value_estimated_bytes(values, value_id) or 0
                edges.append(
                    GraphEdgeSpec(
                        producer,
                        node_id,
                        value_id,
                        edge_id=f"e{len(edges):04d}",
                        forward_transfer_bytes=transfer,
                        backward_transfer_bytes=transfer,
                        communication_weight=transfer * 2,
                    )
                )

    consumers: dict[str, list[str]] = {}
    for edge in edges:
        consumers.setdefault(edge.value_id, []).append(edge.target_node_id)
    values = [
        GraphValueSpec(
            value.value_id,
            value.producer_node_id,
            tuple(consumers.get(value.value_id, ())),
            value.shape,
            value.dtype,
            value.requires_grad,
            value.estimated_bytes,
            value.pytree_path,
        )
        for value in values
    ]

    graph = GenericGraphIR(
        nodes=tuple(nodes),
        values=tuple(values),
        edges=tuple(edges),
        input_value_ids=tuple(input_value_ids),
        output_value_ids=tuple(output_value_ids),
        parameter_owners=_parameter_owners(nodes),
        capture_backend=capture_backend,
        input_pytree_spec=input_pytree_spec,
        output_pytree_spec=output_pytree_spec,
        parameter_uses=_parameter_uses(nodes),
        shared_parameter_ids=_shared_parameter_ids(nodes),
        states=_state_objects(model, nodes, parameter_ids_by_path, buffer_ids_by_path),
    )
    return GenericGraphIR(
        **{
            **graph.__dict__,
            "graph_fingerprint": graph_fingerprint(graph),
        }
    )


def infer_boundary_values(
    graph: GenericGraphIR,
    node_to_stage: Mapping[str, str],
) -> tuple[BoundaryValueSpec, ...]:
    values = {value.value_id: value for value in graph.values}
    producer_stage_by_value = {
        value.value_id: node_to_stage[value.producer_node_id]
        for value in graph.values
        if value.producer_node_id in node_to_stage
    }
    consumers: dict[str, set[str]] = {}
    for edge in graph.edges:
        source_stage = producer_stage_by_value.get(edge.value_id)
        target_stage = node_to_stage.get(edge.target_node_id)
        if source_stage is None or target_stage is None or source_stage == target_stage:
            continue
        consumers.setdefault(edge.value_id, set()).add(target_stage)

    boundaries: list[BoundaryValueSpec] = []
    for value_id, consumer_stages in sorted(consumers.items()):
        value = values[value_id]
        boundaries.append(
            BoundaryValueSpec(
                value_id=value_id,
                producer_stage=producer_stage_by_value[value_id],
                consumer_stages=tuple(sorted(consumer_stages)),
                shape=value.shape,
                dtype=value.dtype,
                requires_grad=value.requires_grad,
                estimated_bytes=value.estimated_bytes,
            )
        )
    return tuple(boundaries)


def module_dependencies_from_graph(
    graph: GenericGraphIR,
) -> tuple[tuple[tuple[str, str], ...], tuple[str, ...]]:
    module_by_node = {node.node_id: node.module_path for node in graph.nodes}
    producers: dict[str, set[str]] = {}
    dependencies: set[tuple[str, str]] = set()
    ordered: list[str] = []

    for node in graph.nodes:
        current = module_by_node[node.node_id]
        input_modules: set[str] = set()
        for value_id in node.input_value_ids:
            input_modules.update(producers.get(value_id, set()))
        if current:
            if current not in ordered:
                ordered.append(current)
            for source in input_modules:
                if source != current:
                    dependencies.add((source, current))
            output_modules = {current}
        else:
            output_modules = input_modules
        for value_id in node.output_value_ids:
            producers[value_id] = set(output_modules)

    used_modules = {path for path in module_by_node.values() if path}
    ordered.extend(sorted(used_modules - set(ordered)))
    return tuple(sorted(dependencies)), tuple(ordered)


def _node_module_path(node: Any) -> str | None:
    if getattr(node, "op", None) == "call_module":
        return str(node.target)
    return None


def _node_parameter_paths(node: Any, module_by_path: Mapping[str, Any]) -> tuple[str, ...]:
    if getattr(node, "op", None) != "call_module":
        return ()
    module_path = str(node.target)
    module = module_by_path.get(module_path)
    if module is None:
        return ()
    return tuple(
        f"{module_path}.{name}" if module_path else name
        for name, _parameter in _module_named_parameters(module)
    )


def _node_buffer_paths(node: Any, module_by_path: Mapping[str, Any]) -> tuple[str, ...]:
    if getattr(node, "op", None) != "call_module":
        return ()
    module_path = str(node.target)
    module = module_by_path.get(module_path)
    if module is None:
        return ()
    return tuple(
        f"{module_path}.{name}" if module_path else name
        for name, _buffer in _module_named_buffers(module)
    )


def _get_attr_state_paths(node: Any, ids_by_path: Mapping[str, str]) -> tuple[str, ...]:
    if getattr(node, "op", None) != "get_attr":
        return ()
    path = str(getattr(node, "target", ""))
    return (path,) if path in ids_by_path else ()


def _unique(items: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(items))


def _canonical_target(node: Any, module_by_path: Mapping[str, Any]) -> str:
    if getattr(node, "op", None) == "call_module":
        module = module_by_path.get(str(node.target))
        return type(module).__name__ if module is not None else "unknown_module"
    target = getattr(node, "target", "")
    name = getattr(target, "__qualname__", None) or getattr(target, "__name__", str(target))
    module = getattr(target, "__module__", "")
    return f"{module}.{name}".strip(".")


def _value_spec(value_id: str, producer: str | None, meta: Any) -> GraphValueSpec:
    shape = tuple(getattr(meta, "shape", ()) or ())
    dtype = None if meta is None else str(getattr(meta, "dtype", "")).replace("torch.", "")
    requires_grad = None if meta is None else getattr(meta, "requires_grad", None)
    estimated_bytes = _estimated_bytes(shape, dtype)
    return GraphValueSpec(
        value_id=value_id,
        producer_node_id=producer,
        shape=shape,
        dtype=dtype or None,
        requires_grad=requires_grad,
        estimated_bytes=estimated_bytes,
    )


def _value_estimated_bytes(
    values: Sequence[GraphValueSpec],
    value_id: str,
) -> int | None:
    for value in values:
        if value.value_id == value_id:
            return value.estimated_bytes
    return None


def _estimated_bytes(shape: Sequence[int | str], dtype: str | None) -> int | None:
    bytes_by_dtype = {
        "bool": 1,
        "uint8": 1,
        "int8": 1,
        "int16": 2,
        "int32": 4,
        "int64": 8,
        "float16": 2,
        "bfloat16": 2,
        "float32": 4,
        "float64": 8,
    }
    itemsize = bytes_by_dtype.get(dtype or "")
    if itemsize is None:
        return None
    total = itemsize
    for dim in shape:
        if not isinstance(dim, int):
            return None
        total *= dim
    return total


def _producer_node_id(values: Sequence[GraphValueSpec], value_id: str) -> str | None:
    for value in values:
        if value.value_id == value_id:
            return value.producer_node_id
    return None


def _parameter_schema(model: Any) -> tuple[dict[str, str], dict[str, int]]:
    ids: dict[str, str] = {}
    bytes_by_path: dict[str, int] = {}
    object_to_id: dict[int, str] = {}
    for index, (path, parameter) in enumerate(_named_parameters(model)):
        object_id = id(parameter)
        parameter_id = object_to_id.setdefault(object_id, f"p{index:04d}")
        ids[path] = parameter_id
        bytes_by_path[path] = parameter.numel() * parameter.element_size()
    return ids, bytes_by_path


def _buffer_schema(model: Any) -> dict[str, str]:
    object_to_id: dict[int, str] = {}
    ids: dict[str, str] = {}
    for index, (path, buffer) in enumerate(_named_buffers(model)):
        ids[path] = object_to_id.setdefault(id(buffer), f"b{index:04d}")
    return ids


def _named_parameters(model: Any) -> tuple[tuple[str, Any], ...]:
    try:
        return tuple(model.named_parameters(remove_duplicate=False))
    except TypeError:
        return tuple(model.named_parameters())


def _module_named_parameters(module: Any) -> tuple[tuple[str, Any], ...]:
    try:
        return tuple(module.named_parameters(recurse=True, remove_duplicate=False))
    except TypeError:
        return tuple(module.named_parameters(recurse=True))


def _named_buffers(model: Any) -> tuple[tuple[str, Any], ...]:
    try:
        return tuple(model.named_buffers(remove_duplicate=False))
    except TypeError:
        return tuple(model.named_buffers())


def _module_named_buffers(module: Any) -> tuple[tuple[str, Any], ...]:
    try:
        return tuple(module.named_buffers(recurse=True, remove_duplicate=False))
    except TypeError:
        return tuple(module.named_buffers(recurse=True))


def _state_objects(
    model: Any,
    nodes: Sequence[GraphNodeSpec],
    parameter_ids_by_path: Mapping[str, str],
    buffer_ids_by_path: Mapping[str, str],
) -> tuple[StateObjectSpec, ...]:
    parameter_users: dict[str, list[str]] = {}
    parameter_owner: dict[str, str] = {}
    buffer_users: dict[str, list[str]] = {}
    buffer_owner: dict[str, str] = {}
    for node in nodes:
        for state_id in node.parameter_ids:
            parameter_users.setdefault(state_id, []).append(node.node_id)
            parameter_owner.setdefault(state_id, node.node_id)
        for state_id in node.buffer_ids:
            buffer_users.setdefault(state_id, []).append(node.node_id)
            buffer_owner.setdefault(state_id, node.node_id)

    records: list[StateObjectSpec] = []
    records.extend(
        _state_records(
            "parameter",
            _named_parameters(model),
            parameter_ids_by_path,
            parameter_owner,
            parameter_users,
        )
    )
    records.extend(
        _state_records(
            "buffer",
            _named_buffers(model),
            buffer_ids_by_path,
            buffer_owner,
            buffer_users,
        )
    )
    return tuple(records)


def _state_records(
    kind: str,
    named_items: Sequence[tuple[str, Any]],
    ids_by_path: Mapping[str, str],
    owner_by_id: Mapping[str, str],
    users_by_id: Mapping[str, Sequence[str]],
) -> tuple[StateObjectSpec, ...]:
    keys_by_id: dict[str, list[str]] = {}
    for path, _item in named_items:
        keys_by_id.setdefault(ids_by_path[path], []).append(path)

    records: list[StateObjectSpec] = []
    for path, item in named_items:
        state_id = ids_by_path[path]
        keys = keys_by_id[state_id]
        records.append(
            StateObjectSpec(
                canonical_state_id=state_id,
                kind=kind,
                state_dict_key=path,
                shape=tuple(getattr(item, "shape", ()) or ()),
                dtype=str(getattr(item, "dtype", "")).replace("torch.", "") or None,
                requires_grad=(
                    bool(item.requires_grad)
                    if hasattr(item, "requires_grad") and kind == "parameter"
                    else False
                ),
                storage_id=state_id,
                shared_group_id=state_id if len(keys) > 1 else None,
                checkpoint_owner_key=keys[0],
                owner_node_ids=(
                    (owner_by_id[state_id],) if state_id in owner_by_id else ()
                ),
                use_node_ids=tuple(sorted(set(users_by_id.get(state_id, ())))),
            )
        )
    return tuple(records)


def _real_parameter_storage_bytes(model: Any) -> int:
    total = 0
    for _name, parameter in _named_parameters(model):
        if getattr(parameter, "is_meta", False):
            continue
        total += parameter.numel() * parameter.element_size()
    return total


def _build_factory_sample(
    spec: ModelFactorySpec,
    *,
    device: str,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    try:
        sample = spec.sample_input_builder(device=device)
    except TypeError:
        sample = spec.sample_input_builder()
    if isinstance(sample, tuple) and len(sample) == 2 and isinstance(sample[1], Mapping):
        return tuple(sample[0]), dict(sample[1])
    return tuple(sample), {}


def _parameter_owners(nodes: Sequence[GraphNodeSpec]) -> dict[str, str]:
    owners: dict[str, str] = {}
    for node in nodes:
        for path in node.parameter_paths:
            if path not in owners:
                owners[path] = node.node_id
    return owners


def _parameter_uses(nodes: Sequence[GraphNodeSpec]) -> tuple[ParameterUseSpec, ...]:
    consumers: dict[str, list[str]] = {}
    canonical_path: dict[str, str] = {}
    for node in nodes:
        for parameter_id, path in zip(node.parameter_ids, node.parameter_paths, strict=True):
            consumers.setdefault(parameter_id, []).append(node.node_id)
            canonical_path.setdefault(parameter_id, path)
    return tuple(
        ParameterUseSpec(
            parameter_id=parameter_id,
            canonical_path=canonical_path[parameter_id],
            consumer_nodes=tuple(sorted(set(nodes))),
        )
        for parameter_id, nodes in sorted(consumers.items())
    )


def _shared_parameter_ids(nodes: Sequence[GraphNodeSpec]) -> tuple[str, ...]:
    counts: dict[str, set[str]] = {}
    for node in nodes:
        for parameter_id in node.parameter_ids:
            counts.setdefault(parameter_id, set()).add(node.node_id)
    return tuple(sorted(parameter_id for parameter_id, users in counts.items() if len(users) > 1))


def _custom_op_diagnostics(graph_module: Any) -> tuple[str, ...]:
    return tuple(
        f"unsupported custom op: {_callable_name(node.target)}"
        for node in graph_module.graph.nodes
        if node.op == "call_function" and not _is_allowed_function(node.target)
    )


def _classify_capture_exception(exc: BaseException) -> FailureCode:
    text = f"{exc.__class__.__name__}: {exc}".lower()
    if any(token in text for token in ("custom op", "custom_op", "no fake impl")):
        return FailureCode.CUSTOM_OP_UNSUPPORTED
    if any(
        token in text
        for token in (
            "data dependent",
            "data-dependent",
            "control flow",
            "could not guard on data-dependent expression",
            "symbolically traced variables cannot be used as inputs to control flow",
        )
    ):
        return FailureCode.DYNAMIC_CONTROL_FLOW_UNSUPPORTED
    if any(token in text for token in ("graph break", "graph_break", "unsupported")):
        return FailureCode.GRAPH_BREAK_UNSUPPORTED
    return FailureCode.MODEL_CAPTURE_UNSUPPORTED


def _is_allowed_function(target: Any) -> bool:
    module = getattr(target, "__module__", "")
    name = getattr(target, "__name__", "")
    if module.startswith("torch"):
        return True
    if module in {"operator", "_operator", "builtins", "math"}:
        return True
    if "VariableFunctionsClass" in type(target).__qualname__:
        return True
    return name in {"getitem", "getattr"}


def _callable_name(target: Any) -> str:
    module = getattr(target, "__module__", "")
    name = getattr(target, "__qualname__", None) or getattr(target, "__name__", repr(target))
    return f"{module}.{name}".strip(".")


def _pytree_name(value: Any) -> str:
    if isinstance(value, Mapping):
        return "dict(" + ",".join(sorted(str(key) for key in value)) + ")"
    if isinstance(value, tuple):
        return f"tuple[{len(value)}]"
    if isinstance(value, list):
        return f"list[{len(value)}]"
    return type(value).__name__


def graph_fingerprint(graph: CanonicalGraphIR) -> str:
    payload = {
        "schema_version": graph.schema_version,
        "nodes": [
            {
                "node_id": node.node_id,
                "op_kind": node.op_kind,
                "canonical_target": node.canonical_target,
                "input_value_ids": list(node.input_value_ids),
                "output_value_ids": list(node.output_value_ids),
                "parameter_ids": list(node.parameter_ids),
                "buffer_ids": list(node.buffer_ids),
            }
            for node in graph.nodes
        ],
        "values": [
            {
                "value_id": value.value_id,
                "producer_node_id": value.producer_node_id,
                "consumer_node_ids": list(value.consumer_node_ids),
                "shape": list(value.shape),
                "dtype": value.dtype,
                "requires_grad": value.requires_grad,
            }
            for value in graph.values
        ],
        "input_value_ids": list(graph.input_value_ids),
        "output_value_ids": list(graph.output_value_ids),
        "edges": [
            {
                "source_node_id": edge.source_node_id,
                "target_node_id": edge.target_node_id,
                "value_id": edge.value_id,
            }
            for edge in graph.edges
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
