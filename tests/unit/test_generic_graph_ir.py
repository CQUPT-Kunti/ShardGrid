from __future__ import annotations

import importlib.util
import inspect
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.models import generic_partition_zoo  # noqa: E402
from examples.models.generic_partition_zoo import build_zoo_model, make_zoo_sample  # noqa: E402

from shardgrid.planner.generic_graph import (  # noqa: E402
    capture_generic_graph,
    infer_boundary_values,
    module_dependencies_from_graph,
)


def _generic_fixture_module() -> ModuleType:
    module_name = "generic_training_models"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    path = Path(__file__).resolve().parents[1] / "fixtures" / "generic_training_models.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load fixture module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _ordinary_case(name: str) -> Any:
    return _generic_fixture_module().generic_training_case(name)


def _capture_ordinary_case(name: str):
    case = _ordinary_case(name)
    graph = capture_generic_graph(
        case.module.eval(),
        sample_args=case.args,
        sample_kwargs=dict(case.kwargs or {}),
    )
    return case, graph


def _registered_module_order(module) -> list[str]:
    return [name for name, _submodule in module.named_modules() if name]


def _executed_module_order(graph) -> list[str]:
    return [node.module_path for node in graph.nodes if node.module_path is not None]


def _functional_targets(graph) -> list[str]:
    return [
        node.target
        for node in graph.nodes
        if node.module_path is None and node.op_kind not in {"placeholder", "output"}
    ]


def _states_by_key(graph) -> dict[str, Any]:
    return {state.state_dict_key: state for state in graph.states}


class BufferStateModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(4)
        self.proj = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.bn(x))

ZOO_MODELS = (
    "mini_resnet",
    "mini_unet",
    "mini_densenet",
    "mini_inception",
    "mini_vit",
    "mini_encoder_decoder",
)


def _capture(name: str):
    model = build_zoo_model(name).eval()
    args, kwargs = make_zoo_sample(name)
    return capture_generic_graph(model, sample_args=args, sample_kwargs=kwargs)


def test_generic_graph_capture_resnet() -> None:
    graph = _capture("mini_resnet")
    dependencies, ordered = module_dependencies_from_graph(graph)

    assert graph.capture_backend == "torch.fx.symbolic_trace"
    assert graph.nodes
    assert graph.values
    assert graph.edges
    assert all(value.value_id.startswith("v") for value in graph.values)
    assert any(
        src.startswith("blocks.0") and dst.startswith("blocks.0")
        for src, dst in dependencies
    )
    assert "stem.conv" in ordered


def test_generic_graph_capture_unet() -> None:
    graph = _capture("mini_unet")
    dependencies, ordered = module_dependencies_from_graph(graph)

    assert "enc1.conv" in ordered
    assert any(src.startswith("enc1") and dst.startswith("dec1") for src, dst in dependencies)


def test_generic_graph_capture_densenet() -> None:
    graph = _capture("mini_densenet")
    dependencies, ordered = module_dependencies_from_graph(graph)

    assert "input" in ordered
    assert any(src == "input" and dst.startswith("layers.") for src, dst in dependencies)


def test_generic_graph_capture_inception() -> None:
    graph = _capture("mini_inception")
    dependencies, _ordered = module_dependencies_from_graph(graph)

    assert any(src.startswith("stem") and dst.startswith("branch") for src, dst in dependencies)
    assert any(dst == "head" for _src, dst in dependencies)


def test_generic_graph_capture_vit() -> None:
    graph = _capture("mini_vit")
    dependencies, ordered = module_dependencies_from_graph(graph)

    assert "patch" in ordered
    assert any("blocks.0.q" in item for pair in dependencies for item in pair)


def test_generic_graph_capture_encoder_decoder() -> None:
    graph = _capture("mini_encoder_decoder")
    dependencies, ordered = module_dependencies_from_graph(graph)

    assert {"src_embed", "tgt_embed"} <= set(ordered)
    assert any(src.startswith("encoder") and dst.startswith("decoder") for src, dst in dependencies)


def test_unet_skip_becomes_boundary_automatically() -> None:
    graph = _capture("mini_unet")
    node_to_stage = {}
    enc1_relu_claimed = False
    for node in graph.nodes:
        if node.module_path and node.module_path.startswith("enc1"):
            node_to_stage[node.node_id] = "stage0"
        elif not enc1_relu_claimed and "relu" in node.target:
            enc1_relu_claimed = True
            node_to_stage[node.node_id] = "stage0"
        elif node.module_path and node.module_path.startswith(("dec1", "out")):
            node_to_stage[node.node_id] = "stage2"
        elif "cat" in node.target:
            node_to_stage[node.node_id] = "stage2"
        else:
            node_to_stage[node.node_id] = "stage1"

    boundaries = infer_boundary_values(graph, node_to_stage)

    assert any(
        boundary.producer_stage == "stage0" and "stage2" in boundary.consumer_stages
        for boundary in boundaries
    )


def test_densenet_multi_consumer_boundary() -> None:
    graph = _capture("mini_densenet")
    cat_index = 0
    node_to_stage = {}
    input_relu_claimed = False
    for node in graph.nodes:
        if node.module_path == "input":
            node_to_stage[node.node_id] = "stage0"
        elif not input_relu_claimed and "relu" in node.target:
            input_relu_claimed = True
            node_to_stage[node.node_id] = "stage0"
        elif "cat" in node.target:
            node_to_stage[node.node_id] = f"stage{1 + min(cat_index, 2)}"
            cat_index += 1
        else:
            node_to_stage[node.node_id] = "stage2"

    boundaries = infer_boundary_values(graph, node_to_stage)

    assert any(
        boundary.producer_stage == "stage0"
        and len(boundary.consumer_stages) >= 2
        for boundary in boundaries
    )


def test_resnet_residual_boundary() -> None:
    graph = _capture("mini_resnet")
    dependencies, _ordered = module_dependencies_from_graph(graph)

    assert any(
        src.endswith("conv1.norm") and dst.endswith("conv2")
        for src, dst in dependencies
    )
    assert any("proj" in src or "proj" in dst for src, dst in dependencies)


def test_inception_branch_merge_boundary() -> None:
    graph = _capture("mini_inception")
    node_to_stage = {
        node.node_id: (
            "stage0"
            if node.module_path and node.module_path.startswith("branch")
            else "stage1"
        )
        for node in graph.nodes
    }

    boundaries = infer_boundary_values(graph, node_to_stage)

    assert any(
        boundary.producer_stage == "stage0"
        and boundary.consumer_stages == ("stage1",)
        for boundary in boundaries
    )


def test_encoder_decoder_cross_stage_dependency() -> None:
    graph = _capture("mini_encoder_decoder")
    dependencies, _ordered = module_dependencies_from_graph(graph)

    assert any(src.startswith("encoder") and dst.startswith("decoder") for src, dst in dependencies)


def test_model_zoo_has_no_model_specific_stage_builders() -> None:
    source = inspect.getsource(generic_partition_zoo)

    assert "Stage" not in source
    assert "stage_builder" not in source
    assert "partition_builder" not in source


def test_ordinary_multibranch_registration_order_is_not_execution_order() -> None:
    case, graph = _capture_ordinary_case("multi_branch")
    registered = _registered_module_order(case.module)
    executed = _executed_module_order(graph)

    assert registered.index("right") < registered.index("gate")
    assert executed.index("gate") < executed.index("right")
    assert registered != executed[: len(registered)]
    assert any("cat" in target for target in _functional_targets(graph))


def test_ordinary_residual_graph_contains_execution_not_named_module_order_only() -> None:
    case, graph = _capture_ordinary_case("residual")
    registered = _registered_module_order(case.module)
    executed = _executed_module_order(graph)
    functional_targets = _functional_targets(graph)

    assert "block" in registered
    assert "block" not in executed
    assert {"proj", "block.0", "block.2", "head"} <= set(executed)
    assert any("add" in target for target in functional_targets)
    assert len(graph.nodes) > len(executed)


def test_ordinary_shared_module_has_multiple_execution_sites_for_one_registration() -> None:
    case, graph = _capture_ordinary_case("shared_module")
    registered = _registered_module_order(case.module)
    executed = _executed_module_order(graph)
    counts = Counter(executed)

    assert registered.count("shared") == 1
    assert counts["shared"] == 2
    assert any("add" in target for target in _functional_targets(graph))


def test_ordinary_functional_ops_are_execution_nodes_without_modules() -> None:
    case, graph = _capture_ordinary_case("unet_like")
    registered = _registered_module_order(case.module)
    functional_targets = _functional_targets(graph)

    assert registered == ["enc1", "enc2", "dec1", "out"]
    assert any("avg_pool2d" in target for target in functional_targets)
    assert any("interpolate" in target for target in functional_targets)
    assert any("cat" in target for target in functional_targets)
    assert len(functional_targets) >= 4


def test_state_objects_capture_parameter_metadata_and_use_sites() -> None:
    case, graph = _capture_ordinary_case("sequential")
    states = _states_by_key(graph)

    first_weight = states["layers.0.weight"]

    assert set(states) == set(case.module.state_dict())
    assert first_weight.kind == "parameter"
    assert first_weight.canonical_state_id == "p0000"
    assert first_weight.shape == tuple(case.module.layers[0].weight.shape)
    assert first_weight.dtype == "float32"
    assert first_weight.requires_grad is True
    assert first_weight.checkpoint_owner_key == "layers.0.weight"
    assert first_weight.owner_node_ids == first_weight.use_node_ids
    assert first_weight.use_node_ids
    assert graph.to_dict()["states"][0]["state_dict_key"] in case.module.state_dict()


def test_state_objects_capture_buffer_metadata_and_use_sites() -> None:
    model = BufferStateModel().eval()
    graph = capture_generic_graph(model, sample_args=(torch.randn(3, 4),))
    states = _states_by_key(graph)
    bn_node = next(node for node in graph.nodes if node.module_path == "bn")

    running_mean = states["bn.running_mean"]

    assert running_mean.kind == "buffer"
    assert running_mean.canonical_state_id.startswith("b")
    assert running_mean.shape == tuple(model.bn.running_mean.shape)
    assert running_mean.dtype == "float32"
    assert running_mean.requires_grad is False
    assert running_mean.owner_node_ids == (bn_node.node_id,)
    assert running_mean.use_node_ids == (bn_node.node_id,)


def test_state_objects_preserve_tied_parameter_keys_with_one_canonical_owner() -> None:
    _case, graph = _capture_ordinary_case("shared_tied_parameter")
    states = _states_by_key(graph)

    embedding = states["embedding.weight"]
    decoder = states["decoder.weight"]

    assert embedding.canonical_state_id == decoder.canonical_state_id
    assert embedding.shared_group_id == decoder.shared_group_id == "p0000"
    assert embedding.checkpoint_owner_key == "embedding.weight"
    assert decoder.checkpoint_owner_key == "embedding.weight"
    assert set(decoder.use_node_ids) == set(embedding.use_node_ids)
    assert len(decoder.use_node_ids) == 2


def test_state_owner_and_use_records_follow_graph_nodes_not_registration_order() -> None:
    case, graph = _capture_ordinary_case("multi_branch")
    registered = _registered_module_order(case.module)
    executed = _executed_module_order(graph)
    states = _states_by_key(graph)

    right_node = next(node for node in graph.nodes if node.module_path == "right")
    gate_node = next(node for node in graph.nodes if node.module_path == "gate")

    assert registered.index("right") < registered.index("gate")
    assert executed.index("gate") < executed.index("right")
    assert states["right.weight"].use_node_ids == (right_node.node_id,)
    assert states["gate.weight"].use_node_ids == (gate_node.node_id,)
    assert int(gate_node.node_id[1:]) < int(right_node.node_id[1:])
