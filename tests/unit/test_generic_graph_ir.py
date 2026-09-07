from __future__ import annotations

import importlib.util
import inspect
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any

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
