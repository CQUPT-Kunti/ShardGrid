"""Static ParallelPlan contract tests (T070).

Validates the static two-stage plan against the REAL T069 Stage0 / Stage1:
world size, stage count, disjoint ranks, non-empty per-stage parameters,
complete non-overlapping coverage of the full model, tensor boundary
metadata, and the limited-support label.  Failure cases (same rank, empty
stage, single-stage ownership, overlap/missing params, metadata mismatch)
must be rejected.
"""

from __future__ import annotations

from examples.models.minimal_transformer import (
    MinimalTransformerConfig,
    build_minimal_transformer,
)
from examples.models.stage0 import build_stage0
from examples.models.stage1 import build_stage1

from shardgrid.engines.static_validation import (
    LIMITED_SUPPORT_LABEL,
    StaticParallelPlan,
    load_static_parallel_plan,
    validate_parameter_coverage,
    validate_parameter_ownership,
    validate_static_plan,
    validate_static_plan_or_raise,
    validate_tensor_boundary,
)

PLAN_PATH = "examples/models/static_parallel_plan.yaml"

CONFIG = MinimalTransformerConfig(
    vocab_size=1024, hidden_size=128, num_hidden_layers=2,
    num_attention_heads=4, max_seq_length=64,
)


def _real_counts() -> dict[str, int]:
    stage0 = build_stage0(CONFIG, seed=42)
    stage1 = build_stage1(CONFIG, seed=42)
    return {
        "stage0": sum(parameter.numel() for parameter in stage0.parameters()),
        "stage1": sum(parameter.numel() for parameter in stage1.parameters()),
    }


def _real_ownership() -> tuple[dict[str, list[str]], list[str]]:
    full = build_minimal_transformer(CONFIG, seed=42)
    full_names = sorted(dict(full.named_parameters()).keys())
    ownership = {
        "stage0": [
            name
            for name in full_names
            if name.startswith(("embed", "pos", "blocks.0"))
        ],
        "stage1": [
            name
            for name in full_names
            if name.startswith(("blocks.1", "ln", "lm_head"))
        ],
    }
    return ownership, full_names


def test_plan_loads_and_basic_fields() -> None:
    plan = load_static_parallel_plan(PLAN_PATH)
    assert isinstance(plan, StaticParallelPlan)
    assert plan.plan_mode == "static"
    assert plan.support_label == LIMITED_SUPPORT_LABEL
    assert plan.engine == "galvatron"
    assert plan.engine_plan_path == "/var/tmp/shardgrid/original-external-plan.json"
    assert plan.world_size == 2
    assert plan.stage_count == 2
    assert plan.model_total_parameter_count == 664832


def test_plan_stages_have_distinct_ranks_and_real_parameters() -> None:
    plan = load_static_parallel_plan(PLAN_PATH)
    stage0, stage1 = plan.stages
    assert stage0.rank == 0
    assert stage1.rank == 1
    assert stage0.rank != stage1.rank
    assert stage0.worker_id == "gpu4060"
    assert stage1.worker_id == "gpu1060"
    assert stage0.parameter_count > 0
    assert stage1.parameter_count > 0
    # both stages own real modules, not passthrough stubs
    assert any("block0" in name for name in stage0.parameter_ownership)
    assert any("block1" in name for name in stage1.parameter_ownership)
    assert "lm_head.weight" in stage1.parameter_ownership


def test_plan_parameter_counts_match_real_models() -> None:
    plan = load_static_parallel_plan(PLAN_PATH)
    counts = _real_counts()
    assert plan.stages[0].parameter_count == counts["stage0"] == 336384
    assert plan.stages[1].parameter_count == counts["stage1"] == 328448
    assert counts["stage0"] + counts["stage1"] == 664832
    assert plan.model_total_parameter_count == 664832


def test_no_single_stage_holds_all_parameters() -> None:
    counts = _real_counts()
    total = sum(counts.values())
    for count in counts.values():
        assert 0 < count < total
    assert counts["stage0"] / total < 0.6
    assert counts["stage1"] / total < 0.6


def test_plan_validation_passes_with_real_evidence() -> None:
    plan = load_static_parallel_plan(PLAN_PATH)
    counts = _real_counts()
    ownership, full_names = _real_ownership()
    problems = validate_static_plan(
        plan,
        stage_parameter_counts=counts,
        full_model_parameter_count=664832,
        stage_ownership=ownership,
        full_parameter_names=full_names,
    )
    assert problems == [], problems
    validate_static_plan_or_raise(
        plan,
        stage_parameter_counts=counts,
        full_model_parameter_count=664832,
        stage_ownership=ownership,
        full_parameter_names=full_names,
    )


def test_plan_validation_uses_plan_metadata_not_literal_two_stage_rule() -> None:
    plan = StaticParallelPlan(
        plan_id="three-stage-static",
        plan_mode="static",
        support_label=LIMITED_SUPPORT_LABEL,
        engine="galvatron",
        engine_plan_path="/tmp/external-plan.json",
        world_size=3,
        model_total_parameter_count=60,
        stages=(
            type(load_static_parallel_plan(PLAN_PATH).stages[0])(
                id="stage0",
                rank=0,
                parameter_count=10,
                worker_id="gpu0",
            ),
            type(load_static_parallel_plan(PLAN_PATH).stages[0])(
                id="stage1",
                rank=1,
                parameter_count=20,
                worker_id="gpu1",
            ),
            type(load_static_parallel_plan(PLAN_PATH).stages[0])(
                id="stage2",
                rank=2,
                parameter_count=30,
                worker_id="gpu2",
            ),
        ),
    )
    problems = validate_static_plan(
        plan,
        stage_parameter_counts={"stage0": 10, "stage1": 20, "stage2": 30},
        full_model_parameter_count=60,
    )
    assert problems == []


def test_parameter_ownership_is_disjoint_and_complete() -> None:
    ownership, full_names = _real_ownership()
    assert validate_parameter_ownership(
        stage_ownership=ownership, full_parameter_names=full_names
    ) == []
    stage0_names = set(ownership["stage0"])
    stage1_names = set(ownership["stage1"])
    assert not stage0_names & stage1_names
    assert stage0_names | stage1_names == set(full_names)


def test_tensor_boundary_metadata_matches() -> None:
    plan = load_static_parallel_plan(PLAN_PATH)
    stage0, stage1 = plan.stages
    assert validate_tensor_boundary(
        activation_out=stage0.activation_shape,
        activation_in=stage1.activation_shape,
        dtype_out=stage0.activation_dtype,
        dtype_in=stage1.activation_dtype,
    ) == []
    assert stage0.activation_shape == ("batch", "seq", 128)
    assert stage0.activation_dtype == "float32"


def test_stage0_activation_and_stage1_gradient_boundary_correspond() -> None:
    # gradient boundary mirrors the activation boundary in the plan document
    import yaml

    raw = yaml.safe_load(open(PLAN_PATH, encoding="utf-8"))
    boundary = raw["tensor_boundary"]
    assert boundary["activation"]["producer_stage"] == "stage0"
    assert boundary["activation"]["consumer_stage"] == "stage1"
    assert boundary["gradient"]["producer_stage"] == "stage1"
    assert boundary["gradient"]["consumer_stage"] == "stage0"
    assert boundary["activation"]["shape"] == boundary["gradient"]["shape"]
    assert boundary["activation"]["dtype"] == boundary["gradient"]["dtype"]
    assert boundary["activation"]["name"] == "hidden"
    assert boundary["gradient"]["name"] == "hidden.grad"


def test_fail_same_rank_rejected() -> None:
    plan = load_static_parallel_plan(PLAN_PATH)
    bad = StaticParallelPlan(
        plan_id="bad",
        plan_mode="static",
        support_label=LIMITED_SUPPORT_LABEL,
        engine="galvatron",
        engine_plan_path="x",
        world_size=2,
        model_total_parameter_count=664832,
        stages=(
            plan.stages[0],
            type(plan.stages[0])(
                id="stage1", rank=0, parameter_count=100, worker_id="gpu1060"
            ),
        ),
    )
    problems = validate_static_plan(bad)
    assert any("distinct ranks" in problem for problem in problems)


def test_fail_stage_count_mismatch_rejected() -> None:
    plan = load_static_parallel_plan(PLAN_PATH)
    bad = StaticParallelPlan(
        plan_id="bad",
        plan_mode="static",
        support_label=LIMITED_SUPPORT_LABEL,
        engine="galvatron",
        engine_plan_path="x",
        world_size=3,
        model_total_parameter_count=664832,
        stages=plan.stages,
    )
    problems = validate_static_plan(bad)
    assert any("stage_count 2 != world_size 3" in problem for problem in problems)


def test_fail_empty_stage_rejected() -> None:
    plan = load_static_parallel_plan(PLAN_PATH)
    bad = StaticParallelPlan(
        plan_id="bad",
        plan_mode="static",
        support_label=LIMITED_SUPPORT_LABEL,
        engine="galvatron",
        engine_plan_path="x",
        world_size=2,
        model_total_parameter_count=664832,
        stages=(
            plan.stages[0],
            type(plan.stages[0])(
                id="stage1", rank=1, parameter_count=0, worker_id="gpu1060"
            ),
        ),
    )
    problems = validate_static_plan(bad)
    assert any("parameter_count 0" in problem for problem in problems)


def test_fail_all_parameters_in_one_stage_rejected() -> None:
    problems = validate_parameter_coverage(
        stage_parameter_counts={"stage0": 664832, "stage1": 0},
        full_model_parameter_count=664832,
    )
    assert any("parameter_count 0" in problem for problem in problems)
    problems2 = validate_parameter_coverage(
        stage_parameter_counts={"stage0": 336384, "stage1": 328448},
        full_model_parameter_count=664832,
    )
    assert problems2 == []


def test_fail_parameter_overlap_rejected() -> None:
    ownership, full_names = _real_ownership()
    overlapped = {
        "stage0": ownership["stage0"] + ["blocks.1.qkv.weight"],
        "stage1": ownership["stage1"],
    }
    problems = validate_parameter_ownership(
        stage_ownership=overlapped, full_parameter_names=full_names
    )
    assert any("repeats" in problem for problem in problems)


def test_fail_parameter_missing_rejected() -> None:
    ownership, full_names = _real_ownership()
    missing = {"stage0": ownership["stage0"], "stage1": ownership["stage1"][1:]}
    problems = validate_parameter_ownership(
        stage_ownership=missing, full_parameter_names=full_names
    )
    assert any("missing from all stages" in problem for problem in problems)


def test_fail_tensor_metadata_mismatch_rejected() -> None:
    problems = validate_tensor_boundary(
        activation_out=("batch", "seq", 128),
        activation_in=("batch", "seq", 64),
        dtype_out="float32",
        dtype_in="float32",
    )
    assert any("activation shape mismatch" in problem for problem in problems)
    problems2 = validate_tensor_boundary(
        activation_out=("batch", "seq", 128),
        activation_in=("batch", "seq", 128),
        dtype_out="float32",
        dtype_in="float16",
    )
    assert any("activation dtype mismatch" in problem for problem in problems2)


def test_plan_rejects_unknown_parameters() -> None:
    ownership, full_names = _real_ownership()
    bad = {
        "stage0": ownership["stage0"],
        "stage1": ownership["stage1"] + ["not_a_parameter"],
    }
    problems = validate_parameter_ownership(
        stage_ownership=bad, full_parameter_names=full_names
    )
    assert any("unknown parameters" in problem for problem in problems)


def test_limited_support_label_present() -> None:
    plan = load_static_parallel_plan(PLAN_PATH)
    assert plan.support_label == "limited_support"
    assert any(
        "limited_support" in limitation for limitation in plan.limitations
    )
    assert any("arbitrary graph partitioning" in limitation for limitation in plan.limitations)
