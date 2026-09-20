"""T057 stress taxonomy consumption contract tests.

The stress runner must classify failures from structured ``failure.code`` /
``failure.stage`` evidence recorded by JobManager/SSHLauncher, not by
parsing message substrings as the primary classifier.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

STRESS_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "stress_dynamic_gpu_multi_job.py"


@pytest.fixture(scope="module")
def stress_module():
    import importlib.util
    import sys

    module_name = "stress_dynamic_gpu_multi_job"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, STRESS_SCRIPT)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load stress script from {STRESS_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _job(failure: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "job_id": "job-stress-classify",
        "instance": "MiniUNet-1",
        "status": (
            {"failure": failure}
            if failure is not None
            else {"state": "training"}
        ),
    }


def test_stress_runner_consumes_structured_code_for_memory_reject(stress_module) -> None:
    job = _job(
        {
            "stage": "PLAN",
            "code": "MEMORY_REJECT",
            "message": "probe out of memory for candidate",
            "retryable": True,
        }
    )

    classification = stress_module.classify_job_failure(job)

    assert classification["classifier"] == "code"
    assert classification["classified"] == "memory_reject"
    assert classification["retryable"] is True


def test_stress_runner_consumes_structured_code_for_infra_failure(stress_module) -> None:
    job = _job(
        {
            "stage": "RENDEZVOUS",
            "code": "NETWORK_FAILURE",
            "message": "ssh host unreachable",
            "retryable": True,
        }
    )

    classification = stress_module.classify_job_failure(job)
    assert classification["classifier"] == "code"
    assert classification["classified"] == "infra_failure"
    assert stress_module.is_infra_failure(job) is True
    assert stress_module.is_safety_failure(job) is False


def test_stress_runner_consumes_structured_code_for_rendezvous_failure(stress_module) -> None:
    job = _job(
        {
            "stage": "RENDEZVOUS",
            "code": "RENDEZVOUS_FAILURE",
            "message": "ranks never completed rendezvous",
            "retryable": True,
        }
    )

    classification = stress_module.classify_job_failure(job)
    assert classification["classifier"] == "code"
    assert classification["classified"] == "infra_failure"
    assert stress_module.is_infra_failure(job) is True


def test_stress_runner_consumes_structured_code_for_process_launch_failure(
    stress_module,
) -> None:
    job = _job(
        {
            "stage": "LAUNCH",
            "code": "PROCESS_LAUNCH_FAILURE",
            "message": "remote launch command failed",
            "retryable": True,
        }
    )

    classification = stress_module.classify_job_failure(job)
    assert classification["classifier"] == "code"
    assert classification["classified"] == "infra_failure"
    assert stress_module.is_infra_failure(job) is True


def test_stress_runner_consumes_structured_code_for_runtime_failure(stress_module) -> None:
    job = _job(
        {
            "stage": "TRAIN",
            "code": "RUNTIME_FAILURE",
            "message": "worker training process failed",
            "retryable": False,
        }
    )

    classification = stress_module.classify_job_failure(job)
    assert classification["classifier"] == "code"
    assert classification["classified"] == "runtime_failure"
    assert stress_module.is_safety_failure(job) is False
    assert stress_module.is_infra_failure(job) is False


def test_stress_runner_consumes_structured_code_for_search_budget(stress_module) -> None:
    job = _job(
        {
            "stage": "PLAN",
            "code": "SEARCH_BUDGET_LIMIT",
            "message": "bounded search budget exhausted",
            "retryable": True,
        }
    )

    classification = stress_module.classify_job_failure(job)
    assert classification["classifier"] == "code"
    assert classification["classified"] == "search_budget_limit"


def test_stress_runner_consumes_structured_code_for_formal_oom(stress_module) -> None:
    job = _job(
        {
            "stage": "TRAIN",
            "code": "FORMAL_TRAINING_OOM",
            "message": "rank 0 exited from a formal training OOM",
            "retryable": False,
        }
    )

    classification = stress_module.classify_job_failure(job)
    assert classification["classifier"] == "code"
    assert classification["classified"] == "safety_failure"
    assert stress_module.is_safety_failure(job) is True
    assert stress_module.is_infra_failure(job) is False


def test_stress_runner_consumes_structured_code_for_cpu_saturation(stress_module) -> None:
    job = _job(
        {
            "stage": "TRAIN",
            "code": "CPU_PROCESS_SATURATION",
            "message": "cpu process saturation",
            "retryable": True,
        }
    )

    classification = stress_module.classify_job_failure(job)
    assert classification["classifier"] == "code"
    assert classification["classified"] == "cpu_saturation"


def test_stress_runner_consumes_structured_code_for_gpu_memory_saturation(
    stress_module,
) -> None:
    job = _job(
        {
            "stage": "TRAIN",
            "code": "GPU_MEMORY_SATURATION",
            "message": "gpu memory saturation proven",
            "retryable": True,
        }
    )

    classification = stress_module.classify_job_failure(job)
    assert classification["classifier"] == "code"
    assert classification["classified"] == "gpu_memory_saturation"


def test_stress_runner_consumes_structured_code_for_test_limit(stress_module) -> None:
    job = _job(
        {
            "stage": "TRAIN",
            "code": "TEST_LIMIT_REACHED",
            "message": "test limit reached",
            "retryable": False,
        }
    )

    classification = stress_module.classify_job_failure(job)
    assert classification["classifier"] == "code"
    assert classification["classified"] == "test_limit_reached"


def test_stress_runner_consumes_structured_code_for_not_proven(stress_module) -> None:
    job = _job(
        {
            "stage": "TRAIN",
            "code": "SATURATION_NOT_PROVEN",
            "message": "candidate coverage insufficient",
            "retryable": False,
        }
    )

    classification = stress_module.classify_job_failure(job)
    assert classification["classifier"] == "code"
    assert classification["classified"] == "saturation_not_proven"


def test_stress_runner_does_not_require_message_parsing(stress_module) -> None:
    job = _job(
        {
            "stage": "TRAIN",
            "code": "INFRA_FAILURE",
            "message": "",
            "retryable": True,
        }
    )

    classification = stress_module.classify_job_failure(job)
    assert classification["classifier"] == "code"
    assert classification["classified"] == "infra_failure"
    assert stress_module.is_infra_failure(job) is True


def test_stress_runner_stage_fallback_for_legacy_records(stress_module) -> None:
    job = _job(
        {
            "stage": "RENDEZVOUS",
            "message": "rendezvous timeout",
            "retryable": True,
        }
    )

    classification = stress_module.classify_job_failure(job)
    assert classification["classifier"] == "stage"
    assert classification["classified"] == "infra_failure"
    assert stress_module.is_infra_failure(job) is True


def test_stress_runner_legacy_message_fallback_still_works(stress_module) -> None:
    job = _job(
        {
            "stage": "TRAIN",
            "message": (
                "rank 0 exited; torch.cuda.OutOfMemoryError: CUDA out of memory"
            ),
            "retryable": False,
        }
    )

    classification = stress_module.classify_job_failure(job)
    assert classification["classifier"] == "message"
    assert classification["classified"] == "safety_failure"
    assert stress_module.is_safety_failure(job) is True


def test_stress_runner_preserves_evidence_references(stress_module) -> None:
    job = _job(
        {
            "stage": "TRAIN",
            "code": "RUNTIME_FAILURE",
            "producer": "ssh_launcher",
            "message": "worker training process failed",
            "retryable": False,
            "log_refs": ["logs/train.rank1.stdout"],
            "artifact_refs": ["checkpoint/model-state.pt"],
        }
    )

    classification = stress_module.classify_job_failure(job)
    assert classification["producer"] == "ssh_launcher"
    assert classification["log_refs"] == ["logs/train.rank1.stdout"]
    assert classification["artifact_refs"] == ["checkpoint/model-state.pt"]


def test_equal_step_count_is_stall_not_progress(stress_module) -> None:
    assert stress_module.classify_step_progress(10, 10) == "no_progress"
    assert stress_module.classify_step_progress(10, 11) == "progress"
    assert stress_module.classify_step_progress(11, 10) == "regression"


def test_probe_isolation_check_flags_equal_steps_as_stall(stress_module) -> None:
    jobs = [
        {"job_id": "job-a", "instance": "MiniUNet-1"},
        {"job_id": "job-b", "instance": "MiniDenseNet-1"},
    ]
    before = {"job-a": 10, "job-b": 10}
    after = {"job-a": 10, "job-b": 12}

    result = stress_module.probe_isolation_check(jobs, before, after)

    assert result is not None
    assert result["reason"].startswith("existing active job optimizer steps stalled")
    assert [item["job_id"] for item in result["no_progress"]] == ["job-a"]
    assert result["no_progress"][0]["before_steps"] == 10
    assert result["no_progress"][0]["after_steps"] == 10


def test_probe_isolation_check_flags_step_regression(stress_module) -> None:
    jobs = [{"job_id": "job-a", "instance": "MiniUNet-1"}]
    before = {"job-a": 10}
    after = {"job-a": 7}

    result = stress_module.probe_isolation_check(jobs, before, after)

    assert result is not None
    assert result["reason"].startswith("existing active job optimizer steps regressed")
    assert [item["job_id"] for item in result["regression"]] == ["job-a"]


def test_probe_isolation_check_passes_on_real_progress(stress_module) -> None:
    jobs = [{"job_id": "job-a", "instance": "MiniUNet-1"}]
    before = {"job-a": 10}
    after = {"job-a": 12}

    assert stress_module.probe_isolation_check(jobs, before, after) is None


def test_steady_state_equal_steps_across_samples_is_stall(stress_module) -> None:
    jobs = [{"job_id": "job-a", "instance": "MiniUNet-1"}]
    samples = [
        {"jobs": {"job-a": {"min_steps": 10}}},
        {"jobs": {"job-a": {"min_steps": 10}}},
        {"jobs": {"job-a": {"min_steps": 10}}},
    ]

    result = stress_module.steady_state_progress_check(samples, jobs)

    assert result is not None
    assert result["stalled"][0]["progress"] == "no_progress"
    assert result["stalled"][0]["sample_steps"] == [10, 10, 10]


def test_steady_state_step_regression_is_detected(stress_module) -> None:
    jobs = [{"job_id": "job-a", "instance": "MiniUNet-1"}]
    samples = [
        {"jobs": {"job-a": {"min_steps": 10}}},
        {"jobs": {"job-a": {"min_steps": 8}}},
    ]

    result = stress_module.steady_state_progress_check(samples, jobs)

    assert result is not None
    assert result["stalled"][0]["progress"] == "regression"
    assert result["stalled"][0]["sample_steps"] == [10, 8]


def test_steady_state_increasing_steps_is_not_stalled(stress_module) -> None:
    jobs = [{"job_id": "job-a", "instance": "MiniUNet-1"}]
    samples = [
        {"jobs": {"job-a": {"min_steps": 10}}},
        {"jobs": {"job-a": {"min_steps": 14}}},
        {"jobs": {"job-a": {"min_steps": 19}}},
    ]

    assert stress_module.steady_state_progress_check(samples, jobs) is None


def test_saturation_requires_complete_family_coverage(stress_module) -> None:
    evidence = {
        "non_gpu_bottlenecks": [],
        "all_families_failed_last_round": False,
        "all_smallest_band_attempted": True,
        "remaining_plausible_candidates": 0,
    }

    result = stress_module.classify_saturation_evidence(evidence)

    assert result["reached"] is False
    assert result["code"] == "SATURATION_NOT_PROVEN"
    assert result["reason"] == "FAMILY_COVERAGE_INCOMPLETE"


def test_saturation_requires_smallest_band_attempt(stress_module) -> None:
    evidence = {
        "non_gpu_bottlenecks": [],
        "all_families_failed_last_round": True,
        "all_smallest_band_attempted": False,
        "remaining_plausible_candidates": 0,
    }

    result = stress_module.classify_saturation_evidence(evidence)

    assert result["reached"] is False
    assert result["code"] == "SEARCH_BUDGET_LIMIT"
    assert result["reason"] == "SMALLEST_BAND_NOT_ATTEMPTED"


def test_saturation_requires_no_remaining_candidates(stress_module) -> None:
    evidence = {
        "non_gpu_bottlenecks": [],
        "all_families_failed_last_round": True,
        "all_smallest_band_attempted": True,
        "remaining_plausible_candidates": 2,
    }

    result = stress_module.classify_saturation_evidence(evidence)

    assert result["reached"] is False
    assert result["code"] == "SEARCH_BUDGET_LIMIT"
    assert result["reason"] == "REMAINING_PLAUSIBLE_CANDIDATES"


def test_saturation_blocked_by_non_gpu_bottleneck(stress_module) -> None:
    evidence = {
        "non_gpu_bottlenecks": ["probe_infra_failure"],
        "all_families_failed_last_round": True,
        "all_smallest_band_attempted": True,
        "remaining_plausible_candidates": 0,
    }

    result = stress_module.classify_saturation_evidence(evidence)

    assert result["reached"] is False
    assert result["code"] == "SATURATION_NOT_PROVEN"
    assert result["reason"] == "NON_GPU_BOTTLENECK_FIRST"


def test_saturation_declared_only_with_complete_evidence(stress_module) -> None:
    evidence = {
        "non_gpu_bottlenecks": [],
        "all_families_failed_last_round": True,
        "all_smallest_band_attempted": True,
        "remaining_plausible_candidates": 0,
    }

    result = stress_module.classify_saturation_evidence(evidence)

    assert result["reached"] is True
    assert result["code"] == "GPU_MEMORY_SATURATION"
    assert result["reason"] == "EVIDENCE_COMPLETE"


def test_saturation_evidence_includes_machine_readable_fields(stress_module) -> None:
    report = {
        "attempts": [
            {"attempt": 1, "model": "MiniUNet", "result": "NO_FEASIBLE_PLAN"},
        ],
        "probe_infra_summary": {"infra_failures": 0},
        "saturation": {},
        "cleanup": {"stopped": {}},
    }
    failed_round = {"MiniUNet": {"result": "NO_FEASIBLE_PLAN"}}
    args = type("Args", (), {"max_attempts": 10})()

    import types

    fake_discover = lambda config_path: {"eligible_gpus": [{"worker_id": "gpu0", "gpu_total_memory_bytes": 1 << 30, "gpu_free_memory_bytes": 1 << 30, "used_memory_bytes": 0}]}
    stress_module.discover = fake_discover

    evidence = stress_module.build_saturation_evidence(
        args=args,
        report=report,
        failed_round=failed_round,
        active_jobs=[],
        cluster_path=Path("/tmp/nonexistent"),
        jobs_root=Path("/tmp/nonexistent-jobs"),
    )

    assert "fresh_gpu_state" in evidence
    assert "fresh_free_memory_mb" in evidence
    assert "family_coverage" in evidence
    assert "attempted_bands_per_family" in evidence
    assert "probe_reject_reasons" in evidence
    assert "candidate_search" in evidence
    assert "non_gpu_bottlenecks" in evidence
    assert "active_job_count" in evidence
    assert "remaining_plausible_candidates" in evidence
    assert "checkpoint_evidence" in evidence
    assert "cleanup_evidence" in evidence


def test_checkpoint_evidence_records_merge_not_just_file_presence(
    stress_module,
    tmp_path: Path,
) -> None:
    import torch

    job_root = tmp_path / "job-a"
    (job_root / "checkpoint").mkdir(parents=True)
    torch.save({"layer.weight": torch.zeros(2, 2)}, job_root / "checkpoint" / "model-state.pt")
    (job_root / "checkpoint" / "model-state-metadata.json").write_text(
        '{"format": "shardgrid-generic-model-state/v1", "parameter_changed": true}',
        encoding="utf-8",
    )
    jobs = [{"job_id": "job-a", "instance": "MiniUNet-1", "model": "mini_unet"}]

    evidence = stress_module.checkpoint_sampling_evidence(tmp_path, jobs)

    assert evidence["sampled_job_count"] == 1
    sample = evidence["samples"][0]
    assert sample["model_state_present"] is True
    assert sample["state_dict_is_mapping"] is True
    assert sample["state_dict_key_count"] == 1
    assert sample["merge_evidence"] == "STATE_DICT_PLAIN_MAPPING"
    assert sample["checkpoint_metadata_format"] == "shardgrid-generic-model-state/v1"


def test_checkpoint_evidence_records_load_failure(stress_module, tmp_path: Path) -> None:
    job_root = tmp_path / "job-bad"
    (job_root / "checkpoint").mkdir(parents=True)
    (job_root / "checkpoint" / "model-state.pt").write_text("not a torch file", encoding="utf-8")
    jobs = [{"job_id": "job-bad", "instance": "MiniDenseNet-1", "model": "mini_densenet"}]

    evidence = stress_module.checkpoint_sampling_evidence(tmp_path, jobs)

    sample = evidence["samples"][0]
    assert sample["model_state_present"] is True
    assert sample["merge_evidence"].startswith("LOAD_FAILED")


def test_checkpoint_strict_load_and_validation_forward_evidence(
    stress_module,
    tmp_path: Path,
) -> None:
    import torch

    from examples.models.generic_partition_zoo.models import (
        build_zoo_model,
        make_zoo_sample,
    )

    parameters = {"base_channels": 64}
    model = build_zoo_model("mini_unet", **parameters).eval()
    sample_args, sample_kwargs = make_zoo_sample("mini_unet", **parameters)
    with torch.no_grad():
        model(*sample_args, **(sample_kwargs or {}))

    evidence = stress_module._checkpoint_strict_load(
        {
            "zoo_model": "mini_unet",
            "model_parameters": parameters,
        },
        model.state_dict(),
    )

    assert evidence["strict_load_evidence"] == "PASS"
    assert evidence["missing_keys"] == []
    assert evidence["unexpected_keys"] == []
    assert evidence["validation_forward_evidence"] == "PASS"


def test_checkpoint_strict_load_reports_not_attempted_without_model(
    stress_module,
) -> None:
    evidence = stress_module._checkpoint_strict_load(
        {"zoo_model": None, "model_parameters": {}},
        {},
    )

    assert evidence["strict_load_evidence"] == "NOT_ATTEMPTED_NO_MODEL"
    assert evidence["validation_forward_evidence"] == "NOT_ATTEMPTED_NO_MODEL"