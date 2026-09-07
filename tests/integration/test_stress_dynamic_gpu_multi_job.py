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