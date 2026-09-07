from __future__ import annotations

import pytest

from shardgrid.common.enums import FailureCode, FailureStage
from shardgrid.common.models import as_worker_id
from shardgrid.jobs.models import FailureRecord

CONTRACT_FAILURE_CODES = (
    "MODEL_CAPTURE_UNSUPPORTED",
    "GRAPH_BREAK_UNSUPPORTED",
    "CUSTOM_OP_UNSUPPORTED",
    "DYNAMIC_CONTROL_FLOW_UNSUPPORTED",
    "PROFILE_FAILURE",
    "PARTITION_FAILURE",
    "PLAN_VALIDATION_FAILURE",
    "NO_FEASIBLE_PLAN",
    "SEARCH_BUDGET_LIMIT",
    "MEMORY_REJECT",
    "FORMAL_TRAINING_OOM",
    "RESOURCE_CHANGED",
    "NETWORK_FAILURE",
    "RENDEZVOUS_FAILURE",
    "PROCESS_LAUNCH_FAILURE",
    "INFRA_FAILURE",
    "RUNTIME_FAILURE",
    "CPU_PROCESS_SATURATION",
    "GPU_MEMORY_SATURATION",
    "TEST_LIMIT_REACHED",
    "SATURATION_NOT_PROVEN",
)


def test_all_contract_failure_codes_are_representable() -> None:
    represented = set(FailureCode.__members__)
    assert set(CONTRACT_FAILURE_CODES) <= represented


def test_failure_code_serializes_and_round_trips_as_stable_string() -> None:
    for code in FailureCode:
        assert code.to_json() == code.value
        assert FailureCode.from_value(code.value) is code
        assert str(code) == code.value


def test_failure_code_values_are_unique_stable_strings() -> None:
    assert len(FailureCode.__members__) == len(set(code.value for code in FailureCode))
    assert set(CONTRACT_FAILURE_CODES) == set(code.value for code in FailureCode)


def test_failure_stage_compatibility_is_preserved() -> None:
    assert {stage.value for stage in FailureStage} == {
        "BOOTSTRAP",
        "PROBE",
        "NETWORK",
        "PROFILE",
        "PLAN",
        "DISTRIBUTE",
        "LAUNCH",
        "RENDEZVOUS",
        "TRAIN",
        "CHECKPOINT",
        "SCHEDULE",
        "GPU_SHARE",
        "STOP",
        "CLEANUP",
    }


def test_failure_record_contract_round_trip() -> None:
    record = FailureRecord(
        stage=FailureStage.TRAIN,
        host="machine-d.local",
        worker_id=as_worker_id("gpu1060"),
        command="python3 train.py",
        exit_code=2,
        stdout_path="/tmp/train.stdout",
        stderr_path="/tmp/train.stderr",
        message="training step failed",
        recommended_action="inspect worker logs and retry",
        retryable=False,
        manual_action_required=False,
    )

    restored = FailureRecord.from_dict(record.to_dict())

    assert restored == record
    assert restored.to_dict()["stage"] == "TRAIN"


def test_failure_record_contract_rejects_incomplete_payloads() -> None:
    with pytest.raises(ValueError, match="failure message"):
        FailureRecord(
            stage=FailureStage.BOOTSTRAP,
            host="machine-a.local",
            recommended_action="rerun bootstrap",
        )

    with pytest.raises(ValueError, match="recommended_action"):
        FailureRecord(
            stage=FailureStage.PROBE,
            host="machine-c.local",
            message="probe failed",
        )
