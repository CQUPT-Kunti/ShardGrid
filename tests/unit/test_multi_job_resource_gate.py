from __future__ import annotations

from tests.multi_host.test_multi_job_resource_gate import (
    _assert_real_training,
    _capacity_reason,
    _job_step_floor,
    _remember_scan_job_ids,
)


def test_remember_scan_job_ids_tracks_all_dry_run_attempts() -> None:
    known = {"existing"}

    _remember_scan_job_ids(
        known,
        [
            {"job_id": "job-scan-0", "state": "failed"},
            {"job_id": "job-scan-1", "state": "snapshotting"},
            {"job_id": None, "state": "failed"},
        ],
    )

    assert known == {"existing", "job-scan-0", "job-scan-1"}


def test_assert_real_training_does_not_require_param_update_ok() -> None:
    _assert_real_training(
        [
            {
                "train": {
                    "steps": 3,
                    "loss_isfinite": True,
                    "distributed_initialized": True,
                    "stage_materialized": True,
                    "optimizer_step_completed": True,
                    "full_model_materialized": False,
                },
            },
            {
                "placement": {"full_model_materialized": False},
                "train": {
                    "steps": 4,
                    "loss_isfinite": True,
                    "distributed_initialized": True,
                    "stage_materialized": True,
                    "optimizer_step_completed": True,
                    "full_model_materialized": False,
                },
            },
        ]
    )


def test_capacity_reason_accepts_no_feasible_and_resource_changed() -> None:
    assert _capacity_reason({"failure": {"message": "automatic planner failed: NO_FEASIBLE_3_WORKER_PLAN"}})
    assert _capacity_reason({"failure": {"message": "RESOURCE_CHANGED: worker gpu1060 required peak 42"}})


def test_job_step_floor_uses_min_rank_steps() -> None:
    assert _job_step_floor(
        [
            {"train": {"steps": 9}},
            {"train": {"steps": 3}},
            {"train": {"steps": 5}},
        ]
    ) == 3
