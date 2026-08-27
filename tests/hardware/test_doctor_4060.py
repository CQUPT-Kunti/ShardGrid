from __future__ import annotations

from pathlib import Path

from tests.hardware.doctor_hardware_acceptance import verify_live_worker_doctor


def test_doctor_rtx4060_hardware_acceptance(tmp_path: Path) -> None:
    verify_live_worker_doctor(
        tmp_path,
        worker_id="gpu4060",
        host="10.87.5.155",
        peer_host="10.87.5.15",
        expected_gpu="NVIDIA GeForce RTX 4060 Laptop GPU",
        expected_hostname="LDJ",
        expected_interface="eth3",
    )
