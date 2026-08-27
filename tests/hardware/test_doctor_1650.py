from __future__ import annotations

from pathlib import Path

from tests.hardware.doctor_hardware_acceptance import verify_live_worker_doctor


def test_doctor_gtx1650_hardware_acceptance(tmp_path: Path) -> None:
    verify_live_worker_doctor(
        tmp_path,
        worker_id="gpu1060",
        host="10.87.5.15",
        peer_host="10.87.5.155",
        expected_gpu="NVIDIA GeForce GTX 1650",
        expected_hostname="LAPTOP-5G3QUOGM",
        expected_interface="eth0",
    )
