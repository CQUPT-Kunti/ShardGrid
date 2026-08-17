from __future__ import annotations

import json

from shardgrid.cli.app import main
from shardgrid.cli.commands import doctor as doctor_command
from shardgrid.common.enums import Health
from shardgrid.control.doctor import ControlDoctorReport


def _report(*, health: Health, exit_code: int) -> ControlDoctorReport:
    return ControlDoctorReport(
        target="control",
        host="host-a",
        os_version="Linux-test",
        timestamp="2026-08-16T00:00:00+00:00",
        environment={"selected_environment": "base"},
        checks=(),
        health=health,
        manual_actions=("operator action: install iperf3",),
        commands_run=(),
        exit_code=exit_code,
    )


def test_doctor_cli_json_uses_report(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        doctor_command,
        "run_control_doctor",
        lambda config: _report(health=Health.HEALTHY, exit_code=0),
    )

    exit_code = main(["doctor", "--target", "control", "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["target"] == "control"
    assert payload["health"] == "healthy"


def test_doctor_cli_human_reports_manual_actions(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        doctor_command,
        "run_control_doctor",
        lambda config: _report(health=Health.BLOCKED_MANUAL_ACTION, exit_code=2),
    )

    exit_code = main(["doctor"])

    captured = capsys.readouterr()
    assert "ShardGrid doctor" in captured.out
    assert "manual actions:" in captured.out
    assert "install iperf3" in captured.out
    assert exit_code == 2


def test_doctor_is_registered_as_real_command(capsys) -> None:
    try:
        main(["--help"])
    except SystemExit as error:
        assert error.code == 0

    captured = capsys.readouterr()
    assert "doctor" in captured.out
    assert "placeholder command; implementation pending" not in captured.out