from __future__ import annotations

import json

from shardgrid.cli.app import main
from shardgrid.cli.commands import doctor as doctor_command
from shardgrid.common.enums import Health
from shardgrid.control.doctor import DoctorReport, DoctorSubjectReport


def _report(*, target: str, health: Health, exit_code: int) -> DoctorReport:
    subject = DoctorSubjectReport(
        subject="control",
        subject_type="control",
        host="host-a",
        runtime="control",
        physical_os="linux",
        runtime_os="linux",
        timestamp="2026-08-27T00:00:00+00:00",
        environment={"selected_environment": "base"},
        checks=(),
        health=health,
        manual_actions=("operator action: install iperf3",),
        commands_run=(),
        exit_code=exit_code,
    )
    return DoctorReport(
        target=target,
        generated_at="2026-08-27T00:00:00+00:00",
        subjects=(subject,),
        health=health,
        exit_code=exit_code,
        checks=(),
    )


def test_doctor_cli_json_uses_report(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        doctor_command,
        "run_doctor",
        lambda target, config=None, fix=False: _report(
            target=target, health=Health.HEALTHY, exit_code=0
        ),
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
        "run_doctor",
        lambda target, config=None, fix=False: _report(
            target=target, health=Health.BLOCKED_MANUAL_ACTION, exit_code=2
        ),
    )

    exit_code = main(["doctor"])

    captured = capsys.readouterr()
    assert "ShardGrid doctor" in captured.out
    assert "manual actions:" in captured.out
    assert "install iperf3" in captured.out
    assert exit_code == 2


def test_doctor_cli_supports_workers_fix(monkeypatch, capsys) -> None:
    seen: list[tuple[str, bool]] = []

    def fake_run(target, config=None, fix=False):
        seen.append((target, fix))
        return _report(target=target, health=Health.DEGRADED, exit_code=1)

    monkeypatch.setattr(doctor_command, "run_doctor", fake_run)

    exit_code = main(["doctor", "--target", "workers", "--fix", "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert seen == [("workers", True)]
    assert payload["target"] == "workers"
    assert exit_code == 1


def test_doctor_is_registered_as_real_command(capsys) -> None:
    try:
        main(["--help"])
    except SystemExit as error:
        assert error.code == 0

    captured = capsys.readouterr()
    assert "doctor" in captured.out
    assert "placeholder command; implementation pending" not in captured.out
