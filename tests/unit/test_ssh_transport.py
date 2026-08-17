from __future__ import annotations

import shlex

from shardgrid.common.config import SSHConfig
from shardgrid.common.enums import FailureStage
from shardgrid.common.process import ProcessResult
from shardgrid.transport import ssh as ssh_module
from shardgrid.transport.ssh import KnownHostPolicy, SSHOptions, SSHTransport


def _result(
    stdout: str = "",
    stderr: str = "",
    exit_code: int = 0,
    timed_out: bool = False,
) -> ProcessResult:
    return ProcessResult(
        args=(),
        recorded_command="",
        shell=False,
        cwd=None,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        timed_out=timed_out,
        runtime_environment={},
    )


def _transport(
    *,
    host: str = "10.87.5.155",
    user: str | None = "shardgrid",
    port: int = 22,
    timeout: float = 15.0,
    known_host_policy: KnownHostPolicy = KnownHostPolicy.STRICT,
    known_hosts_path: str | None = None,
    private_key_path: str | None = None,
    ssh_executable: str = "ssh",
) -> SSHTransport:
    return SSHTransport(
        SSHOptions(
            host=host,
            user=user,
            port=port,
            timeout=timeout,
            known_host_policy=known_host_policy,
            known_hosts_path=known_hosts_path,
            private_key_path=private_key_path,
            ssh_executable=ssh_executable,
        )
    )


def test_assemble_command_basic() -> None:
    argv = _transport().assemble_command(["echo", "hello"])

    assert argv[0] == "ssh"
    assert "-o" in argv
    assert "BatchMode=yes" in argv
    assert "ConnectTimeout=15" in argv
    assert "StrictHostKeyChecking=yes" in argv
    assert "shardgrid@10.87.5.155" in argv
    assert argv[-1] == "echo hello"


def test_assemble_command_user_and_port() -> None:
    transport = _transport(port=2222, user=None)

    argv = transport.assemble_command(["hostname"])

    assert "-p" in argv
    assert "2222" in argv
    assert "10.87.5.155" in argv
    assert "shardgrid@10.87.5.155" not in argv


def test_assemble_command_quoting_with_spaces() -> None:
    transport = _transport()
    command = ["python", "-c", "print('a b')", "arg with spaces"]

    argv = transport.assemble_command(command)

    remote = argv[-1]
    assert isinstance(remote, str)
    assert shlex.split(remote) == command


def test_assemble_command_known_host_policy() -> None:
    strict = _transport(known_host_policy=KnownHostPolicy.STRICT)
    accept_new = _transport(known_host_policy=KnownHostPolicy.ACCEPT_NEW)

    assert "StrictHostKeyChecking=yes" in strict.assemble_command(["true"])
    assert "StrictHostKeyChecking=accept-new" in accept_new.assemble_command(["true"])


def test_assemble_command_known_hosts_and_key_references() -> None:
    transport = _transport(
        known_hosts_path="~/.ssh/known_hosts",
        private_key_path="~/.ssh/id_ed25519",
    )

    argv = transport.assemble_command(["true"])

    assert "UserKnownHostsFile=~/.ssh/known_hosts" in argv
    assert "-i" in argv
    assert "~/.ssh/id_ed25519" in argv


def test_run_returns_stdout_stderr_and_exit_code(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(command: object, **kwargs: object) -> ProcessResult:
        captured["command"] = command
        captured["kwargs"] = kwargs
        return _result(stdout="ok\n", stderr="", exit_code=0)

    monkeypatch.setattr(ssh_module, "run_process", fake_run)
    transport = _transport()

    result = transport.run(["echo", "ok"])

    assert result.ok is True
    assert result.stdout == "ok\n"
    captured_command = captured["command"]
    assert isinstance(captured_command, tuple)
    assert captured_command[-1] == "echo ok"
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["timeout"] == 15.0
    assert kwargs["shell"] is False


def test_run_mocked_connection_failure_keeps_diagnostics(monkeypatch) -> None:
    def fake_run(command: object, **kwargs: object) -> ProcessResult:
        return _result(
            stderr="shardgrid@10.87.5.155: Permission denied (publickey)",
            exit_code=255,
        )

    monkeypatch.setattr(ssh_module, "run_process", fake_run)

    result = _transport().run(["hostname"])

    assert result.ok is False
    assert result.exit_code == 255
    assert "Permission denied" in result.stderr


def test_run_mocked_timeout(monkeypatch) -> None:
    def fake_run(command: object, **kwargs: object) -> ProcessResult:
        return _result(stderr="Connection timed out", exit_code=-1, timed_out=True)

    monkeypatch.setattr(ssh_module, "run_process", fake_run)

    result = _transport().run(["hostname"])

    assert result.timed_out is True
    assert result.ok is False


def test_run_ssh_executable_missing(monkeypatch) -> None:
    monkeypatch.setattr(ssh_module.shutil, "which", lambda exe: None)

    result = _transport(ssh_executable="ssh-not-present").run(["hostname"])

    assert result.exit_code == 127
    assert "ssh executable not found" in result.stderr
    assert result.ok is False


def test_run_redacts_secrets_from_recorded_command(monkeypatch) -> None:
    def fake_run(command: object, **kwargs: object) -> ProcessResult:
        recorded = kwargs.get("secrets", ())
        assert isinstance(recorded, tuple)
        text = " ".join(command) if isinstance(command, tuple) else str(command)
        for secret in recorded:
            text = text.replace(secret, "***")
        return _result(exit_code=0)

    monkeypatch.setattr(ssh_module, "run_process", fake_run)

    result = _transport().run(["echo", "super-secret-token"], secrets=("super-secret-token",))

    assert "super-secret-token" not in result.recorded_command
    assert result.ok is True


def test_options_from_ssh_config() -> None:
    ssh_config = SSHConfig(
        default_port=2222,
        connect_timeout_seconds=9,
        strict_host_key_checking=False,
        known_hosts_path="~/.ssh/known_hosts",
        private_key_path="~/.ssh/id_ed25519",
    )

    options = SSHOptions.from_ssh_config(ssh_config, host="10.87.5.15", user="shardgrid")

    assert options.port == 2222
    assert options.timeout == 9.0
    assert options.known_host_policy == KnownHostPolicy.ACCEPT_NEW
    assert options.known_hosts_path == "~/.ssh/known_hosts"
    assert options.private_key_path == "~/.ssh/id_ed25519"


def test_to_failure_record_reuses_failure_record(monkeypatch) -> None:
    result = _result(
        stderr="shardgrid@10.87.5.155: Permission denied (publickey)",
        exit_code=255,
    )

    failure = _transport().to_failure_record(
        result,
        stage=FailureStage.PROBE,
        message="SSH authentication failed",
        recommended_action="authorize the Machine A public key on the Worker",
    )

    assert failure.stage == FailureStage.PROBE
    assert failure.host == "10.87.5.155"
    assert failure.exit_code == 255
    assert failure.message == "SSH authentication failed"
    assert failure.manual_action_required is False