from __future__ import annotations

from pathlib import Path

import pytest

from shardgrid.artifacts.transport import (
    ArtifactTransferSpec,
    ArtifactTransferStatus,
    ArtifactTransportConfig,
    ArtifactTransportName,
    RemoteArtifactLocation,
    build_transport_config,
    select_artifact_transport,
    serialize_transfer_result,
)
from shardgrid.common.process import ProcessResult

_SECRET = "TEST_PASSWORD_DO_NOT_LEAK"


def _runner_factory(exit_codes: list[int], stderrs: list[str] | None = None):
    calls: list[tuple[list[str], dict[str, object]]] = []
    stderr_values = stderrs or [""] * len(exit_codes)

    def fake_runner(command, **kwargs):
        index = len(calls)
        calls.append((list(command), dict(kwargs)))
        return ProcessResult(
            args=tuple(command),
            recorded_command=" ".join(str(part) for part in command).replace(_SECRET, "***"),
            shell=False,
            cwd=None,
            exit_code=exit_codes[index],
            stdout="",
            stderr=stderr_values[index],
            timed_out=False,
            runtime_environment={},
        )

    return fake_runner, calls


def _which_available(*available: str):
    return lambda executable: executable if executable in set(available) else None


def _spec(tmp_path: Path, *, label: str = "code") -> ArtifactTransferSpec:
    source = tmp_path / "job-001" / "code"
    source.mkdir(parents=True, exist_ok=True)
    return ArtifactTransferSpec(
        label=label,
        source=str(source),
        destination="/remote/job-001/code",
        direction="push",
        local_root=str(tmp_path / "job-001"),
        recursive=True,
    )


def _remote() -> RemoteArtifactLocation:
    return RemoteArtifactLocation(
        host="worker.local",
        user="shardgrid",
        path="/remote/job-001",
        port=2222,
        private_key_path="/keys/id_ed25519",
        connect_timeout_seconds=15,
        command_timeout_seconds=60.0,
        known_host_policy="yes",
        known_hosts_path="/home/test/.ssh/known_hosts",
    )


def test_explicit_select_scp() -> None:
    transport = select_artifact_transport(
        ArtifactTransportConfig(preferred=ArtifactTransportName.SCP),
        which=_which_available("scp"),
    )
    assert transport.name is ArtifactTransportName.SCP


def test_explicit_select_sftp() -> None:
    transport = select_artifact_transport(
        ArtifactTransportConfig(preferred=ArtifactTransportName.SFTP),
        which=_which_available("sftp"),
    )
    assert transport.name is ArtifactTransportName.SFTP


def test_explicit_select_rsync() -> None:
    transport = select_artifact_transport(
        ArtifactTransportConfig(preferred=ArtifactTransportName.RSYNC),
        which=_which_available("rsync"),
    )
    assert transport.name is ArtifactTransportName.RSYNC


def test_auto_select_prefers_first_available_transport() -> None:
    transport = select_artifact_transport(
        ArtifactTransportConfig(),
        which=_which_available("scp", "sftp"),
    )
    assert transport.name is ArtifactTransportName.SCP


def test_requested_transport_missing_is_rejected() -> None:
    with pytest.raises(ValueError, match="unavailable"):
        select_artifact_transport(
            ArtifactTransportConfig(preferred=ArtifactTransportName.RSYNC),
            which=_which_available("scp"),
        )


def test_auto_when_all_tools_missing_is_rejected() -> None:
    with pytest.raises(ValueError, match="no supported artifact transport"):
        select_artifact_transport(ArtifactTransportConfig(), which=_which_available())


def test_permission_failure_is_reported() -> None:
    runner, _ = _runner_factory([1], ["Permission denied"])
    transport = select_artifact_transport(
        ArtifactTransportConfig(preferred=ArtifactTransportName.SCP),
        which=_which_available("scp"),
        runner=runner,
    )
    result = transport.transfer([_spec(Path("/tmp"))], remote=_remote())

    assert result.status is ArtifactTransferStatus.FAILED
    assert result.items[0].retryable is False
    assert "Permission denied" in result.items[0].stderr


def test_partial_transfer_result_is_not_reported_success(tmp_path: Path) -> None:
    runner, _ = _runner_factory([0, 1], ["", "network error"])
    transport = select_artifact_transport(
        ArtifactTransportConfig(preferred=ArtifactTransportName.SCP),
        which=_which_available("scp"),
        runner=runner,
    )
    result = transport.transfer(
        [_spec(tmp_path, label="config"), _spec(tmp_path, label="code")],
        remote=_remote(),
    )

    assert result.status is ArtifactTransferStatus.PARTIAL
    assert [item.status for item in result.items] == [
        ArtifactTransferStatus.SUCCESS,
        ArtifactTransferStatus.FAILED,
    ]


def test_invalid_transport_config_is_rejected() -> None:
    with pytest.raises(ValueError):
        build_transport_config("ftp")


def test_sensitive_credential_not_leaked_to_recorded_command(tmp_path: Path) -> None:
    runner, calls = _runner_factory([0])
    transport = select_artifact_transport(
        ArtifactTransportConfig(preferred=ArtifactTransportName.SCP),
        which=_which_available("scp"),
        runner=runner,
    )
    result = transport.transfer([_spec(tmp_path)], remote=_remote(), secrets=[_SECRET])

    assert _SECRET not in result.items[0].recorded_command
    assert calls[0][0][0] == "scp"


def test_scp_reuses_ssh_flags_and_command_timeout(tmp_path: Path) -> None:
    runner, calls = _runner_factory([0])
    transport = select_artifact_transport(
        ArtifactTransportConfig(preferred=ArtifactTransportName.SCP),
        which=_which_available("scp"),
        runner=runner,
    )

    result = transport.transfer([_spec(tmp_path)], remote=_remote())

    assert result.ok is True
    argv, kwargs = calls[0]
    assert argv[:11] == [
        "scp",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        "UserKnownHostsFile=/home/test/.ssh/known_hosts",
        "-i",
        "/keys/id_ed25519",
    ]
    assert "-P" in argv
    assert kwargs["timeout"] == 60.0


def test_rsync_reuses_ssh_flags_and_command_timeout(tmp_path: Path) -> None:
    runner, calls = _runner_factory([0])
    transport = select_artifact_transport(
        ArtifactTransportConfig(preferred=ArtifactTransportName.RSYNC),
        which=_which_available("rsync"),
        runner=runner,
    )

    result = transport.transfer([_spec(tmp_path)], remote=_remote())

    assert result.ok is True
    argv, kwargs = calls[0]
    assert argv[:3] == ["rsync", "-a", "-e"]
    assert "BatchMode=yes" in argv[3]
    assert "ConnectTimeout=15" in argv[3]
    assert "StrictHostKeyChecking=yes" in argv[3]
    assert "UserKnownHostsFile=/home/test/.ssh/known_hosts" in argv[3]
    assert "-i /keys/id_ed25519" in argv[3]
    assert "-p 2222" in argv[3]
    assert kwargs["timeout"] == 60.0


def test_argument_safety_preserves_space_paths_without_shell_join(tmp_path: Path) -> None:
    runner, calls = _runner_factory([0])
    source_root = tmp_path / "job 001"
    source = source_root / "code dir"
    source.mkdir(parents=True)
    transport = select_artifact_transport(
        ArtifactTransportConfig(preferred=ArtifactTransportName.SCP),
        which=_which_available("scp"),
        runner=runner,
    )
    result = transport.transfer(
        [
            ArtifactTransferSpec(
                label="code",
                source=str(source),
                destination="/remote/code dir",
                direction="push",
                local_root=str(source_root),
                recursive=True,
            )
        ],
        remote=_remote(),
    )

    assert result.ok is True
    assert calls[0][0][-2] == str(source)
    assert calls[0][0][-1].endswith(":/remote/code dir")


def test_path_escape_is_rejected(tmp_path: Path) -> None:
    runner, _ = _runner_factory([0])
    transport = select_artifact_transport(
        ArtifactTransportConfig(preferred=ArtifactTransportName.SCP),
        which=_which_available("scp"),
        runner=runner,
    )
    with pytest.raises(ValueError, match="escaped local_root"):
        transport.transfer(
            [
                ArtifactTransferSpec(
                    label="code",
                    source=str(tmp_path / "outside"),
                    destination="/remote/code",
                    direction="push",
                    local_root=str(tmp_path / "job-001"),
                )
            ],
            remote=_remote(),
        )


def test_pull_creates_nested_destination_parent_before_command(tmp_path: Path) -> None:
    destination = tmp_path / "job-001" / "logs" / "worker-a" / "rank0" / ".incoming.log"
    calls = []

    def runner(command, **kwargs):
        del kwargs
        calls.append(command)
        assert destination.parent.exists()
        return ProcessResult(
            args=tuple(command),
            recorded_command="scp ok",
            shell=False,
            cwd=None,
            exit_code=0,
            stdout="",
            stderr="",
            timed_out=False,
            runtime_environment={},
        )

    transport = select_artifact_transport(
        ArtifactTransportConfig(preferred=ArtifactTransportName.SCP),
        which=_which_available("scp"),
        runner=runner,
    )

    result = transport.transfer(
        [
            ArtifactTransferSpec(
                label="log",
                source="/remote/log",
                destination=str(destination),
                direction="pull",
                local_root=str(tmp_path / "job-001"),
            )
        ],
        remote=_remote(),
    )

    assert result.ok is True
    assert calls


def test_pull_rejects_escaped_destination_before_mkdir(tmp_path: Path) -> None:
    escaped = tmp_path / "outside" / "created-by-bug" / "file.txt"
    transport = select_artifact_transport(
        ArtifactTransportConfig(preferred=ArtifactTransportName.SCP),
        which=_which_available("scp"),
        runner=lambda command, **kwargs: pytest.fail("runner must not be called"),
    )

    with pytest.raises(ValueError, match="escaped local_root"):
        transport.transfer(
            [
                ArtifactTransferSpec(
                    label="log",
                    source="/remote/log",
                    destination=str(escaped),
                    direction="pull",
                    local_root=str(tmp_path / "job-001"),
                )
            ],
            remote=_remote(),
        )

    assert not escaped.parent.exists()


def test_transfer_result_serialization(tmp_path: Path) -> None:
    runner, _ = _runner_factory([0])
    transport = select_artifact_transport(
        ArtifactTransportConfig(preferred=ArtifactTransportName.SFTP),
        which=_which_available("sftp"),
        runner=runner,
    )
    result = transport.transfer([_spec(tmp_path)], remote=_remote())
    payload = serialize_transfer_result(result)

    assert payload["transport"] == "sftp"
    assert payload["status"] == "success"
