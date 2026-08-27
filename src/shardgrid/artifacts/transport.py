"""Artifact transport selection and command execution over mature tools."""

from __future__ import annotations

import shutil
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence, cast

from shardgrid.common.enums import FailureStage, SerializableStrEnum
from shardgrid.common.errors import failure_from_process_result
from shardgrid.common.process import ProcessResult, redact_text, run_process
from shardgrid.jobs.models import FailureRecord

WhichFunc = Callable[[str], str | None]
RunProcessFunc = Callable[..., ProcessResult]


class ArtifactTransportName(SerializableStrEnum):
    AUTO = "auto"
    RSYNC = "rsync"
    SCP = "scp"
    SFTP = "sftp"


class ArtifactTransferStatus(SerializableStrEnum):
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class ArtifactTransportConfig:
    preferred: ArtifactTransportName = ArtifactTransportName.AUTO
    rsync_executable: str = "rsync"
    scp_executable: str = "scp"
    sftp_executable: str = "sftp"

    @classmethod
    def from_value(cls, value: str) -> "ArtifactTransportConfig":
        return cls(preferred=ArtifactTransportName(value))


@dataclass(frozen=True)
class RemoteArtifactLocation:
    host: str
    path: str
    user: str | None = None
    port: int = 22
    private_key_path: str | None = None

    def __post_init__(self) -> None:
        if not self.host.strip():
            raise ValueError("remote host must be a non-empty string")
        if not self.path.strip():
            raise ValueError("remote path must be a non-empty string")

    @property
    def target(self) -> str:
        return f"{self.user}@{self.host}:{self.path}" if self.user else f"{self.host}:{self.path}"


@dataclass(frozen=True)
class ArtifactTransferSpec:
    label: str
    source: str
    destination: str
    direction: str
    local_root: str | None = None
    recursive: bool = False

    def __post_init__(self) -> None:
        if self.direction not in {"push", "pull"}:
            raise ValueError("direction must be push or pull")
        if not self.label.strip():
            raise ValueError("label must be a non-empty string")
        if not self.source.strip() or not self.destination.strip():
            raise ValueError("source and destination must be non-empty strings")


@dataclass(frozen=True)
class ArtifactTransferItemResult:
    label: str
    transport: str
    status: ArtifactTransferStatus
    source: str
    destination: str
    recorded_command: str | None = None
    exit_code: int | None = None
    stderr: str = ""
    retryable: bool = False

    @property
    def ok(self) -> bool:
        return self.status is ArtifactTransferStatus.SUCCESS


@dataclass(frozen=True)
class ArtifactTransferResult:
    transport: str
    status: ArtifactTransferStatus
    items: list[ArtifactTransferItemResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status is ArtifactTransferStatus.SUCCESS


def _serialize(value: Any) -> Any:
    if hasattr(value, "value"):
        return value.value
    if is_dataclass(value):
        return {key: _serialize(item) for key, item in asdict(cast(Any, value)).items()}
    if isinstance(value, dict):
        return {str(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    return value


class ArtifactTransport(Protocol):
    name: ArtifactTransportName

    def transfer(
        self,
        items: Sequence[ArtifactTransferSpec],
        *,
        remote: RemoteArtifactLocation,
        secrets: Sequence[str] = (),
    ) -> ArtifactTransferResult: ...


@dataclass(frozen=True)
class _ToolConfig:
    name: ArtifactTransportName
    executable: str


class CommandArtifactTransport:
    def __init__(
        self,
        tool: _ToolConfig,
        *,
        which: WhichFunc = shutil.which,
        runner: RunProcessFunc = run_process,
    ) -> None:
        self.name = tool.name
        self.executable = tool.executable
        self._which = which
        self._runner = runner

    def transfer(
        self,
        items: Sequence[ArtifactTransferSpec],
        *,
        remote: RemoteArtifactLocation,
        secrets: Sequence[str] = (),
    ) -> ArtifactTransferResult:
        if self._which(self.executable) is None:
            return ArtifactTransferResult(
                transport=self.name.value,
                status=ArtifactTransferStatus.UNAVAILABLE,
                items=[
                    ArtifactTransferItemResult(
                        label=item.label,
                        transport=self.name.value,
                        status=ArtifactTransferStatus.UNAVAILABLE,
                        source=item.source,
                        destination=item.destination,
                        stderr=f"{self.executable} executable not found",
                    )
                    for item in items
                ],
            )

        results: list[ArtifactTransferItemResult] = []
        for item in items:
            self._validate_local_path(item)
            argv, stdin = self._build_command(item, remote)
            process = self._runner(argv, shell=False, secrets=secrets, input=stdin)
            results.append(
                ArtifactTransferItemResult(
                    label=item.label,
                    transport=self.name.value,
                    status=_result_status(process),
                    source=item.source,
                    destination=item.destination,
                    recorded_command=process.recorded_command,
                    exit_code=process.exit_code,
                    stderr=redact_text(process.stderr, secrets) or "",
                    retryable=not _is_permission_denied(process.stderr),
                )
            )
        return ArtifactTransferResult(
            transport=self.name.value,
            status=_overall_status(results),
            items=results,
        )

    def to_failure_record(
        self,
        result: ArtifactTransferItemResult,
        *,
        host: str,
        message: str,
        recommended_action: str,
    ) -> FailureRecord:
        process = ProcessResult(
            args=() if result.recorded_command is None else result.recorded_command,
            recorded_command=result.recorded_command or self.executable,
            shell=False,
            cwd=None,
            exit_code=result.exit_code or 1,
            stdout="",
            stderr=result.stderr,
            timed_out=False,
            runtime_environment={},
        )
        return failure_from_process_result(
            stage=FailureStage.DISTRIBUTE,
            host=host,
            result=process,
            message=message,
            recommended_action=recommended_action,
            retryable=result.retryable,
        )

    def _validate_local_path(self, item: ArtifactTransferSpec) -> None:
        if item.local_root is None:
            return
        root = Path(item.local_root).resolve(strict=False)
        if item.direction == "push":
            path = Path(item.source).resolve(strict=False)
        else:
            path = Path(item.destination).resolve(strict=False)
        if root not in path.parents and path != root:
            raise ValueError("artifact path escaped local_root")

    def _ssh_flags(self, remote: RemoteArtifactLocation) -> list[str]:
        flags: list[str] = []
        if remote.private_key_path:
            flags.extend(["-i", remote.private_key_path])
        if remote.port != 22:
            flags.extend(["-P", str(remote.port)])
        return flags

    def _build_command(
        self, item: ArtifactTransferSpec, remote: RemoteArtifactLocation
    ) -> tuple[list[str], str | None]:
        if self.name is ArtifactTransportName.SCP:
            argv = [self.executable, *self._ssh_flags(remote)]
            if item.recursive:
                argv.append("-r")
            if item.direction == "push":
                argv.extend([item.source, remote.target_for(item.destination)])
            else:
                argv.extend([remote.source_for(item.source), item.destination])
            return argv, None
        if self.name is ArtifactTransportName.RSYNC:
            argv = [self.executable, "-a"]
            if remote.port != 22 or remote.private_key_path:
                ssh_parts = ["ssh"]
                if remote.private_key_path:
                    ssh_parts.extend(["-i", remote.private_key_path])
                if remote.port != 22:
                    ssh_parts.extend(["-p", str(remote.port)])
                argv.extend(["-e", " ".join(ssh_parts)])
            if item.direction == "push":
                argv.extend([item.source, remote.target_for(item.destination)])
            else:
                argv.extend([remote.source_for(item.source), item.destination])
            return argv, None

        argv = [self.executable]
        if remote.private_key_path:
            argv.extend(["-i", remote.private_key_path])
        if remote.port != 22:
            argv.extend(["-P", str(remote.port)])
        argv.append(remote.login_target)
        if item.direction == "push":
            script = f'put {"-r " if item.recursive else ""}{item.source} {item.destination}\n'
        else:
            script = f'get {"-r " if item.recursive else ""}{item.source} {item.destination}\n'
        return argv, script


def select_artifact_transport(
    config: ArtifactTransportConfig,
    *,
    which: WhichFunc = shutil.which,
    runner: RunProcessFunc = run_process,
) -> CommandArtifactTransport:
    preferred = config.preferred
    if preferred is ArtifactTransportName.AUTO:
        for name, executable in (
            (ArtifactTransportName.RSYNC, config.rsync_executable),
            (ArtifactTransportName.SCP, config.scp_executable),
            (ArtifactTransportName.SFTP, config.sftp_executable),
        ):
            if which(executable):
                return CommandArtifactTransport(
                    _ToolConfig(name=name, executable=executable), which=which, runner=runner
                )
        raise ValueError("no supported artifact transport executable is available")

    executable = {
        ArtifactTransportName.RSYNC: config.rsync_executable,
        ArtifactTransportName.SCP: config.scp_executable,
        ArtifactTransportName.SFTP: config.sftp_executable,
    }[preferred]
    if which(executable) is None:
        raise ValueError(f"requested artifact transport is unavailable: {preferred.value}")
    return CommandArtifactTransport(
        _ToolConfig(name=preferred, executable=executable), which=which, runner=runner
    )


def build_transport_config(preferred: str) -> ArtifactTransportConfig:
    return ArtifactTransportConfig.from_value(preferred)


def serialize_transfer_result(result: ArtifactTransferResult) -> dict[str, Any]:
    return _serialize(result)


def _result_status(process: ProcessResult) -> ArtifactTransferStatus:
    if process.ok:
        return ArtifactTransferStatus.SUCCESS
    return ArtifactTransferStatus.FAILED


def _overall_status(items: Sequence[ArtifactTransferItemResult]) -> ArtifactTransferStatus:
    if not items:
        return ArtifactTransferStatus.SUCCESS
    successes = [item for item in items if item.ok]
    if len(successes) == len(items):
        return ArtifactTransferStatus.SUCCESS
    if successes:
        return ArtifactTransferStatus.PARTIAL
    return ArtifactTransferStatus.FAILED


def _is_permission_denied(stderr: str) -> bool:
    return "permission denied" in stderr.lower()


def _remote_target(host: str, user: str | None, path: str) -> str:
    return f"{user}@{host}:{path}" if user else f"{host}:{path}"


RemoteArtifactLocation.target_for = lambda self, path: _remote_target(  # type: ignore[attr-defined]
    self.host, self.user, path
)
RemoteArtifactLocation.source_for = lambda self, path: _remote_target(  # type: ignore[attr-defined]
    self.host, self.user, path
)
RemoteArtifactLocation.login_target = property(  # type: ignore[attr-defined]
    lambda self: f"{self.user}@{self.host}" if self.user else self.host
)
