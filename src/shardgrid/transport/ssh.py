"""SSH transport over the system OpenSSH client (T037).

ShardGrid never implements the SSH protocol: this transport assembles and runs
the system ``ssh`` executable through the existing process wrapper, so every
result is a :class:`ProcessResult` and every failure can become a
:class:`FailureRecord` through the existing error helpers.  Credentials are
never stored here; authentication relies on the user's existing SSH keys or
SSH agent configuration (BatchMode is always on).
"""

from __future__ import annotations

import shlex
import shutil
from dataclasses import dataclass
from typing import Sequence

from shardgrid.common.config import SSHConfig
from shardgrid.common.enums import FailureStage, SerializableStrEnum
from shardgrid.common.errors import failure_from_process_result
from shardgrid.common.process import ProcessResult, redact_command, run_process
from shardgrid.jobs.models import FailureRecord


class KnownHostPolicy(SerializableStrEnum):
    STRICT = "yes"
    ACCEPT_NEW = "accept-new"


@dataclass(frozen=True)
class SSHOptions:
    host: str
    user: str | None = None
    port: int = 22
    timeout: float = 15.0
    known_host_policy: KnownHostPolicy = KnownHostPolicy.STRICT
    known_hosts_path: str | None = None
    private_key_path: str | None = None
    ssh_executable: str = "ssh"

    @classmethod
    def from_ssh_config(
        cls,
        ssh_config: SSHConfig,
        *,
        host: str,
        user: str | None,
        port: int | None = None,
    ) -> SSHOptions:
        policy = (
            KnownHostPolicy.STRICT
            if ssh_config.strict_host_key_checking
            else KnownHostPolicy.ACCEPT_NEW
        )
        return cls(
            host=host,
            user=user,
            port=port if port is not None else ssh_config.default_port,
            timeout=float(ssh_config.connect_timeout_seconds),
            known_host_policy=policy,
            known_hosts_path=ssh_config.known_hosts_path,
            private_key_path=ssh_config.private_key_path,
        )


class SSHTransport:
    """Run remote commands through the system OpenSSH client."""

    def __init__(self, options: SSHOptions) -> None:
        self.options = options

    def _base_argv(self) -> list[str]:
        argv = [self.options.ssh_executable]
        argv.extend(["-o", "BatchMode=yes"])
        argv.extend(["-o", f"ConnectTimeout={int(self.options.timeout)}"])
        argv.extend(
            ["-o", f"StrictHostKeyChecking={self.options.known_host_policy.value}"]
        )
        if self.options.known_hosts_path:
            argv.extend(["-o", f"UserKnownHostsFile={self.options.known_hosts_path}"])
        if self.options.private_key_path:
            argv.extend(["-i", self.options.private_key_path])
        if self.options.port != 22:
            argv.extend(["-p", str(self.options.port)])
        target = self.options.host
        if self.options.user:
            target = f"{self.options.user}@{target}"
        argv.append(target)
        return argv

    def assemble_command(self, command: Sequence[str] | str) -> tuple[str, ...]:
        argv = self._base_argv()
        if isinstance(command, str):
            argv.append(command)
        else:
            argv.append(shlex.join(command))
        return tuple(argv)

    def run(
        self,
        command: Sequence[str] | str,
        *,
        timeout: float | None = None,
        secrets: Sequence[str] = (),
        check: bool = False,
        stdin: str | bytes | None = None,
    ) -> ProcessResult:
        argv = self.assemble_command(command)
        recorded = redact_command(argv, secrets)
        if shutil.which(self.options.ssh_executable) is None:
            return ProcessResult(
                args=argv,
                recorded_command=recorded,
                shell=False,
                cwd=None,
                exit_code=127,
                stdout="",
                stderr=f"ssh executable not found: {self.options.ssh_executable}",
                timed_out=False,
                runtime_environment={},
            )
        try:
            return run_process(
                argv,
                timeout=self.options.timeout if timeout is None else timeout,
                shell=False,
                secrets=secrets,
                check=check,
                input=stdin,
            )
        except FileNotFoundError:
            return ProcessResult(
                args=argv,
                recorded_command=recorded,
                shell=False,
                cwd=None,
                exit_code=127,
                stdout="",
                stderr=f"ssh executable not found: {self.options.ssh_executable}",
                timed_out=False,
                runtime_environment={},
            )

    def to_failure_record(
        self,
        result: ProcessResult,
        *,
        stage: FailureStage = FailureStage.PROBE,
        host: str | None = None,
        message: str,
        recommended_action: str,
        worker_id: str | None = None,
        conda_environment: str | None = None,
        conda_prefix: str | None = None,
        python_executable: str | None = None,
    ) -> FailureRecord:
        return failure_from_process_result(
            stage=stage,
            host=host or self.options.host,
            result=result,
            message=message,
            recommended_action=recommended_action,
            worker_id=worker_id,
            conda_environment=conda_environment,
            conda_prefix=conda_prefix,
            python_executable=python_executable,
        )