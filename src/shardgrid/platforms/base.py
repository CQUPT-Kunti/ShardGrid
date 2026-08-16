"""Platform adapter contract for OS-specific behavior."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import PurePath
from typing import Mapping, Sequence

from shardgrid.common.process import ProcessResult, run_process


@dataclass(frozen=True)
class ManualActionCheck:
    allowed: bool
    reason: str | None = None
    requires_manual_action: bool = False


@dataclass(frozen=True)
class BootstrapStep:
    name: str
    command: tuple[str, ...] | str
    shell: bool = False
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None
    timeout: float | None = None


class PlatformAdapter(ABC):
    """Stable interface for isolating platform-specific behavior."""

    platform_name: str
    shell_program: str

    @abstractmethod
    def detect(self) -> dict[str, str]:
        """Return basic platform identity metadata."""

    @abstractmethod
    def run(
        self,
        command: Sequence[str] | str,
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
        shell: bool = False,
        secrets: Sequence[str] = (),
    ) -> ProcessResult:
        """Execute a command using the platform's process behavior."""

    @abstractmethod
    def path_join(self, *parts: str) -> str:
        """Join path fragments using the platform's path rules."""

    @abstractmethod
    def validate_manual_action(self, action: str) -> ManualActionCheck:
        """Classify whether an action must stop for manual intervention."""

    @abstractmethod
    def bootstrap_step(self, name: str, command: Sequence[str] | str) -> BootstrapStep:
        """Build a reusable bootstrap step definition."""


class FakePlatformAdapter(PlatformAdapter):
    """Test double used for contract tests."""

    platform_name = "fake"
    shell_program = "/bin/sh"

    def detect(self) -> dict[str, str]:
        return {"platform": self.platform_name, "shell": self.shell_program}

    def run(
        self,
        command: Sequence[str] | str,
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
        shell: bool = False,
        secrets: Sequence[str] = (),
    ) -> ProcessResult:
        return run_process(
            command,
            cwd=cwd,
            env=env,
            timeout=timeout,
            shell=shell,
            secrets=secrets,
        )

    def path_join(self, *parts: str) -> str:
        return str(PurePath(*parts))

    def validate_manual_action(self, action: str) -> ManualActionCheck:
        if action.startswith("manual:"):
            return ManualActionCheck(
                allowed=False,
                reason=action.split(":", 1)[1] or "manual action required",
                requires_manual_action=True,
            )
        return ManualActionCheck(allowed=True)

    def bootstrap_step(self, name: str, command: Sequence[str] | str) -> BootstrapStep:
        step_command = tuple(command) if not isinstance(command, str) else command
        return BootstrapStep(name=name, command=step_command)
