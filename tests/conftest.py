from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pytest


@dataclass(frozen=True)
class TestEnvironment:
    repo_root: Path
    system: str
    release: str
    is_linux: bool
    is_windows: bool
    is_wsl: bool
    has_hardware: bool
    has_multi_host: bool
    has_kubernetes: bool
    has_volcano: bool
    has_hami: bool


OPT_IN_MARKERS: Final[dict[str, str]] = {
    "integration": "--run-integration",
    "hardware": "--run-hardware",
    "multi_host": "--run-multi-host",
    "windows": "--run-windows",
    "wsl": "--run-wsl",
    "kubernetes": "--run-kubernetes",
    "volcano": "--run-volcano",
    "hami": "--run-hami",
}

PATH_MARKERS: Final[dict[str, tuple[str, ...]]] = {
    "unit": ("unit", "local"),
    "contract": ("local",),
    "integration": ("integration",),
    "hardware": ("hardware",),
    "multi_host": ("multi_host",),
    "kubernetes": ("kubernetes",),
    "volcano": ("volcano",),
    "hami": ("hami",),
}


def _detect_test_environment() -> TestEnvironment:
    repo_root = Path(__file__).resolve().parent.parent
    release = platform.release().lower()
    system = platform.system()
    is_windows = system == "Windows"
    is_linux = system == "Linux"
    is_wsl = is_linux and ("microsoft" in release or "wsl" in release)
    return TestEnvironment(
        repo_root=repo_root,
        system=system,
        release=platform.release(),
        is_linux=is_linux,
        is_windows=is_windows,
        is_wsl=is_wsl,
        has_hardware=os.environ.get("SHARDGRID_ENABLE_HARDWARE_TESTS") == "1",
        has_multi_host=os.environ.get("SHARDGRID_ENABLE_MULTI_HOST_TESTS") == "1",
        has_kubernetes=os.environ.get("SHARDGRID_ENABLE_KUBERNETES_TESTS") == "1",
        has_volcano=os.environ.get("SHARDGRID_ENABLE_VOLCANO_TESTS") == "1",
        has_hami=os.environ.get("SHARDGRID_ENABLE_HAMI_TESTS") == "1",
    )


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("shardgrid")
    group.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="run integration tests",
    )
    group.addoption(
        "--run-hardware",
        action="store_true",
        default=False,
        help="run hardware tests when the local environment is prepared for them",
    )
    group.addoption(
        "--run-multi-host",
        action="store_true",
        default=False,
        help="run multi-host tests when the local environment is prepared for them",
    )
    group.addoption(
        "--run-windows",
        action="store_true",
        default=False,
        help="run tests that require a Windows host",
    )
    group.addoption(
        "--run-wsl",
        action="store_true",
        default=False,
        help="run tests that require a WSL runtime",
    )
    group.addoption(
        "--run-kubernetes",
        action="store_true",
        default=False,
        help="run Kubernetes tests when the local environment is prepared for them",
    )
    group.addoption(
        "--run-volcano",
        action="store_true",
        default=False,
        help="run Volcano tests when the local environment is prepared for them",
    )
    group.addoption(
        "--run-hami",
        action="store_true",
        default=False,
        help="run HAMi tests when the local environment is prepared for them",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    environment = _detect_test_environment()

    for item in items:
        for directory, markers in PATH_MARKERS.items():
            if directory in item.path.parts:
                for marker in markers:
                    item.add_marker(marker)

        for marker, option in OPT_IN_MARKERS.items():
            if marker in item.keywords and not config.getoption(option):
                item.add_marker(pytest.mark.skip(reason=f"{marker} tests require {option}"))

        if "windows" in item.keywords and not environment.is_windows:
            item.add_marker(pytest.mark.skip(reason="windows tests require a Windows host"))
        if "wsl" in item.keywords and not environment.is_wsl:
            item.add_marker(pytest.mark.skip(reason="wsl tests require a WSL runtime"))
        if "hardware" in item.keywords and not environment.has_hardware:
            item.add_marker(
                pytest.mark.skip(
                    reason="hardware tests require SHARDGRID_ENABLE_HARDWARE_TESTS=1"
                )
            )
        if "multi_host" in item.keywords and not environment.has_multi_host:
            item.add_marker(
                pytest.mark.skip(
                    reason="multi_host tests require SHARDGRID_ENABLE_MULTI_HOST_TESTS=1"
                )
            )
        if "kubernetes" in item.keywords and not environment.has_kubernetes:
            item.add_marker(
                pytest.mark.skip(
                    reason="kubernetes tests require SHARDGRID_ENABLE_KUBERNETES_TESTS=1"
                )
            )
        if "volcano" in item.keywords and not environment.has_volcano:
            item.add_marker(
                pytest.mark.skip(
                    reason="volcano tests require SHARDGRID_ENABLE_VOLCANO_TESTS=1"
                )
            )
        if "hami" in item.keywords and not environment.has_hami:
            item.add_marker(
                pytest.mark.skip(reason="hami tests require SHARDGRID_ENABLE_HAMI_TESTS=1")
            )


@pytest.fixture(scope="session")
def test_environment() -> TestEnvironment:
    return _detect_test_environment()


@pytest.fixture(scope="session")
def repo_root(test_environment: TestEnvironment) -> Path:
    return test_environment.repo_root


@pytest.fixture(scope="session")
def jobs_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("jobs-root")
