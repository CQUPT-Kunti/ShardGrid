from __future__ import annotations

from shardgrid.platforms.linux import LinuxPlatform
from shardgrid.platforms.windows import WindowsPlatform
from shardgrid.platforms.wsl import WSLPlatform


def test_linux_path_join_normal_parts() -> None:
    path = LinuxPlatform().path_join("jobs", "job-0001", "logs")

    assert path == "jobs/job-0001/logs"


def test_linux_path_join_preserves_spaces() -> None:
    path = LinuxPlatform().path_join("jobs root", "my job 42", "log files")

    assert path == "jobs root/my job 42/log files"


def test_linux_path_join_preserves_special_characters() -> None:
    path = LinuxPlatform().path_join("a#b", "c$d", "e&f", "g'h", 'i"j')

    assert path == "a#b/c$d/e&f/g'h/i\"j"


def test_linux_path_join_absolute_posix_path() -> None:
    path = LinuxPlatform().path_join("/opt", "shardgrid", "jobs")

    assert path == "/opt/shardgrid/jobs"


def test_linux_path_join_is_deterministic_and_complete() -> None:
    components = ("a b", "c d", "e f")
    first = LinuxPlatform().path_join(*components)
    second = LinuxPlatform().path_join(*components)

    assert first == second
    for component in components:
        assert component in first


def test_windows_path_join_drive_rooted_path() -> None:
    path = WindowsPlatform().path_join("C:/", "Users", "alice", "AppData")

    assert path == "C:\\Users\\alice\\AppData"


def test_windows_path_join_drive_path_with_spaces() -> None:
    path = WindowsPlatform().path_join("C:/", "Program Files", "ShardGrid 0.1")

    assert path == "C:\\Program Files\\ShardGrid 0.1"


def test_windows_path_join_uses_backslash_separators() -> None:
    path = WindowsPlatform().path_join("jobs", "job-0001", "logs")

    assert path == "jobs\\job-0001\\logs"


def test_windows_path_join_preserves_special_characters() -> None:
    path = WindowsPlatform().path_join("C:/", "a#b", "c$d", "e&f", "g'h", 'i"j')

    assert path == "C:\\a#b\\c$d\\e&f\\g'h\\i\"j"


def test_windows_path_join_is_deterministic_and_complete() -> None:
    components = ("C:/", "a b", "c d", "e f")
    first = WindowsPlatform().path_join(*components)
    second = WindowsPlatform().path_join(*components)

    assert first == second
    assert "a b" in first
    assert "c d" in first
    assert "e f" in first


def test_wsl_path_join_posix_with_spaces() -> None:
    path = WSLPlatform().path_join("home", "alice", "my jobs")

    assert path == "home/alice/my jobs"


def test_wsl_path_join_mntc_windows_view() -> None:
    path = WSLPlatform().path_join("/mnt/c", "Users", "alice")

    assert path == "/mnt/c/Users/alice"


def test_wsl_path_join_preserves_special_characters() -> None:
    path = WSLPlatform().path_join("a#b", "c$d", "e&f", "g'h", 'i"j')

    assert path == "a#b/c$d/e&f/g'h/i\"j"


def test_linux_and_windows_paths_never_cross_contaminate() -> None:
    posix = LinuxPlatform().path_join("jobs", "my job")
    win = WindowsPlatform().path_join("jobs", "my job")

    assert "\\" not in posix
    assert "/" not in win