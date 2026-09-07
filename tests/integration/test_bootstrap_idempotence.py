from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
BASH = shutil.which("bash") or "/bin/bash"
LINUX_BOOTSTRAP = ROOT / "scripts" / "bootstrap-linux.sh"
WINDOWS_BOOTSTRAP = ROOT / "scripts" / "bootstrap-windows.ps1"
WSL_BOOTSTRAP = ROOT / "scripts" / "bootstrap-wsl.sh"


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _link_core_tools(fake_bin: Path, extra: tuple[str, ...] = ()) -> None:
    commands = (
        "awk",
        "date",
        "df",
        "head",
        "hostname",
        "mkdir",
        "tail",
        "uname",
        "python3",
    ) + extra
    for name in commands:
        target = shutil.which(name)
        if target is None:
            raise AssertionError(f"missing system tool for test harness: {name}")
        (fake_bin / name).symlink_to(target)


def _fake_python_source() -> str:
    return textwrap.dedent(
        f"""\
        #!/usr/bin/env python3
        import json
        import os
        import subprocess
        import sys
        from pathlib import Path

        REAL_PYTHON = {sys.executable!r}
        STATE_PATH = Path(__file__).resolve().parents[1] / ".fake-python-state.json"

        def load_state():
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))

        def save_state(state):
            STATE_PATH.write_text(json.dumps(state), encoding="utf-8")

        def pip_install(state, args):
            joined = " ".join(args)
            if "PyYAML" in joined:
                state["yaml"] = True
            if "torch" in joined:
                state["torch"] = True
            if "-e" in args:
                state["yaml"] = True
                state["shardgrid"] = True
                state["pytest"] = True
                state["ruff"] = True
                state["mypy"] = True
            save_state(state)
            return 0

        def import_check(state, code):
            if "import yaml, shardgrid" in code:
                return 0 if state.get("yaml") and state.get("shardgrid") else 1
            if "import yaml" in code and "shardgrid" not in code:
                return 0 if state.get("yaml") else 1
            for module in ("pytest", "ruff", "mypy"):
                if f"import {{module}}" in code:
                    return 0 if state.get(module, False) else 1
            if "torch.cuda.is_available()" in code and "print('torch_version'" in code:
                print("torch_version", state.get("torch_version", "missing"))
                print("cuda_version", state.get("cuda_version", "not_checked"))
                print("cuda_available", bool(state.get("cuda_available", False)))
                return 0
            return None

        def stdin_check(state, script):
            if 'find_spec("yaml")' in script and 'torch.cuda.is_available' in script:
                ready = state.get("yaml") and state.get("torch") and state.get("cuda_available")
                return 0 if ready else 1
            if "import shardgrid, yaml" in script:
                return 0 if state.get("yaml") and state.get("shardgrid") else 1
            if (
                'print(json.dumps({{"status": "missing"}}))' in script
                and "cuda_available" in script
            ):
                if not state.get("torch"):
                    print(json.dumps({{"status": "missing"}}))
                else:
                    print(json.dumps({{
                        "status": "present",
                        "version": state.get("torch_version"),
                        "cuda_version": state.get("cuda_version"),
                        "cuda_available": bool(state.get("cuda_available", False)),
                    }}))
                return 0
            if "import yaml" in script:
                return 0 if state.get("yaml") else 1
            return None

        def main():
            state = load_state()
            args = sys.argv[1:]
            if args == ["--version"]:
                print(state.get("python_version", "Python 3.12.13"))
                return 0
            if len(args) >= 3 and args[0] == "-m" and args[1] == "pip" and args[2] == "install":
                return pip_install(state, args[3:])
            if len(args) >= 2 and args[0] == "-c":
                handled = import_check(state, args[1])
                if handled is not None:
                    return handled
                return subprocess.call([REAL_PYTHON] + args, env=os.environ.copy())
            if args and args[0] == "-":
                script = sys.stdin.read()
                handled = stdin_check(state, script)
                if handled is not None:
                    return handled
                completed = subprocess.run(
                    [REAL_PYTHON] + args,
                    input=script,
                    text=True,
                    env=os.environ.copy(),
                    check=False,
                )
                return completed.returncode
            completed = subprocess.run([REAL_PYTHON] + args, check=False)
            return completed.returncode

        raise SystemExit(main())
        """
    )


def _create_fake_env(prefix: Path, state: dict[str, Any]) -> None:
    bin_dir = prefix / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    (prefix / ".fake-python-state.json").write_text(json.dumps(state), encoding="utf-8")
    source = _fake_python_source()
    _write_executable(bin_dir / "python", source)
    _write_executable(bin_dir / "python3", source)


def _write_fake_conda(
    fake_bin: Path,
    env_root: Path,
    template_dir: Path,
    create_state: dict[str, Any],
) -> None:
    script = textwrap.dedent(
        f"""\
        #!/usr/bin/env python3
        import json
        import shutil
        import sys
        from pathlib import Path

        ENV_ROOT = Path({str(env_root)!r})
        TEMPLATE_DIR = Path({str(template_dir)!r})
        CREATE_STATE = {json.dumps(create_state)!r}

        def env_list():
            for prefix in sorted(ENV_ROOT.iterdir()):
                if (prefix / "bin" / "python").exists():
                    print(f"{{prefix.name}} {{prefix}}")

        def create_env(args):
            name = args[args.index("-n") + 1]
            target = ENV_ROOT / name
            if not target.exists():
                shutil.copytree(TEMPLATE_DIR, target)
                (target / ".fake-python-state.json").write_text(CREATE_STATE, encoding="utf-8")

        def main():
            args = sys.argv[1:]
            if args == ["--version"]:
                print("conda 26.5.3")
                return 0
            if args[:2] == ["env", "list"]:
                env_list()
                return 0
            if args and args[0] == "create":
                create_env(args)
                return 0
            raise SystemExit(f"unexpected fake conda args: {{args}}")

        raise SystemExit(main())
        """
    )
    _write_executable(fake_bin / "conda", script)


def _write_fake_echo_tool(fake_bin: Path, name: str, lines: str) -> None:
    body = textwrap.dedent(
        f"""\
        #!/usr/bin/env bash
        cat <<'EOF'
        {lines}
        EOF
        """
    )
    _write_executable(fake_bin / name, body)


def _write_fake_nvidia_smi(fake_bin: Path, summary: str) -> None:
    body = textwrap.dedent(
        f"""\
        #!/usr/bin/env bash
        if [[ "$*" == *"--query-gpu=name,driver_version"* ]]; then
            echo "{summary}"
        else
            echo "Mon Aug 17 13:00:00 2026       "
        fi
        """
    )
    _write_executable(fake_bin / "nvidia-smi", body)


def _write_fake_root_apt_get(fake_bin: Path) -> None:
    body = textwrap.dedent(
        """\
        #!/usr/bin/env bash
        set -euo pipefail
        if [[ "${1:-}" == "update" ]]; then
            exit 0
        fi
        if [[ "${1:-}" == "install" && "${2:-}" == "-y" && "${3:-}" == "iperf3" ]]; then
            cat >"$(dirname "$0")/iperf3" <<'EOF'
        #!/usr/bin/env bash
        echo "iperf 3.20 (cJSON 1.7.15)"
        EOF
            chmod +x "$(dirname "$0")/iperf3"
            exit 0
        fi
        exit 1
        """
    )
    _write_executable(fake_bin / "apt-get", body)


def _write_fake_root_id(fake_bin: Path) -> None:
    body = textwrap.dedent(
        """\
        #!/usr/bin/env bash
        if [[ "${1:-}" == "-u" ]]; then
            echo 0
            exit 0
        fi
        /usr/bin/id "$@"
        """
    )
    _write_executable(fake_bin / "id", body)


def _run_bootstrap(
    script: Path,
    args: list[str],
    env: dict[str, str],
    findings_dir: Path,
) -> tuple[int, dict[str, Any]]:
    command = [BASH, str(script), *args, "--json", "--findings-dir", str(findings_dir)]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    payload = json.loads(completed.stdout)
    return completed.returncode, payload


def _base_env(tmp_path: Path, fake_bin: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["HOME"] = str(tmp_path / "home")
    env["PATH"] = f"{fake_bin}:{env.get('PATH', '')}"
    return env


def _payload_signature(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "health": payload["health"],
        "manual_actions": payload["manual_actions"],
        "decision": payload["decision"],
        "conda": payload["conda"],
        "python": payload["python"],
        "runtime_tools": payload.get("runtime_tools"),
        "torch": payload.get("torch"),
        "project_dependencies": payload.get("project_dependencies"),
    }


def test_linux_bootstrap_reuses_existing_compatible_conda_environment(tmp_path: Path) -> None:
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    _link_core_tools(fake_bin)
    _write_fake_echo_tool(fake_bin, "ssh", "OpenSSH_9.9p1, OpenSSL 3.0.0")
    _write_fake_echo_tool(fake_bin, "git", "git version 2.53.0")
    _write_fake_echo_tool(fake_bin, "iperf3", "iperf 3.20 (cJSON 1.7.15)")

    env_root = tmp_path / "envs"
    compatible = env_root / "compatible"
    base = env_root / "base"
    _create_fake_env(
        compatible,
        {
            "python_version": "Python 3.12.13",
            "yaml": True,
            "shardgrid": True,
            "pytest": True,
            "ruff": True,
            "mypy": True,
            "torch": False,
            "cuda_available": False,
        },
    )
    _create_fake_env(
        base,
        {
            "python_version": "Python 3.14.6",
            "yaml": False,
            "shardgrid": False,
            "pytest": False,
            "ruff": False,
            "mypy": False,
            "torch": False,
            "cuda_available": False,
        },
    )
    template_dir = tmp_path / "template-env"
    _create_fake_env(template_dir, {"python_version": "Python 3.12.13"})
    _write_fake_conda(fake_bin, env_root, template_dir, {"python_version": "Python 3.12.13"})

    env = _base_env(tmp_path, fake_bin)
    env["CONDA_DEFAULT_ENV"] = "compatible"
    env["CONDA_PREFIX"] = str(compatible)

    first_code, first_payload = _run_bootstrap(
        LINUX_BOOTSTRAP, ["--check"], env, tmp_path / "linux-1"
    )
    second_code, second_payload = _run_bootstrap(
        LINUX_BOOTSTRAP, ["--check"], env, tmp_path / "linux-2"
    )

    assert first_code == 0 == second_code
    assert first_payload["decision"]["reuse_environment"] == "compatible"
    assert first_payload["conda"]["active_environment"] == "compatible"
    assert _payload_signature(first_payload) == _payload_signature(second_payload)


def test_linux_bootstrap_repairs_partial_installation_without_creating_environment(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    _link_core_tools(fake_bin)
    _write_fake_echo_tool(fake_bin, "ssh", "OpenSSH_9.9p1, OpenSSL 3.0.0")
    _write_fake_echo_tool(fake_bin, "git", "git version 2.53.0")
    _write_fake_echo_tool(fake_bin, "iperf3", "iperf 3.20 (cJSON 1.7.15)")

    env_root = tmp_path / "envs"
    active = env_root / "active"
    _create_fake_env(
        active,
        {
            "python_version": "Python 3.12.13",
            "yaml": True,
            "shardgrid": True,
            "pytest": False,
            "ruff": False,
            "mypy": False,
            "torch": False,
            "cuda_available": False,
        },
    )
    template_dir = tmp_path / "template-env"
    _create_fake_env(template_dir, {"python_version": "Python 3.12.13"})
    _write_fake_conda(fake_bin, env_root, template_dir, {"python_version": "Python 3.12.13"})

    env = _base_env(tmp_path, fake_bin)
    env["CONDA_DEFAULT_ENV"] = "active"
    env["CONDA_PREFIX"] = str(active)

    code, payload = _run_bootstrap(
        LINUX_BOOTSTRAP,
        ["--install-deps"],
        env,
        tmp_path / "linux-install",
    )

    assert code == 0
    assert payload["health"] == "healthy"
    assert payload["decision"]["reuse_environment"] == "active"
    assert payload["project_dependencies"]["shardgrid_importable"] is True
    assert payload["project_dependencies"]["pytest_available"] is True
    assert payload["manual_actions"] == []


def test_linux_bootstrap_reports_manual_actions_for_missing_conda_and_tools(tmp_path: Path) -> None:
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    _link_core_tools(fake_bin)

    env = _base_env(tmp_path, fake_bin)
    env["PATH"] = str(fake_bin)

    code, payload = _run_bootstrap(LINUX_BOOTSTRAP, ["--check"], env, tmp_path / "linux-missing")

    assert code == 2
    assert payload["health"] == "blocked_manual_action"
    assert any("install conda" in action for action in payload["manual_actions"])
    assert any("install OpenSSH client" in action for action in payload["manual_actions"])
    assert any("install Git" in action for action in payload["manual_actions"])
    assert any("install iperf3" in action for action in payload["manual_actions"])


def test_wsl_bootstrap_reuses_compatible_runtime_and_is_idempotent(tmp_path: Path) -> None:
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    _link_core_tools(fake_bin)
    _write_fake_echo_tool(fake_bin, "lsb_release", "Ubuntu 26.04 LTS")
    _write_fake_echo_tool(fake_bin, "git", "git version 2.53.0")
    _write_fake_echo_tool(fake_bin, "iperf3", "iperf 3.20 (cJSON 1.7.15)")
    _write_fake_nvidia_smi(fake_bin, "NVIDIA GeForce RTX 4060 Laptop GPU, 566.07")

    env_root = tmp_path / "envs"
    base = env_root / "base"
    shardgrid = env_root / "shardgrid"
    _create_fake_env(
        base,
        {
            "python_version": "Python 3.14.6",
            "yaml": False,
            "shardgrid": False,
            "pytest": False,
            "ruff": False,
            "mypy": False,
            "torch": True,
            "torch_version": "2.13.0+cu130",
            "cuda_version": "13.0",
            "cuda_available": False,
        },
    )
    _create_fake_env(
        shardgrid,
        {
            "python_version": "Python 3.12.13",
            "yaml": True,
            "shardgrid": True,
            "pytest": True,
            "ruff": True,
            "mypy": True,
            "torch": True,
            "torch_version": "2.7.1+cu118",
            "cuda_version": "11.8",
            "cuda_available": True,
        },
    )
    template_dir = tmp_path / "template-env"
    _create_fake_env(template_dir, {"python_version": "Python 3.12.13"})
    _write_fake_conda(fake_bin, env_root, template_dir, {"python_version": "Python 3.12.13"})

    env = _base_env(tmp_path, fake_bin)

    first_code, first_payload = _run_bootstrap(
        WSL_BOOTSTRAP, ["--check"], env, tmp_path / "wsl-1"
    )
    second_code, second_payload = _run_bootstrap(
        WSL_BOOTSTRAP, ["--check"], env, tmp_path / "wsl-2"
    )

    assert first_code == 0 == second_code
    assert first_payload["conda"]["selected_environment"] == "shardgrid"
    assert first_payload["torch"]["version"] == "2.7.1+cu118"
    assert first_payload["torch"]["cuda_available"] == "true"
    assert _payload_signature(first_payload) == _payload_signature(second_payload)


def test_wsl_bootstrap_install_deps_installs_iperf3_when_already_root(tmp_path: Path) -> None:
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    _link_core_tools(fake_bin, extra=("bash", "cat", "chmod", "cp", "dirname", "grep", "pwd"))
    _write_fake_echo_tool(fake_bin, "lsb_release", "Ubuntu 26.04 LTS")
    _write_fake_echo_tool(fake_bin, "git", "git version 2.53.0")
    _write_fake_nvidia_smi(fake_bin, "NVIDIA GeForce RTX 4060 Laptop GPU, 566.07")
    _write_fake_root_apt_get(fake_bin)
    _write_fake_root_id(fake_bin)

    env_root = tmp_path / "envs"
    shardgrid = env_root / "shardgrid"
    _create_fake_env(
        shardgrid,
        {
            "python_version": "Python 3.12.13",
            "yaml": True,
            "shardgrid": True,
            "pytest": True,
            "ruff": True,
            "mypy": True,
            "torch": True,
            "torch_version": "2.7.1+cu118",
            "cuda_version": "11.8",
            "cuda_available": True,
        },
    )
    template_dir = tmp_path / "template-env"
    _create_fake_env(template_dir, {"python_version": "Python 3.12.13"})
    _write_fake_conda(fake_bin, env_root, template_dir, {"python_version": "Python 3.12.13"})

    env = _base_env(tmp_path, fake_bin)
    env["PATH"] = str(fake_bin)

    code, payload = _run_bootstrap(
        WSL_BOOTSTRAP,
        ["--install-deps"],
        env,
        tmp_path / "wsl-install-iperf3",
    )

    assert code == 0
    assert payload["health"] == "healthy"
    assert payload["runtime_tools"]["iperf3"].startswith("iperf 3.20")
    assert payload["manual_actions"] == []


def test_wsl_bootstrap_creates_shardgrid_environment_only_when_needed(tmp_path: Path) -> None:
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    _link_core_tools(fake_bin)
    _write_fake_echo_tool(fake_bin, "lsb_release", "Ubuntu 26.04 LTS")
    _write_fake_echo_tool(fake_bin, "git", "git version 2.53.0")
    _write_fake_echo_tool(fake_bin, "iperf3", "iperf 3.20 (cJSON 1.7.15)")
    _write_fake_nvidia_smi(fake_bin, "NVIDIA GeForce GTX 1650, 527.41")

    env_root = tmp_path / "envs"
    keepme = env_root / "keepme"
    _create_fake_env(
        keepme,
        {
            "python_version": "Python 3.14.6",
            "yaml": True,
            "shardgrid": True,
            "pytest": True,
            "ruff": True,
            "mypy": True,
            "torch": True,
            "torch_version": "2.13.0+cu130",
            "cuda_version": "13.0",
            "cuda_available": False,
        },
    )
    template_dir = tmp_path / "template-env"
    _create_fake_env(
        template_dir,
        {
            "python_version": "Python 3.12.13",
            "yaml": False,
            "shardgrid": True,
            "pytest": True,
            "ruff": True,
            "mypy": True,
            "torch": False,
            "torch_version": "2.7.1+cu118",
            "cuda_version": "11.8",
            "cuda_available": True,
        },
    )
    _write_fake_conda(
        fake_bin,
        env_root,
        template_dir,
        {
            "python_version": "Python 3.12.13",
            "yaml": False,
            "shardgrid": True,
            "pytest": True,
            "ruff": True,
            "mypy": True,
            "torch": False,
            "torch_version": "2.7.1+cu118",
            "cuda_version": "11.8",
            "cuda_available": True,
        },
    )

    env = _base_env(tmp_path, fake_bin)

    run_code, run_payload = _run_bootstrap(
        WSL_BOOTSTRAP, [], env, tmp_path / "wsl-run"
    )
    check_code, check_payload = _run_bootstrap(
        WSL_BOOTSTRAP, ["--check"], env, tmp_path / "wsl-check"
    )

    assert run_code == 0 == check_code
    assert run_payload["conda"]["selected_environment"] == "shardgrid"
    assert check_payload["conda"]["selected_environment"] == "shardgrid"
    assert check_payload["torch"]["cuda_available"] == "true"
    assert (keepme / "bin" / "python").exists()
    assert (env_root / "shardgrid" / "bin" / "python").exists()


def test_windows_bootstrap_contract_keeps_host_conda_and_wsl_conda_separate() -> None:
    text = WINDOWS_BOOTSTRAP.read_text(encoding="utf-8")

    assert "host_only_not_training_runtime" in text
    assert "training_runtime" in text
    assert "install OpenSSH Client Windows capability" in text
    assert "reboot may be required" in text
    assert "do not use Windows-host Conda for training" in text
