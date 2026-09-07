#!/usr/bin/env bash
# ShardGrid idempotent Ubuntu control-node bootstrap (T028)
#
# Safe behavior:
#   - Detect-first: never installs before recording what exists.
#   - Conda-first / reuse-first: reuses the active compatible Conda environment.
#   - Never runs sudo, apt-get, or any elevated/reboot/password action.
#   - Never runs conda create/remove (creating an environment is a manual action).
#   - Never modifies system Python.
#   - Only automatic mutation is optional pip-install of project dependencies
#     into the active Conda environment via --install-deps.
#
# Exit codes:
#   0 healthy
#   1 unexpected detection failure
#   2 blocked by manual action(s)
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FINDINGS_DIR="${SHARDGRID_BOOTSTRAP_DIR:-$HOME/.shardgrid/bootstrap}"

MODE="run"
INSTALL_DEPS=0
JSON_OUT=0

usage() {
    cat <<'EOF'
Usage: bootstrap-linux.sh [--check] [--install-deps] [--json] [--findings-dir DIR]

  --check            Detect and record only; never install anything.
  --install-deps     Install missing project dependencies into the active
                     Conda environment (safe official packages only).
  --json             Emit a JSON summary on stdout.
  --findings-dir DIR Override the findings directory
                     (default: ${SHARDGRID_BOOTSTRAP_DIR:-$HOME/.shardgrid/bootstrap}).
EOF
}

while (($# > 0)); do
    case "$1" in
        --check) MODE="check"; INSTALL_DEPS=0; shift ;;
        --install-deps) INSTALL_DEPS=1; shift ;;
        --json) JSON_OUT=1; shift ;;
        --findings-dir) FINDINGS_DIR="${2:-}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 1 ;;
    esac
done

mkdir -p "$FINDINGS_DIR"

TIMESTAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
HOST="$(hostname 2>/dev/null || echo unknown)"
OS_NAME="$(uname -srmo 2>/dev/null || uname -srm 2>/dev/null || echo unknown)"

MANUAL_ACTIONS=()
COMMANDS_RUN=()
HEALTH="healthy"

add_manual_action() {
    MANUAL_ACTIONS+=("$1")
    HEALTH="blocked_manual_action"
}

record_command() {
    COMMANDS_RUN+=("$*")
}

tool_version() {
    local tool="$1"
    local target="$2"
    shift 2
    if command -v "$tool" >/dev/null 2>&1; then
        COMMANDS_RUN+=("$tool $*")
        local out
        out="$("$@" 2>&1 || true)"
        printf -v "$target" '%s' "$(printf '%s' "$out" | head -n 1)"
    else
        printf -v "$target" '%s' "not_installed"
    fi
}

# --- Conda detection -------------------------------------------------------
CONDA_EXE="$(command -v conda 2>/dev/null || true)"
CONDA_VERSION=""
CONDA_ENV_LIST=""
CONDA_ENV_NAMES=()
CONDA_ENV_PATHS=()
CONDA_ACTIVE_ENV="${CONDA_DEFAULT_ENV:-}"
CONDA_PREFIX="${CONDA_PREFIX:-}"

if [[ -n "$CONDA_EXE" ]]; then
    CONDA_VERSION="$("$CONDA_EXE" --version 2>&1 || true)"
    CONDA_ENV_LIST="$("$CONDA_EXE" env list 2>&1 || true)"
    record_command "$CONDA_EXE" env list
else
    add_manual_action \
        "install conda (requires elevated administrator action; ShardGrid never installs Conda itself)"
fi

while IFS= read -r line; do
    [[ -z "$line" || "$line" =~ ^# ]] && continue
    CONDA_ENV_NAMES+=("$(awk '{print $1}' <<<"$line")")
    CONDA_ENV_PATHS+=("$(awk '{print $NF}' <<<"$line")")
done <<<"$CONDA_ENV_LIST"

if [[ -z "$CONDA_ACTIVE_ENV" && -n "$CONDA_PREFIX" ]]; then
    for i in "${!CONDA_ENV_PATHS[@]}"; do
        if [[ "${CONDA_ENV_PATHS[$i]}" == "$CONDA_PREFIX" ]]; then
            CONDA_ACTIVE_ENV="${CONDA_ENV_NAMES[$i]}"
            break
        fi
    done
fi

CONDA_ENV_NAMES_JOINED="$(IFS=$'\n'; printf '%s' "${CONDA_ENV_NAMES[*]}")"

# --- Python detection ------------------------------------------------------
PYTHON=""
PYTHON_CONDA_MANAGED="false"
if [[ -n "$CONDA_PREFIX" && -x "$CONDA_PREFIX/bin/python" ]]; then
    PYTHON="$CONDA_PREFIX/bin/python"
    PYTHON_CONDA_MANAGED="true"
else
    PYTHON="$(command -v python3 2>/dev/null || true)"
fi
PYTHON_VERSION=""
if [[ -n "$PYTHON" ]]; then
    PYTHON_VERSION="$("$PYTHON" --version 2>&1 || true)"
    record_command "$PYTHON" --version
fi

# --- Tool detection ----------------------------------------------------------
SSH_VERSION=""
GIT_VERSION=""
IPERF3_VERSION=""
tool_version ssh SSH_VERSION ssh -V
tool_version git GIT_VERSION git --version
tool_version iperf3 IPERF3_VERSION iperf3 --version

[[ "$SSH_VERSION" == "not_installed" ]] && add_manual_action \
    "install OpenSSH client (operator action: sudo apt-get install -y openssh-client)"
[[ "$GIT_VERSION" == "not_installed" ]] && add_manual_action \
    "install Git (operator action: sudo apt-get install -y git)"
[[ "$IPERF3_VERSION" == "not_installed" ]] && add_manual_action \
    "install iperf3 (operator action: sudo apt-get install -y iperf3)"

# --- Disk ---------------------------------------------------------------------
DISK_PATH="$HOME"
DISK_FREE_KB=""
if command -v df >/dev/null 2>&1; then
    DISK_FREE_KB="$(df -k "$DISK_PATH" 2>/dev/null | tail -n 1 | awk '{print $4}' || true)"
fi

# --- Project dependencies -----------------------------------------------------
deps_ok() {
    local python_bin="${1:-}"
    [[ -z "$python_bin" ]] && return 1
    PYTHONPATH="$REPO_ROOT/src" "$python_bin" -c "import yaml, shardgrid" >/dev/null 2>&1
}

DEPS_SG="missing"
DEPS_YAML="missing"
DEPS_PYTEST="missing"
DEPS_RUFF="missing"
DEPS_MYPY="missing"
DEPS_OVERALL="missing"
if [[ -n "$PYTHON" ]]; then
    if PYTHONPATH="$REPO_ROOT/src" "$PYTHON" -c "import yaml, shardgrid" >/dev/null 2>&1; then
        DEPS_SG="present"
        DEPS_YAML="present"
    fi
    for dep in pytest ruff mypy; do
        if "$PYTHON" -c "import $dep" >/dev/null 2>&1; then
            case "$dep" in
                pytest) DEPS_PYTEST="present" ;;
                ruff) DEPS_RUFF="present" ;;
                mypy) DEPS_MYPY="present" ;;
            esac
        fi
    done
    if [[ "$DEPS_SG" == "present" && "$DEPS_YAML" == "present" && "$DEPS_PYTEST" == "present" ]]; then
        DEPS_OVERALL="present"
    fi
fi

# --- Reuse / create decision --------------------------------------------------
DECISION_REUSE=""
DECISION_CREATE_NEEDED="false"

if [[ -n "$CONDA_EXE" ]]; then
    if [[ -n "$CONDA_ACTIVE_ENV" && "$PYTHON_CONDA_MANAGED" == "true" ]]; then
        if deps_ok "$PYTHON"; then
            DECISION_REUSE="$CONDA_ACTIVE_ENV"
        else
            for i in "${!CONDA_ENV_PATHS[@]}"; do
                env_py="${CONDA_ENV_PATHS[$i]}/bin/python"
                if deps_ok "$env_py"; then
                    DECISION_REUSE="${CONDA_ENV_NAMES[$i]}"
                    break
                fi
            done
            if [[ -z "$DECISION_REUSE" ]]; then
                DECISION_CREATE_NEEDED="true"
                add_manual_action \
                    "create a ShardGrid Conda environment (conda create is a manual action; operator runs: conda create -n shardgrid -y python)"
            fi
        fi
    elif [[ -n "$CONDA_PREFIX" ]]; then
        DECISION_CREATE_NEEDED="true"
        add_manual_action \
            "create a ShardGrid Conda environment (conda create is a manual action; operator runs: conda create -n shardgrid -y python)"
    fi
fi

if [[ -n "$CONDA_EXE" && -n "$CONDA_ACTIVE_ENV" && -z "$DECISION_REUSE" && -z "$DECISION_CREATE_NEEDED" ]]; then
    DECISION_REUSE="$CONDA_ACTIVE_ENV"
fi

# --- Optional safe install of project dependencies ------------------------------
if [[ "$INSTALL_DEPS" == "1" && "$MODE" != "check" && -n "$CONDA_PREFIX" ]]; then
    if [[ "$DEPS_OVERALL" == "missing" && "$HEALTH" != "blocked_manual_action" ]]; then
        record_command "$PYTHON" -m pip install -e "$REPO_ROOT[dev,test]"
        if "$PYTHON" -m pip install -e "$REPO_ROOT[dev,test]" >/dev/null 2>&1; then
            DEPS_SG="present"
            DEPS_YAML="present"
            DEPS_PYTEST="present"
            DEPS_OVERALL="present"
        else
            add_manual_action \
                "install project dependencies into the selected Conda environment (operator action: python -m pip install -e '$REPO_ROOT[dev,test]')"
        fi
    fi
fi

if [[ "$DEPS_OVERALL" == "missing" && "$HEALTH" == "healthy" ]]; then
    HEALTH="degraded"
fi

# --- Serialize findings ---------------------------------------------------------
JSON_PYTHON="$PYTHON"
[[ -z "$JSON_PYTHON" ]] && JSON_PYTHON="$(command -v python3 2>/dev/null || true)"

export BOOTSTRAP_SCRIPT="bootstrap-linux.sh"
export BOOTSTRAP_MODE="$MODE"
export BOOTSTRAP_TIMESTAMP="$TIMESTAMP"
export BOOTSTRAP_HOST="$HOST"
export BOOTSTRAP_OS="$OS_NAME"
export BOOTSTRAP_REPO_ROOT="$REPO_ROOT"
export BOOTSTRAP_CONDA_EXE="$CONDA_EXE"
export BOOTSTRAP_CONDA_VERSION="$CONDA_VERSION"
export BOOTSTRAP_CONDA_ENVS="$CONDA_ENV_NAMES_JOINED"
export BOOTSTRAP_CONDA_ACTIVE_ENV="$CONDA_ACTIVE_ENV"
export BOOTSTRAP_CONDA_PREFIX="$CONDA_PREFIX"
export BOOTSTRAP_PYTHON="$PYTHON"
export BOOTSTRAP_PYTHON_VERSION="$PYTHON_VERSION"
export BOOTSTRAP_PYTHON_CONDA_MANAGED="$PYTHON_CONDA_MANAGED"
export BOOTSTRAP_SSH="$SSH_VERSION"
export BOOTSTRAP_GIT="$GIT_VERSION"
export BOOTSTRAP_IPERF3="$IPERF3_VERSION"
export BOOTSTRAP_DISK_PATH="$DISK_PATH"
export BOOTSTRAP_DISK_FREE_KB="$DISK_FREE_KB"
export BOOTSTRAP_DEPS_SG="$DEPS_SG"
export BOOTSTRAP_DEPS_YAML="$DEPS_YAML"
export BOOTSTRAP_DEPS_PYTEST="$DEPS_PYTEST"
export BOOTSTRAP_DEPS_RUFF="$DEPS_RUFF"
export BOOTSTRAP_DEPS_MYPY="$DEPS_MYPY"
export BOOTSTRAP_DECISION_REUSE="$DECISION_REUSE"
export BOOTSTRAP_DECISION_CREATE_NEEDED="$DECISION_CREATE_NEEDED"
export BOOTSTRAP_HEALTH="$HEALTH"
MANUAL_ACTIONS_JOINED="$(IFS=$'\x1f'; printf '%s' "${MANUAL_ACTIONS[*]}")"
COMMANDS_RUN_JOINED="$(IFS=$'\x1f'; printf '%s' "${COMMANDS_RUN[*]}")"
export BOOTSTRAP_MANUAL_ACTIONS="$MANUAL_ACTIONS_JOINED"
export BOOTSTRAP_COMMANDS_RUN="$COMMANDS_RUN_JOINED"

if [[ -n "$JSON_PYTHON" ]]; then
    TIMESTAMPED_FILE="$FINDINGS_DIR/findings-$TIMESTAMP.json"
    LATEST_FILE="$FINDINGS_DIR/latest.json"
    "$JSON_PYTHON" - "$TIMESTAMPED_FILE" "$LATEST_FILE" <<'PY'
import json
import os
import sys


def get_list(key, sep="\x1f"):
    value = os.environ.get(key, "")
    return [item for item in value.split(sep) if item] if value else []


def optional(value):
    return value if value else None


data = {
    "script": os.environ.get("BOOTSTRAP_SCRIPT", ""),
    "mode": os.environ.get("BOOTSTRAP_MODE", "run"),
    "timestamp": os.environ.get("BOOTSTRAP_TIMESTAMP", ""),
    "host": os.environ.get("BOOTSTRAP_HOST", ""),
    "os": os.environ.get("BOOTSTRAP_OS", ""),
    "repo_root": os.environ.get("BOOTSTRAP_REPO_ROOT", ""),
    "conda": {
        "executable": optional(os.environ.get("BOOTSTRAP_CONDA_EXE")),
        "version": optional(os.environ.get("BOOTSTRAP_CONDA_VERSION")),
        "environments": get_list("BOOTSTRAP_CONDA_ENVS", "\n"),
        "active_environment": optional(os.environ.get("BOOTSTRAP_CONDA_ACTIVE_ENV")),
        "active_prefix": optional(os.environ.get("BOOTSTRAP_CONDA_PREFIX")),
    },
    "python": {
        "executable": optional(os.environ.get("BOOTSTRAP_PYTHON")),
        "version": optional(os.environ.get("BOOTSTRAP_PYTHON_VERSION")),
        "conda_managed": os.environ.get("BOOTSTRAP_PYTHON_CONDA_MANAGED") == "true",
    },
    "tools": {
        "ssh": os.environ.get("BOOTSTRAP_SSH", "not_checked"),
        "git": os.environ.get("BOOTSTRAP_GIT", "not_checked"),
        "iperf3": os.environ.get("BOOTSTRAP_IPERF3", "not_checked"),
    },
    "disk": {
        "path": optional(os.environ.get("BOOTSTRAP_DISK_PATH")),
        "free_bytes": (
            int(os.environ["BOOTSTRAP_DISK_FREE_KB"]) * 1024
            if os.environ.get("BOOTSTRAP_DISK_FREE_KB")
            else None
        ),
    },
    "project_dependencies": {
        "shardgrid_importable": os.environ.get("BOOTSTRAP_DEPS_SG", "missing") == "present",
        "pyyaml_available": os.environ.get("BOOTSTRAP_DEPS_YAML", "missing") == "present",
        "pytest_available": os.environ.get("BOOTSTRAP_DEPS_PYTEST", "missing") == "present",
        "ruff_available": os.environ.get("BOOTSTRAP_DEPS_RUFF", "missing") == "present",
        "mypy_available": os.environ.get("BOOTSTRAP_DEPS_MYPY", "missing") == "present",
    },
    "decision": {
        "reuse_environment": optional(os.environ.get("BOOTSTRAP_DECISION_REUSE")),
        "create_needed": os.environ.get("BOOTSTRAP_DECISION_CREATE_NEEDED") == "true",
    },
    "health": os.environ.get("BOOTSTRAP_HEALTH", "unknown"),
    "manual_actions": get_list("BOOTSTRAP_MANUAL_ACTIONS"),
    "commands_run": get_list("BOOTSTRAP_COMMANDS_RUN"),
}

for path in sys.argv[1:]:
    with open(path, "w") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
PY
fi

# --- Human report ----------------------------------------------------------------
if [[ "$JSON_OUT" == "1" && -n "$JSON_PYTHON" ]]; then
    "$JSON_PYTHON" -c "import json, os; print(json.dumps(json.load(open(os.path.expanduser('$FINDINGS_DIR/latest.json'))), indent=2, sort_keys=True))"
fi

{
    echo "ShardGrid Linux bootstrap ($MODE mode)"
    echo "host: $HOST | os: $OS_NAME | timestamp: $TIMESTAMP"
    echo "conda: ${CONDA_EXE:-not_found} ${CONDA_VERSION:+($CONDA_VERSION)}"
    echo "  environments: ${CONDA_ENV_NAMES[*]:-none}"
    echo "  active: ${CONDA_ACTIVE_ENV:-none} prefix: ${CONDA_PREFIX:-none}"
    echo "python: ${PYTHON:-not_found} ${PYTHON_VERSION:+($PYTHON_VERSION)} conda_managed=$PYTHON_CONDA_MANAGED"
    echo "tools: ssh=$SSH_VERSION | git=$GIT_VERSION | iperf3=$IPERF3_VERSION"
    echo "disk: path=$DISK_PATH free_kb=${DISK_FREE_KB:-unknown}"
    echo "deps: shardgrid=$DEPS_SG yaml=$DEPS_YAML pytest=$DEPS_PYTEST ruff=$DEPS_RUFF mypy=$DEPS_MYPY"
    echo "decision: reuse=${DECISION_REUSE:-none} create_needed=$DECISION_CREATE_NEEDED"
    echo "health: $HEALTH"
    if ((${#MANUAL_ACTIONS[@]} > 0)); then
        echo "manual actions:"
        for action in "${MANUAL_ACTIONS[@]}"; do
            echo "  - $action"
        done
    fi
    echo "findings: $FINDINGS_DIR"
} >&2

case "$HEALTH" in
    blocked_manual_action) exit 2 ;;
    healthy) exit 0 ;;
    *) exit 1 ;;
esac