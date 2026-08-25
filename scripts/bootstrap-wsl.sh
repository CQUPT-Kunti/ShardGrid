#!/usr/bin/env bash
# ShardGrid idempotent WSL2 Ubuntu training-runtime bootstrap (T030)
#
# Safe behavior:
#   - Detect-first: records runtime identity, Conda, Python, ShardGrid deps,
#     PyTorch, CUDA visibility, nvidia-smi, iperf3, and basic runtime tools.
#   - Conda-first / reuse-first: reuses an active compatible Conda environment,
#     or another existing compatible environment if one already exists.
#   - Creates or repairs a dedicated "shardgrid" Conda environment only when
#     no existing environment provides a CUDA-ready PyTorch runtime.
#   - Never installs a Linux NVIDIA display driver inside WSL.
#   - Never overwrites or deletes an existing Conda environment.
#   - Never runs sudo/apt or any password-prompting action.
#   - Safe automatic mutation is limited to user-space Conda/pip operations
#     inside the selected WSL training environment, plus optional MTU
#     configuration when the script is already running with root privileges.
#
# Exit codes:
#   0 healthy
#   1 unexpected detection failure or degraded
#   2 blocked by manual action(s)
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." 2>/dev/null && pwd || true)"
FINDINGS_DIR="${SHARDGRID_BOOTSTRAP_DIR:-$HOME/.shardgrid/bootstrap}"

MODE="run"
INSTALL_DEPS=0
JSON_OUT=0
RUNTIME_ENV_NAME="${SHARDGRID_RUNTIME_ENV_NAME:-shardgrid}"
RUNTIME_PYTHON_SPEC="${SHARDGRID_RUNTIME_PYTHON_SPEC:-python=3.12}"
TORCH_INDEX_URL="${SHARDGRID_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu118}"
TORCH_PACKAGE_SPEC="${SHARDGRID_TORCH_PACKAGE_SPEC:-torch==2.7.1+cu118}"
SHARDGRID_NCCL_MTU="${SHARDGRID_NCCL_MTU:-1500}"
SHARDGRID_NCCL_PEER_IP="${SHARDGRID_NCCL_PEER_IP:-}"
SHARDGRID_WSL_PERSIST_NCCL_MTU="${SHARDGRID_WSL_PERSIST_NCCL_MTU:-0}"

usage() {
    cat <<'EOF'
Usage: bootstrap-wsl.sh [--check] [--install-deps] [--json] [--findings-dir DIR]

  --check            Detect and record only; never install anything.
  --install-deps     Install missing ShardGrid Python dependencies into the
                     selected existing Conda environment.
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
FILE_TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
HOST="$(hostname 2>/dev/null || echo unknown)"
WSL_DISTRO="${WSL_DISTRO_NAME:-}"
if [[ -z "$WSL_DISTRO" ]] && [[ -r /etc/os-release ]]; then
    # shellcheck disable=SC1091
    . /etc/os-release
    WSL_DISTRO="${PRETTY_NAME:-${NAME:-unknown}}"
fi
KERNEL="$(uname -srmo 2>/dev/null || uname -srm 2>/dev/null || echo unknown)"

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

parse_route_dev() {
    local route_info="${1:-}"
    awk '
{
    for (i = 1; i <= NF; i++) {
        if ($i == "dev" && i + 1 <= NF) {
            print $(i + 1)
            exit
        }
    }
}' <<<"$route_info"
}

parse_link_mtu() {
    local link_info="${1:-}"
    awk '
{
    for (i = 1; i <= NF; i++) {
        if ($i == "mtu" && i + 1 <= NF) {
            print $(i + 1)
            exit
        }
    }
}' <<<"$link_info"
}

build_wsl_boot_mtu_command() {
    local peer_ip="${1:-}"
    local mtu="${2:-1500}"
    printf '%s' "/bin/sh -lc 'peer_ip=$peer_ip; mtu=$mtu; dev=\$(ip route get \"\$peer_ip\" | awk '\''{for(i=1;i<=NF;i++) if(\$i==\"dev\" && i+1<=NF){print \$(i+1); exit}}'\''); [ -n \"\$dev\" ] && ip link set dev \"\$dev\" mtu \"\$mtu\"'"
}

persist_wsl_mtu_command() {
    local peer_ip="${1:-}"
    local mtu="${2:-1500}"
    local wsl_conf="/etc/wsl.conf"
    local backup="/etc/wsl.conf.bak.t072_mtu"
    local boot_command=""
    boot_command="$(build_wsl_boot_mtu_command "$peer_ip" "$mtu")"
    [[ -n "$boot_command" ]] || return 1

    if [[ -f "$wsl_conf" ]] && [[ ! -f "$backup" ]]; then
        cp "$wsl_conf" "$backup" || return 1
    fi

    if [[ -f "$wsl_conf" ]] && grep -Eq '^[[:space:]]*command[[:space:]]*=' "$wsl_conf"; then
        return 2
    fi

    if [[ ! -f "$wsl_conf" ]]; then
        cat >"$wsl_conf" <<EOF
[boot]
command=$boot_command
EOF
        return 0
    fi

    if grep -Eq '^[[:space:]]*\[boot\][[:space:]]*$' "$wsl_conf"; then
        printf '\ncommand=%s\n' "$boot_command" >>"$wsl_conf" || return 1
        return 0
    fi

    cat >>"$wsl_conf" <<EOF

[boot]
command=$boot_command
EOF
}

configure_nccl_path_mtu() {
    NCCL_ROUTE_OUTPUT=""
    NCCL_INTERFACE=""
    NCCL_INTERFACE_MTU_BEFORE=""
    NCCL_INTERFACE_MTU_AFTER=""
    NCCL_MTU_STATUS="not_checked"
    NCCL_DF_1472_STATUS="not_checked"
    NCCL_DF_1473_STATUS="not_checked"
    NCCL_WSL_BOOT_STATUS="not_requested"
    NCCL_WSL_BOOT_COMMAND=""

    if [[ -z "$SHARDGRID_NCCL_PEER_IP" ]]; then
        return 0
    fi

    if [[ -z "${SHARDGRID_NCCL_MTU:-}" ]]; then
        add_manual_action "set SHARDGRID_NCCL_MTU to a valid integer (default is 1500)"
        NCCL_MTU_STATUS="invalid_config"
        return 0
    fi

    NCCL_ROUTE_OUTPUT="$(ip route get "$SHARDGRID_NCCL_PEER_IP" 2>&1 || true)"
    record_command ip route get "$SHARDGRID_NCCL_PEER_IP"
    NCCL_INTERFACE="$(parse_route_dev "$NCCL_ROUTE_OUTPUT")"
    if [[ -z "$NCCL_INTERFACE" ]]; then
        add_manual_action \
            "resolve the WSL2 NCCL route to peer $SHARDGRID_NCCL_PEER_IP (ip route get must return a dev)"
        NCCL_MTU_STATUS="route_dev_missing"
        return 0
    fi

    local link_before=""
    link_before="$(ip link show dev "$NCCL_INTERFACE" 2>&1 || true)"
    record_command ip link show dev "$NCCL_INTERFACE"
    NCCL_INTERFACE_MTU_BEFORE="$(parse_link_mtu "$link_before")"
    if [[ -z "$NCCL_INTERFACE_MTU_BEFORE" ]]; then
        add_manual_action "read MTU for WSL2 interface $NCCL_INTERFACE"
        NCCL_MTU_STATUS="mtu_read_failed"
        return 0
    fi

    if [[ "$MODE" != "check" && "$NCCL_INTERFACE_MTU_BEFORE" != "$SHARDGRID_NCCL_MTU" ]]; then
        if [[ "$(id -u)" == "0" ]]; then
            record_command ip link set dev "$NCCL_INTERFACE" mtu "$SHARDGRID_NCCL_MTU"
            if ! ip link set dev "$NCCL_INTERFACE" mtu "$SHARDGRID_NCCL_MTU" >/dev/null 2>&1; then
                add_manual_action \
                    "set interface $NCCL_INTERFACE MTU to $SHARDGRID_NCCL_MTU for NCCL peer $SHARDGRID_NCCL_PEER_IP"
                NCCL_MTU_STATUS="set_failed"
            fi
        else
            add_manual_action \
                "set interface $NCCL_INTERFACE MTU to $SHARDGRID_NCCL_MTU for NCCL peer $SHARDGRID_NCCL_PEER_IP (requires root/CAP_NET_ADMIN)"
            NCCL_MTU_STATUS="needs_root"
        fi
    fi

    local link_after=""
    link_after="$(ip link show dev "$NCCL_INTERFACE" 2>&1 || true)"
    record_command ip link show dev "$NCCL_INTERFACE"
    NCCL_INTERFACE_MTU_AFTER="$(parse_link_mtu "$link_after")"
    if [[ "$NCCL_INTERFACE_MTU_AFTER" == "$SHARDGRID_NCCL_MTU" ]]; then
        NCCL_MTU_STATUS="PASS"
    elif [[ "$NCCL_INTERFACE_MTU_AFTER" =~ ^[0-9]+$ ]]; then
        NCCL_MTU_STATUS="NCCL_PATH_MTU_UNSAFE"
        add_manual_action \
            "NCCL_PATH_MTU_UNSAFE: peer=$SHARDGRID_NCCL_PEER_IP interface=$NCCL_INTERFACE interface_mtu=$NCCL_INTERFACE_MTU_AFTER expected_mtu=$SHARDGRID_NCCL_MTU"
    fi

    local df1472=""
    df1472="$(ping -M do -c 1 -s 1472 "$SHARDGRID_NCCL_PEER_IP" 2>&1 || true)"
    record_command ping -M do -c 1 -s 1472 "$SHARDGRID_NCCL_PEER_IP"
    if grep -qiE 'command not found|operation not permitted' <<<"$df1472"; then
        NCCL_DF_1472_STATUS="MTU_PROBE_UNAVAILABLE"
    elif grep -qi '1 received' <<<"$df1472"; then
        NCCL_DF_1472_STATUS="PASS"
    else
        NCCL_DF_1472_STATUS="FAIL"
    fi

    local df1473=""
    df1473="$(ping -M do -c 1 -s 1473 "$SHARDGRID_NCCL_PEER_IP" 2>&1 || true)"
    record_command ping -M do -c 1 -s 1473 "$SHARDGRID_NCCL_PEER_IP"
    if grep -qiE 'command not found|operation not permitted' <<<"$df1473"; then
        NCCL_DF_1473_STATUS="MTU_PROBE_UNAVAILABLE"
    elif grep -qi '1 received' <<<"$df1473"; then
        NCCL_DF_1473_STATUS="UNEXPECTED_PASS"
    else
        NCCL_DF_1473_STATUS="EXPECTED_BLOCK"
    fi

    NCCL_WSL_BOOT_COMMAND="$(build_wsl_boot_mtu_command "$SHARDGRID_NCCL_PEER_IP" "$SHARDGRID_NCCL_MTU")"
    if [[ "$SHARDGRID_WSL_PERSIST_NCCL_MTU" == "1" ]]; then
        if [[ "$(id -u)" != "0" ]]; then
            NCCL_WSL_BOOT_STATUS="needs_root"
            add_manual_action "persist WSL boot MTU fix in /etc/wsl.conf (requires root)"
        else
            local persist_rc=0
            if persist_wsl_mtu_command "$SHARDGRID_NCCL_PEER_IP" "$SHARDGRID_NCCL_MTU"; then
                NCCL_WSL_BOOT_STATUS="configured"
            else
                persist_rc=$?
                case "$persist_rc" in
                    2)
                        NCCL_WSL_BOOT_STATUS="manual_merge_required"
                        add_manual_action \
                            "merge the WSL [boot] command in /etc/wsl.conf without overwriting an existing command"
                        ;;
                    *)
                        NCCL_WSL_BOOT_STATUS="persist_failed"
                        add_manual_action "persist WSL boot MTU fix in /etc/wsl.conf"
                        ;;
                esac
            fi
        fi
    fi
}

find_conda() {
    local candidate=""
    if candidate="$(command -v conda 2>/dev/null)"; then
        [[ -n "$candidate" ]] && printf '%s\n' "$candidate" && return 0
    fi

    local -a candidates=(
        "$HOME/miniconda3/bin/conda"
        "$HOME/anaconda3/bin/conda"
        "$HOME/mambaforge/bin/conda"
        "$HOME/miniforge3/bin/conda"
        "/opt/conda/bin/conda"
    )
    for candidate in "${candidates[@]}"; do
        if [[ -x "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}

load_conda_envs() {
    CONDA_ENV_LIST=""
    CONDA_ENV_NAMES=()
    CONDA_ENV_PATHS=()
    [[ -n "$CONDA_EXE" ]] || return 0

    CONDA_ENV_LIST="$("$CONDA_EXE" env list 2>&1 || true)"
    record_command "$CONDA_EXE" env list

    while IFS= read -r line; do
        [[ -z "$line" || "$line" =~ ^# ]] && continue
        CONDA_ENV_NAMES+=("$(awk '{print $1}' <<<"$line")")
        CONDA_ENV_PATHS+=("$(awk '{print $NF}' <<<"$line")")
    done <<<"$CONDA_ENV_LIST"
}

select_compatible_env() {
    SELECTED_ENV=""
    SELECTED_PREFIX=""
    PYTHON=""
    PYTHON_VERSION=""
    PYTHON_CONDA_MANAGED="false"

    if [[ -n "$CONDA_ACTIVE_ENV" && -n "$CONDA_PREFIX" && -x "$CONDA_PREFIX/bin/python" ]]; then
        if conda_env_ready "$CONDA_PREFIX/bin/python"; then
            SELECTED_ENV="$CONDA_ACTIVE_ENV"
            SELECTED_PREFIX="$CONDA_PREFIX"
            PYTHON="$CONDA_PREFIX/bin/python"
            PYTHON_CONDA_MANAGED="true"
        fi
    fi

    if [[ -z "$PYTHON" ]]; then
        for i in "${!CONDA_ENV_PATHS[@]}"; do
            env_python="${CONDA_ENV_PATHS[$i]}/bin/python"
            if conda_env_ready "$env_python"; then
                SELECTED_ENV="${CONDA_ENV_NAMES[$i]}"
                SELECTED_PREFIX="${CONDA_ENV_PATHS[$i]}"
                PYTHON="$env_python"
                PYTHON_CONDA_MANAGED="true"
                break
            fi
        done
    fi
}

bootstrap_runtime_env() {
    local target_env="$RUNTIME_ENV_NAME"
    local target_prefix=""
    local env_exists="false"

    for i in "${!CONDA_ENV_NAMES[@]}"; do
        if [[ "${CONDA_ENV_NAMES[$i]}" == "$target_env" ]]; then
            env_exists="true"
            target_prefix="${CONDA_ENV_PATHS[$i]}"
            break
        fi
    done

    if [[ "$env_exists" != "true" ]]; then
        record_command "$CONDA_EXE" create -n "$target_env" -y --override-channels -c conda-forge "$RUNTIME_PYTHON_SPEC" pip
        "$CONDA_EXE" create -n "$target_env" -y --override-channels -c conda-forge "$RUNTIME_PYTHON_SPEC" pip >/dev/null 2>&1 || return 1
        load_conda_envs
        for i in "${!CONDA_ENV_NAMES[@]}"; do
            if [[ "${CONDA_ENV_NAMES[$i]}" == "$target_env" ]]; then
                target_prefix="${CONDA_ENV_PATHS[$i]}"
                break
            fi
        done
    fi

    [[ -n "$target_prefix" && -x "$target_prefix/bin/python" ]] || return 1

    if ! conda_env_ready "$target_prefix/bin/python"; then
        record_command "$target_prefix/bin/python" -m pip install --upgrade --progress-bar off PyYAML
        "$target_prefix/bin/python" -m pip install --upgrade --progress-bar off PyYAML >/dev/null 2>&1 || return 1
        record_command "$target_prefix/bin/python" -m pip install --upgrade --progress-bar off --index-url "$TORCH_INDEX_URL" "$TORCH_PACKAGE_SPEC"
        "$target_prefix/bin/python" -m pip install --upgrade --progress-bar off --index-url "$TORCH_INDEX_URL" "$TORCH_PACKAGE_SPEC" >/dev/null 2>&1 || return 1
    fi

    return 0
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

conda_env_ready() {
    local python_bin="${1:-}"
    [[ -n "$python_bin" && -x "$python_bin" ]] || return 1
    "$python_bin" - <<'PY' >/dev/null 2>&1
import importlib.util
if importlib.util.find_spec("yaml") is None:
    raise SystemExit(1)
if importlib.util.find_spec("torch") is None:
    raise SystemExit(1)
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
}

shardgrid_deps_present() {
    local python_bin="${1:-}"
    [[ -n "$python_bin" && -x "$python_bin" ]] || return 1
    if [[ -d "$REPO_ROOT/src/shardgrid" ]]; then
        PYTHONPATH="$REPO_ROOT/src" "$python_bin" - <<'PY' >/dev/null 2>&1
import shardgrid, yaml
PY
    else
        "$python_bin" - <<'PY' >/dev/null 2>&1
import yaml
PY
    fi
}

# --- Conda detection ---------------------------------------------------------
CONDA_EXE="$(find_conda 2>/dev/null || true)"
CONDA_VERSION=""
CONDA_ENV_LIST=""
CONDA_ENV_NAMES=()
CONDA_ENV_PATHS=()
CONDA_ACTIVE_ENV="${CONDA_DEFAULT_ENV:-}"
CONDA_PREFIX="${CONDA_PREFIX:-}"

if [[ -n "$CONDA_EXE" ]]; then
    CONDA_VERSION="$("$CONDA_EXE" --version 2>&1 || true)"
    load_conda_envs
else
    add_manual_action \
        "install Conda inside the Ubuntu WSL2 runtime; ShardGrid does not use Windows-host Conda for training"
fi

if [[ -z "$CONDA_ACTIVE_ENV" && -n "$CONDA_PREFIX" ]]; then
    for i in "${!CONDA_ENV_PATHS[@]}"; do
        if [[ "${CONDA_ENV_PATHS[$i]}" == "$CONDA_PREFIX" ]]; then
            CONDA_ACTIVE_ENV="${CONDA_ENV_NAMES[$i]}"
            break
        fi
    done
fi

CONDA_ENV_NAMES_JOINED="$(IFS=$'\n'; printf '%s' "${CONDA_ENV_NAMES[*]}")"

# --- Python selection --------------------------------------------------------
SYSTEM_PYTHON="$(command -v python3 2>/dev/null || true)"
SYSTEM_PYTHON_VERSION=""
PARSER_PYTHON="$SYSTEM_PYTHON"
if [[ -z "$PARSER_PYTHON" ]]; then
    PARSER_PYTHON="$(command -v python 2>/dev/null || true)"
fi
if [[ -n "$SYSTEM_PYTHON" ]]; then
    SYSTEM_PYTHON_VERSION="$("$SYSTEM_PYTHON" --version 2>&1 || true)"
    record_command "$SYSTEM_PYTHON" --version
fi

SELECTED_ENV=""
SELECTED_PREFIX=""
PYTHON=""
PYTHON_VERSION=""
PYTHON_CONDA_MANAGED="false"

if [[ -n "$CONDA_EXE" ]]; then
    select_compatible_env
    if [[ -z "$SELECTED_ENV" && "$MODE" != "check" ]]; then
        if bootstrap_runtime_env; then
            load_conda_envs
            select_compatible_env
        else
            add_manual_action \
                "create or repair a WSL2 Conda training environment with PyYAML and PyTorch"
        fi
    fi
fi

if [[ -z "$PYTHON" && -n "$SYSTEM_PYTHON" ]]; then
    PYTHON="$SYSTEM_PYTHON"
fi

if [[ -n "$PYTHON" ]]; then
    PYTHON_VERSION="$("$PYTHON" --version 2>&1 || true)"
    record_command "$PYTHON" --version
fi

if [[ -n "$CONDA_EXE" && -z "$SELECTED_ENV" ]]; then
    add_manual_action \
        "create or select a WSL2 Conda training environment with PyYAML and PyTorch; do not use system Python for training"
fi

# --- Runtime tool detection --------------------------------------------------
BASH_VERSION_LINE="${BASH_VERSION:-unknown}"
LSB_RELEASE=""
UNAME_VERSION=""
GIT_VERSION=""
IPERF3_VERSION=""
tool_version lsb_release LSB_RELEASE lsb_release -ds
tool_version uname UNAME_VERSION uname -srmo
tool_version git GIT_VERSION git --version
tool_version iperf3 IPERF3_VERSION iperf3 --version

if [[ "$IPERF3_VERSION" == "not_installed" ]]; then
    add_manual_action \
        "install iperf3 inside the Ubuntu WSL2 runtime (operator action: sudo apt-get install -y iperf3)"
fi

configure_nccl_path_mtu

# --- NVIDIA / CUDA visibility ------------------------------------------------
NVIDIA_SMI_PATH="$(command -v nvidia-smi 2>/dev/null || true)"
NVIDIA_SMI_VERSION=""
NVIDIA_GPU_SUMMARY=""
if [[ -n "$NVIDIA_SMI_PATH" ]]; then
    NVIDIA_SMI_VERSION="$("$NVIDIA_SMI_PATH" 2>&1 | head -n 1 || true)"
    NVIDIA_GPU_SUMMARY="$("$NVIDIA_SMI_PATH" --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -n 1 || true)"
    record_command "$NVIDIA_SMI_PATH"
    record_command "$NVIDIA_SMI_PATH" --query-gpu=name,driver_version --format=csv,noheader
else
    add_manual_action \
        "verify CUDA-on-WSL2 visibility from the Ubuntu runtime; do not install a Linux NVIDIA display driver inside WSL"
fi

# --- ShardGrid dependency detection -----------------------------------------
DEPS_SG="not_checked"
DEPS_YAML="not_checked"
DEPS_OVERALL="not_checked"
if [[ -n "$PYTHON" && "$PYTHON_CONDA_MANAGED" == "true" ]]; then
    if shardgrid_deps_present "$PYTHON"; then
        DEPS_SG="present"
        DEPS_YAML="present"
        DEPS_OVERALL="present"
    else
        DEPS_SG="missing"
        DEPS_YAML="missing"
        DEPS_OVERALL="missing"
    fi
fi

# --- PyTorch / CUDA runtime detection ---------------------------------------
TORCH_VERSION="not_checked"
TORCH_CUDA_VERSION="not_checked"
TORCH_CUDA_AVAILABLE="not_checked"
if [[ -n "$PYTHON" && "$PYTHON_CONDA_MANAGED" == "true" ]]; then
    TORCH_INFO="$("$PYTHON" - <<'PY' 2>/dev/null || true
import importlib.util
import json
if importlib.util.find_spec("torch") is None:
    print(json.dumps({"status": "missing"}))
else:
    import torch
    print(json.dumps({
        "status": "present",
        "version": getattr(torch, "__version__", None),
        "cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
        "cuda_available": bool(torch.cuda.is_available()),
    }))
PY
)"
    if [[ -n "$PARSER_PYTHON" ]]; then
        TORCH_VERSION="$("$PARSER_PYTHON" - "$TORCH_INFO" <<'PY'
import json, sys
payload = json.loads(sys.argv[1] or "{}")
print(payload.get("version") or payload.get("status") or "not_checked")
PY
)"
        TORCH_CUDA_VERSION="$("$PARSER_PYTHON" - "$TORCH_INFO" <<'PY'
import json, sys
payload = json.loads(sys.argv[1] or "{}")
value = payload.get("cuda_version")
print(value if value is not None else "not_checked")
PY
)"
        TORCH_CUDA_AVAILABLE="$("$PARSER_PYTHON" - "$TORCH_INFO" <<'PY'
import json, sys
payload = json.loads(sys.argv[1] or "{}")
if "cuda_available" in payload:
    print("true" if payload["cuda_available"] else "false")
else:
    print("not_checked")
PY
)"
    fi
fi

if [[ -n "$SELECTED_ENV" && ( "$TORCH_VERSION" == "missing" || "$TORCH_VERSION" == "not_checked" || "$TORCH_CUDA_AVAILABLE" != "true" ) ]]; then
    add_manual_action \
        "install PyTorch in the selected WSL2 Conda environment and confirm torch.cuda.is_available() from that environment"
fi

# --- Reuse / create decision -------------------------------------------------
DECISION_REUSE="$SELECTED_ENV"
DECISION_CREATE_NEEDED="false"
if [[ -n "$CONDA_EXE" && -z "$SELECTED_ENV" ]]; then
    DECISION_CREATE_NEEDED="true"
fi

# --- Optional safe install of project dependencies ---------------------------
if [[ "$INSTALL_DEPS" == "1" && "$MODE" != "check" && -n "$PYTHON" && "$PYTHON_CONDA_MANAGED" == "true" ]]; then
    if [[ "$DEPS_OVERALL" != "present" && -d "$REPO_ROOT" ]]; then
        record_command "$PYTHON" -m pip install -e "$REPO_ROOT[dev,test]"
        if "$PYTHON" -m pip install -e "$REPO_ROOT[dev,test]" >/dev/null 2>&1; then
            DEPS_SG="present"
            DEPS_YAML="present"
            DEPS_OVERALL="present"
        else
            add_manual_action \
                "install ShardGrid Python dependencies into the selected WSL2 Conda environment (operator action: python -m pip install -e '$REPO_ROOT[dev,test]')"
        fi
    fi
fi

if [[ "$DEPS_OVERALL" == "missing" && "$HEALTH" == "healthy" ]]; then
    HEALTH="degraded"
fi

JSON_PYTHON="$SYSTEM_PYTHON"
[[ -z "$JSON_PYTHON" ]] && JSON_PYTHON="$(command -v python 2>/dev/null || true)"

export BOOTSTRAP_SCRIPT="bootstrap-wsl.sh"
export BOOTSTRAP_MODE="$MODE"
export BOOTSTRAP_TIMESTAMP="$TIMESTAMP"
export BOOTSTRAP_FILE_TIMESTAMP="$FILE_TIMESTAMP"
export BOOTSTRAP_HOST="$HOST"
export BOOTSTRAP_WSL_DISTRO="$WSL_DISTRO"
export BOOTSTRAP_KERNEL="$KERNEL"
export BOOTSTRAP_REPO_ROOT="$REPO_ROOT"
export BOOTSTRAP_CONDA_EXE="$CONDA_EXE"
export BOOTSTRAP_CONDA_VERSION="$CONDA_VERSION"
export BOOTSTRAP_CONDA_ENVS="$CONDA_ENV_NAMES_JOINED"
export BOOTSTRAP_CONDA_ACTIVE_ENV="$CONDA_ACTIVE_ENV"
export BOOTSTRAP_CONDA_PREFIX="$CONDA_PREFIX"
export BOOTSTRAP_SELECTED_ENV="$SELECTED_ENV"
export BOOTSTRAP_SELECTED_PREFIX="$SELECTED_PREFIX"
export BOOTSTRAP_PYTHON="$PYTHON"
export BOOTSTRAP_PYTHON_VERSION="$PYTHON_VERSION"
export BOOTSTRAP_PYTHON_CONDA_MANAGED="$PYTHON_CONDA_MANAGED"
export BOOTSTRAP_SYSTEM_PYTHON="$SYSTEM_PYTHON"
export BOOTSTRAP_SYSTEM_PYTHON_VERSION="$SYSTEM_PYTHON_VERSION"
export BOOTSTRAP_BASH_VERSION="$BASH_VERSION_LINE"
export BOOTSTRAP_LSB_RELEASE="$LSB_RELEASE"
export BOOTSTRAP_UNAME_VERSION="$UNAME_VERSION"
export BOOTSTRAP_GIT="$GIT_VERSION"
export BOOTSTRAP_IPERF3="$IPERF3_VERSION"
export BOOTSTRAP_NVIDIA_SMI_PATH="$NVIDIA_SMI_PATH"
export BOOTSTRAP_NVIDIA_SMI_VERSION="$NVIDIA_SMI_VERSION"
export BOOTSTRAP_NVIDIA_GPU_SUMMARY="$NVIDIA_GPU_SUMMARY"
export BOOTSTRAP_DEPS_SG="$DEPS_SG"
export BOOTSTRAP_DEPS_YAML="$DEPS_YAML"
export BOOTSTRAP_DEPS_OVERALL="$DEPS_OVERALL"
export BOOTSTRAP_TORCH_VERSION="$TORCH_VERSION"
export BOOTSTRAP_TORCH_CUDA_VERSION="$TORCH_CUDA_VERSION"
export BOOTSTRAP_TORCH_CUDA_AVAILABLE="$TORCH_CUDA_AVAILABLE"
export BOOTSTRAP_DECISION_REUSE="$DECISION_REUSE"
export BOOTSTRAP_DECISION_CREATE_NEEDED="$DECISION_CREATE_NEEDED"
export BOOTSTRAP_HEALTH="$HEALTH"
export BOOTSTRAP_NCCL_MTU="$SHARDGRID_NCCL_MTU"
export BOOTSTRAP_NCCL_PEER_IP="$SHARDGRID_NCCL_PEER_IP"
export BOOTSTRAP_NCCL_ROUTE_OUTPUT="$NCCL_ROUTE_OUTPUT"
export BOOTSTRAP_NCCL_INTERFACE="$NCCL_INTERFACE"
export BOOTSTRAP_NCCL_INTERFACE_MTU_BEFORE="$NCCL_INTERFACE_MTU_BEFORE"
export BOOTSTRAP_NCCL_INTERFACE_MTU_AFTER="$NCCL_INTERFACE_MTU_AFTER"
export BOOTSTRAP_NCCL_MTU_STATUS="$NCCL_MTU_STATUS"
export BOOTSTRAP_NCCL_DF_1472_STATUS="$NCCL_DF_1472_STATUS"
export BOOTSTRAP_NCCL_DF_1473_STATUS="$NCCL_DF_1473_STATUS"
export BOOTSTRAP_NCCL_WSL_BOOT_STATUS="$NCCL_WSL_BOOT_STATUS"
export BOOTSTRAP_NCCL_WSL_BOOT_COMMAND="$NCCL_WSL_BOOT_COMMAND"
MANUAL_ACTIONS_JOINED="$(IFS=$'\x1f'; printf '%s' "${MANUAL_ACTIONS[*]}")"
COMMANDS_RUN_JOINED="$(IFS=$'\x1f'; printf '%s' "${COMMANDS_RUN[*]}")"
export BOOTSTRAP_MANUAL_ACTIONS="$MANUAL_ACTIONS_JOINED"
export BOOTSTRAP_COMMANDS_RUN="$COMMANDS_RUN_JOINED"

if [[ -n "$JSON_PYTHON" ]]; then
    TIMESTAMPED_FILE="$FINDINGS_DIR/wsl-$FILE_TIMESTAMP.json"
    LATEST_FILE="$FINDINGS_DIR/wsl-latest.json"
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
    "wsl_distro": optional(os.environ.get("BOOTSTRAP_WSL_DISTRO")),
    "kernel": optional(os.environ.get("BOOTSTRAP_KERNEL")),
    "repo_root": optional(os.environ.get("BOOTSTRAP_REPO_ROOT")),
    "conda": {
        "executable": optional(os.environ.get("BOOTSTRAP_CONDA_EXE")),
        "version": optional(os.environ.get("BOOTSTRAP_CONDA_VERSION")),
        "environments": get_list("BOOTSTRAP_CONDA_ENVS", "\n"),
        "active_environment": optional(os.environ.get("BOOTSTRAP_CONDA_ACTIVE_ENV")),
        "active_prefix": optional(os.environ.get("BOOTSTRAP_CONDA_PREFIX")),
        "selected_environment": optional(os.environ.get("BOOTSTRAP_SELECTED_ENV")),
        "selected_prefix": optional(os.environ.get("BOOTSTRAP_SELECTED_PREFIX")),
    },
    "python": {
        "executable": optional(os.environ.get("BOOTSTRAP_PYTHON")),
        "version": optional(os.environ.get("BOOTSTRAP_PYTHON_VERSION")),
        "conda_managed": os.environ.get("BOOTSTRAP_PYTHON_CONDA_MANAGED") == "true",
        "system_executable": optional(os.environ.get("BOOTSTRAP_SYSTEM_PYTHON")),
        "system_version": optional(os.environ.get("BOOTSTRAP_SYSTEM_PYTHON_VERSION")),
    },
    "runtime_tools": {
        "bash_version": optional(os.environ.get("BOOTSTRAP_BASH_VERSION")),
        "lsb_release": optional(os.environ.get("BOOTSTRAP_LSB_RELEASE")),
        "uname": optional(os.environ.get("BOOTSTRAP_UNAME_VERSION")),
        "git": optional(os.environ.get("BOOTSTRAP_GIT")),
        "iperf3": optional(os.environ.get("BOOTSTRAP_IPERF3")),
    },
    "project_dependencies": {
        "shardgrid_importable": os.environ.get("BOOTSTRAP_DEPS_SG") == "present",
        "pyyaml_available": os.environ.get("BOOTSTRAP_DEPS_YAML") == "present",
        "status": os.environ.get("BOOTSTRAP_DEPS_OVERALL", "not_checked"),
    },
    "torch": {
        "version": optional(os.environ.get("BOOTSTRAP_TORCH_VERSION")),
        "cuda_version": optional(os.environ.get("BOOTSTRAP_TORCH_CUDA_VERSION")),
        "cuda_available": optional(os.environ.get("BOOTSTRAP_TORCH_CUDA_AVAILABLE")),
    },
    "nvidia_smi": {
        "path": optional(os.environ.get("BOOTSTRAP_NVIDIA_SMI_PATH")),
        "version": optional(os.environ.get("BOOTSTRAP_NVIDIA_SMI_VERSION")),
        "gpu_summary": optional(os.environ.get("BOOTSTRAP_NVIDIA_GPU_SUMMARY")),
    },
    "decision": {
        "reuse_environment": optional(os.environ.get("BOOTSTRAP_DECISION_REUSE")),
        "create_needed": os.environ.get("BOOTSTRAP_DECISION_CREATE_NEEDED") == "true",
    },
    "nccl_path_mtu": {
        "peer_ip": optional(os.environ.get("BOOTSTRAP_NCCL_PEER_IP")),
        "expected_mtu": optional(os.environ.get("BOOTSTRAP_NCCL_MTU")),
        "route_output": optional(os.environ.get("BOOTSTRAP_NCCL_ROUTE_OUTPUT")),
        "interface": optional(os.environ.get("BOOTSTRAP_NCCL_INTERFACE")),
        "interface_mtu_before": optional(os.environ.get("BOOTSTRAP_NCCL_INTERFACE_MTU_BEFORE")),
        "interface_mtu_after": optional(os.environ.get("BOOTSTRAP_NCCL_INTERFACE_MTU_AFTER")),
        "status": optional(os.environ.get("BOOTSTRAP_NCCL_MTU_STATUS")),
        "df_1472": optional(os.environ.get("BOOTSTRAP_NCCL_DF_1472_STATUS")),
        "df_1473": optional(os.environ.get("BOOTSTRAP_NCCL_DF_1473_STATUS")),
        "wsl_boot_status": optional(os.environ.get("BOOTSTRAP_NCCL_WSL_BOOT_STATUS")),
        "wsl_boot_command": optional(os.environ.get("BOOTSTRAP_NCCL_WSL_BOOT_COMMAND")),
    },
    "health": os.environ.get("BOOTSTRAP_HEALTH", "unknown"),
    "manual_actions": get_list("BOOTSTRAP_MANUAL_ACTIONS"),
    "commands_run": get_list("BOOTSTRAP_COMMANDS_RUN"),
}

for path in sys.argv[1:]:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
PY
fi

if [[ "$JSON_OUT" == "1" && -n "$JSON_PYTHON" ]]; then
    "$JSON_PYTHON" -c "import json, os; print(json.dumps(json.load(open(os.path.expanduser('$FINDINGS_DIR/wsl-latest.json'))), indent=2, sort_keys=True))"
fi

{
    echo "ShardGrid WSL bootstrap ($MODE mode)"
    echo "host: $HOST | distro: ${WSL_DISTRO:-unknown} | kernel: $KERNEL | timestamp: $TIMESTAMP"
    echo "conda: ${CONDA_EXE:-not_found} ${CONDA_VERSION:+($CONDA_VERSION)}"
    echo "  environments: ${CONDA_ENV_NAMES[*]:-none}"
    echo "  active: ${CONDA_ACTIVE_ENV:-none} prefix: ${CONDA_PREFIX:-none}"
    echo "  selected: ${SELECTED_ENV:-none} prefix: ${SELECTED_PREFIX:-none}"
    echo "python: ${PYTHON:-not_found} ${PYTHON_VERSION:+($PYTHON_VERSION)} conda_managed=$PYTHON_CONDA_MANAGED"
    echo "system python: ${SYSTEM_PYTHON:-not_found} ${SYSTEM_PYTHON_VERSION:+($SYSTEM_PYTHON_VERSION)}"
    echo "torch: version=$TORCH_VERSION cuda_version=$TORCH_CUDA_VERSION cuda_available=$TORCH_CUDA_AVAILABLE"
    echo "nvidia-smi: ${NVIDIA_SMI_PATH:-not_found} ${NVIDIA_GPU_SUMMARY:+($NVIDIA_GPU_SUMMARY)}"
    echo "runtime tools: bash=$BASH_VERSION_LINE lsb_release=$LSB_RELEASE git=$GIT_VERSION iperf3=$IPERF3_VERSION"
    if [[ -n "$SHARDGRID_NCCL_PEER_IP" ]]; then
        echo "ShardGrid NCCL network check"
        echo "  peer: $SHARDGRID_NCCL_PEER_IP"
        echo "  interface: ${NCCL_INTERFACE:-unknown}"
        echo "  interface_mtu: ${NCCL_INTERFACE_MTU_AFTER:-unknown}"
        echo "  expected_mtu: $SHARDGRID_NCCL_MTU"
        echo "  status: $NCCL_MTU_STATUS"
        echo "  df_1472: $NCCL_DF_1472_STATUS"
        echo "  df_1473: $NCCL_DF_1473_STATUS"
    fi
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
