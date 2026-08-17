# Windows GPU Worker Bootstrap

`scripts/bootstrap-windows.ps1` implements T029 for the Windows physical host.
It is a detect-first, rerunnable bootstrap check. It records reality and stops
before any step that needs administrator approval, password entry, reboot, BIOS
changes, risky firewall edits, or elevated Conda installation.

## Usage

Run from an ordinary PowerShell session on each Windows GPU Worker:

```powershell
.\scripts\bootstrap-windows.ps1 -Check
.\scripts\bootstrap-windows.ps1 -Check -Json
.\scripts\bootstrap-windows.ps1 -Check -FindingsDir C:\Users\shardgrid\.shardgrid\bootstrap
```

Exit codes:

- `0`: healthy
- `1`: degraded or unexpected detection failure
- `2`: blocked by manual action

Findings are written to `windows-latest.json` and a timestamped JSON file under
the findings directory.

## What It Checks

- PowerShell version and whether the session is elevated
- OpenSSH Client and Server capability state
- `sshd` service presence and status
- WSL and VirtualMachinePlatform Windows feature state
- Ubuntu WSL distro presence and whether it is WSL2
- `nvidia-smi` visibility and NVIDIA Windows driver compatibility floor
- Windows-host Conda executable, version, active environment, and environment list
- Ubuntu WSL2 training-runtime Conda executable, version, environment list, and
  Python version

## Runtime Boundary

Windows-host Conda is recorded as `host_only_not_training_runtime`.

The actual training runtime is the Ubuntu WSL2 Linux environment recorded under
`wsl_training_conda`. A healthy Windows Python or Conda installation is not a
training PASS. CUDA and PyTorch readiness for training must be checked inside
WSL2 by T030 and later worker probes.

## Manual Actions

The script never performs these actions automatically:

- Install or enable Windows capabilities
- Start or enable `sshd`
- Install WSL2 or the Ubuntu distro
- Convert a distro to WSL2
- Install or upgrade NVIDIA drivers
- Install Conda or create/replace Conda environments
- Change firewall rules
- Ask for or handle passwords
- Reboot or change BIOS/UEFI settings

When one is needed, the script exits `2` and records the exact manual action.

## Verification Status

Local agent validation on the Ubuntu control host is limited to static checks
because this host has no PowerShell runtime.

Real check-mode execution was performed on Windows host `10.87.5.15`
(`LAPTOP-5G3QUOGM`) on 2026-08-17 by copying the script with `scp` and running:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File bootstrap-windows.ps1 -Check -Json
```

Result: `blocked_manual_action`, exit code `2`.

Observed:

- PowerShell `5.1.26100.9168`
- OpenSSH server observed through the active SSH session
- NVIDIA driver major `527`, compatible with the script's CUDA-on-WSL2 floor
- Windows-host Conda present and recorded as host-only
- Ubuntu WSL distro missing
- WSL2 training Conda not checked because the Ubuntu distro is missing

Manual action:

- Install the Ubuntu WSL distro manually, then rerun this check

The required RTX 4060 host `10.87.5.155` was not reachable from Machine A during
this T029 run (`no route to host`), so verification for that specific Worker is
`PENDING`.
