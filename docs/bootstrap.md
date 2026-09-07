# Bootstrap and Version Recording

## Operating Rule

ShardGrid bootstrap and compatibility work starts with detection, not forced installation.

1. Detect Conda before any Python environment operation
2. Record Conda executable, existing environments, active environment, active prefix, and Python executable/version
3. Reuse an already compatible Conda environment or component when it is present
4. Create a ShardGrid-specific Conda environment only when no existing environment satisfies project dependencies
5. Install Conda or adjust system components only when they are missing or proven incompatible and the action is safe or operator-approved

This applies to the Ubuntu control node, Windows workers, and WSL2 training runtimes.

## Version Recording Model

Use `examples/versions.yaml` as the stable recording format for:

- Actual detected versions
- Conda executable, active environment, environment prefix, and environment list
- Installation status
- Detection status
- Compatibility status
- Notes about why a component is accepted, deferred, or needs adjustment

The file is meant to record reality, not to impose one fixed version on every machine.

## Status Meanings

- `installed`: the component is present and a version can be recorded now or later
- `not_installed`: the component is not present
- `not_checked`: the component has not been checked yet
- `null`: the field is intentionally unknown or not recorded yet

## Separation of Concerns

Keep these separate:

- Observed version data
- Compatibility constraints
- Operator decisions

Example:

- Python may be installed with a real detected version on Machine A
- Conda may already provide the selected Python environment and should be reused
- PyTorch may remain `not_installed` on Machine A
- Kubernetes may remain `not_installed` and `not_checked` until the SSH MVP is complete

## Current Phase

This phase defines only the record format and the bootstrap policy.

It does not create, delete, overwrite, or activate any user Conda environment.

It does not:

- install Conda
- install PyTorch
- install CUDA
- install Kubernetes
- install Volcano
- install HAMi
- decide final compatibility for later platform integrations

## Practical Use Later

Later bootstrap and doctor flows should:

1. Detect Conda and current environment identity before Python-related checks
2. Detect a component and write the observed version
3. Mark whether it is `installed`, `not_installed`, or `not_checked`
4. Evaluate compatibility separately
5. Only then recommend reuse, Conda environment creation, installation, or a manual action

## Ubuntu Control-Node Bootstrap (`scripts/bootstrap-linux.sh`)

T028 implements the Ubuntu control-node bootstrap that follows the operating
rule above. It is idempotent and non-destructive by construction.

### Behavior

- Detect-first: records Conda executable, existing environments, active
  environment and prefix, Python executable/version, OpenSSH, Git, iperf3, disk
  free space, and ShardGrid project dependencies before any change decision.
- Conda-first / reuse-first: reuses the active compatible Conda environment
  (`reuse_environment`) and only reports `create_needed=true` when no existing
  environment satisfies the project dependencies.
- Safe only: never runs `sudo`, `apt-get`, password, reboot, BIOS, or firewall
  operations, never runs `conda create`/`conda remove`, and never modifies
  system Python.
- Manual actions: any missing Conda, OpenSSH, Git, or iperf3, or a needed
  Conda environment creation, stops automation with an explicit manual action
  and exits with code `2` (`blocked_manual_action`).
- Optional install: `--install-deps` installs only the project dependencies
  declared in `pyproject.toml` into the active Conda environment when they are
  missing. This is the only automatic mutation and it is skipped in `--check`
  mode or when system Python would be modified.
- Version recording: every run writes a timestamped JSON findings file and
  updates `latest.json` under `$SHARDGRID_BOOTSTRAP_DIR`
  (default `$HOME/.shardgrid/bootstrap`), recording actual detected versions
  and the commands that produced them.

### Usage

```bash
scripts/bootstrap-linux.sh                 # detect, record, report
scripts/bootstrap-linux.sh --check         # detect and record only
scripts/bootstrap-linux.sh --install-deps  # also install missing project deps (Conda env only)
scripts/bootstrap-linux.sh --json          # emit findings as JSON on stdout
scripts/bootstrap-linux.sh --findings-dir DIR
```

### Exit codes

- `0` healthy
- `1` unexpected detection failure
- `2` blocked by one or more manual actions

### Verification (Machine A, 2026-08-17)

- First run and second run produced identical health and decisions
  (`reuse_environment=base`, `create_needed=false`), wrote one timestamped
  findings file each, and did not create or modify any Conda environment.
- iperf3 is genuinely not installed on Machine A, so the real runs stop with a
  manual action for `iperf3` (exit `2`) instead of faking success.
- Simulated environments (restricted `PATH`, without deleting any software):
  - Conda missing -> manual action for Conda, Git, and iperf3 (exit `2`).
  - Git missing with Conda present -> manual action for Git and iperf3.
  - Ghost active environment prefix -> `create_needed=true` plus the
    `conda create` manual action, instead of creating an environment.
