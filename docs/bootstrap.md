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
