# Bootstrap and Version Recording

## Operating Rule

ShardGrid bootstrap and compatibility work starts with detection, not forced installation.

1. Detect the current environment first
2. Reuse an already compatible component when it is present
3. Install or adjust only when a component is missing or proven incompatible

This applies to the Ubuntu control node, Windows workers, and WSL2 training runtimes.

## Version Recording Model

Use `examples/versions.yaml` as the stable recording format for:

- Actual detected versions
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
- PyTorch may remain `not_installed` on Machine A
- Kubernetes may remain `not_installed` and `not_checked` until the SSH MVP is complete

## Current Phase

This phase defines only the record format and the bootstrap policy.

It does not:

- install PyTorch
- install CUDA
- install Kubernetes
- install Volcano
- install HAMi
- decide final compatibility for later platform integrations

## Practical Use Later

Later bootstrap and doctor flows should:

1. Detect a component and write the observed version
2. Mark whether it is `installed`, `not_installed`, or `not_checked`
3. Evaluate compatibility separately
4. Only then recommend reuse, upgrade, installation, or a manual action
