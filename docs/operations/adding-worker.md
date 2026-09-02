# Adding Optional Machine E

T106 adds the third GPU Worker by configuration only. No inventory, launcher,
or planner special-case code is needed.

## Example entry

Use the live CQUPT runtime identity already proven by probe:

```yaml
- id: gpu4060-cqupt
  machine_id: machine-e
  physical_os: windows
  runtime_os: wsl2_linux
  runtime: wsl2
  host: 10.87.5.214
  ssh_user: shardgrid
  ssh_port: 22
  runtime_distro: Ubuntu-22.04
  conda_environment: shardgrid
  conda_prefix: /home/shardgrid/miniconda3/envs/shardgrid
  local_world_size: 1
  enabled: true
  labels:
    gpu: rtx4060
    host_identity: CQUPT
    optional: "true"
```

`host_identity` is descriptive metadata only. The authoritative observed
hostname still comes from `shardgrid probe`.

## Verify

Enabled:

```bash
python -m shardgrid.cli.app --config examples/workers.yaml probe --worker gpu4060-cqupt --json
python -m shardgrid.cli.app --config examples/workers.yaml workers --refresh --json
```

Expected:

- `gpu4060-cqupt` appears in the configured inventory.
- probe reports `windows_identity= cqupt/CQUPT` and `wsl_distro=Ubuntu-22.04`.
- inventory reports the Worker as `HEALTHY` and `eligible=true`.

Disabled:

- set `enabled: false` on `gpu4060-cqupt`
- rerun `workers --refresh --json`

Expected:

- the Worker still appears as configured
- `eligible=false`
- the original `gpu4060` and `gpu1060` remain usable

Restore the shared example config to `enabled: true` after the disable check.
