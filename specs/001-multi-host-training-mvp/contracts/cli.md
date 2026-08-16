# CLI Contract: ShardGrid MVP + Platform Foundation

All commands support human-readable output by default and `--json` for machine-readable output unless noted.

## Global Options

- `--config PATH`: cluster config path. Default is `shardgrid.yaml` or `examples/workers.yaml` in examples.
- `--jobs-root PATH`: override configured job snapshot root.
- `--verbose`: include command details and diagnostics.
- `--json`: emit structured JSON.

## shardgrid doctor

Purpose: Validate Control node, configured Windows Workers, and WSL2 runtime readiness.

Inputs:

- `--target control|workers|all`
- `--fix`: perform safe idempotent fixes only.

Success output:

- Control readiness.
- Worker readiness.
- Runtime readiness.
- Version records.
- Manual actions required, if any.

Failure output:

- `FailureRecord` with `stage=BOOTSTRAP` or `stage=PROBE`.

## shardgrid workers

Purpose: Show configured and probed Worker inventory.

Inputs:

- `--refresh`: run live probe before display.

Success output:

- List of `WorkerResource`.

Failure output:

- Unreachable or unhealthy workers are shown with failure details; command exits non-zero only if `--require-healthy` is supplied.

## shardgrid probe

Purpose: Run Worker and GPU runtime probes.

Inputs:

- `--worker WORKER_ID` optional.

Success output:

- Probe records, GPUResource, backend availability.

Failure output:

- `FailureRecord` with `stage=PROBE`.

## shardgrid network-test

Purpose: Measure pairwise Worker connectivity, latency, and bandwidth.

Inputs:

- `--worker-a WORKER_ID`
- `--worker-b WORKER_ID`
- `--all`

Success output:

- `NetworkState` with `NetworkLink` records.

Failure output:

- `FailureRecord` with `stage=NETWORK`.

## shardgrid dist-test

Purpose: Verify cross-host distributed process group independent of full training.

Inputs:

- `--backend nccl|gloo|auto`
- `--workers WORKER_ID,WORKER_ID`
- `--save-report PATH` optional.

Success output:

- Broadcast, send/recv, and all_reduce results.
- Backend label.
- NCCL diagnostics if fallback was used.

Failure output:

- `FailureRecord` with `stage=RENDEZVOUS`.

## shardgrid train CONFIG

Purpose: Run a TrainingJob from Control node.

Inputs:

- `CONFIG`: TrainingJob config YAML.
- `--backend ssh|kubernetes|volcano|auto`
- `--dry-run`: create and validate ExecutionPlan without launching.
- `--override-worker WORKER_ID` repeatable.

Success output:

- `job_id`
- `state`
- `snapshot_path`
- `execution_plan_path`

Failure output:

- `FailureRecord` at the failed stage.

## shardgrid status JOB

Purpose: Show current or final JobStatus.

Inputs:

- `JOB`: job ID.
- `--watch`: follow until terminal state.

Success output:

- JobState, selected Workers, ranks, stages, backend, loss history, checkpoint reference.

## shardgrid logs JOB

Purpose: Show or locate job logs.

Inputs:

- `JOB`: job ID.
- `--worker WORKER_ID`
- `--rank RANK`
- `--tail N`

Success output:

- Log paths and latest log lines.

## shardgrid stop JOB

Purpose: Stop launched ranks and preserve partial job state.

Inputs:

- `JOB`: job ID.

Success output:

- Stop actions by Worker/rank.
- Final JobState: `stopped` or `failed`.
