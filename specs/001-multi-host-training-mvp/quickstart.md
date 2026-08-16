# Quickstart: ShardGrid MVP + Platform Foundation

This quickstart describes the intended validation flow after implementation tasks are generated and completed. Commands are illustrative contracts, not proof that the current repository already implements them.

## Stage A - Foundation

1. On Machine A, prepare the Ubuntu control node:

   ```bash
   scripts/bootstrap-linux.sh
   shardgrid doctor --target control
   ```

2. On each Windows GPU Worker, run the Windows bootstrap from an appropriate PowerShell session:

   ```powershell
   scripts\bootstrap-windows.ps1
   ```

3. Inside each Worker's WSL2 Ubuntu runtime, prepare the training runtime:

   ```bash
   scripts/bootstrap-wsl.sh
   ```

4. From Machine A, validate all configured machines:

   ```bash
   shardgrid doctor --target all --config examples/workers.yaml
   ```

Expected result:

- Machine A reports Control readiness.
- Machine C reports Windows + WSL2 + RTX 4060 runtime readiness.
- Machine D reports Windows + WSL2 + GTX 1060 runtime readiness.
- Any manual action is explicit and not marked successful.

## Stage B - Real Multi-Host GPU Training

1. Inspect Workers:

   ```bash
   shardgrid workers --refresh --config examples/workers.yaml
   ```

2. Probe network:

   ```bash
   shardgrid network-test --all --config examples/workers.yaml
   ```

3. Run distributed smoke test:

   ```bash
   shardgrid dist-test --backend auto --workers gpu4060,gpu1060 --config examples/workers.yaml
   ```

Expected result:

- NCCL is attempted first.
- Broadcast, send/recv, and all_reduce are tested.
- If NCCL fails, diagnostics are saved and Gloo fallback is explicitly labeled.

4. Run parallel engine compatibility spike:

   ```bash
   shardgrid probe engine galvatron --config examples/workers.yaml
   ```

Expected result:

- Galvatron compatibility report exists.
- If Galvatron fails, DeepSpeed Pipeline, PyTorch pipeline APIs, or nnScaler fallback reports are created.

5. Run minimal static two-stage training if the full engine path is not ready yet:

   ```bash
   shardgrid train examples/train-minimal.yaml --backend ssh
   ```

Expected result:

- Stage0 runs on RTX 4060.
- Stage1 runs on GTX 1060.
- Forward, activation transfer, backward, gradient transfer, optimizer step, loss decrease, and checkpoint are proven.

## Stage C - Formal SSH MVP

1. Run one-command training:

   ```bash
   shardgrid train examples/train-minimal.yaml --backend ssh
   ```

2. Inspect status:

   ```bash
   shardgrid status sg-0001
   ```

3. Inspect logs:

   ```bash
   shardgrid logs sg-0001 --tail 100
   ```

4. Stop a running job when needed:

   ```bash
   shardgrid stop sg-0001
   ```

Formal MVP acceptance:

- The user only launches from Machine A.
- Control performs discovery, probing, planning, snapshotting, distribution, launch, rendezvous, training, monitoring, logging, and result collection.
- Job snapshot contains code, config, plan, logs, environment, diagnostics, and checkpoint metadata.

## Stage D - Kubernetes + Volcano

Do this only after Stage C passes.

1. Run Kubernetes compatibility gate:

   ```bash
   shardgrid platform k8s compatibility --config examples/workers.yaml
   ```

2. If the gate passes, run Kubernetes backend training:

   ```bash
   shardgrid train examples/train-minimal.yaml --backend kubernetes
   ```

3. Install and validate Volcano only after Kubernetes training works:

   ```bash
   shardgrid platform volcano compatibility --config examples/workers.yaml
   shardgrid train examples/train-minimal.yaml --backend volcano
   ```

Expected result:

- SSH backend remains available even if Kubernetes or Volcano is blocked.
- Kubernetes and Volcano each produce compatibility reports.

## Stage E - HAMi + Multi-User Simulation

Do this only after Kubernetes and Volcano are stable.

1. Run HAMi compatibility gate:

   ```bash
   shardgrid platform hami compatibility --config examples/workers.yaml
   ```

2. If compatible, run sharing test:

   ```bash
   shardgrid train examples/share-job-a.yaml --backend volcano
   shardgrid train examples/share-job-b.yaml --backend volcano
   ```

3. Run multi-user simulation:

   ```bash
   shardgrid simulate multi-user examples/multi-user.yaml
   ```

Expected result:

- Multiple jobs can be queued and placed.
- GPU sharing is advertised only if HAMi compatibility passes.
- Isolation, logs, and results are preserved per job.
