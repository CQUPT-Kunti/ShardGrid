# Attention

Updated: 2026-08-17

This file records the practical problems already visible while the current goal
is only to get the end-to-end flow working.

The point is not to solve them all now.  The point is to keep the problems
visible so later agents do not pretend the path is already production-ready.

## 1. Storage nodes will not be enough once users and jobs increase

Current reality:

- the current setup is still a small test environment
- GPU workers are only a few machines
- jobs, logs, checkpoints, and environment evidence will grow quickly

Why this matters:

- local-only snapshot storage on the control node will become a bottleneck
- checkpoint retention and job artifact retention will become an operational
  problem before large-scale scheduling is stable
- adding more GPU workers also increases artifact distribution and collection
  pressure

What this implies later:

- storage planning cannot stay implicit
- artifact retention policy, shared storage, and cleanup policy will eventually
  need to be explicit
- adding worker nodes is not enough if artifact storage does not scale with them

## 2. SSH authentication is a real operational dependency

Current reality:

- new control machines cannot automatically access GPU workers
- each control node needs its own SSH authentication accepted by each target
  worker user
- this was already a real blocker for T038 until Machine A's public key was
  added to the Windows `shardgrid` user on `10.87.5.155` and `10.87.5.15`

Why this matters:

- the project automation uses non-interactive SSH
- password-only assumptions do not hold for the current `SSHTransport`
- a new control node will fail until its key is authorized on each worker

What this implies later:

- define one stable control-node trust model
- document worker onboarding steps for SSH key authorization
- avoid pretending runtime or GPU failures when the actual blocker is
  authentication

## 3. Different GPUs do not automatically mean the same CUDA/runtime path

Current reality:

- worker GPUs are heterogeneous
- current known GPU workers are RTX 4060 and GTX 1650
- the WSL runtime already needed real compatibility handling for PyTorch/CUDA
  wheel selection

Why this matters:

- driver visibility on Windows is not the same thing as training-runtime
  compatibility in WSL
- a package combination that works on one worker may not be the right one on a
  different worker
- heterogeneous GPU fleets need detection-first behavior, not fixed assumptions

What this implies later:

- environment preparation must continue to verify the real runtime result on
  each assigned worker
- runtime package selection should be based on actual worker compatibility
  evidence, not only on a single global default
- scheduler and bootstrap flows will need to keep worker-specific runtime facts
  retrievable

## 4. Virtual environment creation still needs to become part of scheduled worker automation

Current reality:

- environment creation and repair are still tied to worker preparation steps
- user-level workflows are not yet at the point where a job request can cause
  the assigned worker to create or reuse its environment automatically as part
  the full execution path

Why this matters:

- later users should not have to log into the worker manually just to create a
  runtime environment
- environment creation must happen on the node that will actually run the job
- the command must target the assigned compute node, not the control machine

What this implies later:

- environment create/reuse logic has to move into the job execution path for the
  selected worker
- worker assignment and environment preparation cannot stay disconnected
- the system will need to send the create/reuse instruction to the assigned
  compute node automatically before launch when required

## Current stance

Right now the priority is still to get the whole path working honestly:

- control node
- worker access
- WSL runtime
- Conda runtime selection
- distributed launch path

That is enough for the current stage.  The four issues above are not blockers
to recording real progress, but they are real future problems and should stay
visible.
