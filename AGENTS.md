<!-- SPECKIT START -->
For additional context about technologies to be used, project structure,
shell commands, and other important information, read specs/001-multi-host-training-mvp/plan.md
<!-- SPECKIT END -->

## Project Guidance

- Active Spec Kit feature: `001-multi-host-training-mvp`
- Source layout: `src/shardgrid/`
- Primary control/login node: Machine A on Ubuntu
- MVP GPU workers are separate physical machines
- Current default assumption is one GPU per physical worker
- Windows GPU workers use WSL2 Linux as the real training runtime

## Delivery Order

1. Prove real SSH-based cross-host training first
2. Add Kubernetes only after the SSH MVP is working
3. Add Volcano only after Kubernetes is stable
4. Add HAMi only after the Kubernetes/Volcano path is stable

Do not describe later phases as available until their compatibility gates and implementation tasks actually pass.

## Reuse Rules

- Prefer mature components and existing platform capabilities over ShardGrid-owned replacements
- Do not reimplement SSH, distributed runtimes, Kubernetes, Volcano, HAMi, or GPU virtualization when an adapter boundary is sufficient
- Reuse a compatible installed Python/tooling environment when available; do not force one exact patch release without a demonstrated compatibility reason

## Working References

- Spec: `specs/001-multi-host-training-mvp/spec.md`
- Plan: `specs/001-multi-host-training-mvp/plan.md`
- Tasks: `specs/001-multi-host-training-mvp/tasks.md`
