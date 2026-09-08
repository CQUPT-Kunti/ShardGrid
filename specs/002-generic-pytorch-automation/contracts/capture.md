# Capture Contract

## Non-Materializing Capture Artifact

Path: job snapshot under `plan/capture-context.json` plus bounded metadata artifacts.

Required fields:

- entrypoint path, argv, working directory, environment summary;
- capture backend, backend version, and safety evidence;
- model declaration evidence without complete real parameter/buffer materialization;
- graph inputs, outputs, node IDs, value IDs, shapes, dtypes, and device intent;
- original input call structure, including positional args, keyword args, mappings, tuples, lists, masks, labels, and nested batches;
- parameter and buffer metadata, original model-state keys, canonical state IDs, shapes, dtypes, requires-grad flags, and shared/tied groups;
- optimizer class, param-group structure, hyperparameters, accumulation, mixed-precision, scheduler, and checkpoint intent when safely observable;
- counters/evidence proving no full real CPU forward, no real CPU backward, and no real CPU optimizer step occurred before planning;
- unsupported capability diagnostics and fail-closed reason.

## Rules

- Capture must complete before distributed parameter mutation.
- Capture must not require the control plane to hold the complete real model state in RAM.
- Capture must not perform a full real CPU forward/backward/optimizer step as a prerequisite for planning.
- Capture must preserve structured batches and call semantics rather than flattening away user intent.
- Unsupported capture must return a structured failure and stop before mutation.
- User code must remain ordinary PyTorch and must not implement ShardGrid model-provider, sample-input, stage, or registry APIs.

## Unsupported Examples

- model construction or state metadata cannot be represented without full materialization;
- hidden mutation cannot be replayed safely;
- dynamic Python control flow lacks safe graph/shape guards;
- custom ops lack safe metadata/capture support;
- optimizer, mixed-precision, scheduler, or checkpoint semantics cannot be mapped to owned state safely;
- any fallback would require real CPU training execution before planning.

## Gate Evidence

```text
CONTROL_PLANE_FULL_MODEL_BEFORE_PLAN=0
CPU_REAL_FORWARD_BEFORE_PLAN=0
CPU_REAL_BACKWARD_BEFORE_PLAN=0
CPU_REAL_OPTIMIZER_STEP_BEFORE_PLAN=0
ORDINARY_PYTORCH_ENTRYPOINT=PASS
```
