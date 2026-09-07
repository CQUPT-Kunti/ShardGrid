# Capture Contract

## Captured Context Artifact

Path: job snapshot under `plan/capture-context.json` plus any tensor metadata files needed by runtime.

Required fields:

- entrypoint path, argv, working directory, environment summary;
- capture backend and backend version evidence;
- model identity, module/class evidence, parameter count, buffer count;
- first usable batch structure;
- model positional args and keyword args;
- tensor shapes, dtypes, devices, requires-grad flags;
- optimizer class, param-group structure, hyperparameters where observable;
- loss/backward/optimizer/scheduler boundary evidence;
- original `state_dict` key to canonical state ID map;
- unsupported capability diagnostics.

Rules:

- Capture must occur before distributed parameter mutation.
- Capture must preserve structured batches rather than flattening away call semantics.
- ShardGrid internals may adapt/intercept; user code must not implement ShardGrid-specific APIs.
- Unsupported capture must return a structured failure and stop.

Unsupported examples:

- hidden model mutation that cannot be replayed safely;
- dynamic Python control flow not captured by export/FX;
- custom ops without export/fake/meta support;
- optimizer semantics that cannot be mapped to owned state safely.
