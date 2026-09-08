# Checkpoint Contract

## Worker Shard

Each worker shard records:

- checkpoint schema version;
- graph fingerprint;
- plan ID;
- step;
- worker ID, rank, GPU ID;
- owned logical partitions;
- state entries by canonical ID and original model-state key;
- tensor shape and dtype evidence;
- payload reference and checksum/fingerprint;
- ownership and read/write evidence.

## Merge / Finalization

Finalization must:

- reject schema, fingerprint, plan, step, rank, or ownership mismatches;
- reject duplicate canonical IDs and duplicate original state keys;
- reject missing expected parameters or buffers;
- reject shape or dtype mismatches;
- preserve original model-state keys;
- produce a standard model-state-compatible artifact or stream;
- validate compatibility against the captured original model-state structure;
- avoid requiring the control plane to hold the entire model state in memory at once for large-state models.

## Shared State

Shared/tied parameters use one canonical state object and one checkpoint owner. Other partitions may read through explicit runtime dependencies. Ambiguous cross-partition writers are unsupported.

## Large-State Safety

For models larger than the configured control-plane RAM budget:

- shard inspection must be streaming or bounded;
- validation must operate from manifests and chunked payload reads where possible;
- finalization must not rebuild the model by name;
- finalization must not first load all worker state tensors into one in-memory object.

## Out Of Scope

Optimizer-state consolidation is not required unless a specific optimizer state can be proven safe and key-stable without extra user APIs.

## Gate Evidence

```text
STANDARD_STATE_DICT_COMPATIBLE=PASS
CHECKPOINT_CONTROL_PLANE_FULL_STATE_LOAD=0
WORKER_OWNED_STATE_ONLY=PASS
```
