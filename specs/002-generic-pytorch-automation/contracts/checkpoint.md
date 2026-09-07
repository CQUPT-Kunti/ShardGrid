# Checkpoint Contract

## Worker Shard

Each worker shard records:

- checkpoint schema version;
- graph fingerprint;
- plan ID;
- step;
- worker ID, rank, GPU ID;
- owned logical partitions;
- parameter entries by canonical ID and original `state_dict` key;
- buffer entries by canonical ID and original `state_dict` key;
- tensor shape and dtype evidence.

## Merge

Merge must:

- reject schema, fingerprint, plan, step, rank, or ownership mismatches;
- reject duplicate canonical IDs and duplicate original state keys;
- reject missing expected parameters or buffers;
- reject shape or dtype mismatches;
- preserve original `state_dict` keys;
- produce a standard PyTorch model-state artifact;
- validate strict reload against the captured original model state structure.

## Shared State

Shared/tied parameters use one canonical state object and one checkpoint owner. Other partitions may read through explicit runtime dependencies. Ambiguous cross-partition writers are unsupported.

## Out Of Scope

Optimizer-state consolidation is not required for this feature unless a specific optimizer state can be proven safe and key-stable without extra user APIs.
