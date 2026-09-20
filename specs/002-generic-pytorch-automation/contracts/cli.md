# CLI Contract

## `shardgrid run`

```bash
shardgrid run [--config CLUSTER_CONFIG] [--dry-run] [--json] ENTRYPOINT [ARGS...]
```

Required behavior:

- Runs `ENTRYPOINT [ARGS...]` as the user's normal Python training program under ShardGrid bootstrap.
- Does not require user model factories, sample-input functions, ShardGrid model subclasses, or ShardGrid input formats.
- Captures planning metadata before distributed parameter mutation and without full real control-plane model materialization.
- On `--dry-run`, performs non-materializing capture, graph analysis, ownership validation, memory estimation, partitioning, placement, and admission checks without formal training mutation.
- On `--json`, emits machine-readable job/candidate/failure artifacts.
- Production admission must not launch a per-job GPU forward/backward/optimizer trial before formal training.
- Fresh host and GPU total/used/free memory discovery remains part of every planning/admission decision.

Compatibility:

- Existing `shardgrid train` remains available during migration.
- Existing config-driven automatic examples remain runnable as regression coverage.

Failure output:

- CLI must show broad stage, precise failure code, retry class, and path/log references.
- Unsupported model failures must happen before distributed mutation.
- Materialization, CPU capture execution, estimator, artifact payload, worker full-state load, and checkpoint finalization failures must be distinguishable.
