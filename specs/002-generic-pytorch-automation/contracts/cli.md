# CLI Contract

## `shardgrid run`

```bash
shardgrid run [--config CLUSTER_CONFIG] [--dry-run] [--json] ENTRYPOINT [ARGS...]
```

Required behavior:

- Runs `ENTRYPOINT [ARGS...]` as the user's normal Python training program under ShardGrid bootstrap.
- Does not require user model factories, sample-input functions, ShardGrid model subclasses, or ShardGrid input formats.
- Captures the first usable training step before distributed parameter mutation.
- On `--dry-run`, performs capture, graph analysis, ownership validation, partitioning, placement, and admission checks without formal training mutation.
- On `--json`, emits machine-readable job/candidate/failure artifacts.

Compatibility:

- Existing `shardgrid train` remains available during migration.
- Existing config-driven automatic examples remain runnable as regression coverage.

Failure output:

- CLI must show broad stage, precise failure code, retry class, and path/log references.
- Unsupported model failures must happen before distributed mutation.
