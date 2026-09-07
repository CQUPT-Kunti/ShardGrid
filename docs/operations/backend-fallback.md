# Backend Fallback: NCCL first, Gloo only as labelled fallback

Status: active (T050) | Updated: 2026-08-18

## Rule 1: NCCL is always first

The default distributed backend is **NCCL**. Gloo is never used as the primary
backend. A run either starts with NCCL or is an explicitly requested
`--backend gloo` run; there is no automatic path that skips NCCL.

## When is Gloo fallback allowed?

Gloo may only be retried after **both** of these hold:

1. NCCL genuinely failed on the real pair of physical Workers, and the NCCL
   failure evidence has been preserved (per-rank exit codes, errors, and
   `NCCL_DEBUG=INFO` log tails).
2. The baseline conditions required for *any* distributed backend are present:
   - raw TCP reachability between the two Workers
   - rendezvous address/port usable
   - SSH / runtime wrapper working

The fallback decision is made by `run_with_fallback()` /
`decide_fallback()` in `src/shardgrid/distributed/fallback.py`. The decision
always receives the requested backend, the NCCL result and its diagnostics, and
the runtime/network/rendezvous baseline.

## How to recognise `NCCL FAILED`

A genuine NCCL failure is reported with the explicit label **`NCCL FAILED`**
and carries the NCCL diagnostics. This is only ever produced when NCCL itself
failed (process group init failure, NCCL network/interface failure, NCCL
runtime/compatibility failure, or collective failure) on a healthy baseline.

Do **not** wrap the following as "NCCL FAILED -> GLOO FALLBACK": WSL TCP down,
rendezvous port unreachable, Worker unreachable, SSH/runtime failure, or base
network misconfiguration. Those conditions produce **`FALLBACK NOT ALLOWED`**
and stop the run.

## How to recognise `GLOO FALLBACK`

When NCCL failed, the baseline is healthy, and the retry on Gloo passed, the
result is labelled **`GLOO FALLBACK`**. The NCCL failure evidence remains
attached to the same result, so the report reads:

```text
NCCL FAILED
GLOO FALLBACK: PASS
```

It must never be written as `Distributed backend: NCCL PASS`.

## Decision states

| State | Meaning |
|-------|---------|
| `NCCL SUCCESS` | NCCL passed; Gloo was never attempted. |
| `NCCL FAILED` | NCCL failed and no fallback was attempted (explicit `nccl` requested). |
| `GLOO FALLBACK` | NCCL failed; baseline healthy; Gloo retry passed. |
| `FALLBACK NOT ALLOWED` | NCCL failed but baseline (network/rendezvous/runtime) is blocked; no retry. |
| `FALLBACK FAILED` | NCCL failed; baseline healthy; Gloo retry also failed. |

## Key rule

**Gloo success does not mean NCCL success.** A `GLOO FALLBACK` result proves
Gloo works on this pair; it says nothing about NCCL. Only `NCCL SUCCESS` may be
labelled as an NCCL pass.
