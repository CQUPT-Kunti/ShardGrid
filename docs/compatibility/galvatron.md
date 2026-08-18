# Compatibility: Galvatron - Detection Harness (T054)

Status: HARNESS READY (detection-level) | Updated: 2026-08-18 | Owner: Machine A control node

T054 implements the Galvatron compatibility-spike **harness**. It does NOT
declare Galvatron supported or unsupported on RTX 4060 / GTX 1650 / multi-host;
that decision belongs to T061 after T056-T060 capability validation.

## Current environment (selected Conda dev environment)

- Conda executable: `/home/yangjilei/anaconda3/bin/conda`
- Selected environment: `base` (prefix `/home/yangjilei/anaconda3`)
- Python: `/home/yangjilei/anaconda3/bin/python` (3.13.5)
- PyTorch: not installed in this control-plane environment
- CUDA runtime: not detected (no PyTorch)

## Galvatron detection

Check-only detection runs these probes against the selected Conda Python
(no environment change):

- `python -m pip show galvatron`
- `import galvatron` (records `__file__` and `__version__`)
- `git -C <import location> remote get-url origin` (source verification)
- PyTorch probe: `import torch` (version, `torch.version.cuda`, cuda available)

Result statuses are strictly separated:

| Status | Meaning |
|--------|---------|
| `AVAILABLE` | official Galvatron importable in the selected environment with runtime evidence; detection-level only, capability validation is T056-T060 |
| `NOT INSTALLED` | not present; official install command recorded as a manual action |
| `INCOMPATIBLE` | present but import/runtime broken in the selected environment |
| `BLOCKED` | official install would damage the environment, source unverifiable/unofficial, or manual action required |
| `CHECK FAILED` | harness itself could not complete evidence collection |

`AVAILABLE` never means "fully compatible": it only proves detect + import +
version + runtime evidence. Import success alone is never treated as full
compatibility.

## Official sources only

- PyPI package `galvatron` (verified via recorded pip metadata)
- Official GitHub repo: `https://github.com/PKU-DAIR/Hetu-Galvatron`
  (verified via `Home-page` metadata or `git remote get-url origin`)

Unofficial forks, patched sources, or unverifiable locations are reported as
`BLOCKED` with a manual action; the harness never installs or patches them.

## Install path (opt-in, never in check mode)

- Default mode is check-only: no install, no backend change.
- `allow_install=True` runs only the official command after a `pip install
  --dry-run` preflight. If the dry-run would change torch / nvidia / triton /
  cuda / tensorrt packages, installation stops and is recorded `BLOCKED`.
- Editable (GitHub) install additionally requires a detected Conda identity.
- Any install failure is recorded with preserved diagnostics and a manual
  action; Galvatron source is never patched.

## Current compatibility check

`run_galvatron_check()` on Machine A (2026-08-18):

- status: `NOT INSTALLED`
- Galvatron version/source: none detected
- blocker/diagnostics: PyTorch not installed in the selected control-plane
  environment (expected on Machine A; GPU Workers are checked separately in
  T055-T057)
- evidence: saved via `save_galvatron_evidence()` (run-id + `galvatron-latest.json`)

## Rules

- Detect-first / reuse-first: an existing official Galvatron in the selected
  environment is reused; no upgrade or replacement without evidence.
- If installation would break the selected environment, stop and record
  manual action / blocker.
- No final "Galvatron supported/unsupported" conclusion yet: final decision is
  T061.

## Diagnostics

- Implementation: `src/shardgrid/engines/compatibility.py`
- Tests: `tests/unit/test_engine_compatibility.py` (21 tests)
- Evidence schema: `GalvatronCompatibilityResult` (run_id, status, versions,
  Conda/Python/PyTorch/CUDA, commands, per-command diagnostics with output
  tails, blockers, manual actions, timing)
