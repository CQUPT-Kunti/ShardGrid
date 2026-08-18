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

## T055 - Declared requirements vs Worker runtimes (2026-08-18)

### Galvatron declared requirements (official source only)

- Source: `github:PKU-DAIR/Hetu-Galvatron` (official `setup.py`, main branch)
- Version: `2.4.1` (package name `hetu-galvatron`)
- Python: `>=3.8`
- Declared install requirements (verbatim):
  `torch>=2.0.1`, `torchvision>=0.15.2`, `numpy<2.0.0`,
  `transformers==4.49.0`, `h5py>=3.6.0`, `attrs>=21.4.0`, `yacs>=0.1.8`,
  `six>=1.15.0`, `sentencepiece>=0.1.95`, `pybind11>=2.9.1`, `scipy>=1.10.1`
- Conditional (`GALVATRON_FLASH_ATTN_INSTALL=TRUE`): `packaging`,
  `flash-attn>=2.0.8`
- Build requirements: C++ compiler (`csrc/dp_core.cpp`, C++11, `-fPIC`),
  `pybind11>=2.9.1` (setup)
- CUDA: no explicit CUDA requirement declared (not guessed)
- PyPI finding: the PyPI package `galvatron` (0.0.3) is **NOT** the official
  PKU-DAIR project (home page `github.com/kyegomez/Galvatron`); PyPI install
  is rejected as unofficial. Official source = GitHub only.

### RTX 4060 Worker (gpu4060, WSL2 selected Conda `shardgrid`)

- Conda: env `shardgrid`, prefix `/home/shardgrid/miniconda3/envs/shardgrid`
- Python: 3.12.13 | PyTorch: 2.7.1+cu118 | torch CUDA runtime: 11.8
- Driver: 566.07 | GPU: NVIDIA GeForce RTX 4060 Laptop GPU | cap 8.9
- Galvatron: NOT INSTALLED

### GTX 1650 Worker (gpu1060, WSL2 selected Conda `shardgrid`)

- Conda: env `shardgrid`, prefix `/home/shardgrid/miniconda3/envs/shardgrid`
- Python: 3.12.13 | PyTorch: 2.7.1+cu118 | torch CUDA runtime: 11.8
- Driver: 527.41 | GPU: NVIDIA GeForce GTX 1650 | cap 7.5
- Galvatron: NOT INSTALLED

### Compatibility comparison

| Worker | python (>=3.8) | pytorch (>=2.0.1) | cuda (declared) | Galvatron | status |
|--------|----------------|-------------------|-----------------|-----------|--------|
| gpu4060 | MATCH (3.12.13) | MATCH (2.7.1+cu118) | REQUIREMENT UNKNOWN (actual 11.8) | NOT INSTALLED | NOT INSTALLED |
| gpu1060 | MATCH (3.12.13) | MATCH (2.7.1+cu118) | REQUIREMENT UNKNOWN (actual 11.8) | NOT INSTALLED | NOT INSTALLED |

Overall: `NOT INSTALLED` - Python and PyTorch satisfy the declared
requirements on both Workers; no Galvatron installation exists yet.

### Blockers / findings

- Galvatron is not installed on either Worker (expected; T056 install spike
  decides).
- Declared dependencies are mostly absent on both Workers: numpy,
  transformers, torchvision, yacs, sentencepiece, scipy, h5py, attrs, six,
  pybind11, flash_attn are NOT present; only `packaging` 26.3 is installed.
  (Verified live via the WSL selected Conda runtime.)
- PyPI `galvatron` is an unofficial third-party package; only the official
  GitHub source is acceptable.
- No environment change was made: no upgrade/downgrade of Python, PyTorch,
  CUDA, and no Conda env creation/replacement (detect-first / reuse-first).
- T055 PASS means "version compatibility comparison completed" only; it does
  not mean Galvatron ran on the GPUs (that is T056-T060).
- Final Galvatron support decision remains T061.

### Evidence

- `tests/integration/test_engine_versions.py` (13 logic cases + 1 live test)
- `src/shardgrid/engines/compatibility.py` (declared-requirements collection,
  worker version evidence, comparison)
- Live evidence: `/var/tmp/shardgrid/engines/galvatron-versions-latest.json`

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
