# Test Matrix

## Principles

- Only real execution can produce `PASS`
- Missing hardware or platform prerequisites must remain `PENDING`, `SKIPPED`, or `BLOCKED`
- Non-hardware quality checks run separately from later platform gates

## Baseline Automated Quality

| Scope | Environment | Commands | Current Status |
|---|---|---|---|
| Package install and import | Ubuntu CI / Windows CI | `python -m pip install -e ".[dev]"`, `python -c "import shardgrid"` | Configured in workflow |
| Unit and local pytest | Ubuntu CI / Windows CI | `pytest -q` | Configured in workflow |
| Lint | Ubuntu CI / Windows CI | `ruff check` | Configured in workflow |
| Type check | Ubuntu CI / Windows CI | `mypy` | Configured in workflow |

These checks are intended to run without GPU, CUDA, WSL2, Kubernetes, Volcano, HAMi, or Galvatron.

## Hardware and Platform Gates

| Scope | Environment | Trigger | Current Status Rule |
|---|---|---|---|
| Single-GPU smoke tests | Real GPU worker runtime | Explicit hardware test run | `PENDING` until executed on real hardware |
| Multi-host distributed tests | Machine A + real workers | Explicit multi-host test run | `PENDING` until executed on real hosts |
| Windows host validation | Real Windows worker | Explicit Windows validation | `PENDING` until executed on Windows |
| WSL runtime validation | Real WSL2 runtime | Explicit WSL validation | `PENDING` until executed in WSL2 |
| Kubernetes compatibility | Real Kubernetes-capable environment | Explicit Kubernetes test run | `PENDING` until executed |
| Volcano compatibility | Real Kubernetes + Volcano environment | Explicit Volcano test run | `PENDING` until executed |
| HAMi compatibility | Real Kubernetes + Volcano + HAMi environment | Explicit HAMi test run | `PENDING` until executed |

## Local Developer Expectations

Default local development on Ubuntu Machine A should be able to run:

```bash
python3 -m pip install -e ".[dev]"
python3 -c "import shardgrid; print(shardgrid.__file__)"
pytest -q
ruff check
mypy
```

Default local runs must not require:

- GPU access
- CUDA
- Windows
- WSL2
- Kubernetes
- Volcano
- HAMi
- Galvatron

## Marker Expectations

- `unit` and `local` tests are expected to run in normal local and CI quality passes
- `integration`, `hardware`, `multi_host`, `windows`, `wsl`, `kubernetes`, `volcano`, and `hami` tests are opt-in
- If those gated tests have not been executed in the required environment, they must not be reported as passed
