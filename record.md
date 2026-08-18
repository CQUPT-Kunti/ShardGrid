
## T046

- 实现最小 torch.distributed smoke 程序（examples/distributed_smoke/smoke.py）
- 支持 rank / world-size / master-addr / master-port / backend / local-rank 参数
- 仅使用官方 torch.distributed API（init/barrier/destroy），不实现协议
- 记录 Conda env/prefix、Python、PyTorch、torch.version.cuda、backend、rank/world_size/local_rank runtime evidence
- 测试结果：PASS（12/12 unit；全量 240 passed；ruff/mypy 通过；Machine A 无 torch 时 init 失败结构化返回，不影响判定）

## T047

- 实现 multi-host distributed runner（src/shardgrid/distributed/runner.py）
- 一台物理 Worker 一个进程，local_world_size=1，local_rank=0；RTX 4060=rank0、GTX 1650=rank1
- 生成 rank0/rank1 dry-run 命令（复用 T040 WSL2 Conda runtime + T046 smoke 程序），不真实启动
- 复用 Worker/Runtime config、ProcessResult/FailureRecord；无硬编码 IP/python/prefix/distro
- 测试结果：PASS（12/12；全量 240 passed；ruff/mypy 通过）

## T048
- backend/interface/rendezvous selection（src/shardgrid/distributed/backend.py）
- NCCL first；Gloo 仅显式 fallback（T050）
- interface 来自 NetworkState；rendezvous 校验 reachable；diagnostics 支持 NCCL_DEBUG=INFO
- 测试结果：PASS（11/11）

## T049
- 真实执行 RTX 4060 rank0 + GTX 1650 rank1 NCCL collectives
- 2026-08-18 修复并重跑：
  - `WSLRuntimeWrapper.run_script()` 去掉会破坏 `wsl.exe ... /bin/bash -lc` 的 `PATH="$PATH"` 注入，仅保留 `CONDA_PREFIX` / `CONDA_DEFAULT_ENV` 和绝对 Conda Python
  - collectives harness 按本轮 `ip route get` 动态记录每个 rank 的实际接口、`ip_local_port_range`、阶段日志和 NCCL effective env
  - live 测试复用 `tests/address.json` 的真实 Worker 地址，不手写 Python / Conda prefix / WSL distro
- 最新 live 结果：PASS（证据保存在 `/var/tmp/shardgrid/distributed/collectives-latest.json`）
  - rank0：RTX 4060 Worker `10.87.5.155` / `LDJ` / WSL `Ubuntu` / Conda `shardgrid` / interface `eth3`
  - rank1：GTX 1650 Worker `10.87.5.15` / `LAPTOP-5G3QUOGM` / WSL `Ubuntu` / Conda `shardgrid` / interface `eth0`
  - backend：`nccl`
  - rendezvous：`master_addr=10.87.5.155`, `master_port=29500`
  - runtime Python：两边都是 `/home/shardgrid/miniconda3/envs/shardgrid/bin/python`
  - PyTorch / CUDA：两边 `torch 2.7.1+cu118`，runtime CUDA `11.8`
  - process group：`AFTER_INIT` 达成
  - `broadcast`：PASS，tensor=`[11.0, 22.0, 33.0, 44.0]`
  - `send/recv`：PASS，tensor=`[5.0, 6.0, 7.0, 8.0]`
  - `barrier`：PASS
  - `all_reduce`：PASS，tensor=`[3.0, 3.0, 3.0, 3.0]`
- 质量检查：live pytest `1 passed in 46.70s`；相关 pytest `16 passed, 8 skipped`；`ruff check` 通过；`mypy` 通过
- T049 状态：可以标记 `[X]`；Gate 2 已满足进入 T050 的前置条件
