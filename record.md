
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

## T050
- 实现 NCCL -> Gloo fallback（src/shardgrid/distributed/fallback.py）
  - 明确 decision 状态：`NCCL SUCCESS` / `NCCL FAILED` / `GLOO FALLBACK` / `FALLBACK NOT ALLOWED` / `FALLBACK FAILED`
  - NCCL 永远首选；Gloo 只在 NCCL 真失败且 network/rendezvous/runtime baseline 正常时 fallback
  - baseline blocked（TCP/rendezvous/Worker 不可达/SSH/runtime 失败）判为 `FALLBACK NOT ALLOWED`，不误触发
  - NCCL failure evidence（per-rank exit_code / error / log tail）保留在 FallbackDecision.to_dict()
- Gloo fallback backend 明确标记：Gloo 成功标 `GLOO FALLBACK`，绝不当成 NCCL 成功
- 测试：tests/multi_host/test_gloo_fallback.py 12/12 通过（unit；live 测试 multi_host opt-in 未运行）
- docs/operations/backend-fallback.md 已更新
- 测试结果：T050 相关 pytest 12/12 passed；multi_host unit 49 passed；全量 pytest 232 passed（8 个 gpu_probe contract 失败为基线已有，非 T050 引入）；ruff check 通过；mypy 通过
- live result：**BLOCKED**（T049 最新真实 NCCL 结果为 PASS，无有效 NCCL failure evidence，按规则不允许真实 Gloo fallback 复跑）
- T050 implementation：DONE；T050 live fallback：BLOCKED

## T051
- 实现 `shardgrid dist-test`（src/shardgrid/cli/commands/dist_test.py）
- 支持 `--backend auto|nccl|gloo`、`--workers`、`--save-report`；`dist-test` 已从 placeholder 移除并注册真实命令
- 复用 T046 smoke / T047 runner / T048 backend / T049 collectives / T050 fallback / SSHTransport / WSL wrapper，不重新实现 launch
- 修复 T049 collectives harness：`init_process_group` 使用请求的 backend（原来硬编码 nccl）；Gloo 用 CPU tensor（Gloo TCP 对 CUDA P2P send/recv 有 `writev: Bad address` 已知问题，NCCL 仍用 CUDA tensor）
- backend 语义：nccl 失败即失败不切换；gloo 输出 backend=gloo；auto 走 T050 decision（NCCL first，仅在 NCCL 真失败+baseline 正常时 Gloo fallback，保留 NCCL failure evidence）
- `--save-report` 输出 JSON 报告：workers/backend requested+actual/collectives/runtime evidence/network/failure diagnostics；不写密码/凭据
- exit code：NCCL SUCCESS / GLOO FALLBACK / GLOO PASS = 0；NCCL FAILED / GLOO FAILED / FALLBACK FAILED / FALLBACK NOT ALLOWED = 非零
- tests：integration 14/14 passed；multi_host unit 49 passed；ruff 通过；mypy 通过（100 files）；全量 232 passed（8 个 gpu_probe contract 失败为基线已有）
- live result：PASS
  - `dist-test --backend auto`：NCCL SUCCESS，rank0 RTX4060(10.87.5.155, eth3)/rank1 GTX1650(10.87.5.15, eth0)，broadcast/send-recv/all_reduce 全 PASS，exit 0
  - `dist-test --backend gloo`：GLOO PASS（CPU tensor），collectives 全 PASS，exit 0
  - 报告：`/var/tmp/shardgrid/distributed/reports/dist-test-live-auto.json`、`dist-test-live-gloo.json`
- T051 implementation：DONE；T051 live verification：PASS

## T052
- Gate 1 single-GPU acceptance（src/shardgrid/workers/single_gpu_gate.py）
- 统一 Gate 逻辑：一套 smoke script + 一个 evaluator，分别对两张 GPU 独立执行（all-or-nothing）
- 复用 T040 WSLRuntimeWrapper（SSH->WSL2->selected Conda）、T041 probe 约定、ProcessResult；不重新实现 SSH/WSL/Conda/GPU probe
- smoke 校验：physical host / WSL2 runtime / Conda env / Python / torch / torch.version.cuda / cuda_available / device_count / GPU name（必须匹配）/ driver / 1024x1024 CUDA matmul + finite 校验 / timestamp
- 状态语义：PASS（双 GPU 真 smoke PASS）；FAIL（真执行但至少一张失败，含 CUDA 不可用/GPU 不匹配/tensor 失败）；BLOCKED（SSH/WSL/环境/硬件不可访问或 evidence 缺失）；PENDING（未执行）；BLOCKED/PENDING 永不误判 PASS
- RTX 4060 CUDA/PyTorch smoke：**PASS**（RTX 4060 Laptop GPU cap8.9/8187MB，driver 566.07，torch 2.7.1+cu118，CUDA 11.8，matmul finite）
- GTX 1650 CUDA/PyTorch smoke：**PASS**（GTX 1650 cap7.5/4095MB，driver 527.41，torch 2.7.1+cu118，CUDA 11.8，matmul finite）
- Gate 1：**PASS**（evidence：`/var/tmp/shardgrid/gates/gate1-latest.json`）
- docs/compatibility/single-gpu.md 已更新
- tests：hardware logic 19/19 passed；live 1/1 passed（真实双 Worker）；全量回归 232 passed（8 个 gpu_probe contract 失败为基线已有）；ruff/mypy 通过
- T052 implementation：DONE；RTX 4060 live：PASS；GTX 1650 live：PASS；Gate 1：PASS

## T053
- Gate 2 distributed acceptance（src/shardgrid/distributed/distributed_gate.py, tests/multi_host/test_distributed_gate.py）
- 复用 T051 `dist-test` 报告，不新建 distributed resource model；Gate 2 直接消费：
  - backend requested / backend_state / backend_actual
  - workers / ranks / world_size
  - collectives
  - runtime_evidence
  - network
  - nccl_failure_evidence
- 修正 Gate 2 evidence 判定：
  - actual backend 为 Gloo fallback 时，如果顶层 `runtime_evidence.per_rank` 被失败的 NCCL 尝试污染，改为回退读取 `collectives[actual_backend].ranks`
  - 增加 diagnostics evidence 检查
  - 增加缺失 backend label / diagnostics / actual-backend runtime fallback 的测试覆盖
- 本地验证：
  - `SHARDGRID_ENABLE_MULTI_HOST_TESTS=1 pytest -q --run-multi-host tests/multi_host/test_distributed_gate.py -k 'not live_gate2_real_pair'` -> `20 passed`
  - `ruff check` 通过
  - `mypy` 通过
- 本轮真实 pair test：
  - Gate 1：RTX 4060 `PASS`，GTX 1650 `PASS`
  - rank0 Worker：`10.87.5.155` / `LDJ` / RTX 4060
  - rank1 Worker：`10.87.5.15` / `LAPTOP-5G3QUOGM` / GTX 1650
  - physical hosts distinct：YES
  - requested backend：`auto`
  - actual backend：`gloo`
  - backend_state：`FALLBACK FAILED`
  - process group：FAIL
  - broadcast：FAIL
  - send/recv：FAIL
  - all_reduce：FAIL
  - tensor validation：FAIL
  - network interface：rank0=`eth3`，rank1=`eth0`
  - diagnostics：已保存在 `/var/tmp/shardgrid/gates/dist-test-gate2-live.json` 与 `/var/tmp/shardgrid/gates/gate2-latest.json`
  - 保留的 NCCL 真实失败：`socketStartConnect: Connect to 10.87.5.155<60728> failed : Software caused connection abort`
- 结果：FAIL
- T053 状态：未标记 `[X]`；Gate 2 本轮未真实通过；不能进入 T054

## T054
- Galvatron compatibility spike harness（src/shardgrid/engines/compatibility.py）
- detect-first / reuse-first：只检查 selected Conda 环境，默认 check-only，不改 backend、不装包、不替换环境
- 记录：检测命令、Galvatron 是否存在、version/source、Conda env/prefix、Python executable/version、PyTorch version、CUDA runtime version、执行时间、result/failure diagnostics（每命令 stdout/stderr tail）
- 状态明确区分：AVAILABLE / NOT INSTALLED / INCOMPATIBLE / BLOCKED / CHECK FAILED；import 成功不等于完全兼容（AVAILABLE 仅 detection-level，能力验证是 T056-T060）
- 只允许官方来源：PyPI `galvatron`（pip 元数据验证）或 `https://github.com/PKU-DAIR/Hetu-Galvatron`（Home-page / git origin 验证）；unofficial/未验证来源 → BLOCKED + manual action
- 安装路径 opt-in（allow_install）：`pip install --dry-run` preflight，若会改动 torch/nvidia/triton/cuda/tensorrt → BLOCKED；GitHub editable 安装需 Conda 身份；安装失败保留 diagnostics + manual action
- 复用 T016 run_process、T036 detect_conda/detect_python、T014 CompatibilitySpikeReport（to_spike_report 映射）
- 现场 check 结果（Machine A，base env）：NOT INSTALLED（Galvatron 不存在；torch 未安装于控制平面环境，已记录 diagnostics）
- 测试结果：PASS（21/21 unit；全量 257 passed + 148 skipped，8 个 gpu_probe contract 失败为基线已有；ruff/mypy 通过，107 files）
- docs/compatibility/galvatron.md 已更新；未写最终支持/不支持结论（属 T061）
- T054 implementation：DONE；compatibility check（Machine A）：NOT INSTALLED（符合预期）
