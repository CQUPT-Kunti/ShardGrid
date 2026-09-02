# ShardGrid 操作记录

## 2026-09-02 CQUPT (第三 GPU Worker) 环境准备 + T106 接入

**Worker**: CQUPT
**IP**: 10.87.5.214
**SSH user**: shardgrid（Windows OpenSSH，密码认证 + 控制节点 ed25519 密钥授权）
**GPU**: NVIDIA GeForce RTX 4060 Laptop GPU（Windows driver 566.07 / CUDA 12.7 host）
**OS**: Windows + WSL2 Ubuntu-22.04（WSL 默认用户改为 shardgrid）

### SSH / 认证

- 首次密钥认证失败（Permission denied）→ 用户提供密码后密码认证成功。
- 控制节点 `id_ed25519` 公钥已写入 CQUPT `C:\ProgramData\ssh\administrators_authorized_keys`（shardgrid 属 Administrators 组，ACL: Administrators:F / SYSTEM:F）。
- 之后 `BatchMode=yes` 密钥认证可用，ShardGrid CLI（SSHTransport）可无密码访问。

### Windows bootstrap check（bootstrap-windows.ps1 -Check -Json，-ExpectedUbuntuDistro Ubuntu-22.04）

| 项 | 结果 |
|---|---|
| OpenSSH client/server | Installed, sshd Running |
| WSL | Enabled (WSL2, Ubuntu-22.04) |
| NVIDIA driver | 566.07 ≥ 495, compatible |
| Windows-host Conda | anaconda3 @ D:\env\anaconda3（host only，不使用） |
| WSL training Conda | 检测时缺失 → 已安装 |

### WSL bootstrap check（bootstrap-wsl.sh --check --json，shardgrid 用户）

最终 health: **healthy**，无 manual actions。

### 依赖清单与处置

| DEPENDENCY | 处置 | 方式 |
|---|---|---|
| gcc-11/g++-11 11.4.0 | INSTALLED | apt（CUDA 11.8 要求 GCC ≤ 11） |
| cmake 3.22.1 / ninja 1.10.1 / make | INSTALLED | apt |
| iperf3 3.9 / iproute2 / ping | INSTALLED | apt |
| git 2.34.1 / curl / wget | REUSED | 已有 |
| CUDA Toolkit 11.8（nvcc V11.8.89） | INSTALLED | NVIDIA 官方 apt repo（wsl-ubuntu keyring）→ `/usr/local/cuda-11.8` |
| NCCL libnccl2/libnccl-dev 2.31.2-1+cuda12.9 | INSTALLED | NVIDIA 官方 apt repo（ubuntu2204 keyring），与 155 一致 |
| Miniconda (conda 26.7.1) | INSTALLED | 官方 repo.anaconda.com → `/home/shardgrid/miniconda3` |
| conda env `shardgrid` (Python 3.12.14) | CREATED | conda create，复用现有布局（与 155/15 一致） |
| PyYAML 6.0.3 + 项目 Python 依赖 | INSTALLED | pip（numpy 1.26.4, einops 0.8.2, transformers 4.49.0 等） |
| PyTorch 2.7.1+cu118 / torchvision 0.22.1+cu118 / torchaudio 2.7.1+cu118 | INSTALLED | download.pytorch.org/whl/cu118（官方） |
| NCCL backend (torch) | PASS | is_nccl_available=True (2.21.5, nvidia-nccl-cu11) |
| Gloo | PASS | is_gloo_available=True |
| Apex / Galvatron | NOT_REQUIRED（本轮） | T106 probe 不依赖；T107 前按 docs/worker-env-setup.md 官方固定 commit 安装 |

### 下载情况

- 全部 **DIRECT + OFFICIAL SOURCE**，无代理回退。
- CUDA toolkit 下载约 2.7 GB（NVIDIA 官方 apt），Miniconda 198 MB，PyTorch cu118 wheels。
- 无重复下载，无重试。

### 网络检查（不修改 MTU）

- CQUPT → 10.87.5.155: `ip route get` dev **eth0**, src 172.29.249.4, SSH 可达。
- CQUPT → 10.87.5.15: `ip route get` dev **eth0**, SSH 可达。
- ICMP ping 对 155/15 均被 Windows 防火墙拦截（100% loss），但 SSH/TCP 正常，不是主机离线。
- 本轮未改 MTU（无 evidence 表明不安全）。
- 注：第一次 inventory --refresh 时 gpu1060 曾 UNREACHABLE（瞬时 SSH 失败）；重试后三机全部 HEALTHY/eligible。

### Runtime validation（CQUPT WSL shardgrid env）

| 项 | 结果 |
|---|---|
| hostname == CQUPT | PASS (cqupt) |
| WSL2 / distro | PASS, Ubuntu-22.04 |
| nvidia-smi | PASS, RTX 4060, 566.07 |
| torch.cuda.is_available | True |
| device_count | 1 |
| GPU name | NVIDIA GeForce RTX 4060 Laptop GPU |
| torch CUDA version | 11.8 |
| NCCL / Gloo | available / available |
| gcc-11 / g++-11 | 11.4.0 |
| CUDA Toolkit / nvcc | 11.8 / V11.8.89 |
| CUDA matmul + backward + optimizer | PASS (finite grad) |

### T106 结果

- `examples/workers.yaml` 更新：`gpu4060-cqupt` → host `10.87.5.214`, runtime_distro `Ubuntu-22.04`, conda env/prefix 真实值, enabled=true。
- `shardgrid workers --refresh --json`: 3 workers；gpu4060-cqupt **HEALTHY / eligible=True**。
- `shardgrid probe --worker gpu4060-cqupt --json`: **PASS**（windows_identity=cqupt, runtime identity 正确, GPU=RTX 4060, CUDA 11.8, NCCL+Gloo）。
- enabled=false → UNAVAILABLE / eligible=False；enabled=true → HEALTHY / eligible=True。**PASS**
- 未修改任何 launcher/planner 核心逻辑。

### 状态

- T107: 未开始。
- commit: 未执行。
- 遗留：Apex/Galvatron 未安装（本轮 NOT_REQUIRED）；1650 重测可达，三机 inventory 全部 HEALTHY。

## 2026-09-02 T107 Three-Worker Distributed Smoke

**Workers**

- `gpu4060` / `LDJ` / `10.87.5.155` / NVIDIA GeForce RTX 4060 Laptop GPU
- `gpu1060` / `LAPTOP-5G3QUOGM` / `10.87.5.15` / NVIDIA GeForce GTX 1650
- `gpu4060-cqupt` / `CQUPT` / `10.87.5.214` / NVIDIA GeForce RTX 4060 Laptop GPU

**Target topology**

- `world_size=3`
- `local_world_size=1` on every host
- rank mapping: `0 -> gpu4060`, `1 -> gpu1060`, `2 -> gpu4060-cqupt`

**Preflight**

- SSH reachable: PASS on all 3 Workers
- WSL runtime reachable: PASS on all 3 Workers
- `torch import`: PASS on all 3 Workers
- `torch.cuda.is_available() == true`: PASS on all 3 Workers
- `device_count >= 1`: PASS on all 3 Workers
- `torch.distributed.is_nccl_available()`: PASS on all 3 Workers

**Dynamic route/interface evidence**

- `LDJ -> GTX1650`: `eth3`, src `10.87.5.155`, MTU `1500`
- `LDJ -> CQUPT`: `eth3`, src `10.87.5.155`, MTU `1500`
- `GTX1650 -> LDJ`: `eth0`, src `10.87.5.15`, MTU `2800 -> 1500`
- `GTX1650 -> CQUPT`: `eth0`, src `10.87.5.15`, MTU `2800 -> 1500`
- `CQUPT -> LDJ`: `eth0`, src `172.29.249.4`, MTU `1500`
- `CQUPT -> GTX1650`: `eth0`, src `172.29.249.4`, MTU `1500`

**MTU**

- 首次 preflight 发现 `gpu1060` 实际 NCCL 出口接口 `eth0` 的 MTU 为 `2800`，不满足已验证安全值 `1500`。
- 已复用现有 `scripts/bootstrap-wsl.sh --fix-nccl-mtu-only` 动态修复逻辑，对 `gpu1060 -> 10.87.5.155` 执行一次 targeted fix。
- 修复后 `gpu1060` 到 `10.87.5.155` 和 `10.87.5.214` 的 route evidence 均为 `eth0` / MTU `1500`。

**Formal hardware smoke**

- backend: `nccl`
- run_id: `t107-three-worker-smoke-20260902T052019Z`
- hardware smoke runs: `1`
- hardware smoke executed: `YES`
- 结果：`FAIL`

**Observed blocker**

- rank 0 (`gpu4060`) 在 NCCL bootstrap 后未产出 `T107_SMOKE` payload，stdout 停在：
  `NCCL INFO Bootstrap : Using eth3:10.87.5.155<0>`
- rank 0 stderr:
  `[c10d] The hostname of the client socket cannot be retrieved. err=-3`
- rank 0/1/2 均在 `120s` timeout 后退出，未进入可验证的 `init_ok/broadcast/all_reduce/barrier` 阶段。

**Artifacts**

- live evidence: `/var/tmp/shardgrid/distributed/three-worker-smoke-latest.json`
- compatibility note: `docs/compatibility/three-worker.md`
- failure bundle: `.codex-bundles/t107_t107-three-worker-smoke-20260902T052019Z_bundle_20260902_132307.zip`

**状态**

- T107: FAIL
- T107 complete: NO
- Can enter T108: NO
- T108 started: NO
- commit: NO

## 2026-09-02 T107 Network Blocker Follow-up

**目标**

- 严格只处理 `CQUPT WSL is using NAT data-plane instead of the 10.87.5.x overlay network`

**Windows WSL networking mode 对比**

- `LDJ`: `%USERPROFILE%\\.wslconfig` = `[wsl2] networkingMode=mirrored firewall=false`
- `GTX1650`: `%USERPROFILE%\\.wslconfig` = `[wsl2] networkingMode=mirrored firewall=false`
- `CQUPT` before: `.wslconfig` absent
- `CQUPT` after: `[wsl2] networkingMode=mirrored firewall=false`

**CQUPT 变更**

- 备份：`%USERPROFILE%\\.wslconfig.t107.before`，内容 `ABSENT`
- 写入 mirrored 配置后执行 `wsl --shutdown`
- 重启 WSL 后：
  - `ip addr` 新增 overlay 接口 `eth2`
  - `eth2` address = `10.87.5.214/24`
  - `ip route get 10.87.5.155` -> `dev eth2 src 10.87.5.214`
  - `ip route get 10.87.5.15` -> `dev eth2 src 10.87.5.214`

**WSL TCP data-plane 验证**

- `LDJ WSL -> CQUPT WSL`: PASS
- `CQUPT WSL -> LDJ WSL`: PASS
- `GTX1650 WSL -> CQUPT WSL`: PASS
- `CQUPT WSL -> GTX1650 WSL`: PASS
- `LDJ WSL -> GTX1650 WSL`: PASS
- `GTX1650 WSL -> LDJ WSL`: PASS

**CQUPT NCCL interface / MTU**

- 动态 route/interface: `eth2`
- 修复前 MTU: `2800`
- 复用现有 `scripts/bootstrap-wsl.sh --fix-nccl-mtu-only`
- 修复后 MTU: `1500`

**fresh T107 rerun**

- rerun id: `t107-three-worker-smoke-20260902T053606Z`
- network blocker: RESOLVED
- new blocker:
  - rank 0 `EADDRINUSE` on rendezvous port `29500`
  - rank 1 / rank 2 `socketStartConnect: Connect to 10.87.5.155<46225> failed : Software caused connection abort`
- 该 rerun 未进入 `init_ok`

**状态**

- T107 network fix: PASS
- T107 overall: FAIL
- T108 started: NO
- commit: NO

## 2026-09-02 T107 Rendezvous Port Blocker Follow-up

**目标**

- 严格只处理 rank 0 / LDJ / `10.87.5.155` 上的 rendezvous port blocker `EADDRINUSE port 29500`

**Port 29500 owner inspection**

- `ss -ltnp | grep ':29500'`:
  `users:(("python",pid=311,fd=17))`
- `lsof -iTCP:29500 -sTCP:LISTEN`:
  `python 311 shardgrid TCP *:29500 (LISTEN)`
- `ps -p 311 -o lstart,cmd`:
  `Wed Sep 2 13:20:56 2026 /home/shardgrid/miniconda3/envs/shardgrid/bin/python -`
- owner type: **STALE_SHARDGRID**
- 不是无关用户进程，也不是需要保留的非 ShardGrid 服务

**修复**

- `tests/multi_host/test_three_worker_smoke.py` 新增：
  - `29500` owner 识别
  - stale ShardGrid PID 的精确 stop
  - 若默认端口被无关进程占用，则为 fresh run 选择新的可用 rendezvous port
  - rank0/rank1/rank2 共享同一个 `MASTER_PORT`
- 本次 fresh rerun 前：
  - 成功精确清理 LDJ 上旧 ShardGrid residue PID `311`
  - `29500` 在正式启动前重新检查为空闲

**Targeted tests**

- `tests/multi_host/test_three_worker_smoke.py -k 'parse_port_owner or select_rendezvous_port or not live'`: PASS
- `tests/integration/test_ssh_launch.py --run-integration`: PASS
- `python -m py_compile tests/multi_host/test_three_worker_smoke.py`: PASS
- `ruff check tests/multi_host/test_three_worker_smoke.py`: PASS

**fresh T107 rerun**

- rerun id: `t107-three-worker-smoke-20260902T054328Z`
- `MASTER_ADDR=10.87.5.155`
- selected `MASTER_PORT=29500`
- `port_29500_owner`: `NONE`
- `stale_process_cleanup`: `true`
- same rendezvous endpoint on all 3 ranks: intended `PASS` by launch config, but formal verification incomplete because rank payloads were not returned for all ranks

**结果**

- rendezvous port blocker: **RESOLVED**
- rank 0 no longer failed with `EADDRINUSE`
- new first blocker:
  - rank 0 again timed out after NCCL bootstrap with no final `T107_SMOKE` payload
  - rank 0 stdout stopped at:
    `NCCL INFO Bootstrap : Using eth3:10.87.5.155<0>`
  - rank 0 stderr:
    `[c10d] The hostname of the client socket cannot be retrieved. err=-3`
  - rank 1 reached deeper than before and showed:
    `ncclCommInitRank ... Init COMPLETE`
    then `Broadcast: opCount 0 ...`
  - rank 2 still timed out without final payload

**Artifacts**

- live evidence: `/var/tmp/shardgrid/distributed/three-worker-smoke-latest.json`
- new failure bundle:
  `.codex-bundles/t107_t107-three-worker-smoke-20260902T054328Z_bundle_20260902_134608.zip`

**状态**

- T107 rendezvous fix: PASS
- T107 overall: FAIL
- T108 started: NO
- commit: NO
