# Agent Handoff

Updated: 2026-08-18

这个文件只写交接重点。下一位 Agent 先看这里，再碰 Windows / WSL / NCCL / multi-host。

## 当前真实状态

- `T049` 已真实跑通，已经标记为 `[X]`
- Machine A 可以通过 SSH 到两台 Windows GPU Worker
- 真实训练 runtime 是：
  - `Windows Host`
  - `WSL2 Ubuntu`
  - `shardgrid` Conda env
  - `/home/shardgrid/miniconda3/envs/shardgrid/bin/python`
- 不要把 Windows Python / Windows Conda 当成训练 runtime

当前 GPU Worker：

- `10.87.5.155` / `LDJ` / `shardgrid` / RTX 4060 / rank0
- `10.87.5.15` / `LAPTOP-5G3QUOGM` / `shardgrid` / GTX 1650 / rank1

地址来源：

- `tests/address.json`

## 前面真正踩过的坑

### 1. 状态污染

之前有过旧结论污染当前判断的问题：

- 明明 T029 / T030 已完成，却被旧上下文继续判成未完成
- 明明 Worker 已准备好，却被旧状态说成还没准备
- 明明 SSH 已修好，却还在按旧结论说访问 blocked

结论：

- 不要盲信旧聊天摘要
- 先看当前文件：
  - `specs/001-multi-host-training-mvp/tasks.md`
  - `record.md`
  - `docs/operations/*.md`

### 2. SSH 通了，不等于训练链路通了

真正要分层看：

1. Machine A -> Windows SSH
2. Windows -> WSL Ubuntu
3. WSL -> selected Conda env
4. runtime Python 身份
5. raw TCP
6. TCPStore
7. NCCL

不要把 SSH 成功误写成 distributed 成功。

### 3. Windows Host IP、WSL 路由、历史缓存 IP 容易混

前面出过的问题：

- 旧 `NetworkState` 污染了新的 rendezvous 选择
- `master_addr` 曾经被选错

这次真实跑通时的关键值：

- `master_addr=10.87.5.155`
- `master_port=29500`

### 4. 网卡名不能写死

这是实机里最容易误判的一层。

前面出现过：

- 以为接口固定是 `eth0` 或 `eth3`
- 实际上必须按本轮路由动态取

本次真实成功时：

- `10.87.5.155 -> 10.87.5.15` 走 `eth3`
- `10.87.5.15 -> 10.87.5.155` 走 `eth0`

结论：

- 不要写死 `NCCL_SOCKET_IFNAME`
- 每次都用：

```bash
ip route get <peer_ip>
```

取 `dev`

### 5. 不要默认强制改 `ip_local_port_range`

之前试过强制改成 `50000 51000`，结果并不稳定。

这次最终真实跑通时，两台成功使用的是：

```text
44620 48715
```

结论：

- 现在代码里没有强制 `sysctl -w`
- 不要默认在启动前改端口范围
- 先记录当前值，再判断

### 6. 旧 Python 进程会污染下一轮测试

前面出现过：

- 上一轮失败留下 `python -`
- `29500` 被占用
- 看起来像网络问题，实际是残留进程问题

检查方式：

```bash
ps -ef | grep '[p]ython'
ss -ltnp | grep ':29500'
```

### 7. 真正的训练 Python 必须来自 WSL Conda

必须明确记录：

- `CONDA_PREFIX`
- `CONDA_DEFAULT_ENV`
- `python_executable`

正确 runtime Python：

```text
/home/shardgrid/miniconda3/envs/shardgrid/bin/python
```

### 8. 一个真实代码坑：WSL 直连命令 quoting

之前不是网络坏了，是命令包装层被改坏了。

出错点：

- `run_script()` payload 里拼了 `PATH="$PATH"`
- 破坏了 `wsl.exe ... /bin/bash -lc` 的 quoting

修复后：

- 不再给 stdin-fed 脚本强塞 `PATH="$PATH"`
- 只保留：
  - `CONDA_PREFIX`
  - `CONDA_DEFAULT_ENV`
  - 绝对 Conda Python 路径

## T049 真实通过时的结果

- backend：`nccl`
- rank0：RTX 4060
- rank1：GTX 1650
- `broadcast`：PASS
- `send/recv`：PASS
- `barrier`：PASS
- `all_reduce`：PASS

证据文件：

- `/var/tmp/shardgrid/distributed/collectives-latest.json`

这里面有：

- rank0/rank1 的 WSL runtime 证据
- conda env / prefix
- runtime Python
- torch / CUDA
- network interface
- 各阶段日志：
  - `BEFORE_INIT`
  - `AFTER_INIT`
  - `BEFORE_BROADCAST`
  - `AFTER_BROADCAST`
  - `BEFORE_SEND_RECV`
  - `AFTER_SEND_RECV`
  - `BEFORE_BARRIER`
  - `AFTER_BARRIER`
  - `BEFORE_ALL_REDUCE`
  - `AFTER_ALL_REDUCE`

## 建议先看的文件

- `tests/address.json`
- `specs/001-multi-host-training-mvp/tasks.md`
- `record.md`
- `docs/operations/windows-wsl-nccl-troubleshooting.md`
- `docs/operations/remote-access.md`
- `docs/operations/bootstrap-findings.md`

## 下一位 Agent 不要再犯的错

1. 不要把旧摘要当成最新事实
2. 不要把 SSH 成功当成训练链路成功
3. 不要写死 `eth0` / `eth3`
4. 不要默认强制改端口范围
5. 不要忽略旧 `python` 残留进程
6. 不要把 Windows Python 当成训练 Python
7. 不要看到 NCCL timeout 就直接判 GPU/CUDA 问题
