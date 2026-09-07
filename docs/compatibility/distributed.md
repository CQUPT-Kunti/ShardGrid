# Gate 2 Distributed Compatibility

更新日期：2026-08-18

## 范围

本文件只记录 T053 的真实 Gate 2 验收结果，不写泛化设计。

真实 Worker：

- rank0: `gpu4060` / `10.87.5.155` / `LDJ` / RTX 4060
- rank1: `gpu1060` / `10.87.5.15` / `LAPTOP-5G3QUOGM` / GTX 1650

## Gate 1 前置状态

- RTX 4060 single-GPU Gate: `PASS`
- GTX 1650 single-GPU Gate: `PASS`
- Gate 1 evidence: `/var/tmp/shardgrid/gates/gate1-latest.json`

## 本轮 Gate 2 真实执行

执行路径：

- Machine A
- SSH
- Windows physical host
- WSL2 Ubuntu
- selected Conda env `shardgrid`
- PyTorch Distributed

本轮真实 pair report：

- `/var/tmp/shardgrid/gates/dist-test-gate2-live.json`

本轮 Gate 2 evidence：

- `/var/tmp/shardgrid/gates/gate2-latest.json`

## 真实环境证据

两台 Worker 目标训练环境：

- conda env: `shardgrid`
- conda prefix: `/home/shardgrid/miniconda3/envs/shardgrid`
- python: `/home/shardgrid/miniconda3/envs/shardgrid/bin/python`

已知网络接口：

- rank0 interface: `eth3`
- rank1 interface: `eth0`
- master_addr: `10.87.5.155`
- master_port: `29500`

## backend 结果

- requested backend: `auto`
- backend_state: `FALLBACK FAILED`
- actual backend: `gloo`

说明：

- NCCL 先尝试，失败证据已保留
- Gloo fallback 也未建立起 process group
- 因此不能写成 `NCCL PASS`
- 也不能写成 `GLOO FALLBACK PASS`

## collective 结果

### NCCL

- process group: `FAIL`
- broadcast: `FAIL`
- send/recv: `FAIL`
- all_reduce: `FAIL`

已保留的真实失败信息包含：

- rank1 NCCL error:
  - `socketStartConnect: Connect to 10.87.5.155<60728> failed : Software caused connection abort`
- rank0 timeout
- per-rank stdout/stderr tail

### Gloo fallback

- process group: `FAIL`
- broadcast: `FAIL`
- send/recv: `FAIL`
- all_reduce: `FAIL`

本轮 Gloo fallback 的两个 rank 都只到了：

- `BEFORE_INIT`

没有拿到：

- `AFTER_INIT`
- `AFTER_BROADCAST`
- `AFTER_SEND_RECV`
- `AFTER_ALL_REDUCE`

## Gate 2 结论

本轮真实结果：

- Gate 2 status: `FAIL`

失败原因：

1. backend 状态不是允许通过的 `NCCL SUCCESS` 或 `GLOO FALLBACK`
2. process group 未建立
3. required collectives 未成功
4. 本轮 actual backend 的 runtime evidence 不完整

## 当前判断

- T053 implementation：已实现
- Gate 2 hardware PASS：未通过
- T053 不能标记为 `[X]`
- 不能进入 T054
