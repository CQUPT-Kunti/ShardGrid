# Windows + WSL2 + NCCL 多机排障记录

更新日期：2026-08-18

本文档记录 ShardGrid 在真实双 Windows GPU Worker 上做 WSL2 + Conda + PyTorch Distributed / NCCL 联调时，已经实际遇到过的问题、判断方法和处理结论。

目标不是讲原理，而是避免下一个 Agent 从零开始误判。

## 适用范围

当前已验证的真实节点：

- `10.87.5.155` / `LDJ` / `shardgrid` / RTX 4060 / WSL Ubuntu
- `10.87.5.15` / `LAPTOP-5G3QUOGM` / `shardgrid` / GTX 1650 / WSL Ubuntu

当前控制节点：

- Machine A / Ubuntu

当前训练运行时约束：

- 真正的训练 Python 必须来自 WSL2 内的 Conda 环境
- 不允许把 Windows Host Python 当成训练 runtime
- SSH 只是进入 Windows Host 的通道，训练实际发生在 `Windows -> WSL2 Ubuntu -> Conda env`

## 先记住的结论

1. `SSH 能连上` 不等于 `WSL 训练链路可用`。
2. `Windows Host IP 可达` 不等于 `WSL 到 WSL 的 TCP/NCCL 一定可用`。
3. 网卡名不能写死，必须每次按本轮路由动态取。
4. `NCCL_SOCKET_IFNAME` 选错时，会看起来像 NCCL 卡死，但根因其实是接口选错。
5. 不要默认强制改 `ip_local_port_range`。这次最终跑通时，两台实际成功使用的是默认范围 `44620 48715`。
6. 旧的 `python` 残留进程会污染下一次测试，尤其会占住 `29500`。
7. 历史缓存的网络状态会污染新的 rendezvous 选择，不能盲信旧状态。

## 这次真实跑通时的关键条件

### Windows / WSL

两台 Windows 用户 `C:\\Users\\shardgrid\\.wslconfig`：

```ini
[wsl2]
networkingMode=mirrored
firewall=false
```

说明：

- 这里只是记录本次联调时的成功条件
- `firewall=false` 是为了先证明链路能通
- 后续如果要恢复更严格的防火墙策略，要在链路验证通过后单独做

### WSL 内实际路由和接口

4060 节点：

- Windows Host IP：`10.87.5.155`
- 到对端 `10.87.5.15` 的实际接口：`eth3`

1650 节点：

- Windows Host IP：`10.87.5.15`
- 到对端 `10.87.5.155` 的实际接口：`eth0`

注意：

- 不要把 `eth3` / `eth0` 写死到代码里
- 每次运行前都要用 `ip route get <peer_ip>` 动态取 `dev`

### WSL Conda 运行时

两台都使用：

- distro：`Ubuntu`
- user：`shardgrid`
- conda env：`shardgrid`
- conda prefix：`/home/shardgrid/miniconda3/envs/shardgrid`
- python：`/home/shardgrid/miniconda3/envs/shardgrid/bin/python`

## 已经真实遇到过的问题

## 1. SSH 可用，但不是训练失败原因的终点

现象：

- Machine A 可以 SSH 到 Windows Worker
- 但 T038/T049 仍然失败

真实原因：

- SSH 只说明 `Machine A -> Windows Host` 通了
- 后面还有：
  - `Windows -> WSL`
  - `WSL -> selected Conda env`
  - `WSL rank0 -> WSL rank1 TCP`
  - `c10d / NCCL`

处理原则：

- 不要把 `SSH OK` 误写成 `训练链路 OK`
- 必须继续验证 WSL distro、Conda、Python、raw TCP、TCPStore、NCCL

## 2. 历史 NetworkState 污染了新的 rendezvous 选择

现象：

- live 测试拿到了错误的 `master_addr`
- 日志看起来像 `torch.distributed` 或 `NCCL` 本身失败

真实原因：

- 旧的网络探测结果被复用
- 选出来的 `master_addr` 不是本轮真实应该使用的地址

处理方法：

- 本轮真实 Worker 地址必须重新从配置和当前 address book 读取
- 不要依赖旧缓存去猜 `master_addr`
- 当前成功配置中：
  - rank0 / master：`10.87.5.155`
  - master_port：`29500`

## 3. 写死网卡名导致 NCCL 走错接口

现象：

- `raw TCP` 可能通
- 但 `NCCL init` 或后续 collective 卡住

真实原因：

- `NCCL_SOCKET_IFNAME` 写死后，不一定对应本轮真实到对端的出接口
- WSL mirrored networking 下，接口名可能变化，不能靠记忆写 `eth0` 或 `eth3`

正确做法：

在每个 Worker 当前 WSL 内执行：

```bash
ip route get <peer_ip>
```

从输出里取 `dev` 字段。

4060 机器对 1650：

```bash
ip route get 10.87.5.15
```

1650 机器对 4060：

```bash
ip route get 10.87.5.155
```

然后设置：

```bash
export NCCL_SOCKET_IFNAME="=<iface>"
export GLOO_SOCKET_IFNAME="<iface>"
export NCCL_SOCKET_FAMILY=AF_INET
export NCCL_IB_DISABLE=1
export NCCL_NET=Socket
```

## 4. 强制修改 `ip_local_port_range` 会把成功条件改坏

现象：

- 之前尝试把端口范围强制改成 `50000 51000`
- 某些 raw TCP/NCCL 测试反而不稳定

结论：

- 这次最终真实跑通时，两台实际成功使用的是：

```text
net.ipv4.ip_local_port_range = 44620 48715
```

处理原则：

- 不要默认强制 `sysctl -w`
- 先记录当前值，再测试
- 只有在明确知道问题就是动态端口范围时，才做变更

## 5. 旧的 Python / distributed 进程残留会造成假故障

现象：

- `29500` 端口占用
- 报错像网络故障，实际是 `EADDRINUSE`

真实原因：

- 上一次失败的 `python -` 还留在 WSL 里
- 新一轮测试没在干净状态下启动

建议检查：

```bash
ps -ef | grep '[p]ython'
ss -ltnp | grep ':29500'
```

必要时先清理：

```bash
pkill -f /home/shardgrid/miniconda3/envs/shardgrid/bin/python || true
```

注意：

- 这里只应清理本轮测试残留
- 不要粗暴杀掉不相关业务进程

## 6. Windows Host Python 和 WSL Conda Python 混淆

现象：

- 看起来“Python 在远端跑了”
- 但不确定是不是 WSL Conda 环境里的 Python

风险：

- 如果实际跑的是 Windows Python、WSL 系统 Python，结果就不可信

正确判断标准：

- 必须记录真实 `python_executable`
- 当前正确值应类似：

```text
/home/shardgrid/miniconda3/envs/shardgrid/bin/python
```

- 同时记录：
  - `CONDA_PREFIX`
  - `CONDA_DEFAULT_ENV`
  - WSL distro

## 7. raw TCP / TCPStore / NCCL 要分层排查，不能混着猜

正确顺序：

1. `Machine A -> Windows Host SSH`
2. `Windows -> WSL Ubuntu`
3. `WSL Conda Python`
4. `raw TCP`
5. `TCPStore`
6. `NCCL init`
7. `broadcast`
8. `send/recv`
9. `barrier`
10. `all_reduce`

不要出现以下误判：

- `NCCL 失败` 其实是 raw TCP 不通
- `GPU 失败` 其实是 SSH 认证失败
- `CUDA 失败` 其实是跑到了错误的 Python

## 8. raw TCP 通了，不代表代码就一定没问题

现象：

- 一度已经能做最小 `all_reduce`
- 但测试框架仍然失败

真实原因：

- 不是网络退化，而是命令包装层被改坏了
- 当时 `run_script()` 里拼进了：

```text
export PATH=...:"$PATH"
```

- 这会破坏 `wsl.exe ... /bin/bash -lc` 这条链路里的 quoting

修复结论：

- 对 stdin-fed Python 脚本，不要再往 payload 里塞 `PATH="$PATH"`
- 直接使用绝对 Conda Python 路径最稳
- 同时只补：
  - `CONDA_PREFIX`
  - `CONDA_DEFAULT_ENV`

## 推荐排查命令

## Windows 侧

查看 `.wslconfig`：

```powershell
Get-Content C:\Users\shardgrid\.wslconfig
```

查看 Windows 防火墙 profile：

```powershell
Get-NetFirewallProfile | Select-Object Name, Enabled, DefaultInboundAction, DefaultOutboundAction
```

查看 WSL Hyper-V Firewall：

```powershell
Get-NetFirewallHyperVVMCreator
Get-NetFirewallHyperVVMSetting -PolicyStore ActiveStore -Name '{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}'
```

重启 WSL：

```powershell
wsl --shutdown
wsl -d Ubuntu
```

## WSL 侧

查看当前端口范围：

```bash
sysctl net.ipv4.ip_local_port_range
```

查看到对端的真实路由接口：

```bash
ip route get <peer_ip>
```

查看 IP / 路由：

```bash
hostname -I
ip -4 addr
ip route
```

查看监听和连接：

```bash
ss -ltnp | grep python
ss -tanp | grep python
```

## 最小 raw TCP 验证

服务端：

```bash
python - <<'PY'
import socket
s = socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("0.0.0.0", 29500))
s.listen(1)
print("LISTENING 29500", flush=True)
c, a = s.accept()
print("ACCEPTED", a, flush=True)
c.sendall(b"OK")
c.close()
s.close()
PY
```

客户端：

```bash
python - <<'PY'
import socket
s = socket.create_connection(("10.87.5.155", 29500), 5)
print(s.recv(2))
s.close()
PY
```

## 最小 NCCL 验证

每台先动态取接口：

```bash
PEER_IP=<peer_ip>
IFACE=$(ip route get "$PEER_IP" | awk '{for(i=1;i<=NF;i++) if($i=="dev"){print $(i+1); exit}}')
echo "$IFACE"
```

然后设置：

```bash
export NCCL_SOCKET_IFNAME="=$IFACE"
export GLOO_SOCKET_IFNAME="$IFACE"
export NCCL_SOCKET_FAMILY=AF_INET
export NCCL_IB_DISABLE=1
export NCCL_NET=Socket
export NCCL_DEBUG=INFO
```

## 本次最终真实通过的判定标准

以下条件同时满足，才算 T049 真通过：

- 两个不同物理 Host 都参加了同一个 process group
- rank0 是 RTX 4060，rank1 是 GTX 1650
- 两边都运行在各自 WSL Conda Python
- `AFTER_INIT` 出现
- `AFTER_BROADCAST` 出现
- `AFTER_SEND_RECV` 出现
- `AFTER_BARRIER` 出现
- `AFTER_ALL_REDUCE` 出现
- `all_reduce_tensor` 结果正确
- 结构化 evidence 写入：

```text
/var/tmp/shardgrid/distributed/collectives-latest.json
```

## 下一个 Agent 不要再犯的错

1. 不要因为 SSH 通了，就跳过 WSL/Conda/runtime 身份验证。
2. 不要把 Windows Host IP、WSL IP、历史缓存 IP 混为一谈。
3. 不要写死 `eth0` / `eth3`。
4. 不要默认强制改 `ip_local_port_range`。
5. 不要忽略旧 `python` 残留进程。
6. 不要把 Windows Python 或 WSL 系统 Python 当成训练 runtime。
7. 不要看到 NCCL 超时就直接判 GPU/CUDA 问题，先分层确认 raw TCP 和接口选择。
8. 不要在 `wsl.exe ... /bin/bash -lc` 的 payload 里随便拼会破坏 quoting 的内容。
