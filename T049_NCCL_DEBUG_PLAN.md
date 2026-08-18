# T049 NCCL 排查计划

## 目标

对当前 `T049` 的 `init PASS -> broadcast HANG` 做代码级排查。

这份文档只处理当前最可疑的代码问题，不再重复已经验证过的基础环境问题。

真实节点：

- rank0: `10.87.5.155` / RTX 4060 / Windows + WSL2 Ubuntu
- rank1: `10.87.5.15` / GTX 1650 / Windows + WSL2 Ubuntu
- WSL user: `shardgrid`
- Conda env: `shardgrid`
- backend: `nccl`
- rendezvous: `10.87.5.155:29500`

当前已知真实状态：

- SSH: PASS
- WSL2 Ubuntu: PASS
- CUDA visible: PASS
- raw TCP: PASS（在正确网络/防火墙状态下）
- TCPStore/rendezvous: PASS
- NCCL `init_process_group()` 当前代码路径可以返回
- 第一个 CUDA collective `broadcast` 仍可能 HANG
- WSL 接口名会随 `wsl --shutdown` / 重启变化，不能写死 `eth0/eth2/eth3`
- WSL mirrored networking 的 `ip_local_port_range` 可能随启动变化；不要在本轮排查里继续强制修改端口范围

---

# P0-1：`NCCL_SOCKET_IFNAME` 使用 `setdefault()`，可能没有覆盖 `.bashrc` 中的旧值

## 可疑代码

文件：`src/shardgrid/distributed/collectives.py`（上传版本 `collectives.py`）

当前：

```python
os.environ.setdefault("NCCL_DEBUG", "INFO")
os.environ.setdefault("NCCL_DEBUG_SUBSYS", "INIT,BOOTSTRAP,NET")
os.environ.setdefault("NCCL_SOCKET_IFNAME", "=__INTERFACE__")
os.environ.setdefault("GLOO_SOCKET_IFNAME", "__INTERFACE__")
os.environ.setdefault("NCCL_SOCKET_FAMILY", "AF_INET")
os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("NCCL_NET", "Socket")
```

同时 `WSLRuntimeWrapper.run_script()` 在启动 Python 前执行：

```bash
source ~/.bashrc >/dev/null 2>&1 || true
```

所以如果某台 Worker 的 `~/.bashrc` 曾经残留：

```bash
export NCCL_SOCKET_IFNAME='=eth2'
```

而本轮动态探测得到 `eth3`，那么：

```python
os.environ.setdefault("NCCL_SOCKET_IFNAME", "=eth3")
```

**不会覆盖旧的 `=eth2`。**

## Agent 要做

### A. 先查两台 Worker 是否有旧环境变量

两台 WSL 都执行：

```bash
grep -nE 'NCCL_|GLOO_|MASTER_|WORLD_SIZE|RANK' ~/.bashrc ~/.profile ~/.bash_profile 2>/dev/null || true
```

以及：

```bash
env | grep -E '^(NCCL|GLOO|MASTER|WORLD_SIZE|RANK|LOCAL_RANK)=' | sort
```

把结果写入本轮 evidence。

### B. 修改代码

运行时确定性的 NCCL/Gloo 配置不要用 `setdefault()`，改成强制赋值：

```python
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_DEBUG_SUBSYS"] = "INIT,BOOTSTRAP,NET,COLL"
os.environ["NCCL_SOCKET_IFNAME"] = "=__INTERFACE__"
os.environ["GLOO_SOCKET_IFNAME"] = "__INTERFACE__"
os.environ["NCCL_SOCKET_FAMILY"] = "AF_INET"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_NET"] = "Socket"
```

### C. 在初始化前打印“实际生效值”

必须打印 Python 进程最终看到的值，而不是调用方以为传进去的值：

```python
print(
    "NCCL_EFFECTIVE_ENV",
    json.dumps(
        {
            "rank": rank,
            "interface_arg": interface,
            "NCCL_SOCKET_IFNAME": os.environ.get("NCCL_SOCKET_IFNAME"),
            "GLOO_SOCKET_IFNAME": os.environ.get("GLOO_SOCKET_IFNAME"),
            "NCCL_SOCKET_FAMILY": os.environ.get("NCCL_SOCKET_FAMILY"),
            "NCCL_IB_DISABLE": os.environ.get("NCCL_IB_DISABLE"),
            "NCCL_NET": os.environ.get("NCCL_NET"),
        },
        sort_keys=True,
    ),
    flush=True,
)
```

## 通过标准

155 当前实际 peer route 假设为 `eth3` 时：

```text
NCCL_SOCKET_IFNAME = =eth3
GLOO_SOCKET_IFNAME = eth3
```

15 当前实际 peer route 假设为 `eth0` 时：

```text
NCCL_SOCKET_IFNAME = =eth0
GLOO_SOCKET_IFNAME = eth0
```

**任何一台实际打印值和 `ip route get <peer>` 不一致，都先停止 NCCL 测试。**

---

# P0-2：当前 `init_ok=True` 可能只是 rendezvous 完成，真正 NCCL communicator 延迟到第一个 broadcast

## 可疑代码

当前：

```python
torch.cuda.set_device(local_rank)

dist.init_process_group(
    backend=backend,
    init_method="tcp://%s:%d" % (master, port),
    rank=rank,
    world_size=world_size,
)

out["init_ok"] = True
```

随后第一个 CUDA collective：

```python
dist.broadcast(broadcast_tensor, src=0)
```

当前现象正是：

```text
before init
after init
broadcast HANG
```

这可能意味着真正的 NCCL communicator / peer channels 在第一个 collective 才建立。

## Agent 要做

显式创建 CUDA device，并把 `device_id` 传给 `init_process_group()`：

```python
device = torch.device(f"cuda:{local_rank}")
torch.cuda.set_device(device)

dist.init_process_group(
    backend=backend,
    init_method="tcp://%s:%d" % (master, port),
    rank=rank,
    world_size=world_size,
    device_id=device,
)
```

同时增加阶段日志：

```python
print("STAGE before_init", flush=True)

dist.init_process_group(..., device_id=device)

print("STAGE after_real_nccl_init", flush=True)
```

## 通过/失败解释

如果修改后卡在：

```text
STAGE before_init
```

而看不到：

```text
STAGE after_real_nccl_init
```

那么之前所谓 `NCCL init PASS` 实际只是延迟初始化造成的假象；真正问题仍在 NCCL communicator/bootstrap/peer socket 建立。

如果：

```text
STAGE after_real_nccl_init
```

两边都出现，但仍卡在 broadcast，则才可以明确分类为：

```text
communicator init PASS
first collective data path HANG
```

---

# P0-3：每个 rank 的接口必须运行时动态发现，不能依赖历史 NetworkState 或写死值

## 已知事实

WSL 重启后已经观察到接口变化：

```text
同一个 10.87.5.155 曾经落在 eth2，也曾经落在 eth3
```

因此：

```text
interface = eth3
```

不能作为持久配置。

## 当前 live T049 中较好的部分

`tests/multi_host/test_collectives.py` 的 live test 已经调用：

```python
rank0_interface = _discover_interface(w0, str(entry1["ip"]))
rank1_interface = _discover_interface(w1, str(entry0["ip"]))
```

`_discover_interface()` 使用：

```bash
ip route get <peer_ip>
```

这是正确方向。

## 仍然需要 Agent 检查的地方

### A. 搜索整个代码库是否还写死 `eth0/eth2/eth3`

```bash
rg -n 'eth[0-9]+|NCCL_SOCKET_IFNAME|GLOO_SOCKET_IFNAME' src tests examples
```

已知上传代码里：

`tests/unit/test_gloo_fallback.py` 的 live fallback test 存在：

```python
master, rank0_interface, rank1_interface = "10.87.5.155", "eth3", "eth0"
```

这个 live 路径必须改成动态探测。

### B. 检查生产路径是否仍使用旧 `NetworkState.selected_interfaces`

`backend.py` 中：

```python
configured = state.selected_interfaces.get(source_worker_id)
if configured:
    return configured
```

这里如果保存的是上一次 WSL 启动得到的 `ethX`，会重新引入 stale-interface 问题。

Agent 必须确认：

- live T049 是否绕开 stale `NetworkState`
- production distributed launch 是否仍可能使用 stale `selected_interfaces`
- 对 WSL Worker，接口是否应该每次 launch 前重新 `ip route get peer_ip`

## 建议规则

对于 WSL live runtime：

```text
NetworkState 中的 ethX 只能作为 evidence，不应作为跨 WSL 重启的持久真值。
```

每次 launch 前都重新探测。

---

# P1-1：Evidence 只保存了一个 `interface`，但两个 rank 实际接口可能不同

## 可疑代码

`save_collectives_evidence()` 当前顶层只保存：

```python
"interface": interface,
```

而 live test 调用时传入的是：

```python
interface=selection.interface
```

`selection` 又是根据 rank0 interface 构造的。

但真实情况是：

```text
rank0 = eth3
rank1 = eth0
```

所以 evidence 顶层 `interface=eth3` 容易让排查人员误以为两个 rank 都在使用 `eth3`。

## Agent 要做

每个 rank 的 result/evidence 必须记录：

```json
{
  "rank": 0,
  "peer_ip": "10.87.5.15",
  "route_interface": "eth3",
  "effective_nccl_socket_ifname": "=eth3"
}
```

以及：

```json
{
  "rank": 1,
  "peer_ip": "10.87.5.155",
  "route_interface": "eth0",
  "effective_nccl_socket_ifname": "=eth0"
}
```

不要再用一个顶层 `interface` 表示两台 Worker。

---

# P1-2：`run_script()` 会 `source ~/.bashrc`，必须确认这是不是 T049 所需要的行为

## 当前代码

`runtime.py`：

```python
payload = (
    "source ~/.bashrc >/dev/null 2>&1 || true; "
    f"{python_executable} -"
)
```

这会把用户交互 shell 的历史配置全部注入测试进程，例如：

- 旧 NCCL interface
- 旧 MASTER_ADDR / MASTER_PORT
- CUDA 相关变量
- proxy/network 变量
- Conda 自动初始化副作用

当前脚本已经使用绝对 Conda Python：

```text
/home/shardgrid/miniconda3/envs/shardgrid/bin/python
```

因此 T049 是否真的需要 `source ~/.bashrc` 必须重新评估。

## Agent 排查方式

先不要马上删除。

做两个最小版本：

### A. 当前行为

```bash
source ~/.bashrc; <conda-python> -
```

### B. clean environment 行为

不 source `.bashrc`，只执行：

```bash
<conda-python> -
```

并由脚本显式设置所有 NCCL/Gloo 必需环境。

如果 B 稳定、A 不稳定，则 `.bashrc` 污染成立。

## 建议长期设计

训练 runtime 应尽量是 deterministic environment：

```text
明确 Conda Python
+ 明确 env mapping
+ 不依赖交互式 ~/.bashrc
```

---

# P1-3：为每一个 collective 增加前后阶段标记，避免 timeout 时不知道卡在哪一行

当前 `_COLLECTIVES_SCRIPT` 只在最终成功后写 `COLLECTIVE_RESULT`。

如果远端被 timeout kill，就可能完全没有 structured result。

## Agent 要增加

```python
print("STAGE before_init", flush=True)
...
print("STAGE after_init", flush=True)

print("STAGE before_broadcast", flush=True)
dist.broadcast(...)
torch.cuda.synchronize()
print("STAGE after_broadcast", flush=True)

print("STAGE before_send_recv", flush=True)
...
print("STAGE after_send_recv", flush=True)

print("STAGE before_barrier", flush=True)
...
print("STAGE after_barrier", flush=True)

print("STAGE before_all_reduce", flush=True)
...
print("STAGE after_all_reduce", flush=True)
```

并在 evidence 中保留足够 stdout，不要只靠最后一个 JSON。

---

# P1-4：NCCL barrier 显式绑定 device

当前：

```python
dist.barrier()
```

建议 NCCL 路径改成：

```python
dist.barrier(device_ids=[local_rank])
```

这不是当前 `broadcast` HANG 的第一嫌疑，因为代码尚未执行到 barrier，但应该顺手消除后续 device mapping 歧义。

---

# P2-1：确认 `broadcast` 两边 tensor 契约完全一致

当前代码看起来是对称的：

rank0：

```python
expected_broadcast.clone()
```

rank1：

```python
torch.zeros(4, dtype=torch.float32, device="cuda")
```

两边：

```python
dist.broadcast(..., src=0)
```

当前没有看到明显 rank-order bug。

但为了排除 runtime 差异，Agent 应在 broadcast 前打印：

```python
print(
    "BROADCAST_META",
    rank,
    broadcast_tensor.shape,
    broadcast_tensor.dtype,
    broadcast_tensor.device,
    torch.cuda.current_device(),
    flush=True,
)
```

两边必须一致：

```text
shape=torch.Size([4])
dtype=torch.float32
device=cuda:0
current_device=0
```

---

# P2-2：检查两台 PyTorch / CUDA / NCCL build 是否一致

两台 Worker 在同一轮执行：

```bash
python - <<'PY'
import torch
import torch.distributed as dist
print('torch=', torch.__version__)
print('torch_cuda=', torch.version.cuda)
print('cuda_available=', torch.cuda.is_available())
print('nccl_available=', dist.is_nccl_available())
print('nccl_version=', torch.cuda.nccl.version() if torch.cuda.is_available() else None)
print('gpu=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
```

把输出写入 evidence。

版本不一致不一定必然失败，但当前是异构 GPU + WSL，多一个变量就多一个排查维度，因此先记录真实值。

---

# P2-3：不要继续把 `ip_local_port_range` 强制成某个固定值作为本轮代码修复

本轮目标是排代码，不再同时修改网络模型。

Agent 应：

```bash
sysctl net.ipv4.ip_local_port_range
```

只记录，不修改。

尤其不要在 `WSLRuntimeWrapper.run()` / `run_script()` 里再次加入自动 root `sysctl -w`。

---

# 建议的最小代码修改顺序

## 第 1 轮：只增加观测，不改 collective 语义

1. `setdefault()` 改成强制 `os.environ[...]`
2. 打印 effective NCCL/Gloo env
3. 打印当前 `ip route get peer`
4. 打印 current CUDA device
5. 在每个 collective 前后打印 STAGE
6. 每个 rank evidence 分别保存 interface

运行最小 NCCL。

如果仍是：

```text
STAGE after_init
STAGE before_broadcast
HANG
```

进入第 2 轮。

## 第 2 轮：强制 NCCL communicator 在 init 阶段建立

修改：

```python
device = torch.device(f"cuda:{local_rank}")
torch.cuda.set_device(device)

dist.init_process_group(
    backend=backend,
    init_method=f"tcp://{master}:{port}",
    rank=rank,
    world_size=world_size,
    device_id=device,
)
```

再次运行。

### 分类 A

如果卡在 init：

```text
before_init
HANG
```

结论：

```text
真正的 NCCL communicator/bootstrap/peer connection 仍有问题。
之前 after_init 是 lazy-init 假象。
```

### 分类 B

如果 init 真正 PASS，但 broadcast HANG：

```text
after_real_nccl_init
before_broadcast
HANG
```

结论：

```text
communicator 建立成功，问题进入 collective data path。
```

这时再抓 NCCL socket / tcpdump，而不是继续改 init。

---

# 建议的下一轮运行日志

两台都打开：

```text
NCCL_DEBUG=INFO
NCCL_DEBUG_SUBSYS=INIT,BOOTSTRAP,NET,COLL
NCCL_DEBUG_FILE=/tmp/nccl.%h.%p.log
TORCH_DISTRIBUTED_DEBUG=DETAIL
```

失败后采集：

```bash
cat /tmp/nccl.*
ss -ltnp | grep python || true
ss -tanp | grep python || true
ip -4 addr
ip route
sysctl net.ipv4.ip_local_port_range
```

155：

```bash
ip route get 10.87.5.15
```

15：

```bash
ip route get 10.87.5.155
```

---

# Agent 最终需要返回的结果格式

```text
CODE CHECK
- collectives.py setdefault removed: YES/NO
- run_script sources bashrc: YES/NO
- hard-coded ethX remaining: <locations>
- stale NetworkState interface possible: YES/NO + explanation
- device_id passed to init_process_group: YES/NO

RANK0
- peer_ip:
- ip route get:
- interface_arg:
- effective NCCL_SOCKET_IFNAME:
- effective GLOO_SOCKET_IFNAME:
- port_range:
- torch:
- torch CUDA:
- NCCL version:
- last STAGE reached:
- relevant NCCL log tail:
- live sockets:

RANK1
- peer_ip:
- ip route get:
- interface_arg:
- effective NCCL_SOCKET_IFNAME:
- effective GLOO_SOCKET_IFNAME:
- port_range:
- torch:
- torch CUDA:
- NCCL version:
- last STAGE reached:
- relevant NCCL log tail:
- live sockets:

CLASSIFICATION
- rendezvous issue: YES/NO
- stale interface issue: YES/NO
- bashrc env pollution: YES/NO
- lazy NCCL init confirmed: YES/NO
- communicator init PASS: YES/NO
- broadcast data path PASS: YES/NO
- next smallest action:
```

---

# 当前优先级总结

按排查价值排序：

1. **P0：`setdefault()` + `source ~/.bashrc` 导致动态接口可能没有真正生效**
2. **P0：`init_process_group()` 未传 `device_id`，当前 `init PASS` 可能只是 lazy init**
3. **P0：任何 live/production 路径仍写死或缓存 `ethX`**
4. **P1：evidence 只记录 rank0 的单一 interface，掩盖两端接口不对称**
5. **P1：阶段日志不足，timeout 后无法知道准确阻塞行**
6. **P1：barrier 未显式指定 device_ids**
7. **P2：版本/tensor metadata 等一致性检查**

本轮不要同时重新设计 firewall、端口范围或 WSL 网络模式。先把上述代码路径变成可确定、可观测，再根据新 evidence 决定网络侧是否还有剩余问题。
