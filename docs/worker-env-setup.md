# GPU Worker 训练环境安装手册（RTX 4060 / GTX 1650）

本文档记录从 Windows + WSL 裸机到完整训练环境（CUDA 11.8 → PyTorch cu118 → Apex → Galvatron v2.4.0）的**全部实测成功命令**。环境重建后按本文档顺序执行即可复现。

适用机型（两台实测通过）：

| Worker | Windows IP | Windows 主机 | GPU | capability | VRAM |
|--------|-----------|--------------|-----|-----------|------|
| gpu4060 | 10.87.5.155 | `LDJ` | RTX 4060 Laptop GPU | 8.9 | 8188 MiB |
| gpu1060 | 10.87.5.15 | `LAPTOP-5G3QUOGM` | GTX 1650 | 7.5 | 4096 MiB |

---

## 0. Windows 用户与 WSL 版本问题（最容易踩的坑）

### Windows SSH 用户

- SSH 登录用户：`shardgrid`（Windows 用户，形如 `ldj\shardgrid` / `laptop-5g3quogm\shardgrid`）。
- SSH 登录后**落在 Windows 层**（默认 shell 是 cmd/PowerShell），`pwd`、`ls` 等 Linux 命令不可用（报"不是内部或外部命令"）。
- **必须显式进入 WSL**，不要依赖默认发行版：

```powershell
whoami                       # 期望: ldj\shardgrid 或 laptop-5g3quogm\shardgrid
wsl -l -v                    # 期望: Ubuntu-22.04 ... VERSION 2
wsl -d Ubuntu-22.04          # 显式进入，禁止 wsl（默认）或 wsl -d Ubuntu
```

WSL 内验证：

```bash
whoami                       # 期望: shardgrid
cat /etc/os-release          # 期望: VERSION_ID="22.04"（22.04.5 LTS）
nvidia-smi                   # 期望: GPU 可见（驱动由 Windows 提供，WSL 内不装驱动）
```

### Ubuntu 版本问题

- **必须 Ubuntu 22.04**（22.04.5 LTS）。CUDA 11.8 官方 WSL repo 与 NCCL 的 ubuntu2204 repo 都绑定 22.04。
- 禁止 `do-release-upgrade`、`apt upgrade`、`apt full-upgrade`。
- **WSL 发行版名必须是 `Ubuntu-22.04`**。写 `Ubuntu` 会报 `WSL_E_DISTRO_NOT_FOUND`。
- WSL2 GPU 驱动由 Windows NVIDIA driver 提供（`nvidia-smi` 显示 CUDA 12.7 只是 Windows 驱动能力，**不要**据此升级 Toolkit，固定 CUDA 11.8）。
- 禁止删除 / 重建 / 导入 WSL 发行版；禁止操作其他 WSL。

### 网络与防火墙

- 正常网络优先；失败时启用 Windows 侧代理（WSL 内可直接访问 `127.0.0.1:7890`）：

```bash
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
```

  不要把代理写入 `~/.bashrc`。apt 走代理用 `-o Acquire::http::Proxy="http://127.0.0.1:7890"`。
- **Windows 防火墙只放行了项目 rendezvous 端口 29500**。双机 NCCL/Gloo 测试必须用 29500；其他端口（29501+）会被拦截导致 NCCL 卡住（`socket.cpp ... err=-3` 在成功路径也会出现，是噪音日志；真实失败原因是端口被拦）。
- 各机器 WSL 内承载 10.87.5.x 的接口**不固定**（实测 4060=eth3，1650=eth0），必须用 `ip route get <对端IP>` 动态探测，禁止写死接口名。

---

## 1. 需要下载什么（清单）

| 序号 | 内容 | 来源（官方） | 落点 |
|------|------|-------------|------|
| 1 | CUDA WSL repo keyring `cuda-keyring_1.1-1_all.deb` | `developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/` | 系统 apt |
| 2 | CUDA Toolkit 11.8（`cuda-toolkit-11-8`，约 2.7 GB） | 同上（apt 安装） | `/usr/local/cuda-11.8` |
| 3 | NCCL repo keyring `cuda-keyring_1.1-1_all.deb` | `developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/` | 系统 apt |
| 4 | NCCL `libnccl2` / `libnccl-dev`（2.31.2-1+cuda12.9） | 同上（apt 安装） | 系统层 |
| 5 | Miniconda `Miniconda3-latest-Linux-x86_64.sh` | `repo.anaconda.com/miniconda/` | `~/miniconda3` |
| 6 | PyTorch 2.7.1+cu118 / torchvision 0.22.1 / torchaudio 2.7.1 | `download.pytorch.org/whl/cu118`（pip） | conda env |
| 7 | NVIDIA Apex（固定 commit `9e3568a6f90fbc1996a06f8f9e99310bdaf2253a`） | `github.com/NVIDIA/apex` | `~/apex-spike` |
| 8 | Galvatron v2.4.0（固定 commit `498bcadeb6ff80cd246bdc4321124da0f4b2d89b`） | `github.com/PKU-DAIR/Hetu-Galvatron` tag `v2.4.0` | `~/galvatron-spike-v2.4.0` |
| 9 | pip 依赖包（见 §6 列表） | PyPI 官方 | conda env |

---

## 2. 基础工具 + GCC/G++ 11（STAGE 1）

```bash
sudo apt update
sudo apt install -y \
  git wget curl vim cmake ninja-build pkg-config \
  build-essential openssh-server gcc-11 g++-11

/usr/bin/gcc-11 --version   # 期望: gcc-11 (Ubuntu 11.4.0-1ubuntu1~22.04.3) 11.4.0
/usr/bin/g++-11 --version
```

**版本问题**：CUDA 11.8 的 nvcc 只支持 GCC ≤ 11，必须装 `gcc-11/g++-11` 并用 `-ccbin /usr/bin/g++-11`；不要用系统默认 gcc-12。

---

## 3. 系统级 CUDA Toolkit 11.8（STAGE 2）

禁止用 Conda/Pip 装 CUDA Toolkit；禁止 Ubuntu 的 `nvidia-cuda-toolkit`；禁止装 Linux 驱动。

```bash
cd ~
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update          # 新增 repo 后允许一次

# 网络慢时走代理
sudo apt-get \
  -o Acquire::http::Proxy="http://127.0.0.1:7890" \
  -o Acquire::https::Proxy="http://127.0.0.1:7890" \
  install -y cuda-toolkit-11-8

/usr/local/cuda-11.8/bin/nvcc --version   # 期望: Cuda compilation tools, release 11.8, V11.8.89
```

### CUDA 环境变量（~/.bashrc，幂等追加）

```bash
grep -q '^export CUDA_HOME=/usr/local/cuda-11.8$' ~/.bashrc || cat >> ~/.bashrc <<'BASH'

export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
export CUDAHOSTCXX=/usr/bin/g++-11
BASH
source ~/.bashrc
```

验证：

```bash
echo "$CUDA_HOME"; which nvcc; nvcc --version   # nvcc=/usr/local/cuda-11.8/bin/nvcc, 11.8
```

### CUDA Headers 验证

```bash
for h in cuda_runtime.h cuda_profiler_api.h cublas_v2.h cusparse.h cusolverDn.h; do
  test -f /usr/local/cuda-11.8/include/$h && echo "PASS $h"
done
```

禁止用 Conda/PyPI 目录里的 CUDA header 代替系统层。

---

## 4. 需要编译什么（清单）

| 项目 | 编译内容 | 关键命令 |
|------|---------|---------|
| 1 | CUDA native smoke（`~/test_cuda.cu`） | `nvcc -ccbin /usr/bin/g++-11 ~/test_cuda.cu -o ~/test_cuda` |
| 2 | NVIDIA Apex CUDA/C++ 扩展（FusedAdam 等） | `APEX_CPP_EXT=1 APEX_CUDA_EXT=1 python -m pip install -v --no-build-isolation .` |
| 3 | Galvatron `galvatron_dp_core` C++ 扩展 | `python -m pip install --no-deps .`（官方 setup.py） |

编译前置环境（必须一致）：

```bash
export CUDA_HOME=/usr/local/cuda-11.8
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
export CUDAHOSTCXX=/usr/bin/g++-11
```

### Native CUDA 编译测试

```bash
cat > ~/test_cuda.cu <<'EOF'
#include <cuda_runtime.h>
#include <stdio.h>
int main() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) { printf("CUDA ERROR: %s\n", cudaGetErrorString(err)); return 1; }
    printf("GPU count: %d\n", count);
    return count == 1 ? 0 : 2;
}
EOF

/usr/local/cuda-11.8/bin/nvcc -ccbin /usr/bin/g++-11 ~/test_cuda.cu -o ~/test_cuda
~/test_cuda   # 必须输出: GPU count: 1
```

禁止 `--allow-unsupported-compiler`；禁止修改 CUDA headers。

---

## 5. NCCL（STAGE 6）

```bash
cd ~
wget -O cuda-keyring-ubuntu2204.deb \
  https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring-ubuntu2204.deb
sudo apt update          # 新增 repo 后允许一次

apt-cache policy libnccl2 libnccl-dev   # 期望候选 2.31.2-1+cuda12.9 / +cuda13.3
sudo apt-get install -y libnccl2 libnccl-dev   # 装 2.31.2-1+cuda12.9（与两台 Worker 一致）

dpkg -l | grep -E 'libnccl2|libnccl-dev'
```

注意：`2.31.2` 只有 `+cuda12.9` / `+cuda13.3` 变体（无 `+cuda11.8`），与 CUDA 11.8 系统共存已验证可用（deb 只依赖 libc6/libstdc++）。不要为了 NCCL 更换 CUDA Toolkit。

---

## 6. Miniconda + shardgrid 环境（STAGE 7-8）

```bash
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p "$HOME/miniconda3"
"$HOME/miniconda3/bin/conda" init bash
source ~/.bashrc
conda --version

# 如果 Conda ToS 阻止创建环境（新版必现）:
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

conda create -n shardgrid python=3.12 -y
conda activate shardgrid
which python    # 必须: $HOME/miniconda3/envs/shardgrid/bin/python
python --version   # 3.12.13
```

非交互脚本中 `conda activate` 需要：`source "$HOME/miniconda3/etc/profile.d/conda.sh"`。

### Python 依赖（STAGE 9-10）

```bash
python -m pip install \
  torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu118

python -m pip install --no-deps setuptools==80.10.2

python -m pip install \
  ninja packaging wheel cython "numpy<2" einops==0.8.2 h5py attrs yacs six \
  sentencepiece pybind11 scipy transformers==4.49.0
```

PyTorch 验证：

```python
import torch
print(torch.__version__)          # 2.7.1+cu118
print(torch.version.cuda)         # 11.8
print(torch.cuda.is_available())  # True
print(torch.cuda.get_device_name(0))        # NVIDIA GeForce RTX 4060 Laptop GPU / GTX 1650
print(torch.cuda.get_device_capability(0))  # (8, 9) / (7, 5)
```

---

## 7. NVIDIA Apex（STAGE 11-12）

固定 commit：`9e3568a6f90fbc1996a06f8f9e99310bdaf2253a`

```bash
cd ~
git clone https://github.com/NVIDIA/apex.git apex-spike
cd ~/apex-spike
git checkout 9e3568a6f90fbc1996a06f8f9e99310bdaf2253a
git rev-parse HEAD    # 必须匹配

# 编译前置确认: which python(Conda shardgrid) / echo $CUDA_HOME / nvcc 11.8 / gcc-11 / torch 2.7.1+cu118
APEX_CPP_EXT=1 APEX_CUDA_EXT=1 python -m pip install -v --no-build-isolation .
```

验证：

```bash
cd ~
python - <<'PY'
import apex
from apex.optimizers import FusedAdam
print(apex.__file__)      # .../site-packages/apex/__init__.py
print("FusedAdam PASS")
PY
```

---

## 8. Galvatron v2.4.0（STAGE 13-14）

固定 tag/commit：`v2.4.0` / `498bcadeb6ff80cd246bdc4321124da0f4b2d89b`。禁止 PyPI `galvatron` 包（第三方 kyegomez 版本）。

```bash
cd ~
git clone --branch v2.4.0 https://github.com/PKU-DAIR/Hetu-Galvatron.git galvatron-spike-v2.4.0
cd ~/galvatron-spike-v2.4.0
git checkout 498bcadeb6ff80cd246bdc4321124da0f4b2d89b
git rev-parse HEAD    # 必须匹配

# v2.4.0 的 galvatron/site_package 是纯内容目录（megatron/nccl-tests），不是 pip 安装目录。
# 用根目录官方 setup.py 安装，--no-deps 防止 resolver 替换 torch 2.7.1+cu118:
python -m pip install --no-deps .
```

验证：

```bash
cd ~
python - <<'PY'
import galvatron
import galvatron.core.profiler
print(galvatron.__file__)
print("Galvatron profiler import PASS")
PY
```

已知限制：Galvatron Hardware Profiler（`profile_hardware.py`）在 WSL2 上受 CUPTI 限制（`CUPTI_ERROR_*` / torch.profiler SIGSEGV，exitcode -11）→ 记为 `BLOCKED_BY_WSL2_CUPTI`，不影响 import / 普通 CUDA 运行。

---

## 9. 最终验证（每台实机）

```bash
python - <<'PY'
import torch
import galvatron
import galvatron.core.profiler
from apex.optimizers import FusedAdam

print("torch:", torch.__version__, "CUDA:", torch.version.cuda)
x = torch.randn(1024, 1024, device="cuda", requires_grad=True)
y = x @ x
y.sum().backward()
opt = torch.optim.SGD([x], lr=0.01); opt.step()
torch.cuda.synchronize()
assert torch.isfinite(x.grad).all().item()
print("CUDA forward/backward/optimizer PASS")
PY
```

双机分布式验证（Machine A，项目 CLI，**端口必须 29500**，config 中 `runtime_distro` 必须写 `Ubuntu-22.04`）：

```bash
shardgrid --config <config> dist-test --backend gloo --workers gpu4060,gpu1060
shardgrid --config <config> dist-test --backend nccl  --workers gpu4060,gpu1060
shardgrid --config <config> dist-test --backend auto --workers gpu4060,gpu1060
```

---

## 10. 常见问题速查

| 现象 | 原因 / 处理 |
|------|------------|
| SSH 后 `pwd` 报"不是内部或外部命令" | 落在 Windows 层，先 `wsl -d Ubuntu-22.04` |
| `WSL_E_DISTRO_NOT_FOUND` | distro 名必须是 `Ubuntu-22.04`，不是 `Ubuntu` |
| `sudo: a password is required` | 本环境 sudo 需要密码，用 `echo <pass> \| sudo -S <cmd>` |
| `wsl -l -v` 输出乱码 / 带 `\x00` | Windows 输出是 UTF-16LE，解析时去掉 `\x00` 或按 UTF-16 解码 |
| apt/pip 下载很慢 | 临时启用 `127.0.0.1:7890` 代理（不写入 bashrc） |
| `nvcc: No such file or directory` | 未加载 CUDA 环境变量（`source ~/.bashrc` 或按 §3 导出） |
| NCCL 双机卡住 / 超时 | 端口必须是 29500（防火墙只放行该端口）；`NCCL_SOCKET_IFNAME` 用 `ip route get` 探测的接口 |
| `socket.cpp ... hostname ... err=-3` | 噪音日志，成功路径也会出现；真正失败看是否端口被拦 / 残留 python 进程 |
| 双机测试残留 python 进程 | 测试前 `pkill -9 -f miniconda3/envs/shardgrid/bin/python`（项目 dist-test 自动处理） |
| `galvatron.core.profiler` 报 `No module named 'apex'/'einops'` | 缺依赖，按 §6/§7 补齐 Apex、einops 等 |
| 编译 Apex 报 CUDA header 缺失 | CUDA 11.8 系统层 headers 必须完整（§3 验证），禁止用 PyPI nvidia wheel 的头文件 |
| Galvatron hardware profiler SIGSEGV (-11) | WSL2 CUPTI 限制，`BLOCKED_BY_WSL2_CUPTI`，不影响训练 runtime |

---

## 11. 已实测验证的版本基线

| 组件 | 版本 |
|------|------|
| Ubuntu | 22.04.5 LTS（VERSION_ID="22.04"） |
| WSL | 2（发行版名 `Ubuntu-22.04`） |
| GCC / G++ | 11.4.0（`/usr/bin/gcc-11`） |
| CUDA Toolkit | 11.8.0（`/usr/local/cuda-11.8`，nvcc V11.8.89） |
| NCCL | 2.31.2-1+cuda12.9（libnccl2 / libnccl-dev） |
| Miniconda | conda 26.5.3（`~/miniconda3`） |
| Python | 3.12.13（conda env `shardgrid`） |
| setuptools | 80.10.2 |
| numpy | 1.26.4 |
| einops | 0.8.2 |
| transformers | 4.49.0 |
| PyTorch / torchvision / torchaudio | 2.7.1+cu118 / 0.22.1+cu118 / 2.7.1+cu118 |
| Apex | commit `9e3568a6f90fbc1996a06f8f9e99310bdaf2253a`（CUDA ext 构建） |
| Galvatron | v2.4.0，commit `498bcadeb6ff80cd246bdc4321124da0f4b2d89b`（hetu-galvatron 1.0.0） |
| 驱动（4060 / 1650） | 566.07 / 527.41（Windows 提供，WSL 内不装） |
| 网络接口（本次实测） | 4060=`eth3`，1650=`eth0`（动态探测，勿写死） |