你现在运行在 Ubuntu Machine A。

严格执行 T057 GTX 1650 live verification。

重要：
这是验收测试阶段，不是环境配置阶段。

不要重新安装 CUDA。
不要重新安装 PyTorch。
不要重新修改 gcc/g++。
不要修改项目代码。
不要修改测试逻辑。
不要执行 T058。

如果任何验证失败：
立即停止并报告错误。
不要自动修复。
不要修改环境后重跑。


========================================
当前 GTX 1650 Worker 已完成环境准备
========================================

Worker:
- GPU: NVIDIA GeForce GTX 1650
- WSL2 Ubuntu
- Conda environment: shardgrid

PyTorch:
- torch: 2.7.1+cu118
- torch.version.cuda: 11.8
- torch.cuda.is_available(): True

CUDA compiler:
- nvcc: 11.8.89
- nvcc path: $CONDA_PREFIX/bin/nvcc

Host compiler:
- 使用 Conda gcc/g++ 11.4
- 不要使用系统默认 gcc/g++

当前已验证编译链：

CUDA nvcc 11.8
+
Conda gcc/g++ 11.4
+
CUDA development headers

CUDA headers 已存在，包括：
- cuda_runtime.h
- cuda_profiler_api.h

这套环境应与已经通过 T056 的 RTX 4060 Worker 保持一致。


========================================
Galvatron baseline
========================================

使用已经确定的官方 Galvatron：

source:
https://github.com/PKU-DAIR/Hetu-Galvatron

ref:
v2.4.0

不要：
- 跟随 main
- 使用第三方 PyPI galvatron
- 修改 Galvatron 源码
- 切换其他版本


========================================
T057 验证内容
========================================

1. Environment evidence

记录：

- Worker identity
- physical/runtime identity
- Conda environment
- Conda prefix
- Python executable
- Python version
- PyTorch version
- torch.version.cuda


2. GPU evidence

确认并记录：

- GPU name = NVIDIA GeForce GTX 1650
- compute capability
- VRAM
- CUDA available


3. CUDA build environment

确认：

- nvcc 11.8.89
- CUDA_HOME 指向 selected Conda prefix
- CC/CXX 使用 Conda gcc/g++ 11.4
- cuda_runtime.h 可用
- cuda_profiler_api.h 可用

不要切换到系统 gcc/g++。


4. Apex

检查当前 selected Conda environment 中 Apex 是否已经可用。

验证：

import apex

from apex.optimizers import FusedAdam

如果 Apex 尚未安装，可以使用已经在 RTX 4060 上验证过的相同官方 Apex 安装/编译方案。

必须保持：

- CUDA 11.8
- Conda gcc/g++ 11.4
- 当前 PyTorch 2.7.1+cu118

不要升级或替换 PyTorch/CUDA。

如果 Apex 编译失败：
立即停止。


5. Galvatron

验证：

import galvatron

并确认 Galvatron 可以在当前 GTX 1650 selected runtime 中正常 import。

如果 import 失败：
立即停止。


6. CUDA runtime smoke

执行真实 GTX 1650 CUDA smoke：

- CUDA tensor allocation
- 1024x1024 CUDA matmul
- torch.cuda.synchronize()
- finite validation

必须确认真实 GPU 是 GTX 1650。


========================================
严格失败规则
========================================

如果任何一步失败：

立即停止。

报告：

- status: FAIL 或 BLOCKED
- failure stage
- error message
- Worker
- evidence path

失败后禁止：

- 修改项目代码
- 修改 harness
- 修改测试
- 自动安装新的未知依赖
- 更换 CUDA
- 更换 PyTorch
- 更换 gcc/g++
- 修改 Galvatron 源码
- 降低验证标准
- 跳过失败步骤
- 自动再次运行尝试绕过问题

失败后等待人工决定下一步。


========================================
成功规则
========================================

只有以下全部真实通过：

- environment evidence PASS
- GTX 1650 identity PASS
- CUDA build environment PASS
- Apex PASS
- FusedAdam PASS
- Galvatron import PASS
- CUDA matmul PASS
- finite validation PASS

才允许：

更新：

- docs/compatibility/galvatron.md
- record.md
- tasks.md

并标记：

T057 [X]

GTX 1650 Galvatron compatibility: PASS


========================================
输出要求
========================================

任务结束只简单报告：

T057 GTX 1650：

- Conda/Python：
- PyTorch/CUDA：
- GPU：
- CUDA compiler：
- host compiler：
- Apex：
- FusedAdam：
- Galvatron import：
- CUDA runtime smoke：
- result：PASS / FAIL / BLOCKED
- evidence：

判断：

- T057 是否可以标记 [X]：YES / NO
- GTX 1650 Galvatron compatibility：PASS / FAIL / BLOCKED
- 是否可以进入 T058：YES / NO

完成后停止。

不要执行 T058。
不要创建 commit。