# Project #2: CUDA Integration — 事件清单

## 主要修改文件

### 需要新建的文件
| 文件 | 作用 |
|------|------|
| `xmake/nvidia.lua` | **新建** — CUDA 编译配置（nvcc 编译器、CUDA 库路径） |
| `src/device/nvidia/nvidia_runtime_api.cu` | **新建** — CUDA Runtime API 12 个函数的实现 |
| `src/ops/argmax/nvidia/argmax_nvidia.cu` | **新建** — argmax 的 CUDA kernel |
| `src/ops/embedding/nvidia/embedding_nvidia.cu` | **新建** — embedding 的 CUDA kernel |
| `src/ops/linear/nvidia/linear_nvidia.cu` | **新建** — linear 的 CUDA kernel（可选调用 cuBLAS） |
| `src/ops/rms_norm/nvidia/rms_norm_nvidia.cu` | **新建** — rms_norm 的 CUDA kernel |
| `src/ops/rope/nvidia/rope_nvidia.cu` | **新建** — rope 的 CUDA kernel |
| `src/ops/self_attention/nvidia/self_attention_nvidia.cu` | **新建** — self_attention 的 CUDA kernel |
| `src/ops/swiglu/nvidia/swiglu_nvidia.cu` | **新建** — swiglu 的 CUDA kernel |

### 需要修改的已有文件
| 文件 | 修改内容 |
|------|----------|
| [`xmake.lua`](file:///c:/Code/LLAISYS/llaisys/xmake.lua) | **修改** — 添加 CUDA 编译选项支持和 `llaisys-ops-nvidia` 目标 |
| `xmake/cpu.lua` | 可能需**查看**（理解 Lua 宏和函数） |
| `src/ops/argmax/op.cpp` | **添加** CUDA 设备分支 |
| `src/ops/embedding/op.cpp` | **添加** CUDA 设备分支 |
| `src/ops/linear/op.cpp` | **添加** CUDA 设备分支 |
| `src/ops/rms_norm/op.cpp` | **添加** CUDA 设备分支 |
| `src/ops/rope/op.cpp` | **添加** CUDA 设备分支 |
| `src/ops/self_attention/op.cpp` | **添加** CUDA 设备分支 |
| `src/ops/swiglu/op.cpp` | **添加** CUDA 设备分支 |
| [`xmake.lua`](file:///c:/Code/LLAISYS/llaisys/xmake.lua) 中的 `llaisys` 共享库目标 | 添加 nvidia runtime 和 ops 依赖 |

### 需要参考的已有文件
| 文件 | 作用 |
|------|------|
| [`src/device/cpu/cpu_runtime_api.cpp`](file:///c:/Code/LLAISYS/llaisys/src/device/cpu/cpu_runtime_api.cpp) | **重要参考** — CPU 的 Runtime API 实现，CUDA 版本需要实现同样的 12 个函数 |

## 需更改的配置
- **`xmake.lua`**：添加 `has_config("nv-gpu")` 条件分支，启用 CUDA 编译
- **`xmake/nvidia.lua`**：配置 nvcc 编译器、CUDA toolkit 路径、链接 `cudart.lib` / `libcudart.so`

## 建议阅读层次

> 参考 [遍历式阅读顺序](../thinking/遍历式阅读顺序.md)

| 层次 | 内容 | 必读文件 |
|:--:|------|------|
| **第 4 层** | Device | [src/device/runtime_api.hpp](file:///c:/Code/LLAISYS/llaisys/src/device/runtime_api.hpp) — 理解 `LlaisysRuntimeAPI` 函数表，新建 CUDA Runtime 要填充同样的结构体 |
| | | [src/device/device_resource.hpp](file:///c:/Code/LLAISYS/llaisys/src/device/device_resource.hpp) — 理解 `DeviceResource` 工厂模式，CUDA 设备需注册自己的工厂 |
| **第 5 层** | CPU Runtime（参考） | [src/device/cpu/cpu_runtime_api.cpp](file:///c:/Code/LLAISYS/llaisys/src/device/cpu/cpu_runtime_api.cpp) — **精读参考**：12 个函数指针如何从 `malloc`/`memcpy` 映射，CUDA 版同理映射到 `cudaMalloc`/`cudaMemcpy` |
| | | [src/device/cpu/cpu_resource.cpp](file:///c:/Code/LLAISYS/llaisys/src/device/cpu/cpu_resource.cpp) — 参考 `registerResource()` 注册流程 |
| **第 7 层** | Tensor | [src/tensor/tensor.cpp](file:///c:/Code/LLAISYS/llaisys/src/tensor/tensor.cpp) — `load()` 里 `memcpy_sync(H2D)` 在 CUDA 上会自动走 `cudaMemcpyHostToDevice` |
| **第 8 层** | **算子（本任务）** | [src/ops/add/op.cpp](file:///c:/Code/LLAISYS/llaisys/src/ops/add/op.cpp) — 参考设备分支模式，每个算子的 `.cpp` 需添加 CUDA 分支 |
| | | `src/ops/add/cpu/add_cpu.cpp` — 参考 CPU kernel 逻辑，CUDA kernel 算法相同但并行策略不同 |
| | | `src/ops/*/nvidia/*.cu` — **新建**：7 个算子的 CUDA kernel |
| **第 11 层** | 构建系统 | [xmake.lua](file:///c:/Code/LLAISYS/llaisys/xmake.lua) — 添加 `llaisys-device-nvidia` / `llaisys-ops-nvidia` target |
| | | `xmake/nvidia.lua` — **新建**：nvcc 编译器配置 |

> 本任务核心是：读懂 **第 4-5 层**的 CPU 实现作为参考，在 **第 4 层**新建 CUDA Runtime、**第 8 层**新建 CUDA kernels、**第 11 层**添加 nvcc 编译配置。Tensor 层（第 7 层）无需修改。

---

## 背景知识

### CUDA 编程模型
- **Host**：CPU 端代码，负责分配显存、启动 kernel、同步
- **Device**：GPU 端代码，kernel 函数在 GPU 上并行执行
- **Grid → Block → Thread**：三级并行层次，`<<<gridDim, blockDim>>>` 配置

### CUDA Runtime API（12 个函数）
| 分类 | 函数 | 说明 |
|------|------|------|
| 设备管理 | `init` / `destroy` / `reset` / `synchronize` | 初始化 GPU、全局同步 |
| 内存管理 | `mem_alloc` / `mem_free` / `mem_copy` / `memset` | GPU 显存操作 |
| 设备属性 | `props` / `device_count` | 查询 GPU 属性（名称、显存、SM 数等） |
| 流管理 | `stream_create` / `stream_destroy` / `stream_sync` | CUDA 流用于异步并行 |

### 算子 CUDA Kernel 设计要点
| 算子 | 并行策略 | 说明 |
|------|----------|------|
| argmax | 每块处理一段数据，块内 reduce | 二维 grid |
| embedding | 每个 thread 处理一个 token 的嵌入 | 一维 grid |
| linear | 每个 thread 计算一个输出元素 | 二维 grid，或直接用 cuBLAS |
| rms_norm | 每 block 处理一行 | 一维 grid |
| rope | 每个 thread 处理一对元素 | 一维 grid |
| self_attention | 每个 block 处理一个 head 的一对 (q,k) | 多维 grid |
| swiglu | 逐元素，每个 thread 计算一个 | 一维 grid |

---

## 任务清单

### 阶段一：CUDA 编译环境搭建

#### 任务 2.1：创建 `xmake/nvidia.lua`

- [ ] **配置 nvcc 编译器**
  - 设置 nvcc 路径
  - 设置 CUDA 架构（如 `-arch=sm_75` 对应 Turing）
  - 添加 CUDA include 目录

- [ ] **配置 CUDA 库链接**
  - `cudart`：CUDA Runtime
  - `cublas`：CUDA BLAS（可选，用于 linear 加速）

#### 任务 2.2：修改 `xmake.lua`

- [ ] **添加 CUDA 选项**
  - 参照 `cpu.lua` 的方式
  - 使用 `option("nv-gpu")` 控制是否启用 CUDA 编译

- [ ] **添加 `llaisys-ops-nvidia` 编译目标**
  - 包含 `src/ops/*/nvidia/*.cu`
  - 依赖 CUDA Runtime

- [ ] **在 `llaisys` 共享库中链接 CUDA 组件**
  - 根据 `has_config("nv-gpu")` 条件链接

---

### 阶段二：CUDA Runtime API 实现

#### 任务 2.3：创建 `src/device/nvidia/nvidia_runtime_api.cu`

- [ ] **仔细阅读 `cpu_runtime_api.cpp` 了解每个 API 签名和预期行为**
  - CPU 版本已打印每个 API 调用，能快速了解调用序列

- [ ] **实现设备管理函数**
  - `init(device_id)` — `cudaSetDevice` + 初始化 CUDA 资源
  - `destroy()` — 清理所有资源
  - `reset()` — `cudaDeviceReset`
  - `synchronize()` — `cudaDeviceSynchronize`

- [ ] **实现内存管理函数**
  - `mem_alloc(size)` — `cudaMalloc`
  - `mem_free(ptr)` — `cudaFree`
  - `mem_copy(dst, src, size, kind)` — `cudaMemcpy`，处理 H2D/D2H/D2D
  - `memset(ptr, value, size)` — `cudaMemset`

- [ ] **实现设备属性函数**
  - `device_count()` — `cudaGetDeviceCount`
  - `props(device_id)` — `cudaGetDeviceProperties`，返回 GPU 名称/显存/CUDA 核心数等

- [ ] **实现流管理函数（可选）**
  - `stream_create()` / `stream_destroy()` / `stream_sync()` — CUDA 流 API

#### 任务 2.4：验证 CUDA Runtime

- [ ] **运行 CUDA 运行时测试**
  - 编译带 CUDA 支持的版本
  - 确认能正确检测 GPU、分配/释放显存、数据传输

---

### 阶段三：CUDA 算子实现

#### 任务 2.5：实现 CUDA Kernel

- [ ] **每个算子的 kernel 模式**
  - 头文件声明（`*_nvidia.cuh` 或 `.cu` 中的 `__global__` 函数）
  - Host 封装函数（在 `.cu` 中，负责调用 kernel 和检查错误）

- [ ] **按顺序实现各算子的 CUDA 版本**
  - argmax_nvidia
  - embedding_nvidia
  - linear_nvidia（可直接用 cuBLAS `cublasSgemm` 代替手写 kernel）
  - rms_norm_nvidia
  - rope_nvidia
  - self_attention_nvidia
  - swiglu_nvidia

- [ ] **每个 kernel 注意**
  - 正确的 grid/block 维度配置
  - 边界检查（线程索引不超出数据范围）
  - 使用 `__syncthreads()` 进行 block 内同步
  - 错误检查：`cudaGetLastError()` + `cudaDeviceSynchronize()`

#### 任务 2.6：在 op.cpp 中添加 CUDA 分派

- [ ] **每个 `op.cpp` 增加 CUDA 设备判断**
  - 检查 context device 类型
  - 如果是 CUDA 设备 → 调用 `*_nvidia()` 函数
  - 如果是 CPU 设备 → 调用原有的 `*_cpu()` 函数

#### 任务 2.7：验证 CUDA 算子

- [ ] **运行 CUDA 算子的测试**
- [ ] **对比 CPU 和 CUDA 的输出一致性**
  - 浮点误差在可接受范围内（受 CUDA 的 FMA 和计算顺序影响）

---

### 阶段四：CUDA 模型推理

#### 任务 2.8：在 GPU 上运行模型推理

- [ ] **将模型权重加载到 GPU 显存**
- [ ] **运行推理测试**
  - ```bash
    python test_infer.py --model <模型路径> --device cuda
    ```

#### 任务 2.9：性能评估

- [ ] **测量 GPU 推理吞吐量并与 CPU 对比**
  - 记录 tokens/second
  - 分析 GPU 利用率（nsight systems / nvidia-smi）