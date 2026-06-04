# Project #2: CUDA Integration -- 事件清单

## 主要修改文件

### 需要新建的文件
| 文件 | 作用 |
|------|------|
| `xmake/nvidia.lua` | **新建** -- CUDA 编译配置（nvcc 编译器、CUDA 库路径） |
| `src/device/nvidia/nvidia_runtime_api.cu` | **新建** -- CUDA Runtime API 12 个函数的实现 |
| `src/ops/argmax/nvidia/argmax_nvidia.cu` | **新建** -- argmax 的 CUDA kernel |
| `src/ops/embedding/nvidia/embedding_nvidia.cu` | **新建** -- embedding 的 CUDA kernel |
| `src/ops/linear/nvidia/linear_nvidia.cu` | **新建** -- linear 的 CUDA kernel |
| `src/ops/rms_norm/nvidia/rms_norm_nvidia.cu` | **新建** -- rms_norm 的 CUDA kernel |
| `src/ops/rope/nvidia/rope_nvidia.cu` | **新建** -- rope 的 CUDA kernel |
| `src/ops/self_attention/nvidia/self_attention_nvidia.cu` | **新建** -- self_attention 的 CUDA kernel |
| `src/ops/swiglu/nvidia/swiglu_nvidia.cu` | **新建** -- swiglu 的 CUDA kernel |

### 需要修改的已有文件
| 文件 | 修改内容 |
|------|----------|
| [`xmake.lua`](file:///c:/Code/LLAISYS/llaisys/xmake.lua) | **修改** -- 添加 CUDA 编译选项支持，启用 `nv-gpu` 时将 CUDA 源文件直接编译到 `llaisys` 共享库 |
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
| [`src/device/cpu/cpu_runtime_api.cpp`](file:///c:/Code/LLAISYS/llaisys/src/device/cpu/cpu_runtime_api.cpp) | **重要参考** -- CPU 的 Runtime API 实现，CUDA 版本需要实现同样的 12 个函数 |

## 需更改的配置
- **`xmake.lua`**：添加 `has_config("nv-gpu")` 条件分支，启用 CUDA 编译
- **`xmake/nvidia.lua`**：配置 nvcc 编译器、CUDA toolkit 路径、链接 `cudart.lib` / `libcudart.so`

## 建议阅读层次

> 参考 [遍历式阅读顺序](../thinking/遍历式阅读顺序.md)

| 层次 | 内容 | 必读文件 |
|:--:|------|------|
| **第 4 层** | Device | [src/device/runtime_api.hpp](file:///c:/Code/LLAISYS/llaisys/src/device/runtime_api.hpp) -- 理解 `LlaisysRuntimeAPI` 函数表，新建 CUDA Runtime 要填充同样的结构体 |
| | | [src/device/device_resource.hpp](file:///c:/Code/LLAISYS/llaisys/src/device/device_resource.hpp) -- 理解 `DeviceResource` 工厂模式，CUDA 设备需注册自己的工厂 |
| **第 5 层** | CPU Runtime（参考） | [src/device/cpu/cpu_runtime_api.cpp](file:///c:/Code/LLAISYS/llaisys/src/device/cpu/cpu_runtime_api.cpp) -- **精读参考**：12 个函数指针如何从 `malloc`/`memcpy` 映射，CUDA 版同理映射到 `cudaMalloc`/`cudaMemcpy` |
| | | [src/device/cpu/cpu_resource.cpp](file:///c:/Code/LLAISYS/llaisys/src/device/cpu/cpu_resource.cpp) -- 参考 `registerResource()` 注册流程 |
| **第 7 层** | Tensor | [src/tensor/tensor.cpp](file:///c:/Code/LLAISYS/llaisys/src/tensor/tensor.cpp) -- `load()` 里 `memcpy_sync(H2D)` 在 CUDA 上会自动走 `cudaMemcpyHostToDevice` |
| **第 8 层** | **算子（本任务）** | [src/ops/add/op.cpp](file:///c:/Code/LLAISYS/llaisys/src/ops/add/op.cpp) -- 参考设备分支模式，每个算子的 `.cpp` 需添加 CUDA 分支 |
| | | `src/ops/add/cpu/add_cpu.cpp` -- 参考 CPU kernel 逻辑，CUDA kernel 算法相同但并行策略不同 |
| | | `src/ops/*/nvidia/*.cu` -- **新建**：7 个算子的 CUDA kernel |
| **第 11 层** | 构建系统 | [xmake.lua](file:///c:/Code/LLAISYS/llaisys/xmake.lua) -- 添加 `llaisys-device-nvidia` / `llaisys-ops-nvidia` target |
| | | `xmake/nvidia.lua` -- **新建**：nvcc 编译器配置 |

> 本任务核心是：读懂 **第 4-5 层**的 CPU 实现作为参考，在 **第 4 层**新建 CUDA Runtime、**第 8 层**新建 CUDA kernels、**第 11 层**添加 nvcc 编译配置。Tensor 层（第 7 层）无需修改。

---

## 背景知识

### CUDA 编程模型
- **Host**：CPU 端代码，负责分配显存、启动 kernel、同步
- **Device**：GPU 端代码，kernel 函数在 GPU 上并行执行
- **Grid -> Block -> Thread**：三级并行层次，`<<<gridDim, blockDim>>>` 配置

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
| argmax | 每块处理一段数据，块内归约 + 主机最终归约 | 两阶段归约策略，避免原子操作冲突 |
| embedding | 每个 thread 处理一个 token 的嵌入 | 一维 grid |
| linear | 每个 thread 计算一个输出元素 (dot product) | 二维 grid（rows x col_blocks） |
| rms_norm | 每 block 处理一行，shared memory 行内归约 | 一维 grid + block 内归约 |
| rope | 每 block 处理一对 (seq, head)，thread 遍历 head_dim | 二维 grid  |
| self_attention | 每个 block 处理一个 head，batch thread 遍历 kv_len | 二维 grid + GQA 支持 |
| swiglu | 逐元素，每个 thread 计算一个 gate*sigmoid(gate)*up | 一维 grid |

---

## 任务清单

### 阶段一：CUDA 编译环境搭建

#### 任务 2.1：创建 `xmake/nvidia.lua`

- [x] **配置 nvcc 编译器**
  - 设置 nvcc 路径
  - 设置 CUDA 架构为 `compute_89`（适配 Ada Lovelace RTX 40 系列）
  - 添加 CUDA include 目录 `$(env CUDA_PATH)/include`

- [x] **配置 CUDA 库链接**
  - `cudart`：CUDA Runtime
  - CUDA 源文件直接编译到 `llaisys` 共享库中（`add_files` 条件编译方式）

#### 任务 2.2：修改 `xmake.lua`

- [x] **添加 CUDA 选项**
  - 使用 `option("nv-gpu")` 控制是否启用 CUDA 编译
  - 通过 `has_config("nv-gpu")` 条件编译

- [x] **CUDA 源文件编译**
  - 通过 `add_files("src/ops/*/nvidia/*.cu")` 条件将 CUDA 源文件添加到 `llaisys` 共享库
  - 启用 `-DENABLE_NVIDIA_API` 预处理器宏

- [x] **在 `llaisys` 共享库中链接 CUDA 组件**
  - 通过 `xmake/nvidia.lua` 加载 CUDA 工具链
  - 链接 `cudart` 库

---

### 阶段二：CUDA Runtime API 实现

#### 任务 2.3：创建 `src/device/nvidia/nvidia_runtime_api.cu`

- [x] **仔细阅读 `cpu_runtime_api.cpp` 了解每个 API 签名和预期行为**

- [x] **实现设备管理函数**
  - `init(device_id)` -- `cudaSetDevice(device_id)`
  - `destroy()` -- 清理所有资源
  - `reset()` -- `cudaDeviceReset()`
  - `synchronize()` -- `cudaDeviceSynchronize()`

- [x] **实现内存管理函数**
  - `mem_alloc(size)` -- `cudaMalloc`
  - `mem_free(ptr)` -- `cudaFree`
  - `mem_copy(dst, src, size, kind)` -- `cudaMemcpy`，处理 H2D/D2H/D2D
  - `memset(ptr, value, size)` -- `cudaMemset`

- [x] **实现设备属性函数**
  - `device_count()` -- `cudaGetDeviceCount`
  - `props(device_id)` -- `cudaGetDeviceProperties`，返回 GPU 名称/显存/CUDA 核心数等

- [x] **实现流管理函数**
  - `stream_create()` / `stream_destroy()` / `stream_sync()` -- CUDA 流 API

#### 任务 2.4：验证 CUDA Runtime

- [x] **运行 CUDA 运行时测试**
  - GPU 检测正常：`Found 1 nvidia devices`
  - 显存分配/释放正常
  - H2D/D2H 数据传输正常

---

### 阶段三：CUDA 算子实现

#### 任务 2.5：实现 CUDA Kernel

- [x] **每个算子的 kernel 模式**
  - 头文件声明（`*_nvidia.cuh`）
  - Host 封装函数（在 `.cu` 中，负责调用 kernel 和检查错误）

- [x] **按顺序实现各算子的 CUDA 版本**
  - argmax_nvidia -- 两阶段归约：块内并行归约 + 主机端最终归约
  - embedding_nvidia -- 支持 int64 索引，每个 thread 处理一个 token 的嵌入
  - linear_nvidia -- 手写 GEMM kernel，二维 grid
  - rms_norm_nvidia -- 行内 shared memory 归约 + 广播归一化
  - rope_nvidia -- 二维 grid (seq_len x n_heads)，thread 遍历 half_dim
  - self_attention_nvidia -- 支持 GQA，batch thread 遍历 kv_len
  - swiglu_nvidia -- 逐元素 gate*sigmoid(gate)*up

- [x] **每个 kernel 注意**
  - 正确的 grid/block 维度配置
  - 边界检查（线程索引不超出数据范围）
  - 使用 `__syncthreads()` 进行 block 内同步（rms_norm）
  - 错误检查：`cudaGetLastError()` 

#### 任务 2.6：在 op.cpp 中添加 CUDA 分派

- [x] **每个 `op.cpp` 增加 CUDA 设备判断**
  - 检查 context device 类型
  - 如果是 CUDA 设备 -> 调用 `*_nvidia()` 函数（受 `#ifdef ENABLE_NVIDIA_API` 保护）
  - 如果是 CPU 设备 -> 调用原有的 `*_cpu()` 函数

#### 任务 2.7：验证 CUDA 算子

- [x] **运行 CUDA 算子的正确性测试**（[bench_cuda_ops.py](file:///c:/Code/LLAISYS/llaisys/test/bench_cuda_ops.py)）
- [x] **所有 7 个算子全部通过 CPU vs CUDA 输出一致性验证**

**测试结果汇总：**

| 算子 | 测试配置 | 结果 |
|------|----------|------|
| argmax | 4 种尺寸 (4 ~ 16384) | **PASS** |
| embedding | 9 种配置 (vocab 128~32000, 1~16 tokens) | **PASS** |
| linear | 4 种配置 (含无 bias 场景, 最大 512x4096x4096) | **PASS** |
| rms_norm | 4 种尺寸 (4x8 ~ 128x4096) | **PASS** |
| rope | 3 种配置 (seq 4~32, heads 2~8, dim 8~128) | **PASS** |
| self_attention | 3 种配置 (含 GQA nkvh=2) | **PASS** |
| swiglu | 4 种尺寸 (4x8 ~ 512x11008) | **PASS** |

> All CUDA operator tests passed!

#### 调试记录

| 问题 | 原因 | 修复 |
|------|------|------|
| `Found 0 nvidia devices` | 动态链接缺失 `cudart64_12.dll` | 将 DLL 复制到 `python/llaisys/libllaisys/` |
| argmax 返回错误值 (inf) | 原子操作 `atomicCAS` 冲突导致结果异常 | 重构为两阶段归约：块内归约 + `cudaMemcpy` 到主机端最终归约 |
| argmax 段错误 | 主机端最终归约直接解引用 GPU 指针 | 改用 `cudaMemcpy(Device)` 写回结果 |
| linear 结果偏差大 | kernel 中 weight 访问模式错误 `weight[j*n+col]` | 改为 `weight[col*k+j]`（weight 为 (n,k) 行优先存储） |
| rms_norm 结果 inf | `rms` 局部变量非共享，只有 thread 0 计算了正确的值 | 将 rms 存入 shared memory `smem[0]`，`__syncthreads()` 后所有线程读取 |
| embedding "Unsupported data type: int64" | embedding kernel 未处理 int64 类型索引 | switch-case 添加 `LLIASYS_DTYPE_I64` 分支 |
| linear None bias 崩溃 | Python 绑定无法处理 None bias | 修改 `ops.cc` 中 `llaisysLinear` 支持 nullptr bias |

---

### 阶段四：性能基准测试

#### 任务 2.8：性能评估

- [x] **测量 CUDA 算子执行时间**（[bench_cuda_ops.py](file:///c:/Code/LLAISYS/llaisys/test/bench_cuda_ops.py)）

**CUDA 算子性能基准结果：**

| 算子 | 数据规模 | CUDA 耗时 | 备注 |
|------|----------|-----------|------|
| argmax | (16384,) | 0.18 ms | 两阶段归约，含 H2D 传输 |
| linear | 512x4096x4096 | 370.06 ms | 手写 GEMM kernel，未使用 Tensor Core / tiling |
| rms_norm | (128, 4096) | 0.03 ms | shared memory 行内归约 |
| rope | (32, 8, 128) | 0.05 ms | 二维 grid |
| swiglu | (512, 11008) | 0.19 ms | 逐元素计算 |

> **说明**：linear 的手写 GEMM kernel 仅为功能验证，未使用 shared memory tiling 和 Tensor Core，因此性能未达到最优。实际推理场景中应集成 cuBLAS 以获得峰值性能（约 15 TFLOPS 理论值 vs 当前 ~5.4 GFLOPS）。

---

## 总结

项目 2 完成了以下工作：

1. **CUDA Runtime API**：实现 12 个核心函数，覆盖设备管理、内存管理、设备属性查询、流管理
2. **构建系统**：通过 `xmake.lua` 条件编译 CUDA 源文件，直接集成到 `llaisys` 共享库
3. **7 个 CUDA 算子**：argmax、embedding、linear、rms_norm、rope、self_attention、swiglu 的 CUDA kernel 实现
4. **设备分派**：所有算子在 op.cpp 中根据设备类型自动选择 CPU/CUDA 实现
5. **正确性验证**：所有 7 个算子通过 CPU 参照验证，输出一致性在浮点精度范围内
6. **性能基准**：完成 CUDA 算子的执行时间测量