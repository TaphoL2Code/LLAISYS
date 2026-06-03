# Project #5: Distributed Inference — 事件清单

需要多设备，硬件条件暂不足，跳过

## 主要修改文件

### C++ 后端
| 文件 | 修改内容 |
|------|----------|
| [`src/models/qwen2/qwen2.hpp`](file:///c:/Code/LLAISYS/llaisys/src/models/qwen2/qwen2.hpp) | **修改** — 支持张量并行（Tensor Parallelism），权重分片 |
| [`src/models/qwen2/qwen2.cpp`](file:///c:/Code/LLAISYS/llaisys/src/models/qwen2/qwen2.cpp) | **修改** — 分布式前向传播逻辑、跨设备通信 |
| [`include/llaisys/models/qwen2.h`](file:///c:/Code/LLAISYS/llaisys/include/llaisys/models/qwen2.h) | **修改** — 扩展 API（多设备模型创建） |
| `src/device/nvidia/nvidia_runtime_api.cu` | **修改** — 添加 NCCL 通信 API |
| `src/device/cpu/cpu_runtime_api.cpp` | **修改** — 添加 MPI 通信 API |
| `src/ops/linear/nvidia/linear_nvidia.cu` | **修改** — 支持分片后的列并行/行并行 linear |
| `src/ops/linear/cpu/linear_cpu.cpp` | **修改** — 同上（CPU 版本） |
| `src/ops/self_attention/nvidia/self_attention_nvidia.cu` | **修改** — 支持分头并行 attention |

### 可能需要新建的文件
| 文件 | 作用 |
|------|------|
| `src/device/communicator.hpp` | **新建** — 设备间通信抽象接口 |
| `src/device/nvidia/nccl_communicator.cu` | **新建** — NCCL 通信实现 |
| `src/device/cpu/mpi_communicator.cpp` | **新建** — MPI 通信实现 |

### Python 前端
| 文件 | 修改内容 |
|------|----------|
| [`python/llaisys/models/qwen2.py`](file:///c:/Code/LLAISYS/llaisys/python/llaisys/models/qwen2.py) | **修改** — 支持多设备模型创建 |

### 编译配置
| 文件 | 修改内容 |
|------|----------|
| [`xmake.lua`](file:///c:/Code/LLAISYS/llaisys/xmake.lua) | 添加 NCCL/MPI 编译选项和链接 |
| `xmake/nvidia.lua` | 添加 NCCL 依赖 |

## 需更改的配置
- **NCCL**（NVIDIA GPU）：链接 `libnccl.so` / `nccl.lib`，添加 `ENABLE_NCCL` 宏
- **MPI**（CPU 集群）：链接 `libmpi.so`，添加 `ENABLE_MPI` 宏
- **xmake**：添加 `--nv-gpu=y --nccl=y` 等选项

## 建议阅读层次

> 参考 [遍历式阅读顺序](../thinking/遍历式阅读顺序.md)

| 层次 | 内容 | 必读文件 |
|:--:|------|------|
| **第 4 层** | Device（通信抽象） | [src/device/device_resource.hpp](file:///c:/Code/LLAISYS/llaisys/src/device/device_resource.hpp) — 理解工厂模式，新增 `Communicator` 抽象接口 |
| | | `src/device/communicator.hpp` — **新建**：跨设备通信抽象（AllReduce/AllGather/ReduceScatter） |
| **第 5 层** | 通信实现 | `src/device/nvidia/nccl_communicator.cu` — **新建**：NCCL 实现（`ncclAllReduce` 等） |
| | | `src/device/cpu/mpi_communicator.cpp` — **新建**：MPI 实现（`MPI_Allreduce` 等） |
| | | [src/device/cpu/cpu_runtime_api.cpp](file:///c:/Code/LLAISYS/llaisys/src/device/cpu/cpu_runtime_api.cpp) — 参考：添加 MPI 通信函数到 API 表 |
| **第 7 层** | Tensor | [src/tensor/tensor.cpp](file:///c:/Code/LLAISYS/llaisys/src/tensor/tensor.cpp) — 分片后 Tensor 的 shape 和 strides 变化 |
| **第 8 层** | **算子（本任务）** | [src/ops/linear/op.cpp](file:///c:/Code/LLAISYS/llaisys/src/ops/linear/op.cpp) — **核心改动**：分片后的列并行/行并行 Linear |
| | | [src/ops/self_attention/op.cpp](file:///c:/Code/LLAISYS/llaisys/src/ops/self_attention/op.cpp) — 分头并行 Attention |
| **第 9-10 层** | 模型接口 | [src/models/qwen2/qwen2.cpp](file:///c:/Code/LLAISYS/llaisys/src/models/qwen2/qwen2.cpp) — forward 流程加入 AllReduce/AllGather 通信 |
| | | [python/llaisys/models/qwen2.py](file:///c:/Code/LLAISYS/llaisys/python/llaisys/models/qwen2.py) — 多设备模型创建 |
| **第 11 层** | 构建系统 | [xmake.lua](file:///c:/Code/LLAISYS/llaisys/xmake.lua) — 添加 NCCL/MPI 链接 |

> 本任务以 **第 4 层通信抽象 + 第 8 层算子分片**为核心，需在 Project 2（CUDA）完成后进行。如果不做 CUDA 版本，只用 CPU+MPI 也可以验证逻辑。

---

## 背景知识：张量并行（Tensor Parallelism）

### 核心思想
将模型的一层切分到多个设备上，每个设备计算一部分，然后通过通信同步结果。

### 两种切分方式

**1. 列并行（Column Parallel）**
- 将权重矩阵按列切分：W = [W₁ | W₂]（W₁ 在 GPU0，W₂ 在 GPU1）
- 输入 X 复制到所有设备
- 每个设备计算 Yᵢ = X · Wᵢ
- 输出在列维度拼接：Y = [Y₁ | Y₂]
- 用于：Q/K/V 投影、FFN 的 Gate/Up 投影

**2. 行并行（Row Parallel）**
- 将权重矩阵按行切分：W = [W₁; W₂]（W₁ 在 GPU0，W₂ 在 GPU1）
- 输入 X 也按列切分：X = [X₁ | X₂]
- 每个设备计算 Yᵢ = Xᵢ · Wᵢ
- 输出求和：Y = Y₁ + Y₂（AllReduce）
- 用于：O 投影、FFN 的 Down 投影

### Qwen2 中需要并行的层
```
Embedding    → 不切分（或词表切分，需 AllGather）
Attention:
  Q 投影     → 列并行（按 head 维度切分）
  K 投影     → 列并行
  V 投影     → 列并行
  Attention  → 各设备独立计算（各自 head）
  O 投影     → 行并行 + AllReduce
  RMS Norm   → 不切分（元素级，各设备独立计算）
FFN:
  Gate 投影  → 列并行
  Up 投影    → 列并行
  SwiGLU     → 元素级，各设备独立
  Down 投影  → 行并行 + AllReduce
```

---

## 任务清单

### 阶段一：通信层实现

#### 任务 5.1：设计通信抽象接口

- [ ] **创建 `src/device/communicator.hpp`**
  ```cpp
  class Communicator {
  public:
      virtual void allReduce(tensor_t data) = 0;    // 求和同步
      virtual void allGather(tensor_t data) = 0;    // 收集拼接
      virtual void broadcast(tensor_t data, int root) = 0;
      virtual int rank() const = 0;
      virtual int worldSize() const = 0;
      virtual void barrier() = 0;
  };
  ```

- [ ] **实现 NCCL Communicator（GPU）**
  - `ncclAllReduce` / `ncclAllGather` / `ncclBroadcast`
  - 在 `nvidia_resource.cu` 中存储 `ncclComm_t`

- [ ] **实现 MPI Communicator（CPU）**
  - `MPI_Allreduce` / `MPI_Allgather` / `MPI_Bcast`

#### 任务 5.2：初始化通信环境

- [ ] **NCCL：初始化 NCCL 通信器**
  - 为每个 GPU 创建独立的 `ncclComm_t`
  - 分配 NCCL ID 并在设备间共享

- [ ] **MPI：初始化 MPI 环境**
  - `MPI_Init` + `MPI_Comm_rank` + `MPI_Comm_size`

---

### 阶段二：模型分片

#### 任务 5.3：修改模型创建支持多设备

- [ ] **修改 `llaisysQwen2ModelCreate`**
  - 接受 `device_ids` 数组和 `ndevice`
  - 每个设备创建部分模型（只持有自己负责的权重分片）

- [ ] **实现权重分片加载**
  - 列并行权重：每设备持有 `W[:, start:end]`
  - 行并行权重：每设备持有 `W[start:end, :]`
  - 加载 safetensors 时按分片规则切分

#### 任务 5.4：修改 Attention 层

- [ ] **Q/K/V 投影（列并行）**
  - 各设备计算自己负责的 head 的 Q/K/V
  - 无需通信

- [ ] **Self-Attention**
  - 各设备独立计算自己负责的 head
  - K/V 的 head 数需按设备数缩小

- [ ] **O 投影（行并行 + AllReduce）**
  - 各设备独立计算输出
  - 调用 AllReduce 求和得到完整结果

#### 任务 5.5：修改 FFN 层

- [ ] **Gate/Up 投影（列并行）**
  - 各设备独立计算

- [ ] **SwiGLU**
  - 各设备独立计算（元素级操作）

- [ ] **Down 投影（行并行 + AllReduce）**
  - 各设备独立计算
  - AllReduce 求和

#### 任务 5.6：修改 RMS Norm

- [ ] **RMS Norm**
  - 各设备独立计算（每行独立）
  - 不需要通信

---

### 阶段三：验证

#### 任务 5.7：单设备兼容性

- [ ] **确保单设备模式仍正常工作**
  - ndevice=1 时不引入额外通信开销
  - 所有原有测试仍然通过

#### 任务 5.8：多设备验证

- [ ] **双设备推理验证**
  - 在两块 GPU（或两个 MPI 进程）上运行推理
  - 对比单设备推理结果（数值应一致）
  - 运行 `test_infer.py` 确认输出 tokens 相同

#### 任务 5.9：性能评估

- [ ] **测量分布式推理的性能**
  - 对比单设备 vs 多设备的吞吐量
  - 测量通信开销占比
  - 评估扩展性（2 设备 → 4 设备 → 8 设备）

---

## 关键难点提示

### 通信开销
- AllReduce 是主要瓶颈
- 尽量减少通信次数：将 RMS Norm 前后的通信合并
- NCCL 的 AllReduce 使用 Ring 或 Tree 算法，高效但需同步

### 数值精度
- 分片推理中，AllReduce 的顺序可能影响浮点累加结果
- 与单设备推理的结果可能有微小差异（通常在可接受范围内）

### 负载均衡
- 确保各设备计算量基本均衡
- 按 head 数切分时：nhead 需要能被 ndevice 整除
- 按 hidden_dim 切分时：hidden_dim / ndevice 无余数