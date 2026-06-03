# Layer08_Ops

问：*为什么算子层是"推理引擎的核心"？*

答：*算子是 LLM 推理的**计算原子**——每个算子对应 Transformer 架构中的一个数学操作。当前 add 已完成 CPU 实现（含 BF16 支持），其余 8 个算子均为 `TO_BE_IMPLEMENTED()` 占位，等待 Project 1-3 逐一实现。*

- [x] ## 第 8 层：Ops（算子） — 9 组算子，共 20 个文件

**理解"每个算子 = 公共接口(op.hpp) + 分发逻辑(op.cpp) + 设备实现(cpu/cuda)"三层架构。**

| 序号 | 算子 | 文件 | 行数 | 核心看点 |
|:--:|------|------|:--:|------|
| 8.1 | [add](file:///c:/Code/LLAISYS/llaisys/src/ops/add/) | op.hpp + op.cpp + cpu/add_cpu.hpp + cpu/add_cpu.cpp | 4 文件 | **唯一已完成的算子**：`op.cpp` 做设备分发 + 参数校验，`add_cpu.cpp` 做模板化 kernel（BF16/FP16/F32） |
| 8.2 | [argmax](file:///c:/Code/LLAISYS/llaisys/src/ops/argmax/) | op.hpp + op.cpp | 2 文件 | `TO_BE_IMPLEMENTED()` 占位——找最大值索引（Transformer 输出层/采样前） |
| 8.3 | [embedding](file:///c:/Code/LLAISYS/llaisys/src/ops/embedding/) | op.hpp + op.cpp | 2 文件 | `TO_BE_IMPLEMENTED()` 占位——查表（weight[index]），参数量最大的算子 |
| 8.4 | [linear](file:///c:/Code/LLAISYS/llaisys/src/ops/linear/) | op.hpp + op.cpp | 2 文件 | `TO_BE_IMPLEMENTED()` 占位——矩阵乘法 out = in × W^T + b（计算量最大） |
| 8.5 | [rms_norm](file:///c:/Code/LLAISYS/llaisys/src/ops/rms_norm/) | op.hpp + op.cpp | 2 文件 | `TO_BE_IMPLEMENTED()` 占位——RMS 归一化（Qwen2 的 LayerNorm 替代） |
| 8.6 | [rope](file:///c:/Code/LLAISYS/llaisys/src/ops/rope/) | op.hpp + op.cpp | 2 文件 | `TO_BE_IMPLEMENTED()` 占位——旋转位置编码（RoPE），Qwen2 的位置信息注入 |
| 8.7 | [self_attention](file:///c:/Code/LLAISYS/llaisys/src/ops/self_attention/) | op.hpp + op.cpp | 2 文件 | `TO_BE_IMPLEMENTED()` 占位——Softmax(QK^T/√d) × V，注意力计算核心 |
| 8.8 | [swiglu](file:///c:/Code/LLAISYS/llaisys/src/ops/swiglu/) | op.hpp + op.cpp | 2 文件 | `TO_BE_IMPLEMENTED()` 占位——SiLU 门控激活（Qwen2 FFN 使用） |
| 8.9 | [rearrange](file:///c:/Code/LLAISYS/llaisys/src/ops/rearrange/) | op.hpp + op.cpp | 2 文件 | `TO_BE_IMPLEMENTED()` 占位——张量重排/reshape，用于 Qwen2 的 QKV 拆分 |

---

- [x] ### 8.1 add — 唯一已完成的算子

**设计模式**：`op.hpp` 声明接口 → `op.cpp` 做参数校验 + 设备分发 → `cpu/add_cpu.hpp` 声明 CPU kernel → `cpu/add_cpu.cpp` 实现模板化 kernel。

```
// op.hpp
#pragma once
#include "../../tensor/tensor.hpp"
namespace llaisys::ops {
void add(tensor_t c, tensor_t a, tensor_t b);
}
```

接口声明：`C = A + B`，三个 `tensor_t`（`shared_ptr<Tensor>`）参数。注意 `c` 是**输出参数**（预分配），不是返回值——这是推理引擎的常见模式：避免每次运算都 `new` 新 Tensor，复用预分配内存。

```
// op.cpp
#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/add_cpu.hpp"

namespace llaisys::ops {
void add(tensor_t c, tensor_t a, tensor_t b) {
    CHECK_SAME_DEVICE(c, a, b);
    CHECK_SAME_SHAPE(c->shape(), a->shape(), b->shape());
    CHECK_SAME_DTYPE(c->dtype(), a->dtype(), b->dtype());
    ASSERT(c->isContiguous() && a->isContiguous() && b->isContiguous(),
           "Add: all tensors must be contiguous.");
```

**参数校验四步**：设备一致 → 形状一致 → 数据类型一致 → 连续存储。全部通过 `check.hpp` 中的宏实现，校验失败则抛异常。

```
    if (c->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::add(c->data(), a->data(), b->data(), c->dtype(), c->numel());
    }

    llaisys::core::context().setDevice(c->deviceType(), c->deviceId());

    switch (c->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::add(c->data(), a->data(), b->data(), c->dtype(), c->numel());
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
```

**设备分发**：

1. CPU 快速路径：`deviceType() == CPU` 时直接调 CPU kernel（跳过 `setDevice` 开销）
2. 非 CPU：`setDevice` 切换到目标设备，`switch` 按设备类型分发
3. NVIDIA：`TO_BE_IMPLEMENTED()` 占位，等待 Project 2 的 CUDA kernel 实现

```
// cpu/add_cpu.hpp
#pragma once
#include "llaisys.h"
#include <cstddef>
namespace llaisys::ops::cpu {
void add(std::byte *c, const std::byte *a, const std::byte *b, llaisysDataType_t type, size_t size);
}
```

CPU kernel 接口：参数是 `std::byte*`（原始字节指针），解耦了 Tensor 对象——kernel 只关心"内存地址 + 数据类型 + 元素个数"。

```
// cpu/add_cpu.cpp
template <typename T>
void add_(T *c, const T *a, const T *b, size_t numel) {
    for (size_t i = 0; i < numel; i++) {
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            c[i] = llaisys::utils::cast<T>(
                llaisys::utils::cast<float>(a[i]) + llaisys::utils::cast<float>(b[i]));
        } else {
            c[i] = a[i] + b[i];
        }
    }
}
```

**模板化 kernel**：`if constexpr` 在编译期分流——F32 直接 `+` 运算，BF16/FP16 先 `cast<float>()` 转 F32 计算再 `cast<T>()` 转回。这是 CPU 推理的共同模式：CPU 没有原生 BF16 硬件指令，用 F32 模拟。

```
namespace llaisys::ops::cpu {
void add(std::byte *c, const std::byte *a, const std::byte *b, llaisysDataType_t type, size_t numel) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return add_(reinterpret_cast<float *>(c), ...);
    case LLAISYS_DTYPE_BF16:
        return add_(reinterpret_cast<llaisys::bf16_t *>(c), ...);
    case LLAISYS_DTYPE_F16:
        return add_(reinterpret_cast<llaisys::fp16_t *>(c), ...);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
```

**类型分发**：`switch(type)` + `reinterpret_cast` 将 `std::byte*` 转为具体类型指针，然后调用模板函数。这是 C++ 中"运行时多态"的经典模式——`switch` 在运行时决定类型，`reinterpret_cast` 在编译时保证内存布局正确。

---

- [x] ### 8.2~8.9 argmax / embedding / linear / rms_norm / rope / self_attention / swiglu / rearrange — 待实现算子

8 个算子当前状态完全一致：

```
// op.hpp（以 rms_norm 为例）
#pragma once
#include "../../tensor/tensor.hpp"
namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps);
}

// op.cpp
#include "op.hpp"
namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    TO_BE_IMPLEMENTED();
}
}
```

**每个算子的数学模型**：

| 算子 | 输入 | 公式 | 在 Qwen2 中的位置 |
|------|------|------|-------------------|
| `argmax` | `vals[N]` | `max_idx = argmax(vals)` | 输出层采样前，找概率最大的 token |
| `embedding` | `index[B,T]`, `weight[V,D]` | `out[B,T,D] = weight[index]` | 输入层，token→向量 |
| `linear` | `in[M,K]`, `weight[N,K]`, `bias[N]` | `out = in × W^T + bias` | Q/K/V 投影、FFN 上下投影 |
| `rms_norm` | `in[N,D]`, `weight[D]`, `eps` | `out = in / RMS(in) × weight` | 每个 Attention 和 FFN 之前 |
| `rope` | `in[B,H,T,D]`, `pos_ids[T]`, `theta` | 旋转位置编码 | Q/K 计算后，attention 前 |
| `self_attention` | `q,k,v[B,H,T,D]`, `scale` | `Softmax(QK^T/√d) × V` | Transformer 核心 |
| `swiglu` | `gate[N,D]`, `up[N,D]` | `out = gate × SiLU(up)` | FFN 激活函数 |
| `rearrange` | `in[B,T,H,D]` | 张量重排 | QKV 拆分/合并 |

**实现任务分配**：

- Project 1（CPU 推理）：完成所有 9 个算子的 CPU kernel
- Project 2（CUDA 集成）：为支持 BF16 的算子增加 CUDA kernel
- Project 3（模型推理）：算子串联成完整的前向传播流程

**完整调用链（以 add 为例）**：
```
用户代码: ops::add(c, a, b)
  → 参数校验（CHECK_SAME_DEVICE/SHAPE/DTYPE + ASSERT(contiguous)）
  → 设备分发（switch device_type）
    → [CPU] cpu::add(c->data(), a->data(), b->data(), dtype, numel)
      → switch(dtype) → reinterpret_cast<T*>
        → add_<T>(c, a, b, numel)
          → for i in range(numel): c[i] = a[i] + b[i]
            → [BF16] cast<float> → + → cast<bf16_t>
    → [NVIDIA] TO_BE_IMPLEMENTED()（Project 2 实现）
```