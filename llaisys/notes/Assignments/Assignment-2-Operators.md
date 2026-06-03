# Assignment #2: Operators — 事件清单

## 主要修改文件

### 需要创建/修改的 CPU 实现文件
| 算子 | 头文件 (新建/修改) | CPU 实现 (新建/修改) |
|------|--------------------|-----------------------|
| argmax | `src/ops/argmax/cpu/argmax_cpu.hpp` | `src/ops/argmax/cpu/argmax_cpu.cpp` |
| embedding | `src/ops/embedding/cpu/embedding_cpu.hpp` | `src/ops/embedding/cpu/embedding_cpu.cpp` |
| linear | `src/ops/linear/cpu/linear_cpu.hpp` | `src/ops/linear/cpu/linear_cpu.cpp` |
| rms_norm | `src/ops/rms_norm/cpu/rms_norm_cpu.hpp` | `src/ops/rms_norm/cpu/rms_norm_cpu.cpp` |
| rope | `src/ops/rope/cpu/rope_cpu.hpp` | `src/ops/rope/cpu/rope_cpu.cpp` |
| self_attention | `src/ops/self_attention/cpu/self_attention_cpu.hpp` | `src/ops/self_attention/cpu/self_attention_cpu.cpp` |
| swiglu | `src/ops/swiglu/cpu/swiglu_cpu.hpp` | `src/ops/swiglu/cpu/swiglu_cpu.cpp` |

### 需要修改的已有文件
| 文件 | 修改内容 |
|------|----------|
| [`src/ops/argmax/op.cpp`](file:///c:/Code/LLAISYS/llaisys/src/ops/argmax/op.cpp) | 根据设备类型分派到 CPU 实现 |
| [`src/ops/embedding/op.cpp`](file:///c:/Code/LLAISYS/llaisys/src/ops/embedding/op.cpp) | 同上 |
| [`src/ops/linear/op.cpp`](file:///c:/Code/LLAISYS/llaisys/src/ops/linear/op.cpp) | 同上 |
| [`src/ops/rms_norm/op.cpp`](file:///c:/Code/LLAISYS/llaisys/src/ops/rms_norm/op.cpp) | 同上 |
| [`src/ops/rope/op.cpp`](file:///c:/Code/LLAISYS/llaisys/src/ops/rope/op.cpp) | 同上 |
| [`src/ops/self_attention/op.cpp`](file:///c:/Code/LLAISYS/llaisys/src/ops/self_attention/op.cpp) | 同上 |
| [`src/ops/swiglu/op.cpp`](file:///c:/Code/LLAISYS/llaisys/src/ops/swiglu/op.cpp) | 同上 |

## 需更改的配置
**无需更改配置**。`xmake/cpu.lua` 中 `llaisys-ops-cpu` 编译目标已通过 `add_files("../src/ops/\*/cpu/\*.cpp")` 自动包含所有 CPU 算子实现文件。

## 建议阅读层次

> 参考 [遍历式阅读顺序](../thinking/遍历式阅读顺序.md)

| 层次 | 内容 | 必读文件 |
|:--:|------|------|
| **第 7 层** | Tensor（前置） | [src/tensor/tensor.cpp](file:///c:/Code/LLAISYS/llaisys/src/tensor/tensor.cpp) — 回顾 `data()`/`numel()`/`dtype()` 如何在 kernel 中获取数据和类型 |
| **第 8 层** | **算子（本任务）** | [src/ops/add/op.cpp](file:///c:/Code/LLAISYS/llaisys/src/ops/add/op.cpp) — **精读参考实现**：设备分派模式 |
| | | [src/ops/add/cpu/add_cpu.cpp](file:///c:/Code/LLAISYS/llaisys/src/ops/add/cpu/add_cpu.cpp) — **精读参考实现**：CPU kernel 模板模式 |
| **第 9 层** | C API | [src/llaisys/ops.cc](file:///c:/Code/LLAISYS/llaisys/src/llaisys/ops.cc) — 理解 `extern "C"` 导出如何调 `ops::xxx()` |
| **第 10 层** | Python 绑定 | [python/llaisys/libllaisys/ops.py](file:///c:/Code/LLAISYS/llaisys/python/llaisys/libllaisys/ops.py) — ctypes 函数签名 |
| | | [python/llaisys/ops.py](file:///c:/Code/LLAISYS/llaisys/python/llaisys/ops.py) — Python 算子封装 |
| **第 12 层** | 测试 | [test/ops/add.py](file:///c:/Code/LLAISYS/llaisys/test/ops/add.py) — 参考测试模式用于验证 |

> 第 0-6 层已在 Assignment 1 中读过，直接复用基础即可。本任务核心是读懂 **第 8 层的 add 参考实现**。

---

## 前置准备：理解算子架构

- [ ] **仔细阅读参考实现 `src/ops/add/`**
  - [src/ops/add/op.hpp](file:///c:/Code/LLAISYS/llaisys/src/ops/add/op.hpp) — C++ 内部声明
  - [src/ops/add/op.cpp](file:///c:/Code/LLAISYS/llaisys/src/ops/add/op.cpp) — 设备分派器
  - [src/ops/add/cpu/add_cpu.hpp](file:///c:/Code/LLAISYS/llaisys/src/ops/add/cpu/add_cpu.hpp) — CPU 实现的声明
  - [src/ops/add/cpu/add_cpu.cpp](file:///c:/Code/LLAISYS/llaisys/src/ops/add/cpu/add_cpu.cpp) — CPU 实际计算（模板支持多种数据类型）
  - [src/llaisys/ops.cc](file:///c:/Code/LLAISYS/llaisys/src/llaisys/ops.cc) — C API 桥接
  - 理解**数据流**：Python → ctypes → C API → ops.cc → op.cpp（分派器）→ cpu/xxx.cpp（计算）
  - 理解**编译链**：`xmake/cpu.lua` 将 `src/ops/*/cpu/*.cpp` 编译为 `llaisys-ops-cpu` 静态库

- [ ] **理解通用模式（每个算子都遵守）**
  - `op.cpp` 中做设备一致性检查（`CHECK_SAME_DEVICE`, `CHECK_SAME_SHAPE`, `CHECK_SAME_DTYPE`）
  - 通过 `c->data()` 获取原始 `std::byte*` 指针
  - CPU 实现用模板处理 Float32/Float16/BFloat16 三种数据类型
  - 使用 `llaisys::utils::cast()` 做类型转换（自动处理 fp16 ↔ f32）

- [ ] **运行 add 测试验证理解**
  - ```bash
    python test/ops/add.py
    ```

---

## 任务清单

### 任务 2.1：argmax

- [ ] **理解 argmax 数学定义**
  - 沿 `vals` 最后一维找最大值的位置和值
  - `vals` 是 1D 张量（shape = [N]）
  - `max_idx` 和 `max_val` 是 shape = [1] 的张量

- [ ] **创建 `src/ops/argmax/cpu/argmax_cpu.hpp`**
  - 声明模板函数 `void argmax(...)`，放在 `llaisys::ops::cpu` 命名空间

- [ ] **创建 `src/ops/argmax/cpu/argmax_cpu.cpp`**
  - 遍历 `vals` 找到最大值的位置和值
  - 支持至少 F32、F16、BF16
  - 使用 `dtype` 分发到具体类型实现

- [ ] **修改 `src/ops/argmax/op.cpp`**
  - 参照 `add/op.cpp` 的模式实现设备分派

- [ ] **验证**
  - ```bash
    python test/ops/argmax.py
    ```

### 任务 2.2：embedding

- [ ] **理解 embedding 数学定义**
  - 从 `weight`（2D, shape=[vocab_size, embed_dim]）中按 `index`（1D, Int64）取出行
  - `output` 是 2D，shape = [len(index), embed_dim]

- [ ] **创建 `src/ops/embedding/cpu/embedding_cpu.hpp` + `.cpp`**

- [ ] **修改 `src/ops/embedding/op.cpp`**

- [ ] **验证**
  - ```bash
    python test/ops/embedding.py
    ```

### 任务 2.3：linear

- [ ] **理解 linear 数学定义**
  - Y = X · W^T + b
  - `in` (2D), `weight` (2D), `bias` (1D, 可选), `out` (2D)
  - 需要支持 bias 为 nullptr（不提供偏置）

- [ ] **创建 `src/ops/linear/cpu/linear_cpu.hpp` + `.cpp`**
  - 实现三层嵌套循环的矩阵乘法

- [ ] **修改 `src/ops/linear/op.cpp`**

- [ ] **验证**
  - ```bash
    python test/ops/linear.py
    ```

### 任务 2.4：rms_norm

- [ ] **理解 rms_norm 数学定义**
  - Y_i = W_i * X_i / sqrt(mean(X_i^2) + eps)
  - `in` (2D), `weight` (1D), `out` (2D)

- [ ] **创建 `src/ops/rms_norm/cpu/rms_norm_cpu.hpp` + `.cpp`**

- [ ] **修改 `src/ops/rms_norm/op.cpp`**

- [ ] **验证**
  - ```bash
    python test/ops/rms_norm.py
    ```

### 任务 2.5：rope（旋转位置编码）

- [ ] **理解 RoPE 数学定义**
  - a'_j = a_j * cos(θ_j) - b_j * sin(θ_j)
  - b'_j = b_j * cos(θ_j) + a_j * sin(θ_j)
  - θ_j = pos_id / (theta^(2j/d))

- [ ] **创建 `src/ops/rope/cpu/rope_cpu.hpp` + `.cpp`**

- [ ] **修改 `src/ops/rope/op.cpp`**

- [ ] **验证**
  - ```bash
    python test/ops/rope.py
    ```

### 任务 2.6：self_attention

- [ ] **理解 self_attention 数学定义**
  - A = Q · K^T * scale → causal mask → softmax → Y = softmax(A) · V
  - 支持 GQA（Group Query Attention）

- [ ] **创建 `src/ops/self_attention/cpu/self_attention_cpu.hpp` + `.cpp`**

- [ ] **修改 `src/ops/self_attention/op.cpp`**

- [ ] **验证**
  - ```bash
    python test/ops/self_attention.py
    ```

### 任务 2.7：swiglu

- [ ] **理解 swiglu 数学定义**
  - $$
    out_i = up_i * gate_i / (1 + e^{(-gate_i)})
    $$
  
    
  
- [ ] **创建 `src/ops/swiglu/cpu/swiglu_cpu.hpp` + `.cpp`**

- [ ] **修改 `src/ops/swiglu/op.cpp`**

- [ ] **验证**
  - ```bash
    python test/ops/swiglu.py
    ```

### 任务 2.8：运行全部算子测试

- [ ] **依次运行每个算子测试并提交代码**

### 任务 2.9（可选）：rearrange

- [ ] **创建 `src/ops/rearrange/cpu/rearrange_cpu.hpp` + `.cpp`**
- [ ] **修改 `src/ops/rearrange/op.cpp`**

---

## 整体验证
- [ ] 所有 7 个算子测试均通过
- [ ] 每个算子至少支持 F32、F16、BF16
- [ ] 理解每个算子在 Qwen2 模型推理中的具体用途