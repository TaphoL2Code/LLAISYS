# Assignment #3: Large Language Model Inference — 事件清单

## 主要修改文件

### 需要新建/修改的 C++ 后端文件
| 文件 | 作用 |
|------|------|
| [`include/llaisys/models/qwen2.h`](file:///c:/Code/LLAISYS/llaisys/include/llaisys/models/qwen2.h) | 可能需要扩展（模型元信息、权重结构、API 声明） |
| `src/models/qwen2/qwen2.hpp` | **新建** — Qwen2 模型 C++ 类声明 |
| `src/models/qwen2/qwen2.cpp` | **新建** — Qwen2 模型 C++ 核心实现（模型推理流程） |
| `src/models/qwen2/qwen2_weight_loader.cpp` | **新建** — 从 safetensors 加载权重到 C++ 后端 |
| `src/llaisys/qwen2.cc` | **新建** — Qwen2 的 C API 桥接（将 C 句柄转为 C++ 调用） |
| [`xmake.lua`](file:///c:/Code/LLAISYS/llaisys/xmake.lua) | **修改** — 添加 qwen2 模块的编译目标 |

### 需要修改的 Python 文件
| 文件 | 作用 |
|------|------|
| [`python/llaisys/models/qwen2.py`](file:///c:/Code/LLAISYS/llaisys/python/llaisys/models/qwen2.py) | **实现**模型加载（safetensors→C++）和 generate 函数 |
| [`python/llaisys/libllaisys/ops.py`](file:///c:/Code/LLAISYS/llaisys/python/llaisys/libllaisys/ops.py) | 可能需要**添加** Qwen2 模型相关的 ctypes 函数签名 |
| `python/llaisys/libllaisys/` | **新建** qwen2 的 ctypes 绑定文件 |

### 可参考的测试文件
| 文件 | 作用 |
|------|------|
| [`test/test_infer.py`](file:///c:/Code/LLAISYS/llaisys/test/test_infer.py) | 最终的端到端验证测试 |

## 需更改的配置
- **[`xmake.lua`](file:///c:/Code/LLAISYS/llaisys/xmake.lua)**：添加 qwen2 模块的静态库目标（参照 `llaisys-ops`），并将新文件链接到最终的 `llaisys` 共享库

## 建议阅读层次

> 参考 [遍历式阅读顺序](../thinking/遍历式阅读顺序.md)

| 层次 | 内容 | 必读文件 |
|:--:|------|------|
| **第 7 层** | Tensor（前置） | [src/tensor/tensor.cpp](file:///c:/Code/LLAISYS/llaisys/src/tensor/tensor.cpp) — 回顾 `create()`/`load()`/`data()`，模型权重加载和中间结果都需要创建 Tensor |
| **第 8 层** | 算子（前置） | [src/ops/argmax/op.cpp](file:///c:/Code/LLAISYS/llaisys/src/ops/argmax/op.cpp) ~ [src/ops/swiglu/op.cpp](file:///c:/Code/LLAISYS/llaisys/src/ops/swiglu/op.cpp) — 各算子 op.hpp 中的签名，模型 forward 要调它们 |
| **第 9 层** | C API | [src/llaisys/tensor.cc](file:///c:/Code/LLAISYS/llaisys/src/llaisys/tensor.cc) — 参考 `extern "C"` 模式，新建 `qwen2.cc` 需同样处理 `void* handle` |
| | | [include/llaisys/models/qwen2.h](file:///c:/Code/LLAISYS/llaisys/include/llaisys/models/qwen2.h) — 模型 C API 声明 |
| **第 10 层** | Python 绑定 | [python/llaisys/models/qwen2.py](file:///c:/Code/LLAISYS/llaisys/python/llaisys/models/qwen2.py) — 待实现模型加载和生成 |
| | | [python/llaisys/libllaisys/__init__.py](file:///c:/Code/LLAISYS/llaisys/python/llaisys/libllaisys/__init__.py) — 参考 ctypes CDLL 绑定模式 |
| **第 11 层** | 构建系统 | [xmake.lua](file:///c:/Code/LLAISYS/llaisys/xmake.lua) — 参考 `llaisys-ops` target 添加方式 |
| **第 12 层** | 测试 | [test/test_infer.py](file:///c:/Code/LLAISYS/llaisys/test/test_infer.py) — 端到端验证：`hf_infer()` vs `llaisys_infer()` |

> 第 0-6 层已在 Assignment 1 中读过。本任务核心是读懂 **第 7-8 层**的数据结构和算子，然后新建 **第 9-10 层**的文件。

---

## 关键难点提示

### KV Cache
- KV Cache 是实现高效自回归推理的关键
- Prefill 阶段（首次处理输入序列）：计算所有 input tokens 的 K/V 并存入 cache
- Decode 阶段（每次生成一个新 token）：只计算新 token 的 K/V 并追加到 cache

### 权重加载
- safetensors 中的权重可能以 BF16 存储，需转换为模型使用的数据类型
- 注意权重的 shape 匹配
- 需要正确映射 safetensors 中的参数名到 `LlaisysQwen2Weights` 中的对应张量

### 数值精度
- 与 PyTorch 的 bf16 推理对比时，注意容差设置
- 中间结果随层数增加可能有累积误差
- 使用 `atol=1e-5, rtol=1e-5` 作为对比标准

---

## 任务清单

### 阶段一：模型后端 C++ 实现

#### 任务 3.1：设计模型 C++ 类语义

- [ ] **创建 `src/models/qwen2/qwen2.hpp`**
  - 定义 `Qwen2Model` 类，持有超参数和所有权重张量
  - 持有 KV Cache 状态
  - 声明 `infer(token_ids, ntoken)` 方法

#### 任务 3.2：实现 KV Cache

- [ ] **设计 KV Cache 数据结构**
  - 每层需要一个 K Cache 和一个 V Cache（共 2 × nlayer 个张量）
  - Cache 维度：`[max_seq_len, nkv_head, head_dim]`
  - 需要记录当前已缓存的序列长度

- [ ] **实现 KV Cache 的初始化/更新逻辑**
  - 首次推理：将计算出的 K/V 存入 Cache
  - 后续推理：将新的 K/V 追加到现有 Cache 末尾

#### 任务 3.3：实现单层 Transformer Block

- [ ] **实现 Attention 子层**
  - Q/K/V 投影（`llaisysLinear`）
  - RoPE 位置编码（`llaisysROPE`）
  - KV Cache 拼接
  - Self-Attention（`llaisysSelfAttention`）
  - O 投影（`llaisysLinear`）
  - 残差连接（`llaisysAdd`）

- [ ] **实现 FFN 子层**
  - Gate/Up 投影（`llaisysLinear`）
  - SwiGLU 激活（`llaisysSwiGLU`）
  - Down 投影（`llaisysLinear`）
  - 残差连接（`llaisysAdd`）

- [ ] **实现完整层的前向传播**
  - Attention 前：RMS Norm → Attention → 残差
  - FFN 前：RMS Norm → FFN → 残差

#### 任务 3.4：实现完整推理流程

- [ ] **实现 `infer` 方法（完整 pipeline）**
  - Step 1: Embedding 查表（`llaisysEmbedding`）
  - Step 2: 循环 nlayer 次（每层调用 Transformer Block）
  - Step 3: 最终 RMS Norm（`llaisysRmsNorm`）
  - Step 4: Output 投影到词表（`llaisysLinear`）
  - Step 5: Argmax 采样（获取下一个 token）
  - Step 6: 预填充（Prefill）顺序处理所有输入 tokens，然后自回归生成

#### 任务 3.5：实现 C API 桥接

- [ ] **创建 `src/llaisys/qwen2.cc`**
  - `llaisysQwen2ModelCreate` — 创建模型实例、分配 KV Cache
  - `llaisysQwen2ModelDestroy` — 销毁模型、释放资源
  - `llaisysQwen2ModelWeights` — 返回权重结构体指针
  - `llaisysQwen2ModelInfer` — 执行推理，返回生成的 token 序列

#### 任务 3.6：修改 xmake.lua 编译配置

- [ ] **添加 `llaisys-qwen2` 静态库目标**
  - 依赖 `llaisys-ops`
  - 包含 `src/models/qwen2/*.cpp`
  - 将 `llaisys-qwen2` 作为 `llaisys` 共享库的依赖

---

### 阶段二：Python 前端实现

#### 任务 3.7：实现 ctypes 绑定

- [ ] **创建/扩展 `python/llaisys/libllaisys/` 中的 Qwen2 绑定**
  - 注册 C API 函数的 ctypes 签名
  - 定义 Meta 和 Weights 的 ctypes.Structure

#### 任务 3.8：实现 Python 模型类

- [ ] **实现 `python/llaisys/models/qwen2.py` 的 `__init__`**
  - 读取 `config.json` 获取模型超参数
  - 调用 C API `llaisysQwen2ModelCreate` 创建后端模型
  - 遍历所有 `.safetensors` 文件加载权重

- [ ] **实现 `generate` 方法**
  - Tokenizer 编码 → chat template → prefill → 自回归生成 → 解码

---

### 阶段三：调试与验证

#### 任务 3.9：逐层调试对比

- [ ] **使用 `debug()` 函数对比每个中间结果**
  - Embedding → Q/K/V 投影 → RoPE → Self-Attention → O 投影 → FFN → logits
  - 与 PyTorch 版本同一层的输出做数值对比

#### 任务 3.10：端到端验证

- [ ] **运行推理测试并提交代码**
  - ```bash
    python test/test_infer.py --model <模型路径> --test
    ```