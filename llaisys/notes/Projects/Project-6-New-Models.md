# Project #6: Support New Models — 事件清单

## 主要修改文件

### 需要新建的 C++ 文件
| 文件 | 作用 |
|------|------|
| `include/llaisys/models/<new_model>.h` | **新建** — 新模型的 C API 头文件（Meta、Weights、API） |
| `src/models/<new_model>/<new_model>.hpp` | **新建** — 新模型 C++ 类声明 |
| `src/models/<new_model>/<new_model>.cpp` | **新建** — 新模型 C++ 核心实现 |
| `src/models/<new_model>/<new_model>_weight_loader.cpp` | **新建** — 权重加载 |
| `src/llaisys/<new_model>.cc` | **新建** — C API 桥接 |

### 需要新建的 Python 文件
| 文件 | 作用 |
|------|------|
| `python/llaisys/models/<new_model>.py` | **新建** — 新模型的 Python 封装 |
| `python/llaisys/libllaisys/<new_model>.py` | **新建** — ctypes 绑定 |

### 需要修改的已有文件
| 文件 | 修改内容 |
|------|----------|
| [`xmake.lua`](file:///c:/Code/LLAISYS/llaisys/xmake.lua) | 添加新模型的编译目标 |
| [`python/llaisys/models/__init__.py`](file:///c:/Code/LLAISYS/llaisys/python/llaisys/models/__init__.py) | 导出新模型类 |
| [`python/llaisys/libllaisys/__init__.py`](file:///c:/Code/LLAISYS/llaisys/python/llaisys/libllaisys/__init__.py) | 加载新模型的 ctypes 绑定 |
| [`test/test_infer.py`](file:///c:/Code/LLAISYS/llaisys/test/test_infer.py) | 可能需要**修改/扩展**以支持新模型测试 |

### 可能需要的额外算子
| 文件 | 作用 |
|------|------|
| `src/ops/<new_op>/` | **新建** — 如果新模型使用了现有算子不支持的运算 |
| [`include/llaisys/ops.h`](file:///c:/Code/LLAISYS/llaisys/include/llaisys/ops.h) | **修改** — 如需要，添加新算子的 C API |

## 需更改的配置
- **`xmake.lua`**：添加新模型的静态库编译目标（参照 qwen2 的配置方式）
- 无需其他配置变更

## 建议阅读层次

> 参考 [遍历式阅读顺序](../thinking/遍历式阅读顺序.md)

| 层次 | 内容 | 必读文件 |
|:--:|------|------|
| **第 0 层** | 公共头文件 | [include/llaisys/models/qwen2.h](file:///c:/Code/LLAISYS/llaisys/include/llaisys/models/qwen2.h) — 参考 Qwen2 的 C API 结构，新建新模型头文件 |
| **第 7 层** | Tensor | [src/tensor/tensor.cpp](file:///c:/Code/LLAISYS/llaisys/src/tensor/tensor.cpp) — 理解权重张量的创建和管理 |
| **第 8 层** | 算子 | [src/ops/](file:///c:/Code/LLAISYS/llaisys/src/ops/) — 检查新模型是否需要额外的算子（如 GQA、MoE routing、新激活函数） |
| **第 9 层** | C API | [src/llaisys/tensor.cc](file:///c:/Code/LLAISYS/llaisys/src/llaisys/tensor.cc) — 参考 `extern "C"` 模式，新建 `<model>.cc` |
| | | `src/models/qwen2/qwen2.hpp` + `qwen2.cpp` — **精读参考**：模型 C++ 实现模式（权重结构、forward/repeat_kv/apply_rotary_pos_emb） |
| **第 10 层** | Python 绑定 | [python/llaisys/models/qwen2.py](file:///c:/Code/LLAISYS/llaisys/python/llaisys/models/qwen2.py) — **精读参考**：Python 端模型加载和生成 |
| | | [python/llaisys/libllaisys/__init__.py](file:///c:/Code/LLAISYS/llaisys/python/llaisys/libllaisys/__init__.py) — 参考 ctypes 绑定模式 |
| **第 11 层** | 构建系统 | [xmake.lua](file:///c:/Code/LLAISYS/llaisys/xmake.lua) — 参考 `llaisys` target 的依赖配置 |
| **第 12 层** | 测试 | [test/test_infer.py](file:///c:/Code/LLAISYS/llaisys/test/test_infer.py) — 扩展支持新模型测试 |

> 本任务以 **第 9-10 层的模型实现**为核心，强烈建议**精读 Qwen2 的所有代码**作为参考模板。如果新模型架构与 Qwen2 相似（如 LLaMA），90% 的代码可以直接复用，只需修改权重大小和少量逻辑。

---

## 推荐候选模型及其挑战

### 选择 1：LLaMA 系列（LLaMA-2 / LLaMA-3 / LLaMA-3.1）
- **相似度**：与 Qwen2 架构 90% 相似
- **主要差异**：
  - 使用 **SwiGLU**（相同）✓
  - 使用 **RMS Norm**（相同）✓
  - 使用 **RoPE**（相同）✓
  - **GQA**（相同）✓
  - 无 bias（Qwen2 的 Q/K/V/O 投影有 bias，LLaMA 没有）⚠ 需处理可选 bias
- **难度**：⭐⭐（低）

### 选择 2：Mistral / Mixtral（MoE）
- **相似度**：与 Qwen2 架构 70% 相似
- **主要差异**：
  - **MoE（Mixture of Experts）**：FFN 替换为多个 expert + router
  - 需要实现 **Top-K routing** 算子
  - 需要实现 expert 负载均衡
- **难度**：⭐⭐⭐⭐（高）

### 选择 3：Gemma-2
- **相似度**：与 Qwen2 架构 80% 相似
- **主要差异**：
  - 使用 **GeGLU**（而非 SwiGLU）→ 需要新算子
  - Pre-Norm + Post-Norm 混合
  - 可能使用 **logit soft-capping**
- **难度**：⭐⭐⭐（中）

### 选择 4：ChatGLM 系列
- **相似度**：与 Qwen2 架构 60% 相似
- **主要差异**：
  - **双向注意力 + 自回归混合**
  - 不同的位置编码方式
  - 不同的 Norm 位置
- **难度**：⭐⭐⭐⭐（高）

### 选择 5：Phi-3 / Phi-4
- **相似度**：与 Qwen2 架构 85% 相似
- **主要差异**：
  - Block Sparse Attention（可选）
  - 部分使用 GELU 激活（而非 SwiGLU）
- **难度**：⭐⭐⭐（中）

---

## 任务清单

### 阶段一：选型与分析

#### 任务 6.1：选择目标模型

- [ ] **评估候选模型**
  - 考察与现有算子的兼容度
  - 考察模型架构的特殊性
  - 考察模型的流行度和实用价值

- [ ] **研究模型架构**
  - 阅读 HuggingFace 上的模型定义代码（`modeling_<model>.py`）
  - 画出模型前向传播的数据流图
  - 列出与 Qwen2 的差异点

- [ ] **确定需要新增的算子**
  - 是否所有现有算子都适用？
  - 哪些新算子必须实现？
  - 哪些可以用现有算子组合实现？

---

### 阶段二：后端实现

#### 任务 6.2：实现新算子（如需要）

- [ ] **为新模型特有的操作创建算子**
  - 遵循 `src/ops/<op_name>/op.hpp` + `op.cpp` + `cpu/` 的标准结构
  - 至少支持 F32
  - 编写对应的测试文件 `test/ops/<op_name>.py`

- [ ] **在 C API 中注册新算子**
  - `include/llaisys/ops.h` 添加声明
  - `src/llaisys/ops.cc` 添加桥接
  - `python/llaisys/libllaisys/ops.py` 注册 ctypes
  - `python/llaisys/ops.py` 添加 Python 方法

#### 任务 6.3：创建模型头文件

- [ ] **创建 `include/llaisys/models/<new_model>.h`**
  - 定义模型超参数结构体（Meta）
  - 定义权重结构体（Weights）— 列出所有需要的权重张量
  - 声明 C API：Create, Destroy, Weights（获取权重结构体）, Infer

#### 任务 6.4：实现模型 C++ 类

- [ ] **创建 `src/models/<new_model>/<new_model>.hpp` + `.cpp`**
  - 参照 `src/models/qwen2/` 的实现模式
  - 实现 Transformer 层的前向传播
  - 实现 KV Cache
  - 实现完整推理 pipeline（Embedding → N×Layers → Norm → Head）

- [ ] **处理与 Qwen2 的差异**
  - 例如 LLaMA 的 attention 没有 bias
  - 例如 MoE 的路由逻辑
  - 不同的 Norm 位置（Pre-Norm vs Post-Norm）

#### 任务 6.5：实现 C API 桥接

- [ ] **创建 `src/llaisys/<new_model>.cc`**
  - 将 C++ 模型类的方法包装为 C 函数
  - 模型创建 → `new` + C++ 构造函数
  - 模型销毁 → `delete`
  - 模型推理 → 调用 C++ 类的 infer 方法

#### 任务 6.6：修改编译配置

- [ ] **在 `xmake.lua` 中添加新模型目标**
  - 参照 `llaisys-qwen2` 的配置
  - 依赖 `llaisys-ops`
  - 添加到 `llaisys` 共享库的依赖列表

---

### 阶段三：Python 前端

#### 任务 6.7：实现 ctypes 绑定

- [ ] **创建 `python/llaisys/libllaisys/<new_model>.py`**
  - 注册 Create/Destroy/Weights/Infer 的 ctypes 函数签名
  - 定义 Meta 和 Weights 的 ctypes.Structure

- [ ] **在 `python/llaisys/libllaisys/__init__.py` 中加载**
  - 调用新模块的加载函数

#### 任务 6.8：实现 Python 模型类

- [ ] **创建 `python/llaisys/models/<new_model>.py`**
  - `__init__`：读取 config.json → 创建后端模型 → 加载 safetensors 权重
  - `generate`：Tokenizer 编码 → Prefill → 自回归生成 → 解码
  - 支持 Temperature/Top-K/Top-P 采样

- [ ] **在 `python/llaisys/models/__init__.py` 中导出**

---

### 阶段四：测试与验证

#### 任务 6.9：编写推理测试

- [ ] **在 `test/` 中创建新模型的推理测试**
  - 参照 `test/test_infer.py` 的结构
  - 加载新模型
  - 用 argmax 采样对比 PyTorch 结果
  - 验证 token 序列完全一致

#### 任务 6.10：端到端验证

- [ ] **运行测试**
  - ```bash
    python test/test_<new_model>_infer.py --model <模型路径> --test
    ```
  - 确认绿色 `Test passed!`

- [ ] **性能对比**
  - 对比 LLAISYS vs PyTorch 的推理速度
  - 记录吞吐量（tokens/second）

---

## 通用适配检查清单

无论选择哪个新模型，都需要逐项检查以下内容并处理差异：

- [ ] **Embedding 层**：是否支持 token type embeddings（如 BERT）？
- [ ] **位置编码**：RoPE / ALiBi / Learned Positional Embedding / NoPE？
- [ ] **Norm 方式**：RMS Norm / Layer Norm / 组合使用？Pre-Norm 还是 Post-Norm？
- [ ] **Attention 类型**：MHA / MQA / GQA / Sliding Window / Sparse？
- [ ] **Attention bias**：Q/K/V/O 投影有无 bias？
- [ ] **激活函数**：SwiGLU / GeGLU / GELU / ReLU / SiLU？
- [ ] **FFN 类型**：标准 FFN / MoE（多个 Expert）？
- [ ] **残差连接**：标准残差 / Pre + Post 残差？
- [ ] **输出层**：是否需要 tie word embeddings（weight tying）？
- [ ] **特殊结构**：如 Mamba 的 SSM / RWKV 的 WKV 等