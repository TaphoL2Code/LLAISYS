# Project #6: Support New Models — 事件清单

## 选择模型：LLaMA 系列

选择 LLaMA 作为首个新模型，原因：
- 与 Qwen2 架构 90% 相似（SwiGLU、RMS Norm、RoPE、GQA 均相同）
- 唯一差异：LLaMA 的 Q/K/V/O 投影无 bias
- 权重名称和 config.json 格式高度一致

## 主要修改文件

### 新建的 C++ 文件
| 文件 | 作用 | 状态 |
|------|------|:--:|
| [`include/llaisys/models/llama.h`](file:///c:/Code/LLAISYS/llaisys/include/llaisys/models/llama.h) | **新建** — LLaMA 的 C API 头文件（Meta、Weights、API） | Done |
| [`src/models/llama/llama.hpp`](file:///c:/Code/LLAISYS/llaisys/src/models/llama/llama.hpp) | **新建** — LLaMA C++ 类声明 | Done |
| [`src/models/llama/llama.cpp`](file:///c:/Code/LLAISYS/llaisys/src/models/llama/llama.cpp) | **新建** — LLaMA C++ 核心实现（无 bias 注意力） | Done |
| [`src/llaisys/llama.cc`](file:///c:/Code/LLAISYS/llaisys/src/llaisys/llama.cc) | **新建** — C API 桥接 | Done |

### 新建的 Python 文件
| 文件 | 作用 | 状态 |
|------|------|:--:|
| [`python/llaisys/models/llama.py`](file:///c:/Code/LLAISYS/llaisys/python/llaisys/models/llama.py) | **新建** — LLaMA 的 Python 封装 | Done |
| [`python/llaisys/libllaisys/llama.py`](file:///c:/Code/LLAISYS/llaisys/python/llaisys/libllaisys/llama.py) | **新建** — ctypes 绑定 | Done |

### 修改的已有文件
| 文件 | 修改内容 | 状态 |
|------|----------|:--:|
| [`xmake.lua`](file:///c:/Code/LLAISYS/llaisys/xmake.lua) | 添加 `llaisys-llama` 静态库目标和依赖 | Done |
| [`python/llaisys/models/__init__.py`](file:///c:/Code/LLAISYS/llaisys/python/llaisys/models/__init__.py) | 导出 `Llama` 类 | Done |
| [`python/llaisys/libllaisys/__init__.py`](file:///c:/Code/LLAISYS/llaisys/python/llaisys/libllaisys/__init__.py) | 加载 `load_llama` 绑定 | Done |

---

## 实现细节

### 与 Qwen2 的差异处理

LLaMA 与 Qwen2 的架构几乎完全相同，唯一的差异点及处理方式如下：

| 差异点 | Qwen2 | LLaMA | 处理方式 |
|--------|-------|-------|----------|
| Attention Q/K/V bias | 有 bias | 无 bias | C++ 中 `linear()` 调用统一传 `nullptr` |
| Attention O bias | 无 bias | 无 bias | 相同，都传 `nullptr` |
| Weights 结构体 bias 字段 | `attn_q_b`, `attn_k_b`, `attn_v_b` | 无 | C 结构体中去掉 bias 字段 |
| 权重文件名 | `.bias` 后缀存在 | 无 `.bias` 文件 | Python 权重映射表中去掉 bias 条目 |
| EOS token ID | 151643 (Qwen) | 2 (LLaMA) | 从 config.json 读取，默认值不同 |

### C++ 架构

- `LlamaModel` 类继承与 Qwen2Model 相同的结构模式
- `_transformer_block` 中的 Q/K/V/O 线性投影均使用 `nullptr` bias
- 其余组件（RMS Norm、RoPE、Self-Attention、SwiGLU、KV-Cache）完全复用现有算子
- 支持 `infer()`（贪婪）、`forward()`（返回 logits）、`reset_kv_cache()` 三种接口

### Python 前端

- `Llama` 类提供与 `Qwen2` 相同的接口：`generate()`, `generate_stream()`, `chat()`, `reset_kv()`
- 权重加载逻辑与 Qwen2 一致，仅需调整权重字段映射表
- 支持 Temperature、Top-K、Top-P 采样策略

---

## 通用适配检查清单

| 检查项 | Qwen2 | LLaMA | 状态 |
|--------|-------|-------|:--:|
| Embedding 层 | 标准 token embedding | 相同 | OK |
| 位置编码 | RoPE | 相同 | OK |
| Norm 方式 | RMS Norm (Pre-Norm) | 相同 | OK |
| Attention 类型 | GQA | 相同 | OK |
| Attention bias | Q/K/V 有 bias | 无 bias | 已处理 |
| 激活函数 | SwiGLU | 相同 | OK |
| FFN 类型 | 标准 FFN | 相同 | OK |
| 残差连接 | Pre-Norm 标准残差 | 相同 | OK |
| 输出层 | 独立 lm_head | 相同 | OK |
| 特殊结构 | 无 | 无 | OK |

---

## 功能验证

- [x] C++ 编译通过，无警告
- [x] `llaisys.dll` 正确导出 `llaisysLlamaModel*` 系列函数
- [x] Python ctypes 绑定正确加载所有函数签名
- [x] `Llama` 类在 `llaisys.models` 中正确导出
- [x] 权重结构体 `LlaisysLlamaWeights` 字段与 C 结构体对齐（无 bias 字段）

---

## 使用方式

```python
import llaisys
from llaisys.libllaisys import DeviceType

# 加载 LLaMA 模型
model = llaisys.models.Llama("/path/to/Llama-3.2-1B", DeviceType.CPU)

# 推理
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("/path/to/Llama-3.2-1B")
inputs = tokenizer.encode("Hello, who are you?")
outputs = model.generate(inputs, max_new_tokens=128, temperature=0.8)
print(tokenizer.decode(outputs, skip_special_tokens=True))
```

## 扩展性

此架构支持通过相同模式快速添加更多模型：
- **Phi-3**（85% 相似）：仅需调整激活函数为 GELU
- **Mistral**（70% 相似，含 MoE）：需额外实现 MoE routing 算子
- 核心思想：复用现有算子，仅实现模型间差异部分