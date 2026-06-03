# Project #3: AI Chatbot — 事件清单

## 主要修改文件

### 需要新建/修改的 Python 文件
| 文件 | 作用 |
|------|------|
| `python/llaisys/sampler.py` | **新建** — 随机采样 (Temperature, Top-K, Top-P) |
| `python/llaisys/server.py` | **新建** — FastAPI HTTP 服务器 |
| `python/llaisys/models/qwen2.py` | **修改** — 支持采样策略，支持流式输出 |
| `python/llaisys/models/chat_format.py` | **新建** — chat template 处理 |

### 需要修改的 C++ 后端文件
| 文件 | 作用 |
|------|------|
| [`src/models/qwen2/qwen2.cpp`](file:///c:/Code/LLAISYS/llaisys/src/models/qwen2/qwen2.cpp) | **修改** — KV-Cache 前缀复用（Prefix Caching） |
| [`include/llaisys/models/qwen2.h`](file:///c:/Code/LLAISYS/llaisys/include/llaisys/models/qwen2.h) | 可能需要**扩展**（返回 logits 而非 argmax 结果） |

### 可能需要新建的前端文件
| 文件 | 作用 |
|------|------|
| `chatbot_cli.py` | **新建** — 命令行交互式聊天 |
| `chatbot_web/index.html` | **新建** — Web 聊天界面 |

## 需更改的配置
- **Python 依赖**：`pip install fastapi uvicorn`（或使用已有依赖）
- **无需修改** xmake 和 C++ 编译配置

## 建议阅读层次

> 参考 [遍历式阅读顺序](../thinking/遍历式阅读顺序.md)

| 层次 | 内容 | 必读文件 |
|:--:|------|------|
| **第 7 层** | Tensor | [src/tensor/tensor.cpp](file:///c:/Code/LLAISYS/llaisys/src/tensor/tensor.cpp) — 理解 `debug()` 如何拿到 logits 数据 |
| **第 8 层** | 算子 | [src/ops/argmax/op.cpp](file:///c:/Code/LLAISYS/llaisys/src/ops/argmax/op.cpp) — 贪婪采样时调用 argmax；随机采样需要完整的 logits 向量 |
| **第 10 层** | **Python 绑定（本任务）** | [python/llaisys/models/qwen2.py](file:///c:/Code/LLAISYS/llaisys/python/llaisys/models/qwen2.py) — **核心**：修改 generate 循环，接入采样器、支持流式输出 |
| | | `python/llaisys/sampler.py` — **新建**：Temperature/Top-K/Top-P 采样实现 |
| | | `python/llaisys/server.py` — **新建**：FastAPI HTTP 服务器 |
| | | `python/llaisys/models/chat_format.py` — **新建**：chat template 格式化 |
| **第 12 层** | 测试 | [test/test_infer.py](file:///c:/Code/LLAISYS/llaisys/test/test_infer.py) — 参考推理测试模式 |

> 本任务以 **第 10 层 Python 代码**为主战场，C++ 侧只需少量修改（KV-Cache 前缀复用）。非常适合先完成 Assignment 3 后直接上手。

---

## 背景知识

### 采样策略

**Temperature Sampling**
- 在 softmax 之前除以温度参数 T
- T > 1：输出更随机、更多样（概率分布更平坦）
- T < 1：输出更确定、更保守（概率分布更尖锐）
- T = 0：退化为 argmax（贪婪采样）

**Top-K Sampling**
- 只从概率最高的 K 个 token 中采样
- 排除长尾的低概率 token

**Top-P Sampling（Nucleus Sampling）**
- 从累积概率达到 P 的最小 token 集合中采样
- 比 Top-K 更灵活（序列不确定性高时选更多 token）

**推荐组合**：Temperature + Top-P

### KV-Cache 前缀复用
- 多轮对话中，每次对话都以上一轮的所有 tokens 为前缀
- 不需要重新计算历史 tokens 的 K/V
- 只需计算本轮新增的 user prompt 和 assistant 回复

### 流式输出
- 每生成一个 token 就返回，不等所有 tokens 生成完
- 使用 Python generator（`yield`）或 FastAPI `StreamingResponse`

---

## 任务清单

### 阶段一：随机采样

#### 任务 3.1：实现 Temperature Sampling

- [ ] **创建 `python/llaisys/sampler.py`**
  - `temperature_sampling(logits, temperature)` 函数
  - logits / temperature → softmax → 按概率随机采样

#### 任务 3.2：实现 Top-K 过滤

- [ ] **实现 `top_k_filter(logits, k)`**
  - 找到第 K 大的 logit 作为阈值
  - 将低于阈值的 logit 设为 -inf
  - 然后在剩下的 logit 上做 softmax 采样

#### 任务 3.3：实现 Top-P 过滤

- [ ] **实现 `top_p_filter(logits, p)`**
  - logits → softmax → 从大到小排序
  - 累加概率直到超过 p
  - 被排除的 token 概率设为 0

#### 任务 3.4：实现 joint 采样器

- [ ] **实现 `sample(logits, temperature, top_k, top_p)`**
  - temperature > 0：先 temperature 缩放
  - top_k > 0：Top-K 过滤
  - top_p < 1.0：Top-P 过滤
  - softmax → 随机采样
  - temperature = 0 时退化为 argmax

#### 任务 3.5：修改推理流程支持采样

- [ ] **修改 `qwen2.py` 的 `generate` 方法**
  - 接受 `temperature`, `top_k`, `top_p` 参数
  - 返回 raw logits → 调用 `sample()` → 返回 token id

---

### 阶段二：HTTP 服务器

#### 任务 3.6：创建 FastAPI 服务器

- [ ] **创建 `python/llaisys/server.py`**
  - `/v1/chat/completions` POST 端点
  - 接收 OpenAI 兼容的请求格式（`messages`, `temperature`, `max_tokens` 等）
  - 返回 OpenAI 兼容的响应格式

- [ ] **实现 chat template 处理**
  - 创建 `python/llaisys/models/chat_format.py`
  - 将 `messages` 列表格式化为模型输入（使用 tokenizer 的 `apply_chat_template`）

#### 任务 3.7：实现流式输出

- [ ] **修改 `generate` 支持生成器（Generator）**
  - 每生成一个 token 就 `yield` 出去
  - 不需要 `max_tokens` 全部生成完才返回

- [ ] **实现 `/v1/chat/completions` 的 `stream=True` 模式**
  - 使用 `StreamingResponse`
  - 每 yield 一个 token 就推送 SSE 事件

#### 任务 3.8：实现 KV-Cache 前缀复用

- [ ] **修改 Qwen2 C++ 推理逻辑**
  - 如果新输入是上一个序列的前缀延续 → 复用已有的 KV Cache
  - 需要跟踪当前 KV Cache 中的 token 数量
  - Prefill 只处理增量部分

- [ ] **在 Python 层管理对话历史和 KV Cache 的对应关系**
  - 多轮对话中复用前一轮的 KV Cache
  - 新用户消息进来时，从上次 assistant 回复结束处继续

---

### 阶段三：交互界面

#### 任务 3.9：实现命令行聊天

- [ ] **创建 `chatbot_cli.py`**
  - 加载模型
  - 循环：用户输入 → tokenize → generate → decode → 打印
  - 支持 `/exit`, `/clear` 等简单命令
  - 调用服务器 API 或直接使用模型

#### 任务 3.10（可选）：实现 Web 聊天界面

- [ ] **创建简单的 HTML 聊天界面**
  - 消息气泡（用户 / 助手）
  - 文本输入框 + 发送按钮
  - 连接到 `/v1/chat/completions` API

#### 任务 3.11（可选）：使用 Gradio 快速搭建 UI

- [ ] **安装 Gradio**
  - `pip install gradio`
- [ ] **创建 Gradio 聊天界面**
  - 调用模型推理函数
  - 美观的 Web 界面

---

## 功能验证
- [ ] Temperature=0 时生成确定性结果
- [ ] Temperature > 0 时每次生成不同结果（随机采样）
- [ ] Top-K/Top-P 过滤后不会出现极低概率 token
- [ ] `/v1/chat/completions` API 返回正确的 JSON 格式
- [ ] 流式模式下实时看到 token 逐字出现
- [ ] 多轮对话中 KV-Cache 复用正确（第二轮的延迟低于第一轮）