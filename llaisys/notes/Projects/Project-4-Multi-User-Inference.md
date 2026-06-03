# Project #4: Multi-user Inference Service — 事件清单

> **前提条件**：需完成 Project #2（CUDA 集成）和 Project #3（流式输出）。

需要Web后端开发相关技能，跳过，后面需要学习补充

## 主要修改文件

### Python 后端
| 文件 | 修改内容 |
|------|----------|
| `python/llaisys/server.py` | **新建/重写** — 多用户并发服务器 |
| `python/llaisys/request_queue.py` | **新建** — 请求队列和调度器 |
| `python/llaisys/batch_scheduler.py` | **新建** — 连续批处理调度器 |
| `python/llaisys/kv_cache_pool.py` | **新建** — KV-Cache 池管理（前缀匹配复用） |
| `python/llaisys/models/qwen2.py` | **修改** — 支持批量推理、绑定不同 KV Cache |

### C++ 后端
| 文件 | 修改内容 |
|------|----------|
| [`src/models/qwen2/qwen2.cpp`](file:///c:/Code/LLAISYS/llaisys/src/models/qwen2/qwen2.cpp) | **修改** — 支持批量推理（batch infer）、多个独立的 KV Cache |
| [`include/llaisys/models/qwen2.h`](file:///c:/Code/LLAISYS/llaisys/include/llaisys/models/qwen2.h) | 可能需要**扩展** API（批量推理接口、KV Cache 管理） |
| `src/ops/linear/cpu/linear_cpu.cpp` / nvidia | **修改** — 支持批量矩阵乘法（3D 输入） |

## 需更改的配置
- **Python 依赖**：添加 `asyncio` 相关支持、可能需要 `redis` 或消息队列用于请求队列持久化
- **无需更改** xmake 配置（除非 operator 需要支持批量计算的新实现）

## 建议阅读层次

> 参考 [遍历式阅读顺序](../thinking/遍历式阅读顺序.md)

| 层次 | 内容 | 必读文件 |
|:--:|------|------|
| **第 7 层** | Tensor | [src/tensor/tensor.hpp](file:///c:/Code/LLAISYS/llaisys/src/tensor/tensor.hpp) — 理解 Tensor 共享 storage 机制，多个请求的 KV Cache 需要独立 storage |
| **第 8 层** | 算子 | [src/ops/self_attention/op.cpp](file:///c:/Code/LLAISYS/llaisys/src/ops/self_attention/op.cpp) — 批量推理时 attention 需处理多个不等长序列的 mask |
| **第 9 层** | C API | [src/llaisys/models/](file:///c:/Code/LLAISYS/llaisys/src/llaisys/) — 可能需要新增批量推理 C API |
| **第 10 层** | **Python（本任务）** | `python/llaisys/request_queue.py` — **新建**：请求排队管理 |
| | | `python/llaisys/batch_scheduler.py` — **新建**：连续批处理调度器（核心算法） |
| | | `python/llaisys/kv_cache_pool.py` — **新建**：KV-Cache 池管理 / 前缀复用 |
| | | [python/llaisys/server.py](file:///c:/Code/LLAISYS/llaisys/python/llaisys/server.py) — 在 Project 3 server 基础上改造为异步并发 |
| | | [python/llaisys/models/qwen2.py](file:///c:/Code/LLAISYS/llaisys/python/llaisys/models/qwen2.py) — 支持批量推理、绑定不同 KV Cache |
| **第 12 层** | 测试 | [test/test_infer.py](file:///c:/Code/LLAISYS/llaisys/test/test_infer.py) — 可扩展为多请求并发测试 |

> 本任务以 **第 10 层 Python 调度逻辑**为核心，C++ 侧（第 7-8 层）需要支持批量推理和独立 KV Cache。建议先完成 Project 2（CUDA）+ Project 3（Chatbot Server）。

---

## 背景知识

### 为什么需要连续批处理（Continuous Batching）
- 传统的静态批处理：等所有请求的生成都结束 → 组新批次 → 开始下一轮
  - 问题：短请求要等长请求结束，GPU 利用率低
- 连续批处理：每迭代一次（生成一个 token）就重新组批次
  - 新请求可以随时加入
  - 已完成的请求移出批次
  - 大幅提升吞吐量

### 为什么需要 KV-Cache Pool
- 每个请求有独立的 KV Cache
- 多个请求可能共享相同的前缀（如相同的 system prompt）
- 前缀匹配复用避免重复计算

---

## 任务清单

### 阶段一：多用户并发请求处理

#### 任务 4.1：设计请求队列

- [ ] **设计请求数据结构**
  ```python
  class InferenceRequest:
      request_id: str
      messages: list[dict]          # 对话历史
      params: dict                   # temperature, top_p, max_tokens 等
      state: str                     # "pending" | "running" | "done"
      generated_tokens: list[int]    # 已生成的 tokens
      kv_cache_id: str              # 关联的 KV Cache ID
      created_at: float
  ```

- [ ] **实现请求队列**
  - 线程安全的队列（`queue.Queue` 或 `asyncio.Queue`）
  - API 端点接收请求后立即加入队列并返回（可采用轮询模式或 WebSocket）
  - 支持查询请求状态

#### 任务 4.2：实现异步 API 端点

- [ ] **非阻塞的 `/v1/chat/completions`**
  - 接收请求 → 创建 `InferenceRequest` → 加入队列 → 立即返回 `request_id`
  - 客户端通过 `/v1/requests/{request_id}` 轮询结果

- [ ] **（推荐）WebSocket 或 SSE 推送结果**
  - 建立长连接
  - 推理线程生成 token 后推送给客户端
  - 更接近真实生产环境

---

### 阶段二：连续批处理（Continuous Batching）

#### 任务 4.3：设计批处理调度器

- [ ] **创建 `batch_scheduler.py`**
  - 循环运行的独立线程/协程
  - 每轮迭代：
    1. 从队列中取新的 pending 请求
    2. 与当前 running 的请求组成批次
    3. 执行一次批量推理（每个请求生成一个 token）
    4. 将完成的请求标记为 done，移除出批次
    5. 未完成的请求保留在 running 集合

#### 任务 4.4：实现批量推理

- [ ] **设计批量推理的数据布局**
  - 每个请求可能处于不同的序列长度
  - 使用 padding 使输入序列对齐，或用 `ragged batch` 格式
  - 最简单的方案：padding 到 max_len，统一输入

- [ ] **修改 C++ 后端支持批量推理**
  - 新增 API：`llaisysQwen2ModelBatchInfer(models, batch_inputs, ...)`
  - 或通过现有的 `llaisysQwen2ModelInfer` 逐模型调用（更简单但性能差）
  - 批量 linear：输入 shape 变为 `[batch_size, seq_len, hidden_dim]`

- [ ] **批量算子支持**
  - Linear：批量 GEMM `[B, M, K] @ [K, N] → [B, M, N]`
  - Attention：每个请求独立的 KV Cache 和因果掩码
  - 其他逐元素算子自动支持批量（增加 batch 维度即可）

#### 任务 4.5：优化批次组批策略

- [ ] **最大化批次利用率**
  - 优先加入序列较短的请求
  - 避免一个超长序列拖慢整个批次
  - 可设置 `max_batch_tokens`（批次内总 token 数上限）

---

### 阶段三：KV-Cache 池管理

#### 任务 4.6：设计 KV-Cache 数据结构

- [ ] **KV-Cache 池**
  - 预分配一组 KV Cache 块（block）
  - 每个请求按需申请若干块
  - 块大小：如每块 = 16 个 token 位置

- [ ] **Radix Tree / Trie 前缀管理**
  - 将每个请求的 token 序列插入 Radix Tree
  - 相同前缀的请求共享 KV Cache
  - 新请求先查找最长公共前缀

#### 任务 4.7：实现前缀匹配复用

- [ ] **查找共享前缀**
  - 新请求进入时，在 Radix Tree 中查找最长匹配前缀
  - 如果找到匹配，直接复用该部分的 KV Cache
  - 只计算新部分（prefill 只处理新 tokens）

- [ ] **引用计数和释放**
  - 每个 KV Cache 块有引用计数
  - 请求完成时减少引用计数
  - 引用计数归零时释放块

---

### 阶段四：测试与评估

#### 任务 4.8：编写并发测试

- [ ] **模拟多用户并发请求**
  - 同时发送 10+ 个请求
  - 验证所有请求都能正确完成
  - 测试不同长度的请求混合场景

#### 任务 4.9：性能评估

- [ ] **测量吞吐量指标**
  - Tokens per second (TPS)
  - Requests per second (RPS)
  - Time to first token (TTFT)
  - 对比单请求串行 vs 批处理的加速比

---

## 架构总览

```
┌─────────────────────────────────────────────────────────┐
│                    HTTP API Server                       │
│           (FastAPI + Uvicorn, async)                     │
│  POST /v1/chat/completions  →  加入请求队列              │
│  GET  /v1/requests/{id}     →  查询请求状态              │
│  WS   /v1/ws                 →  WebSocket 实时推送       │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────┐
│                   Request Queue                          │
│              (asyncio.Queue, 线程安全)                   │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────┐
│                Batch Scheduler                           │
│            (独立推理线程，循环调度)                        │
│  每轮: 取新请求 → 组批次 → 批量推理 → 标记完成           │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────┐
│                KV-Cache Pool                             │
│          (Radix Tree 管理，前缀匹配复用)                   │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────┐
│              LLAISYS C++ Backend                         │
│          (批量推理 + 设备计算)                             │
└─────────────────────────────────────────────────────────┘
```

## 关键难点提示

### 连续批处理
- 核心挑战：每个请求序列长度不同，如何高效组批
- 方案1：Padding → 简单但有浪费
- 方案2：Ragged batch（可变长度）→ 高效但实现复杂
- 推荐：先用 Padding 方案实现，再优化

### KV Cache 管理
- GPU 显存有限，需要合理管理 Cache 块
- Eviction 策略：LRU（最近最少使用）或 FIFO
- 前缀匹配可以大幅减少 Prefill 开销

### 并发安全
- LLAISYS 的 Context 是线程局部的
- 需要在推理线程中正确管理设备上下文
- 考虑使用一个推理线程绑定一个 GPU