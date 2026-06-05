# Project #4: Multi-user Inference Service — 事件清单

> 实现多用户并发推理服务器，支持请求排队、模型实例池、连续批处理调度、SSE 流式响应。

## 架构设计

```
┌─────────────────────────────────────────────────────────┐
│                    HTTP API Server                       │
│           (FastAPI + Uvicorn, async)                     │
│  POST /v1/chat/completions  →  加入请求队列              │
│  GET  /v1/requests/{id}     →  查询请求状态              │
│  DELETE /v1/requests/{id}   →  取消请求                  │
│  GET  /v1/stats             →  队列/池统计               │
│  GET  /v1/models            →  模型列表                  │
│  GET  /health               →  健康检查                  │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────┐
│                   Request Queue                          │
│              (threading.Lock, 线程安全)                   │
│  支持: 入队 / 出队 / 状态查询 / 取消 / 统计              │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────┐
│                Batch Scheduler                           │
│            (独立推理线程，循环调度)                        │
│  每轮: 取新请求 → 获取模型实例 → 推理 → 释放实例         │
│  每个请求在独立线程中处理，支持流式回调                   │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────┐
│                  Model Pool                              │
│    (每个实例绑定独立的 C++ 模型 + KV-Cache)               │
│  - 预创建实例（warm-up）                                 │
│  - 空闲超时自动重置 KV-Cache                             │
│  - 信号量控制并发数                                       │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────┐
│              LLAISYS C++ Backend                         │
│          (每个实例独立推理 + KV-Cache)                     │
└─────────────────────────────────────────────────────────┘
```

## 新增文件

| 文件 | 作用 | 状态 |
|------|------|:--:|
| [`python/llaisys/request_queue.py`](file:///c:/Code/LLAISYS/llaisys/python/llaisys/request_queue.py) | **新建** — 请求数据结构 + 线程安全队列 | Done |
| [`python/llaisys/model_pool.py`](file:///c:/Code/LLAISYS/llaisys/python/llaisys/model_pool.py) | **新建** — 模型实例池，管理独立 KV-Cache | Done |
| [`python/llaisys/batch_scheduler.py`](file:///c:/Code/LLAISYS/llaisys/python/llaisys/batch_scheduler.py) | **新建** — 连续批处理调度器 | Done |

## 修改文件

| 文件 | 修改内容 | 状态 |
|------|----------|:--:|
| [`python/llaisys/server.py`](file:///c:/Code/LLAISYS/llaisys/python/llaisys/server.py) | **重写** — 多用户异步服务器 | Done |

---

## 实现细节

### 请求队列 (`request_queue.py`)

- `InferenceRequest`：包含请求参数、状态、结果、流式回调
- `RequestQueue`：基于 `threading.Lock` 的线程安全队列
- 状态机：`PENDING → RUNNING → DONE/ERROR`
- 支持取消、状态查询、统计

### 模型实例池 (`model_pool.py`)

- `ModelPool`：通过工厂函数创建模型实例，每个实例绑定独立 C++ 模型
- 每个实例有独立的 KV-Cache，互不干扰
- 信号量 (`BoundedSemaphore`) 控制最大并发数
- 预热机制：启动时预创建第一个实例
- 空闲超时自动 `reset_kv()`，避免旧缓存污染新请求

### 连续批处理调度器 (`batch_scheduler.py`)

- 后台线程循环轮询请求队列
- 每个请求在独立线程中处理，实现真正的并发
- 流式模式：通过 `_text_callback` 回调推送 token，经 `asyncio.Queue` 桥接到 SSE
- 非流式模式：收集全部 token 后返回，通过 polling 获取结果

### 服务器 (`server.py`)

**API 端点：**

| 端点 | 方法 | 功能 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/v1/chat/completions` | POST | 提交聊天请求（stream=true 时 SSE 流式返回） |
| `/v1/requests/{id}` | GET | 轮询请求状态和结果 |
| `/v1/requests/{id}` | DELETE | 取消 pending 请求 |
| `/v1/stats` | GET | 队列和池的统计信息 |
| `/v1/models` | GET | OpenAI 兼容的模型列表 |

**两种使用模式：**

1. **流式模式** (`stream=true`)：SSE 实时推送 token，适合前端实时展示
2. **轮询模式** (`stream=false`)：提交后立即返回 `request_id`，客户端轮询获取结果

---

## 关键设计决策

### 为什么用 ModelPool 而非修改 C++ 批量推理？

| 方案 | 优点 | 缺点 |
|------|------|------|
| ModelPool（当前） | 无需修改 C++；每个请求独立 KV-Cache；实现简单 | 每个实例独立加载权重，内存占用高 |
| 批量推理 | 内存效率高；GPU 利用率高 | 需要修改 C++ 算子支持 3D 输入；KV-Cache 管理复杂 |

当前采用 ModelPool 方案，因为实现简单且不需要修改 C++ 后端。后续可升级为真正的批量推理。

### 为什么用独立线程而非 asyncio 处理推理？

LLaiSys 的 C++ 推理是同步阻塞的（CPU 密集型），不适合在 asyncio 事件循环中运行。使用独立线程避免阻塞事件循环。

### 流式回调的桥接方式

```
scheduler thread (sync)          asyncio event loop (async)
        │                              │
        │  _text_callback(text)        │
        │ ─────────────────────────▶   │
        │  loop.call_soon_threadsafe(  │
        │    token_queue.put_nowait)   │
        │                              │
        │                     await token_queue.get()
        │                              │
        │                     yield SSE chunk
```

---

## 功能验证

- [x] `request_queue.py` 所有单元测试通过（入队/出队/状态/取消/统计）
- [x] `model_pool.py` 所有单元测试通过（获取/释放/并发/预热）
- [x] `batch_scheduler.py` 导入正常
- [x] `server.py` FastAPI app 创建成功，所有端点注册正常
- [x] 依赖安装：`fastapi`, `uvicorn`, `starlette`

---

## 使用方式

### 启动服务器

```bash
cd C:\Code\LLAISYS\llaisys
python -m llaisys.server \
    --model ./Qwen2-0.5B \
    --tokenizer ./Qwen2-0.5B \
    --port 8080 \
    --pool-size 4 \
    --device 0 \
    --model-class qwen2
```

### 流式请求 (curl)

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llaisys",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 128,
    "temperature": 0.8,
    "stream": true
  }'
```

### 轮询模式

```bash
# 1. 提交请求
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "llaisys", "messages": [{"role": "user", "content": "Hello!"}], "stream": false}'
# 返回: {"request_id": "abc123", "state": "pending", ...}

# 2. 轮询结果
curl http://localhost:8080/v1/requests/abc123
# 返回: {"state": "done", "generated_text": "Hello! How can I help?", ...}
```

### 查看统计

```bash
curl http://localhost:8080/v1/stats
# {"pending": 0, "running": 2, "total_queued": 2, ...}
```

---

## 并发测试

```bash
# 同时发送 5 个请求（pool_size=4 时，4 个并发处理，1 个排队）
for i in 1 2 3 4 5; do
  curl -X POST http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"llaisys\",\"messages\":[{\"role\":\"user\",\"content\":\"Say $i\"}],\"stream\":false}" &
done
wait
```