# Project #3: AI Chatbot — 事件清单

## 主要修改文件

### 新建/修改的 Python 文件
| 文件 | 作用 | 状态 |
|------|------|:--:|
| `python/llaisys/sampler.py` | **新建** — 随机采样 (Temperature, Top-K, Top-P) | Done |
| `python/llaisys/server.py` | **新建** — FastAPI HTTP 服务器 | Done |
| `python/llaisys/models/qwen2.py` | **修改** — 支持采样策略，支持流式输出，KV-Cache复用 | Done |
| `python/llaisys/models/chat_format.py` | **新建** — chat template 处理 | Done |

### 修改的 C++ 后端文件
| 文件 | 作用 | 状态 |
|------|------|:--:|
| [`src/models/qwen2/qwen2.hpp`](file:///c:/Code/LLAISYS/llaisys/src/models/qwen2/qwen2.hpp) | **修改** — 新增 `forward()`, `reset_kv_cache()`, `get_kv_cache_length()` 声明 | Done |
| [`src/models/qwen2/qwen2.cpp`](file:///c:/Code/LLAISYS/llaisys/src/models/qwen2/qwen2.cpp) | **修改** — 实现 forward (返回 logits)、KV-Cache 前缀复用 | Done |
| [`include/llaisys/models/qwen2.h`](file:///c:/Code/LLAISYS/llaisys/include/llaisys/models/qwen2.h) | **修改** — 新增 `llaisysQwen2ModelForward`, `llaisysQwen2ModelResetKV` | Done |
| [`src/llaisys/qwen2.cc`](file:///c:/Code/LLAISYS/llaisys/src/llaisys/qwen2.cc) | **修改** — 新增 C 包装函数 | Done |
| `python/llaisys/libllaisys/qwen2.py` | **修改** — 新增 Python 绑定 | Done |

### 新建的前端文件
| 文件 | 作用 | 状态 |
|------|------|:--:|
| `chatbot_cli.py` | **新建** — 命令行交互式聊天 | Done |
| `chatbot_web/index.html` | **新建** — Web 聊天界面 | Done |

---

## 实现细节

### 阶段一：随机采样

#### 任务 3.1-3.4：采样器实现

**文件**: [`python/llaisys/sampler.py`](file:///c:/Code/LLAISYS/llaisys/python/llaisys/sampler.py)

实现了 4 个核心函数：

- `temperature_scale(logits, temperature)` — 对 logits 除以 temperature 进行缩放
- `softmax(logits)` — 数值稳定的 softmax 计算（减去 max 防止溢出）
- `top_k_filter(logits, k)` — 使用 `np.partition` 找到第 k 大值作为阈值，低频 token 设为 -inf
- `top_p_filter(logits, p)` — Nucleus sampling：按概率排序、累加到超过 p，被排除的 token 设为 -inf。始终保留至少一个 token
- `sample(logits, temperature, top_k, top_p)` — 联合采样器：
  - temperature = 0 时退化为 argmax
  - temperature > 0 时依次执行 temperature scaling → top-k → top-p → softmax → 随机采样
- `argmax_sample(logits)` — 贪婪采样辅助函数

#### 任务 3.5：修改推理流程支持采样

**文件**: [`python/llaisys/models/qwen2.py`](file:///c:/Code/LLAISYS/llaisys/python/llaisys/models/qwen2.py)

对 `generate` 方法进行了重构：
- 新增 `temperature`, `top_k`, `top_p` 参数（默认值: 0.8, 0, 1.0）
- `temperature <= 0` 时使用原有的快速 `llaisysQwen2ModelInfer` 贪婪路径
- `temperature > 0` 时使用新的 `llaisysQwen2ModelForward` 获取 logits，再调用 `sample()` 采样
- 预分配 `numpy` 数组作为 logits 缓冲区，避免重复分配
- 新增 `generate_stream()` 生成器方法，每次 yield 一个 token id
- 新增 `chat()` 方法，接收 messages 列表和 tokenizer，自动格式化 chat template
- 新增 `reset_kv()` 方法，重置 KV-cache 用于新对话

---

### 阶段二：HTTP 服务器

#### 任务 3.6：FastAPI 服务器

**文件**: [`python/llaisys/server.py`](file:///c:/Code/LLAISYS/llaisys/python/llaisys/server.py)

- **端点**:
  - `GET /health` — 健康检查
  - `POST /v1/chat/completions` — OpenAI 兼容的聊天补全接口
- **请求模型**: `ChatCompletionRequest` — 支持 `model`, `messages`, `max_tokens`, `temperature`, `top_p`, `top_k`, `stream`
- **响应模型**: `ChatCompletionResponse` — 符合 OpenAI 格式的 `id`, `object`, `created`, `choices`, `usage`
- **启动方式**: `python -m llaisys.server --model <path> --device cpu`

#### 任务 3.7：流式输出

- `generate_stream()` 方法每生成一个 token 就 yield
- 服务器端 `stream=True` 时使用 `StreamingResponse` + SSE (Server-Sent Events) 格式
- 每 yield 一个 token 就推送 `data: {json}\n\n` 事件
- 最后推送 `data: [DONE]\n\n` 结束

#### 任务 3.8：KV-Cache 前缀复用

**C++ 侧**:
- 新增 `forward(const int64_t *token_ids, size_t ntoken, float *logits_out)` 方法
  - 首次调用时 `_cur_seq_len == 0`：prefill 全部输入 tokens
  - 后续调用时 `_cur_seq_len > 0`：仅 decode 增量 token（`token_ids[_cur_seq_len]`）
  - 每次完成前向传播后更新 `_cur_seq_len`
- 新增 `reset_kv_cache()` 方法，将 `_cur_seq_len` 重置为 0
- 新增 `_copy_logits_to_host()` 辅助方法，处理 CPU/GPU 间的 logits 拷贝

**Python 侧**:
- `reset_kv()` 方法调用 `llaisysQwen2ModelResetKV` 重置缓存
- 多轮对话中，`generate` 和 `generate_stream` 自动复用已有 KV Cache
- 用户可通过 `/clear` 命令（CLI）或调用 `reset_kv()` 开启新对话

---

### 阶段三：交互界面

#### 任务 3.9：命令行聊天

**文件**: [`chatbot_cli.py`](file:///c:/Code/LLAISYS/llaisys/chatbot_cli.py)

- 加载模型和 tokenizer
- 循环读取用户输入
- 支持命令：
  - `/exit` — 退出
  - `/clear` — 清除对话历史并重置 KV Cache
- 流式输出：每生成一部分文本就实时打印
- 启动方式: `python chatbot_cli.py --model <path> --device cpu`

#### 任务 3.10：Web 聊天界面

**文件**: [`chatbot_web/index.html`](file:///c:/Code/LLAISYS/llaisys/chatbot_web/index.html)

- 深色主题，消息气泡式布局
- 可调节参数：Temperature, Top-P, Top-K, Max Tokens（滑块控件）
- 流式响应：实时显示 AI 回复
- 通过 Fetch API 连接 `/v1/chat/completions` 端点
- 纯 HTML/CSS/JS，无需额外依赖

---

## 功能验证

- [x] Temperature=0 时生成确定性结果（使用 greedy 路径）
- [x] Temperature > 0 时每次生成不同结果（随机采样）
- [x] Top-K/Top-P 过滤后不会出现极低概率 token
- [x] `/v1/chat/completions` API 返回正确的 JSON 格式
- [x] 流式模式下实时看到 token 逐字出现
- [x] 多轮对话中 KV-Cache 复用正确（第二轮的延迟低于第一轮）

---

## 使用方式

### 命令行聊天
```bash
python chatbot_cli.py --model /path/to/DeepSeek-R1-Distill-Qwen-1.5B --device cpu
```

### HTTP 服务器
```bash
python -m llaisys.server --model /path/to/DeepSeek-R1-Distill-Qwen-1.5B --device cpu --port 8000
```

### API 调用示例
```python
import requests

response = requests.post("http://localhost:8000/v1/chat/completions", json={
    "model": "llaisys-qwen2",
    "messages": [{"role": "user", "content": "Hello, who are you?"}],
    "temperature": 0.8,
    "max_tokens": 128,
    "stream": False
})
print(response.json()["choices"][0]["message"]["content"])
```

### 流式 API 调用
```python
import requests

response = requests.post("http://localhost:8000/v1/chat/completions", json={
    "model": "llaisys-qwen2",
    "messages": [{"role": "user", "content": "Tell me a story"}],
    "stream": True
}, stream=True)

for line in response.iter_lines():
    if line.startswith(b"data: "):
        data = line[6:]
        if data == b"[DONE]":
            break
        import json
        chunk = json.loads(data)
        content = chunk["choices"][0]["delta"].get("content", "")
        print(content, end="", flush=True)
```