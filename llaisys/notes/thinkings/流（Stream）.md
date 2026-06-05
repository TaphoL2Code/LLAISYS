# 流（Stream）详解

## 什么是流

流（Stream）是一种**对数据序列的抽象**，它将数据的产生、传输、转换和消费建模为连贯的流动过程。核心思想是：**不需要一次性拥有全部数据，而是按需、逐步地处理数据**。

流的关键特征：
- **顺序性**：数据按 FIFO 顺序流动
- **连续性**：数据可以无限长（如网络连接、传感器读数）
- **惰性求值**：数据在被消费时才产生
- **背压（Backpressure）**：消费者可以控制生产者的速率

## 不同领域的流

### 一、C/C++ 标准 I/O 流

C++ 的 `iostream` 库是最经典的流实现：

```cpp
#include <iostream>
#include <fstream>
#include <sstream>

// 标准流对象
std::cin   >> x;     // 标准输入流
std::cout  << x;     // 标准输出流
std::cerr  << x;     // 标准错误流（无缓冲）
std::clog  << x;     // 标准日志流（有缓冲）

// 文件流
std::ifstream in("input.txt");    // 输入文件流
std::ofstream out("output.txt");  // 输出文件流
std::fstream   io("data.bin", std::ios::binary | std::ios::in | std::ios::out);

// 字符串流
std::stringstream ss;
ss << "Hello " << 42;
std::string s = ss.str();  // "Hello 42"

// 自定义 streambuf → 控制流的底层行为
class hexdump_streambuf : public std::streambuf {
    // 重写 overflow() 实现十六进制转储
};
```

**设计模式**：流通过 `operator<<` 和 `operator>>` 实现类型安全的格式化 I/O，通过 `streambuf` 解耦数据源与格式化逻辑。

### 二、CUDA 流（Stream）

CUDA Stream 是 GPU 上的**命令队列**，用于管理异步操作的执行顺序：

```cpp
cudaStream_t stream;
cudaStreamCreate(&stream);

// 两个独立的流可以并行执行
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

// 异步内存拷贝
cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream1);
cudaMemcpyAsync(dst2, src2, size, cudaMemcpyHostToDevice, stream2);

// 异步 kernel 启动
kernel<<<grid, block, 0, stream1>>>(dst, ...);
kernel<<<grid, block, 0, stream2>>>(dst2, ...);

// 同步
cudaStreamSynchronize(stream1);
cudaStreamDestroy(stream1);
```

LLaiSys 中的 CUDA Stream 使用：

```cpp
// src/core/llaisys_core.hpp — 简化的异步执行模型
// 每个 device 维护一个 stream，所有 kernel 在同一 stream 中顺序执行
// 保证操作顺序，同时与 host 代码异步
```

**CUDA Stream 的核心价值**：
- 数据传输与 kernel 执行重叠（H2D copy + kernel 并行）
- 多个 kernel 在不同 stream 中并行执行
- 事件（cudaEvent）实现跨 stream 同步

### 三、网络流（TCP Stream / HTTP Streaming）

#### TCP 字节流

TCP 本身就是**字节流**协议——没有消息边界，数据作为连续的字节序列传输：

```python
import socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(("localhost", 8080))
sock.send(b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n")
response = b""
while True:
    chunk = sock.recv(4096)  # 流式接收
    if not chunk:
        break
    response += chunk
```

#### HTTP 流式响应（Server-Sent Events / Chunked Transfer）

LLaiSys Chatbot 中的流式输出：

```python
# server.py — FastAPI 流式响应
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    async def generate():
        for token in model.generate_stream(inputs, ...):
            chunk = {
                "choices": [{"delta": {"content": token_text}, "index": 0}],
                "object": "chat.completion.chunk"
            }
            yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )
```

**关键优势**：首 token 延迟极低（TTFT），用户立即看到回复，无需等待完整生成。

### 四、编程语言中的流式迭代器

#### Python Generator / Iterator

```python
# Generator — 惰性生成序列
def fibonacci():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

# 按需消费，不占用无限内存
fib = fibonacci()
for _ in range(10):
    print(next(fib))  # 0, 1, 1, 2, 3, 5, 8, 13, 21, 34

# 流式处理管线
from itertools import islice
squares = (x**2 for x in fibonacci())          # 平方流
even_squares = (x for x in squares if x % 2 == 0)  # 过滤流
first_5 = list(islice(even_squares, 5))            # 消费 5 个
```

LLaiSys 中的使用：

```python
# models/llama.py — 流式 token 生成
def generate_stream(self, inputs, max_new_tokens=128, temperature=0.8, ...):
    for _ in range(max_new_tokens):
        # 每次只计算一个 token
        logits = self._forward(...)
        next_token = sample(logits, temperature, top_k, top_p)
        yield next_token  # 立即 yield，不等待后续 token
        if next_token == self._end_token:
            break
```

#### Java Stream API

```java
List<String> result = names.stream()
    .filter(n -> n.startsWith("A"))
    .map(String::toUpperCase)
    .sorted()
    .limit(5)
    .collect(Collectors.toList());
// 惰性求值：直到 collect() 才实际执行
```

#### Rust Iterator

```rust
let sum: u64 = (0..)
    .map(|x| x * x)
    .filter(|x| x % 2 == 0)
    .take(10)
    .sum();  // 惰性：take(10) 后才开始计算
```

### 五、Node.js Stream

Node.js 将流作为一等公民，四种基础类型：

```javascript
const { Readable, Writable, Transform, Duplex } = require('stream');

// Readable — 数据源
const readable = new Readable({
    read(size) {
        this.push(String(Math.random()));
    }
});

// Writable — 数据目的地
const writable = fs.createWriteStream('output.txt');

// Transform — 转换流
const uppercase = new Transform({
    transform(chunk, encoding, callback) {
        callback(null, chunk.toString().toUpperCase());
    }
});

// Duplex — 双向流（如 TCP socket）
// Pipeline 连接
readable.pipe(uppercase).pipe(writable);
```

### 六、响应式编程中的流（RxJS / Reactive Streams）

```javascript
// RxJS — 事件流
const clickStream = fromEvent(document, 'click');
const throttled = clickStream.pipe(
    throttleTime(300),
    map(event => ({ x: event.clientX, y: event.clientY }))
);
throttled.subscribe(pos => console.log(pos));
```

## 流 vs 批处理

| 维度 | 批处理 | 流处理 |
|------|--------|--------|
| 内存占用 | O(n) 全部数据 | O(1) 当前窗口 |
| 延迟 | 处理完所有数据才出结果 | 逐项立即产出 |
| 适用场景 | 离线分析、训练 | 在线推理、实时监控 |
| 可组合性 | 通常需要中间结果物化 | 链式 pipe 无中间物化 |
| 错误处理 | 一次失败全部重来 | 可逐项恢复 |

## 流的核心设计模式

### 1. 生产者-消费者解耦

```python
# 流作为生产者和消费者之间的缓冲抽象
def producer(stream):
    for item in data_source:
        stream.put(item)   # 生产
    stream.put(SENTINEL)   # 结束信号

def consumer(stream):
    while True:
        item = stream.get()  # 消费
        if item == SENTINEL:
            break
        process(item)
```

### 2. 管线（Pipeline）/ 过滤器链

```python
# 每个阶段是独立的 Transform，通过 pipe 连接
data_source
    → filter(invalid)     # 过滤
    → map(normalize)      # 变换
    → batch(32)           # 组批
    → model.predict()     # 推理
    → aggregate(results)  # 聚合
```

### 3. 背压（Backpressure）

```python
import asyncio

async def producer(queue: asyncio.Queue):
    for i in range(1000):
        await queue.put(i)  # 如果队列满，自动等待
    await queue.put(None)   # 结束信号

async def consumer(queue: asyncio.Queue):
    while True:
        item = await queue.get()
        if item is None:
            break
        await slow_process(item)  # 慢消费者
        queue.task_done()
```

## LLaiSys 中的流应用

| 场景 | 流类型 | 实现 |
|------|--------|------|
| CUDA 异步执行 | CUDA Stream | `src/core/` — 每个 device 一个 stream |
| Token 生成 | Python Generator | `generate_stream()` — 逐 token yield |
| HTTP 响应 | SSE (Server-Sent Events) | `server.py` — `StreamingResponse` |
| 模型推理 | 惰性序列 | KV-Cache 复用避免重复计算 |
| 数据加载 | 流式预处理 | 边读边 tokenize，不阻塞推理 |

## 总结

流的核心思想是**将数据从静态的"集合"变为动态的"过程"**：

1. **C/C++ I/O 流** — 统一的格式化输入输出抽象
2. **CUDA 流** — GPU 异步命令队列，实现计算与传输重叠
3. **网络流** — 字节序列的连续传输（TCP、HTTP SSE）
4. **语言迭代器** — 惰性序列，无限数据处理
5. **响应式流** — 事件驱动的异步数据流

共同的设计哲学：**不要等数据，让数据流过来**。