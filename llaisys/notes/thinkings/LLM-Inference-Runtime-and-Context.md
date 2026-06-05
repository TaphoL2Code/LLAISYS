# 大模型推理中的 Runtime 与 Context 详解

## 概述

在大模型推理系统（如 LLaiSys、vLLM、llama.cpp）中，"Runtime"和"Context"是两个核心但容易混淆的概念。它们出现在不同层面，含义各有侧重：

| 层面 | Runtime | Context |
|------|---------|---------|
| **系统层** | 设备运行时：GPU/CPU 驱动接口的抽象 | 设备上下文：当前激活的设备 + 其状态 |
| **模型层** | — | 推理上下文：KV-Cache、序列长度、会话状态 |
| **框架层** | 推理引擎：ONNX Runtime、TensorRT、llama.cpp | 执行上下文：内存分配器、线程池、CUDA Stream |

---

## 一、系统层 Runtime：设备运行时抽象

### 1.1 概念

**Runtime** 是推理框架对底层硬件（GPU/CPU）驱动 API 的**统一封装**。它解决的问题是：不同硬件（NVIDIA GPU、AMD GPU、CPU）有不同的 API，但框架需要一套统一的接口来管理设备内存、启动计算、同步数据。

### 1.2 LLaiSys 中的实现

以 LLaiSys 为例，Runtime 的核心是一个**函数指针结构体**：

```c
// include/llaisys/runtime.h — Runtime API 定义
struct LlaisysRuntimeAPI {
    // 设备管理
    get_device_count_api    get_device_count;     // 获取可用设备数量
    set_device_api          set_device;           // 设置当前设备
    device_synchronize_api  device_synchronize;   // 同步设备

    // 流管理
    create_stream_api       create_stream;        // 创建执行流
    destroy_stream_api      destroy_stream;       // 销毁执行流
    stream_synchronize_api  stream_synchronize;   // 同步流

    // 内存管理
    malloc_device_api       malloc_device;        // 分配设备内存（GPU显存）
    free_device_api         free_device;          // 释放设备内存
    malloc_host_api         malloc_host;          // 分配页锁定主机内存
    free_host_api           free_host;            // 释放页锁定主机内存

    // 数据传输
    memcpy_sync_api         memcpy_sync;          // 同步内存拷贝
    memcpy_async_api        memcpy_async;         // 异步内存拷贝（通过流）
};
```

**为什么需要这个抽象？** — 看一个具体对比：

```cpp
// 没有 Runtime 抽象时，每个算子必须写两套代码：
#ifdef CUDA
    cudaMalloc(&ptr, size);      // CUDA API
#else
    ptr = aligned_alloc(64, size); // CPU API
#endif

// 有了 Runtime 抽象后，一套代码通用：
auto* api = runtime.api();
ptr = api->malloc_device(size);  // 统一接口，底层自动分发
```

### 1.3 Runtime 的完整生命周期

LLaiSys 中 `Runtime` 类封装了完整的设备运行时：

```cpp
// src/core/runtime/runtime.hpp, runtime.cpp
class Runtime {
    llaisysDeviceType_t _device_type;   // CPU / NVIDIA
    int _device_id;                     // 第几张卡
    const LlaisysRuntimeAPI *_api;     // 函数指针表
    MemoryAllocator *_allocator;       // 内存分配器
    llaisysStream_t _stream;           // 默认执行流
    bool _is_active;                   // 是否激活

    void _activate() {
        _api->set_device(_device_id);  // cudaSetDevice(0) 或 CPU 空操作
        _is_active = true;
    }
};
```

**一个 Runtime 实例 = 一个设备（如 GPU 0）的完整操作环境**，包括：
- 可以调用哪些 API（`_api`）
- 内存从哪里分配（`_allocator`）
- 任务在哪个流上执行（`_stream`）

### 1.4 类比理解

| 概念 | 类比 |
|------|------|
| `LlaisysRuntimeAPI` | 汽车的操作手册（方向盘、油门、刹车的位置） |
| `Runtime` | 一辆具体的汽车（已启动、挂好档、在某个设备上） |
| `Runtime::_activate()` | 坐到驾驶座上，握住方向盘 |
| `Runtime::_deactivate()` | 下车，换另一辆车（切换设备） |

---

## 二、系统层 Context：设备上下文管理器

### 2.1 概念

**Context** 管理**所有设备的 Runtime**，并提供"当前活动设备"的切换能力。如果 Runtime 是汽车的实例，Context 就是停车场管理员——知道所有车在哪，告诉你现在该开哪一辆。

### 2.2 LLaiSys 中的实现

```cpp
// src/core/context/context.hpp, context.cpp
class Context {
    // 所有 Runtime 的注册表：{设备类型 → [Runtime列表]}
    unordered_map<llaisysDeviceType_t, vector<Runtime*>> _runtime_map;
    Runtime *_current_runtime;  // 当前激活的 Runtime

    Context() {
        // 初始化时：枚举所有设备类型，创建 Runtime
        // 优先激活 GPU，没有 GPU 则用 CPU
        for (auto device_type : [NVIDIA, CPU]) {
            int count = api->get_device_count();
            for (int i = 0; i < count; i++) {
                if (_current_runtime == nullptr) {
                    auto runtime = new Runtime(device_type, i);
                    runtime->_activate();
                    _current_runtime = runtime;
                }
            }
        }
    }

    void setDevice(device_type, device_id) {
        // 切换当前设备：先 deactivate 旧的，再 activate 新的
        _current_runtime->_deactivate();
        _runtime_map[device_type][device_id]->_activate();
        _current_runtime = _runtime_map[device_type][device_id];
    }

    Runtime& runtime() {
        return *_current_runtime;  // 获取当前活动 Runtime
    }
};
```

### 2.3 线程局部 Context

LLaiSys 中 Context 是 **thread-local** 的：

```cpp
// src/core/context/context.cpp
Context& context() {
    thread_local Context thread_context;  // 每个线程独立
    return thread_context;
}
```

这意味着：
- 线程 A 可以操作 GPU 0，线程 B 同时操作 GPU 1，互不干扰
- 每个线程有自己的 `_current_runtime` 和内存分配器

### 2.4 Context 的核心职责

```
Context 对象
├── _runtime_map           → 管理所有 Runtime
│   ├── [NVIDIA]           → [GPU0 Runtime, GPU1 Runtime, ...]
│   └── [CPU]              → [CPU0 Runtime]
│
├── _current_runtime       → 当前激活的 Runtime（全局通过 context().runtime() 访问）
│   ├── api()              → 获取当前设备的 API 函数表
│   ├── allocateDeviceStorage() → 在当前设备上分配内存
│   ├── stream()           → 当前设备的执行流
│   └── synchronize()      → 等待当前设备所有操作完成
│
└── setDevice()            → 切换当前设备
```

---

## 三、大模型推理中的"上下文"（Inference Context / KV-Cache Context）

### 3.1 概念

在 LLM 推理中，"上下文"有另一层含义：**模型的推理状态**。每次推理不仅仅是"输入 → 输出"，还涉及：

- **当前序列长度**（`_cur_seq_len`）：已经处理了多少 token
- **KV-Cache**：之前所有 token 的 Key 和 Value 张量缓存
- **位置编码状态**：当前 token 的位置 ID

### 3.2 LLaiSys 中的实现

```cpp
// src/models/llama/llama.hpp — 推理上下文嵌入在模型对象中
class LlamaModel {
    // 模型架构参数（不变的）
    LlaisysLlamaMeta _meta;

    // 推理上下文（随会话变化的）
    vector<tensor_t> _k_cache;   // 每层的 Key 缓存 [maxseq, nkvh, dh]
    vector<tensor_t> _v_cache;   // 每层的 Value 缓存 [maxseq, nkvh, dh]
    size_t _cur_seq_len = 0;     // 当前已处理的序列长度
};
```

**推理上下文的生命周期**：

```
第一次推理（prefill）：
    输入："你好，请"
    _cur_seq_len = 0 → 3
    _k_cache[0..nlayer] = 前3个token的K值
    _v_cache[0..nlayer] = 前3个token的V值

第二次推理（decode）：
    输入：(复用KV-Cache) + "问"
    _cur_seq_len = 3 → 4
    _k_cache[0..nlayer][3] = 第4个token的K值（追加）
    _v_cache[0..nlayer][3] = 第4个token的V值（追加）

重置上下文（新对话）：
    reset_kv_cache()
    _cur_seq_len = 0
```

### 3.3 为什么 KV-Cache 是"上下文"

每生成一个新 token，模型需要 attend 到**所有之前的 token**。如果没有 KV-Cache，每次都要重新计算所有历史 token 的 K 和 V，计算量是 O(n²)。有了 KV-Cache，每次只需计算 O(n)：

```
没有 KV-Cache：
    Token 1: 计算 K1, V1, Attention(Q1, [K1], [V1])
    Token 2: 计算 K1, K2, V1, V2, Attention(Q2, [K1,K2], [V1,V2])  ← K1/V1 重复计算！
    Token 3: 计算 K1, K2, K3, V1, V2, V3, ...                      ← K1-K2/V1-V2 重复计算！

有 KV-Cache：
    Token 1: 计算 K1, V1 → 存入缓存, Attention(Q1, cache_K, cache_V)
    Token 2: 计算 K2, V2 → 追加到缓存, Attention(Q2, cache_K[0:2], cache_V[0:2])
    Token 3: 计算 K3, V3 → 追加到缓存, Attention(Q3, cache_K[0:3], cache_V[0:3])
```

### 3.4 多轮对话中的上下文管理

```python
# 多轮对话示例
model = Llama("Llama-3.2-1B")

# 第一轮
response1 = model.chat([{"role": "user", "content": "1+1=?"}])
# 此时 _cur_seq_len = N（整轮对话的 token 数）

# 第二轮：复用之前的 KV-Cache
response2 = model.chat([{"role": "user", "content": "那 2+2 呢？"}])
# _cur_seq_len 从 N 继续增长，历史 token 已缓存

# 开始新对话：清空上下文
model.reset_kv()
# _cur_seq_len = 0，KV-Cache 清空
```

---

## 四、两个"Context"的对比

| 维度 | 设备上下文（Device Context） | 推理上下文（Inference Context） |
|------|---------------------------|-------------------------------|
| 所在层 | 系统层（`core/context/`） | 模型层（`models/llama/`） |
| 管理者 | `Context` 类 | `LlamaModel` 类 |
| 生命周期 | 进程级，thread-local | 会话级，随 `reset_kv()` 重置 |
| 核心数据 | 当前激活的 Runtime（设备） | KV-Cache + 序列长度 |
| 切换方式 | `context().setDevice(GPU, 1)` | `model.reset_kv()` |
| 数量 | 每个线程一个 Context | 每个模型实例一个推理上下文 |
| 用途 | 确定"在哪个设备上算" | 确定"已经算到哪了" |

---

## 五、完整的数据流

```
用户调用 model.generate(["你", "好"])
    │
    ▼
Python Llama.generate()
    │ 调用 C++ infer()
    ▼
C++ LlamaModel::infer(token_ids=[105, 206], ntoken=2)
    │
    ├── 获取当前 Runtime： context().runtime()
    │       │
    │       ├── api() → 获取 cudaMalloc / cudaMemcpy 等函数指针
    │       ├── stream() → 获取 CUDA Stream
    │       └── allocateDeviceStorage() → 分配 GPU 显存
    │
    ├── 检查推理上下文：
    │       │
    │       ├── _cur_seq_len == 0 → Prefill 模式
    │       │   ├── 对全部 tokens 做 embedding
    │       │   ├── 全部 transformer layers 前向
    │       │   └── KV-Cache 写入 0..ntoken-1 位置
    │       │
    │       └── _cur_seq_len > 0 → Decode 模式
    │           ├── 只对新 token 做 embedding
    │           ├── 单层前向 + 从 KV-Cache 读取历史
    │           └── KV-Cache 追加写入 _cur_seq_len 位置
    │
    └── 返回 next_token
```

---

## 六、业界框架中的 Runtime/Context 对比

| 框架 | Runtime 概念 | Context 概念 |
|------|-------------|-------------|
| **LLaiSys** | `Runtime` 类 + `LlaisysRuntimeAPI` 函数表 | 设备 Context（`Context` 类）+ 推理 Context（KV-Cache） |
| **ONNX Runtime** | `OrtSession` — 封装推理会话 | `OrtMemoryInfo` — 内存分配器上下文 |
| **TensorRT** | `nvinfer1::IRuntime` — 反序列化引擎 | `IExecutionContext` — 绑定输入/输出、设置 batch size |
| **vLLM** | `Worker` — 管理 GPU 和模型实例 | `SequenceGroup` — 管理一个请求的 KV-Cache 和生成状态 |
| **llama.cpp** | `ggml_backend` — 后端抽象（CUDA/Metal/CPU） | `llama_context` — 包含 KV-Cache + 计算图缓冲区 |
| **PyTorch** | `torch.cuda` — CUDA 设备管理 | `torch.cuda.stream()` — 当前 CUDA 流 |

### llama.cpp 中的典型用法

```cpp
// 1. 创建 Runtime（后端）
ggml_backend_t backend = ggml_backend_cuda_init(0);  // GPU 0

// 2. 创建模型
llama_model *model = llama_model_load_from_file("model.gguf", params);

// 3. 创建推理上下文
llama_context *ctx = llama_context_new(model, params);
// ctx 内部包含：
//   - KV-Cache（所有层的 K/V 缓存）
//   - 计算图缓冲区（临时张量）
//   - 当前序列状态

// 4. 推理（上下文自动管理 KV-Cache）
llama_decode(ctx, batch);  // 等价于 LLaiSys 的 infer()

// 5. 切换会话：清空上下文
llama_kv_cache_clear(ctx);  // 等价于 LLaiSys 的 reset_kv()
```

---

## 七、总结

| 概念 | 一句话总结 |
|------|-----------|
| **Runtime API** | 硬件驱动的函数指针表，回答"这个设备能做什么操作" |
| **Runtime 实例** | 一个具体设备的操作环境，回答"我现在在哪个设备上" |
| **Context（设备）** | 所有 Runtime 的注册表 + 当前设备切换，回答"要换到哪个设备" |
| **Context（推理）** | 模型的运行时状态（KV-Cache + 序列长度），回答"已经算到哪了" |

两者的关系：**设备 Context 决定在"哪里"算，推理 Context 决定"算到哪了"**。每次前向传播时，算子通过 `context().runtime().api()` 找到对应的设备函数，同时通过模型的 `_k_cache` / `_v_cache` 获取历史计算结果。