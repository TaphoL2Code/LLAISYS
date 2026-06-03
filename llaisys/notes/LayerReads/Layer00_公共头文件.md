# Layer00_公共头文件

问：*为什么要设计公共头文件？*

答：*构建一个 **跨平台（Windows、Linux、macOS）、跨语言（C、C++、Python）、多设备（CPU、NVIDIA GPU）** 的底层运行时库的基础类型系统*

- [x] ## 第 0 层：公共头文件 — 5 个文件

**先认清项目里流通的所有"数据类型"和"接口契约"。**

| 序号 | 文件                                                         | 行数 | 核心看点                                                     |
| :--: | ------------------------------------------------------------ | :--: | ------------------------------------------------------------ |
| 0.1  | [include/llaisys.h](file:///c:/Code/LLAISYS/llaisys/include/llaisys.h) | 106  | **全部枚举**：`llaisysDataType_t`（11 种 dtype 及对应字节数）、`llaisysDeviceType_t`、`llaisysMemcpyKind_t`（H2H/H2D/D2H/D2D）、`llaisysStatus_t` |
| 0.2  | [include/llaisys/runtime.h](file:///c:/Code/LLAISYS/llaisys/include/llaisys/runtime.h) |  29  | `LlaisysRuntimeAPI` 结构体——所有**设备**操作的函数指针表（malloc/free/memcpy/memset 等） |
| 0.3  | [include/llaisys/tensor.h](file:///c:/Code/LLAISYS/llaisys/include/llaisys/tensor.h) |  37  | **Tensor** 的 C API 声明：`tensorCreate`/`tensorLoad`/`tensorView`/`tensorPermute`/`tensorSlice` 等 |
| 0.4  | [include/llaisys/ops.h](file:///c:/Code/LLAISYS/llaisys/include/llaisys/ops.h) |  18  | 全部**算子**的 C API 声明：`opsAdd`/`opsArgmax`/`opsEmbedding`/`opsLinear`... |
| 0.5  | [include/llaisys/models/qwen2.h](file:///c:/Code/LLAISYS/llaisys/include/llaisys/models/qwen2.h) |  39  | `qwen2Create`/`qwen2Forward`/`qwen2Generate` 等**模型**推理 C API |

- [x] ### 0.1 include/llaisys.h

```
#ifndef __LLAISYS_H__
#define __LLAISYS_H__
...
#endif // __LLAISYS_H__
```

头文件保护符：用于防止同一个头文件被重复包含，引起重复定义的错误

```
#if defined(_WIN32)
#define __export __declspec(dllexport)
#elif defined(__GNUC__) && ((__GNUC__ >= 4) || (__GNUC__ == 3 && __GNUC_MINOR__ >= 3))
#define __export __attribute__((visibility("default")))
#else
#define __export
#endif
```

平台相关的符号导出宏：跨平台地定义 `__export` 宏，用于标记需要从动态库（DLL/so）中导出的函数或变量

**为了实现跨平台（Windows、Linux、macOS）**

```
#ifdef __cplusplus
#define __C extern "C"
#include <cstddef>
#include <cstdint>
#else
#define __C
#include <stddef.h>
#include <stdint.h>
#endif
```

 C/CPP兼容性处理：实现头文件再C/CPP编译器下的兼容性

**为了实现跨语言（C、C++、Python）**

```
// Device Types
typedef enum {
    LLAISYS_DEVICE_CPU = 0,
    //// TODO: Add more device types here. Numbers need to be consecutive.
    LLAISYS_DEVICE_NVIDIA = 1,
    LLAISYS_DEVICE_TYPE_COUNT
} llaisysDeviceType_t;
```

设备类型枚举：定义支持的计算设备类型，用于区分后端设备（如CPU、NVIDIA GPU等）

**为了实现多设备（CPU、NVIDIA GPU）**

```
// Data Types
typedef enum {
    LLAISYS_DTYPE_INVALID = 0,
    LLAISYS_DTYPE_BYTE = 1,
    LLAISYS_DTYPE_BOOL = 2,
    LLAISYS_DTYPE_I8 = 3,
    LLAISYS_DTYPE_I16 = 4,
    LLAISYS_DTYPE_I32 = 5,
    LLAISYS_DTYPE_I64 = 6,
    LLAISYS_DTYPE_U8 = 7,
    LLAISYS_DTYPE_U16 = 8,
    LLAISYS_DTYPE_U32 = 9,
    LLAISYS_DTYPE_U64 = 10,
    LLAISYS_DTYPE_F8 = 11,
    LLAISYS_DTYPE_F16 = 12,
    LLAISYS_DTYPE_F32 = 13,
    LLAISYS_DTYPE_F64 = 14,
    LLAISYS_DTYPE_C16 = 15,
    LLAISYS_DTYPE_C32 = 16,
    LLAISYS_DTYPE_C64 = 17,
    LLAISYS_DTYPE_C128 = 18,
    LLAISYS_DTYPE_BF16 = 19,
} llaisysDataType_t;
```

数据类型枚举：定义张量/缓冲区中支持的所有数据格式，包含：

原始字节、布尔、有/无符号整数（8/16/32/64 bit）、浮点数（8/16/32/64 bit）、复数（16/32/64/128 bit）、

Brain Floating Point Format 即BF16（见thinking）

```
// Runtime Types
// Stream
typedef void *llaisysStream_t;
```

运行时类型定义：定义流（Stream）的不透明句柄类型，

流：“流”是对数据序列的抽象，可以是从某个源读取数据（输入流），或向某个目标写入数据（输出流）

句柄：是一个不透明的标识符，用于引用某个内部对象（如流、文件、设备）

对流的访问需要句柄

```
// Memory Copy Directions
typedef enum {
    LLAISYS_MEMCPY_H2H = 0,
    LLAISYS_MEMCPY_H2D = 1,
    LLAISYS_MEMCPY_D2H = 2,
    LLAISYS_MEMCPY_D2D = 3,
} llaisysMemcpyKind_t;
```

内存拷贝方向枚举：（主机、设备）×（主机、设备）一共四种


- [x] ### 0.2 include/llaisys/runtime.h

```
#ifndef LLAISYS_RUNTIME_H
#define LLAISYS_RUNTIME_H
...
#endif // LLAISYS_RUNTIME_H
```

头文件保护符：与 `llaisys.h` 风格一致，`#include "../llaisys.h"` 引入基础类型

```
__C {
```

`__C` 宏展开为 `extern "C"`（C++ 编译时）或空（C 编译时），让C/CPP都可以正常编译这部分代码，确保以下所有函数和结构体使用 C 链接方式，便于 Python ctypes 等外部调用

**为了实现跨语言（C、C++、Python）**

```
    // Runtime API Functions
    // Device
    typedef int (*get_device_count_api)();
    typedef void (*set_device_api)(int);
    typedef void (*device_synchronize_api)();
    // Stream
    typedef llaisysStream_t (*create_stream_api)();
    typedef void (*destroy_stream_api)(llaisysStream_t);
    typedef void (*stream_synchronize_api)(llaisysStream_t);
    // Memory
    typedef void *(*malloc_device_api)(size_t);
    typedef void (*free_device_api)(void *);
    typedef void *(*malloc_host_api)(size_t);
    typedef void (*free_host_api)(void *);
    // Memory copy
    typedef void (*memcpy_sync_api)(void *, const void *, size_t, llaisysMemcpyKind_t);
    typedef void (*memcpy_async_api)(void *, const void *, size_t, llaisysMemcpyKind_t, llaisysStream_t);
```

**函数指针类型定义：定义了 12 种运行时操作的标准签名，分四大类：**

- **设备管理（3 个）**：`get_device_count`（获取设备数量）、`set_device`（切换当前设备）、`device_synchronize`（同步设备）
- **流管理（3 个）**：`create_stream`（创建异步流）、`destroy_stream`（销毁流）、`stream_synchronize`（等待流完成）
- **内存管理（4 个）**：`malloc_device`（设备内存分配）、`free_device`（释放设备内存）、`malloc_host`（主机页锁定内存分配）、`free_host`（释放页锁定内存）
- **内存拷贝（2 个）**：`memcpy_sync`（同步拷贝，阻塞等待完成）、`memcpy_async`（异步拷贝，通过流控制）

函数指针V.S.指针函数（见thinking）

```
    struct LlaisysRuntimeAPI {
        get_device_count_api get_device_count;
        set_device_api set_device;
        device_synchronize_api device_synchronize;
        create_stream_api create_stream;
        destroy_stream_api destroy_stream;
        stream_synchronize_api stream_synchronize;
        malloc_device_api malloc_device;
        free_device_api free_device;
        malloc_host_api malloc_host;
        free_host_api free_host;
        memcpy_sync_api memcpy_sync;
        memcpy_async_api memcpy_async;
    };
```

运行时 API 函数表：`LlaisysRuntimeAPI` 是一个纯 C 结构体，包含 12 个函数指针。**这是整个项目的设备抽象核心**——不同设备（CPU/NVIDIA）各自填充这个结构体，上层代码通过统一的函数指针调用，无需关心底层是 CPU 还是 GPU。

CPU 实现会将 `malloc_device` 指向 `malloc`、`memcpy_sync` 指向 `memcpy`；NVIDIA 实现则指向 `cudaMalloc`、`cudaMemcpy` 等。

**为了实现多设备（CPU、NVIDIA GPU）**

```
    // Llaisys API for getting the runtime APIs
    __export const LlaisysRuntimeAPI *llaisysGetRuntimeAPI(llaisysDeviceType_t);

    // Llaisys API for switching device context
    __export void llaisysSetContextRuntime(llaisysDeviceType_t, int);
```

全局导出函数：

- `llaisysGetRuntimeAPI`：根据设备类型返回对应的 `LlaisysRuntimeAPI` 函数表指针（CPU 返回 CPU 函数表，NVIDIA 返回 CUDA 函数表）
- `llaisysSetContextRuntime`：切换当前线程的设备上下文（设置设备类型和设备 ID），后续所有操作都在该设备上执行

**为了实现多设备（CPU、NVIDIA GPU）**

```
} // end __C
```

- [x] ### 0.3 include/llaisys/tensor.h

```
#ifndef LLAISYS_TENSOR_H
#define LLAISYS_TENSOR_H
...
#endif // LLAISYS_TENSOR_H
```

头文件保护符，`#include "../llaisys.h"` 引入基础类型枚举

```
__C {
    typedef struct LlaisysTensor *llaisysTensor_t;
```

不透明句柄：`llaisysTensor_t` 是一个指向 `LlaisysTensor` 结构体的指针，但结构体定义不对外暴露（定义在 `src/tensor/tensor.hpp` 中）。Python 端通过 ctypes 持有这个 `void*` 指针，调用 C API 时传入。

```
    __export llaisysTensor_t tensorCreate(
        size_t * shape,
        size_t ndim,
        llaisysDataType_t dtype,
        llaisysDeviceType_t device_type,
        int device_id);
```

创建张量：传入 shape 数组、维度数、数据类型、目标设备类型和 ID，在指定设备上分配内存并返回张量句柄

```
    __export void tensorDestroy(
        llaisysTensor_t tensor);
```

销毁张量：释放张量占用的设备内存和元数据

```
    __export void *tensorGetData(
        llaisysTensor_t tensor);
```

获取数据指针：返回张量底层存储的原始指针（`std::byte*`），注意这个指针指向的是设备内存，CPU 和 GPU 上的地址空间不同

```
    __export size_t tensorGetNdim(
        llaisysTensor_t tensor);
```

获取维度数：返回张量的 `ndim`（几维张量），标量=0，向量=1，矩阵=2

```
    __export void tensorGetShape(
        llaisysTensor_t tensor,
        size_t * shape);
```

获取形状：将张量各维度的大小写入 `shape` 数组（调用者需预分配 `ndim` 个 `size_t`）

```
    __export void tensorGetStrides(
        llaisysTensor_t tensor,
        ptrdiff_t * strides);
```

获取步长：将张量各维度的 stride 写入数组。stride 表示"沿该维度走一步需要跳过多少字节/元素"，是 `view()`/`permute()` 等操作的关键

```
    __export llaisysDataType_t tensorGetDataType(
        llaisysTensor_t tensor);
    __export llaisysDeviceType_t tensorGetDeviceType(
        llaisysTensor_t tensor);
    __export int tensorGetDeviceId(
        llaisysTensor_t tensor);
```

获取元信息：数据类型、设备类型、设备 ID——三个 getter 函数，分别返回张量的 dtype、所在设备和设备编号

```
    __export void tensorDebug(
        llaisysTensor_t tensor);
```

调试打印：将张量数据从设备内存拷贝到主机（D2H memcpy），然后按张量形状打印到 stdout。**内部会做一次跨设备拷贝**，所以 GPU 张量也能打印

```
    __export uint8_t tensorIsContiguous(
        llaisysTensor_t tensor);
```

连续性检查：检查张量的内存布局是否连续（`strides` 是否满足 row-major 连续条件）。连续的张量可以直接当一维数组操作，`view()` 要求输入连续

```
    __export void tensorLoad(
        llaisysTensor_t tensor,
        const void *data);
```

加载数据：将主机内存 `data` 拷贝到张量的设备内存（H2D memcpy）。如果张量在 GPU 上，会触发 `cudaMemcpyHostToDevice`

```
    __export llaisysTensor_t tensorView(
        llaisysTensor_t tensor,
        size_t * shape,
        size_t ndim);
```

视图变换：返回一个新的张量，**共享同一块底层内存**（storage），但 shape 和 strides 不同。不复制数据，O(1) 操作

```
    __export llaisysTensor_t tensorPermute(
        llaisysTensor_t tensor,
        size_t * order);
```

维度置换：按 `order` 指定的顺序重排维度。例如 `order=[1,0]` 等价于矩阵转置。同样共享 storage，不复制数据

```
    __export llaisysTensor_t tensorSlice(
        llaisysTensor_t tensor,
        size_t dim,
        size_t start,
        size_t end);
```

切片：沿指定维度 `dim` 取 `[start, end)` 范围的子张量。通过修改 offset 和调整 shape 实现，共享底层 storage

- [x] ### 0.4 include/llaisys/ops.h

```
#ifndef LLAISYS_OPS_H
#define LLAISYS_OPS_H
...
#endif // LLAISYS_OPS_H
```

头文件保护符，`#include "tensor.h"` 引入 `llaisysTensor_t` 类型

```
__C {
    __export void llaisysAdd(llaisysTensor_t c, llaisysTensor_t a, llaisysTensor_t b);
    __export void llaisysArgmax(llaisysTensor_t max_idx, llaisysTensor_t max_val, llaisysTensor_t vals);
    __export void llaisysEmbedding(llaisysTensor_t out, llaisysTensor_t index, llaisysTensor_t weight);
    __export void llaisysLinear(llaisysTensor_t out, llaisysTensor_t in, llaisysTensor_t weight, llaisysTensor_t bias);
    __export void llaisysRearrange(llaisysTensor_t out, llaisysTensor_t in);
    __export void llaisysRmsNorm(llaisysTensor_t out, llaisysTensor_t in, llaisysTensor_t weight, float eps);
    __export void llaisysROPE(llaisysTensor_t out, llaisysTensor_t in, llaisysTensor_t pos_ids, float theta);
    __export void llaisysSelfAttention(llaisysTensor_t attn_val, llaisysTensor_t q, llaisysTensor_t k, llaisysTensor_t v, float scale);
    __export void llaisysSwiGLU(llaisysTensor_t out, llaisysTensor_t gate, llaisysTensor_t up);
}
```

全部 9 个算子的 C API 声明。每个函数都是 `extern "C"` 导出，接收 `llaisysTensor_t` 句柄作为输入/输出：

| 函数 | 功能 | 输入 | 输出 |
|:--|:--|:--|:--|
| `llaisysAdd` | 逐元素加法 | a, b | c = a + b |
| `llaisysArgmax` | 沿维度取最大值索引 | vals | max_idx, max_val |
| `llaisysEmbedding` | 词嵌入查表 | index, weight | out = weight[index] |
| `llaisysLinear` | 线性变换 | in, weight, bias | out = in × W^T + b |
| `llaisysRearrange` | 维度重排 | in | out（按特定模式重排） |
| `llaisysRmsNorm` | RMS 归一化 | in, weight, eps | out = RMSNorm(in) |
| `llaisysROPE` | 旋转位置编码 | in, pos_ids, theta | out（RoPE 旋转后） |
| `llaisysSelfAttention` | 自注意力 | q, k, v, scale | attn_val = softmax(QK^T/√d)·V |
| `llaisysSwiGLU` | SwiGLU 激活 | gate, up | out = gate × SiLU(up) |

注意：所有输出张量需要在调用前由调用者预先创建好（`tensorCreate`），算子只负责填充数据，不负责分配内存。

- [x] ### 0.5 include/llaisys/models/qwen2.h

```
#ifndef LLAISYS_MODELS_QWEN2_H
#define LLAISYS_MODELS_QWEN2_H
...
#endif // LLAISYS_MODELS_QWEN2_H
```

头文件保护符，`#include "../tensor.h"` 引入 `llaisysTensor_t` 和基础类型

```
    struct LlaisysQwen2Meta {
        llaisysDataType_t dtype;
        size_t nlayer, hs, nh, nkvh, dh, di, maxseq, voc;
        float epsilon, theta;
        int64_t end_token;
    };
```

模型元信息结构体：描述 Qwen2 模型的超参数，所有字段含义如下：

| 字段 | 全称 | 含义 |
|:--|:--|:--|
| `dtype` | data type | 模型权重和激活值的数据类型 |
| `nlayer` | number of layers | Transformer 层数 |
| `hs` | hidden size | 隐藏层维度 |
| `nh` | number of heads | 注意力头数 |
| `nkvh` | number of KV heads | KV 头数（GQA：Grouped Query Attention） |
| `dh` | dimension per head | 每个注意力头的维度 |
| `di` | intermediate dimension | FFN 中间层维度 |
| `maxseq` | max sequence length | 最大序列长度 |
| `voc` | vocabulary size | 词表大小 |
| `epsilon` | epsilon | RMSNorm 的 epsilon 参数 |
| `theta` | theta | RoPE 位置编码的 theta 参数 |
| `end_token` | end token | 序列结束标记的 token ID |

```
    struct LlaisysQwen2Weights {
        llaisysTensor_t in_embed;
        llaisysTensor_t out_embed;
        llaisysTensor_t out_norm_w;   // a.k.a. model.norm.weight
        llaisysTensor_t *attn_norm_w; // a.k.a. input_layernorm.weight
        llaisysTensor_t *attn_q_w;
        llaisysTensor_t *attn_q_b;
        llaisysTensor_t *attn_k_w;
        llaisysTensor_t *attn_k_b;
        llaisysTensor_t *attn_v_w;
        llaisysTensor_t *attn_v_b;
        llaisysTensor_t *attn_o_w;
        llaisysTensor_t *mlp_norm_w; // a.k.a. post_attention_layernorm.weight
        llaisysTensor_t *mlp_gate_w;
        llaisysTensor_t *mlp_up_w;
        llaisysTensor_t *mlp_down_w;
    };
```

模型权重结构体：存储 Qwen2 模型的所有权重张量。分两类：

- **全局权重（标量指针）**：`in_embed`（输入嵌入）、`out_embed`（输出嵌入）、`out_norm_w`（最终 LayerNorm 权重）——每模型只有一份
- **逐层权重（指针数组）**：`attn_norm_w`（注意力层归一化）、`attn_q/k/v/o_w/b`（Q/K/V/O 投影权重和偏置）、`mlp_norm_w`（FFN 层归一化）、`mlp_gate_w`（门控投影）、`mlp_up_w`（上投影）、`mlp_down_w`（下投影）——`*` 表示数组，长度为 `nlayer`，每层一份

每个权重都是一个 `llaisysTensor_t`，实际存储在设备内存中。

```
    struct LlaisysQwen2Model;

    __export struct LlaisysQwen2Model *llaisysQwen2ModelCreate(
        const LlaisysQwen2Meta *meta,
        llaisysDeviceType_t device,
        int *device_ids,
        int ndevice);
```

创建模型：根据元信息 `meta` 在指定设备（`device` + `device_ids`）上创建 Qwen2 模型实例。`device_ids` 数组和 `ndevice` 支持多 GPU 分布式推理；单卡时 `ndevice=1`，`device_ids=[0]`

```
    __export void llaisysQwen2ModelDestroy(
        struct LlaisysQwen2Model * model);
```

销毁模型：释放模型占用的所有设备内存（权重、KV Cache 等）

```
    __export struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(
        struct LlaisysQwen2Model * model);
```

获取权重句柄：返回模型内部 `LlaisysQwen2Weights` 结构体的指针，外部可用 `tensorLoad()` 将预训练权重数据加载到这些张量中

```
    __export int64_t llaisysQwen2ModelInfer(
        struct LlaisysQwen2Model * model,
        int64_t * token_ids,
        size_t ntoken);
```

模型推理：输入 `token_ids` 数组和 `ntoken` 长度，执行一次前向传播，返回预测的下一个 token ID。内部会更新 KV Cache，多次调用可实现自回归生成

