# `void*` 详解：类型擦除的万能指针

> `void*` 是 C/C++ 中唯一可以合法指向"任何类型"的指针，是底层系统编程和框架设计中最核心的类型擦除机制。本文从语法、原理、LLAISYS 中的实际用法三个层面展开。

---

## 目录

1. [什么是 `void*`](#1-什么是-void)
2. [为什么会需要 `void*`](#2-为什么会需要-void)
3. [核心规则](#3-核心规则)
4. [LLAISYS 中的三种典型用法](#4-llaisys-中的三种典型用法)
   - [4.1 类型分发：`std::byte*` 替代 `void*`](#41-类型分发stdbyte-替代-void)
   - [4.2 内存管理：原始指针与 `malloc/free`](#42-内存管理原始指针与-mallocfree)
   - [4.3 C 接口：`const void*` 作为万能输入](#43-c-接口const-void-作为万能输入)
5. [完整调用链：从 Python 到 SIMD](#5-完整调用链从-python-到-simd)
6. [`void*` vs `std::byte*` vs `std::any` vs `std::variant`](#6-void-vs-stdbyte-vs-stdany-vs-stdvariant)
7. [常见陷阱](#7-常见陷阱)

---

## 1. 什么是 `void*`

`void*` 是一个**无类型指针**（typeless pointer）。它存储的是内存地址，但不携带任何类型信息。

```cpp
int    x = 42;
float  y = 3.14f;
char   z = 'A';

void* p1 = &x;   // OK: int* → void*
void* p2 = &y;   // OK: float* → void*
void* p3 = &z;   // OK: char* → void*

// void* 可以指向任何类型，但自身不记录类型
```

与 `void` 的关系：
- `void` 作为返回类型 → 函数无返回值
- `void*` 作为指针类型 → 指向未知类型的数据

---

## 2. 为什么会需要 `void*`

### 用 C 语言回答：泛型编程的替代

C 语言没有模板（template），也没有函数重载。如果需要一个可以操作任意类型数据的函数，只能用 `void*`：

```c
// C 的 qsort — 用 void* 实现泛型排序
void qsort(void *base, size_t nmemb, size_t size,
           int (*compar)(const void *, const void *));

// 可以排序任意类型：
int    arr1[] = {3, 1, 2};
double arr2[] = {3.0, 1.0, 2.0};

qsort(arr1, 3, sizeof(int),    cmp_int);
qsort(arr2, 3, sizeof(double), cmp_double);
```

### 更根本的答案：计算机内存没有类型

CPU 的内存是扁平的字节序列。`0x7FFE1234` 这个地址上存储的 4 个字节，可以是 `int`、`float`、`uint32_t`、4 个 `char`，或者一个指针。**类型是人类施加在比特上的解释**。

`void*` 直接反映了这个底层事实：它说"我知道这里有个地址，但我不告诉你如何解释里面的数据——你自己负责"。

---

## 3. 核心规则

### 规则 1：任何指针可以隐式转换为 `void*`

```cpp
int*  ip = &x;
void* vp = ip;   // 隐式转换，无需 cast
```

### 规则 2：`void*` 不能隐式转换回具体类型（C++ 中）

```cpp
void* vp = &x;
int*  ip = vp;          // ❌ C++ 编译错误
int*  ip = (int*)vp;    // ✅ C 风格强制转换
int*  ip = static_cast<int*>(vp);     // ✅ C++ 风格
int*  ip = reinterpret_cast<int*>(vp); // ✅ 更明确（推荐）
```

> C 语言允许 `void*` 隐式转换回任何类型，但 C++ 不允许。这是 C 和 C++ 的重要区别之一。

### 规则 3：不能解引用 `void*`

```cpp
void* vp = &x;
*vp = 10;       // ❌ 编译错误：void 是不完整类型，无法解引用
*(int*)vp = 10;  // ✅ 先转换再解引用
```

### 规则 4：不能对 `void*` 做指针算术

```cpp
void* vp = &x;
vp++;           // ❌ 编译错误：不知道步长是多少
((char*)vp)++;   // ✅ 按字节步进
```

### 规则 5：`void*` 不携带类型信息

```cpp
void* vp = new int(42);
delete vp;      // ❌ 未定义行为：不知道调用哪个析构函数
delete (int*)vp; // ✅
```

---

## 4. LLAISYS 中的三种典型用法

### 4.1 类型分发：`std::byte*` 替代 `void*`

**问题**：LLM 推理需要支持多种数据类型（F32、BF16、F16、I32、I64），但算子的核心算法逻辑是相同的，只有元素类型不同。

**解决方案**：用 `std::byte*`（语义上等价于 `void*`）作为统一的入口参数，然后在函数内部根据 `llaisysDataType_t` 分发到正确的模板实例化。

**出现在**：所有 `src/ops/*/cpu/*_cpu.cpp` 的对外接口函数。

**完整示例**（[`linear_cpu.cpp`](file:///c:/Code/LLAISYS/llaisys/src/ops/linear/cpu/linear_cpu.cpp)）：

```cpp
// 步骤 1：模板化的核心算法 — 在编译时就知道类型 T
template <typename T>
void linear_(T *out, const T *in, const T *weight, const T *bias,
             size_t m, size_t k, size_t n) {
    // 具体计算逻辑，使用 T 的乘法、加法、累加
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            float sum = 0.0f;
            for (size_t p = 0; p < k; p++) {
                sum += static_cast<float>(in[i * k + p]) *
                       static_cast<float>(weight[j * k + p]);
            }
            out[i * n + j] = static_cast<T>(sum);
        }
    }
}

// 步骤 2：类型分发的入口 — 运行时根据 dtype 选择实例化
namespace llaisys::ops::cpu {
void linear(std::byte *out, const std::byte *in,
            const std::byte *weight, const std::byte *bias,
            llaisysDataType_t type, size_t m, size_t k, size_t n) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return linear_(reinterpret_cast<float *>(out),
                       reinterpret_cast<const float *>(in),
                       reinterpret_cast<const float *>(weight),
                       reinterpret_cast<const float *>(bias),
                       m, k, n);
    case LLAISYS_DTYPE_BF16:
        return linear_(reinterpret_cast<llaisys::bf16_t *>(out), ...);
    case LLAISYS_DTYPE_F16:
        return linear_(reinterpret_cast<llaisys::fp16_t *>(out), ...);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
}
```

**关键设计点**：

| 层面 | 指针类型 | 何时确定类型 | 作用 |
|------|:--:|:--:|------|
| 入口函数签名 | `std::byte*` | 运行时（通过 `dtype` 参数） | 统一接口，调用方不需要知道具体类型 |
| 模板函数签名 | `T*` | 编译时（通过模板实例化） | 编译器生成类型安全的优化代码 |
| 转换点 | `reinterpret_cast<const float*>(in)` | 分发时 | 把无类型指针重新解释为具体类型 |

**为什么用 `std::byte*` 而不是 `void*`？**

C++17 引入了 `std::byte`，它的语义是"这是一块原始内存，不是对象"。与 `void*` 相比：
- `std::byte*` 明确表达"这是字节序列"的意图
- `void*` 更通用，但语义模糊（"不知道是什么" vs "就是原始字节"）
- 在 LLAISYS 中，`std::byte*` 出现的位置恰好是"数据指针"的位置，语义更精确

**LLaMA.cpp 对比**：LLaMA.cpp 做同样的事，但用的是 `void*`：

```cpp
// llama.cpp 风格（C 语言）
void ggml_compute_forward_add(
    const struct ggml_compute_params *params,
    struct ggml_tensor *dst) {
    // 用 void* 做类型擦除
    const void *src0 = dst->src[0]->data;
    switch (dst->src[0]->type) {
        case GGML_TYPE_F32: /* 用 float* 处理 */ break;
        case GGML_TYPE_F16: /* 用 ggml_fp16_t* 处理 */ break;
    }
}
```

---

### 4.2 内存管理：原始指针与 `malloc/free`

**问题**：设备内存分配函数需要返回一个指向已分配内存的指针，但分配器不知道调用者会如何解释这块内存。

**解决方案**：`mallocDevice`/`mallocHost` 返回 `void*`，`freeDevice`/`freeHost` 接收 `void*`。

**出现在**：`src/device/cpu/cpu_runtime_api.cpp`、`src/device/nvidia/nvidia_runtime_api.cu`、`src/device/runtime_api.cpp`。

```cpp
// CPU 端实现（简化的 malloc/wrapper）
void *mallocDevice(size_t size) {
    return std::malloc(size);  // 返回 void*，不承诺类型
}

void freeDevice(void *ptr) {
    std::free(ptr);            // 接收 void*，不关心类型
}

// CUDA 端实现
void *mallocDevice(size_t size) {
    void *ptr;
    cudaMalloc(&ptr, size);    // cudaMalloc 也返回 void**
    return ptr;
}

void freeDevice(void *ptr) {
    cudaFree(ptr);             // cudaFree 接收 void*
}
```

**设计原理**：`malloc` 和 `free` 只关心"多少字节"，不关心"这些字节表示什么"。这是 `void*` 最经典的用法，从 C 标准库开始就是如此。

**在 LLAISYS 中的调用链**：

```
Tensor::create()
  → runtime.allocateDeviceStorage(size)   // 返回 void* 的 Storage
  → Storage 持有 void* 指针
  → Tensor::data() 返回 std::byte*        // 从 void* 转为 std::byte*
```

---

### 4.3 C 接口：`const void*` 作为万能输入

**问题**：C API 的 `tensorLoad` 函数需要接受来自 Python（NumPy）的任意类型数据。

**解决方案**：`const void*` — 接受任何类型的输入，内部通过 `memcpy` 拷贝字节。

**出现在**：[`src/llaisys/tensor.cc`](file:///c:/Code/LLAISYS/llaisys/src/llaisys/tensor.cc) 和 [`src/tensor/tensor.cpp`](file:///c:/Code/LLAISYS/llaisys/src/tensor/tensor.cpp)。

```cpp
// C API 层（tensor.cc）
void tensorLoad(llaisysTensor_t tensor, const void *data) {
    tensor->tensor->load(data);    // 直接传递 void*
}

// C++ 实现层（tensor.cpp）
void Tensor::load(const void *src_) {
    size_t size = numel() * elementSize();
    const auto *api = core::context().runtime().api();
    // memcpy 以 void* 工作 → 拷贝字节，不关心类型
    api->memcpy_sync(data(), src_, size, LLAISYS_MEMCPY_H2D);
}
```

**Python 调用链**：

```python
# Python 端
import numpy as np
t = llaisys.Tensor([3, 4], llaisys.DataType.F32, llaisys.DeviceType.CPU)
data = np.random.randn(3, 4).astype(np.float32)
t.load(data)  # NumPy 数组的 data_ptr 作为 void* 传入
```

```
Python (ctypes)
  → numpy.ndarray.ctypes.data_as(c_void_p)
  → tensorLoad(tensor, void* data)       ← C API
  → Tensor::load(const void* src)        ← C++ 实现
  → memcpy_sync(dst, src, size, H2D)     ← 字节拷贝
```

---

## 5. 完整调用链：从 Python 到 SIMD

以 `linear` 算子为例，展示 `void*` / `std::byte*` 在整个调用链中的角色：

```
[Python]
  llaisys.Ops.linear(out, inp, weight, bias)
  → ctypes 调用 C 函数符号

[C API: ops.cc]
  void llaisysLinear(llaisysTensor_t out, llaisysTensor_t in, ...)
  → auto* o = out->tensor->data();  // 返回 std::byte*
  → auto* i = in->tensor->data();   // 返回 std::byte*
  → ops::linear(dtype, o, i, w, b, m, k, n)

[算子分发: op.cpp]
  void linear(tensor_t out, tensor_t in, ...)
  → auto* o = out->data();  // std::byte*
  → ops::cpu::linear(o, i, w, b, dtype, m, k, n)

[CPU 类型分发: linear_cpu.cpp]
  void linear(std::byte *out, const std::byte *in, ...)
  → switch (dtype) {
        case F32: linear_(reinterpret_cast<float*>(out), ...)
        case BF16: linear_(reinterpret_cast<bf16_t*>(out), ...)
    }

[模板化核心算法: linear_<T>()]
  void linear_(float *out, const float *in, ...)
  → AVX2 _mm256_fmadd_ps(out, in, weight)  ← 具体类型 + SIMD
```

**关键观察**：在整个调用链中，`std::byte*` 从 `Tensor::data()` 开始，一直传递到 CPU 分发函数，在 `switch (dtype)` 处才被 `reinterpret_cast` 转换为具体类型。这是**延迟类型绑定**（late type binding）的经典模式。

---

## 6. `void*` vs `std::byte*` vs `std::any` vs `std::variant`

| 特性 | `void*` | `std::byte*` | `std::any` | `std::variant<A,B,C>` |
|------|:--:|:--:|:--:|:--:|
| 引入版本 | C89 | C++17 | C++17 | C++17 |
| 是否知道类型 | 否 | 否 | 是（运行时 RTTI） | 是（编译时枚举） |
| 类型安全 | 不安全 | 不安全 | 安全（bad_any_cast 异常） | 安全（编译时检查） |
| 内存开销 | 8 bytes | 8 bytes | 16+ bytes（堆分配） | 最大类型大小 + 索引 |
| 可以指向任何类型 | 是 | 是（字节） | 是 | 仅限预定义的类型集合 |
| 访问方式 | 强制转换 | 强制转换 | `any_cast<T>()` | `std::get<T>()` / `std::visit` |
| 指针算术 | 禁止 | 允许（按字节） | 禁止 | 禁止 |
| 主要用途 | C 风格泛型、内存管理 | 原始内存操作 | 类型安全的万能容器 | 安全的类型联合 |

### 为什么 LLAISYS 不用 `std::variant`？

LLaMA 系列模型的数据类型是有限的（F32、BF16、F16），理论上可以用 `std::variant`。但选择 `std::byte*` 的原因：

1. **零开销**：`std::byte*` 只是一个 8 字节指针，`std::variant` 需要存储类型标签
2. **C 兼容性**：C API 必须用 `void*`，`std::variant` 无法跨 C 边界
3. **批量操作**：`std::byte*` 指向连续内存，可以整个 buffer 做 `memcpy`；`std::variant` 是逐元素的
4. **运行时在已知上下文**：`dtype` 已经作为单独参数传递，不需要 variant 的标签

---

## 7. 常见陷阱

### 陷阱 1：对齐问题

```cpp
char buf[8];
float* fp = reinterpret_cast<float*>(buf);  // 危险：buf 可能未对齐到 4 字节
*fp = 3.14f;  // 在某些架构上会 SIGBUS
```

**正确做法**：使用 `alignas` 或从对齐的分配器获取内存。

### 陷阱 2：严格别名规则（Strict Aliasing）

```cpp
float f = 3.14f;
int* ip = reinterpret_cast<int*>(&f);  // 危险：违反严格别名规则
int i = *ip;  // 未定义行为
```

**正确做法**：使用 `memcpy` 或 `std::bit_cast`（C++20）：

```cpp
float f = 3.14f;
int i;
std::memcpy(&i, &f, sizeof(float));  // 安全
// 或
int i = std::bit_cast<int>(f);       // C++20
```

### 陷阱 3：忘记转换类型就解引用

```cpp
void process(void* data) {
    // 直接解引用 → 编译错误
    // *data = 42;  // ❌
    
    // 忘记转换的常见错误
    int* p = (int*)data;
    *p = 42;  // ✅ 但前提是 data 确实指向 int
}
```

### 陷阱 4：`void*` 上的 `delete` 未调用析构函数

```cpp
struct Foo {
    ~Foo() { std::cout << "destroyed\n"; }
};

void* p = new Foo();
delete p;         // ❌ 未定义行为：~Foo() 不会被调用
delete (Foo*)p;   // ✅ 正确
```

### 陷阱 5：`reinterpret_cast` 类型不匹配

```cpp
// 假设 data 实际指向 float，但错误地转换为 int
std::byte* data = tensor->data();  // 实际是 F32 张量
auto* ip = reinterpret_cast<int*>(data);  // 运行时错误，但编译通过
// 用 ip 读取到的值完全错误（float 的比特模式 ≠ int 的比特模式）
```

**LLASYS 的防护**：`switch (dtype)` 确保 `reinterpret_cast` 的目标类型与张量的实际数据类型一致。

---

## 总结

| 用法 | 位置 | 为什么用 |
|------|------|------|
| `std::byte*` 类型分发 | `src/ops/*/cpu/*_cpu.cpp` | 统一接口，延迟类型绑定到分发点 |
| `void*` 内存分配 | `src/device/*/runtime_api.cpp` | 标准 C 语义，分配器不关心类型 |
| `const void*` C API | `src/llaisys/tensor.cc` | 跨语言边界，接受任意数据源 |
| `std::byte*` 数据指针 | `src/tensor/tensor.cpp` | 张量内部存储的统一表示 |

核心思想：**`void*`（及其现代替代 `std::byte*`）是类型擦除的最底层机制。它把"类型是什么"从编译时推迟到运行时，让框架可以用一份代码处理 F32/BF16/F16，代价是要求调用者自己保证类型安全。**