# Layer02_Allocator

问：*为什么需要内存分配器抽象？*

答：*统一 CPU `malloc`/`free` 和 GPU `cudaMalloc`/`cudaFree` 的调用接口——上层代码通过**虚函数多态**调用 `allocate/release`，不知道也不关心内存实际在哪。是为了实现**多设备***

- [x] ## 第 2 层：Allocator（内存分配器） — 3 个文件

**先理解"内存从哪来"，再理解"内存怎么管理"。**

| 序号 | 文件 | 行数 | 核心看点 |
|:--:|------|:--:|------|
| 2.1 | [src/core/allocator/allocator.hpp](file:///c:/Code/LLAISYS/llaisys/src/core/allocator/allocator.hpp) | 22 | 抽象 `MemoryAllocator` 类：`allocate(size)` / `release(ptr)` 纯虚函数 |
| 2.2 | [src/core/allocator/naive_allocator.hpp](file:///c:/Code/LLAISYS/llaisys/src/core/allocator/naive_allocator.hpp) | 14 | `NaiveAllocator` 声明——最简单的 `malloc`/`free` 实现 |
| 2.3 | [src/core/allocator/naive_allocator.cpp](file:///c:/Code/LLAISYS/llaisys/src/core/allocator/naive_allocator.cpp) | 11 | 实现：`allocate()` 调 `_api->malloc_device()`，`release()` 调 `_api->free_device()` |

---

- [x] ### 2.1 src/core/allocator/allocator.hpp

**`MemoryAllocator` 是设备内存分配器的抽象基类，让整个框架能够以统一的方式在不同设备上分配/释放内存，同时保持设备相关的实现隔离在各自的子类中。**

```
#pragma once

#include "llaisys/runtime.h"

#include "../storage/storage.hpp"

namespace llaisys::core {
class MemoryAllocator {
protected:
    const LlaisysRuntimeAPI *_api;
    MemoryAllocator(const LlaisysRuntimeAPI *runtime_api) : _api(runtime_api){};

public:
    virtual ~MemoryAllocator() = default;
    virtual std::byte *allocate(size_t size) = 0;
    virtual void release(std::byte *memory) = 0;
};

} // namespace llaisys::core
```

抽象基类设计：

- `_api`：持有设备运行时 API 的函数表指针（来自 `LlaisysRuntimeAPI`），子类通过 `_api->malloc_device()` 等完成实际分配
- 构造函数 `protected`：不允许外部直接实例化，只能通过子类构造
- 析构函数 `virtual`：确保 `delete` 基类指针时正确调用子类析构
- `allocate` / `release` 为纯虚函数（`= 0`）：强制子类实现，构成策略模式——CPU 版调 `malloc`，CUDA 版调 `cudaMalloc`
- 返回值 `std::byte*`：C++17 的字节类型，明确表示"这是一块原始内存"，不是任何具体类型的数组

---

- [x] ### 2.2 src/core/allocator/naive_allocator.hpp

**运行时基础设施层**——为上层算子、模型和 API 提供统一的错误处理、混合精度数据类型、跨设备内存分配能力，是构建高性能推理引擎的基石。

```
#pragma once

#include "allocator.hpp"

namespace llaisys::core::allocators {
class NaiveAllocator : public MemoryAllocator {
public:
    NaiveAllocator(const LlaisysRuntimeAPI *runtime_api);
    ~NaiveAllocator() = default;
    std::byte *allocate(size_t size) override;
    void release(std::byte *memory) override;
};
} // namespace llaisys::core::allocators
```

`NaiveAllocator` 是 `MemoryAllocator` 的最简实现——"naive"（朴素），因为没有任何内存池、缓存或对齐优化，就是直接调用运行时 API。

构造函数接受 `LlaisysRuntimeAPI*` 并传给基类，这意味着**同一个 `NaiveAllocator` 类可以服务不同设备**——CPU 上 `_api` 指向 CPU 函数表（`malloc`/`free`），NVIDIA 上指向 CUDA 函数表（`cudaMalloc`/`cudaFree`）。

---

- [x] ### 2.3 src/core/allocator/naive_allocator.cpp

**适配任意设备运行时 API 的“透传”分配器，不做任何优化，直接转发内存请求给底层 C 函数。**

```
#include "naive_allocator.hpp"

#include "../runtime/runtime.hpp"

namespace llaisys::core::allocators {
NaiveAllocator::NaiveAllocator(const LlaisysRuntimeAPI *runtime_api) : MemoryAllocator(runtime_api) {
}

std::byte *NaiveAllocator::allocate(size_t size) {
    return static_cast<std::byte *>(_api->malloc_device(size));
}

void NaiveAllocator::release(std::byte *memory) {
    _api->free_device(memory);
}
} // namespace llaisys::core::allocators
```

实现极简——只有两个函数：

- `allocate(size)` → 调用 `_api->malloc_device(size)`，返回值 `void*` 强制转为 `std::byte*`
- `release(ptr)` → 调用 `_api->free_device(ptr)`

**关键理解**：分配器本身不关心内存在哪。如果 `_api` 是 CPU 运行时，`malloc_device` 就是 `malloc`；如果 `_api` 是 CUDA 运行时，`malloc_device` 就是 `cudaMalloc`。这就是**依赖注入 + 多态**的威力——上层代码零改动即可切换设备。

**调用链追踪**：

```
Tensor::create()
  → Runtime::allocateDeviceStorage(size)
    → NaiveAllocator::allocate(size)
      → _api->malloc_device(size)
        → [CPU] std::malloc(size) 或 [NVIDIA] cudaMalloc(&ptr, size)
```