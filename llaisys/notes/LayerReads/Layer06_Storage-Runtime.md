# Layer06_Storage-Runtime

问：*为什么要把 Storage 和 Runtime 放在同一层？*

答：*Storage 是 **Runtime 的工厂类**——Storage 的构造函数是 private 的，只有 Runtime 才能创建 Storage。这层理解"内存块如何被 Runtime 生产和管理"。*

- [x] ## 第 6 层：Storage + Runtime — 4 个文件

**Storage 和 Runtime 是"内聚"的——Storage 只在 Runtime 内部创建，两者一起理解。**

| 序号 | 文件 | 行数 | 核心看点 |
|:--:|------|:--:|------|
| 6.1 | [src/core/runtime/runtime.hpp](file:///c:/Code/LLAISYS/llaisys/src/core/runtime/runtime.hpp) | 47 | `Runtime` 类：持有 `_device_type`、`_device_id`、`_api`、`_allocator`、`_stream`；`allocateDeviceStorage()` / `allocateHostStorage()` |
| 6.2 | [src/core/runtime/runtime.cpp](file:///c:/Code/LLAISYS/llaisys/src/core/runtime/runtime.cpp) | 73 | 关键实现：构造时创建 Allocator 和 Stream；`allocateDeviceStorage` → `new Storage(allocator→allocate(size), ...)` |
| 6.3 | [src/core/storage/storage.hpp](file:///c:/Code/LLAISYS/llaisys/src/core/storage/storage.hpp) | 28 | `Storage` 类：`_memory`（原始指针）、`_size`、`_runtime`（引用）、`_is_host`；构造函数 `private`，`friend class Runtime` |
| 6.4 | [src/core/storage/storage.cpp](file:///c:/Code/LLAISYS/llaisys/src/core/storage/storage.cpp) | 28 | 析构 `Runtime::freeStorage(this)`；getter 方法 |

---

- [x] ### 6.1 src/core/runtime/runtime.hpp

```
#pragma once
#include "../core.hpp"

#include "../../device/runtime_api.hpp"
#include "../allocator/allocator.hpp"

namespace llaisys::core {
class Runtime {
private:
    llaisysDeviceType_t _device_type;
    int _device_id;
    const LlaisysRuntimeAPI *_api;
    MemoryAllocator *_allocator;
    bool _is_active;
    void _activate();
    void _deactivate();
    llaisysStream_t _stream;
    Runtime(llaisysDeviceType_t device_type, int device_id);
```

Runtime 私有成员：

- `_device_type` / `_device_id`：标识设备
- `_api`：设备函数表（CPU 或 CUDA）
- `_allocator`：内存分配器（`NaiveAllocator`，构造时 `new`）
- `_is_active`：是否被 `Context` 激活
- `_activate()` / `_deactivate()`：由 `Context::setDevice()` 调用
- `_stream`：设备流（CPU 上是 0，GPU 上是 `cudaStream_t`）
- 构造函数 `private`：只有 `Context` 才能创建 `Runtime`

```
public:
    friend class Context;

    ~Runtime();

    // Prevent copying
    Runtime(const Runtime &) = delete;
    Runtime &operator=(const Runtime &) = delete;

    // Prevent moving
    Runtime(Runtime &&) = delete;
    Runtime &operator=(Runtime &&) = delete;

    llaisysDeviceType_t deviceType() const;
    int deviceId() const;
    bool isActive() const;

    const LlaisysRuntimeAPI *api() const;

    storage_t allocateDeviceStorage(size_t size);
    storage_t allocateHostStorage(size_t size);
    void freeStorage(Storage *storage);

    llaisysStream_t stream() const;
    void synchronize() const;
};
```

公共接口：

- `allocateDeviceStorage` / `allocateHostStorage`：返回 `storage_t`（即 `shared_ptr<Storage>`），区分设备内存和主机内存
- `freeStorage`：释放一个 Storage 的内存
- `synchronize()`：同步设备流（GPU 上等所有操作完成，CPU 上无操作）
- 禁止拷贝/移动：`Runtime` 不允许复制

---

- [x] ### 6.2 src/core/runtime/runtime.cpp

```
Runtime::Runtime(llaisysDeviceType_t device_type, int device_id)
    : _device_type(device_type), _device_id(device_id), _is_active(false) {
    _api = llaisys::device::getRuntimeAPI(_device_type);
    _stream = _api->create_stream();
    _allocator = new allocators::NaiveAllocator(_api);
}
```

构造过程：

1. 通过 `getRuntimeAPI(device_type)` 获取函数表（CPU 用 `std::malloc`，NVIDIA 用 `cudaMalloc`）
2. 创建设备流 `_stream`（CPU 返回 0，GPU 返回 `cudaStream_t`）
3. 创建 `NaiveAllocator`——传入 `_api`，分配器通过函数表调用设备 API
4. 初始状态 `_is_active = false`——等待 `Context::setDevice()` 激活

```
Runtime::~Runtime() {
    if (!_is_active) {
        std::cerr << "Mallicious destruction of inactive runtime." << std::endl;
    }
    delete _allocator;
    _allocator = nullptr;
    _api->destroy_stream(_stream);
    _api = nullptr;
}
```

析构：如果 Runtime 未被激活就被销毁，打印警告。然后清理 `_allocator` 和 `_stream`。

```
void Runtime::_activate() {
    _api->set_device(_device_id);
    _is_active = true;
}

void Runtime::_deactivate() {
    _is_active = false;
}
```

激活/停用：`_activate` 调用 `_api->set_device(device_id)`——CPU 上什么也不做，GPU 上调用 `cudaSetDevice(device_id)` 设置当前活跃的 GPU。

```
storage_t Runtime::allocateDeviceStorage(size_t size) {
    return std::shared_ptr<Storage>(new Storage(_allocator->allocate(size), size, *this, false));
}

storage_t Runtime::allocateHostStorage(size_t size) {
    return std::shared_ptr<Storage>(new Storage((std::byte *)_api->malloc_host(size), size, *this, true));
}
```

**关键理解**——两种内存分配方式：

- **Device Storage**：通过 `_allocator->allocate()` → `_api->malloc_device()`。CPU 上就是 `malloc`，GPU 上就是 `cudaMalloc`。返回 GPU 显存
- **Host Storage**：直接调用 `_api->malloc_host()`。CPU 上就是 `malloc`，GPU 上就是 `cudaMallocHost`（分配锁页内存，用于加速 CPU↔GPU 传输）

返回值都是 `shared_ptr<Storage>`——引用计数管理，最后一个引用消失时自动析构。

```
void Runtime::freeStorage(Storage *storage) {
    if (storage->isHost()) {
        _api->free_host(storage->memory());
    } else {
        _allocator->release(storage->memory());
    }
}
```

释放内存：根据 `isHost()` 区分——Host 内存直接调 `_api->free_host`，Device 内存通过 Allocator 释放。**对称于分配逻辑**。

---

- [x] ### 6.3 src/core/storage/storage.hpp

```
#pragma once
#include "llaisys.h"

#include "../core.hpp"

#include <memory>

namespace llaisys::core {
class Storage {
private:
    std::byte *_memory;
    size_t _size;
    Runtime &_runtime;
    bool _is_host;
    Storage(std::byte *memory, size_t size, Runtime &runtime, bool is_host);

public:
    friend class Runtime;
    ~Storage();

    std::byte *memory() const;
    size_t size() const;
    llaisysDeviceType_t deviceType() const;
    int deviceId() const;
    bool isHost() const;
};
```

Storage 私有成员：

- `_memory`：原始内存指针（`std::byte*`）
- `_size`：分配的字节数
- `_runtime`：**引用**（不是指针）——Storage 的生命周期绑定到 Runtime，Runtime 先于 Storage 销毁
- `_is_host`：区分 Host 内存和 Device 内存
- 构造函数 `private` + `friend class Runtime`：**只有 Runtime 能创建 Storage**，外部无法直接 `new Storage(...)`

**设计意图**：`Storage` 是"内存块"的最小抽象，不包含任何设备信息——设备信息通过 `_runtime` 间接获取（`deviceType()` / `deviceId()` 代理到 `_runtime`）。

---

- [x] ### 6.4 src/core/storage/storage.cpp

```
#include "storage.hpp"

#include "../runtime/runtime.hpp"

namespace llaisys::core {
Storage::Storage(std::byte *memory, size_t size, Runtime &runtime, bool is_host)
    : _memory(memory), _size(size), _runtime(runtime), _is_host(is_host) {
}

Storage::~Storage() {
    _runtime.freeStorage(this);
}
```

**析构自动释放内存**：`Storage` 析构时调用 `_runtime.freeStorage(this)`，不需要手动 `free`。这是 RAII（Resource Acquisition Is Initialization）模式——内存生命周期完全由 `shared_ptr<Storage>` 管理。

```
std::byte *Storage::memory() const {
    return _memory;
}

size_t Storage::size() const {
    return _size;
}

llaisysDeviceType_t Storage::deviceType() const {
    return _runtime.deviceType();
}

int Storage::deviceId() const {
    return _runtime.deviceId();
}

bool Storage::isHost() const {
    return _is_host;
}
```

代理方法：`deviceType()` 和 `deviceId()` 不存储在 Storage 内部，而是**代理到 `_runtime`**。这样 Storage 保持最小化设计——只关心"内存指针 + 大小 + 是否为 Host"，设备信息通过 Runtime 获取。

**生命周期链条**：
```
Context（线程级单例，全局唯一）
  └─ Runtime（每个设备一个，由 Context 管理）
       └─ NaiveAllocator（Runtime 持有，负责分配内存）
       └─ Storage（shared_ptr，引用计数管理）
            └─ std::byte* _memory（原始内存块）
```

**Storage 析构链**：
```
shared_ptr<Storage> 引用计数归零
  → Storage::~Storage()
    → _runtime.freeStorage(this)
      → [isHost?] _api->free_host(memory) 或 _allocator->release(memory)
        → [CPU] std::free(ptr) 或 [NVIDIA] cudaFree(ptr)
```