# Layer03_Context

问：*为什么需要 Context？*

答：*管理**线程级的多设备运行时环境**——每个线程维护自己的设备映射和当前活跃设备，实现 `setDevice()` 设备切换。*

- [x] ## 第 3 层：Context（上下文管理器） — 2 个文件

**理解"线程级单例"和"多设备切换"的机制。**

| 序号 | 文件 | 行数 | 核心看点 |
|:--:|------|:--:|------|
| 3.1 | [src/core/context/context.hpp](file:///c:/Code/LLAISYS/llaisys/src/core/context/context.hpp) | 41 | `Context` 类：`_runtime_map`（设备→Runtime 映射）、`_current_runtime`、`setDevice()` / `runtime()` |
| 3.2 | [src/core/context/context.cpp](file:///c:/Code/LLAISYS/llaisys/src/core/context/context.cpp) | 77 | **重点**：`thread_local Context thread_context`（线程级单例）、`core::context()` 返回该单例的引用；`setDevice() ` 从 map 中查找 Runtime 并激活 |

---

- [x] ### 3.1 src/core/context/context.hpp

**`Context` 是线程级的设备运行时管理器，通过单例模式维护不同设备的 Runtime 映射，负责切换和获取当前设备的活跃 Runtime，并禁止拷贝/移动以保证唯一性**

```
#pragma once

#include "llaisys.h"

#include "../core.hpp"

#include "../runtime/runtime.hpp"

#include <unordered_map>
#include <vector>

namespace llaisys::core {
class Context {
private:
    std::unordered_map<llaisysDeviceType_t, std::vector<Runtime *>> _runtime_map;
    Runtime *_current_runtime;
    Context();
```

核心数据结构：

- `_runtime_map`：`unordered_map<设备类型, vector<Runtime*>>`——每种设备类型（CPU/NVIDIA）对应一个 `Runtime` 指针数组，数组索引即 `device_id`。例如 `_runtime_map[LLAISYS_DEVICE_NVIDIA][2]` 就是第 3 张 NVIDIA GPU 的 Runtime
- `_current_runtime`：当前活跃的 Runtime 指针，后续所有操作（分配内存、拷贝数据、执行算子）都发生在该 Runtime 代表的设备上
- 构造函数 `private`：**外部不能 new Context**，只能通过友元函数 `context()` 获取

```
public:
    ~Context();

    // Prevent copy
    Context(const Context &) = delete;
    Context &operator=(const Context &) = delete;

    // Prevent move
    Context(Context &&) = delete;
    Context &operator=(Context &&) = delete;
```

禁止拷贝/移动：`Context` 是线程级单例，拷贝或移动会导致多个实例管理同一套 Runtime，造成 double-free。`= delete` 是 C++11 的显式禁用语法。

```
    void setDevice(llaisysDeviceType_t device_type, int device_id);
    Runtime &runtime();

    friend Context &context();
};
} // namespace llaisys::core
```

两个核心接口 + 友元声明：

- `setDevice(type, id)`：切换当前线程的设备上下文。如果当前 Runtime 已经是目标设备，则什么都不做（幂等）；否则 `_deactivate` 旧的、`_activate` 新的
- `runtime()`：返回当前活跃 Runtime 的引用。调用前必须已通过 `setDevice` 激活。所有需要设备操作的地方都通过 `context().runtime()` 获取 Runtime
- `friend Context &context()`：友元函数声明，使全局函数 `context()` 能访问私有构造函数

---

- [x] ### 3.2 src/core/context/context.cpp

**`context.cpp` 通过 `thread_local` 实现线程级单例的 `Context`，按优先级（GPU 优先，CPU 兜底）初始化设备 Runtime，并在 `setDevice` 中实现延迟创建与设备切换，确保每个线程可独立绑定不同设备。**

```
#include "context.hpp"
#include "../../utils.hpp"
#include <thread>
```

`<thread>` 用于 `thread_local` 关键字。

```
Context::Context() {
    // All device types, put CPU at the end
    std::vector<llaisysDeviceType_t> device_typs;
    for (int i = 1; i < LLAISYS_DEVICE_TYPE_COUNT; i++) {
        device_typs.push_back(static_cast<llaisysDeviceType_t>(i));
    }
    device_typs.push_back(LLAISYS_DEVICE_CPU);
```

构造逻辑（初始化设备列表）：先把 GPU 等加速设备排前面，CPU 放最后。通过 `for (int i = 1; ...)` 跳过 `LLAISYS_DEVICE_CPU (= 0)`，遍历完后 `push_back(CPU)`。**意图明确**：优先使用 GPU 等加速设备，CPU 作为兜底。

```
    for (auto device_type : device_typs) {
        const LlaisysRuntimeAPI *api_ = llaisysGetRuntimeAPI(device_type);
        int device_count = api_->get_device_count();
        std::vector<Runtime *> runtimes_(device_count);
        for (int device_id = 0; device_id < device_count; device_id++) {

            if (_current_runtime == nullptr) {
                auto runtime = new Runtime(device_type, device_id);
                runtime->_activate();
                runtimes_[device_id] = runtime;
                _current_runtime = runtime;
            }
        }
        _runtime_map[device_type] = runtimes_;
    }
```

遍历设备并初始化：对每种设备类型，调用 `llaisysGetRuntimeAPI()` 获取函数表，`get_device_count()` 获取设备数量，然后**只为第一个有能力的设备创建 Runtime 并激活**（`if (_current_runtime == nullptr)` 只对第一个设备为 true）。

**重要细节**：CPU 被放到最后遍历，所以如果系统有 GPU，GPU 会先被创建并激活；如果没有任何加速设备，CPU 作为兜底被激活。`_runtime_map` 中空的 Runtime 指针（未激活的设备）在 `setDevice()` 时按需创建。

```
Context::~Context() {
    delete _current_runtime;
    for (auto &runtime_entry : _runtime_map) {
        std::vector<Runtime *> runtimes = runtime_entry.second;
        for (auto runtime : runtimes) {
            if (runtime != nullptr && runtime != _current_runtime) {
                runtime->_activate();
                delete runtime;
            }
        }
    }
    _current_runtime = nullptr;
}
```

析构顺序：先删当前 Runtime，再遍历所有未激活的 Runtime 逐一激活后删除。激活是为了确保 `delete` 时 Runtime 能正确调用设备 API 清理资源（如释放 CUDA 上下文）。

```
void Context::setDevice(llaisysDeviceType_t device_type, int device_id) {
    if (_current_runtime == nullptr || _current_runtime->deviceType() != device_type || _current_runtime->deviceId() != device_id) {
        auto runtimes = _runtime_map[device_type];
        CHECK_ARGUMENT((size_t)device_id < runtimes.size() && device_id >= 0, "invalid device id");
        if (_current_runtime != nullptr) {
            _current_runtime->_deactivate();
        }
        if (runtimes[device_id] == nullptr) {
            runtimes[device_id] = new Runtime(device_type, device_id);
        }
        runtimes[device_id]->_activate();
        _current_runtime = runtimes[device_id];
    }
}
```

设备切换（幂等设计）：

1. 先检查：如果目标设备已经是当前设备，什么都不做（避免不必要的 deactivate/activate）
2. 校验 `device_id` 合法性
3. 旧 Runtime `_deactivate()`（设置为 inactive）
4. 如果目标 Runtime 还没创建（`nullptr`），**延迟创建**（Lazy Initialization）——只在第一次被使用时才 `new Runtime`
5. 新 Runtime `_activate()`（调用 `set_device` 设置设备上下文）
6. 更新 `_current_runtime` 指针

```
Runtime &Context::runtime() {
    ASSERT(_current_runtime != nullptr, "No runtime is activated, please call setDevice() first.");
    return *_current_runtime;
}
```

获取当前 Runtime：用 `ASSERT` 确保有活跃 Runtime，否则抛异常。注意返回的是**引用**而非指针——调用者假设 Runtime 一定存在，无需判空。

```
Context &context() {
    thread_local Context thread_context;
    return thread_context;
}
```

**核心——线程级单例**：

`thread_local` 是 C++11 关键字，每个线程拥有独立的 `thread_context` 实例。这意味着：

- 线程 A 调用 `context().setDevice(CPU, 0)`，只影响线程 A
- 线程 B 调用 `context().setDevice(NVIDIA, 1)`，只影响线程 B
- 两个线程互不干扰，各自管理自己的设备上下文

这是多用户并发推理（Project 4）的基础——每个请求在独立线程中处理，各自绑定不同的 GPU。

**调用链**：

```
用户代码: context().setDevice(NVIDIA, 0)
  → thread_local Context 构造（首次调用时）
    → 遍历设备类型，优先激活 GPU
  → setDevice() 切换到指定设备

用户代码: context().runtime().allocateDeviceStorage(1024)
  → runtime() 返回当前活跃 Runtime 引用
  → allocateDeviceStorage → NaiveAllocator → cudaMalloc
```