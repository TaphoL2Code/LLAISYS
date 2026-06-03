# Layer04_Device

问：*为什么需要 Device 层？*

答：*实现**设备运行时 API 的分发机制**——通过函数表 + 工厂方法，将 CPU 和 GPU 的操作统一抽象，上层代码无需 if-else 判断设备类型。*

- [x] ## 第 4 层：Device（设备抽象） — 5 个文件

**理解"CPU 和 GPU 怎么统一抽象"。**

| 序号 | 文件 | 行数 | 核心看点 |
|:--:|------|:--:|------|
| 4.1 | [src/device/runtime_api.hpp](file:///c:/Code/LLAISYS/llaisys/src/device/runtime_api.hpp) | 27 | `getRuntimeAPI()` 分发函数声明、`getUnsupportedRuntimeAPI()` 兜底声明 |
| 4.2 | [src/device/runtime_api.cpp](file:///c:/Code/LLAISYS/llaisys/src/device/runtime_api.cpp) | 30 | **分发实现**：switch(device_type) → CPU/NVIDIA，以及 NOOP 空实现（所有函数抛异常） |
| 4.3 | [src/device/device_resource.hpp](file:///c:/Code/LLAISYS/llaisys/src/device/device_resource.hpp) | 33 | `DeviceResource` 基类——持有 `_device_type` 和 `_device_id` 两个属性 |
| 4.4 | [src/device/cpu/cpu_resource.hpp](file:///c:/Code/LLAISYS/llaisys/src/device/cpu/cpu_resource.hpp) | 15 | `CPUResource` 声明——继承 `DeviceResource`，固定 CPU 设备 |
| 4.5 | [src/device/cpu/cpu_resource.cpp](file:///c:/Code/LLAISYS/llaisys/src/device/cpu/cpu_resource.cpp) | 18 | CPU 资源构造——`DeviceResource(LLAISYS_DEVICE_CPU, 0)` |

---

- [x] ### 4.1 src/device/runtime_api.hpp

```
#pragma once
#include "llaisys/runtime.h"

#include "../utils.hpp"

namespace llaisys::device {
const LlaisysRuntimeAPI *getRuntimeAPI(llaisysDeviceType_t device_type);

const LlaisysRuntimeAPI *getUnsupportedRuntimeAPI();

namespace cpu {
const LlaisysRuntimeAPI *getRuntimeAPI();
}

#ifdef ENABLE_NVIDIA_API
namespace nvidia {
const LlaisysRuntimeAPI *getRuntimeAPI();
}
#endif
} // namespace llaisys::device
```

API 分发头文件：

- `getRuntimeAPI(device_type)`：全局分发函数——根据设备类型枚举返回对应的 `LlaisysRuntimeAPI` 函数表指针
- `getUnsupportedRuntimeAPI()`：兜底函数——返回一个所有函数都抛异常的"空"函数表，用于不支持该设备时的安全报错
- `cpu::getRuntimeAPI()`：CPU 的 API 获取函数
- `nvidia::getRuntimeAPI()`：NVIDIA CUDA 的 API 获取函数——用 `#ifdef ENABLE_NVIDIA_API` 条件编译，只有在构建时启用了 CUDA 支持才存在

---

- [x] ### 4.2 src/device/runtime_api.cpp

```
#include "runtime_api.hpp"

namespace llaisys::device {

int getDeviceCount() { return 0; }
void setDevice(int) { EXCEPTION_UNSUPPORTED_DEVICE; }
void deviceSynchronize() { EXCEPTION_UNSUPPORTED_DEVICE; }
llaisysStream_t createStream() { EXCEPTION_UNSUPPORTED_DEVICE; return nullptr; }
void destroyStream(llaisysStream_t) { EXCEPTION_UNSUPPORTED_DEVICE; }
void streamSynchronize(llaisysStream_t) { EXCEPTION_UNSUPPORTED_DEVICE; }
void *mallocDevice(size_t) { EXCEPTION_UNSUPPORTED_DEVICE; return nullptr; }
void freeDevice(void *) { EXCEPTION_UNSUPPORTED_DEVICE; }
void *mallocHost(size_t) { EXCEPTION_UNSUPPORTED_DEVICE; return nullptr; }
void freeHost(void *) { EXCEPTION_UNSUPPORTED_DEVICE; }
void memcpySync(void *, const void *, size_t, llaisysMemcpyKind_t) { EXCEPTION_UNSUPPORTED_DEVICE; }
void memcpyAsync(void *, const void *, size_t, llaisysMemcpyKind_t, llaisysStream_t) { EXCEPTION_UNSUPPORTED_DEVICE; }
```

NOOP 默认实现：12 个函数，全部调用 `EXCEPTION_UNSUPPORTED_DEVICE` 抛异常。只有 `getDeviceCount()` 返回 0（表示无设备）。这是"不安全设备"的兜底实现。

```
static const LlaisysRuntimeAPI NOOP_RUNTIME_API = {
    &getDeviceCount, &setDevice, &deviceSynchronize,
    &createStream, &destroyStream, &streamSynchronize,
    &mallocDevice, &freeDevice, &mallocHost, &freeHost,
    &memcpySync, &memcpyAsync};

const LlaisysRuntimeAPI *getUnsupportedRuntimeAPI() {
    return &NOOP_RUNTIME_API;
}
```

全局静态空函数表：`NOOP_RUNTIME_API` 是一个 `static const` 全局变量，整个程序生命周期内只有一份。`getUnsupportedRuntimeAPI()` 返回其指针。

```
const LlaisysRuntimeAPI *getRuntimeAPI(llaisysDeviceType_t device_type) {
    switch (device_type) {
    case LLAISYS_DEVICE_CPU:
        return llaisys::device::cpu::getRuntimeAPI();
    case LLAISYS_DEVICE_NVIDIA:
#ifdef ENABLE_NVIDIA_API
        return llaisys::device::nvidia::getRuntimeAPI();
#else
        return getUnsupportedRuntimeAPI();
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
        return nullptr;
    }
}
```

分发核心——switch 分派：这是整个设备抽象的关键枢纽。`Context::Context()` 构造时调用此函数获取每种设备的 API 表。NVIDIA 分支用了条件编译——如果编译时没有启用 CUDA，则返回 NOOP 空表（安全报错而非编译失败）。

---

- [x] ### 4.3 src/device/device_resource.hpp

```
#pragma once
#include "llaisys.h"

#include "../utils.hpp"

namespace llaisys::device {
class DeviceResource {
private:
    llaisysDeviceType_t _device_type;
    int _device_id;

public:
    DeviceResource(llaisysDeviceType_t device_type, int device_id)
        : _device_type(device_type), _device_id(device_id) {}
    ~DeviceResource() = default;

    llaisysDeviceType_t getDeviceType() const { return _device_type; }
    int getDeviceId() const { return _device_id; };
};
} // namespace llaisys::device
```

设备资源基类：最简单的值对象（Value Object）——只持有两个属性 (`_device_type`, `_device_id`) 和两个 getter。不包含任何设备操作逻辑。

**设计意图**：为 Project 2（CUDA 集成）和 Project 5（分布式推理）的扩展预留。CUDA 设备资源可能额外持有 `cudaStream_t`、`cublasHandle_t` 等句柄，统一继承自此基类。

---

- [x] ### 4.4 src/device/cpu/cpu_resource.hpp

```
#pragma once

#include "../device_resource.hpp"

namespace llaisys::device::cpu {
class Resource : public llaisys::device::DeviceResource {
public:
    Resource();
    ~Resource() = default;
};
} // namespace llaisys::device::cpu
```

CPU 资源类：继承 `DeviceResource`，没有任何额外成员。仅多了一个自定义构造函数（固定传入 `LLAISYS_DEVICE_CPU` 和 `0`）。

---

- [x] ### 4.5 src/device/cpu/cpu_resource.cpp

```
#include "cpu_resource.hpp"

namespace llaisys::device::cpu {
Resource::Resource() : llaisys::device::DeviceResource(LLAISYS_DEVICE_CPU, 0) {}
} // namespace llaisys::device::cpu
```

CPU 资源构造：将设备类型固定为 `LLAISYS_DEVICE_CPU`，设备 ID 固定为 `0`（CPU 只有一个）。这与 NVIDIA GPU 形成对比——GPU 可能有多个，`device_id` 可能是 0/1/2/3...。

**当前代码中 `Resource` 类还没有被实际使用**（Context 直接管理 `Runtime`，不经过 `DeviceResource`）。它为 Project 2/5 预留——CUDA 设备上线后，`nvidia::Resource` 可能持有 `cudaStream_t` 等额外资源。