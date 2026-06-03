# Layer05_CPU-Runtime

问：*为什么 CPU 需要一个"运行时"？*

答：*CPU 虽然天然支持 `malloc`/`free`/`memcpy`，但为了与 GPU 保持**统一接口**，也需要实现 `LlaisysRuntimeAPI` 函数表——这就是"空壳模式"：大部分函数为空操作，但接口完全一致。*

- [x] ## 第 5 层：CPU Runtime API — 1 个文件

**理解"CPU 怎么填充统一函数表"。**

| 序号 | 文件 | 行数 | 核心看点 |
|:--:|------|:--:|------|
| 5.1 | [src/device/cpu/cpu_runtime_api.cpp](file:///c:/Code/LLAISYS/llaisys/src/device/cpu/cpu_runtime_api.cpp) | 75 | **12 个 CPU 函数** + 静态函数表 `RUNTIME_API` + `getRuntimeAPI()` 返回函数表指针 |

---

- [x] ### 5.1 src/device/cpu/cpu_runtime_api.cpp

```
#include "../runtime_api.hpp"

#include <cstdlib>
#include <cstring>

namespace llaisys::device::cpu {

namespace runtime_api {
int getDeviceCount() {
    return 1;
}
```

设备数量：CPU 永远返回 1——只有一个 CPU，"显卡"不存在多卡概念。

```
void setDevice(int) {
    // do nothing
}

void deviceSynchronize() {
    // do nothing
}
```

空操作：
- `setDevice`：CPU 无需切换设备，什么都不做
- `deviceSynchronize`：CPU 上的操作天然是同步的，无需等待

```
llaisysStream_t createStream() {
    return (llaisysStream_t)0; // null stream
}

void destroyStream(llaisysStream_t stream) {
    // do nothing
}
void streamSynchronize(llaisysStream_t stream) {
    // do nothing
}
```

Stream 空实现：CPU 上没有异步流的概念。`createStream` 返回 `0`（空流），`destroyStream`/`streamSynchronize` 都是空操作。

```
void *mallocDevice(size_t size) {
    return std::malloc(size);
}

void freeDevice(void *ptr) {
    std::free(ptr);
}

void *mallocHost(size_t size) {
    return mallocDevice(size);
}

void freeHost(void *ptr) {
    freeDevice(ptr);
}
```

内存管理：CPU 上 `mallocDevice` 就是标准 `std::malloc`。`mallocHost` 直接复用 `mallocDevice`（CPU 上 Device 和 Host 是同一块内存，没有 GPU 那种"显存 vs 主存"的区别）。同理 `freeHost` 复用 `freeDevice`。

```
void memcpySync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind) {
    std::memcpy(dst, src, size);
}

void memcpyAsync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind, llaisysStream_t stream) {
    memcpySync(dst, src, size, kind);
}
```

数据拷贝：CPU 上 `memcpySync` 就是标准 `std::memcpy`（忽略 `kind` 参数——CPU 上无所谓 Device→Host 还是 Host→Device）。`memcpyAsync` 直接调用 `memcpySync`（CPU 没有异步拷贝）。

```
static const LlaisysRuntimeAPI RUNTIME_API = {
    &getDeviceCount,
    &setDevice,
    &deviceSynchronize,
    &createStream,
    &destroyStream,
    &streamSynchronize,
    &mallocDevice,
    &freeDevice,
    &mallocHost,
    &freeHost,
    &memcpySync,
    &memcpyAsync};

} // namespace runtime_api

const LlaisysRuntimeAPI *getRuntimeAPI() {
    return &runtime_api::RUNTIME_API;
}
```

静态函数表 + 访问函数：`RUNTIME_API` 是 `static const` 全局变量，保存 12 个函数指针。`getRuntimeAPI()` 被 `device::getRuntimeAPI(LLAISYS_DEVICE_CPU)` 调用，返回函数表指针。

**与 NOOP 的对比**：

| 函数 | NOOP 实现 | CPU 实现 |
|------|-----------|----------|
| `getDeviceCount` | `return 0` | `return 1` |
| `mallocDevice` | 抛异常 | `std::malloc` |
| `freeDevice` | 抛异常 | `std::free` |
| `memcpySync` | 抛异常 | `std::memcpy` |
| 其他 8 个 | 抛异常 | 空操作 |

NOOP 是"这个设备不存在"，CPU 是"这个设备就是 CPU 自己"。**这就是统一接口的威力——换一个函数表，不换一行上层代码。**