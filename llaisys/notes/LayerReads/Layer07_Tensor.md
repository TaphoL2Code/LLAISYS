# Layer07_Tensor

问：*Tensor 层前面 6 层已经铺垫了内存分配、设备管理、运行时——Tensor 还需要做什么？*

答：*Tensor 在底层基础设施之上实现了**形状/步长/偏移的元数据管理**、**数据加载**和**张量变换**（permute/view/slice）。这是上层算子操作的数据载体。*

- [x] ## 第 7 层：Tensor（张量） — 2 个文件

**理解"张量 = 元数据 + 存储 + 偏移"这一核心公式。**

| 序号 | 文件 | 行数 | 核心看点 |
|:--:|------|:--:|------|
| 7.1 | [src/tensor/tensor.hpp](file:///c:/Code/LLAISYS/llaisys/src/tensor/tensor.hpp) | 60 | `TensorMeta` 结构体（dtype/shape/strides）、`Tensor` 类（`_meta`/`_storage`/`_offset` 三件套）、`create()` 工厂方法 |
| 7.2 | [src/tensor/tensor.cpp](file:///c:/Code/LLAISYS/llaisys/src/tensor/tensor.cpp) | 205 | `create()` 的完整实现（分配内存 + 计算 strides）；`data()` 返回 `_storage->memory() + _offset`；`debug()` 支持跨设备打印 |

---

- [x] ### 7.1 src/tensor/tensor.hpp

```
#pragma once
#include "../core/llaisys_core.hpp"

#include <vector>
namespace llaisys {
class Tensor;
using tensor_t = std::shared_ptr<Tensor>;

struct TensorMeta {
    llaisysDataType_t dtype;
    std::vector<size_t> shape;
    std::vector<ptrdiff_t> strides;
};
```

`TensorMeta` 结构体：存储"逻辑形状"信息，与底层内存解耦。`shape` 是张量的维度大小，`strides` 是每个维度"跳一步"需要的元素个数（Row-major：`strides[0]` 最大，`strides[ndim-1] = 1`）。

```
class Tensor {
private:
    TensorMeta _meta;
    core::storage_t _storage;
    size_t _offset;
    Tensor(TensorMeta meta, core::storage_t storage, size_t offset = 0);
```

**Tensor 三件套**：

- `_meta`：形状/步长/数据类型元数据
- `_storage`：`shared_ptr<Storage>`——底层内存块的引用计数指针
- `_offset`：偏移量（字节），用于 slice/view 等"零拷贝"操作——新 Tensor 共享同一块 Storage，只改变 offset

构造函数 `private`——只能通过 `create()` 或元数据变换方法（permute/view/slice）创建。

```
public:
    static tensor_t create(
        const std::vector<size_t> &shape,
        llaisysDataType_t dtype,
        llaisysDeviceType_t device_type = LLAISYS_DEVICE_CPU,
        int device = 0);
    ~Tensor() = default;
```

`create()` 工厂方法：唯一的外部创建入口。默认 `device_type = CPU`，`device = 0`。

```
    // Info
    std::byte *data();
    const std::byte *data() const;
    size_t ndim() const;
    const std::vector<size_t> &shape() const;
    const std::vector<ptrdiff_t> &strides() const;
    llaisysDataType_t dtype() const;
    llaisysDeviceType_t deviceType() const;
    int deviceId() const;
    size_t numel() const;
    size_t elementSize() const;

    std::string info() const;
    void debug() const;

    bool isContiguous() const;
```

查询接口：`data()` 返回 `_storage->memory() + _offset`（含偏移量的原始指针）；`numel()` 返回总元素数（`shape[0] × shape[1] × ...`）；`elementSize()` 返回每个元素的字节数（`dsize(dtype)`）；`isContiguous()` 判断是否连续存储（`strides` 是否满足 row-major 条件）。

```
    // Meta Transform
    tensor_t permute(const std::vector<size_t> &order) const;
    tensor_t slice(size_t dim, size_t start, size_t end) const;
    tensor_t view(const std::vector<size_t> &shape) const;

    // Load data from host memory
    void load(const void *src);

    // Challenging features
    tensor_t contiguous() const;
    tensor_t reshape(const std::vector<size_t> &shape) const;
    tensor_t to(llaisysDeviceType_t device_type, int device = -1) const;
};
```

核心操作（当前多数为 `TO_BE_IMPLEMENTED()` 占位）：

- `permute(order)`：维度重排（如 `[0,1,2] → [2,1,0]`），重新排列 `strides` 和 `shape`
- `slice(dim, start, end)`：沿某维度切片，修改 `offset` 和 `shape[dim]`
- `view(shape)`：改变形状但不改变数据（要求连续存储）
- `load(src)`：从 Host 内存拷贝数据到 Tensor
- `contiguous()`：返回连续存储的副本
- `to(device_type)`：跨设备拷贝（CPU→GPU 或 GPU→CPU）

---

- [x] ### 7.2 src/tensor/tensor.cpp

```
tensor_t Tensor::create(const std::vector<size_t> &shape,
                        llaisysDataType_t dtype,
                        llaisysDeviceType_t device_type,
                        int device) {
    size_t ndim_ = shape.size();
    std::vector<ptrdiff_t> strides(ndim_);
    size_t stride = 1;
    for (size_t i = 1; i <= ndim_; i++) {
        strides[ndim_ - i] = stride;
        stride *= shape[ndim_ - i];
    }
    TensorMeta meta{dtype, shape, strides};
    size_t total_elems = stride;
    size_t dtype_size = utils::dsize(dtype);
```

**strides 计算（Row-major）**：从最后一维向前计算。例如 `shape = [2, 3, 4]`：
- `strides[2] = 1`（最后一维，相邻元素差 1）
- `strides[1] = 4`（跨第二维需跳 4 个元素）
- `strides[0] = 12`（跨第一维需跳 12 个元素）

`total_elems = stride`（循环结束后 `stride` 就是总元素数）。

```
    if (device_type == LLAISYS_DEVICE_CPU && core::context().runtime().deviceType() != LLAISYS_DEVICE_CPU) {
        auto storage = core::context().runtime().allocateHostStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    } else {
        core::context().setDevice(device_type, device);
        auto storage = core::context().runtime().allocateDeviceStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    }
}
```

**智能内存分配**：

- 如果目标是 CPU 但当前活跃设备是 GPU → 分配 **Host Storage**（锁页内存），便于 GPU↔CPU 传输
- 否则 → 调用 `setDevice` 切换到目标设备，分配 **Device Storage**

这是"推理侧"特有的优化——CPU Tensor 在 GPU 活跃时用 Host 内存，避免 `cudaMemcpy` 的 pageable→pinned 转换开销。

```
std::byte *Tensor::data() {
    return _storage->memory() + _offset;
}
```

**偏移量访问**：`data()` 返回存储基址 + 偏移量。这是 slice/view 等零拷贝操作的基础——新 Tensor 共享同一块 `_storage`，只是 `_offset` 不同。

```
void Tensor::debug() const {
    core::context().setDevice(this->deviceType(), this->deviceId());
    core::context().runtime().api()->device_synchronize();
    std::cout << this->info() << std::endl;
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        debug_print(this->data(), this->shape(), this->strides(), this->dtype());
    } else {
        auto tmp_tensor = create({this->_storage->size()}, this->dtype());
        core::context().runtime().api()->memcpy_sync(
            tmp_tensor->data(),
            this->data(),
            this->numel() * this->elementSize(),
            LLAISYS_MEMCPY_D2H);
        debug_print(tmp_tensor->data(), this->shape(), this->strides(), this->dtype());
    }
}
```

**跨设备调试打印**：

1. `setDevice` + `device_synchronize`：确保设备上所有操作完成，数据一致
2. CPU Tensor → 直接 `debug_print` 打印
3. GPU Tensor → 先创建临时 CPU Tensor，`memcpy_sync` 拷贝数据（D2H = Device to Host），再打印

`debug_print` 函数内部用 `switch(dtype)` 做类型分发，`reinterpret_cast` 为具体类型后调用模板函数 `print_data`——递归遍历多维数组，BF16/FP16 用 `utils::cast<float>()` 转为 float 后打印。

```
bool Tensor::isContiguous() const {
    TO_BE_IMPLEMENTED();
    return true;
}

tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}
```

**注意**：`permute`/`view`/`slice` 等当前返回的是 `new Tensor(_meta, _storage)`——即**共享同一块 Storage 和 Meta**，没有真正实现变换。这是 Project 1 的核心任务——实现这些零拷贝操作的元数据变换。

**Tensor 数据流**：
```
Tensor::create(shape, dtype, device)
  → 计算 strides（row-major）
  → 计算总字节数 = numel × dsize(dtype)
  → [CPU] allocateHostStorage / [GPU] allocateDeviceStorage
    → Runtime → NaiveAllocator → _api->malloc_device
  → new Tensor(meta, storage, offset=0)
  → 返回 shared_ptr<Tensor>
```