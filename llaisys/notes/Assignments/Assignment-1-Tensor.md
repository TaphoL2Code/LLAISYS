# Assignment #1: Tensor — 事件清单

## 主要修改文件
| 文件 | 修改内容 |
|------|----------|
| [`src/tensor/tensor.hpp`](file:///c:/Code/LLAISYS/llaisys/src/tensor/tensor.hpp) | 无需修改（声明已存在） |
| [`src/tensor/tensor.cpp`](file:///c:/Code/LLAISYS/llaisys/src/tensor/tensor.cpp) | **实现 `load`, `isContiguous`, `view`, `permute`, `slice`** |
| [`src/core/runtime/runtime.cpp`](file:///c:/Code/LLAISYS/llaisys/src/core/runtime/runtime.cpp) | 可能需要查看（理解 Runtime 和 Storage 的关系） |
| [`src/core/storage/storage.hpp`](file:///c:/Code/LLAISYS/llaisys/src/core/storage/storage.hpp) | 可能需要查看（理解内存块管理） |

## 需更改的配置
**无需更改配置**。代码编译和链接已在 `xmake.lua` 中配置好。

## 建议阅读层次

> 参考 [遍历式阅读顺序](../thinking/遍历式阅读顺序.md)

| 层次 | 内容 | 必读文件 |
|:--:|------|------|
| **第 0 层** | 公共头文件 | [include/llaisys.h](file:///c:/Code/LLAISYS/llaisys/include/llaisys.h) — 认识 `LLAISYS_MEMCPY_H2D` 等枚举 |
| **第 1 层** | 内部工具 | [src/utils/check.hpp](file:///c:/Code/LLAISYS/llaisys/src/utils/check.hpp) — `CHECK_ARGUMENT`/`ASSERT`/`TO_BE_IMPLEMENTED` |
| **第 2 层** | Allocator | [src/core/allocator/naive_allocator.cpp](file:///c:/Code/LLAISYS/llaisys/src/core/allocator/naive_allocator.cpp) — malloc/free 封装 |
| **第 3 层** | Context | [src/core/context/context.cpp](file:///c:/Code/LLAISYS/llaisys/src/core/context/context.cpp) — `thread_local` 单例、`setDevice()` |
| **第 4 层** | Device | [src/device/runtime_api.hpp](file:///c:/Code/LLAISYS/llaisys/src/device/runtime_api.hpp) — `LlaisysRuntimeAPI` 函数表结构 |
| **第 5 层** | CPU Runtime | [src/device/cpu/cpu_runtime_api.cpp](file:///c:/Code/LLAISYS/llaisys/src/device/cpu/cpu_runtime_api.cpp) — `memcpy_sync` 在 CPU 上就是 `memcpy` |
| **第 6 层** | Storage & Runtime | [src/core/storage/storage.cpp](file:///c:/Code/LLAISYS/llaisys/src/core/storage/storage.cpp) — `data()` = `_memory + _offset` |
| **第 7 层** | **Tensor（本任务）** | [src/tensor/tensor.hpp](file:///c:/Code/LLAISYS/llaisys/src/tensor/tensor.hpp) + [src/tensor/tensor.cpp](file:///c:/Code/LLAISYS/llaisys/src/tensor/tensor.cpp) |

---

## 背景知识

### Tensor 对象三个核心字段
- **`storage`**：`shared_ptr<Storage>`，真正持有数据内存的智能指针，可被多个 Tensor 共享
- **`offset`**：在 Storage 中的字节偏移量，用于 view/slice/permute 等零拷贝操作
- **`meta`**（`TensorMeta`）：包含 `dtype`（数据类型）、`shape`（各维度大小）、`strides`（各维度的字节步长）

### 行主序（Row-Major）存储
LLAISYS 使用行主序。例如 shape = (2, 3, 5) 的 float32 张量：
- strides = (60, 20, 4)，即 (3×5×4, 5×4, 4)
- 偏移公式：`offset = sum(stride[i] * index[i])`

---

## 任务清单

### 任务 1.1：实现 `load(const void *src)`

- [x] **理解当前 Context/Runtime 架构**
  - 阅读 [src/core/context/context.hpp](file:///c:/Code/LLAISYS/llaisys/src/core/context/context.hpp) — 理解 `context()` 获取线程局部 Context
  - 阅读 [src/core/runtime/runtime.hpp](file:///c:/Code/LLAISYS/llaisys/src/core/runtime/runtime.hpp) — 理解 `api()` 获取 Runtime API
  - 理解 `LlaisysRuntimeAPI` 中的 `memcpy_sync` 函数指针

- [x] **实现 void load(const void *src)**
  - 从 `_meta` 获取数据，计算需要拷贝的字节数（`numel() * elementSize()`）
  - 通过 `_storage->runtime()` 或当前 Context 获取 Runtime API
  - 调用 `memcpy_sync(dst, src, size, H2D)` 将 CPU 数据拷贝到设备端
  - 目标地址 = `_storage->memory() + _offset`

实现

```
void Tensor::load(const void *src_) {
    size_t size = numel() * elementSize();
    core::context().setDevice(deviceType(), deviceId());
    const auto *api = core::context().runtime().api();
    api->memcpy_sync(data(), src_, size, LLAISYS_MEMCPY_H2D);
}
```

### 任务 1.2：实现 `isContiguous() const`

- [x] **理解连续性的定义**
  - 张量是行主序的
  - 从最后一维开始，当前维的 stride 应等于下一维的 stride × shape[next_dim]
  - 即：`strides[i] == strides[i+1] * shape[i+1]`，且最后一维 `strides[-1] == elementSize()`

- [x] **实现 bool isContiguous() const**
  - 从倒数第二维向前遍历
  - 检查每次 stride 关系是否匹配
  - 返回 true/false

实现：

```
bool Tensor::isContiguous() const {
    size_t ndim_ = ndim();
    if (ndim_ == 0) return true;
    size_t stride = 1;
    for (size_t i = 0; i < ndim_; i++) {
        if (_meta.strides[ndim_ - 1 - i] != (ptrdiff_t)stride) {
            return false;
        }
        stride *= _meta.shape[ndim_ - 1 - i];
    }
    return true;
}
```

### 任务 1.3：实现 `view(const std::vector<size_t> &shape) const`

- [x] **理解 view 的约束条件**
  - 总元素数不能变（`numel()` 不变）
  - 新 shape 必须能从原 shape 的连续维度合并/拆分得到
  - 关键：只能沿"被一起遍历的维度"合并、只能将"步长连续的维度"拆分

- [x] **实现 tensor_t view(const std::vector<size_t> &shape) const**
  - 验证 `numel()` 一致
  - 计算新的 strides，确保新视图与原存储兼容
  - 如果不兼容（例如原 strides 不规则时），抛出异常
  - 创建新 Tensor，共享同一 Storage，只改变 `_meta.shape` 和 `_meta.strides`
  - 典型例子：(2, 3, 5) strides=(60,20,4) → view(2, 15) strides=(60,4) ✓

实现：

```
tensor_t Tensor::view(const std::vector<size_t> &shape) const {
    CHECK_ARGUMENT(isContiguous(), "view() requires a contiguous tensor");
    size_t total = std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<size_t>());
    CHECK_ARGUMENT(total == numel(), "view() shape must have the same number of elements as the original tensor");
    size_t ndim_ = shape.size();
    std::vector<ptrdiff_t> new_strides(ndim_);
    size_t stride = 1;
    for (size_t i = 0; i < ndim_; i++) {
        new_strides[ndim_ - 1 - i] = stride;
        stride *= shape[ndim_ - 1 - i];
    }
    TensorMeta new_meta{_meta.dtype, shape, new_strides};
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, _offset));
}
```



### 任务 1.4：实现 `permute(const std::vector<size_t> &order) const`

- [x] **理解 permute 的语义**
  - 重新排列维度顺序，不涉及数据拷贝（零开销）
  - 只改变 `shape` 和 `strides` 的顺序

- [x] **实现 tensor_t permute(const std::vector<size_t> &order) const**
  - 验证 order 是有效排列（包含 0~ndim-1 各一次）
  - 按 `order` 重排 `shape` 和 `strides`
  - 创建新 Tensor，共享同一 Storage

实现：

```
tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    CHECK_ARGUMENT(order.size() == ndim(), "permute() order size must match tensor ndim");
    std::vector<size_t> new_shape(ndim());
    std::vector<ptrdiff_t> new_strides(ndim());
    for (size_t i = 0; i < ndim(); i++) {
        new_shape[i] = _meta.shape[order[i]];
        new_strides[i] = _meta.strides[order[i]];
    }
    TensorMeta new_meta{_meta.dtype, new_shape, new_strides};
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, _offset));
}
```



### 任务 1.5：实现 `slice(size_t dim, size_t start, size_t end) const`

- [x] **理解 slice 的语义**
  - 沿指定维度取 `[start, end)` 范围的子张量
  - 不涉及数据拷贝
  - 只需改变该维度的 shape 和 offset

- [x] **实现 tensor_t slice(size_t dim, size_t start, size_t end) const**
  - 验证 dim 有效、start < end、end ≤ shape[dim]
  - 新 shape：该维度变为 `end - start`，其他维度不变
  - 新 strides：与原 strides 相同
  - 新 offset：原 offset + `start * strides[dim]`
  - 创建新 Tensor，共享同一 Storage

实现：

```
tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
    CHECK_ARGUMENT(dim < ndim(), "slice() dim is out of range");
    CHECK_ARGUMENT(start < _meta.shape[dim], "slice() start is out of range");
    CHECK_ARGUMENT(end <= _meta.shape[dim], "slice() end is out of range");
    CHECK_ARGUMENT(start < end, "slice() start must be less than end");
    std::vector<size_t> new_shape = _meta.shape;
    new_shape[dim] = end - start;
    size_t new_offset = _offset + start * _meta.strides[dim] * elementSize();
    TensorMeta new_meta{_meta.dtype, new_shape, _meta.strides};
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, new_offset));
}
```



### 任务 1.6：验证与提交

ctrl+s保存->xmake->xmake install->进入文件目录->进入虚拟环境->运行测试代码

![](C:\Code\LLAISYS\llaisys\screenshot\Assignment-1-Tensor\Snipaste_2026-06-03_08-36-59.png)

- [x] **运行张量测试**
  
  - ```bash
    python test/test_tensor.py
    ```
  - 确认所有测试通过（绿色 `Test passed!`）

验证：

![](C:\Code\LLAISYS\llaisys\screenshot\Assignment-1-Tensor\Snipaste_2026-06-03_08-37-35.png)

![](C:\Code\LLAISYS\llaisys\screenshot\Assignment-1-Tensor\Snipaste_2026-06-03_08-37-44.png)

![](C:\Code\LLAISYS\llaisys\screenshot\Assignment-1-Tensor\Snipaste_2026-06-03_08-37-52.png)

![](C:\Code\LLAISYS\llaisys\screenshot\Assignment-1-Tensor\Snipaste_2026-06-03_08-38-01.png)

- [x] **提交代码并推送**
  - ```bash
    git add src/tensor/tensor.cpp
    git commit -m "Assignment #1: Implement Tensor operations"
    git push
    ```
  - 检查 GitHub Actions 中 Assignment #1 的自动测试通过

验证：



---

## 自我验证方法
- [x] 调用 `tensor.load(torch_tensor.data_ptr())` 后，`check_equal` 通过
- [x] `isContiguous()` 对刚创建的张量返回 true，对 permute 后返回 false
- [x] `view` 改变 shape 后，`check_equal` 仍通过
- [x] `permute` 后 shape 和 strides 正确重排
- [x] `slice` 后取出的子张量数据与原张量对应位置一致

## 提示
- `view` 是最容易出错的：不能简单只改变 shape，要用 strides 验证兼容性
- 零拷贝操作（view/permute/slice）后多个 Tensor 共享同一 Storage，注意生命周期管理（使用 `shared_ptr`）
- `load` 需要理解设备内存模型：CPU → 设备端的数据传输