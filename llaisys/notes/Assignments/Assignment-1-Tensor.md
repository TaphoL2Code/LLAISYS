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
    // 计算需要拷贝的总字节数：元素个数 × 每个元素的字节数
    size_t size = numel() * elementSize(); 
    // 根据张量当前所在的设备类型（如 CPU/CUDA）和设备 ID，设置当前上下文的活动设备
    // 这样后续的内存操作（如 memcpy）就会在正确的目标设备上执行
    core::context().setDevice(deviceType(), deviceId()); // 找活动设备（目标地址）
    // 获取当前运行时（runtime）的底层 API 接口指针，该接口提供了同步内存拷贝等底层函数
    const auto *api = core::context().runtime().api();   // 找运行时对象
    // 调用底层 API 的同步内存拷贝函数，将数据从主机（src_）拷贝到设备端张量的数据指针（data()）
    // LLAISYS_MEMCPY_H2D 表示拷贝方向：Host to Device（主机到设备）
    api->memcpy_sync(data(), src_, size, LLAISYS_MEMCPY_H2D); // 拷贝
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
    // 获取张量的维度数
    size_t ndim_ = ndim();
    // 0 维张量（标量）规定为连续
    if (ndim_ == 0) return true;
    // 从最内层（最后一维）开始的期望步长，初始为 1（最后一个元素相邻间隔为 1）
    size_t stride = 1;
    // 从最后一个维度向第一个维度遍历（i = 0 对应最后一维，i = ndim_-1 对应第一维）
    for (size_t i = 0; i < ndim_; i++) {
        // 检查当前维度的实际步长是否等于期望的步长
        // 注意：_meta.strides 存储的是每个维度的实际步长（字节数或元素个数间隔）
        if (_meta.strides[ndim_ - 1 - i] != (ptrdiff_t)stride) {
            return false;   // 一旦有任何维度步长不符，说明内存不连续
        }
        // 更新期望步长：下一个外层维度的步长 = 当前维度步长 × 当前维度的长度
        stride *= _meta.shape[ndim_ - 1 - i];
    }
    // 所有维度都符合行优先连续的条件，返回 true
    return true;
}
```

理解Stride:[【闪客】你管这破玩意叫张量？_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1SB2gBFEyu/?spm_id_from=333.337.search-card.all.click&vd_source=58f16a4c0a88e0fb3322fd63829f82ce)的前半部分

![](C:\Code\LLAISYS\llaisys\notes\thinkings\image\Snipaste_2026-06-03_15-23-03.png)

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
    // 要求原张量在内存中连续，否则无法安全地重新解释形状
    CHECK_ARGUMENT(isContiguous(), "view() requires a contiguous tensor");
    
    // 计算新形状下的总元素个数：将 shape 中所有维度相乘
    size_t total = std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<size_t>());
    // 新形状的总元素数必须与原张量的元素总数相等，否则视图非法
    CHECK_ARGUMENT(total == numel(), "view() shape must have the same number of elements as the original tensor");
    
    size_t ndim_ = shape.size();                 // 新形状的维度数
    std::vector<ptrdiff_t> new_strides(ndim_);  // 存储计算出的新步长（每个维度相邻元素间隔的字节数）
    size_t stride = 1;                          // 从最后一个维度开始累积步长（行优先，即 C 连续）
    
    // 倒序计算每个维度的步长：最内层（最后一维）步长为1，向前依次为 shape[i+1]*stride[i+1]
    for (size_t i = 0; i < ndim_; i++) {
        // 从后往前填充 new_strides：第 ndim_-1-i 维（即倒数第 i+1 维）的步长设为当前累积值
        new_strides[ndim_ - 1 - i] = stride;
        // 更新步长：乘以当前维度的长度，作为前一个维度的步长
        stride *= shape[ndim_ - 1 - i];
    }
    
    // 构造新的元数据：使用原来的数据类型、新的形状、计算出的新步长
    TensorMeta new_meta{_meta.dtype, shape, new_strides};
    // 创建新的 Tensor 对象，共享原有的存储（_storage）和偏移量（_offset），但使用新的视图元数据
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
    // 检查 order 的长度必须等于当前张量的维度数，否则抛出参数错误
    CHECK_ARGUMENT(order.size() == ndim(), "permute() order size must match tensor ndim");
    // 创建新形状向量，长度与当前维度数相同
    std::vector<size_t> new_shape(ndim());
    // 创建新步长向量，长度与当前维度数相同
    std::vector<ptrdiff_t> new_strides(ndim());
    // 遍历新维度的每个位置 i
    for (size_t i = 0; i < ndim(); i++) {
        // 新形状的第 i 维 = 原形状中第 order[i] 维的大小
        new_shape[i] = _meta.shape[order[i]];
        // 新步长的第 i 维 = 原步长中第 order[i] 维的大小
        new_strides[i] = _meta.strides[order[i]];
    }
    // 构造新的张量元数据：数据类型沿用原张量，使用计算出的新形状和新步长
    TensorMeta new_meta{_meta.dtype, new_shape, new_strides};
    // 创建新的 Tensor 对象，共享原张量的存储（_storage）和偏移（_offset），但使用新的视图元数据
    // 返回智能指针管理的 Tensor 实例
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
    // 检查切片的维度 dim 是否在有效范围内 [0, ndim())
    CHECK_ARGUMENT(dim < ndim(), "slice() dim is out of range");
    // 检查起始索引 start 是否小于该维度的长度
    CHECK_ARGUMENT(start < _meta.shape[dim], "slice() start is out of range");
    // 检查结束索引 end 是否不超过该维度的长度
    CHECK_ARGUMENT(end <= _meta.shape[dim], "slice() end is out of range");
    // 确保 start < end，即切片范围非空
    CHECK_ARGUMENT(start < end, "slice() start must be less than end");
    // 复制原张量的形状（std::vector<size_t>）
    std::vector<size_t> new_shape = _meta.shape;
    // 修改新形状：指定维度的大小变为 end - start（切片后的长度）
    new_shape[dim] = end - start;
    // 计算新视图在底层存储中的偏移量（字节为单位）
    // 原偏移 _offset + (start * 该维度的步长 * 每个元素字节数)
    size_t new_offset = _offset + start * _meta.strides[dim] * elementSize();
    // 创建新的张量元数据：数据类型不变，使用新的形状，但步长数组与原张量相同（因为切片不改变步长）
    TensorMeta new_meta{_meta.dtype, new_shape, _meta.strides};
    // 构造新的 Tensor 对象，共享原存储（_storage）和计算出的新偏移量，使用新元数据
    // 返回智能指针管理的 Tensor 实例（tensor_t 是 shared_ptr<Tensor> 的别名）
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