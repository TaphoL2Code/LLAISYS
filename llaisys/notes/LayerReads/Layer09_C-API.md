# Layer09_C-API

问：*为什么需要 C API 层？*

答：*C++ 的 **name mangling** 和 ABI 不兼容使得不同编译器编译的 C++ 库无法互调。C API 用 `extern "C"` 导出**稳定符号**，Python 通过 `ctypes` 加载 `.dll/.so` 后直接调用这些符号——这是 C++ → Python 的桥梁。*

- [x] ## 第 9 层：C API — 4 个文件

**理解"C++ 核心怎么暴露给 Python"。**

| 序号 | 文件 | 行数 | 核心看点 |
|:--:|------|:--:|------|
| 9.1 | [src/llaisys/llaisys_tensor.hpp](file:///c:/Code/LLAISYS/llaisys/src/llaisys/llaisys_tensor.hpp) | 10 | `LlaisysTensor` 包装结构体：`typedef struct { llaisys::tensor_t tensor; } LlaisysTensor` + `__C` 宏 |
| 9.2 | [src/llaisys/tensor.cc](file:///c:/Code/LLAISYS/llaisys/src/llaisys/tensor.cc) | 96 | `extern "C"` 导出：`tensorCreate`→`Tensor::create`、`tensorLoad`→`tensor->load`、`tensorView`→`tensor->view` 等 14 个函数 |
| 9.3 | [src/llaisys/runtime.cc](file:///c:/Code/LLAISYS/llaisys/src/llaisys/runtime.cc) | 13 | `llaisysSetContextRuntime()` / `llaisysGetRuntimeAPI()` → Context 和 RuntimeAPI 的 C 接口 |
| 9.4 | [src/llaisys/ops.cc](file:///c:/Code/LLAISYS/llaisys/src/llaisys/ops.cc) | 43 | `llaisysAdd`→`ops::add`、`llaisysLinear`→`ops::linear` 等 9 个算子的 C A |

---

- [x] ### 9.1 src/llaisys/llaisys_tensor.hpp

```
#pragma once
#include "llaisys/tensor.h"

#include "../tensor/tensor.hpp"

__C {
    typedef struct LlaisysTensor {
        llaisys::tensor_t tensor;
    } LlaisysTensor;
}
```

**关键设计**：`LlaisysTensor` 是一个 C 风格的 `struct`，内部只包含一个 `shared_ptr<Tensor>`。Python 通过 `ctypes.c_void_p` 持有这个结构体的指针（不透明句柄），调用 C API 函数时传入。

`__C` 宏定义在 `include/llaisys.h` 中：
- `extern "C"` 时 → `__C` 展开为 `extern "C" { }`，禁用 C++ name mangling
- 纯 C 编译器时 → 展开为空

---

- [x] ### 9.2 src/llaisys/tensor.cc

```
__C {
    llaisysTensor_t tensorCreate(
        size_t * shape, size_t ndim, llaisysDataType_t dtype,
        llaisysDeviceType_t device_type, int device_id) {
        std::vector<size_t> shape_vec(shape, shape + ndim);
        return new LlaisysTensor{llaisys::Tensor::create(shape_vec, dtype, device_type, device_id)};
    }
```

`tensorCreate`：将 C 数组 `shape` 转为 `vector<size_t>`，调用 `Tensor::create()`，用 `new` 分配 `LlaisysTensor`（在堆上），返回指针给 Python。

```
    void tensorDestroy(llaisysTensor_t tensor) {
        delete tensor;
    }
```

`tensorDestroy`：`delete` 释放 `LlaisysTensor`。`shared_ptr<Tensor>` 在析构时自动引用计数减一，最终释放底层 Storage。

```
    void *tensorGetData(llaisysTensor_t tensor) {
        return tensor->tensor->data();
    }
```

`tensorGetData`：返回原始数据指针。Python 拿到这个 `void*` 后，可以通过 `memcpy` 或 `torch.as_strided` 访问数据。

```
    void tensorGetShape(llaisysTensor_t tensor, size_t * shape) {
        std::copy(tensor->tensor->shape().begin(), tensor->tensor->shape().end(), shape);
    }
```

`tensorGetShape`：将 `vector<size_t>` 拷贝到用户提供的 `shape` 缓冲区。Python 需预分配 `(c_size_t * ndim)()` 数组。

```
    void tensorLoad(llaisysTensor_t tensor, const void *data) {
        tensor->tensor->load(data);
    }

    llaisysTensor_t tensorView(llaisysTensor_t tensor, size_t * shape, size_t ndim) {
        std::vector<size_t> shape_vec(shape, shape + ndim);
        return new LlaisysTensor{tensor->tensor->view(shape_vec)};
    }

    llaisysTensor_t tensorPermute(llaisysTensor_t tensor, size_t * order) {
        std::vector<size_t> order_vec(order, order + tensor->tensor->ndim());
        return new LlaisysTensor{tensor->tensor->permute(order_vec)};
    }

    llaisysTensor_t tensorSlice(llaisysTensor_t tensor, size_t dim, size_t start, size_t end) {
        return new LlaisysTensor{tensor->tensor->slice(dim, start, end)};
    }
}
```

**模式总结**：每个 C API 函数做三件事：

1. **参数转换**：C 类型 → C++ 类型（`size_t*` → `vector`, `void*` → `const void*`）
2. **调用 C++ 核心**：`tensor->tensor->xxx()`
3. **返回值转换**：C++ 返回值 → C 类型（`new LlaisysTensor{...}` → `llaisysTensor_t`）

---

- [x] ### 9.3 src/llaisys/runtime.cc

```
__C void llaisysSetContextRuntime(llaisysDeviceType_t device_type, int device_id) {
    llaisys::core::context().setDevice(device_type, device_id);
}

__C const LlaisysRuntimeAPI *llaisysGetRuntimeAPI(llaisysDeviceType_t device_type) {
    return llaisys::device::getRuntimeAPI(device_type);
}
```

两个 Runtime 相关的 C API：

- `llaisysSetContextRuntime`：Python 调用此函数切换设备，等价于 `context().setDevice()`
- `llaisysGetRuntimeAPI`：返回设备函数表指针。Python 通过 `ctypes` 的 `Structure` 解析函数表，直接调用 `malloc_device`/`memcpy_sync` 等底层函数

---

- [x] ### 9.4 src/llaisys/ops.cc

```
__C {
    void llaisysAdd(llaisysTensor_t c, llaisysTensor_t a, llaisysTensor_t b) {
        llaisys::ops::add(c->tensor, a->tensor, b->tensor);
    }
    void llaisysArgmax(llaisysTensor_t max_idx, llaisysTensor_t max_val, llaisysTensor_t vals) {
        llaisys::ops::argmax(max_idx->tensor, max_val->tensor, vals->tensor);
    }
    void llaisysEmbedding(llaisysTensor_t out, llaisysTensor_t index, llaisysTensor_t weight) {
        llaisys::ops::embedding(out->tensor, index->tensor, weight->tensor);
    }
    void llaisysLinear(llaisysTensor_t out, llaisysTensor_t in, llaisysTensor_t weight, llaisysTensor_t bias) {
        llaisys::ops::linear(out->tensor, in->tensor, weight->tensor, bias->tensor);
    }
    void llaisysRearrange(llaisysTensor_t out, llaisysTensor_t in) {
        llaisys::ops::rearrange(out->tensor, in->tensor);
    }
    void llaisysRmsNorm(llaisysTensor_t out, llaisysTensor_t in, llaisysTensor_t weight, float eps) {
        llaisys::ops::rms_norm(out->tensor, in->tensor, weight->tensor, eps);
    }
    void llaisysROPE(llaisysTensor_t out, llaisysTensor_t in, llaisysTensor_t pos_ids, float theta) {
        llaisys::ops::rope(out->tensor, in->tensor, pos_ids->tensor, theta);
    }
    void llaisysSelfAttention(llaisysTensor_t attn_val, llaisysTensor_t q, llaisysTensor_t k, llaisysTensor_t v, float scale) {
        llaisys::ops::self_attention(attn_val->tensor, q->tensor, k->tensor, v->tensor, scale);
    }
    void llaisysSwiGLU(llaisysTensor_t out, llaisysTensor_t gate, llaisysTensor_t up) {
        llaisys::ops::swiglu(out->tensor, gate->tensor, up->tensor);
    }
}
```

9 个算子的 C API 包装：每个函数都是简单的"解包 → 调用 → 返回"模式。`c->tensor` 从 `LlaisysTensor` 中取出 `tensor_t`（`shared_ptr<Tensor>`），传给 `ops::xxx()`。

**完整调用链（Python → C API → C++ 核心）**：
```
Python: ops.add(tensor_a, tensor_b)
  → Ops.add() [python/llaisys/ops.py]
    → LIB_LLAISYS.llaisysAdd(c.lib_tensor(), a.lib_tensor(), b.lib_tensor())
      → llaisysAdd() [src/llaisys/ops.cc]
        → c->tensor, a->tensor, b->tensor（解包 shared_ptr）
          → ops::add(c, a, b) [src/ops/add/op.cpp]
            → 参数校验 + 设备分发
              → cpu::add() [src/ops/add/cpu/add_cpu.cpp]
```