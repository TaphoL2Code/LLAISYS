# Layer01_内部工具

问：*为什么需要内部工具层？*

答：*为整个项目提供**类型转换、错误检查和通用宏**，是所有其他源文件的第一层依赖。*

- [x] ## 第 1 层：内部工具 — 3 个文件

**项目内部的底层工具，被所有其他文件引用。**

| 序号 | 文件 | 行数 | 核心看点 |
|:--:|------|:--:|------|
| 1.1 | [src/utils.hpp](file:///c:/Code/LLAISYS/llaisys/src/utils.hpp) | 37 | 全局工具头文件：`dsize()`（dtype→字节数）、`dtype_to_str()`、`check_device()` 等声明；`#include` 了所有子工具头文件 |
| 1.2 | [src/utils/check.hpp](file:///c:/Code/LLAISYS/llaisys/src/utils/check.hpp) | 35 | 断言宏：`CHECK_ARGUMENT(cond, msg)`（抛 `std::invalid_argument`）、`ASSERT(cond, msg)`（抛 `std::runtime_error`）、`TO_BE_IMPLEMENTED()`（抛 `std::logic_error`） |
| 1.3 | [src/utils/types.hpp](file:///c:/Code/LLAISYS/llaisys/src/utils/types.hpp) + [src/utils/types.cpp](file:///c:/Code/LLAISYS/llaisys/src/utils/types.cpp) | 29+46 | `dsize()` 的 switch-case 实现（I8=1, I64=8, F32=4, BF16=2...）；`dtype_to_str()` 和 `device_to_str()` 字符串转换 |

---

- [x] ### 1.1 src/utils.hpp

**伞形头文件**（umbrella header），**将子模块头文件聚合到一起**，方便项目其他部分统一包含

```
#pragma once
#include "utils/check.hpp"
#include "utils/types.hpp"
```

聚合头文件：不定义任何新内容，只做两件事——(1) 用 `#pragma once` 防止重复包含，(2) 聚合引入 `check.hpp` 和 `types.hpp` 两个子模块。项目其他文件只需 `#include "../utils.hpp"` 即可获得所有工具宏和函数，无需分别引用。这是 C++ 项目中常见的"伞形头文件"模式。

---

- [x] ### 1.2 src/utils/check.hpp

**轻量级的运行时检查与错误报告工具集**

```
#include <iostream>
#include <stdexcept>
```

引入标准库：`iostream` 用于错误输出，`stdexcept` 提供标准异常类。

```
#define EXCEPTION_LOCATION_MSG \
    " from " << __func__ << " at " << __FILE__ << ":" << __LINE__ << "."
```

位置信息宏：展开为 `" from functionName at path/to/file.cpp:123."` 的流式字符串。利用编译器的 `__func__`、`__FILE__`、`__LINE__` 预定义宏，在异常消息中自动附加调用位置——出 bug 时直接定位到具体函数和行号。

```
#define EXCEPTION_UNSUPPORTED_DEVICE                                                      \
    do {                                                                                  \
        std::cerr << "[ERROR] Unsupported device" << EXCEPTION_LOCATION_MSG << std::endl; \
        throw std::runtime_error("Unsupported device");                                   \
    } while (0)
```

不支持的设备异常：打印错误日志并抛出 `std::runtime_error`。`do { ... } while(0)` 是 C/C++ 宏的标准写法——确保宏在任何上下文中（if/else/for 等）都作为一个完整语句执行，不会因分号产生语法歧义。

```
#define EXCEPTION_UNSUPPORTED_DATATYPE(DT__)              \
    do {                                                  \
        std::cerr << "[ERROR] Unsupported data type: "    \
                  << llaisys::utils::dtype_to_str(DT__)   \
                  << EXCEPTION_LOCATION_MSG << std::endl; \
        throw std::runtime_error("Unsupported device");   \
    } while (0)
```

不支持的数据类型异常：与上面类似，但接受一个参数 `DT__`——会调用 `dtype_to_str()` 将枚举值转为可读字符串（如 `"bfloat16"`），打印具体是哪种类型不支持。

```
#define CHECK_ARGUMENT(condition, message)                                                 \
    do {                                                                                   \
        if (!(condition)) {                                                                \
            std::cerr << "[ERROR] Invalid argument: " << message << EXCEPTION_LOCATION_MSG \
                      << std::endl;                                                        \
            throw std::invalid_argument(message);                                          \
        }                                                                                  \
    } while (0)
```

参数校验宏：`!condition` 为真时抛 `std::invalid_argument`。用于验证函数参数——如 `CHECK_ARGUMENT(device_id >= 0, "device_id must be non-negative")`。失败时打印自定义 message 和位置信息。

```
#define ASSERT(condition, message)                            \
    do {                                                      \
        if (!(condition)) {                                   \
            std::cerr << "[ERROR] " << message << std::endl   \
                      << "Assertion failed: " << #condition   \
                      << EXCEPTION_LOCATION_MSG << std::endl; \
            throw std::runtime_error("Assertion failed");     \
        }                                                     \
    } while (0)
```

断言宏：与 `CHECK_ARGUMENT` 的区别在于——抛 `std::runtime_error` 而不是 `invalid_argument`，且会用 `#condition`（字符串化运算符）把条件表达式本身也打印出来，便于调试。

```
#define TO_BE_IMPLEMENTED()                                                                   \
    do {                                                                                      \
        std::cerr << "[ERROR] Unimplemented function" << EXCEPTION_LOCATION_MSG << std::endl; \
        throw std::runtime_error("Unimplemented function");                                   \
    } while (0)
```

占位标记宏：嵌入尚未实现的函数中，调用时直接报错退出。这是"骨架代码"的标记方式——先把函数签名和调用链写好，具体实现用 `TO_BE_IMPLEMENTED()` 占位，后续逐功能填充。当前项目大部分算子的 `op.cpp` 都使用此宏。

```
#define CHECK_SAME(ERR, FIRST, ...)                \
    do {                                           \
        for (const auto &arg___ : {__VA_ARGS__}) { \
            if (FIRST != arg___) {                 \
                { ERR; }                           \
            }                                      \
        }                                          \
    } while (0)
```

通用比较宏：将 `FIRST` 与可变参数 `__VA_ARGS__` 中的每个值比较，任一不相等则执行 `ERR` 宏。`{__VA_ARGS__}` 利用 C++11 的初始化列表将可变参数打包成可迭代的 `initializer_list`。

```
#define CHECK_SAME_SHAPE(FIRST, ...) \
    CHECK_SAME(EXCEPTION_SHAPE_MISMATCH, FIRST, __VA_ARGS__)

#define CHECK_SAME_DTYPE(FIRST, ...) \
    CHECK_SAME(EXCEPTION_DATATYPE_MISMATCH, FIRST, __VA_ARGS__)

#define CHECK_SAME_DEVICE(FIRST, ...)                            \
    do {                                                         \
        for (const auto &tensor___ : {__VA_ARGS__}) {            \
            if (FIRST->deviceType() != tensor___->deviceType()   \
                || FIRST->deviceId() != tensor___->deviceId()) { \
                { EXCEPTION_DEVICE_MISMATCH; }                   \
            }                                                    \
        }                                                        \
    } while (0)
```

一致性检查宏：

- `CHECK_SAME_SHAPE`：确保所有输入张量形状一致（逐元素算子如 add 的前置条件）
- `CHECK_SAME_DTYPE`：确保所有输入张量数据类型一致
- `CHECK_SAME_DEVICE`：确保所有张量在同一设备上（CPU 和 GPU 不能混用）。此宏直接访问 `deviceType()`/`deviceId()` 而不是用 `==` 比较，因为设备检查同时需要类型和编号都匹配

---

- [x] ### 1.3 src/utils/types.hpp + src/utils/types.cpp

**自定义半精度浮点类型及转换基础设施**

```
#include "llaisys.h"

#include <iostream>
#include <stdexcept>

namespace llaisys {
struct CustomFloat16 {
    uint16_t _v;
};
typedef struct CustomFloat16 fp16_t;

struct CustomBFloat16 {
    uint16_t _v;
};
typedef struct CustomBFloat16 bf16_t;
```

自定义浮点类型：C++ 标准库没有原生的 FP16 和 BF16 类型，项目用 `uint16_t` 包裹的结构体模拟。`CustomFloat16` 和 `CustomBFloat16` 都是 2 字节大小，但位模式（1-5-10 vs 1-8-7）的解释方式不同。

```
inline size_t dsize(llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_I8:     return sizeof(int8_t);   // 1
    case LLAISYS_DTYPE_I32:    return sizeof(int32_t);  // 4
    case LLAISYS_DTYPE_F32:    return sizeof(float);    // 4
    case LLAISYS_DTYPE_BF16:   return 2;                // bfloat16
    case LLAISYS_DTYPE_F16:    return 2;                // 16-bit float
    case LLAISYS_DTYPE_INVALID:
    default: throw std::invalid_argument("...");
    }
}
```

数据类型→字节数映射（`inline` 表示在头文件中定义，编译时内联展开）：Tensor 创建时通过此函数计算需要分配的内存字节数（`numel × dsize(dtype)`）。

```
inline const char *dtype_to_str(llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:  return "float32";
    case LLAISYS_DTYPE_BF16: return "bfloat16";
    ...
    }
}
```

数据类型→字符串映射：用于错误消息和调试输出，将枚举值转为人类可读的字符串。

```
float _f16_to_f32(fp16_t val);
fp16_t _f32_to_f16(float val);
float _bf16_to_f32(bf16_t val);
bf16_t _f32_to_bf16(float val);
```

浮点转换函数声明（实现在 `types.cpp` 中，详见 [LLAISYS_DTYPE_BF16详细解读](../thinkings/LLAISYS_DTYPE_BF16详细解读.md)）：

- `_bf16_to_f32`：左移 16 位补零（BF16 和 FP32 指数相同，只需补零尾数）
- `_f32_to_bf16`：round-to-nearest-even 截断低 16 位
- `_f16_to_f32`：需处理指数偏置转换（FP16 偏置 15 → FP32 偏置 127）和次正规数
- `_f32_to_f16`：同样需指数偏置转换和舍入

```
template <typename TypeTo, typename TypeFrom>
TypeTo cast(TypeFrom val) {
    if constexpr (std::is_same<TypeTo, TypeFrom>::value) {
        return val;  // 同类型直接返回
    } else if constexpr (std::is_same<TypeTo, bf16_t>::value && ...) {
        return _f32_to_bf16(val);
    } else if constexpr (std::is_same<TypeFrom, bf16_t>::value && ...) {
        return _bf16_to_f32(val);
    } else {
        return static_cast<TypeTo>(val);  // 普通类型用标准转换
    }
}
```

通用类型转换模板（`src/utils/types.hpp` 后半段）：利用 `if constexpr` 在**编译期**根据类型组合选择转换路径。核心逻辑：

- 同类型 → 直接返回
- BF16/FP16 ↔ float → 调用专用转换函数
- BF16/FP16 ↔ 其他类型 → 先转 float 再转目标类型（float 是中间桥梁）
- 普通类型之间 → 标准 `static_cast`

`cast<T>()` 被 `add_cpu.cpp` 等算子 kernel 广泛使用，使得模板化的 kernel 代码能统一处理 BF16/FP16/float 等多种数据类型。

```
// types.cpp
float _bf16_to_f32(bf16_t val) {
    uint32_t bits32 = static_cast<uint32_t>(val._v) << 16;  // 左移 16 位
    float out;
    std::memcpy(&out, &bits32, sizeof(out));  // 位模式解释为 float
    return out;
}

bf16_t _f32_to_bf16(float val) {
    uint32_t bits32;
    std::memcpy(&bits32, &val, sizeof(bits32));
    const uint32_t rounding_bias = 0x00007FFF + ((bits32 >> 16) & 1);
    uint16_t bf16_bits = static_cast<uint16_t>((bits32 + rounding_bias) >> 16);
    return bf16_t{bf16_bits};
}
```

BF16 转换实现（详见 [BF16 解读文档](../thinkings/LLAISYS_DTYPE_BF16详细解读.md#三转换算法详解srcutilstypescpp)）：BF16→F32 只需左移补零（因为指数位宽相同），F32→BF16 用 round-to-nearest-even 舍入。