# C/C++ 预处理指令详解

## 一、什么是预处理

预处理（Preprocessing）是 C/C++ 编译流程的第一步，在真正的编译（词法分析、语法分析、代码生成）之前执行。预处理器根据预处理指令对源代码进行文本替换、条件编译、文件包含等操作，生成一个"纯净"的源代码供编译器处理。

编译流程：**预处理 → 编译 → 汇编 → 链接**

---

## 二、文件包含：`#include`

### 语法

```cpp
#include <header>     // 从系统标准路径搜索
#include "header"     // 先从当前目录搜索，再从系统路径搜索
```

### 行为

预处理器将指定文件的**全部内容**原样插入到 `#include` 所在位置。

### 搜索路径差异

| 形式 | 搜索顺序 |
|------|---------|
| `<>` | 编译器指定的系统/标准库路径（如 `/usr/include`） |
| `""` | 1. 当前源文件所在目录<br>2. `-I` 指定的目录<br>3. 系统标准路径 |

### 项目中的实际应用

```cpp
// 项目内部头文件：用 ""
#include "llaisys.h"
#include "../../utils.hpp"

// 标准库：用 <>
#include <cstddef>
#include <iostream>
```

### 头文件保护（Guard）

防止同一头文件被多次包含导致重复定义：

```cpp
// 传统方式
#ifndef MY_HEADER_H
#define MY_HEADER_H
// ... 头文件内容 ...
#endif

// 现代方式（推荐）
#pragma once
```

---

## 三、宏定义：`#define` / `#undef`

### 对象宏（Object-like Macro）

```cpp
#define PI 3.14159
#define MAX_SIZE 1024
```

预处理器将所有 `PI` 替换为 `3.14159`，`MAX_SIZE` 替换为 `1024`。

### 函数宏（Function-like Macro）

```cpp
#define SQUARE(x) ((x) * (x))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
```

**注意括号陷阱**：不加括号会导致错误：
```cpp
#define BAD_SQUARE(x) x * x
// BAD_SQUARE(1+2) → 1+2*1+2 = 5（错误！）
// SQUARE(1+2)    → ((1+2)*(1+2)) = 9（正确）
```

### 多行宏

使用 `\` 续行：

```cpp
#define LOG_ERROR(msg) \
    do { \
        std::cerr << "[ERROR] " << msg << std::endl; \
    } while (0)
```

`do { ... } while(0)` 惯用法确保宏在任何上下文中都作为单个语句安全使用。

### 取消宏定义

```cpp
#undef MAX_SIZE  // 之后 MAX_SIZE 不再有效
```

### 项目中的实际应用

项目 [check.hpp](file:///c:/Code/LLAISYS/llaisys/src/utils/check.hpp) 大量使用函数宏封装错误处理：

```cpp
#define EXCEPTION_UNSUPPORTED_DEVICE \
    do { \
        std::cerr << "[ERROR] Unsupported device" << EXCEPTION_LOCATION_MSG << std::endl; \
        throw std::runtime_error("Unsupported device"); \
    } while (0)

#define CHECK_ARGUMENT(condition, message) \
    do { \
        if (!(condition)) { \
            std::cerr << "[ERROR] Invalid argument: " << message << EXCEPTION_LOCATION_MSG << std::endl; \
            throw std::invalid_argument(message); \
        } \
    } while (0)

#define CHECK_SAME(ERR, FIRST, ...) \
    do { \
        for (const auto &arg___ : {__VA_ARGS__}) { \
            if (FIRST != arg___) { \
                { ERR; } \
            } \
        } \
    } while (0)
```

### 可变参数宏：`__VA_ARGS__`

```cpp
#define CHECK_SAME_DTYPE(FIRST, ...) \
    CHECK_SAME(EXCEPTION_DATATYPE_MISMATCH, FIRST, __VA_ARGS__)
```

`__VA_ARGS__` 代表 `...` 对应的所有参数。

---

## 四、条件编译：`#if` / `#ifdef` / `#ifndef` / `#else` / `#elif` / `#endif`

### 基本形式

```cpp
#ifdef MACRO_NAME
    // 如果 MACRO_NAME 已定义，编译此段
#endif

#ifndef MACRO_NAME
    // 如果 MACRO_NAME 未定义，编译此段
#endif

#if CONDITION
    // 如果 CONDITION 为真（非零），编译此段
#elif OTHER_CONDITION
    // 否则如果 OTHER_CONDITION 为真
#else
    // 否则
#endif
```

### `defined()` 操作符

```cpp
#if defined(DEBUG) && defined(VERBOSE)
    // DEBUG 和 VERBOSE 都已定义
#endif
```

### 项目中的实际应用

设备分派中的条件编译：

```cpp
switch (max_idx->deviceType()) {
case LLAISYS_DEVICE_CPU:
    return cpu::argmax(...);
#ifdef ENABLE_NVIDIA_API
case LLAISYS_DEVICE_NVIDIA:
    TO_BE_IMPLEMENTED();
    return;
#endif
default:
    EXCEPTION_UNSUPPORTED_DEVICE;
}
```

当 `ENABLE_NVIDIA_API` 未定义时，NVIDIA 分支的代码在预处理阶段就被删除，编译后的二进制不含 GPU 相关代码。

---

## 五、预定义宏

编译器自动提供的宏：

| 宏 | 含义 |
|----|------|
| `__FILE__` | 当前源文件名（字符串） |
| `__LINE__` | 当前行号（整数） |
| `__func__` | 当前函数名（字符串） |
| `__DATE__` | 编译日期 |
| `__TIME__` | 编译时间 |
| `__cplusplus` | C++ 版本（如 `201703L` 表示 C++17） |

### 项目中的实际应用

```cpp
#define EXCEPTION_LOCATION_MSG \
    " from " << __func__ << " at " << __FILE__ << ":" << __LINE__ << "."
```

错误信息会包含函数名、文件名和行号，便于定位问题。

---

## 六、字符串化 `#` 和连接 `##`

### `#` 操作符：字符串化

将宏参数转换为字符串字面量：

```cpp
#define STRINGIFY(x) #x
// STRINGIFY(hello) → "hello"
```

### `##` 操作符：Token 连接

将两个 token 连接成一个：

```cpp
#define CONCAT(a, b) a##b
// CONCAT(var, 1) → var1
```

---

## 七、`#pragma` 指令

平台/编译器特定的指令：

```cpp
#pragma once           // 头文件只包含一次（替代 include guard）
#pragma pack(1)        // 设置结构体对齐为 1 字节
#pragma warning(disable: 4996)  // 禁用特定警告（MSVC）
```

---

## 八、`#error` 和 `#line`

### `#error`：编译时错误

```cpp
#ifndef REQUIRED_MACRO
#error "REQUIRED_MACRO must be defined"
#endif
```

### `#line`：修改行号和文件名

```cpp
#line 100 "custom_file.cpp"
// 此后 __LINE__ 从 100 开始，__FILE__ 为 "custom_file.cpp"
```

---

## 九、预处理指令完整列表

| 指令 | 用途 |
|------|------|
| `#include` | 包含文件 |
| `#define` | 定义宏 |
| `#undef` | 取消宏定义 |
| `#if` | 条件编译（表达式） |
| `#ifdef` | 如果已定义 |
| `#ifndef` | 如果未定义 |
| `#else` | 条件分支 |
| `#elif` | 条件分支 |
| `#endif` | 结束条件编译 |
| `#pragma` | 编译器特定指令 |
| `#error` | 产生编译错误 |
| `#line` | 修改行号/文件名 |
| `#` | 字符串化 |
| `##` | Token 连接 |

---

## 十、最佳实践

1. **头文件保护**：统一使用 `#pragma once`
2. **宏中的括号**：参数和整体都要加括号
3. **多语句宏**：用 `do { ... } while(0)` 包裹
4. **避免复杂宏**：能用 `constexpr`/`inline` 函数替代的尽量不用宏
5. **条件编译**：用于平台差异和设备分派，而非控制程序逻辑
6. **命名规范**：宏名全部大写，用下划线分隔