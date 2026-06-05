# C/C++ 预定义宏详解

## 什么是预定义宏

预定义宏（Predefined Macros）是编译器在编译时自动定义的宏，无需用户通过 `#define` 手动声明。它们提供了关于编译器、平台、语言标准、编译时间等元信息，广泛用于条件编译和跨平台适配。

## 分类

### 一、语言标准宏

| 宏 | 含义 | 示例值 |
|----|------|--------|
| `__cplusplus` | C++ 标准版本 | `199711L` (C++98), `201103L` (C++11), `201703L` (C++17), `202002L` (C++20) |
| `__STDC__` | 是否为标准 C | 定义为 `1` |
| `__STDC_VERSION__` | C 标准版本 | `199901L` (C99), `201112L` (C11), `201710L` (C17) |
| `__STDC_HOSTED__` | 是否为托管实现 | `1` (有完整标准库), `0` (嵌入式/裸机) |

```cpp
// 使用示例：按 C++ 版本选择实现
#if __cplusplus >= 201703L
    // C++17 特性: if constexpr, structured bindings
    if constexpr (std::is_same_v<T, int>) { ... }
#else
    // 回退到旧方式
    ...
#endif
```

### 二、编译器标识宏

| 宏 | 编译器 | 典型用途 |
|----|--------|----------|
| `__GNUC__` | GCC | GCC 主版本号（如 `12`） |
| `__GNUC_MINOR__` | GCC | GCC 次版本号 |
| `__clang__` | Clang | 检测是否为 Clang |
| `__clang_major__` | Clang | Clang 主版本号 |
| `_MSC_VER` | MSVC | 版本编码（如 `1936` = VS 2022 17.6） |
| `_MSC_FULL_VER` | MSVC | 完整版本号 |
| `__INTEL_COMPILER` | Intel ICC | ICC 版本 |
| `__INTEL_LLVM_COMPILER` | Intel ICX | ICX 版本（基于 LLVM） |
| `__NVCC__` | NVIDIA CUDA | NVCC 编译器 |
| `__ARMCC_VERSION` | ARM Compiler | ARM 编译器版本 |

```cpp
// 使用示例：编译器特定优化
#ifdef _MSC_VER
    #define FORCE_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
    #define FORCE_INLINE __attribute__((always_inline)) inline
#else
    #define FORCE_INLINE inline
#endif
```

### 三、平台/操作系统宏

| 宏 | 平台 | 含义 |
|----|------|------|
| `_WIN32` | Windows (32/64 位) | 所有 Windows 平台（含 Win64） |
| `_WIN64` | Windows 64 位 | 仅在 64 位 Windows 定义 |
| `__linux__` | Linux | Linux 系统 |
| `__linux` | Linux | Linux 系统（旧式） |
| `__APPLE__` | macOS | 所有 Apple 平台 |
| `__MACH__` | macOS | Mach 内核 |
| `__ANDROID__` | Android | Android 系统 |
| `__FreeBSD__` | FreeBSD | FreeBSD 系统 |
| `__unix__` | Unix-like | 所有 Unix-like 系统 |
| `__EMSCRIPTEN__` | Emscripten | WebAssembly 编译 |
| `__CYGWIN__` | Cygwin | Windows 上的 POSIX 层 |
| `__MINGW32__` | MinGW | Windows 上的 GCC 移植 |
| `__MINGW64__` | MinGW-w64 | 64 位 MinGW |

```cpp
// 使用示例：平台特定代码
#ifdef _WIN32
    #include <windows.h>
    #define PATH_SEP '\\'
#elif defined(__linux__) || defined(__APPLE__)
    #include <unistd.h>
    #define PATH_SEP '/'
#endif
```

### 四、CPU 架构宏

| 宏 | 架构 | 说明 |
|----|------|------|
| `__x86_64__` | x86-64 | 64 位 x86（GCC/Clang） |
| `_M_X64` | x86-64 | 64 位 x86（MSVC） |
| `__i386__` | x86-32 | 32 位 x86（GCC/Clang） |
| `_M_IX86` | x86-32 | 32 位 x86（MSVC） |
| `__aarch64__` | ARM64 | 64 位 ARM（GCC/Clang） |
| `__arm__` | ARM32 | 32 位 ARM（GCC/Clang） |
| `_M_ARM64` | ARM64 | 64 位 ARM（MSVC） |
| `__powerpc64__` | POWER | 64 位 PowerPC |
| `__riscv` | RISC-V | RISC-V 架构 |
| `__SSE__`, `__SSE2__`, `__SSE3__`, `__SSE4_1__`, `__AVX__`, `__AVX2__`, `__AVX512F__` | x86 SIMD | SIMD 指令集支持 |

```cpp
// 使用示例：架构相关的 SIMD 选择
#if defined(__AVX512F__)
    #define SIMD_WIDTH 512
#elif defined(__AVX2__)
    #define SIMD_WIDTH 256
#elif defined(__SSE2__)
    #define SIMD_WIDTH 128
#else
    #define SIMD_WIDTH 0  // 纯标量
#endif
```

### 五、编译信息宏

| 宏 | 含义 | 示例 |
|----|------|------|
| `__FILE__` | 当前源文件名 | `"main.cpp"` |
| `__LINE__` | 当前行号 | `42` |
| `__FUNCTION__` | 当前函数名（C99） | `"main"` |
| `__PRETTY_FUNCTION__` | 带签名的函数名（GCC/Clang） | `"void foo(int)"` |
| `__DATE__` | 编译日期 | `"Jun 05 2026"` |
| `__TIME__` | 编译时间 | `"14:30:00"` |
| `__TIMESTAMP__` | 完整时间戳 | `"Mon Jun 05 14:30:00 2026"` |
| `__COUNTER__` | 编译期递增计数器 | `0`, `1`, `2`, ... |
| `__BASE_FILE__` | 编译入口文件（GCC/Clang） | `"main.cpp"` |

```cpp
// 使用示例：编译期断言消息
#define STATIC_ASSERT(cond, msg) \
    static_assert(cond, msg " [" __FILE__ ":" STRINGIFY(__LINE__) "]")
```

### 六、C/C++ 特性检测宏

| 宏 | 含义 |
|----|------|
| `__has_include(header)` | 头文件是否可用（C++17） |
| `__has_cpp_attribute(attr)` | 属性是否可用（C++20） |
| `__has_builtin(builtin)` | 内建函数是否可用（Clang） |
| `__has_feature(feature)` | 编译器特性是否可用（Clang） |
| `__has_extension(ext)` | 编译器扩展是否可用（Clang） |
| `__cpp_*` | C++ 特性测试宏（C++20 起） |

```cpp
// 使用示例：可选头文件检测
#if __has_include(<optional>)
    #include <optional>
    using std::optional;
#elif __has_include(<experimental/optional>)
    #include <experimental/optional>
    using std::experimental::optional;
#else
    #error "No <optional> available"
#endif
```

### 七、调试与优化宏

| 宏 | 含义 |
|----|------|
| `__OPTIMIZE__` | GCC/Clang 优化开启 |
| `__OPTIMIZE_SIZE__` | GCC/Clang `-Os` 优化 |
| `__NO_INLINE__` | GCC/Clang 无内联优化 |
| `__SANITIZE_ADDRESS__` | GCC AddressSanitizer 开启 |
| `_DEBUG` | MSVC Debug 模式 |
| `NDEBUG` | 标准 assert 禁用宏（Release 模式） |

```cpp
// 使用示例：调试日志
#ifdef NDEBUG
    #define LOG_DEBUG(...)  ((void)0)
#else
    #define LOG_DEBUG(fmt, ...) \
        fprintf(stderr, "[DEBUG] %s:%d: " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__)
#endif
```

### 八、DLL 导出/导入宏

| 宏 | 编译器 | 含义 |
|----|--------|------|
| `__declspec(dllexport)` | MSVC | 导出符号到 DLL |
| `__declspec(dllimport)` | MSVC | 从 DLL 导入符号 |
| `__attribute__((visibility("default")))` | GCC/Clang | 控制符号可见性 |

LLaiSys 项目中的实际使用模式：

```cpp
// include/llaisys.h
#ifdef _WIN32
    #ifdef LLAISYS_EXPORTS
        #define __export __declspec(dllexport)
    #else
        #define __export __declspec(dllimport)
    #endif
#else
    #define __export __attribute__((visibility("default")))
#endif
```

## 实战：跨平台代码骨架

```cpp
// platform.h — 统一的跨平台宏定义
#pragma once

// =========================================================================
// 1. 编译器检测
// =========================================================================
#if defined(_MSC_VER)
    #define COMPILER_MSVC 1
    #define COMPILER_GCC   0
    #define COMPILER_CLANG 0
#elif defined(__clang__)
    #define COMPILER_MSVC 0
    #define COMPILER_GCC   0
    #define COMPILER_CLANG 1
#elif defined(__GNUC__)
    #define COMPILER_MSVC 0
    #define COMPILER_GCC   1
    #define COMPILER_CLANG 0
#endif

// =========================================================================
// 2. 平台检测
// =========================================================================
#if defined(_WIN32)
    #define PLATFORM_WINDOWS 1
    #define PLATFORM_LINUX   0
    #define PLATFORM_MACOS   0
#elif defined(__APPLE__)
    #define PLATFORM_WINDOWS 0
    #define PLATFORM_LINUX   0
    #define PLATFORM_MACOS   1
#elif defined(__linux__)
    #define PLATFORM_WINDOWS 0
    #define PLATFORM_LINUX   1
    #define PLATFORM_MACOS   0
#endif

// =========================================================================
// 3. 架构检测
// =========================================================================
#if defined(__x86_64__) || defined(_M_X64)
    #define ARCH_X86_64 1
    #define ARCH_ARM64  0
#elif defined(__aarch64__) || defined(_M_ARM64)
    #define ARCH_X86_64 0
    #define ARCH_ARM64  1
#endif

// =========================================================================
// 4. 统一的导出/导入宏
// =========================================================================
#if PLATFORM_WINDOWS
    #ifdef BUILDING_DLL
        #define API_EXPORT __declspec(dllexport)
    #else
        #define API_EXPORT __declspec(dllimport)
    #endif
#else
    #define API_EXPORT __attribute__((visibility("default")))
#endif

// =========================================================================
// 5. 内联/对齐/分支预测
// =========================================================================
#if COMPILER_MSVC
    #define FORCE_INLINE __forceinline
    #define ALIGN(n)     __declspec(align(n))
    #define LIKELY(x)   (x)
    #define UNLIKELY(x) (x)
#else
    #define FORCE_INLINE __attribute__((always_inline)) inline
    #define ALIGN(n)     __attribute__((aligned(n)))
    #define LIKELY(x)   __builtin_expect(!!(x), 1)
    #define UNLIKELY(x) __builtin_expect(!!(x), 0)
#endif
```

## 查看预定义宏的方法

```bash
# GCC / Clang
gcc -dM -E - < /dev/null           # 所有预定义宏
gcc -dM -E -x c++ - < /dev/null    # C++ 模式
clang -dM -E -x c++ - < /dev/null -target x86_64-windows  # 交叉编译

# MSVC (从 VS Developer Command Prompt)
cl /EP /Zc:preprocessor /PD empty.c 2>nul

# 针对特定平台
gcc -dM -E -mavx2 - < /dev/null    # 有 AVX2 支持
gcc -dM -E -m32 - < /dev/null      # 32 位编译
```

## 常见陷阱

1. **`_WIN32` 在 64 位 Windows 上也会定义** — 用 `_WIN64` 区分位数
2. **`__GNUC__` 在 Clang 中也会定义** — Clang 伪装成 GCC，先检测 `__clang__`
3. **`__APPLE__` 和 `__MACH__` 同时定义** — 用 `__APPLE__` 检测 macOS 即可
4. **`__cplusplus` 在 MSVC 默认值偏低** — 需要 `/Zc:__cplusplus` 标志才能获得正确值
5. **`NDEBUG` 在 MSVC Debug 模式不定义** — 用 `_DEBUG` 代替
6. **交叉编译时宏反映目标平台** — 而非编译主机平台