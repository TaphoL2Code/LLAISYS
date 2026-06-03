# Layer11_构建系统

问：*构建系统做什么？*

答：*将 20+ 个 C++ 源文件编译成 **1 个共享库**（`llaisys.dll`/`libllaisys.so`），Python 通过 `ctypes.CDLL` 加载。xmake 管理 9 个 target 之间的依赖关系和编译选项。*

- [x] ## 第 11 层：构建系统 — 2 个文件

**理解"C++ 源码怎么变成 Python 可加载的 .dll/.so"。**

| 序号 | 文件 | 行数 | 核心看点 |
|:--:|------|:--:|------|
| 11.1 | [xmake.lua](file:///c:/Code/LLAISYS/llaisys/xmake.lua) | 122 | 主构建文件：7 个 target（llaisys-utils→llaisys-device→llaisys-core→llaisys-tensor→llaisys-ops→llaisys→llaisys_python）；条件编译 NVIDIA；`mode.debug/release` |
| 11.2 | [xmake/cpu.lua](file:///c:/Code/LLAISYS/llaisys/xmake/cpu.lua) | 27 | CPU 子构建：`llaisys-device-cpu` 和 `llaisys-ops-cpu` 两个 target |

---

- [x] ### 11.1 xmake.lua — 主构建文件

```
add_rules("mode.debug", "mode.release")
set_encodings("utf-8")
add_includedirs("include")
```

全局配置：`debug`/`release` 模式切换（`xmake f -m debug` 或 `xmake f -m release`）；`utf-8` 编码；`include/` 目录加入头文件搜索路径。

```
includes("xmake/cpu.lua")

option("nv-gpu")
    set_default(false)
    set_showmenu(true)
    set_description("Whether to compile implementations for Nvidia GPU")
option_end()

if has_config("nv-gpu") then
    add_defines("ENABLE_NVIDIA_API")
    includes("xmake/nvidia.lua")
end
```

**条件编译**：`xmake f --nv-gpu=y` 启用 NVIDIA GPU 支持。启用后定义 `ENABLE_NVIDIA_API` 宏，C++ 代码中的 `#ifdef ENABLE_NVIDIA_API` 分支生效。`includes("xmake/nvidia.lua")` 加载 CUDA 编译配置（当前文件不存在，Project 2 需创建）。

```
target("llaisys-utils")
    set_kind("static")
    set_languages("cxx17")
    add_files("src/utils/*.cpp")
```

**Target 依赖链**（从底层到顶层）：

| Target | 类型 | 依赖 | 源文件 |
|--------|------|------|--------|
| `llaisys-device-cpu` | static | — | `src/device/cpu/*.cpp` |
| `llaisys-ops-cpu` | static | `llaisys-tensor` | `src/ops/*/cpu/*.cpp` |
| `llaisys-utils` | static | — | `src/utils/*.cpp` |
| `llaisys-device` | static | `llaisys-utils` + `llaisys-device-cpu` | `src/device/*.cpp` |
| `llaisys-core` | static | `llaisys-utils` + `llaisys-device` | `src/core/*/*.cpp` |
| `llaisys-tensor` | static | `llaisys-core` | `src/tensor/*.cpp` |
| `llaisys-ops` | static | `llaisys-ops-cpu` | `src/ops/*/*.cpp` |
| `llaisys` | **shared** | 以上全部 | `src/llaisys/*.cc` |

**关键**：`llaisys` 是唯一的 `shared`（共享库）target，链接所有 `static` 库。`static` 库的好处是编译时内联优化 + 链接后只有一个 `.dll` 文件。

```
target("llaisys")
    set_kind("shared")
    add_deps("llaisys-utils", "llaisys-device", "llaisys-core", "llaisys-tensor", "llaisys-ops")
    add_files("src/llaisys/*.cc")
    after_install(function (target)
        if is_plat("windows") then
            os.cp("bin/*.dll", "python/llaisys/libllaisys/")
        end
        if is_plat("linux") then
            os.cp("lib/*.so", "python/llaisys/libllaisys/")
        end
    end)
```

**安装后复制**：`xmake install` 后自动将 `.dll`/`.so` 复制到 `python/llaisys/libllaisys/` 目录，Python 的 `load_shared_library()` 从该目录加载。

---

- [x] ### 11.2 xmake/cpu.lua — CPU 子构建

```
target("llaisys-device-cpu")
    set_kind("static")
    add_files("../src/device/cpu/*.cpp")

target("llaisys-ops-cpu")
    set_kind("static")
    add_deps("llaisys-tensor")
    add_files("../src/ops/*/cpu/*.cpp")
```

两个 CPU 专用 target：

- `llaisys-device-cpu`：编译 `cpu_runtime_api.cpp` + `cpu_resource.cpp`
- `llaisys-ops-cpu`：编译 `src/ops/*/cpu/*.cpp`（通配符匹配所有算子 CPU kernel，当前只有 `add/cpu/add_cpu.cpp`）

**为什么把 CPU 拆出来？** GPU 构建时，`llaisys-device-cpu` 和 `llaisys-ops-cpu` 仍然被编译（因为所有 target 都依赖它们），但 xmake.lua 中会额外 `includes("xmake/nvidia.lua")` 添加 CUDA target。CPU 和 GPU 的 target 并列，`llaisys` shared 库链接所有。

**编译选项**（每个 target 都设置）：
- `set_languages("cxx17")`：C++17 标准（用于 `if constexpr`、`std::byte` 等）
- `set_warnings("all", "error")`：所有警告视为错误
- `-fPIC`：位置无关代码（Linux 下生成 `.so` 必需）
- `-Wno-unknown-pragmas`：忽略未知 pragma 警告（OpenMP 的 `#pragma omp` 在无 OpenMP 时会产生警告）