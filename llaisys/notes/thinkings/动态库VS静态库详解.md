# 动态库 vs 静态库 详解

## 一、基本概念

### 静态库（Static Library）

- **Windows**: `.lib` 文件
- **Linux**: `.a` 文件（Archive）
- 在**链接阶段**被直接嵌入到可执行文件中
- 本质是一组 `.o`/`.obj` 文件的打包集合

### 动态库（Dynamic Library / Shared Library）

- **Windows**: `.dll` 文件（Dynamic Link Library）
- **Linux**: `.so` 文件（Shared Object）
- 在**运行时**（程序启动后或按需）被加载到内存
- 多个进程可以共享同一份动态库在内存中的副本

---

## 二、链接方式对比

```
静态链接：
┌──────────┐     ┌──────────┐     ┌──────────────┐
│  main.o  │  +  │  lib.a   │  →  │  a.out/.exe  │  (库代码嵌入)
└──────────┘     └──────────┘     └──────────────┘

动态链接：
┌──────────┐          ┌──────────────┐    运行时加载  ┌──────────┐
│  main.o  │  ─────→  │  a.out/.exe  │  ───────────→  │  lib.so  │
└──────────┘          └──────────────┘                │  lib.dll │
                                                      └──────────┘
```

---

## 三、详细对比

| 特性 | 静态库 | 动态库 |
|------|--------|--------|
| **链接时机** | 编译时（链接阶段） | 运行时（加载时或按需） |
| **文件大小** | 可执行文件较大（包含库代码） | 可执行文件较小 |
| **内存占用** | 每个进程独立一份 | 多进程共享同一份（代码段） |
| **更新方式** | 重新编译链接整个程序 | 只需替换库文件 |
| **启动速度** | 快（无需加载额外文件） | 稍慢（需加载和解析库） |
| **部署复杂度** | 简单（单文件分发） | 需确保库文件存在且版本匹配 |
| **版本兼容** | 编译时确定 | 可以运行时灵活切换 |
| **符号可见性** | 全部符号可见 | 可控制导出符号 |

---

## 四、项目中的实际应用

### LLAISYS 项目的库结构

项目 [xmake.lua](file:///c:/Code/LLAISYS/llaisys/xmake.lua) 中定义了两个编译目标：

```lua
-- 静态库：算子实现
target("llaisys-ops")
    set_kind("static")
    add_files("../src/ops/*/op.cpp")

-- 静态库：CPU 算子实现
target("llaisys-ops-cpu")
    set_kind("static")
    add_files("../src/ops/*/cpu/*.cpp")

-- 动态库：最终产出
target("llaisys")
    set_kind("shared")
    add_deps("llaisys-ops", "llaisys-ops-cpu")
```

**设计理念**：
- 算子实现（`llaisys-ops`、`llaisys-ops-cpu`）编译为**静态库**
- 最终产出 `llaisys.dll` 是**动态库**，把静态库链接进去
- 好处：内部模块化编译（静态库），对外提供统一接口（动态库）

### Python 加载动态库

```python
# ops.py 中通过 ctypes 加载 DLL
from .libllaisys import LIB_LLAISYS

# 调用 DLL 中的 C 函数
LIB_LLAISYS.llaisysArgmax(max_idx.lib_tensor(), max_val.lib_tensor(), vals.lib_tensor())
```

---

## 五、符号导出

### Windows 动态库

Windows 上的 DLL 需要**显式导出符号**：

```cpp
// 方式一：__declspec
__declspec(dllexport) void myFunction();

// 方式二：.def 文件
EXPORTS
    myFunction

// 使用方需要 __declspec(dllimport)
__declspec(dllimport) void myFunction();
```

LLASYS 项目使用 `extern "C"` 包装导出函数以避免 C++ 名称修饰（Name Mangling）：

```cpp
__C {
    void llaisysArgmax(llaisysTensor_t max_idx, llaisysTensor_t max_val, llaisysTensor_t vals) {
        llaisys::ops::argmax(max_idx->tensor, max_val->tensor, vals->tensor);
    }
    // ...
}
```

### Linux 动态库

Linux 默认所有符号都导出，但可以通过 `-fvisibility=hidden` 控制：

```cpp
__attribute__((visibility("default"))) void myFunction();
```

---

## 六、动态库加载方式

### 启动时加载（Load-time Linking）

程序启动时由操作系统自动加载依赖的 `.dll`/`.so`：

```cpp
// 编译时链接导入库（.lib / 链接器）
// 程序启动时自动加载 DLL
```

### 运行时加载（Run-time Dynamic Linking）

程序运行时按需加载：

```cpp
// Windows
HMODULE hLib = LoadLibrary("mylib.dll");
auto func = (FuncType)GetProcAddress(hLib, "functionName");
func(args);
FreeLibrary(hLib);

// Linux
void *handle = dlopen("mylib.so", RTLD_LAZY);
auto func = (FuncType)dlsym(handle, "functionName");
func(args);
dlclose(handle);
```

---

## 七、DLL Hell 问题

**问题**：多个程序依赖同一 DLL 的不同版本，导致不兼容。

**解决方案**：
1. **语义化版本**：`mylib-1.2.3.dll`
2. **Side-by-Side Assembly**（Windows）：通过 manifest 指定 DLL 版本
3. **静态链接**：彻底避免 DLL 依赖
4. **容器化**：Docker 等将依赖打包隔离

---

## 八、性能考量

| 方面 | 静态库 | 动态库 |
|------|--------|--------|
| **调用开销** | 直接函数调用，可能内联 | 通过 PLT/GOT 间接跳转，有少量开销 |
| **编译器优化** | 可跨模块内联和优化（LTO） | 优化受限 |
| **缓存效率** | 更好（代码在可执行文件内） | 需额外内存页 |
| **PIC 开销** | 无 | 位置无关代码需额外寄存器 |

---

## 九、选择建议

| 场景 | 推荐 |
|------|------|
| 库会被多个程序使用 | 动态库 |
| 需要热更新/插件系统 | 动态库 |
| 单文件分发 | 静态库 |
| 嵌入式系统 | 静态库 |
| 内部模块（不对外暴露） | 静态库 |
| 对外发布的 SDK | 动态库 + 静态库双版本 |
| 需要调用方链接 C ABI 的库 | 动态库（`extern "C"` 导出） |

---

## 十、xmake 中的配置

```lua
-- 静态库
target("mylib")
    set_kind("static")

-- 动态库
target("mylib")
    set_kind("shared")

-- 仅头文件库
target("mylib")
    set_kind("headeronly")
```