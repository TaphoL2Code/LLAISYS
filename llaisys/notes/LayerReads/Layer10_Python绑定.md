# Layer10_Python绑定

问：*Python 绑定层做了什么？*

答：*三层封装：**ctypes 类型映射** → **ctypes 函数签名绑定** → **Pythonic 类封装**。让用户能用 `Tensor(shape, dtype)` 和 `ops.add(a, b)` 这样自然的 Python 语法操作 C++ 推理引擎。*

- [x] ## 第 10 层：Python 绑定 — 10 个文件

**理解"Python 怎么优雅地调用 C++ 核心"。**

| 序号 | 文件 | 行数 | 核心看点 |
|:--:|------|:--:|------|
| 10.1 | [python/llaisys/libllaisys/llaisys_types.py](file:///c:/Code/LLAISYS/llaisys/python/llaisys/libllaisys/llaisys_types.py) | 63 | **ctypes 类型映射层**：`DataType`/`DeviceType`/`MemcpyKind` 三个 `IntEnum`；`llaisysDataType_t = ctypes.c_int` 等 ctypes 类型别名 |
| 10.2 | [python/llaisys/libllaisys/__init__.py](file:///c:/Code/LLAISYS/llaisys/python/llaisys/libllaisys/__init__.py) | 55 | **动态库加载**：`load_shared_library()` → `ctypes.CDLL(`llaisys.dll`)`；调用 `load_runtime`/`load_tensor`/`load_ops` 绑定函数签名 |
| 10.3 | [python/llaisys/libllaisys/tensor.py](file:///c:/Code/LLAISYS/llaisys/python/llaisys/libllaisys/tensor.py) | 78 | **函数签名绑定**：`lib.tensorCreate.argtypes = [POINTER(c_size_t), c_size_t, ...]`；`lib.tensorCreate.restype = llaisysTensor_t` |
| 10.4 | [python/llaisys/libllaisys/runtime.py](file:///c:/Code/LLAISYS/llaisys/python/llaisys/libllaisys/runtime.py) | 48 | **函数表结构体**：`class LlaisysRuntimeAPI(Structure)` 含 12 个 `CFUNCTYPE` 字段；`load_runtime()` 绑定 `llaisysGetRuntimeAPI` 等函数签名 |
| 10.5 | [python/llaisys/libllaisys/ops.py](file:///c:/Code/LLAISYS/llaisys/python/llaisys/libllaisys/ops.py) | 36 | 9 个算子 ctypes 函数签名绑定：`lib.llaisysAdd.argtypes = [llaisysTensor_t, ...]` |
| 10.6 | [python/llaisys/tensor.py](file:///c:/Code/LLAISYS/llaisys/python/llaisys/tensor.py) | 97 | **Python 类封装**：`class Tensor` —— `__init__`→`tensorCreate`、`load`→`tensorLoad`、`view`→`tensorView`、`__del__`→`tensorDestroy` |
| 10.7 | [python/llaisys/runtime.py](file:///c:/Code/LLAISYS/llaisys/python/llaisys/runtime.py) | 68 | `class RuntimeAPI` —— 包装 `LlaisysRuntimeAPI` 结构体，提供 `malloc_device`/`memcpy_sync` 等 Python 方法 |
| 10.8 | [python/llaisys/ops.py](file:///c:/Code/LLAISYS/llaisys/python/llaisys/ops.py) | 55 | `class Ops` —— 9 个 `@staticmethod`，每个包装一个算子 C API |
| 10.9 | [python/llaisys/__init__.py](file:///c:/Code/LLAISYS/llaisys/python/llaisys/__init__.py) | 20 | 模块入口：`from .tensor import Tensor`、`from .ops import Ops`、`from .models import *` |
| 10.10 | [python/llaisys/models/qwen2.py](file:///c:/Code/LLAISYS/llaisys/python/llaisys/models/qwen2.py) | 33 | `class Qwen2` —— 模型加载 + 推理循环（骨架代码，`TO_BE_IMPLEMENTED`） |

---

- [x] ### 10.1 libllaisys/llaisys_types.py — ctypes 类型映射

```
from ctypes import c_int, c_void_p
from enum import IntEnum

class DataType(IntEnum):
    INVALID = 0
    BYTE = 1
    ...
    BF16 = 19

llaisysDataType_t = ctypes.c_int

class DeviceType(IntEnum):
    CPU = 0
    NVIDIA = 1
    COUNT = 2

class MemcpyKind(IntEnum):
    H2H = 0
    H2D = 1
    D2H = 2
    D2D = 3
```

**三重映射**：C 枚举 → `IntEnum`（Python 层面友好的枚举）→ `ctypes.c_int`（传递给 C 函数的实际类型）。`IntEnum` 继承 `int`，可以和 `c_int` 直接互转。

---

- [x] ### 10.2 libllaisys/__init__.py — 动态库加载

```
def load_shared_library():
    lib_dir = Path(__file__).parent
    if sys.platform.startswith("linux"):
        libname = "libllaisys.so"
    elif sys.platform == "win32":
        libname = "llaisys.dll"
    elif sys.platform == "darwin":
        libname = "llaisys.dylib"
    lib_path = os.path.join(lib_dir, libname)
    return ctypes.CDLL(str(lib_path))

LIB_LLAISYS = load_shared_library()
load_runtime(LIB_LLAISYS)
load_tensor(LIB_LLAISYS)
load_ops(LIB_LLAISYS)
```

**平台自适应加载**：根据操作系统选择 `.dll`/`.so`/`.dylib`。`ctypes.CDLL()` 加载动态库，`load_xxx()` 函数绑定 C 函数的参数类型和返回值类型——这是 `ctypes` 的必需步骤，否则 Python 不知道怎么传参。

---

- [x] ### 10.4 libllaisys/runtime.py — 函数表结构体

```
get_device_count_api = CFUNCTYPE(c_int)
set_device_api = CFUNCTYPE(None, c_int)
device_synchronize_api = CFUNCTYPE(None)
...
malloc_device_api = CFUNCTYPE(c_void_p, c_size_t)
free_device_api = CFUNCTYPE(None, c_void_p)
memcpy_sync_api = CFUNCTYPE(None, c_void_p, c_void_p, c_size_t, llaisysMemcpyKind_t)

class LlaisysRuntimeAPI(Structure):
    _fields_ = [
        ("get_device_count", get_device_count_api),
        ("set_device", set_device_api),
        ...
        ("memcpy_async", memcpy_async_api),
    ]
```

**C 结构体映射到 Python**：`CFUNCTYPE` 定义函数指针类型，`Structure._fields_` 定义结构体字段。`ctypes` 自动处理内存布局对齐。Python 拿到 `LlaisysRuntimeAPI*` 指针后，用 `_api.contents.malloc_device(size)` 直接调用 C 函数。

---

- [x] ### 10.6 python/llaisys/tensor.py — Python 类封装

```
class Tensor:
    def __init__(self, shape, dtype=DataType.F32, device=DeviceType.CPU, device_id=0, tensor=None):
        if tensor:
            self._tensor = tensor  # 从已有句柄构造（view/permute 等）
        else:
            _shape = (c_size_t * len(shape))(*shape)
            self._tensor = LIB_LLAISYS.tensorCreate(_shape, c_size_t(len(shape)), ...)

    def __del__(self):
        if hasattr(self, "_tensor") and self._tensor is not None:
            LIB_LLAISYS.tensorDestroy(self._tensor)
```

**RAII 包装**：`__init__` 创建 C 端 Tensor，`__del__` 自动销毁。Python 的垃圾回收触发 `tensorDestroy`，`shared_ptr` 确保底层 Storage 正确释放。

```
    def view(self, *shape):
        _shape = (c_size_t * len(shape))(*shape)
        return Tensor(tensor=LIB_LLAISYS.tensorView(self._tensor, _shape, c_size_t(len(shape))))
```

**返回新 Python 对象**：`tensorView` 返回新的 `llaisysTensor_t`，包装成新的 `Tensor` Python 对象。两者共享底层 Storage（`shared_ptr` 引用计数 +1），但有不同的 `_tensor` 句柄。

---

- [x] ### 10.10 python/llaisys/models/qwen2.py — 模型推理

```
class Qwen2:
    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        model_path = Path(model_path)
        for file in sorted(model_path.glob("*.safetensors")):
            data_ = safetensors.safe_open(file, framework="numpy", device="cpu")
            for name_ in data_.keys():
                pass  # TODO: load the model weights

    def generate(self, inputs, max_new_tokens=None, top_k=1, top_p=0.8, temperature=0.8):
        return []  # TODO: Implement generate function
```

**骨架代码**：`Qwen2` 类目前只有框架——`safetensors` 解析模型文件、参数接口定义好了，但内部的权重复制和推理循环都是 `TODO`。这是 Project 3（模型推理）的核心任务。

**三层架构总览**：
```
Python 用户层              C API 桥接层               C++ 核心层
─────────────────────────────────────────────────────────────────
Tensor(shape, dtype)  →  tensorCreate()       →  Tensor::create()
ops.add(a, b)         →  llaisysAdd()         →  ops::add()
RuntimeAPI(CPU)       →  llaisysGetRuntimeAPI →  getRuntimeAPI()
Qwen2.generate()      →  llaisysLinear()...   →  ops::linear()...
```