# LLAISYS_DTYPE_BF16 详细解读

## 一、什么是 BF16？

**BF16（Brain Floating Point 16）** 是由 Google Brain 团队提出的 16 位浮点格式，专为深度学习训练和推理优化设计。

### 位布局

```
FP32:  [S][--------8 位指数--------][-----------23 位尾数-----------]
       1     8                        23                          = 32 bits

FP16:  [S][--5 位指数--][----------10 位尾数----------]
       1     5                 10                         = 16 bits

BF16:  [S][--------8 位指数--------][--7 位尾数--]
       1     8                        7                  = 16 bits
```

| 格式 | 符号位 | 指数位 | 尾数位 | 总位数 | 数值范围 | 精度 |
|:--|:--:|:--:|:--:|:--:|:--|:--|
| **FP32** | 1 | 8 | 23 | 32 | ±3.4×10³⁸ | ~7 位十进制 |
| **FP16** | 1 | 5 | 10 | 16 | ±65,504 | ~3 位十进制 |
| **BF16** | 1 | 8 | 7 | 16 | ±3.4×10³⁸ | ~2 位十进制 |

### 核心特点

**BF16 = FP32 的高 16 位截断。** BF16 保留了和 FP32 完全相同的 8 位指数，只牺牲了 16 位尾数精度。这意味着：

- ✅ **数值范围与 FP32 相同**（±3.4×10³⁸），不会像 FP16 那样在 ±65,504 处溢出
- ✅ **FP32 ↔ BF16 转换极其简单**：只需截断/补零低 16 位，无需处理指数偏置
- ✅ **训练友好**：梯度不会因范围溢出而丢失，无需 loss scaling
- ❌ **精度较低**：只有约 2 位十进制有效数字

---

## 二、在 LLAISYS 中的定义

### 2.1 枚举值（include/llaisys.h）

在 [include/llaisys.h](file:///c:/Code/LLAISYS/llaisys/include/llaisys.h) 中，`LLAISYS_DTYPE_BF16 = 19`，是枚举中的最后一个数据类型的：

```c
typedef enum {
    LLAISYS_DTYPE_INVALID = 0,
    ...
    LLAISYS_DTYPE_BF16 = 19,   // ← 第 19 号，排在所有其他类型之后
} llaisysDataType_t;
```

### 2.2 内部结构体（src/utils/types.hpp）

在 [src/utils/types.hpp](file:///c:/Code/LLAISYS/llaisys/src/utils/types.hpp) 中定义了 BF16 的内部存储结构：

```cpp
struct CustomBFloat16 {
    uint16_t _v;       // 16 位原始位模式
};
typedef struct CustomBFloat16 bf16_t;
```

与 FP16 的 `CustomFloat16` 结构完全对称，都是 `uint16_t` 包裹。不同之处在于 `_v` 中存储的位模式含义不同。

### 2.3 字节大小（dsize）

在 `dsize()` 函数中：

```cpp
case LLAISYS_DTYPE_BF16:
    return 2; // bfloat16 — 和 FP16 一样占 2 字节
```

### 2.4 字符串名称（dtype_to_str）

```cpp
case LLAISYS_DTYPE_BF16:
    return "bfloat16";
```

---

## 三、转换算法详解（src/utils/types.cpp）

### 3.1 BF16 → FP32（左移补零）

```cpp
float _bf16_to_f32(bf16_t val) {
    uint32_t bits32 = static_cast<uint32_t>(val._v) << 16;  // 左移 16 位
    float out;
    std::memcpy(&out, &bits32, sizeof(out));                // 位模式解释为 float
    return out;
}
```

**原理图解**：

```
BF16 位模式:    [S][E7 E6 E5 E4 E3 E2 E1 E0][M6 M5 M4 M3 M2 M1 M0]
                     ↓ 左移 16 位
FP32 位模式:    [S][E7 E6 E5 E4 E3 E2 E1 E0][M6 M5 M4 M3 M2 M1 M0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
                 ↑ 符号位原样保留     ↑ 8 位指数完全相同           ↑ 7 位尾数后补 16 个零
```

**为什么这么简单？** 因为 BF16 和 FP32 的指数位宽相同（都是 8 位），指数偏置也相同（都是 127），所以不需要任何指数转换——直接把 16 位左移到高 16 位，低 16 位补零即可。

**精度损失**：BF16 的 7 位尾数变成 FP32 的 23 位尾数时，低 16 位全为 0。这意味着转换回 FP32 的数值会比原始 FP32 精度低，但数值范围完全不变。

### 3.2 FP32 → BF16（舍入截断）

```cpp
bf16_t _f32_to_bf16(float val) {
    uint32_t bits32;
    std::memcpy(&bits32, &val, sizeof(bits32));  // 读取 FP32 的位模式

    const uint32_t rounding_bias = 0x00007FFF    // 0111 1111 1111 1111
                                 + ((bits32 >> 16) & 1);  // 加上第 16 位的值

    uint16_t bf16_bits = static_cast<uint16_t>((bits32 + rounding_bias) >> 16);
    return bf16_t{bf16_bits};
}
```

**步骤分解**：

```
输入 FP32:     [S][E7 E6 E5 E4 E3 E2 E1 E0][M22 M21 ... M16 M15 M14 ... M0]
                                                    ↑ 高 7 位保留  ↑ 低 16 位舍入

Step 1: rounding_bias = 0x7FFF + bit16
        0x7FFF = 0111 1111 1111 1111（二进制）
        + bit16（FP32 位模式中第 16 位的值，即要保留的最高丢弃位）

Step 2: (bits32 + rounding_bias) >> 16
        → 加 bias 后右移 16 位 = 舍入到最近偶数 (round-to-nearest-even)

Step 3: 取低 16 位 = BF16 位模式
```

**舍入策略：Round-to-Nearest-Even（IEEE 754 默认舍入模式）**

| 被丢弃的低 16 位值 | 舍入行为 |
|:--|:--|
| < 0x8000（小于一半） | 向下舍（截断） |
| > 0x8000（大于一半） | 向上舍（+1） |
| == 0x8000（正好一半） | 向偶数舍（看 bit16，=0 则截断，=1 则 +1） |

`rounding_bias` 公式中的 `+ ((bits32 >> 16) & 1)` 就是"向偶数舍"的关键：当正好一半时，bit16=0 则 bias=0x7FFF（向下），bit16=1 则 bias=0x8000（向上）。

### 3.3 通用类型转换（cast 模板）

在 `types.hpp` 中，`cast<T>()` 模板函数通过 `if constexpr` 在编译期分派：

```cpp
// BF16 → float（直接调 _bf16_to_f32）
else if constexpr (std::is_same<TypeFrom, bf16_t>::value && std::is_same<TypeTo, float>::value) {
    return _bf16_to_f32(val);
}

// float → BF16（直接调 _f32_to_bf16）
else if constexpr (std::is_same<TypeTo, bf16_t>::value && std::is_same<TypeFrom, float>::value) {
    return _f32_to_bf16(val);
}

// BF16 → 其他类型（先转 float 再转目标类型）
else if constexpr (std::is_same<TypeFrom, bf16_t>::value && !std::is_same<TypeTo, float>::value) {
    return static_cast<TypeTo>(_bf16_to_f32(val));
}

// 其他类型 → BF16（先转 float 再转 BF16）
else if constexpr (std::is_same<TypeTo, bf16_t>::value && !std::is_same<TypeFrom, float>::value) {
    return _f32_to_bf16(static_cast<float>(val));
}
```

**转换链**：`任意类型 → float → BF16` 或 `BF16 → float → 任意类型`。float 是中间桥梁。

---

## 四、在项目中的实际使用

### 4.1 算子计算（add_cpu.cpp）

在 [src/ops/add/cpu/add_cpu.cpp](file:///c:/Code/LLAISYS/llaisys/src/ops/add/cpu/add_cpu.cpp) 中，BF16 的加法运算：

```cpp
template <typename T>
void add_(T *c, const T *a, const T *b, size_t numel) {
    for (size_t i = 0; i < numel; i++) {
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            // BF16/FP16 不能直接做加法（CPU 没有 BF16 硬件指令）
            // 先各自转 float → float 加法 → 转回 BF16
            c[i] = llaisys::utils::cast<T>(
                llaisys::utils::cast<float>(a[i]) + llaisys::utils::cast<float>(b[i])
            );
        } else {
            c[i] = a[i] + b[i];  // float/int 等原生类型直接加
        }
    }
}
```

**关键点**：x86 CPU 没有 BF16 的硬件加法指令，所以每次加法要做 3 次类型转换（2 次 BF16→float + 1 次 float→BF16），每步都有精度损失。

### 4.2 Tensor 调试打印（tensor.cpp）

在 [src/tensor/tensor.cpp](file:///c:/Code/LLAISYS/llaisys/src/tensor/tensor.cpp) 的 `debug_print` 中：

```cpp
case LLAISYS_DTYPE_BF16:
    return print_data(reinterpret_cast<const bf16_t *>(data), shape, strides, 0);
```

BF16 数据通过 `print_data` 打印时，会调用 `cast<float>()` 转为 float 再输出，因为 `std::cout` 没有 BF16 的原生输出支持。

### 4.3 Python 端（test_utils.py）

在 [test/test_utils.py](file:///c:/Code/LLAISYS/llaisys/test/test_utils.py) 中，BF16 在 Python 侧的映射：

```python
def torch_dtype(dtype_name: str):
    if dtype_name == "bf16":
        return torch.bfloat16    # PyTorch 原生支持 BF16

def llaisys_dtype(dtype_name: str):
    if dtype_name == "bf16":
        return llaisys.DataType.BF16  # 映射到枚举值 19
```

测试中通过 `torch.bfloat16` 与 LLAISYS 的 BF16 进行交叉验证。

---

## 五、BF16 vs FP16 对比

| 维度 | BF16 | FP16 |
|:--|:--|:--|
| **位布局** | 1-8-7 | 1-5-10 |
| **指数范围** | 与 FP32 相同（±3.4×10³⁸） | 仅 ±65,504 |
| **尾数精度** | 7 位（~2 位十进制） | 10 位（~3 位十进制） |
| **FP32 互转** | 截断/补零，O(1) | 需处理指数偏置（127→15） |
| **溢出风险** | 低（范围同 FP32） | 高（需 loss scaling） |
| **硬件支持** | Intel AMX、NVIDIA A100+ | 广泛（NVIDIA V100+） |
| **LLM 推理适用** | ✅ 权重压缩、KV Cache 量化 | ✅ 推理计算 |
| **LLM 训练适用** | ✅ 混合精度训练（前向 BF16，梯度 FP32） | ⚠️ 需要 loss scaling |

### 在 LLAISYS 中的选择建议

- **模型权重存储**：BF16 优于 FP16（范围安全，不会溢出）
- **KV Cache**：BF16 优于 FP16（范围安全）
- **中间激活计算**：FP16 优于 BF16（精度更高，且 NVIDIA GPU 有 FP16 Tensor Core）
- **CPU 推理**：FP16 和 BF16 都需要转 float 计算，成本相同

---

## 六、代码追踪路径

```
include/llaisys.h           → LLAISYS_DTYPE_BF16 = 19（枚举定义）
    ↓
src/utils/types.hpp          → CustomBFloat16 / bf16_t（内部结构体）
                            → dsize() 返回 2
                            → dtype_to_str() 返回 "bfloat16"
                            → cast<T>() 模板分派
    ↓
src/utils/types.cpp          → _bf16_to_f32()（左移 16 位补零）
                            → _f32_to_bf16()（round-to-nearest-even 截断）
    ↓
src/ops/add/cpu/add_cpu.cpp  → add_<bf16_t>()（float 中转加法）
src/tensor/tensor.cpp        → debug_print()（cast<float> 后打印）
    ↓
python/llaisys/libllaisys/llaisys_types.py → DataType.BF16 = 19
test/test_utils.py           → torch.bfloat16 ↔ llaisys.DataType.BF16 映射
```

---

## 七、一句话总结

**BF16 是 FP32 的高 16 位截断，牺牲精度换范围安全。** 在 LLAISYS 中，BF16 通过 `CustomBFloat16{uint16_t}` 存储，与 FP32 互转只需移位操作（无需指数偏置转换），在 CPU 上做算术运算时需通过 float 中转。