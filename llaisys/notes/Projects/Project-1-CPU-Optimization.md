# Project #1: CPU Optimization — 事件清单

## 主要修改文件

### SIMD 优化
| 文件 | 修改内容 |
|------|----------|
| `src/ops/linear/cpu/linear_cpu.cpp` | **优化** — 使用 SIMD 加速矩阵乘法 |
| `src/ops/self_attention/cpu/self_attention_cpu.cpp` | **优化** — SIMD 加速 Q·K^T 和 softmax(A)·V |
| `src/ops/rms_norm/cpu/rms_norm_cpu.cpp` | **优化** — SIMD 加速平方求和和缩放 |
| `src/ops/rope/cpu/rope_cpu.cpp` | **优化** — SIMD 加速旋转计算 |
| `src/ops/swiglu/cpu/swiglu_cpu.cpp` | **优化** — SIMD 加速逐元素操作 |

### OpenMP / 多线程优化
| 文件 | 修改内容 |
|------|----------|
| 同上所有 cpu/*.cpp 文件 | 添加 OpenMP `#pragma omp parallel for` 并行化外层循环 |

### 第三方库集成
| 文件 | 修改内容 |
|------|----------|
| [`xmake.lua`](file:///c:/Code/LLAISYS/llaisys/xmake.lua) | 添加 OpenBLAS / Eigen / MKL 依赖和编译选项 |
| `xmake/cpu.lua` | 添加第三方库的查找和链接逻辑 |
| `src/ops/linear/cpu/linear_cpu.cpp` | **重写** — 替换为 OpenBLAS `cblas_sgemm` 调用 |

## 需更改的配置
- **`xmake.lua`**：添加 OpenBLAS/Eigen/MKL 库的 find_package 和链接
- 可能需要设置环境变量：`OPENBLAS_ROOT`、`MKL_ROOT`
- 编译时添加宏：`-DUSE_SIMD`、`-DUSE_OPENMP`、`-DUSE_OPENBLAS` 等

## 建议阅读层次

> 参考 [遍历式阅读顺序](../thinking/遍历式阅读顺序.md)

| 层次 | 内容 | 必读文件 |
|:--:|------|------|
| **第 7 层** | Tensor（前置） | [src/tensor/tensor.cpp](file:///c:/Code/LLAISYS/llaisys/src/tensor/tensor.cpp) — `data()` 获取 raw pointer 用于 SIMD 循环 |
| **第 8 层** | **算子（本任务）** | `src/ops/linear/cpu/linear_cpu.cpp` — 主优化目标：矩阵乘法 SIMD 化 |
| | | `src/ops/self_attention/cpu/self_attention_cpu.cpp` — 注意力计算并行化 |
| | | `src/ops/rms_norm/cpu/rms_norm_cpu.cpp` — 平方求和 SIMD 加速 |
| | | `src/ops/rope/cpu/rope_cpu.cpp` — 旋转编码向量化 |
| | | `src/ops/swiglu/cpu/swiglu_cpu.cpp` — 逐元素门控 SIMD |
| **第 11 层** | 构建系统 | [xmake/cpu.lua](file:///c:/Code/LLAISYS/llaisys/xmake/cpu.lua) — 添加 `-march=native`/`-fopenmp` 编译选项 |
| | | [xmake.lua](file:///c:/Code/LLAISYS/llaisys/xmake.lua) — 添加 OpenBLAS/Eigen/MKL 依赖 |

> 本任务专注于 **第 8 层算子层的算法优化**，不涉及上层逻辑变更。需先确保 Assignment 2 的算子功能正确后，再进行性能优化。

---

## 背景知识

### SIMD（Single Instruction, Multiple Data）
- 一条指令同时处理多个数据元素
- x86: SSE (128bit, 4×f32), AVX2 (256bit, 8×f32), AVX-512 (512bit, 16×f32)
- ARM: NEON (128bit)

### OpenMP
- `#pragma omp parallel for` 自动并行化循环
- `#pragma omp simd` 提示编译器使用 SIMD
- 需在 xmake 中添加 `-fopenmp` 编译选项
- 运行时控制线程数：`OMP_NUM_THREADS=8`

### 第三方库对比
| 库 | 优势 | 劣势 |
|----|------|------|
| **OpenBLAS** | 开源免费、广泛使用 | 仅提供 BLAS 接口 |
| **Intel MKL** | 性能最强、功能全面 | 商业许可、仅 Intel CPU |
| **Eigen** | 纯头文件、模板化、易集成 | 性能略低于 MKL/OpenBLAS |

---

## 任务清单

### 阶段一：SIMD 向量化

#### 任务 1.1：理解现有实现的计算瓶颈

- [ ] **Profile 现有 CPU 算子**
  - 使用 Python 的 `time` 测量每个算子的耗时
  - 确定最耗时的算子（通常是 linear 和 self_attention）
  - 使用 `perf`（Linux）或 VTune 做热点分析

#### 任务 1.2：linear 的 SIMD 优化

- [ ] **替换三层循环 GEMM**
  - 使用 AVX2 `_mm256_fmadd_ps` 做 8 路 FMA（Fused Multiply-Add）
  - 内层循环每次处理 8 个 float
  - 处理不足 8 个的边界情况（tail loop）

- [ ] **添加 packed 内存布局（可选优化）**
  - 将 weight 矩阵转置重排以优化 cache 访问
  - 避免 gather/scatter 操作

#### 任务 1.3：self_attention 的 SIMD 优化

- [ ] **Q·K^T 矩阵乘法**
  - 对 batch 维度使用 SIMD
  - 与 linear 类似但矩阵较小

- [ ] **Softmax 的 SIMD 优化**
  - exp 运算使用 SIMD 近似（如 AVX2 `_mm256_exp_ps`）
  - reduce sum 使用 SIMD 累加

- [ ] **softmax(A)·V 矩阵乘法**
  - 同上使用 SIMD GEMM

#### 任务 1.4：逐元素算子的 SIMD 优化

- [ ] **rms_norm**
  - 平方求和用 SIMD reduce
  - rsqrt 用 `_mm256_rsqrt_ps`
  - 缩放用 SIMD multiply

- [ ] **rope**
  - sin/cos 查表批量计算
  - 逐对旋转

- [ ] **swiglu**
  - sigmoid + multiply 使用 SIMD

#### 任务 1.5：验证 SIMD 正确性与加速比

- [ ] **运行所有算子测试确认结果不变**
- [ ] **测量加速比**：记录优化前后每个算子的耗时

---

### 阶段二：OpenMP 多线程

#### 任务 1.6：为线性运算添加多线程

- [ ] **linear 外层循环并行化**
  - `#pragma omp parallel for` 并行化 output 行循环
  - 注意线程安全（每个线程操作不同的输出行，天然安全）

#### 任务 1.7：为 Attentions 添加多线程

- [ ] **self_attention 的外层循环并行化**
  - batch 和 head 维度可并行

#### 任务 1.8：编译配置和测试

- [ ] **在 xmake 中添加 `-fopenmp`**
- [ ] **测试不同线程数的加速比**（1, 2, 4, 8 线程）

---

### 阶段三：第三方库集成

#### 任务 1.9：OpenBLAS 集成

- [ ] **在 `xmake.lua` 中添加 OpenBLAS 查找和链接**
  - `add_packages("openblas")` 或手动 `add_linkdirs`
  - 添加 `USE_OPENBLAS` 宏

- [ ] **重写 `linear_cpu.cpp` 使用 `cblas_sgemm`**
  - F32: `cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)`

- [ ] **测量 OpenBLAS GEMM 的加速比**

---

## 性能评估
- [ ] 记录每个优化阶段的推理吞吐量（tokens/second）
- [ ] 生成完整的性能对比报告（基线 vs SIMD vs +OpenMP vs +OpenBLAS）