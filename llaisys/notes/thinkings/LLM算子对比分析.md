# LLM 核心算子对比分析

> 本文档对比 LLM 中关键算子的常见变体，分析它们在训练侧和推理侧的差异，为算子设计和优化决策提供参考。

---

## 目录

1. [RMS 归一化 vs Layer 归一化](#1-rms-归一化-vs-layer-归一化)
2. [SwiGLU 激活 vs ReLU / GELU 激活](#2-swiglu-激活-vs-relu--gelu-激活)
3. [RoPE 编码 vs 正余弦位置编码](#3-rope-编码-vs-正余弦位置编码)
4. [GQA / MQA vs MHA（分组/多查询注意力 vs 多头注意力）](#4-gqa--mqa-vs-mha分组多查询注意力-vs-多头注意力)
5. [Pre-Norm vs Post-Norm](#5-pre-norm-vs-post-norm)
6. [SiLU / Swish vs GELU](#6-silu--swish-vs-gelu)
7. [FP16 vs BF16 vs FP32（精度对比）](#7-fp16-vs-bf16-vs-fp32精度对比)
8. [KV Cache vs 无缓存](#8-kv-cache-vs-无缓存)
9. [总结表](#9-总结表)

---

## 1. RMS 归一化 vs Layer 归一化

### 数学定义

**Layer Normalization（层归一化）：**


$$
给定输入向量 x \in \mathbb{R}^d规定：
$$

$$
\begin{aligned}
\mu &= \frac{1}{d}\sum_{i=1}^{d} x_i \\
\sigma^2 &= \frac{1}{d}\sum_{i=1}^{d} (x_i - \mu)^2 \\
\hat{x}_i &= \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} \\
y_i &= \gamma \cdot \hat{x}_i + \beta
\end{aligned}
$$

**RMS Normalization（均方根归一化）：**

$$
\begin{aligned}
\text{RMS}(x) &= \sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2} \\
\hat{x}_i &= \frac{x_i}{\text{RMS}(x) + \epsilon} \\
y_i &= \gamma \cdot \hat{x}_i
\end{aligned}
$$

### 核心差异

| 维度 | Layer Normalization | RMS Normalization |
|------|:--:|:--:|
| **均值中心化** | 减去均值\mu（re-centering） | 不做均值中心化 |
| **缩放** | 除以标准差 $\sigma$ | 除以 RMS |
| **可学习参数** | $\gamma$ (scale) + $\beta$ (shift) | 仅 $\gamma$ (scale) |
| **参数量** | $2d$ | $d$ |
| **计算量** | 计算均值 + 方差 + 归一化 | 仅计算平方和 + 归一化 |

### 训练侧差异

| | Layer Normalization | RMS Normalization |
|------|------|------|
| 反向传播 | 需要计算 $\partial\mu$ 和 $\partial\sigma^2$ 的梯度 | 不需要 $\partial\mu$，梯度路径更短 |
| 梯度稳定性 | 均值中心化有助于稳定训练初期 | 论文证明移除 re-centering 不影响收敛 |
| 内存占用 | 保存 $\mu, \sigma^2$ 用于反向传播 | 仅保存 $\sum x_i^2$ |
| 计算开销 | 约 $4d$ 次运算（均值、方差、归一化、仿射） | 约 $3d$ 次运算（平方和、归一化、缩放） |

### 推理侧差异

| | Layer Normalization | RMS Normalization |
|------|------|------|
| 延迟 | 略高（多一次减法遍历） | 更低（约减少 25% 计算） |
| 融合优化 | 可与前一层融合（如 linear+LN） | 同样可与前一层融合 |
| 量化友好度 | 均值中心化可能导致量化偏移 | 无中心化，更利于 INT8 量化 |

**推理侧量化**，是指在大模型已经训练完成（通常使用高精度如 FP32 或 FP16）后，在将其部署到实际应用中进行**推理（即用模型回答问题）** 之前，把模型的**权重**（Weights）和/或**激活值**（Activations）从高精度数值（如 32 位浮点数）转换成低精度整数（如 8 位整数 INT8、4 位整数 INT4）的过程。

### 为什么 LLaMA 系列选择 RMS Norm

1. **计算量减少 25%**：省略均值计算，prefill 阶段对长序列有显著收益
2. **参数减半**：去掉 $\beta$，节省 $d$ 个参数（对大模型节省可观）
3. **实验验证**：论文 *Root Mean Square Layer Normalization* (Zhang & Sennrich, 2019) 证明 re-centering 不必要
4. **推理加速**：在 decode 阶段，每次只处理 1 个 token，RMS Norm 更轻量

---

## 2. SwiGLU 激活 vs ReLU / GELU 激活

### 数学定义

**ReLU (Rectified Linear Unit)：**

$$
\text{ReLU}(x) = \max(0, x)
$$

**GELU (Gaussian Error Linear Unit)：**

$$
\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}\left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]
$$

近似形式（tanh）：
$$
\text{GELU}(x) \approx 0.5x\left[1 + \tanh\left(\sqrt{\frac{2}{\pi}}(x + 0.044715x^3)\right)\right]
$$

**SwiGLU (Swish-Gated Linear Unit)：**

$$
\begin{aligned}
\text{SwiGLU}(x) &= \text{SiLU}(xW_1) \odot (xW_2) \\
\text{SiLU}(x) &= x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}
\end{aligned}
$$

$$
其中 W_1, W_2 \in \mathbb{R}^{d \times d_{ff}}，\odot为逐元素乘法。
$$



### 核心差异

| 维度 | ReLU | GELU | SwiGLU |
|------|:--:|:--:|:--:|
| **非线性来源** | 截断（0 以下置零） | 高斯分布的平滑门控 | Sigmoid 门控 + 逐元素乘 |
| **可微性** | 不可微（x=0 处） | 处处可微 | 处处可微 |
| **负值处理** | 完全置零 | 接近零但非零的平滑过渡 | 趋近于零但非零 |
| **计算复杂度** | 极低（一次比较） | 中等（erf/tanh） | 较高（sigmoid + 乘） |
| **参数量** | 0 | 0 | 需额外 $3d \cdot d_{ff}$ 权重 |

### 训练侧差异

| | ReLU | GELU | SwiGLU |
|------|------|------|------|
| 梯度消失 | 严重（"Dead ReLU"，负半区梯度为 0） | 轻微（负半区有小梯度） | 无（SiLU 处处有梯度） |
| 收敛速度 | 较快但可能不稳定 | 稳定 | 更快且稳定 |
| 激活稀疏性 | 约 50% 神经元为零 | 约 10-20% 接近零 | 约 10-20% 接近零 |
| 显存占用 | 低（仅保存 mask） | 中（保存中间值） | 高（门控值 + up 投影） |
| 训练吞吐 | 高 | 中 | 较低（额外矩阵乘） |

### 推理侧差异

| | ReLU | GELU | SwiGLU |
|------|------|------|------|
| 延迟 | 极低（~1 cycle） | 中等（erf/tanh 查表） | 较高（sigmoid + elemwise） |
| 权重加载 | 标准 FFN 权重 | 标准 FFN 权重 | 3 个权重矩阵（gate/up/down） |
| 内存带宽 | 低 | 低 | 高（额外权重 I/O） |
| 量化 | 极易（单阈值） | 较难（需查表） | 较难（sigmoid 量化敏感） |

### 为什么 LLaMA 选择 SwiGLU

1. **更好的收敛性**：*GLU Variants Improve Transformer* (Shazeer, 2020) 表明 SwiGLU 在多项任务上优于 GELU
2. **门控机制**：动态控制信息流，类似 LSTM 的 gate 思想
3. **实践验证**：PaLM、LLaMA、Qwen 等主流模型均采用 SwiGLU

---

## 3. RoPE 编码 vs 正余弦位置编码

### 数学定义

**Sinusoidal Position Encoding（正弦余弦位置编码，原始 Transformer）：**

$$
\begin{aligned}
PE_{(pos, 2i)} &= \sin\left(\frac{pos}{10000^{2i/d}}\right) \\
PE_{(pos, 2i+1)} &= \cos\left(\frac{pos}{10000^{2i/d}}\right)
\end{aligned}
$$

$$
其中 pos 为位置，i 为维度索引，d 为 embedding 维度。
$$

**RoPE (Rotary Position Embedding)：**

$$
对查询向量q和键向量k的第i对维度施加旋转：
$$


$$
\begin{aligned}
f(q, m) &= \begin{pmatrix} \cos m\theta_1 & -\sin m\theta_1 \\ \sin m\theta_1 & \cos m\theta_1 \end{pmatrix} \begin{pmatrix} q_0 \\ q_1 \end{pmatrix} \\
\theta_i &= 10000^{-2i/d}
\end{aligned}
$$

$$
其中 m为位置索引，d 为 head_{dim}。
$$



### 核心差异

| 维度 | Sinusoidal PE | RoPE |
|------|:--:|:--:|
| **注入方式** | 直接加到 embedding 上（加法） | 旋转 q/k 向量（乘法旋转） |
| **位置感知** | 绝对位置 | 相对位置（通过旋转矩阵的性质） |
| **外推能力** | 弱（超出训练长度效果差） | 较强（可通过调整 $\theta$ 扩展） |
| **计算位置** | 在 embedding 层 | 在每层 attention 的 q/k 上 |
| **参数量** | 0（固定公式） | 0（固定公式） |
| **相对位置** | 不直接编码 | 内积天然包含相对位置信息 |

### 训练侧差异

| | Sinusoidal PE | RoPE |
|------|------|------|
| 训练稳定性 | 好 | 好 |
| 长序列泛化 | 差（超出 max_len 效果显著下降） | 较好（NTK-aware 缩放可扩展） |
| 显存 | 需存储 PE 矩阵 | 不需额外存储 |
| 计算开销 | 几乎为零（预计算） | 每层每头需旋转操作 |
| 实现复杂度 | 极简单 | 中等（需实现旋转 kernel） |

### 推理侧差异

| | Sinusoidal PE | RoPE |
|------|------|------|
| Prefill 阶段 | 零开销（预计算查表） | 需对 q/k 做旋转（有开销） |
| Decode 阶段 | 零开销（查表） | 仅需对当前 token 的 q 旋转（开销小） |
| 与 KV Cache 兼容 | 简单（位置信息独立） | 需注意旋转顺序（先旋转再 cache） |
| 序列长度扩展 | 困难（需重新训练） | 可用 NTK-aware / YaRN 等方法 |
| 融合优化 | 天然可融合 | 旋转操作可与 attention 融合 |

### 为什么 LLaMA 选择 RoPE

1. **相对位置编码**：天然捕捉 token 间相对距离，比绝对位置更符合语言规律
2. **外推能力**：通过 NTK-aware scaling 等方法，可将 4K 上下文扩展到 32K+ 而不重训
3. **理论优雅**：内积 q_m^T k_n仅依赖于相对位置 (m-n)
4. **业界标准**：LLaMA、Qwen、Mistral、DeepSeek 等均采用

---

## 4. GQA / MQA vs MHA（分组/多查询注意力 vs 多头注意力）

### 数学定义

**MHA (Multi-Head Attention)：**

每个注意力头有独立的 Q、K、V 投影：
$$
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$
$$
其中 W_i^Q, W_i^K, W_i^V \in \mathbb{R}^{d \times d_h}，h 个头各有独立参数。
$$

**MQA (Multi-Query Attention)：**

所有头共享同一个 K、V 投影，仅 Q 独立：
$$
\text{head}_i = \text{Attention}(QW_i^Q, KW^K, VW^V)
$$
$$
其中 W^K, W^V \in \mathbb{R}^{d \times d_h}，所有头共享。
$$

**GQA (Grouped-Query Attention)：**

$$
将 h 个头分为 g组，每组共享 K、V：
$$

$$
\text{head}_i = \text{Attention}(QW_i^Q, KW_{\lfloor i/g \rfloor}^K, VW_{\lfloor i/g \rfloor}^V)
$$

### 核心差异

| 维度 | MHA | MQA | GQA |
|------|:--:|:--:|:--:|
| KV 头数 | $h$ | 1 | $g$（$1 < g < h$） |
| KV 参数量 | $2hd \cdot d_h$ | $2d \cdot d_h$ | $2g d \cdot d_h$ |
| KV Cache 大小 | $2h \cdot d_h \cdot L$ | $2 \cdot d_h \cdot L$ | $2g \cdot d_h \cdot L$ |
| 模型质量 | 最高 | 略降 | 接近 MHA |
| 推理吞吐 | 最低 | 最高 | 中等 |

### 训练侧差异

| | MHA | MQA | GQA |
|------|------|------|------|
| 参数量 | 最多 | 最少 | 中等 |
| 训练吞吐 | 最低 | 最高 | 中等 |
| 收敛质量 | 最好 | 略差 | 接近 MHA |
| 梯度计算 | 各头独立 | KV 梯度需聚合 | 组内 KV 梯度聚合 |

### 推理侧差异

| | MHA | MQA | GQA |
|------|------|------|------|
| KV Cache 大小 | $2h d_h L$ | $2d_h L$ | $2g d_h L$ |
| 内存带宽 | 瓶颈（decode 受限于 KV 读取） | 宽松 | 适中 |
| Decode 延迟 | 最高 | 最低 | 中等 |
| 长序列支持 | 差（KV Cache 线性增长） | 好 | 好 |
| batch 推理 | 受限于 KV Cache 容量 | 支持更大 batch | 支持更大 batch |

> $$
> 实例：Qwen2-1.5B 使用 GQA（h=12, g=2），KV Cache 为 MHA 的 1/6。
> $$
>
> 

---

## 5. Pre-Norm vs Post-Norm

### 架构差异

**Post-Norm（原始 Transformer）：**
$$
\text{output} = \text{LayerNorm}(x + \text{Sublayer}(x))
$$

**Pre-Norm（现代 LLM 主流）：**
$$
\text{output} = x + \text{Sublayer}(\text{LayerNorm}(x))
$$

### 核心差异

| 维度 | Post-Norm | Pre-Norm |
|------|:--:|:--:|
| 归一化位置 | 残差连接之后 | 残差连接之前 |
| 梯度流动 | 可能梯度消失（深层） | 梯度更稳定 |
| 训练稳定性 | 需 warmup + 小心调参 | 天然稳定，无需 warmup |
| 最终性能 | 理论上限更高 | 收敛更快 |
| 主流采用 | 原始 Transformer、ViT | LLaMA、GPT-3、Qwen |

### 训练侧差异

| | Post-Norm | Pre-Norm |
|------|------|------|
| 学习率敏感度 | 高（需 warmup） | 低 |
| 深层训练 | 困难（梯度消失） | 容易 |
| 收敛速度 | 慢 | 快 |
| 最终 loss | 可能更低 | 略高但差异小 |

### 推理侧差异

| | Post-Norm | Pre-Norm |
|------|------|------|
| 计算量 | 相同 | 相同 |
| 融合机会 | 可与 sublayer 融合 | 先 Norm 再 sublayer，融合稍复杂 |
| 影响 | 无显著差异 | 无显著差异 |

---

## 6. SiLU / Swish vs GELU

### 数学定义

**SiLU / Swish（Sigmoid Linear Unit）：**

$$
\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}
$$

**GELU：**

$$
\text{GELU}(x) = x \cdot \Phi(x) \approx 0.5x\left[1 + \tanh\left(\sqrt{\frac{2}{\pi}}(x + 0.044715x^3)\right)\right]
$$

### 差异

| 维度 | SiLU / Swish | GELU |
|------|:--:|:--:|
| 门控函数 | $\sigma(x)$（sigmoid） | $\Phi(x)$（高斯 CDF） |
| 负值行为 | $x \to -\infty$ 时趋于 0 | $x \to -\infty$ 时趋于 0 |
| 正值行为 | 接近 $x$（线性） | 接近 $x$（线性） |
| 最小值 | 约 $-0.278$（在 $x \approx -1.278$） | 约 $-0.169$（在 $x \approx -0.8$） |
| 计算速度 | 有 SIMD 多项式近似 | 有 erf/tanh 近似 |
| 负值激活 | 略强（min 更小） | 弱 |

---

## 7. FP16 vs BF16 vs FP32（精度对比）

### 数值格式

| 格式 | 总位数 | 指数位 | 尾数位 | 动态范围 | 精度 |
|------|:--:|:--:|:--:|:--:|:--:|
| FP32 | 32 | 8 | 23 | $\pm 3.4 \times 10^{38}$ | $\sim 7$ 位十进制 |
| FP16 | 16 | 5 | 10 | $\pm 65504$ | $\sim 3.3$ 位十进制 |
| BF16 | 16 | 8 | 7 | $\pm 3.4 \times 10^{38}$ | $\sim 2$ 位十进制 |

### 训练侧差异

| | FP32 | FP16 | BF16 |
|------|------|------|------|
| 显存占用 | 4 bytes/元素 | 2 bytes/元素 | 2 bytes/元素 |
| 溢出风险 | 无 | 高（需 loss scaling） | 低（范围同 FP32） |
| 训练吞吐 | 基准 | 2x（理论） | 2x（理论） |
| 混合精度 | 不需要 | 需 FP32 master weights | 需 FP32 master weights |
| 实现复杂度 | 简单 | 复杂（loss scaling） | 简单（截断即可） |

### 推理侧差异

| | FP32 | FP16 | BF16 |
|------|------|------|------|
| 权重存储 | 4 bytes | 2 bytes | 2 bytes |
| 延迟 | 基准 | 更快（Tensor Core） | 更快（Tensor Core） |
| 量化到 INT8/INT4 | 需量化 | 需量化 | 需量化 |
| 精度损失 | 无 | 可能（需验证） | 略大但可接受 |

---

## 8. KV Cache vs 无缓存

### 核心差异

| 维度 | 无 KV Cache | 有 KV Cache |
|------|:--:|:--:|
| Decode 阶段 | 重新计算所有历史 K/V | 直接读取缓存的 K/V |
| 计算复杂度 | $O(L^2)$ | $O(L)$（仅计算新 token） |
| 显存占用 | 低 | 随序列长度线性增长 |
| Prefill 阶段 | 相同 | 相同（写入缓存） |

> 现代 LLM 推理几乎必须使用 KV Cache，否则 decode 延迟不可接受。

---

## 9. 总结表

### 训练侧优先级

| 优化目标 | 推荐方案 |
|------|------|
| 降低显存 | GQA + BF16 + RMS Norm |
| 加速收敛 | Pre-Norm + SwiGLU + RoPE |
| 稳定训练 | Pre-Norm + BF16 + RMS Norm |
| 最高精度 | Post-Norm + MHA + GELU（但训练成本高） |

### 推理侧优先级

| 优化目标 | 推荐方案 |
|------|------|
| 降低延迟 | MQA/GQA + RMS Norm + KV Cache |
| 减少显存 | GQA + INT4 量化 + RMS Norm |
| 长序列支持 | RoPE (NTK-aware) + GQA + KV Cache |
| 最高吞吐 | MQA + FP16 + KV Cache |

### 各主流模型算子选择

| 模型 | 归一化 | 激活 | 位置编码 | 注意力 | Norm 位置 |
|------|:--:|:--:|:--:|:--:|:--:|
| Transformer (2017) | LayerNorm | ReLU | Sinusoidal | MHA | Post-Norm |
| GPT-2/3 | LayerNorm | GELU | Learned | MHA | Pre-Norm |
| LLaMA / LLaMA 2 | RMSNorm | SwiGLU | RoPE | MHA | Pre-Norm |
| LLaMA 3 | RMSNorm | SwiGLU | RoPE | GQA | Pre-Norm |
| Qwen / Qwen2 | RMSNorm | SwiGLU | RoPE | GQA | Pre-Norm |
| Mistral | RMSNorm | SwiGLU | RoPE | GQA (sliding window) | Pre-Norm |
| DeepSeek-V2 | RMSNorm | SwiGLU | RoPE | MLA (Multi-head Latent Attention) | Pre-Norm |

---

## 参考

- *Root Mean Square Layer Normalization* (Zhang & Sennrich, 2019)
- *GLU Variants Improve Transformer* (Shazeer, 2020)
- *RoFormer: Enhanced Transformer with Rotary Position Embedding* (Su et al., 2021)
- *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints* (Ainslie et al., 2023)
- *On Layer Normalization in the Transformer Architecture* (Xiong et al., 2020)
- *LLaMA: Open and Efficient Foundation Language Models* (Touvron et al., 2023)