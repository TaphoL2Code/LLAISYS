# 训练侧 vs 推理侧：相同点与不同点

## 一、总体对比

| 维度 | 训练 (Training) | 推理 (Inference) |
|------|:--:|:--:|
| **目标** | 调整模型参数，最小化 loss | 用固定参数，生成输出 |
| **数据流** | forward → backward → update | forward only |
| **输入** | 大规模数据集（TB 级） | 单条或多条用户输入 |
| **输出** | 梯度 + loss 值 | 预测结果 / 生成文本 |
| **权重状态** | 不断更新 | 冻结不变 |
| **运行次数** | 一次（或几次） | 成千上万次 |
| **硬件关注点** | 算力（TFLOPS） | 延迟（ms）+ 吞吐（QPS） |
| **精度** | FP16/BF16 混合精度训练 | FP16/INT8/INT4 量化推理 |

---

## 二、核心差异详解

### 2.1 计算图

```
训练：
  input → [linear] → [activation] → [linear] → output
                     ↓                          ↓
              保存中间激活              loss = CrossEntropy(output, label)
                     ↓                          ↓
              backward 时复用            loss.backward()
                     ↓
              grad 沿反向传播更新权重

推理：
  input → [linear] → [activation] → [linear] → output → (softmax) → token
                                                     ↑
                                           不需保存中间值，不需反向
```

#### 相同点
- Forward pass 的计算逻辑完全一致：矩阵乘法、激活函数、归一化等
- 模型结构相同：层数、隐藏维度、注意力头数等超参数一致
- 权重矩阵形状和排列顺序相同

#### 不同点
| | 训练 | 推理 |
|------|------|------|
| 计算图 | 动态构建并保存，用于反向传播 | 不需要计算图，直接执行 |
| 中间激活 | **必须保存**所有层输出，供 backward 计算梯度 | **不需要保存**，处理完即丢弃（或仅保留 KV Cache） |
| 显存峰值 | 权重 + 优化器状态 + 中间激活 ≈ **模型大小的 4-6 倍** | 权重 + KV Cache ≈ **模型大小的 1-2 倍** |
| backward 计算 | 需要实现每个算子的反向传播（grad） | 不需要 |
| 自动微分 | 需要 autograd 引擎 | 不需要 |

---

### 2.2 算子实现

以 `Linear`（全连接层）为例：

```
训练：
  Forward:  Y = XW + b
  Backward: dW = Xᵀ · dY
            dX = dY · Wᵀ
            db = sum(dY, dim=0)

  需要实现：3 个 kernel (forward + 2 个 backward)
  需要保存：X（供 dW 计算用）、W（供 dX 计算用）

推理：
  Forward:  Y = XW + b
  需要实现：1 个 kernel
  不需要保存：任何中间结果
```

#### 各算子的训练/推理差异

| 算子 | 训练需要额外实现 | 训练需要保存的状态 | 推理需要保存的状态 |
|------|:--:|------|------|
| **Linear** | backward: dW, dX, db | 输入 X | 无 |
| **Embedding** | backward: 查表梯度累加 | 输入索引 | 无 |
| **LayerNorm / RMSNorm** | backward: 对均值/方差的梯度 | 均值、方差、输入 | 无 |
| **Softmax** | backward: 对概率分布的梯度 | softmax 输出 | 无 |
| **Attention** | backward: Q/K/V 各自梯度 | Q, K, V, attention weights | **KV Cache**（推理独有） |
| **Dropout** | backward: 按 mask 回传梯度 | dropout mask | **推理时关掉**，恒等映射 |
| **BatchNorm** | backward: 对 γ, β, μ, σ² 的梯度 | 输入、均值、方差 | 用训练时累积的 running_mean/var |

---

### 2.3 KV Cache（推理独有）

这是推理侧最重要的优化，训练侧没有对应物。

```
训练（Teacher Forcing）：
  输入 tokens:  [A, B, C, D]           ← 一次性输入完整序列
  Attention:    每个 token 同时计算，互相能看到
  计算量:       O(n²) 但高度并行

推理（自回归生成）：
  Step 1:  输入 [A]         → Attention(A, A)  → 生成 B
  Step 2:  输入 [A, B]      → Attention(B, AB) → 生成 C
  Step 3:  输入 [A, B, C]   → Attention(C, ABC)→ 生成 D

  不用 KV Cache：每个 step 重新计算所有 K/V，O(n²) × n = O(n³)
  使用 KV Cache：只计算新 token 的 Q，复用历史的 K/V，O(n²)
```

| | 训练 | 推理（无 KV Cache） | 推理（有 KV Cache） |
|------|:--:|:--:|:--:|
| 输入方式 | 完整序列一次输入 | 逐 token 输入 | 逐 token 输入 |
| K/V 计算 | 每层一次 | 每层每 step 重复计算 | 每层每 step 只算新 token |
| 计算复杂度 | O(n²) | O(n³) | O(n²) |
| 额外内存 | 无 | 无 | 每层存 K/V（2×layers×n×d×bytes） |

---

### 2.4 归一化层

| | BatchNorm | LayerNorm / RMSNorm |
|------|:--:|:--:|
| 归一化维度 | 沿 batch 维度 | 沿 feature 维度（每个样本独立） |
| 训练时 | 计算当前 batch 的 μ, σ² | 计算当前样本的 μ, σ² |
| 推理时 | 用训练期间累积的 **running_mean / running_var** | **完全一致**，直接计算 |
| 对 batch size 的依赖 | 大（小 batch 时不稳定） | **无依赖** |

为什么 LLM 全用 LayerNorm/RMSNorm 而不是 BatchNorm：
- 推理时 batch size 经常是 1（单条请求），BatchNorm 的 running_stats 在 batch=1 时无意义
- LayerNorm 训练和推理的计算逻辑完全相同，不需要区分模式

---

### 2.5 Dropout / 正则化

| | 训练 | 推理 |
|------|:--:|:--:|
| Dropout | 以概率 p 随机置零神经元，缩放剩余值 | **关闭**，所有权重直接通过（恒等映射） |
| 等价操作 | `mask = Bernoulli(p); output = input * mask / (1-p)` | `output = input` |
| 目的 | 防止过拟合，强制冗余表示 | 不需要 |

实现时需要处理：
```python
# 训练
if self.training:
    mask = torch.bernoulli(torch.full_like(x, 1 - p))
    return x * mask / (1 - p)
# 推理
else:
    return x  # 恒等映射
```

在 LLAISYS 项目中，因为只做推理，Dropout 层可以**完全省略**。

---

### 2.6 精度与量化

| | 训练 | 推理 |
|------|:--:|:--:|
| 典型精度 | FP16/BF16 混合精度 | FP16 / INT8 / INT4 |
| 权重存储 | FP32（master copy）+ FP16（forward copy） | 仅量化权重 |
| 梯度计算 | FP16 forward, FP32 backward | 不适用 |
| 量化感知训练 (QAT) | 模拟量化误差进行训练 | - |
| 训练后量化 (PTQ) | - | 直接量化已训练的权重 |
| 动态量化 | - | 推理时动态量化中间激活 |

LLM 推理常用 INT4 量化（如 GPTQ、AWQ）将 7B 模型从 14GB 压到 4GB，在消费级 GPU 上运行。

---

### 2.7 批处理策略

| | 训练 | 推理 |
|------|:--:|:--:|
| 批量处理 | 固定 batch size，所有样本等长（padding 对齐） | 动态 batching / 连续批处理 |
| 样本长度 | 通过 padding + attention mask 对齐 | 不等长，各自独立生成 |
| 完成时机 | 同时完成（等长） | 不同时完成（连续批处理需动态管理） |
| 显存调度 | 预分配，一次分配 | 动态分配/释放 KV Cache |

**连续批处理（Continuous Batching）**（Project 4 涉及）：
```
传统 static batching：
  Req1: [==========]   ← 长短不一，短的要等长的
  Req2: [====]
  Req3: [==========]
  → 空闲时间浪费算力

连续批处理：
  Req1: [====]已结束 → 从 batch 移除，释放 KV Cache
  Req2: [=========]
  Req3: [=======]
  → 新请求立即加入，持续调度
```

---

### 2.8 优化器与学习率

| | 训练 | 推理 |
|------|:--:|:--:|
| 优化器 | Adam/AdamW/SGD，需要维护动量、二阶矩等状态 | **不需要** |
| 优化器状态内存 | 参数的 2-3 倍（Adam 需要 m + v） | 0 |
| 学习率调度 | cosine decay / warmup / step decay | 不适用 |
| 梯度裁剪 | 防止梯度爆炸 | 不适用 |

---

### 2.9 分布式策略对比

| | 训练 | 推理 |
|------|:--:|:--:|
| **数据并行 (DP)** | ✅ 每卡完整模型副本，不同数据 | ❌ 推理不需要（单条数据不需要复制到多卡） |
| **张量并行 (TP)** | ✅ 将权重矩阵切分到多卡，forward 中通信 | ✅ Project 5 涉及：大模型单卡放不下时使用 |
| **流水线并行 (PP)** | ✅ 按层切分到不同卡，微批次流水线 | ⚠️ 较少使用（延迟高） |
| **ZeRO 优化** | ✅ ZeRO-1/2/3 分片优化器状态和梯度 | ❌ 推理无优化器 |
| **通信模式** | AllReduce（梯度同步, DP） + AllGather/ReduceScatter（TP） | 仅 AllGather/ReduceScatter（TP） |

推理的**张量并行**（TP）比训练简单：不需要跨卡同步梯度和优化器状态，只需要 forward 过程中的激活通信。以 Linear 层为例：

```
训练（列并行 Linear + 反向）：
  Forward:  f = XWᵢ; Y = AllGather([f₀, f₁])
  Backward: dY → dYᵢ(scatter); dXᵢ = dYᵢWᵢᵀ; dX = AllReduce(dXᵢ)
            dWᵢ = XᵀdYᵢ

推理（列并行 Linear，仅 forward）：
  Forward:  f = XWᵢ; Y = AllGather([f₀, f₁])
  → 少了 AllReduce 和 scatter，通信量减半
```

---

## 三、LLM 特有的训练/推理差异

### 3.1 损失函数

| | 训练 | 推理 |
|------|:--:|:--:|
| 语言建模 | CrossEntropy（预测每个 token 的概率 vs 真实标签） | 不需要计算 loss |
| 生成方式 | Teacher Forcing（用真实标签作为下一步输入） | 自回归（用自己生成的 token 作为下一步输入） |

### 3.2 采样策略（推理独有）

训练时用 CrossEntropy 直接算概率分布，推理时需要从分布中采样生成下一个 token：

| 方法 | 说明 | 训练中有对应吗？ |
|------|------|:--:|
| **Greedy** | 选概率最高的 token | ❌ 训练不需要 |
| **Temperature** | softmax(x/T)，T>1 更随机，T<1 更确定 | ❌ |
| **Top-K** | 只从概率最高的 K 个中采样 | ❌ |
| **Top-P (Nucleus)** | 从累积概率达 P 的最小集合中采样 | ❌ |
| **Beam Search** | 维护多条候选序列，选总分最高 | ❌（但 NLP 的传统 Seq2Seq 训练中常用） |
| **Repetition Penalty** | 对已生成的 token 降权 | ❌ |

### 3.3 位置编码（RoPE）

| | 训练 | 推理 |
|------|:--:|:--:|
| RoPE 计算 | 一次性计算整个序列的旋转位置编码 | 每步只算新 token 的，历史 token 的位置编码不变 |
| 与 KV Cache 的交互 | 无交互 | KV Cache 存储的是 **施加 RoPE 之后** 的 K/V，后续 token 无需重新编码 |

---

## 四、代码实现层面的差异

### 4.1 Module 定义

```python
# 训练框架（PyTorch）
class Linear(nn.Module):
    def forward(self, x):
        return F.linear(x, self.weight, self.bias)
    # 训练时 PyTorch autograd 自动生成 backward

# 推理框架（LLM 推理引擎）
class Linear:
    def forward(self, x):
        return matmul(x, self.weight) + self.bias
    # 没有 backward，没有 autograd
```

### 4.2 推理特有的组件

| 组件 | 作用 | 训练中对应？ |
|------|------|:--:|
| `KVCache` | 存储每层每个 token 的 Key/Value | ❌ |
| `Sampler` | Temperature/Top-K/Top-P 采样 | ❌ |
| `RequestQueue` | 排队管理多个推理请求 | ❌（DataLoader 类似但逻辑不同） |
| `Scheduler` | 连续批处理调度 | ❌ |
| `Tokenizer` | 文本↔token 转换 | ✅ 完全相同 |
| `ModelLoader` | 加载 safetensors/pth 权重 | ✅ 完全相同 |

---

## 五、总结：什么时候可以复用训练代码？

| 组件 | 可直接复用 | 需要修改 | 需要重写 |
|------|:--:|:--:|:--:|
| 模型权重加载 | ✅ | - | - |
| Tokenizer | ✅ | - | - |
| LayerNorm / RMSNorm | ✅ | - | - |
| Linear（forward） | ✅ | - | - |
| Embedding（forward） | ✅ | - | - |
| Softmax（forward） | ✅ | - | - |
| Activation（GELU/SiLU） | ✅ | - | - |
| RoPE（forward） | ✅ | - | - |
| Self-Attention | - | ✅ 去掉 dropout mask，增加 KV Cache | - |
| Dropout | - | ✅ 恒等映射 | - |
| Sampler | - | - | ✅ |
| KV Cache | - | - | ✅ |
| Scheduler | - | - | ✅ |
| 所有 backward | - | - | ❌ 完全不需要 |

**一句话总结**：推理是训练的**子集**——所有 forward 计算逻辑相同，但删掉了 backward、优化器、Dropout，增加了 KV Cache 和采样器。