# Layer12_测试

问：*测试怎么验证 LLAISYS 的正确性？*

答：***PyTorch 作为参考实现**——每个 LLAISYS 操作都与 PyTorch 的等价操作对比，用 `check_equal()` 逐元素验证。CPU 上用 `memcpy_sync` 直拷数据，GPU 上通过 `torch.as_strided` 共享内存。*

- [x] ## 第 12 层：测试 — 12 个文件

**理解"怎么用 PyTorch 验证 LLAISYS"。**

| 序号 | 文件 | 行数 | 核心看点 |
|:--:|------|:--:|------|
| 12.1 | [test/test_utils.py](file:///c:/Code/LLAISYS/llaisys/test/test_utils.py) | 250 | **核心工具**：`random_tensor()`（同随机种子生成 PyTorch/LLAISYS Tensor）、`check_equal()`（对比结果）、`benchmark()`（性能对比） |
| 12.2 | [test/test_runtime.py](file:///c:/Code/LLAISYS/llaisys/test/test_runtime.py) | 62 | Runtime 测试：`get_device_count`、`set_device`、`malloc_device`、`memcpy_sync`（H2D→D2D→D2H 往返） |
| 12.3 | [test/test_tensor.py](file:///c:/Code/LLAISYS/llaisys/test/test_tensor.py) | 55 | Tensor 测试：`load`/`view`/`permute`/`slice`/`isContiguous`，与 PyTorch 对比 |
| 12.4 | [test/ops/add.py](file:///c:/Code/LLAISYS/llaisys/test/ops/add.py) | — | add 算子测试（待实现） |
| 12.5 | [test/ops/argmax.py](file:///c:/Code/LLAISYS/llaisys/test/ops/argmax.py) | — | argmax 算子测试（待实现） |
| 12.6 | [test/ops/embedding.py](file:///c:/Code/LLAISYS/llaisys/test/ops/embedding.py) | — | embedding 算子测试（待实现） |
| 12.7 | [test/ops/linear.py](file:///c:/Code/LLAISYS/llaisys/test/ops/linear.py) | — | linear 算子测试（待实现） |
| 12.8 | [test/ops/rms_norm.py](file:///c:/Code/LLAISYS/llaisys/test/ops/rms_norm.py) | — | rms_norm 算子测试（待实现） |
| 12.9 | [test/ops/rope.py](file:///c:/Code/LLAISYS/llaisys/test/ops/rope.py) | — | rope 算子测试（待实现） |
| 12.10 | [test/ops/self_attention.py](file:///c:/Code/LLAISYS/llaisys/test/ops/self_attention.py) | — | self_attention 算子测试（待实现） |
| 12.11 | [test/ops/swiglu.py](file:///c:/Code/LLAISYS/llaisys/test/ops/swiglu.py) | — | swiglu 算子测试（待实现） |
| 12.12 | [test/test_infer.py](file:///c:/Code/LLAISYS/llaisys/test/test_infer.py) | 149 | **端到端推理测试**：加载 DeepSeek-R1-Distill-Qwen-1.5B，HuggingFace vs LLAISYS 对比输出 tokens |

---

- [x] ### 12.1 test/test_utils.py — 核心测试工具

```
def random_tensor(shape, dtype_name, device_name, device_id=0, scale=None, bias=None):
    torch_tensor = torch.rand(shape, dtype=torch_dtype(dtype_name), device=torch_device(device_name, device_id))
    ...
    llaisys_tensor = llaisys.Tensor(shape, dtype=llaisys_dtype(dtype_name), device=llaisys_device(device_name), device_id=device_id)
    api = llaisys.RuntimeAPI(llaisys_device(device_name))
    api.memcpy_sync(llaisys_tensor.data_ptr(), torch_tensor.data_ptr(), bytes_, llaisys.MemcpyKind.D2D)
    return torch_tensor, llaisys_tensor
```

**同步随机数据**：先生成 PyTorch 随机 Tensor，再用 `memcpy_sync`（`D2D` = Device to Device）将数据拷贝到 LLAISYS Tensor。两个 Tensor 内容完全相同，保证对比的公平性。

```
def check_equal(llaisys_result, torch_answer, atol=1e-5, rtol=1e-5, strict=False):
    shape = llaisys_result.shape()
    strides = llaisys_result.strides()

    right = 0
    for i in range(len(shape)):
        right += strides[i] * (shape[i] - 1)

    tmp = torch.zeros((right + 1,), dtype=torch_answer.dtype, ...)
    result = torch.as_strided(tmp, shape, strides)
    api.memcpy_sync(result.data_ptr(), llaisys_result.data_ptr(), (right + 1) * tmp.element_size(), llaisys.MemcpyKind.D2D)

    if strict:
        return torch.equal(result, torch_answer)
    else:
        return torch.allclose(result, torch_answer, atol=atol, rtol=rtol)
```

**跨步长验证**：`torch.as_strided(tmp, shape, strides)` 创建一个共享 `tmp` 内存的视图，步长与 LLAISYS Tensor 一致。然后 `memcpy_sync`（D2D）将 LLAISYS 数据拷贝到这个视图，最后用 `torch.allclose` 对比。**这样即使 LLAISYS Tensor 的 stride 与 PyTorch 不同，也能正确对比。**

---

- [x] ### 12.2 test/test_runtime.py — Runtime 测试

```
def test_memcpy(api, size_bytes: int):
    a = torch.zeros((size_bytes,), dtype=torch.uint8, device=torch_device("cpu"))
    b = torch.ones_like(a)
    device_a = api.malloc_device(size_bytes)
    device_b = api.malloc_device(size_bytes)

    api.memcpy_sync(device_a, a.data_ptr(), size_bytes, llaisys.MemcpyKind.H2D)  # Host → Device
    api.memcpy_sync(device_b, device_a, size_bytes, llaisys.MemcpyKind.D2D)      # Device → Device
    api.memcpy_sync(b.data_ptr(), device_b, size_bytes, llaisys.MemcpyKind.D2H)  # Device → Host

    torch.testing.assert_close(a, b)
```

**往返测试（Round-Trip）**：全零 Tensor `a` H2D→Device，D2D→另一个 Device 缓冲区，D2H→全一 Tensor `b`。最终 `a` 和 `b` 应该相等（都是全零）。这个测试验证了 `malloc_device`/`free_device` 和三种 `memcpy` 方向的正确性。

---

- [x] ### 12.3 test/test_tensor.py — Tensor 操作测试

```
def test_tensor():
    torch_tensor = torch.arange(60, dtype=torch_dtype("i64")).reshape(3, 4, 5)
    llaisys_tensor = llaisys.Tensor((3, 4, 5), dtype=llaisys_dtype("i64"))

    # Test load
    llaisys_tensor.load(torch_tensor.data_ptr())
    assert check_equal(llaisys_tensor, torch_tensor)

    # Test view
    torch_tensor_view = torch_tensor.view(6, 10)
    llaisys_tensor_view = llaisys_tensor.view(6, 10)
    assert check_equal(llaisys_tensor_view, torch_tensor_view)

    # Test permute
    torch_tensor_perm = torch_tensor.permute(2, 0, 1)
    llaisys_tensor_perm = llaisys_tensor.permute(2, 0, 1)
    assert check_equal(llaisys_tensor_perm, torch_tensor_perm)

    # Test slice
    torch_tensor_slice = torch_tensor[:, :, 1:4]
    llaisys_tensor_slice = llaisys_tensor.slice(2, 1, 4)
    assert check_equal(llaisys_tensor_slice, torch_tensor_slice)
```

**四个核心操作测试**：`load`（数据拷贝）、`view`（重塑）、`permute`（维度重排）、`slice`（切片）。每个操作都在 PyTorch 和 LLAISYS 上执行等价操作，然后对比结果。测试数据是 `arange(60).reshape(3,4,5)`——有序数据便于发现索引错误。

---

- [x] ### 12.12 test/test_infer.py — 端到端推理测试

```
def hf_infer(prompt, tokenizer, model, max_new_tokens=128, top_p=0.8, top_k=50, temperature=0.8):
    inputs = tokenizer.encode(input_content, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(inputs, max_new_tokens=max_new_tokens, top_k=top_k, top_p=top_p, temperature=temperature)
    return outputs[0].tolist(), result

def llaisys_infer(prompt, tokenizer, model, max_new_tokens=128, top_p=0.8, top_k=50, temperature=0.8):
    inputs = tokenizer.encode(input_content)
    outputs = model.generate(inputs, max_new_tokens=max_new_tokens, top_k=top_k, top_p=top_p, temperature=temperature)
    return outputs, tokenizer.decode(outputs, skip_special_tokens=True)
```

**双引擎对比**：同一个 prompt，分别用 HuggingFace Transformers 和 LLAISYS 推理，对比输出 token 序列。`--test` 模式下使用确定性采样（`top_p=1.0, top_k=1, temperature=1.0`），确保结果可复现。

**测试流程**：
```
1. 加载 HuggingFace 模型 → 推理 → 获取参考 tokens
2. 卸载 HuggingFace 模型（释放内存）
3. 加载 LLAISYS 模型（Qwen2）→ 推理 → 获取 LLAISYS tokens
4. 对比：assert llaisys_tokens == tokens
```

**测试模型**：`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`（15 亿参数，BF16 约 3GB），这是 Project 3 的最终验证目标。

---

- [x] ### 12.4~12.11 算子测试（8 个文件，待实现）

每个算子测试的预期模式（以 `add.py` 为例）：
```
def test_add():
    for shape in [(128,), (64, 64), (32, 32, 32)]:
        for dtype in ["f32", "bf16", "f16"]:
            a_torch, a_lla = random_tensor(shape, dtype, "cpu")
            b_torch, b_lla = random_tensor(shape, dtype, "cpu")
            c_torch = a_torch + b_torch
            c_lla = llaisys.Tensor(shape, dtype=llaisys_dtype(dtype))
            llaisys.Ops.add(c_lla, a_lla, b_lla)
            assert check_equal(c_lla, c_torch)
```

**测试维度**：多种形状 × 多种数据类型 = 覆盖边界情况。测试完成后输出绿色的 `\033[92mTest passed!\033[0m`。

**完整测试流程**：
```
cd llaisys
xmake build llaisys          # 编译 C++ 核心
xmake install -o python/llaisys/libllaisys  # 复制 .dll/.so 到 Python 包
python test/test_tensor.py   # 运行 Tensor 测试
python test/ops/add.py       # 运行 add 算子测试
python test/test_infer.py --test  # 运行端到端推理测试
```