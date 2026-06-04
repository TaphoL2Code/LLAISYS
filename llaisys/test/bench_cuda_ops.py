"""CUDA Operator Test Script - does not require PyTorch CUDA
Tests correctness of all 7 CUDA operators by comparing against numpy CPU results.
"""
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
import llaisys
import numpy as np
import time
import argparse


def np_dtype(llaisys_dtype):
    """Convert llaisys dtype to numpy dtype."""
    mapping = {
        llaisys.DataType.F32: np.float32,
        llaisys.DataType.F64: np.float64,
        llaisys.DataType.I32: np.int32,
        llaisys.DataType.I64: np.int64,
    }
    return mapping.get(llaisys_dtype, np.float32)


def cuda_sync():
    """Synchronize CUDA device."""
    api = llaisys.RuntimeAPI(llaisys.DeviceType.NVIDIA)
    api.device_synchronize()


def test_argmax():
    """Test argmax CUDA kernel."""
    print("=== Testing argmax (CUDA) ===")
    for shape in [(4,), (256,), (4096,), (16384,)]:
        for dtype_name in ["f32"]:
            dtype = llaisys_dtype(dtype_name)
            npy = np.random.randn(*shape).astype(np_dtype(dtype))

            # Create llaisys CUDA tensors
            vals = llaisys.Tensor(shape, dtype=dtype, device=llaisys.DeviceType.NVIDIA, device_id=0)
            max_idx = llaisys.Tensor((1,), dtype=llaisys.DataType.I64, device=llaisys.DeviceType.NVIDIA, device_id=0)
            max_val = llaisys.Tensor((1,), dtype=dtype, device=llaisys.DeviceType.NVIDIA, device_id=0)

            # Copy input to GPU
            api = llaisys.RuntimeAPI(llaisys.DeviceType.NVIDIA)
            api.memcpy_sync(vals.data_ptr(), npy.ctypes.data, npy.nbytes, llaisys.MemcpyKind.H2D)
            cuda_sync()

            # Run kernel
            llaisys.Ops.argmax(max_idx, max_val, vals)
            cuda_sync()

            # Copy result back to CPU
            result_idx = np.zeros((1,), dtype=np.int64)
            result_val = np.zeros((1,), dtype=np_dtype(dtype))
            api.memcpy_sync(result_idx.ctypes.data, max_idx.data_ptr(), 8, llaisys.MemcpyKind.D2H)
            api.memcpy_sync(result_val.ctypes.data, max_val.data_ptr(), result_val.nbytes, llaisys.MemcpyKind.D2H)
            cuda_sync()

            # Expected
            expected_idx = np.argmax(npy)
            expected_val = npy.flatten()[expected_idx]

            ok = (result_idx[0] == expected_idx) and (abs(result_val[0] - expected_val) < 1e-5)
            status = "PASS" if ok else "FAIL"
            print(f"  shape={shape} {status}: idx={result_idx[0]} (expected {expected_idx}), val={result_val[0]:.6f} (expected {expected_val:.6f})")
            if not ok:
                return False
    return True


def test_embedding():
    """Test embedding CUDA kernel."""
    print("=== Testing embedding (CUDA) ===")
    for vocab_size, dim in [(128, 64), (1024, 256), (32000, 512)]:
        for n_tokens in [1, 4, 16]:
            dtype = llaisys.DataType.F32
            # Create random embedding table
            embed_npy = np.random.randn(vocab_size, dim).astype(np.float32)
            # Create random input token IDs
            input_ids = np.random.randint(0, vocab_size, (n_tokens,)).astype(np.int64)

            # Create llaisys CUDA tensors
            embed = llaisys.Tensor((vocab_size, dim), dtype=dtype, device=llaisys.DeviceType.NVIDIA, device_id=0)
            ids = llaisys.Tensor((n_tokens,), dtype=llaisys.DataType.I64, device=llaisys.DeviceType.NVIDIA, device_id=0)
            out = llaisys.Tensor((n_tokens, dim), dtype=dtype, device=llaisys.DeviceType.NVIDIA, device_id=0)

            api = llaisys.RuntimeAPI(llaisys.DeviceType.NVIDIA)
            api.memcpy_sync(embed.data_ptr(), embed_npy.ctypes.data, embed_npy.nbytes, llaisys.MemcpyKind.H2D)
            api.memcpy_sync(ids.data_ptr(), input_ids.ctypes.data, input_ids.nbytes, llaisys.MemcpyKind.H2D)
            cuda_sync()

            llaisys.Ops.embedding(out, ids, embed)
            cuda_sync()

            result = np.zeros((n_tokens, dim), dtype=np.float32)
            api.memcpy_sync(result.ctypes.data, out.data_ptr(), result.nbytes, llaisys.MemcpyKind.D2H)
            cuda_sync()

            expected = embed_npy[input_ids]
            ok = np.allclose(result, expected, atol=1e-5)
            status = "PASS" if ok else "FAIL"
            print(f"  vocab={vocab_size} dim={dim} tokens={n_tokens} {status}")
            if not ok:
                print(f"    Max diff: {np.max(np.abs(result - expected))}")
                return False
    return True


def test_linear():
    """Test linear CUDA kernel."""
    print("=== Testing linear (CUDA) ===")
    for m, k, n, use_bias in [
        (2, 4, 3, True),
        (2, 4, 3, False),
        (32, 64, 128, True),
        (512, 4096, 4096, True),
    ]:
        dtype = llaisys.DataType.F32
        x_npy = np.random.randn(m, k).astype(np.float32) * 0.1
        w_npy = np.random.randn(n, k).astype(np.float32) * 0.01
        bias_npy = np.random.randn(n).astype(np.float32) if use_bias else None

        # Create llaisys CUDA tensors
        out = llaisys.Tensor((m, n), dtype=dtype, device=llaisys.DeviceType.NVIDIA, device_id=0)
        x = llaisys.Tensor((m, k), dtype=dtype, device=llaisys.DeviceType.NVIDIA, device_id=0)
        w = llaisys.Tensor((n, k), dtype=dtype, device=llaisys.DeviceType.NVIDIA, device_id=0)
        bias = llaisys.Tensor((n,), dtype=dtype, device=llaisys.DeviceType.NVIDIA, device_id=0) if use_bias else None

        api = llaisys.RuntimeAPI(llaisys.DeviceType.NVIDIA)
        api.memcpy_sync(x.data_ptr(), x_npy.ctypes.data, x_npy.nbytes, llaisys.MemcpyKind.H2D)
        api.memcpy_sync(w.data_ptr(), w_npy.ctypes.data, w_npy.nbytes, llaisys.MemcpyKind.H2D)
        if use_bias:
            api.memcpy_sync(bias.data_ptr(), bias_npy.ctypes.data, bias_npy.nbytes, llaisys.MemcpyKind.H2D)
        cuda_sync()

        llaisys.Ops.linear(out, x, w, bias)
        cuda_sync()

        result = np.zeros((m, n), dtype=np.float32)
        api.memcpy_sync(result.ctypes.data, out.data_ptr(), result.nbytes, llaisys.MemcpyKind.D2H)
        cuda_sync()

        expected = x_npy @ w_npy.T
        if use_bias:
            expected += bias_npy

        ok = np.allclose(result, expected, atol=1e-3, rtol=1e-3)
        status = "PASS" if ok else "FAIL"
        print(f"  m={m} k={k} n={n} bias={use_bias} {status}")
        if not ok:
            print(f"    Max diff: {np.max(np.abs(result - expected))}")
            return False
    return True


def test_rms_norm():
    """Test rms_norm CUDA kernel."""
    print("=== Testing rms_norm (CUDA) ===")
    for shape, eps in [
        ((4, 8), 1e-6),
        ((32, 128), 1e-6),
        ((128, 4096), 1e-6),
        ((1, 4096), 1e-6),
    ]:
        dtype = llaisys.DataType.F32
        x_npy = np.random.randn(*shape).astype(np.float32)
        weight_npy = np.ones(shape[-1], dtype=np.float32)

        out = llaisys.Tensor(shape, dtype=dtype, device=llaisys.DeviceType.NVIDIA, device_id=0)
        x = llaisys.Tensor(shape, dtype=dtype, device=llaisys.DeviceType.NVIDIA, device_id=0)
        weight = llaisys.Tensor((shape[-1],), dtype=dtype, device=llaisys.DeviceType.NVIDIA, device_id=0)

        api = llaisys.RuntimeAPI(llaisys.DeviceType.NVIDIA)
        api.memcpy_sync(x.data_ptr(), x_npy.ctypes.data, x_npy.nbytes, llaisys.MemcpyKind.H2D)
        api.memcpy_sync(weight.data_ptr(), weight_npy.ctypes.data, weight_npy.nbytes, llaisys.MemcpyKind.H2D)
        cuda_sync()

        llaisys.Ops.rms_norm(out, x, weight, eps)
        cuda_sync()

        result = np.zeros(shape, dtype=np.float32)
        api.memcpy_sync(result.ctypes.data, out.data_ptr(), result.nbytes, llaisys.MemcpyKind.D2H)
        cuda_sync()

        # Expected: x * weight / sqrt(mean(x^2) + eps)
        rms = np.sqrt(np.mean(x_npy ** 2, axis=-1, keepdims=True) + eps)
        expected = x_npy * weight_npy / rms

        ok = np.allclose(result, expected, atol=1e-3, rtol=1e-3)
        status = "PASS" if ok else "FAIL"
        print(f"  shape={shape} {status}")
        if not ok:
            print(f"    Max diff: {np.max(np.abs(result - expected))}")
            return False
    return True


def test_rope():
    """Test rope CUDA kernel."""
    print("=== Testing rope (CUDA) ===")
    for seq_len, n_heads, head_dim in [
        (4, 2, 8),
        (8, 4, 64),
        (32, 8, 128),
    ]:
        dtype = llaisys.DataType.F32
        shape = (seq_len, n_heads, head_dim)
        x_npy = np.random.randn(*shape).astype(np.float32)

        out = llaisys.Tensor(shape, dtype=dtype, device=llaisys.DeviceType.NVIDIA, device_id=0)
        x = llaisys.Tensor(shape, dtype=dtype, device=llaisys.DeviceType.NVIDIA, device_id=0)
        pos_ids_npy = np.arange(seq_len, dtype=np.int64)
        pos_ids = llaisys.Tensor((seq_len,), dtype=llaisys.DataType.I64, device=llaisys.DeviceType.NVIDIA, device_id=0)

        api = llaisys.RuntimeAPI(llaisys.DeviceType.NVIDIA)
        api.memcpy_sync(x.data_ptr(), x_npy.ctypes.data, x_npy.nbytes, llaisys.MemcpyKind.H2D)
        api.memcpy_sync(pos_ids.data_ptr(), pos_ids_npy.ctypes.data, pos_ids_npy.nbytes, llaisys.MemcpyKind.H2D)
        cuda_sync()

        llaisys.Ops.rope(out, x, pos_ids, 10000.0)
        cuda_sync()

        result = np.zeros(shape, dtype=np.float32)
        api.memcpy_sync(result.ctypes.data, out.data_ptr(), result.nbytes, llaisys.MemcpyKind.D2H)
        cuda_sync()

        # Expected: apply RoPE rotation (first half vs second half)
        expected = x_npy.copy()
        half_dim = head_dim // 2
        for pos in range(seq_len):
            for h in range(n_heads):
                for j in range(half_dim):
                    theta_freq = 1.0 / (10000.0 ** (2.0 * j / head_dim))
                    angle = pos * theta_freq
                    cos_a, sin_a = np.cos(angle), np.sin(angle)
                    x0 = expected[pos, h, j]
                    x1 = expected[pos, h, j + half_dim]
                    expected[pos, h, j] = x0 * cos_a - x1 * sin_a
                    expected[pos, h, j + half_dim] = x0 * sin_a + x1 * cos_a

        ok = np.allclose(result, expected, atol=1e-3, rtol=1e-3)
        status = "PASS" if ok else "FAIL"
        print(f"  seq={seq_len} heads={n_heads} dim={head_dim} {status}")
        if not ok:
            print(f"    Max diff: {np.max(np.abs(result - expected))}")
            return False
    return True


def test_self_attention():
    """Test self_attention CUDA kernel."""
    print("=== Testing self_attention (CUDA) ===")
    for qlen, kvlen, nh, nkvh, hd in [
        (4, 4, 2, 2, 8),
        (8, 8, 4, 4, 32),
        (16, 16, 8, 2, 64),
    ]:
        dtype = llaisys.DataType.F32
        q_npy = np.random.randn(qlen, nh, hd).astype(np.float32) * 0.1
        k_npy = np.random.randn(kvlen, nkvh, hd).astype(np.float32) * 0.1
        v_npy = np.random.randn(kvlen, nkvh, hd).astype(np.float32) * 0.1
        scale = 1.0 / np.sqrt(hd)

        out_shape = (qlen, nh, hd)
        out = llaisys.Tensor(out_shape, dtype=dtype, device=llaisys.DeviceType.NVIDIA, device_id=0)
        q = llaisys.Tensor((qlen, nh, hd), dtype=dtype, device=llaisys.DeviceType.NVIDIA, device_id=0)
        k = llaisys.Tensor((kvlen, nkvh, hd), dtype=dtype, device=llaisys.DeviceType.NVIDIA, device_id=0)
        v = llaisys.Tensor((kvlen, nkvh, hd), dtype=dtype, device=llaisys.DeviceType.NVIDIA, device_id=0)

        api = llaisys.RuntimeAPI(llaisys.DeviceType.NVIDIA)
        api.memcpy_sync(q.data_ptr(), q_npy.ctypes.data, q_npy.nbytes, llaisys.MemcpyKind.H2D)
        api.memcpy_sync(k.data_ptr(), k_npy.ctypes.data, k_npy.nbytes, llaisys.MemcpyKind.H2D)
        api.memcpy_sync(v.data_ptr(), v_npy.ctypes.data, v_npy.nbytes, llaisys.MemcpyKind.H2D)
        cuda_sync()

        llaisys.Ops.self_attention(out, q, k, v, scale)
        cuda_sync()

        result = np.zeros(out_shape, dtype=np.float32)
        api.memcpy_sync(result.ctypes.data, out.data_ptr(), result.nbytes, llaisys.MemcpyKind.D2H)
        cuda_sync()

        # Expected: softmax(QK^T / sqrt(d)) * V
        expected = np.zeros_like(result)
        for q_seq in range(qlen):
            for qh in range(nh):
                kvh = qh // (nh // nkvh)
                scores = np.zeros(kvlen)
                for kv_seq in range(kvlen):
                    scores[kv_seq] = np.dot(q_npy[q_seq, qh, :], k_npy[kv_seq, kvh, :])
                scores = scores * scale
                scores = scores - np.max(scores)
                probs = np.exp(scores) / np.sum(np.exp(scores))
                for d in range(hd):
                    expected[q_seq, qh, d] = np.sum(probs * v_npy[:, kvh, d])

        ok = np.allclose(result, expected, atol=1e-2, rtol=1e-2)
        status = "PASS" if ok else "FAIL"
        print(f"  qlen={qlen} kvlen={kvlen} nh={nh} nkvh={nkvh} hd={hd} {status}")
        if not ok:
            print(f"    Max diff: {np.max(np.abs(result - expected))}")
            return False
    return True


def test_swiglu():
    """Test swiglu CUDA kernel."""
    print("=== Testing swiglu (CUDA) ===")
    for shape in [(4, 8), (32, 128), (128, 4096), (512, 11008)]:
        dtype = llaisys.DataType.F32
        x_npy = np.random.randn(*shape).astype(np.float32) * 0.1

        out = llaisys.Tensor(shape, dtype=dtype, device=llaisys.DeviceType.NVIDIA, device_id=0)
        x = llaisys.Tensor(shape, dtype=dtype, device=llaisys.DeviceType.NVIDIA, device_id=0)

        api = llaisys.RuntimeAPI(llaisys.DeviceType.NVIDIA)
        api.memcpy_sync(x.data_ptr(), x_npy.ctypes.data, x_npy.nbytes, llaisys.MemcpyKind.H2D)
        cuda_sync()

        llaisys.Ops.swiglu(out, x, x)
        cuda_sync()

        result = np.zeros(shape, dtype=np.float32)
        api.memcpy_sync(result.ctypes.data, out.data_ptr(), result.nbytes, llaisys.MemcpyKind.D2H)
        cuda_sync()

        # Expected: gate * sigmoid(gate) * up (with gate=up=x, this is x^2 * sigmoid(x))
        sigmoid = 1.0 / (1.0 + np.exp(-x_npy))
        expected = x_npy * sigmoid * x_npy

        ok = np.allclose(result, expected, atol=1e-3, rtol=1e-3)
        status = "PASS" if ok else "FAIL"
        print(f"  shape={shape} {status}")
        if not ok:
            print(f"    Max diff: {np.max(np.abs(result - expected))}")
            return False
    return True


def llaisys_dtype(name):
    mapping = {
        "f32": llaisys.DataType.F32,
        "f64": llaisys.DataType.F64,
        "i32": llaisys.DataType.I32,
        "i64": llaisys.DataType.I64,
    }
    return mapping.get(name, llaisys.DataType.F32)


def benchmark_ops():
    """Run performance benchmarks for all CUDA operators."""
    print("\n=== CUDA Operator Performance Benchmarks ===")
    api = llaisys.RuntimeAPI(llaisys.DeviceType.NVIDIA)
    dtype = llaisys.DataType.F32

    repeat = 50
    warmup = 10

    # --- argmax ---
    shape = (16384,)
    npy = np.random.randn(*shape).astype(np.float32)
    vals = llaisys.Tensor(shape, dtype=dtype, device=llaisys.DeviceType.NVIDIA, device_id=0)
    max_idx = llaisys.Tensor((1,), dtype=llaisys.DataType.I64, device=llaisys.DeviceType.NVIDIA, device_id=0)
    max_val = llaisys.Tensor((1,), dtype=dtype, device=llaisys.DeviceType.NVIDIA, device_id=0)
    api.memcpy_sync(vals.data_ptr(), npy.ctypes.data, npy.nbytes, llaisys.MemcpyKind.H2D)
    api.device_synchronize()

    for _ in range(warmup):
        llaisys.Ops.argmax(max_idx, max_val, vals)
    api.device_synchronize()
    start = time.time()
    for _ in range(repeat):
        llaisys.Ops.argmax(max_idx, max_val, vals)
    api.device_synchronize()
    elapsed = (time.time() - start) / repeat
    print(f"  argmax   shape={shape}: {elapsed*1000:.4f} ms")

    # --- linear ---
    m, k, n = 512, 4096, 4096
    x_npy = np.random.randn(m, k).astype(np.float32) * 0.1
    w_npy = np.random.randn(n, k).astype(np.float32) * 0.01
    out = llaisys.Tensor((m, n), dtype=dtype, device=llaisys.DeviceType.NVIDIA, device_id=0)
    x = llaisys.Tensor((m, k), dtype=dtype, device=llaisys.DeviceType.NVIDIA, device_id=0)
    w = llaisys.Tensor((n, k), dtype=dtype, device=llaisys.DeviceType.NVIDIA, device_id=0)
    bias = llaisys.Tensor((n,), dtype=dtype, device=llaisys.DeviceType.NVIDIA, device_id=0)
    api.memcpy_sync(x.data_ptr(), x_npy.ctypes.data, x_npy.nbytes, llaisys.MemcpyKind.H2D)
    api.memcpy_sync(w.data_ptr(), w_npy.ctypes.data, w_npy.nbytes, llaisys.MemcpyKind.H2D)
    api.memcpy_sync(bias.data_ptr(), np.zeros(n, dtype=np.float32).ctypes.data, n * 4, llaisys.MemcpyKind.H2D)
    api.device_synchronize()

    for _ in range(warmup):
        llaisys.Ops.linear(out, x, w, bias)
    api.device_synchronize()
    start = time.time()
    for _ in range(repeat):
        llaisys.Ops.linear(out, x, w, bias)
    api.device_synchronize()
    elapsed = (time.time() - start) / repeat
    print(f"  linear   m={m} k={k} n={n}: {elapsed*1000:.4f} ms")

    # --- rms_norm ---
    shape = (128, 4096)
    x_npy = np.random.randn(*shape).astype(np.float32)
    weight_npy = np.ones(shape[-1], dtype=np.float32)
    out = llaisys.Tensor(shape, dtype=dtype, device=llaisys.DeviceType.NVIDIA, device_id=0)
    x = llaisys.Tensor(shape, dtype=dtype, device=llaisys.DeviceType.NVIDIA, device_id=0)
    weight = llaisys.Tensor((shape[-1],), dtype=dtype, device=llaisys.DeviceType.NVIDIA, device_id=0)
    api.memcpy_sync(x.data_ptr(), x_npy.ctypes.data, x_npy.nbytes, llaisys.MemcpyKind.H2D)
    api.memcpy_sync(weight.data_ptr(), weight_npy.ctypes.data, weight_npy.nbytes, llaisys.MemcpyKind.H2D)
    api.device_synchronize()

    for _ in range(warmup):
        llaisys.Ops.rms_norm(out, x, weight, 1e-6)
    api.device_synchronize()
    start = time.time()
    for _ in range(repeat):
        llaisys.Ops.rms_norm(out, x, weight, 1e-6)
    api.device_synchronize()
    elapsed = (time.time() - start) / repeat
    print(f"  rms_norm shape={shape}: {elapsed*1000:.4f} ms")

    # --- rope ---
    shape = (32, 8, 128)
    x_npy = np.random.randn(*shape).astype(np.float32)
    out = llaisys.Tensor(shape, dtype=dtype, device=llaisys.DeviceType.NVIDIA, device_id=0)
    x = llaisys.Tensor(shape, dtype=dtype, device=llaisys.DeviceType.NVIDIA, device_id=0)
    pos_ids_npy = np.arange(shape[0], dtype=np.int64)
    pos_ids = llaisys.Tensor((shape[0],), dtype=llaisys.DataType.I64, device=llaisys.DeviceType.NVIDIA, device_id=0)
    api.memcpy_sync(x.data_ptr(), x_npy.ctypes.data, x_npy.nbytes, llaisys.MemcpyKind.H2D)
    api.memcpy_sync(pos_ids.data_ptr(), pos_ids_npy.ctypes.data, pos_ids_npy.nbytes, llaisys.MemcpyKind.H2D)
    api.device_synchronize()

    for _ in range(warmup):
        llaisys.Ops.rope(out, x, pos_ids, 10000.0)
    api.device_synchronize()
    start = time.time()
    for _ in range(repeat):
        llaisys.Ops.rope(out, x, pos_ids, 10000.0)
    api.device_synchronize()
    elapsed = (time.time() - start) / repeat
    print(f"  rope     shape={shape}: {elapsed*1000:.4f} ms")

    # --- swiglu ---
    shape = (512, 11008)
    x_npy = np.random.randn(*shape).astype(np.float32) * 0.1
    out = llaisys.Tensor(shape, dtype=dtype, device=llaisys.DeviceType.NVIDIA, device_id=0)
    x = llaisys.Tensor(shape, dtype=dtype, device=llaisys.DeviceType.NVIDIA, device_id=0)
    api.memcpy_sync(x.data_ptr(), x_npy.ctypes.data, x_npy.nbytes, llaisys.MemcpyKind.H2D)
    api.device_synchronize()

    for _ in range(warmup):
        llaisys.Ops.swiglu(out, x, x)
    api.device_synchronize()
    start = time.time()
    for _ in range(repeat):
        llaisys.Ops.swiglu(out, x, x)
    api.device_synchronize()
    elapsed = (time.time() - start) / repeat
    print(f"  swiglu   shape={shape}: {elapsed*1000:.4f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmarks")
    args = parser.parse_args()

    print("CUDA Operator Correctness Tests")
    print("=" * 50)

    all_pass = True
    all_pass &= test_argmax()
    all_pass &= test_embedding()
    all_pass &= test_linear()
    all_pass &= test_rms_norm()
    all_pass &= test_rope()
    all_pass &= test_self_attention()
    all_pass &= test_swiglu()

    if all_pass:
        print("\n\033[92mAll CUDA operator tests passed!\033[0m")
    else:
        print("\n\033[91mSome tests FAILED!\033[0m")

    if args.benchmark:
        benchmark_ops()