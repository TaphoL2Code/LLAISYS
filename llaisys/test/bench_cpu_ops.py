"""
Project #1: CPU Optimization Benchmark
Measures individual operator execution time for baseline, SIMD, OpenMP, and OpenBLAS configurations.
"""
import gc
import numpy as np
import time
import llaisys

# ============================================================
# Configuration - Simulated Qwen2-1.5B model dimensions
# ============================================================
# Linear: hidden=1536, intermediate=8960
# Attention: qlen=1 (decode), kvlen varies, nh=12, nkvh=2, hd=128
# RMSNorm: rows=1, cols=1536
# SwiGLU: gate/up dim=8960
# RoPE: seq_len=1, n_heads=12, head_dim=128

CONFIG = {
    "linear": {"m": 1, "k": 1536, "n": 8960},
    "rms_norm": {"rows": 1, "cols": 1536},
    "swiglu": {"numel": 8960},
    "rope": {"seq_len": 1, "n_heads": 12, "head_dim": 128},
    "self_attention": {"qlen": 1, "kvlen": 256, "nh": 12, "nkvh": 2, "hd": 128},
}

WARMUP = 10
REPEAT = 100


def _make_tensor(shape, dtype=llaisys.DataType.F32):
    """Create a random llaisys tensor."""
    t = llaisys.Tensor(shape, dtype, llaisys.DeviceType.CPU)
    npy = np.random.randn(*shape).astype(np.float32)
    t.load(npy)
    return t


def bench_linear():
    m, k, n = CONFIG["linear"]["m"], CONFIG["linear"]["k"], CONFIG["linear"]["n"]
    inp = _make_tensor([m, k])
    weight = _make_tensor([n, k])
    bias = _make_tensor([n])
    out = llaisys.Tensor([m, n], llaisys.DataType.F32, llaisys.DeviceType.CPU)

    # Warmup
    for _ in range(WARMUP):
        llaisys.Ops.linear(out, inp, weight, bias)

    gc.disable()
    start = time.perf_counter()
    for _ in range(REPEAT):
        llaisys.Ops.linear(out, inp, weight, bias)
    elapsed = time.perf_counter() - start
    gc.enable()
    return elapsed / REPEAT * 1000  # ms


def bench_rms_norm():
    rows, cols = CONFIG["rms_norm"]["rows"], CONFIG["rms_norm"]["cols"]
    inp = _make_tensor([rows, cols])
    weight = _make_tensor([cols])
    out = llaisys.Tensor([rows, cols], llaisys.DataType.F32, llaisys.DeviceType.CPU)

    for _ in range(WARMUP):
        llaisys.Ops.rms_norm(out, inp, weight, 1e-6)

    gc.disable()
    start = time.perf_counter()
    for _ in range(REPEAT):
        llaisys.Ops.rms_norm(out, inp, weight, 1e-6)
    elapsed = time.perf_counter() - start
    gc.enable()
    return elapsed / REPEAT * 1000


def bench_swiglu():
    numel = CONFIG["swiglu"]["numel"]
    gate = _make_tensor([numel])
    up = _make_tensor([numel])
    out = llaisys.Tensor([numel], llaisys.DataType.F32, llaisys.DeviceType.CPU)

    for _ in range(WARMUP):
        llaisys.Ops.swiglu(out, gate, up)

    gc.disable()
    start = time.perf_counter()
    for _ in range(REPEAT):
        llaisys.Ops.swiglu(out, gate, up)
    elapsed = time.perf_counter() - start
    gc.enable()
    return elapsed / REPEAT * 1000


def bench_rope():
    sl, nh, hd = CONFIG["rope"]["seq_len"], CONFIG["rope"]["n_heads"], CONFIG["rope"]["head_dim"]
    inp = _make_tensor([sl, nh, hd])
    pos_ids = llaisys.Tensor([sl], llaisys.DataType.I64, llaisys.DeviceType.CPU)
    pos_ids.load(np.array([0], dtype=np.int64))
    out = llaisys.Tensor([sl, nh, hd], llaisys.DataType.F32, llaisys.DeviceType.CPU)

    for _ in range(WARMUP):
        llaisys.Ops.rope(out, inp, pos_ids, 10000.0)

    gc.disable()
    start = time.perf_counter()
    for _ in range(REPEAT):
        llaisys.Ops.rope(out, inp, pos_ids, 10000.0)
    elapsed = time.perf_counter() - start
    gc.enable()
    return elapsed / REPEAT * 1000


def bench_self_attention():
    qlen, kvlen, nh, nkvh, hd = (
        CONFIG["self_attention"]["qlen"],
        CONFIG["self_attention"]["kvlen"],
        CONFIG["self_attention"]["nh"],
        CONFIG["self_attention"]["nkvh"],
        CONFIG["self_attention"]["hd"],
    )
    q = _make_tensor([qlen, nh, hd])
    k = _make_tensor([kvlen, nkvh, hd])
    v = _make_tensor([kvlen, nkvh, hd])
    attn_val = llaisys.Tensor([qlen, nh, hd], llaisys.DataType.F32, llaisys.DeviceType.CPU)

    scale = 1.0 / np.sqrt(hd)

    for _ in range(WARMUP):
        llaisys.Ops.self_attention(attn_val, q, k, v, scale)

    gc.disable()
    start = time.perf_counter()
    for _ in range(REPEAT):
        llaisys.Ops.self_attention(attn_val, q, k, v, scale)
    elapsed = time.perf_counter() - start
    gc.enable()
    return elapsed / REPEAT * 1000


def main():
    print("=" * 70)
    print("Project #1: CPU Optimization Benchmark")
    print("=" * 70)

    print(f"\nWarmup: {WARMUP} iterations, Repeat: {REPEAT} iterations")
    print(f"Operator config: {CONFIG}")
    print()

    benchmarks = [
        ("linear", bench_linear),
        ("rms_norm", bench_rms_norm),
        ("swiglu", bench_swiglu),
        ("rope", bench_rope),
        ("self_attention", bench_self_attention),
    ]

    results = {}
    total_time = 0.0
    for name, func in benchmarks:
        t = func()
        results[name] = t
        total_time += t
        print(f"  {name:20s}: {t:10.4f} ms")

    print(f"\n  {'Total':20s}: {total_time:10.4f} ms")

    # Save results
    with open("benchmark_report.txt", "w") as f:
        f.write("Project #1: CPU Optimization - Benchmark Report\n")
        f.write("=" * 60 + "\n\n")
        f.write("Configuration:\n")
        for k, v in CONFIG.items():
            f.write(f"  {k}: {v}\n")
        f.write(f"\nWarmup: {WARMUP}, Repeat: {REPEAT}\n\n")
        f.write("Results (per-op avg, ms):\n")
        f.write("-" * 40 + "\n")
        for name, t in results.items():
            f.write(f"  {name:20s}: {t:10.4f} ms\n")
        f.write("-" * 40 + "\n")
        f.write(f"  {'Total':20s}: {total_time:10.4f} ms\n")

    print("\nReport saved to benchmark_report.txt")


if __name__ == "__main__":
    main()