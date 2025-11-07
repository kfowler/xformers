#!/usr/bin/env python3
"""
Comprehensive performance benchmark for xformers on macOS.

Compares xformers memory-efficient attention against PyTorch native
scaled_dot_product_attention on CPU.
"""

import torch
import torch.nn.functional as F
from xformers.ops import memory_efficient_attention
import time
import sys


def benchmark_operation(fn, *args, warmup=10, iterations=100, **kwargs):
    """Benchmark a function with warmup"""
    # Warmup
    for _ in range(warmup):
        fn(*args, **kwargs)

    # Benchmark
    start = time.time()
    for _ in range(iterations):
        fn(*args, **kwargs)
    end = time.time()

    avg_time_ms = (end - start) / iterations * 1000
    return avg_time_ms


def xformers_attention(query, key, value, attn_bias=None, scale=None):
    """Wrapper for xformers attention"""
    return memory_efficient_attention(query, key, value, attn_bias=attn_bias, scale=scale)


def pytorch_attention(query, key, value, is_causal=False, scale=None):
    """Wrapper for PyTorch native attention"""
    # Convert from BMHK to BHMD format
    q = query.transpose(1, 2)
    k = key.transpose(1, 2)
    v = value.transpose(1, 2)

    out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, scale=scale)
    return out.transpose(1, 2)


def run_benchmark_suite():
    """Run comprehensive benchmark suite"""

    print("="*70)
    print("xformers macOS Performance Benchmark")
    print("="*70)
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: CPU")
    print()

    # Test configurations
    configs = [
        # (batch, seq_len, num_heads, head_dim, name)
        (1, 32, 4, 64, "Tiny (GPT-2 Small-like)"),
        (2, 64, 8, 64, "Small"),
        (2, 128, 8, 64, "Medium (BERT-like)"),
        (1, 256, 12, 64, "Large"),
        (1, 512, 8, 64, "Very Large"),
        (4, 128, 8, 64, "Batch Processing"),
    ]

    results = []

    for batch, seq_len, num_heads, head_dim, name in configs:
        print(f"\n{'='*70}")
        print(f"Configuration: {name}")
        print(f"  Shape: batch={batch}, seq_len={seq_len}, heads={num_heads}, dim={head_dim}")
        print(f"  Memory: ~{(batch * seq_len * num_heads * head_dim * 4 * 3 / 1024):.1f} KB")
        print(f"{'='*70}")

        # Create test tensors
        query = torch.randn(batch, seq_len, num_heads, head_dim)
        key = torch.randn(batch, seq_len, num_heads, head_dim)
        value = torch.randn(batch, seq_len, num_heads, head_dim)

        # Benchmark xformers
        time_xf = benchmark_operation(
            xformers_attention, query, key, value,
            warmup=5, iterations=50
        )

        # Benchmark PyTorch native
        time_pt = benchmark_operation(
            pytorch_attention, query, key, value,
            warmup=5, iterations=50
        )

        overhead = ((time_xf - time_pt) / time_pt) * 100
        speedup = time_pt / time_xf

        print(f"\nResults:")
        print(f"  xformers:       {time_xf:7.2f} ms")
        print(f"  PyTorch native: {time_pt:7.2f} ms")
        print(f"  Overhead:       {overhead:+6.1f}%")
        print(f"  Speedup:        {speedup:6.2f}x")

        if abs(overhead) < 15:
            status = "✓ Excellent"
        elif abs(overhead) < 30:
            status = "✓ Good"
        else:
            status = "⚠ High overhead"

        print(f"  Status:         {status}")

        results.append({
            'name': name,
            'batch': batch,
            'seq_len': seq_len,
            'num_heads': num_heads,
            'head_dim': head_dim,
            'time_xf': time_xf,
            'time_pt': time_pt,
            'overhead': overhead,
            'speedup': speedup,
        })

    return results


def test_dtype_performance():
    """Test performance across different dtypes"""
    print(f"\n{'='*70}")
    print("Data Type Performance Comparison")
    print(f"{'='*70}")

    config = (2, 128, 8, 64)  # batch, seq_len, heads, dim
    dtypes = [torch.float32, torch.float16, torch.bfloat16]

    for dtype in dtypes:
        query = torch.randn(*config, dtype=dtype)
        key = torch.randn(*config, dtype=dtype)
        value = torch.randn(*config, dtype=dtype)

        time_xf = benchmark_operation(
            xformers_attention, query, key, value,
            warmup=5, iterations=50
        )

        print(f"  {str(dtype):20s}: {time_xf:7.2f} ms")


def test_causal_performance():
    """Test causal attention performance"""
    print(f"\n{'='*70}")
    print("Causal Attention Performance")
    print(f"{'='*70}")

    from xformers.ops.fmha.attn_bias import LowerTriangularMask

    config = (2, 128, 8, 64)
    query = torch.randn(*config)
    key = torch.randn(*config)
    value = torch.randn(*config)

    # Without causal mask
    time_normal = benchmark_operation(
        xformers_attention, query, key, value,
        warmup=5, iterations=50
    )

    # With causal mask
    causal_bias = LowerTriangularMask()
    time_causal = benchmark_operation(
        xformers_attention, query, key, value,
        attn_bias=causal_bias,
        warmup=5, iterations=50
    )

    overhead = ((time_causal - time_normal) / time_normal) * 100

    print(f"  Normal attention:  {time_normal:7.2f} ms")
    print(f"  Causal attention:  {time_causal:7.2f} ms")
    print(f"  Overhead:          {overhead:+6.1f}%")


def print_summary(results):
    """Print benchmark summary"""
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")

    avg_overhead = sum(r['overhead'] for r in results) / len(results)
    avg_speedup = sum(r['speedup'] for r in results) / len(results)

    print(f"\nAverage overhead: {avg_overhead:+.1f}%")
    print(f"Average speedup:  {avg_speedup:.2f}x")

    if abs(avg_overhead) < 15:
        conclusion = "✓ EXCELLENT - xformers overhead is minimal"
    elif abs(avg_overhead) < 30:
        conclusion = "✓ GOOD - xformers overhead is acceptable"
    else:
        conclusion = "⚠ HIGH - xformers has significant overhead"

    print(f"\nConclusion: {conclusion}")
    print("\nNote: On CPU, both xformers and PyTorch use similar backends,")
    print("so similar performance is expected. The real benefits of xformers")
    print("come on GPU with specialized kernels (Flash Attention, CUTLASS).")


def main():
    # Run main benchmark suite
    try:
        results = run_benchmark_suite()
    except Exception as e:
        print(f"\n✗ Main benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Run dtype tests
    try:
        test_dtype_performance()
    except Exception as e:
        print(f"\n✗ Dtype test failed: {e}")

    # Run causal attention test
    try:
        test_causal_performance()
    except Exception as e:
        print(f"\n✗ Causal test failed: {e}")

    # Print summary
    print_summary(results)

    print(f"\n{'='*70}")
    print("✓ Benchmark completed successfully!")
    print(f"{'='*70}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
