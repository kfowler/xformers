#!/usr/bin/env python3
"""
Comprehensive test suite for xformers on macOS.
Tests CPU-only functionality and verifies the build works correctly.
"""

import sys
import torch
import traceback

def test_import():
    """Test basic imports"""
    print("\n1. Testing imports...")
    try:
        import xformers
        import xformers.ops
        import xformers.ops.fmha as fmha
        from xformers.ops.fmha import pytorch_native
        print(f"   ✓ xformers version: {xformers.__version__}")
        print(f"   ✓ All imports successful")
        return True
    except Exception as e:
        print(f"   ✗ Import failed: {e}")
        traceback.print_exc()
        return False


def test_info():
    """Test xformers.info module"""
    print("\n2. Testing xformers.info...")
    try:
        import xformers.info as info
        # This will print build info
        info.print_info()
        print(f"   ✓ Info module works")
        return True
    except Exception as e:
        print(f"   ✗ Info failed: {e}")
        traceback.print_exc()
        return False


def test_basic_attention():
    """Test basic attention operation on CPU"""
    print("\n3. Testing basic attention operation...")
    try:
        from xformers.ops import memory_efficient_attention

        # Create simple attention inputs
        batch_size = 2
        seq_len = 16
        num_heads = 4
        head_dim = 32

        query = torch.randn(batch_size, seq_len, num_heads, head_dim)
        key = torch.randn(batch_size, seq_len, num_heads, head_dim)
        value = torch.randn(batch_size, seq_len, num_heads, head_dim)

        print(f"   Input shape: {query.shape}")
        print(f"   Device: {query.device}")

        # Run attention
        output = memory_efficient_attention(query, key, value)

        print(f"   Output shape: {output.shape}")
        assert output.shape == query.shape, "Output shape mismatch"
        assert not torch.isnan(output).any(), "Output contains NaN"

        print(f"   ✓ Basic attention works")
        return True
    except Exception as e:
        print(f"   ✗ Basic attention failed: {e}")
        traceback.print_exc()
        return False


def test_attention_with_mask():
    """Test attention with causal mask"""
    print("\n4. Testing attention with causal mask...")
    try:
        from xformers.ops import memory_efficient_attention
        from xformers.ops.fmha.attn_bias import LowerTriangularMask

        batch_size = 2
        seq_len = 16
        num_heads = 4
        head_dim = 32

        query = torch.randn(batch_size, seq_len, num_heads, head_dim)
        key = torch.randn(batch_size, seq_len, num_heads, head_dim)
        value = torch.randn(batch_size, seq_len, num_heads, head_dim)

        # Apply causal mask
        attn_bias = LowerTriangularMask()
        output = memory_efficient_attention(query, key, value, attn_bias=attn_bias)

        assert output.shape == query.shape, "Output shape mismatch"
        assert not torch.isnan(output).any(), "Output contains NaN"

        print(f"   ✓ Causal mask attention works")
        return True
    except Exception as e:
        print(f"   ✗ Masked attention failed: {e}")
        traceback.print_exc()
        return False


def test_attention_backward():
    """Test attention backward pass"""
    print("\n5. Testing attention backward pass...")
    try:
        from xformers.ops import memory_efficient_attention

        batch_size = 2
        seq_len = 16
        num_heads = 4
        head_dim = 32

        query = torch.randn(batch_size, seq_len, num_heads, head_dim, requires_grad=True)
        key = torch.randn(batch_size, seq_len, num_heads, head_dim, requires_grad=True)
        value = torch.randn(batch_size, seq_len, num_heads, head_dim, requires_grad=True)

        # Forward pass
        output = memory_efficient_attention(query, key, value)

        # Backward pass
        loss = output.sum()
        loss.backward()

        assert query.grad is not None, "Query gradient not computed"
        assert key.grad is not None, "Key gradient not computed"
        assert value.grad is not None, "Value gradient not computed"
        assert not torch.isnan(query.grad).any(), "Query gradient contains NaN"

        print(f"   ✓ Backward pass works")
        return True
    except Exception as e:
        print(f"   ✗ Backward pass failed: {e}")
        traceback.print_exc()
        return False


def test_different_dtypes():
    """Test attention with different data types"""
    print("\n6. Testing different data types...")
    try:
        from xformers.ops import memory_efficient_attention

        batch_size = 2
        seq_len = 16
        num_heads = 4
        head_dim = 32

        dtypes = [torch.float32, torch.float16]
        if hasattr(torch, 'bfloat16'):
            dtypes.append(torch.bfloat16)

        for dtype in dtypes:
            query = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=dtype)
            key = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=dtype)
            value = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=dtype)

            output = memory_efficient_attention(query, key, value)
            assert output.dtype == dtype, f"Output dtype mismatch for {dtype}"
            assert not torch.isnan(output).any(), f"Output contains NaN for {dtype}"
            print(f"   ✓ {dtype} works")

        return True
    except Exception as e:
        print(f"   ✗ Dtype test failed: {e}")
        traceback.print_exc()
        return False


def test_operator_dispatch():
    """Test that the correct operator is being dispatched"""
    print("\n7. Testing operator dispatch...")
    try:
        from xformers.ops.fmha import dispatch, pytorch_native
        from xformers.ops.fmha.common import Inputs

        batch_size = 2
        seq_len = 16
        num_heads = 4
        head_dim = 32

        query = torch.randn(batch_size, seq_len, num_heads, head_dim)
        key = torch.randn(batch_size, seq_len, num_heads, head_dim)
        value = torch.randn(batch_size, seq_len, num_heads, head_dim)

        inp = Inputs(query=query, key=key, value=value)

        # Dispatch forward operator
        op = dispatch._dispatch_fw(inp, needs_gradient=False)

        print(f"   Selected operator: {op.NAME}")
        assert op == pytorch_native.FwOp, "Wrong operator selected for CPU"
        print(f"   ✓ Correct CPU operator dispatched (pytorch_native)")
        return True
    except Exception as e:
        print(f"   ✗ Operator dispatch test failed: {e}")
        traceback.print_exc()
        return False


def test_performance_baseline():
    """Test performance and compare with PyTorch native"""
    print("\n8. Running performance baseline...")
    try:
        import time
        from xformers.ops import memory_efficient_attention

        batch_size = 4
        seq_len = 128
        num_heads = 8
        head_dim = 64

        query = torch.randn(batch_size, seq_len, num_heads, head_dim)
        key = torch.randn(batch_size, seq_len, num_heads, head_dim)
        value = torch.randn(batch_size, seq_len, num_heads, head_dim)

        # Warmup
        for _ in range(3):
            _ = memory_efficient_attention(query, key, value)

        # Time xformers
        start = time.time()
        for _ in range(10):
            output = memory_efficient_attention(query, key, value)
        xformers_time = (time.time() - start) / 10

        # Time PyTorch native
        q_native = query.transpose(1, 2)
        k_native = key.transpose(1, 2)
        v_native = value.transpose(1, 2)

        start = time.time()
        for _ in range(10):
            output_native = torch.nn.functional.scaled_dot_product_attention(
                q_native, k_native, v_native
            )
        pytorch_time = (time.time() - start) / 10

        print(f"   xformers time: {xformers_time*1000:.2f}ms")
        print(f"   PyTorch time:  {pytorch_time*1000:.2f}ms")
        print(f"   ✓ Performance test complete")
        return True
    except Exception as e:
        print(f"   ✗ Performance test failed: {e}")
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("xformers macOS Functionality Tests")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: CPU (macOS)")
    print(f"CUDA available: {torch.cuda.is_available()}")

    tests = [
        test_import,
        test_info,
        test_basic_attention,
        test_attention_with_mask,
        test_attention_backward,
        test_different_dtypes,
        test_operator_dispatch,
        test_performance_baseline,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"\n✗ Test {test.__name__} crashed: {e}")
            traceback.print_exc()
            results.append(False)

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("\n✓ All tests passed!")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
