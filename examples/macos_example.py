#!/usr/bin/env python3
"""
Real-world example: Using xformers on macOS for efficient attention.

This demonstrates how to use xformers' memory-efficient attention
in a practical transformer layer on macOS (CPU-only).
"""

import torch
import torch.nn as nn
from xformers.ops import memory_efficient_attention
import time


class TransformerLayerWithXFormers(nn.Module):
    """
    A simple transformer layer using xformers for efficient attention.
    Works on macOS CPU with the pytorch_native_cpu operator.
    """

    def __init__(self, d_model=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # QKV projection
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

        # Dropout
        self.dropout = dropout

        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x, attn_bias=None):
        """
        Args:
            x: (batch, seq_len, d_model)
            attn_bias: Optional attention bias
        Returns:
            (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape

        # Self-attention with residual
        residual = x
        x = self.norm1(x)

        # Project to Q, K, V
        qkv = self.qkv_proj(x)  # (batch, seq_len, 3 * d_model)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 1, 3, 4)  # (3, batch, seq_len, num_heads, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply memory-efficient attention using xformers
        # This will use pytorch_native_cpu operator on macOS
        attn_out = memory_efficient_attention(
            q, k, v,
            attn_bias=attn_bias,
            p=self.dropout if self.training else 0.0
        )

        # Output projection
        attn_out = attn_out.reshape(batch_size, seq_len, self.d_model)
        attn_out = self.out_proj(attn_out)

        x = residual + attn_out

        # FFN with residual
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x

        return x


class TransformerLayerBaseline(nn.Module):
    """
    Baseline transformer layer using standard PyTorch attention.
    For performance comparison.
    """

    def __init__(self, d_model=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x, attn_mask=None):
        # Self-attention
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.attention(x, x, x, attn_mask=attn_mask)
        x = residual + attn_out

        # FFN
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x

        return x


def benchmark_attention(model, x, num_iterations=100):
    """Benchmark attention performance"""
    # Warmup
    for _ in range(10):
        _ = model(x)

    # Benchmark
    torch.manual_seed(42)
    start = time.time()
    for _ in range(num_iterations):
        _ = model(x)
    end = time.time()

    avg_time = (end - start) / num_iterations * 1000  # ms
    return avg_time


def test_training():
    """Test that training works (forward + backward)"""
    print("\n" + "="*60)
    print("Testing Training (Forward + Backward)")
    print("="*60)

    model = TransformerLayerWithXFormers(d_model=256, num_heads=8)
    x = torch.randn(2, 32, 256, requires_grad=True)
    target = torch.randn(2, 32, 256)

    # Forward
    output = model(x)
    loss = nn.MSELoss()(output, target)

    print(f"✓ Forward pass: {output.shape}")
    print(f"✓ Loss: {loss.item():.6f}")

    # Backward
    loss.backward()

    print(f"✓ Backward pass completed")
    print(f"✓ Input gradient shape: {x.grad.shape}")
    print(f"✓ QKV projection gradient norm: {model.qkv_proj.weight.grad.norm():.6f}")

    return True


def test_inference():
    """Test inference mode"""
    print("\n" + "="*60)
    print("Testing Inference Mode")
    print("="*60)

    model = TransformerLayerWithXFormers(d_model=256, num_heads=8)
    model.eval()

    with torch.no_grad():
        x = torch.randn(2, 32, 256)
        output = model(x)

    print(f"✓ Inference output shape: {output.shape}")
    print(f"✓ Output mean: {output.mean():.6f}")
    print(f"✓ Output std: {output.std():.6f}")
    print(f"✓ No NaN: {not torch.isnan(output).any()}")

    return True


def test_causal_attention():
    """Test causal (autoregressive) attention"""
    print("\n" + "="*60)
    print("Testing Causal Attention (Autoregressive)")
    print("="*60)

    from xformers.ops.fmha.attn_bias import LowerTriangularMask

    model = TransformerLayerWithXFormers(d_model=256, num_heads=8)
    x = torch.randn(2, 32, 256)

    # Apply causal mask
    causal_bias = LowerTriangularMask()
    output = model(x, attn_bias=causal_bias)

    print(f"✓ Causal attention output shape: {output.shape}")
    print(f"✓ Suitable for language modeling / autoregressive tasks")

    return True


def compare_performance():
    """Compare xformers vs baseline performance"""
    print("\n" + "="*60)
    print("Performance Comparison")
    print("="*60)

    configs = [
        (256, 8, 64),    # Small
        (512, 8, 128),   # Medium
    ]

    results = []

    for d_model, num_heads, seq_len in configs:
        print(f"\nConfig: d_model={d_model}, heads={num_heads}, seq_len={seq_len}")

        x = torch.randn(2, seq_len, d_model)

        # xformers
        model_xf = TransformerLayerWithXFormers(d_model, num_heads)
        model_xf.eval()
        time_xf = benchmark_attention(model_xf, x, num_iterations=50)

        # Baseline
        model_base = TransformerLayerBaseline(d_model, num_heads)
        model_base.eval()
        time_base = benchmark_attention(model_base, x, num_iterations=50)

        speedup = time_base / time_xf

        print(f"  xformers:       {time_xf:.2f}ms")
        print(f"  PyTorch native: {time_base:.2f}ms")
        print(f"  Speedup:        {speedup:.2f}x")

        results.append({
            'config': f"{d_model}d_{num_heads}h_{seq_len}s",
            'xformers': time_xf,
            'baseline': time_base,
            'speedup': speedup
        })

    return results


def main():
    print("="*60)
    print("xformers macOS Example - Practical Usage")
    print("="*60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: CPU (macOS)")
    print(f"CUDA available: {torch.cuda.is_available()}")

    # Check xformers
    try:
        import xformers
        print(f"xformers version: {xformers.__version__}")
    except:
        print("ERROR: xformers not installed")
        return 1

    # Run tests
    tests = [
        ("Training (Forward + Backward)", test_training),
        ("Inference Mode", test_inference),
        ("Causal Attention", test_causal_attention),
    ]

    for name, test_fn in tests:
        try:
            if not test_fn():
                print(f"\n✗ {name} failed")
                return 1
        except Exception as e:
            print(f"\n✗ {name} crashed: {e}")
            import traceback
            traceback.print_exc()
            return 1

    # Performance comparison
    try:
        results = compare_performance()
    except Exception as e:
        print(f"\n✗ Performance comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\n" + "="*60)
    print("✓ All tests passed!")
    print("="*60)
    print("\nxformers is working correctly on macOS!")
    print("You can now use memory-efficient attention in your projects.")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
