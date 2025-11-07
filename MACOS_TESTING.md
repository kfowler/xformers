# macOS Testing Guide for xformers

This guide explains how to test the macOS support for xformers.

## Prerequisites

1. **macOS System**: These tests must be run on an actual Mac (macOS 10.13 or later)
2. **Python 3.9+**: Install via Homebrew or python.org
3. **PyTorch**: Must be installed before building xformers

## Quick Start

### 1. Install PyTorch

```bash
# Install PyTorch (CPU version for macOS)
pip3 install torch
```

### 2. Clone and Build xformers

```bash
# Navigate to the xformers directory
cd /path/to/xformers

# Initialize submodules (not required for CPU-only build, but doesn't hurt)
git submodule update --init --recursive

# Run the automated test script
./test_macos_build.sh
```

The script will:
- Verify you're on macOS
- Check Python and PyTorch installation
- Clean previous build artifacts
- Build xformers (takes 5-10 minutes)
- Run comprehensive functionality tests
- Report results

## Manual Testing

If you prefer to test manually:

### Build xformers

```bash
# Clean any previous builds
python3 setup.py clean --all
rm -rf build dist *.egg-info

# Build and install in development mode
pip3 install -v --no-build-isolation -e .
```

### Run Tests

```bash
# Run the functionality tests
python3 test_macos_functionality.py
```

### Check Build Info

```bash
# Display build information
python3 -m xformers.info
```

## Expected Output

### Successful Build

You should see:
- ✓ No CUDA/ROCm extensions built (expected on macOS)
- ✓ CPU-only extensions compiled successfully
- ✓ No errors about missing CUTLASS (not needed for CPU builds)

### Successful Tests

All tests should pass:
1. ✓ Imports
2. ✓ Info module
3. ✓ Basic attention operation
4. ✓ Attention with causal mask
5. ✓ Backward pass / gradients
6. ✓ Multiple data types (float32, float16, bfloat16)
7. ✓ Operator dispatch (should use `pytorch_native_cpu`)
8. ✓ Performance baseline

## What's Being Tested

The test suite verifies:

1. **Build System**:
   - CPU-only build works without CUDA
   - macOS-specific compiler flags are applied
   - No CUTLASS dependency for CPU builds

2. **Operator Dispatch**:
   - `pytorch_native.FwOp` is selected for CPU
   - Falls back to PyTorch's `scaled_dot_product_attention`

3. **Functionality**:
   - Forward pass works
   - Backward pass / autograd works
   - Multiple data types supported
   - Attention masks work (causal)
   - No NaN/Inf in outputs

4. **Performance**:
   - Benchmarks against PyTorch native implementation
   - Verifies reasonable performance (should be similar since using same backend)

## Troubleshooting

### Build Fails with "CUTLASS not found"

**Issue**: Old code that checked for CUTLASS unconditionally
**Solution**: Make sure you're on the correct branch with macOS fixes:
```bash
git checkout claude/xformers-macos-support-011CUtLrgcxxoTCYeeBnS3Ma
```

### Import Error: "No module named torch"

**Issue**: PyTorch not installed
**Solution**:
```bash
pip3 install torch
```

### Build Fails with Compiler Errors

**Issue**: Missing Xcode command line tools
**Solution**:
```bash
xcode-select --install
```

### Tests Fail: "Wrong operator selected"

**Issue**: Dispatch logic not selecting CPU operator
**Solution**: Check that you're running on CPU (not CUDA):
```python
import torch
print(torch.cuda.is_available())  # Should be False on macOS
```

### Performance Much Slower than Expected

**Note**: CPU attention is naturally slower than GPU. Comparison should be made against PyTorch's native `scaled_dot_product_attention` on CPU, which should be similar.

## Verification Checklist

After testing, verify:

- [ ] Build completes without errors
- [ ] No CUTLASS-related errors
- [ ] Import `xformers` succeeds
- [ ] All 8 tests pass
- [ ] Operator dispatch selects `pytorch_native_cpu`
- [ ] `python -m xformers.info` shows CPU build
- [ ] No warnings about missing CUDA

## Reporting Issues

If tests fail, please provide:
1. macOS version: `sw_vers`
2. Python version: `python3 --version`
3. PyTorch version: `python3 -c "import torch; print(torch.__version__)"`
4. Build log: `build.log` (created by test script)
5. Test output: Full output from `test_macos_functionality.py`
6. Any error messages or stack traces

## Next Steps

Once tests pass:
- xformers is ready to use on macOS
- All CPU operations will use PyTorch's native attention
- Can be used for development, testing, and CPU inference
- GPU operations require CUDA (not available on macOS)

## Limitations on macOS

**What works:**
- ✓ All CPU operations
- ✓ Attention (via PyTorch native)
- ✓ Training and inference on CPU
- ✓ All Python APIs

**What doesn't work:**
- ✗ CUDA operations (no NVIDIA GPUs on Apple Silicon or Intel Macs)
- ✗ Flash Attention (GPU only)
- ✗ CUTLASS kernels (GPU only)
- ✗ GPU acceleration

For GPU acceleration on Mac, consider PyTorch's MPS (Metal Performance Shaders) backend, though xformers doesn't currently support MPS directly.
