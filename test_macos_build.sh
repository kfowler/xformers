#!/bin/bash
# macOS Build and Test Script for xformers
# Run this script on a Mac to verify the macOS support

set -e

echo "========================================="
echo "xformers macOS Build and Test Script"
echo "========================================="
echo ""

# Check if we're on macOS
if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "ERROR: This script must be run on macOS"
    exit 1
fi

echo "✓ Running on macOS $(sw_vers -productVersion)"
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"
echo ""

# Check if PyTorch is installed
echo "Checking PyTorch installation..."
if ! python3 -c "import torch" 2>/dev/null; then
    echo "ERROR: PyTorch is not installed"
    echo "Please install PyTorch first:"
    echo "  pip3 install torch"
    exit 1
fi

torch_version=$(python3 -c "import torch; print(torch.__version__)")
echo "✓ PyTorch version: $torch_version"
echo "✓ CUDA available: $(python3 -c "import torch; print(torch.cuda.is_available())")"
echo ""

# Clean previous build artifacts
echo "Cleaning previous build artifacts..."
python3 setup.py clean --all 2>/dev/null || true
rm -rf build dist *.egg-info
rm -rf xformers/_C*.so xformers/*.so
echo "✓ Clean complete"
echo ""

# Build xformers
echo "Building xformers (this may take 5-10 minutes)..."
echo "----------------------------------------"
pip3 install -v --no-build-isolation -e . 2>&1 | tee build.log

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Build failed. Check build.log for details"
    exit 1
fi

echo ""
echo "✓ Build successful!"
echo ""

# Run basic import test
echo "Testing xformers import..."
python3 -c "import xformers; print(f'✓ xformers version: {xformers.__version__}')" || {
    echo "ERROR: Failed to import xformers"
    exit 1
}
echo ""

# Run comprehensive tests
echo "Running comprehensive tests..."
echo "----------------------------------------"
python3 test_macos_functionality.py

echo ""
echo "========================================="
echo "✓ All tests completed successfully!"
echo "========================================="
