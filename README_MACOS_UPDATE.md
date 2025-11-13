# Suggested README.md Updates for macOS Support

## Recommendation

Add the following section after the existing installation instructions (after line 57):

---

## Installation Section Addition

Add after the "Install from source" section:

```markdown
* **macOS (CPU-only)**: Build from source for CPU-based development and testing

```bash
# Install PyTorch first
pip install torch

# Clone and build xformers
git clone https://github.com/facebookresearch/xformers.git
cd xformers
python setup.py develop --user

# Verify installation
python -m xformers.info
```

**Note for macOS users:**
- macOS builds use CPU-only operators (no CUDA/GPU support)
- All attention operations use PyTorch's native `scaled_dot_product_attention`
- Performance is suitable for development, testing, and CPU inference
- For production GPU workloads, use Linux with CUDA

See [MACOS_TESTING.md](MACOS_TESTING.md) for complete macOS setup and testing guide.
```

---

## Alternative: Minimal Addition

If you prefer a more minimal change, just add this single bullet point:

```markdown
* **macOS support**: CPU-only builds are now supported. See [MACOS_TESTING.md](MACOS_TESTING.md) for details.
```

---

## Platform Support Matrix (Optional Enhancement)

Consider adding a platform support table after the installation section:

```markdown
## Platform Support

| Platform | GPU Acceleration | Status | Notes |
|----------|------------------|--------|-------|
| Linux | CUDA (NVIDIA) | ✅ Full Support | Recommended for production |
| Linux | ROCm (AMD) | ✅ Experimental | Supported GPUs listed in docs |
| Windows | CUDA (NVIDIA) | ✅ Full Support | |
| macOS | None (CPU only) | ✅ Supported | Development & testing |

**macOS Notes:**
- CPU-only builds using PyTorch native attention
- Full API compatibility
- Suitable for development and CPU inference
- See [MACOS_TESTING.md](MACOS_TESTING.md) for details
```

---

## Usage Section Addition (Optional)

Consider adding a macOS-specific note in the "Using xFormers" section:

```markdown
### Platform-Specific Notes

**macOS**: xFormers automatically uses CPU fallback operators on macOS. All features work, but GPU-specific optimizations (Flash Attention, CUTLASS) are not available. Performance is comparable to PyTorch's native attention on CPU.
```

---

## Reasons for These Changes

1. **Visibility**: Users searching for macOS support will find it clearly documented
2. **Expectations**: Sets clear expectations about CPU-only functionality
3. **Guidance**: Points to comprehensive macOS testing guide
4. **Completeness**: Documents all supported platforms in one place

---

## Implementation Priority

1. **High Priority**: Add at minimum the "macOS (CPU-only)" installation section
2. **Medium Priority**: Add platform support matrix for clarity
3. **Low Priority**: Add platform-specific notes in usage section

---

## Example PR Description for README Update

```markdown
# Add macOS Support Documentation to README

## Changes
- Added macOS installation instructions
- Noted CPU-only limitation
- Referenced MACOS_TESTING.md for details
- Added platform support table (optional)

## Context
macOS support was added in PR #XXX. This update ensures users can easily find
installation instructions and understand the limitations of CPU-only builds.

## Testing
- [x] README renders correctly on GitHub
- [x] Links work
- [x] Instructions tested on macOS
```
