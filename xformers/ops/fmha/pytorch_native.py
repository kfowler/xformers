# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""
CPU fallback operator using PyTorch's native scaled_dot_product_attention.
This provides basic attention functionality on CPU for macOS and other platforms.
"""

from typing import Any, List, Mapping, Optional, Set, Tuple

import torch
import torch.nn.functional as F

from .attn_bias import AttentionBias, LowerTriangularMask
from .common import AttentionBwOpBase, AttentionFwOpBase, Context, Inputs


class FwOp(AttentionFwOpBase):
    """CPU fallback using PyTorch's native scaled_dot_product_attention"""

    OPERATOR = None
    SUPPORTED_DEVICES: Set[str] = {"cpu"}
    SUPPORTED_DTYPES: Set[torch.dtype] = {torch.float32, torch.float16, torch.bfloat16}
    SUPPORTED_MAX_K = 2**16
    SUPPORTED_ATTN_BIAS_TYPES: Set[Any] = {type(None), LowerTriangularMask}
    SUPPORTS_DROPOUT = True
    SUPPORTS_CUSTOM_SCALE = True
    NAME = "pytorch_native_cpu"

    @classmethod
    def not_supported_reasons(cls, d: Inputs) -> List[str]:
        reasons = super().not_supported_reasons(d)

        # Check if we're on CPU
        device_type = d.query.device.type
        if device_type not in cls.SUPPORTED_DEVICES:
            reasons.append(
                f"device={device_type} (supported: {cls.SUPPORTED_DEVICES})"
            )

        # Check attention bias type
        if type(d.attn_bias) not in cls.SUPPORTED_ATTN_BIAS_TYPES:
            reasons.append(
                f"attn_bias type {type(d.attn_bias)} not supported "
                f"(supported: {cls.SUPPORTED_ATTN_BIAS_TYPES})"
            )

        return reasons

    @classmethod
    def apply(
        cls, inp: Inputs, needs_gradient: bool
    ) -> Tuple[torch.Tensor, Optional[Context]]:
        """Apply attention using PyTorch's native implementation"""

        query = inp.query
        key = inp.key
        value = inp.value

        # Handle different input shapes (BMHK or BHM format expected by SDPA)
        # xformers typically uses BMHK format
        if query.ndim == 4:
            # BMHK -> BHMD (batch, heads, seq, dim)
            # Assuming query is (B, M, H, K)
            B, M, H, K = query.shape
            query = query.transpose(1, 2)  # (B, H, M, K)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
        elif query.ndim == 3:
            # BMK format - add head dimension
            query = query.unsqueeze(1)
            key = key.unsqueeze(1)
            value = value.unsqueeze(1)

        # Handle attention mask
        attn_mask = None
        is_causal = False
        if isinstance(inp.attn_bias, LowerTriangularMask):
            is_causal = True
        elif inp.attn_bias is not None:
            # For other bias types, we'd need to convert them
            # For now, we just pass None and let PyTorch handle it
            pass

        # Apply scaled dot product attention
        dropout_p = inp.p if inp.p is not None else 0.0
        scale = inp.scale

        out = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p if needs_gradient else 0.0,
            is_causal=is_causal,
            scale=scale,
        )

        # Restore original shape
        if inp.query.ndim == 4:
            # BHMD -> BMHD
            out = out.transpose(1, 2)
        elif inp.query.ndim == 3:
            out = out.squeeze(1)

        # Context not needed for CPU fallback
        ctx = None
        if needs_gradient:
            # Create a minimal context for backward pass
            # PyTorch's autograd will handle the actual backward
            ctx = Context(
                out=out,
                lse=None,  # Log-sum-exp not computed for CPU fallback
            )

        return out, ctx


class BwOp(AttentionBwOpBase):
    """Backward pass for CPU - handled by PyTorch autograd"""

    OPERATOR = None
    SUPPORTED_DEVICES = FwOp.SUPPORTED_DEVICES
    SUPPORTED_DTYPES = FwOp.SUPPORTED_DTYPES
    SUPPORTED_MAX_K = FwOp.SUPPORTED_MAX_K
    SUPPORTED_ATTN_BIAS_TYPES = FwOp.SUPPORTED_ATTN_BIAS_TYPES
    SUPPORTS_DROPOUT = FwOp.SUPPORTS_DROPOUT
    SUPPORTS_CUSTOM_SCALE = FwOp.SUPPORTS_CUSTOM_SCALE
    NAME = "pytorch_native_cpu_bw"

    @classmethod
    def not_supported_reasons(cls, d: Inputs) -> List[str]:
        return FwOp.not_supported_reasons(d)

    @classmethod
    def apply(cls, ctx: Context, inp: Inputs, grad: torch.Tensor) -> "Gradients":
        # This should not be called directly as PyTorch's autograd handles it
        # But we define it for compatibility
        raise NotImplementedError(
            "CPU backward is handled by PyTorch autograd, "
            "this should not be called directly"
        )
