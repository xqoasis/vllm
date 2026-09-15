# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fuse the FP32 arithmetic in the DeepSeek vision tower's norms and RoPE."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _rms_norm_kernel(
    X, W, VAR, Y, stride_x, D: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * stride_x + col, col < D, 0).to(tl.float32)
    weight = tl.load(W + col, col < D, 0).to(tl.float32)
    variance = tl.load(VAR + row)
    normalized = x * tl.rsqrt(variance + EPS)
    tl.store(Y + row * D + col, weight * normalized, col < D)


@triton.jit
def _rotary_kernel(
    X,
    COS,
    SIN,
    Y,
    N,
    stride_x_token,
    stride_x_head,
    stride_cos,
    stride_sin,
    H: tl.constexpr,
    HALF: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = index < N * H * HALF
    dim = index % HALF
    head = index // HALF % H
    token = index // (HALF * H)
    offset = token * stride_x_token + head * stride_x_head + dim
    x1 = tl.load(X + offset, mask, 0).to(tl.float32)
    x2 = tl.load(X + offset + HALF, mask, 0).to(tl.float32)
    cos = tl.load(COS + token * stride_cos + dim, mask, 0).to(tl.float32)
    sin = tl.load(SIN + token * stride_sin + dim, mask, 0).to(tl.float32)
    out = token * H * HALF * 2 + head * HALF * 2 + dim
    tl.store(Y + out, x1 * cos - x2 * sin, mask)
    tl.store(Y + out + HALF, x2 * cos + x1 * sin, mask)


def vision_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    output = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    if x.numel():
        # Preserve PyTorch's reduction order: tiny variance changes can cross BF16
        # rounding boundaries and accumulate across the 32-layer vision tower.
        variance = x.float().square().mean(-1, keepdim=True)
        _rms_norm_kernel[(x.shape[0],)](
            x,
            weight,
            variance,
            output,
            x.stride(0),
            x.shape[1],
            eps,
            triton.next_power_of_2(x.shape[1]),
            enable_fp_fusion=False,
        )
    return output


def vision_rotary(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    n, heads, dim = x.shape
    output = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    if n:
        # Q/K are views into packed QKV; their token stride is not heads * dim.
        _rotary_kernel[(triton.cdiv(n * heads * (dim // 2), 256),)](
            x,
            cos,
            sin,
            output,
            n,
            x.stride(0),
            x.stride(1),
            cos.stride(0),
            sin.stride(0),
            heads,
            dim // 2,
            256,
            enable_fp_fusion=False,
        )
    return output
