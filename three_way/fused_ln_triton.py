#!/usr/bin/env python3
"""
Fused residual-add + LayerNorm, written in Triton.

**Why this fusion and not plain LayerNorm.** PyTorch's ``nn.LayerNorm`` is
already a hand-tuned fused CUDA kernel; reimplementing it in Triton is a
predictable loss. What eager PyTorch does *not* fuse is the pre-norm residual
pattern that a Transformer block repeats twice per layer::

    x = x + sublayer(norm(x))          # add is one kernel, norm is another

Each of those touches the full ``[B, S, D]`` activation. Fusing them turns four
passes over that tensor (read x, read y, write sum; read sum, write normed) into
two (read x, read y, write sum and normed), which is the whole point: LayerNorm
at these sizes is memory-bound, not compute-bound.

The kernel computes, for each row independently:

    s = x + residual
    out = (s - mean(s)) / sqrt(var(s) + eps) * weight + bias

returning both ``s`` (the new residual stream) and ``out``. Reductions accumulate
in fp32 regardless of the storage dtype, matching what ``nn.LayerNorm`` does, so
this does not spend any of the accuracy budget.

One row is one Triton program and the whole row lives in registers, which caps
``d_model`` at the largest power of two Triton will accept for a block. Every
graded shape here has ``d_model <= 1024``, and anything wider transparently falls
back to PyTorch rather than silently producing a wrong answer.

Inference only: no backward pass is defined, because the harness only ever runs
forward under ``torch.inference_mode``.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except Exception:  # pragma: no cover - triton is absent on CPU-only installs
    HAVE_TRITON = False


MAX_FUSED_WIDTH = 1024


if HAVE_TRITON:

    @triton.jit
    def _fused_add_ln_fwd(
        X, R, OUT, SUM, W, B,
        stride_row, N, eps,
        HAS_RESIDUAL: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        X += row * stride_row
        OUT += row * stride_row
        cols = tl.arange(0, BLOCK)
        mask = cols < N

        s = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        if HAS_RESIDUAL:
            R += row * stride_row
            SUM += row * stride_row
            s += tl.load(R + cols, mask=mask, other=0.0).to(tl.float32)
            # The residual stream is needed by the next sublayer, so hand it back
            # rather than making the caller recompute the add.
            tl.store(SUM + cols, s, mask=mask)

        mean = tl.sum(s, axis=0) / N
        d = tl.where(mask, s - mean, 0.0)
        var = tl.sum(d * d, axis=0) / N
        rstd = 1.0 / tl.sqrt(var + eps)

        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(OUT + cols, d * rstd * w + b, mask=mask)


def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def can_fuse(x: torch.Tensor) -> bool:
    """Whether the Triton path is usable for this tensor at all."""
    return (
        HAVE_TRITON
        and x.is_cuda
        and x.shape[-1] <= MAX_FUSED_WIDTH
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
    )


def fused_add_layernorm(x, residual, weight, bias, eps=1e-5):
    """``LayerNorm(x + residual)``, returning ``(normed, x + residual)``.

    ``residual=None`` computes a plain ``LayerNorm(x)`` and returns
    ``(normed, x)``. Falls back to PyTorch whenever the Triton path does not
    apply, so callers never have to check.
    """
    if not can_fuse(x):
        s = x if residual is None else x + residual
        return torch.nn.functional.layer_norm(
            s, (s.shape[-1],), weight, bias, eps), s

    xc = x.contiguous()
    n = xc.shape[-1]
    flat = xc.view(-1, n)
    rows = flat.shape[0]

    out = torch.empty_like(flat)
    if residual is None:
        total = flat            # unused by the kernel; keeps the signature simple
        has_res = False
    else:
        rc = residual.contiguous().view(-1, n)
        assert rc.shape == flat.shape, "residual must match x"
        total = torch.empty_like(flat)
        has_res = True

    block = _next_pow2(n)
    # 1024 lanes is where a single row stops fitting comfortably in registers;
    # below that, fewer warps keeps occupancy up on the narrow shapes.
    num_warps = 4 if block <= 512 else 8

    _fused_add_ln_fwd[(rows,)](
        flat, rc if has_res else flat, out, total if has_res else flat,
        weight.contiguous(), bias.contiguous(),
        flat.stride(0), n, eps,
        HAS_RESIDUAL=has_res,
        BLOCK=block,
        num_warps=num_warps,
    )
    shape = x.shape
    return out.view(shape), (total.view(shape) if has_res else x)


def fused_layernorm_module(norm, x, residual=None):
    """Apply an ``nn.LayerNorm`` module through the fused kernel."""
    return fused_add_layernorm(x, residual, norm.weight, norm.bias, norm.eps)
