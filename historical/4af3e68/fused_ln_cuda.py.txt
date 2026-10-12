# -*- coding: utf-8 -*-
"""Fused residual-add + LayerNorm，CUDA C++ 实现（NVRTC 运行时编译）。

与 kernels/fused_layernorm.py 的 Triton 版同语义、同 API，用于逐项对照：
Triton 的 tl.sum 隐藏了行内归约，这里必须自己写 warp shuffle + shared 跨 warp 汇总。
"""
from __future__ import annotations
import os, functools
import torch, cupy as cp

_HERE = os.path.dirname(os.path.abspath(__file__))
MAX_PER_THREAD = 8
_CTYPE = {torch.float32: "float", torch.float16: "__half"}
_CPDT  = {torch.float32: cp.float32, torch.float16: cp.float16}


@functools.lru_cache(maxsize=1)
def _module():
    src = open(os.path.join(_HERE, "kernel.cu"), encoding="utf-8").read()
    names = [f"fused_add_ln_{v}<{t}>" for v in ("2pass", "1pass")
             for t in ("float", "__half")]
    return cp.RawModule(code=src, options=("--std=c++17",), name_expressions=names)


@functools.lru_cache(maxsize=8)
def _fn(variant: str, ctype: str):
    return _module().get_function(f"fused_add_ln_{variant}<{ctype}>")


def _as_cupy(t: torch.Tensor) -> cp.ndarray:
    """零拷贝把 torch CUDA 张量包成 cupy 数组（只借指针，不搬数据）。"""
    mem = cp.cuda.UnownedMemory(t.data_ptr(), t.numel() * t.element_size(), t)
    return cp.ndarray(tuple(t.shape), dtype=_CPDT[t.dtype],
                      memptr=cp.cuda.MemoryPointer(mem, 0))


def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def _threads_for(n: int) -> int:
    """每线程约 4 个元素；不足 32 补满一个 warp，上限 1024。"""
    return max(32, min(1024, _next_pow2(max(1, (n + 3) // 4))))


def fused_add_layernorm(x, residual, weight, bias, eps=1e-5, variant="2pass"):
    """LayerNorm(x + residual)，返回 (normed, x + residual)。

    residual=None 时退化为 LayerNorm(x)，返回 (normed, x)。
    不满足 CUDA 路径条件时回落到 PyTorch，调用方无需判断。
    """
    ok = (x.is_cuda and x.dtype in _CTYPE and x.shape[-1] <= 1024
          and weight.dtype == x.dtype and bias.dtype == x.dtype)
    if ok:
        n = x.shape[-1]
        ok = (n + _threads_for(n) - 1) // _threads_for(n) <= MAX_PER_THREAD
    if not ok:
        s = x if residual is None else x + residual
        return torch.nn.functional.layer_norm(
            s, (s.shape[-1],), weight, bias, eps), s

    xc = x.contiguous()
    n = xc.shape[-1]
    flat = xc.view(-1, n)
    rows = flat.shape[0]
    out = torch.empty_like(flat)

    has_res = residual is not None
    if has_res:
        rc = residual.contiguous().view(-1, n)
        assert rc.shape == flat.shape, "residual must match x"
        total = torch.empty_like(flat)
    else:
        rc, total = flat, flat          # kernel 不会读写，占位保持签名一致

    _fn(variant, _CTYPE[x.dtype])(
        (rows,), (_threads_for(n),),
        (_as_cupy(flat), _as_cupy(rc), _as_cupy(out), _as_cupy(total),
         _as_cupy(weight.contiguous()), _as_cupy(bias.contiguous()),
         flat.stride(0), n, float(eps), int(has_res)),
    )
    shape = x.shape
    return out.view(shape), (total.view(shape) if has_res else x)
