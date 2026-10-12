# -*- coding: utf-8 -*-
"""Fused residual-add + LayerNorm，CUDA C++ 实现（NVRTC 运行时编译）。

与 kernels/fused_layernorm.py 的 Triton 版同语义、同 API，用于逐项对照：
Triton 的 tl.sum 隐藏了行内归约，这里必须自己写 warp shuffle + shared 跨 warp 汇总。
"""
from __future__ import annotations
import os, functools, threading
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
MAX_PER_THREAD = 8
_CTYPE = {torch.float32: "float", torch.float16: "__half"}
_CUDA_DEVICE = None
_CUDA_CONTEXT_LOCK = threading.Lock()


@functools.lru_cache(maxsize=1)
def _cupy():
    # CPU/unsupported-shape fallback must not require a CUDA-only dependency.
    try:
        import cupy
    except ImportError as exc:
        raise RuntimeError("CUDA fused path requires a compatible CuPy installation") from exc
    return cupy


def _bind_cuda_device(device):
    """The existing module/function caches belong to one device per process."""
    global _CUDA_DEVICE
    with _CUDA_CONTEXT_LOCK:
        if _CUDA_DEVICE is not None and _CUDA_DEVICE != device:
            raise ValueError("cached CUDA functions belong to another device; use a fresh process")
        _CUDA_DEVICE = device


def _cuda_for_tensor(t):
    cp = _cupy()
    current = cp.cuda.runtime.getDevice()
    if t.device.index != current:
        raise ValueError("CuPy current device must match the tensor CUDA device")
    _bind_cuda_device(current)
    return cp


@functools.lru_cache(maxsize=1)
def _module():
    cp = _cupy()
    _bind_cuda_device(cp.cuda.runtime.getDevice())
    src = open(os.path.join(_HERE, "kernel.cu"), encoding="utf-8").read()
    names = [f"fused_add_ln_{v}<{t}>" for v in ("2pass", "1pass")
             for t in ("float", "__half")]
    return cp.RawModule(code=src, options=("--std=c++17",), name_expressions=names)


@functools.lru_cache(maxsize=8)
def _fn(variant: str, ctype: str):
    return _module().get_function(f"fused_add_ln_{variant}<{ctype}>")


def _as_cupy(t: torch.Tensor):
    """零拷贝把 torch CUDA 张量包成 cupy 数组（只借指针，不搬数据）。"""
    if not t.is_cuda or t.dtype not in _CTYPE or not t.is_contiguous():
        raise ValueError("raw pointer conversion requires a contiguous supported CUDA tensor")
    cp = _cuda_for_tensor(t)
    dtype = {torch.float32: cp.float32, torch.float16: cp.float16}[t.dtype]
    mem = cp.cuda.UnownedMemory(t.data_ptr(), t.numel() * t.element_size(), t)
    return cp.ndarray(tuple(t.shape), dtype=dtype,
                      memptr=cp.cuda.MemoryPointer(mem, 0))


def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def _threads_for(n: int) -> int:
    """每线程约 4 个元素；不足 32 补满一个 warp，上限 1024。"""
    return max(32, min(1024, _next_pow2(max(1, (n + 3) // 4))))


def _validate_cuda_inputs(x, residual, weight, bias):
    """Validate raw-pointer assumptions before allocations, reshaping or CuPy."""
    n = x.shape[-1]
    for name, value in (("weight", weight), ("bias", bias)):
        if value.device != x.device:
            raise ValueError(f"{name} must be on the same CUDA device as x")
        if tuple(value.shape) != (n,):
            raise ValueError(f"{name} must have shape ({n},)")
    if residual is not None:
        if not isinstance(residual, torch.Tensor):
            raise TypeError("CUDA residual must be a tensor")
        if residual.device != x.device:
            raise ValueError("residual must be on the same CUDA device as x")
        if residual.dtype != x.dtype:
            raise ValueError("CUDA residual dtype must match x")
        if tuple(residual.shape) != tuple(x.shape):
            raise ValueError("CUDA residual must match x's original shape exactly")


def fused_add_layernorm(x, residual, weight, bias, eps=1e-5, variant="2pass"):
    """LayerNorm(x + residual)，返回 (normed, x + residual)。

    residual=None 时退化为 LayerNorm(x)，返回 (normed, x)。
    不满足 CUDA 路径条件时回落到 PyTorch，调用方无需判断。
    """
    if not isinstance(x, torch.Tensor) or x.dim() < 1:
        raise ValueError("x must be a tensor with at least one dimension")
    for name, value in (("weight", weight), ("bias", bias)):
        if value is not None and not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a tensor or None")
    ok = (x.is_cuda and x.dtype in _CTYPE and 0 < x.shape[-1] <= 1024 and x.numel() > 0
          and weight is not None and bias is not None
          and weight.dtype == x.dtype and bias.dtype == x.dtype)
    if ok:
        n = x.shape[-1]
        ok = (n + _threads_for(n) - 1) // _threads_for(n) <= MAX_PER_THREAD
    if not ok:
        s = x if residual is None else x + residual
        return torch.nn.functional.layer_norm(
            s, (s.shape[-1],), weight, bias, eps), s

    _validate_cuda_inputs(x, residual, weight, bias)
    # Do not silently move contexts or reuse a cached function on another GPU.
    _cuda_for_tensor(x)
    xc = x.contiguous()
    n = xc.shape[-1]
    flat = xc.view(-1, n)
    rows = flat.shape[0]
    out = torch.empty_like(flat)

    has_res = residual is not None
    if has_res:
        rc = residual.contiguous().view(-1, n)
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
