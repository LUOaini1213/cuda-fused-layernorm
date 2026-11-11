# -*- coding: utf-8 -*-
"""Accuracy gate shared by bench.py and three_way/bench3.py.

Until 2026-11-03 the benchmarks only *recorded* max|err| against a float64
reference; nothing failed when a fused kernel was far worse than eager. That is
how the eps argument bug (see fused_ln_cuda._scalar_args) went unnoticed: the
committed fp32 errors are about 20x eager's. The rules below turn the record
into a gate:

1. every max|err| is finite (no NaN/inf);
2. a fused implementation's max|err| <= max(RATIO * eager max|err|, FLOOR[dtype]);
3. zero-variance rows (constant x + residual) return the bias, with no NaN.

The functions are plain PyTorch so the gate itself is unit-tested on CPU
(tests/test_accuracy_gate.py). Running it on real kernels needs a GPU.
"""
from __future__ import annotations

import math

import torch

RATIO = 5.0
FLOOR = {"fp32": 1e-5, "fp16": 1e-3}
CONSTANT_TOL = {torch.float32: 1e-6, torch.float16: 1e-3}


def error_failures(errs: dict, dtype_tag: str, reference: str = "eager") -> list[str]:
    """Rules 1-2 for one shape. ``errs`` maps implementation name -> max|err|."""
    ref = errs[reference]
    if not math.isfinite(ref):
        return [f"{reference}: max|err| is {ref}"]
    bound = max(RATIO * ref, FLOOR[dtype_tag])
    failures = []
    for name, value in errs.items():
        if name == reference:
            continue
        if not math.isfinite(value):
            failures.append(f"{name}: max|err| is {value}")
        elif value > bound:
            failures.append(f"{name}: max|err| {value:.3e} > bound {bound:.3e} "
                            f"(= max({RATIO:g} x {reference} {ref:.3e}, {FLOOR[dtype_tag]:g}))")
    return failures


def constant_row_failures(fn, D: int, dtype, device="cuda", name: str = "impl") -> list[str]:
    """Rule 3. ``fn(x, residual, weight, bias)`` returns ``(normed, x + residual)``.

    0.5 + 0.25 and its row sums are exact in fp16/fp32 for power-of-two D, so the
    true variance is exactly 0 and LayerNorm must return the bias. With eps lost
    (the pre-2026-11-03 bug) rsqrtf(0 + eps_garbage) is NaN.
    """
    rows = 4
    x = torch.full((rows, D), 0.5, dtype=dtype, device=device)
    r = torch.full((rows, D), 0.25, dtype=dtype, device=device)
    g = torch.Generator(device="cpu").manual_seed(1)
    w = torch.randn(D, generator=g).to(device=device, dtype=dtype)
    b = torch.randn(D, generator=g).to(device=device, dtype=dtype)
    out, _ = fn(x, r, w, b)
    out = out.float()
    expected = b.float().expand_as(out)
    if torch.isnan(out).any() or torch.isinf(out).any():
        return [f"{name}: constant row (D={D}) produced NaN/inf; expected the bias"]
    worst = (out - expected).abs().max().item()
    if worst > CONSTANT_TOL[dtype]:
        return [f"{name}: constant row (D={D}) differs from the bias by {worst:.3e}"]
    return []


def report(failures: list[str]) -> int:
    """Print PASS/FAIL and return a process exit code."""
    if failures:
        print(f"\nACCURACY GATE: FAIL ({len(failures)})")
        for line in failures:
            print("  - " + line)
        return 1
    print("\nACCURACY GATE: PASS")
    return 0
