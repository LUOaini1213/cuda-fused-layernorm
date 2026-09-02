# -*- coding: utf-8 -*-
"""受控三方对照：PyTorch eager / Triton / CUDA C++，同一张 GPU、同一组形状、同一精度门。

此前 CUDA 版只能与 Kaggle T4 上记录的 Triton 结果做跨卡参照（同为 sm_75 但芯片不同），
只能得出「量级相当」。本脚本三份实现跑在同一张卡上，是真正的受控对照。

精度以 float64 为真值。计时按单次耗时自适应迭代次数，使每组测量总时长约 200 ms，
避免小形状被噪声主导。精度与计时分阶段，阶段间释放张量——4 GiB 卡上若让三份实现的
输出同时驻留，大形状会因显存压力把计时打歪。
"""
import gc, json, statistics as st, time
import torch

import fused_ln_cuda as CUDA
import fused_ln_triton as TRI

CASES = [                       # (标签, rows, D)
    ("shape 1/5/9-11", 8192,    128),
    ("shape 2",        128,     128),
    ("shape 6",        1280000, 128),
    ("shape 7",        8192,    32),
    ("shape 8",        8192,    1024),
    ("shape 13",       65536,   128),
]
IMPLS = ["eager", "triton", "cuda2", "cuda1"]


def make(rows, D, dtype):
    torch.manual_seed(0)
    return (torch.randn(rows, D, device="cuda", dtype=dtype),
            torch.randn(rows, D, device="cuda", dtype=dtype),
            torch.randn(D, device="cuda", dtype=dtype),
            torch.randn(D, device="cuda", dtype=dtype))


def call(impl, x, r, w, b):
    if impl == "eager":
        s = x + r
        return torch.nn.functional.layer_norm(s, (s.shape[-1],), w, b, 1e-5), s
    if impl == "triton":
        return TRI.fused_add_layernorm(x, r, w, b, 1e-5)
    return CUDA.fused_add_layernorm(x, r, w, b, 1e-5,
                                    variant="2pass" if impl == "cuda2" else "1pass")


def free():
    gc.collect(); torch.cuda.empty_cache()


def accuracy(rows, D, dtype):
    x, r, w, b = make(rows, D, dtype)
    k = min(rows, 4096)
    ref = torch.nn.functional.layer_norm(
        x[:k].double() + r[:k].double(), (D,), w.double(), b.double(), 1e-5)
    errs = {}
    for impl in IMPLS:
        o, _ = call(impl, x, r, w, b)
        errs[impl] = (o[:k].double() - ref).abs().max().item()
        del o; free()
    del x, r, w, b, ref; free()
    return errs


def time_one(impl, rows, D, dtype, budget_ms=200.0):
    x, r, w, b = make(rows, D, dtype)
    fn = lambda: call(impl, x, r, w, b)
    for _ in range(10): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(5): fn()
    torch.cuda.synchronize()
    probe = (time.perf_counter() - t0) / 5 * 1e3
    iters = max(20, min(2000, int(budget_ms / max(probe, 1e-3))))
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    del x, r, w, b, fn; free()
    return st.median(ts)


def run(dtype, tag):
    print(f"\n{'='*104}\n### dtype = {tag}\n{'='*104}")
    print(f"{'case':<16}{'rows':>9}{'D':>6}{'eager':>9}{'triton':>9}{'cuda2':>9}{'cuda1':>9}"
          f"{'TRI/eager':>11}{'CUDA/eager':>12}{'CUDA/TRI':>10}{'err_cuda2':>11}")
    out = []
    for label, rows, D in CASES:
        try:
            errs = accuracy(rows, D, dtype)
            t = {i: time_one(i, rows, D, dtype) for i in IMPLS}
            print(f"{label:<16}{rows:>9}{D:>6}"
                  f"{t['eager']:>9.4f}{t['triton']:>9.4f}{t['cuda2']:>9.4f}{t['cuda1']:>9.4f}"
                  f"{t['eager']/t['triton']:>11.3f}{t['eager']/t['cuda2']:>12.3f}"
                  f"{t['triton']/t['cuda2']:>10.3f}{errs['cuda2']:>11.2e}")
            out.append(dict(case=label, rows=rows, D=D, dtype=tag,
                            **{f"{i}_ms": t[i] for i in IMPLS},
                            tri_vs_eager=t['eager']/t['triton'],
                            cuda_vs_eager=t['eager']/t['cuda2'],
                            cuda_vs_triton=t['triton']/t['cuda2'],
                            **{f"err_{i}": errs[i] for i in IMPLS}))
        except torch.cuda.OutOfMemoryError:
            print(f"{label:<16}{rows:>9}{D:>6}   OOM (4GB)"); free()
    return out


if __name__ == "__main__":
    p = torch.cuda.get_device_properties(0)
    import triton
    print(f"GPU: {p.name} | {p.total_memory/2**30:.1f} GiB | sm_{p.major}{p.minor} | "
          f"{p.multi_processor_count} SM")
    print(f"torch {torch.__version__} | triton {triton.__version__} | 同卡受控对照")
    out = run(torch.float32, "fp32") + run(torch.float16, "fp16")
    json.dump(out, open("bench3_results.json", "w"), indent=1)
    for k, name in [("tri_vs_eager", "Triton  对 eager"),
                    ("cuda_vs_eager", "CUDA    对 eager"),
                    ("cuda_vs_triton", "CUDA    对 Triton")]:
        v = [r[k] for r in out]
        print(f"\n{name}：中位 {st.median(v):.3f}×   范围 {min(v):.3f}× – {max(v):.3f}×")
