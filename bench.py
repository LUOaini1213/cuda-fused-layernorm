# -*- coding: utf-8 -*-
"""fused add+LayerNorm：PyTorch eager vs CUDA C++（两遍/单遍归约）。

形状取自 Track 3 仓库 results/kaggle_t4_triton_bench.log 的孤立 LayerNorm 微基准，
与该日志的 Triton 结果同形状对照（T4 与 GTX 1650 同为 sm_75 Turing）。

精度与计时分两个阶段跑，阶段间释放全部张量：4GB 卡上若让三份实现的输出同时驻留，
大形状会因显存压力把计时打歪（实测 shape 6 会从 28.6ms 劣化到 137ms）。
"""
import gc, json, statistics as st, time
import torch
import fused_ln_cuda as F

CASES = [                       # (标签, rows, D, T4 上 Triton 的 vs_eager)
    ("shape 1/5/9-11", 8192,    128,  1.486),
    ("shape 2",        128,     128,  0.580),
    ("shape 6",        1280000, 128,  1.920),
    ("shape 7",        8192,    32,   1.210),
    ("shape 8",        8192,    1024, 1.107),
    ("shape 13",       65536,   128,  1.797),
]
IMPLS = ["eager", "2pass", "1pass"]


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
    return F.fused_add_layernorm(x, r, w, b, variant=impl)


def free():
    gc.collect(); torch.cuda.empty_cache()


def accuracy(rows, D, dtype):
    """以 float64 为真值，取前 4096 行评估三份实现的最大绝对误差。"""
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
    """单独计时：只让该实现所需的张量驻留，避免大形状下的显存压力干扰。

    迭代次数按单次耗时自适应，使每组测量的总时长都凑到 budget_ms 量级：
    固定 50 次时，0.08ms 的小形状总共才测 4ms，结果被噪声主导（中位加速比
    在两次运行间可以从 1.48 漂到 1.34）。
    """
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
    print(f"\n{'='*94}\n### dtype = {tag}\n{'='*94}")
    print(f"{'case':<16}{'rows':>9}{'D':>6}{'eager_ms':>10}{'cuda2_ms':>10}{'cuda1_ms':>10}"
          f"{'vs_eager':>10}{'GiB/s':>9}{'err_2p':>11}{'T4/Triton':>11}")
    out = []
    for label, rows, D, tri in CASES:
        try:
            errs = accuracy(rows, D, dtype)
            t = {i: time_one(i, rows, D, dtype) for i in IMPLS}
            # 融合路径的显存流量：读 x、读 r、写 sum、写 out
            gbs = rows * D * torch.tensor([], dtype=dtype).element_size() * 4
            gbs = gbs / t["2pass"] * 1e3 / 2**30
            print(f"{label:<16}{rows:>9}{D:>6}{t['eager']:>10.4f}{t['2pass']:>10.4f}"
                  f"{t['1pass']:>10.4f}{t['eager']/t['2pass']:>10.3f}{gbs:>9.1f}"
                  f"{errs['2pass']:>11.2e}{tri:>11.3f}")
            out.append(dict(case=label, rows=rows, D=D, dtype=tag,
                            **{f"{i}_ms": t[i] for i in IMPLS},
                            vs_eager=t["eager"] / t["2pass"], gibs=gbs,
                            **{f"err_{i}": errs[i] for i in IMPLS},
                            triton_t4_vs_eager=tri))
        except torch.cuda.OutOfMemoryError:
            print(f"{label:<16}{rows:>9}{D:>6}   OOM (4GB)")
            free()
    return out


if __name__ == "__main__":
    p = torch.cuda.get_device_properties(0)
    print(f"GPU: {p.name} | {p.total_memory/2**30:.1f} GiB | sm_{p.major}{p.minor} "
          f"| torch {torch.__version__} | {p.multi_processor_count} SM")
    out = run(torch.float32, "fp32") + run(torch.float16, "fp16")
    json.dump(out, open("bench_results.json", "w"), indent=1)
    sp = [r["vs_eager"] for r in out]
    print(f"\n中位加速比（对 eager，{len(sp)} 组）: {st.median(sp):.3f}×")
    print(f"同形状 Triton(T4) 中位: {st.median([r['triton_t4_vs_eager'] for r in out]):.3f}×")
