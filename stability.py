# -*- coding: utf-8 -*-
"""单遍 vs 两遍归约的数值稳定性实验。

两遍：mean = Σs/N；var = Σ(s-mean)²/N        —— 与 Triton 版同语义
单遍：var = Σs²/N - mean²                    —— 少一次块规约往返，更快

单遍公式在 |mean| >> std 时发生灾难性相消：两个量级接近的大数相减，
有效位数被吃光，方差可以算成负数，rsqrtf(负数+eps) 产出 NaN。
本脚本给标准正态数据加偏置，扫描 offset 观察两者误差如何分岔。
真值取 float64 计算。
"""
import torch
import fused_ln_cuda as F

D, ROWS, EPS = 256, 4096, 1e-5


def main():
    p = torch.cuda.get_device_properties(0)
    print(f"GPU: {p.name} | sm_{p.major}{p.minor} | torch {torch.__version__}")
    print(f"数据: {ROWS}x{D} 标准正态 + offset，float32；真值 float64\n")
    torch.manual_seed(0)
    x0 = torch.randn(ROWS, D, device="cuda")
    r = torch.randn_like(x0)
    w = torch.randn(D, device="cuda")
    b = torch.randn(D, device="cuda")

    print(f"{'offset':>10}{'|mean|/std':>13}{'err_2pass':>14}{'err_1pass':>14}{'劣化':>10}")
    print("-" * 61)
    for off in [0.0, 1e1, 1e2, 1e3, 1e4, 1e5]:
        x = x0 + off
        s64 = x.double() + r.double()
        ref = torch.nn.functional.layer_norm(
            s64, (D,), w.double(), b.double(), EPS)
        ratio = (s64.mean(-1).abs() / s64.std(-1)).mean().item()

        errs = []
        for v in ("2pass", "1pass"):
            o, _ = F.fused_add_layernorm(x, r, w, b, EPS, variant=v)
            errs.append((o.double() - ref).abs().max().item())
        e2, e1 = errs
        deg = "NaN" if e1 != e1 else f"{e1 / max(e2, 1e-30):.0f}×"
        print(f"{off:>10.0e}{ratio:>13.1f}{e2:>14.3e}{e1:>14.3e}{deg:>10}")

    print("\n结论：offset=0（均值≈0）时两者等价；|mean|/std 增大后单遍公式迅速劣化，")
    print("     偏置达 1e4 时方差被算成负数，输出 NaN。两遍归约多一次 __syncthreads，")
    print("     换来的是全域数值稳定 —— 这也是 Triton 版选两遍的原因。")


if __name__ == "__main__":
    main()
