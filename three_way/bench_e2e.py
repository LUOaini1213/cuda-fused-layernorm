# -*- coding: utf-8 -*-
"""端到端：把融合算子接进真实 Transformer block，量前向的整体收益。

孤立算子的加速比回答不了「值不值得做」——LayerNorm 只占前向的一部分，
2× 的算子加速到了端到端可能只剩几个百分点。本脚本给出那个百分比。

pre-norm 结构每层用两次「残差加 + LayerNorm」：
    s = x + residual;  normed = LN(s);  residual = s
融合版把这两步并成一个 kernel，并把新的残差流一并返回，避免调用方重算。

三份实现共用同一套权重与同一个注意力实现（F.scaled_dot_product_attention），
差别只在 norm 这一处，因此测出的差值就是融合算子的端到端贡献。
"""
import gc, json, statistics as st, time
import torch
import torch.nn as nn
import torch.nn.functional as F

import fused_ln_cuda as CUDA
try:
    import fused_ln_triton as TRI
    HAVE_TRI = True
except Exception:
    HAVE_TRI = False


class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dtype):
        super().__init__()
        self.h = n_heads
        self.dk = d_model // n_heads
        self.ln1 = nn.LayerNorm(d_model, dtype=dtype)
        self.ln2 = nn.LayerNorm(d_model, dtype=dtype)
        self.qkv = nn.Linear(d_model, 3 * d_model, dtype=dtype)
        self.proj = nn.Linear(d_model, d_model, dtype=dtype)
        self.fc1 = nn.Linear(d_model, d_ff, dtype=dtype)
        self.fc2 = nn.Linear(d_ff, d_model, dtype=dtype)

    def _attn(self, x):
        B, S, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        sh = lambda t: t.view(B, S, self.h, self.dk).transpose(1, 2)
        o = F.scaled_dot_product_attention(sh(q), sh(k), sh(v))
        return self.proj(o.transpose(1, 2).reshape(B, S, D))

    def _mlp(self, x):
        return self.fc2(F.gelu(self.fc1(x)))

    def forward(self, x, residual, mode):
        """x = 上一子层输出，residual = running 残差流。返回 (新 x, 新 residual)。"""
        if mode == "eager":
            s = x + residual
            n1 = F.layer_norm(s, (s.shape[-1],), self.ln1.weight, self.ln1.bias, self.ln1.eps)
            residual = s
            a = self._attn(n1)
            s = a + residual
            n2 = F.layer_norm(s, (s.shape[-1],), self.ln2.weight, self.ln2.bias, self.ln2.eps)
            residual = s
            return self._mlp(n2), residual
        fn = TRI.fused_add_layernorm if mode == "triton" else CUDA.fused_add_layernorm
        n1, residual = fn(x, residual, self.ln1.weight, self.ln1.bias, self.ln1.eps)
        a = self._attn(n1)
        n2, residual = fn(a, residual, self.ln2.weight, self.ln2.bias, self.ln2.eps)
        return self._mlp(n2), residual


class Stack(nn.Module):
    def __init__(self, n_layers, d_model, n_heads, d_ff, dtype):
        super().__init__()
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_ff, dtype) for _ in range(n_layers)])

    def forward(self, x, mode):
        residual = torch.zeros_like(x)
        for b in self.blocks:
            x, residual = b(x, residual, mode)
        return x + residual


CONFIGS = [                     # (标签, B, S, d_model, heads, d_ff, layers)
    ("小 batch 短序列", 1,  128,  512,  8, 2048, 6),
    ("中等",            8,  256,  512,  8, 2048, 6),
    ("长序列",          2, 1024,  512,  8, 2048, 6),
    ("宽模型",          4,  256,  1024, 16, 4096, 6),
]


def timeit(fn, budget_ms=300.0):
    for _ in range(5): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(3): fn()
    torch.cuda.synchronize()
    probe = (time.perf_counter() - t0) / 3 * 1e3
    iters = max(10, min(500, int(budget_ms / max(probe, 1e-3))))
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return st.median(ts)


@torch.inference_mode()
def run(dtype, tag):
    modes = ["eager"] + (["triton"] if HAVE_TRI else []) + ["cuda"]
    print(f"\n{'='*100}\n### dtype = {tag}\n{'='*100}")
    hdr = f"{'配置':<16}{'B×S×D':>16}{'层':>4}" + "".join(f"{m:>10}" for m in modes)
    print(hdr + f"{'CUDA增益':>10}" + (f"{'TRI增益':>10}" if HAVE_TRI else "") + f"{'max|err|':>11}")
    out = []
    for label, B, S, D, H, FF, L in CONFIGS:
        torch.manual_seed(0)
        m = Stack(L, D, H, FF, dtype).cuda().eval()
        x = torch.randn(B, S, D, device="cuda", dtype=dtype)

        ref = m(x, "eager")
        errs = {}
        for mode in modes[1:]:
            o = m(x, mode)
            errs[mode] = (o.float() - ref.float()).abs().max().item()
            del o
        t = {mode: timeit(lambda mode=mode: m(x, mode)) for mode in modes}

        row = f"{label:<16}{f'{B}x{S}x{D}':>16}{L:>4}" + "".join(f"{t[mo]:>10.3f}" for mo in modes)
        row += f"{t['eager']/t['cuda']:>9.3f}x"
        if HAVE_TRI: row += f"{t['eager']/t['triton']:>9.3f}x"
        row += f"{errs['cuda']:>11.2e}"
        print(row)
        out.append(dict(case=label, B=B, S=S, D=D, layers=L, dtype=tag,
                        **{f"{mo}_ms": t[mo] for mo in modes},
                        cuda_gain=t['eager']/t['cuda'],
                        **({'triton_gain': t['eager']/t['triton']} if HAVE_TRI else {}),
                        **{f"err_{k}": v for k, v in errs.items()}))
        del m, x, ref; gc.collect(); torch.cuda.empty_cache()
    return out


if __name__ == "__main__":
    p = torch.cuda.get_device_properties(0)
    print(f"GPU: {p.name} | sm_{p.major}{p.minor} | torch {torch.__version__} | "
          f"Triton {'可用' if HAVE_TRI else '不可用'}")
    print("说明：三份实现共用同一套权重与同一注意力实现，差别只在 norm 这一处。")
    out = run(torch.float32, "fp32") + run(torch.float16, "fp16")
    json.dump(out, open("bench_e2e_results.json", "w"), indent=1)
    g = [r["cuda_gain"] for r in out]
    print(f"\nCUDA 融合算子的端到端增益：中位 {st.median(g):.3f}×  范围 {min(g):.3f}× – {max(g):.3f}×")
    print("对照：同一算子孤立测量时中位 1.428×。两者的差距就是「算子加速被整层摊薄」的幅度。")
