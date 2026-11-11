# -*- coding: utf-8 -*-
"""Recompute every README number from the committed result files.

    python check_readme_numbers.py --check

Sources: bench_results.json (GTX 1650 tables), results_t4/t4_three_way_and_e2e.txt
(T4 three-way table and end-to-end gains) and stability.txt. Each README snippet
below is formatted from those files; a snippet missing from README.md, or a
summary line that disagrees with its per-row data, is a failure. Numbers from
uncommitted earlier runs are listed separately and must stay labelled as such.
No GPU or third-party package is needed.
"""
from __future__ import annotations

import json
import re
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TOL = 0.001          # one unit in the last printed digit of a 3-decimal ratio


def _rows(n: int) -> str:
    return f"{n:,}".replace(",", " ")


def _gtx1650(readme: str, errors: list[str]) -> None:
    rows = json.loads((ROOT / "bench_results.json").read_text(encoding="utf-8"))
    if len(rows) != 12:
        errors.append(f"bench_results.json: expected 12 rows, found {len(rows)}")

    def bold(v: float) -> str:
        s = f"{v:.3f}×"
        return f"**{s}**" if v > 1.1 else s

    for r in rows:
        if r["dtype"] == "fp32":
            line = (f"| {r['case']} | {_rows(r['rows'])} | {_rows(r['D'])} | {r['eager_ms']:.4f} | "
                    f"{r['2pass_ms']:.4f} | {bold(r['vs_eager'])} | {r['gibs']:.1f} GiB/s | "
                    f"{r['err_2pass']:.2e} |")
        else:
            line = (f"| {r['case']} | {r['eager_ms']:.4f} | {r['2pass_ms']:.4f} | "
                    f"{bold(r['vs_eager'])} | {r['gibs']:.1f} GiB/s |")
        if line not in readme:
            errors.append(f"README row missing or changed ({r['dtype']} {r['case']}): {line}")

    by = {(r["dtype"], r["case"]): r for r in rows}
    fp32_eager = [r["err_eager"] for r in rows if r["dtype"] == "fp32"]
    s2, s6, s8, s13 = (by[("fp32", c)] for c in ("shape 2", "shape 6", "shape 8", "shape 13"))
    snippets = [
        f"**12 组中位加速比 {st.median(r['vs_eager'] for r in rows):.3f}×。**",
        f"对 eager 的中位为 {st.median(r['triton_t4_vs_eager'] for r in rows):.3f}×",
        f"eager 的 fp32 误差为 {min(fp32_eager):.2e} – {max(fp32_eager):.2e}",
        f"（{s2['vs_eager']:.2f}× / {by[('fp16', 'shape 2')]['vs_eager']:.2f}×）",
        f"有效带宽仅 {s2['gibs']:.1f} GiB/s",
        f"已经跑到 {s8['gibs']:.1f} GiB/s",
        f"几乎没有头寸（{s8['vs_eager']:.3f}×）",
        f"拿到 {s6['vs_eager']:.2f}× / {s13['vs_eager']:.2f}×，有效带宽 {s6['gibs']:.1f} / {s13['gibs']:.1f} GiB/s",
        f"已提交的第三次运行是 {s6['2pass_ms']:.2f} ms / {s6['vs_eager']:.3f}×",
        f"三次的中位加速比为 1.481× / 1.342× / {st.median(r['vs_eager'] for r in rows):.3f}×",
    ]
    third = {c: by[("fp32", c)]["vs_eager"] for c in ("shape 6", "shape 13", "shape 8", "shape 7", "shape 2")}
    for case, v in third.items():
        snippets.append(re.compile(rf"\| {case} \| [0-9.]+ / [0-9.]+ / {v:.3f} \|"))
    for s in snippets:
        ok = s.search(readme) if isinstance(s, re.Pattern) else s in readme
        if not ok:
            errors.append(f"README snippet missing or changed: {getattr(s, 'pattern', s)}")
    fp32 = [r for r in rows if r["dtype"] == "fp32"]
    errs = [r["err_2pass"] for r in fp32]
    ratio = st.median(r["err_2pass"] / r["err_eager"] for r in fp32)
    if not (2e-5 <= min(errs) and max(errs) <= 4e-5 and 15 <= ratio <= 25):
        errors.append(f"fp32 err_2pass {min(errs):.2e}-{max(errs):.2e}, median {ratio:.1f}x eager: "
                      "no longer '~2-4e-5, about 20x eager'")
    for s in ("fp32 errors ~2–4e−5, about 20× eager", "fp32 误差约为 eager 的 20 倍"):
        if s not in readme:
            errors.append(f"README snippet missing or changed: {s}")
    if "前两次没有提交日志" not in readme or "未提交日志的早期运行" not in readme:
        errors.append("README must label the uncommitted earlier-run numbers as such")


def _t4(readme: str, errors: list[str]) -> None:
    text = (ROOT / "results_t4/t4_three_way_and_e2e.txt").read_text(encoding="utf-8")
    iso = re.findall(r"^shape [^\n]*?\s+\d+\s+\d+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)"
                     r"\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.e+-]+)$", text, re.M)
    if len(iso) != 12:
        errors.append(f"T4 isolated table: expected 12 rows, found {len(iso)}")
        return
    cols = {"tri": 4, "cuda": 5, "cuda_tri": 6}
    for row in iso:
        eager, tri, cuda2 = (float(row[i]) for i in range(3))
        for key, expect in (("tri", eager / tri), ("cuda", eager / cuda2), ("cuda_tri", tri / cuda2)):
            if abs(float(row[cols[key]]) - expect) > 0.003:
                errors.append(f"T4 row ratio {key} {row[cols[key]]} != ms ratio {expect:.4f}")
    summary = {}
    for name, key in (("Triton  对 eager", "tri"), ("CUDA    对 eager", "cuda"), ("CUDA    对 Triton", "cuda_tri")):
        m = re.search(re.escape(name) + r"：中位 ([\d.]+)×\s+范围 ([\d.]+)× – ([\d.]+)×", text)
        if not m:
            errors.append(f"T4 summary line missing: {name}")
            return
        med, lo, hi = m.groups()
        vals = [float(r[cols[key]]) for r in iso]
        for printed, recomputed in ((med, st.median(vals)), (lo, min(vals)), (hi, max(vals))):
            if abs(float(printed) - recomputed) > TOL:
                errors.append(f"T4 {name}: printed {printed} vs recomputed {recomputed:.4f}")
        summary[key] = (med, lo, hi)
    (tm, tl, th), (cm, cl, ch), (xm, xl, xh) = summary["tri"], summary["cuda"], summary["cuda_tri"]
    wins = sum(float(r[cols["cuda_tri"]]) > 1.0 for r in iso)
    snippets = [
        f"| Triton 对 eager | **{tm}×** | {tl}× – {th}× |",
        f"| CUDA C++ 对 eager | {cm}× | {cl}× – {ch}× |",
        f"| **CUDA C++ 对 Triton** | **{xm}×** | {xl}× – {xh}× |",
        f"CUDA/Triton median **{xm}×**, range {xl}×–{xh}×",
        f"Triton/eager {tm}×, CUDA/eager {cm}×",
        f"孤立算子的 CUDA 对 eager 中位是 {cm}×",
        f"最高 {xh}×",
    ]
    if wins:
        errors.append(f"README says CUDA never beats Triton, but {wins} T4 rows have CUDA/TRI > 1")

    e2e = re.findall(r"(\d+)x(\d+)x(\d+)\s+6\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)x\s+([\d.]+)x",
                     text, re.M)
    if len(e2e) != 8:
        errors.append(f"T4 end-to-end table: expected 8 rows, found {len(e2e)}")
        return
    gains = [float(r[6]) for r in e2e]
    for r in e2e:
        if abs(float(r[3]) / float(r[5]) - float(r[6])) > 0.003:
            errors.append(f"e2e gain {r[6]} != eager/cuda {float(r[3]) / float(r[5]):.4f}")
    m = re.search(r"端到端增益：中位 ([\d.]+)×\s+范围 ([\d.]+)× – ([\d.]+)×", text)
    if not m:
        errors.append("T4 end-to-end summary line missing")
        return
    med, lo, hi = m.groups()
    for printed, recomputed in ((med, st.median(gains)), (lo, min(gains)), (hi, max(gains))):
        if abs(float(printed) - recomputed) > TOL:
            errors.append(f"e2e summary printed {printed} vs recomputed {recomputed:.4f}")
    small32, small16 = e2e[0], e2e[4]
    snippets += [
        f"整体增益中位仅 **{med}×**（范围 {lo}× – {hi}×）",
        f"median **{med}×** end to end",
        f"唯一的 {hi}×",
        f"fp32 {float(small32[3]):.3f} ms，同配置 fp16 {float(small16[3]):.3f} ms",
        f"同配置 fp16 是 {float(small16[6]):.3f}×",
    ]
    if max(gains) != float(small32[6]) or sum(g > 1.05 for g in gains) != 1:
        errors.append("README calls the small-batch fp32 cell the only end-to-end win; data disagrees")
    for s in snippets:
        if s not in readme:
            errors.append(f"README snippet missing or changed: {s}")


def _stability(readme: str, errors: list[str]) -> None:
    lines = (ROOT / "stability.txt").read_text(encoding="utf-8").splitlines()
    rows = [l.split() for l in lines if re.match(r"^\s+\de\+\d\d\s", l)]
    if len(rows) != 6:
        errors.append(f"stability.txt: expected 6 rows, found {len(rows)}")
        return
    for off, ratio, e2, e1, deg in rows:
        exp = int(off.split("e+")[1])
        label = "0" if exp == 0 else f"1e{exp}"
        ratio_s = f"{float(ratio):,.1f}".replace(",", " ")
        e1_s = "**NaN**" if e1 == "nan" else e1
        deg_s = "—" if deg == "NaN" else deg
        line = f"| {label} | {ratio_s} | {e2} | {e1_s} | {deg_s} |"
        if line not in readme:
            errors.append(f"README stability row missing or changed: {line}")


def check() -> list[str]:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    errors: list[str] = []
    _gtx1650(readme, errors)
    _t4(readme, errors)
    _stability(readme, errors)
    return errors


def main(argv: list[str]) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    errors = check()
    for e in errors:
        print("FAIL:", e)
    if not errors:
        print("README numbers match bench_results.json, results_t4/ and stability.txt")
    return 1 if errors and "--check" in argv else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
