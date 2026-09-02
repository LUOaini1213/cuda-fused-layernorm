# Fused Residual-Add + LayerNorm — CUDA C++ 实现与对照

> ### 验证状态（2026-09-02）
>
> 表中所有数字均在单张 GTX 1650（`sm_75`，4 GiB）上实测、可复跑。
> 与 Triton 版的对照此前只能跨卡参照（本机 GTX 1650 vs 记录中的 Kaggle T4），**不是受控 A/B**。
> `three_way/bench3.py` 与 `cloud_gpu_verify.ipynb` 提供同卡三方受控对照，结果待补。


把 [TikTok TechJam 2026 Track 3](https://github.com/LUOaini1213/tiktok-techjam-2026-track3)
里的 Triton 融合算子 `kernels/fused_layernorm.py` 用 **CUDA C++ 重写**，并在同形状、
同架构代次上与 PyTorch eager 及原 Triton 版对照。

## 为什么挑这个算子

Transformer 的 pre-norm 残差模式每层重复两次：

```python
x = x + sublayer(norm(x))     # add 一个 kernel，norm 另一个
```

两个 kernel 各自完整读写 `[B, S, D]` 激活。融合后由四趟显存往返（读 x、读 y、写 sum；
读 sum、写 normed）降到两趟。LayerNorm 在这些尺寸下是 **memory-bound**，省的就是带宽。

**Triton 版用 `tl.sum(s, axis=0)` 一行完成行内归约。CUDA 版必须自己实现**：
warp 内 `__shfl_down_sync` 蝶形规约 → shared memory 跨 warp 汇总 → 广播回全体线程。
这是移植过程中真正需要动脑的部分，也是本项目的意义所在。

## 构建

不需要 CUDA Toolkit，也不需要 MSVC —— 内核经 **NVRTC 运行时编译**（CuPy `RawModule`），
`.cu` 源码是标准 CUDA C++。

```bash
pip install torch cupy-cuda12x
python bench.py        # 性能与精度
python stability.py    # 数值稳定性实验
```

## 实现要点

| | 两遍归约 `fused_add_ln_2pass` | 单遍归约 `fused_add_ln_1pass` |
|---|---|---|
| 均值 | `mean = Σs/N` | 同左 |
| 方差 | `Σ(s-mean)²/N` | `Σs²/N − mean²` |
| 块规约次数 | 2 | 1 |
| 数值行为 | 全域稳定 | `\|mean\|≫std` 时相消失效 |

一行一个 block；行内元素缓存在寄存器（`float s_loc[8]`），第二遍不再读显存。
归约无论存储 dtype 一律 fp32 累加，与 `nn.LayerNorm` 一致，不占精度预算。
`d_model > 1024` 或 dtype 不支持时透明回落 PyTorch，调用方无需判断。

## 结果

GTX 1650（4 GiB，sm_75，14 SM）· torch 2.7.1+cu126 · cupy 13.6.0 ·
形状取自原仓库 `results/kaggle_t4_triton_bench.log` 的孤立 LayerNorm 微基准。

### fp32

| case | rows | D | eager (ms) | CUDA 2pass (ms) | vs eager | 有效带宽 | max\|err\| |
|---|---|---|---|---|---|---|---|
| shape 1/5/9-11 | 8 192 | 128 | 0.2908 | 0.1876 | **1.550×** | 83.3 GiB/s | 2.68e-05 |
| shape 2 | 128 | 128 | 0.0581 | 0.0922 | 0.631× | 2.6 GiB/s | 2.40e-05 |
| shape 6 | 1 280 000 | 128 | 36.0922 | 17.8411 | **2.023×** | 136.8 GiB/s | 3.75e-05 |
| shape 7 | 8 192 | 32 | 0.1949 | 0.1223 | **1.593×** | 31.9 GiB/s | 3.44e-05 |
| shape 8 | 8 192 | 1 024 | 1.0671 | 1.0345 | 1.032× | 120.8 GiB/s | 3.34e-05 |
| shape 13 | 65 536 | 128 | 1.8722 | 0.9748 | **1.921×** | 128.2 GiB/s | 3.02e-05 |

### fp16

| case | eager (ms) | CUDA 2pass (ms) | vs eager | 有效带宽 |
|---|---|---|---|---|
| shape 1/5/9-11 | 0.1945 | 0.1903 | 1.022× | 41.0 GiB/s |
| shape 2 | 0.0614 | 0.1089 | 0.564× | 1.1 GiB/s |
| shape 6 | 27.5989 | 16.7070 | **1.652×** | 73.1 GiB/s |
| shape 7 | 0.1692 | 0.1290 | **1.311×** | 15.1 GiB/s |
| shape 8 | 0.5796 | 0.6759 | 0.858× | 92.5 GiB/s |
| shape 13 | 1.4356 | 0.9297 | **1.544×** | 67.2 GiB/s |

**12 组中位加速比 1.428×。** 同形状原 Triton 版在 T4 上对 eager 的中位为 1.348×。

> ⚠️ 这**不是**受控 A/B：T4 与 GTX 1650 同属 sm_75 Turing，但显存带宽与 SM 数不同。
> 两列只能说明「同形状、同架构代次下量级相当」，不能得出 CUDA 版快于 Triton 版的结论。

### 测量方法与重复性

三次独立运行的中位加速比为 1.481× / 1.342× / 1.428×，波动全部来自小形状：

| case | 三次 vs_eager | 波动 |
|---|---|---|
| shape 6 | 2.014 / 2.014 / 2.023 | ±0.4% |
| shape 13 | 1.902 / 1.900 / 1.921 | ±0.6% |
| shape 8 | 1.013 / 1.036 / 1.032 | ±1.2% |
| shape 7 | 1.741 / 1.607 / 1.593 | ±9% |
| shape 2 | 0.507 / 0.692 / 0.631 | ±20% |

**大形状的结论是稳的，小形状的单个数字不要单独引用。**
初版固定 50 次迭代时，shape 2 单次仅 0.08 ms，一组测量总共才 4 ms，完全被噪声主导；
现改为按单次耗时自适应迭代（每组凑约 200 ms，20–2000 次），小形状的抖动已收窄但仍存在——
这类形状本就由 kernel launch 开销主导，抖动是其固有性质，不是测量缺陷。

## 三个发现

### 1. 加速比高度依赖形状，小 batch 反而是负收益

`shape 2` 只有 128 行，两种融合实现都**慢于** eager（0.63× / 0.56×）——
有效带宽仅 2.6 GiB/s，说明总工作量太小，kernel launch 开销主导，融合省下的那两趟
显存往返根本不够抵。`shape 8`（D=1024）已经跑到 120.8 GiB/s 接近带宽上限，
eager 同样饱和，所以融合几乎没有头寸（1.032×）。

**融合算子的收益窗口是「行数足够多、单行足够窄」**：shape 6 / 13 正落在窗口里，
拿到 2.02× / 1.92×，有效带宽 136.8 / 128.2 GiB/s。

### 2. 单遍归约省一次 `__syncthreads`，代价是会产出 NaN

`stability.py` 给标准正态数据加偏置，扫描 `|mean|/std`：

| offset | \|mean\|/std | err 两遍 | err 单遍 | 劣化 |
|---|---|---|---|---|
| 0 | 0.0 | 3.091e-05 | 3.091e-05 | 1× |
| 1e1 | 7.1 | 3.126e-05 | 6.429e-05 | 2× |
| 1e2 | 70.8 | 4.338e-05 | 4.631e-03 | 107× |
| 1e3 | 708.1 | 2.523e-04 | 3.833e-01 | 1519× |
| 1e4 | 7 081.2 | 4.211e-03 | **NaN** | — |
| 1e5 | 70 811.6 | 3.389e-02 | **NaN** | — |

`Σs²/N − mean²` 是两个量级接近的大数相减，有效位被吃光后方差算成负数，
`rsqrtf(负数 + eps)` 直接 NaN。均值≈0 时两者完全等价，所以**用随机正态数据测不出这个问题**——
必须刻意构造偏置才能暴露。两遍归约多一次块规约往返，换全域稳定，这是正确的取舍。

### 3. 4 GiB 卡上，benchmark 本身会把结果测歪

初版 benchmark 在同一作用域里同时持有 eager / 2pass / 1pass 三份实现的输入与输出，
`shape 6` fp32 下 8 个 655 MB 张量共 5.2 GB > 4 GiB 显存，
测得 **137 ms（0.988×，看起来毫无收益）**。

改成精度与计时分阶段、阶段间释放张量后，同一 kernel 测得 **17.9 ms（2.014×）**，
有效带宽 136 GiB/s。**同一份代码，两个结论，差别只在测量方法。**
显存受限设备上做性能对照，必须隔离每份实现的驻留集。

## 文件

| 文件 | 说明 |
|---|---|
| `kernel.cu` | CUDA C++ 内核：warp shuffle + shared 块规约，两遍/单遍两个模板实例 |
| `fused_ln_cuda.py` | NVRTC 编译、torch↔cupy 零拷贝、不满足条件时回落 PyTorch |
| `bench.py` | 性能与精度对照（真值取 float64） |
| `stability.py` | 单遍 vs 两遍数值稳定性扫描 |
| `bench_results.json` | 上表原始数据 |
