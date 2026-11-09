# Fused Residual-Add + LayerNorm — CUDA C++ 实现与对照

**English summary.** A hand-written CUDA C++ port (NVRTC via CuPy) of a Triton fused residual-add + LayerNorm,
compared with PyTorch eager and the original Triton kernel on one Tesla T4 (12 shape/dtype cases).
**Result: negative.** The CUDA port is slower than Triton in all 12 cases (CUDA/Triton median **0.812×**, range 0.468×–0.997×;
Triton/eager 1.341×, CUDA/eager 1.087×). Inside a 6-layer Transformer forward the CUDA norm gives a median **0.983×** end to end.
**Bug found 2026-11-03:** the wrapper passed `eps` as a Python float, which CuPy packs as float64, so the kernel's `float eps` read about −1.4e−33.
All committed error columns were measured with that bug (fp32 errors ~2–4e−5, about 20× eager); the wrapper is fixed, the error columns need a GPU rerun.
**Limits:** one T4 run (Colab, 2026-09-02) plus one committed GTX 1650 run (of three); timings are expected but not yet re-verified after the fix;
why Triton is faster is an untested hypothesis (fixed launch heuristics on both sides; at D=128 Triton uses 128 threads per row, this kernel 32).

> ### 受控对照结果（2026-09-02，Tesla T4）
>
> 此前与 Triton 版的比较只能跨卡参照（本机 GTX 1650 vs 记录中的 Kaggle T4），
> 那不是受控 A/B，只能说「量级相当」。现已在 **单张 T4 上跑完三方受控对照**，
> 结论与跨卡参照给人的印象**相反**：
>
> | 同卡对照（T4，12 组形状） | 中位 | 范围 |
> |---|---|---|
> | Triton 对 eager | **1.341×** | 0.592× – 2.054× |
> | CUDA C++ 对 eager | 1.087× | 0.278× – 1.620× |
> | **CUDA C++ 对 Triton** | **0.812×** | 0.468× – 0.997× |
>
> **本 CUDA 实现慢于 Triton 版，12 组中无一组胜出（最高 0.997×）。**
> 两边都是**固定启发式**，都没有 autotune：Triton 版 `num_warps = 4 if BLOCK <= 512 else 8`
> （`three_way/fused_ln_triton.py`，D≤512 时每行 128 线程），本实现每线程约 4 元素、至少一个 warp
> （D=128 时每行 32 线程）。线程数差异是一个**尚未验证的假设**，需要在 GPU 上扫线程数才能确认。
>
> **端到端**：把算子接进真实 Transformer block（6 层，注意力与 MLP 共用同一实现，
> 差别只在 norm 一处）后，CUDA 版的整体增益中位仅 **0.983×**（范围 0.642× – 1.609×），
> 而同一张 T4 上孤立算子的 CUDA 对 eager 中位是 1.087×：LayerNorm 占 block 成本的比例太小，
> 注意力与四个 GEMM 才是大头。
> 唯一的 1.609×（fp32「小 batch 短序列」）**疑似预热伪影**：这一格是全程第一个计时格，
> 只有 eager 一列变慢（fp32 4.075 ms，同配置 fp16 1.641 ms），Triton / CUDA 两列几乎不随 dtype 变化；
> 同配置 fp16 是 0.642×。`three_way/bench_e2e.py` 已加全局预热、轮换计时顺序、每配置 3 轮，结论待重跑。
>
> **精度更正（2026-11-03）**：wrapper 把 `eps` 以 Python float 传给 CuPy `RawKernel`，
> CuPy 按 float64 打包，内核的 `const float eps` 只读低 4 字节，读到约 −1.4e−33，等于没有 eps。
> 本仓所有误差列（下文 GTX 1650 表、`stability.txt`、T4 原始结果里的 `err_cuda2`）都是带着这个 bug 测的，
> fp32 误差约为 eager 的 20 倍；方差为 0 的行会得到 NaN。wrapper 已改为传 `np.float32` / `np.int32`，
> 基准加了真正会失败的精度门（`accuracy_gate.py`），CPU 上用内核数学的 float32 仿真复现了误差量级。
> **误差列需在 GPU 上重跑后更新**；速度列预计不受影响，但同样要等重跑确认。
>
> 本仓保留全部数据，包括不利于本实现的那些。复现见 `cloud_gpu_verify.ipynb`。


把 [TikTok TechJam 2026 Track 3](https://github.com/LUOaini1213/fp16x3-transformer)
（仓库已改名为 `fp16x3-transformer`）里的 Triton 融合算子 `kernels/fused_layernorm.py`
用 **CUDA C++ 重写**，并在同形状、同架构代次上与 PyTorch eager 及原 Triton 版对照。

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

## 构建与运行

Python 3.11。不需要 CUDA Toolkit，也不需要 MSVC —— 内核经 **NVRTC 运行时编译**（CuPy `RawModule`），
`.cu` 源码是标准 CUDA C++。GPU 依赖按 T4 那次运行的版本钉在 `requirements-gpu.txt`。

```bash
pip install -r requirements-gpu.txt
python bench.py        # 性能与精度（GTX 1650 表），精度门不过时退出码为 1
python stability.py    # 数值稳定性实验
cd three_way && python bench3.py && python bench_e2e.py   # T4 三方对照与端到端
```

无 GPU 的检查（Windows / Linux 均可，CI 两个平台都跑）：

```bash
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -B -m unittest discover -s tests -v
python check_readme_numbers.py --check
```

CPU 测试覆盖：输入守卫、CPU 回落、内核标量参数的 C 类型（逐个对照 `kernel.cu` 签名）、
精度门的规则与接线、本 README 里的数字与已提交结果文件一致。它们**不能**证明 GPU 上的数值或性能。

## 实现要点

| | 两遍归约 `fused_add_ln_2pass` | 单遍归约 `fused_add_ln_1pass` |
|---|---|---|
| 均值 | `mean = Σs/N` | 同左 |
| 方差 | `Σ(s-mean)²/N` | `Σs²/N − mean²` |
| 块规约次数 | 2 | 1 |
| 数值行为 | 全域稳定 | `\|mean\|≫std` 时相消失效 |

一行一个 block；行内元素缓存在寄存器（`float s_loc[8]`），第二遍不再读显存。
归约无论存储 dtype 一律 fp32 累加，与 `nn.LayerNorm` 的累加精度相同。
但已提交的误差列显示 fp32 误差约为 eager 的 20 倍，原因是上面的 eps 传参 bug，不是累加方式；
修复后的精度要等 GPU 重跑才能下结论。
`d_model > 1024` 或 dtype 不支持时透明回落 PyTorch，调用方无需判断。

## 结果

GTX 1650（4 GiB，sm_75，14 SM）· torch 2.7.1+cu126 · cupy 13.6.0 ·
形状取自原仓库 `results/kaggle_t4_triton_bench.log` 的孤立 LayerNorm 微基准。

### fp32

| case | rows | D | eager (ms) | CUDA 2pass (ms) | vs eager | 有效带宽 | max\|err\|（eps bug，待重跑） |
|---|---|---|---|---|---|---|---|
| shape 1/5/9-11 | 8 192 | 128 | 0.2908 | 0.1876 | **1.550×** | 83.3 GiB/s | 2.68e-05 |
| shape 2 | 128 | 128 | 0.0581 | 0.0922 | 0.631× | 2.6 GiB/s | 2.40e-05 |
| shape 6 | 1 280 000 | 128 | 36.0922 | 17.8411 | **2.023×** | 136.8 GiB/s | 3.75e-05 |
| shape 7 | 8 192 | 32 | 0.1949 | 0.1223 | **1.593×** | 31.9 GiB/s | 3.44e-05 |
| shape 8 | 8 192 | 1 024 | 1.0671 | 1.0345 | 1.032× | 120.8 GiB/s | 3.34e-05 |
| shape 13 | 65 536 | 128 | 1.8722 | 0.9748 | **1.921×** | 128.2 GiB/s | 3.02e-05 |

同一批数据里 eager 的 fp32 误差为 1.13e-06 – 2.33e-06（`bench_results.json` 的 `err_eager`）。
CPU 上按内核数学做 float32 仿真、把 eps 换成 −1.37e−33，也得到 2.7e−5 – 3.1e−5 的误差
（`tests/test_accuracy_gate.py`），与上表量级一致。

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
> 同卡受控对照见文首：CUDA 版慢于 Triton。

### 测量方法与重复性

GTX 1650 上共跑过三次。**只有第三次的原始数据提交在 `bench_results.json`**（即上表）；
前两次没有提交日志，下面前两列是当时记下的数字，无法从仓库复核。
三次的中位加速比为 1.481× / 1.342× / 1.428×，波动全部来自小形状：

| case | 三次 vs_eager（前两次无日志） | 波动 |
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

`stability.py` 给标准正态数据加偏置，扫描 `|mean|/std`（`stability.txt`，GTX 1650；
误差列同样是带着 eps bug 测的，待重跑）：

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
（偏置 1e4 时 fp32 的舍入误差远大于 1e−5，正确传入 eps 预计不会消除这里的 NaN，但这同样要重跑确认。）

### 3. 4 GiB 卡上，benchmark 本身会把结果测歪

初版 benchmark 在同一作用域里同时持有 eager / 2pass / 1pass 三份实现的输入与输出，
`shape 6` fp32 下 8 个 655 MB 张量共 5.2 GB > 4 GiB 显存，
测得 **137 ms（0.988×，看起来毫无收益）**。

改成精度与计时分阶段、阶段间释放张量后，同一 kernel 测得 **17.9 ms（2.014×）**，
有效带宽 136 GiB/s。**同一份代码，两个结论，差别只在测量方法。**
显存受限设备上做性能对照，必须隔离每份实现的驻留集。
（137 ms 与 17.9 ms 两次都来自未提交日志的早期运行；已提交的第三次运行是 17.84 ms / 2.023×。）

## 文件

| 文件 | 说明 |
|---|---|
| `kernel.cu` | CUDA C++ 内核：warp shuffle + shared 块规约，两遍/单遍两个模板实例 |
| `fused_ln_cuda.py` | NVRTC 编译、torch↔cupy 零拷贝、不满足条件时回落 PyTorch |
| `accuracy_gate.py` | 精度门：误差 ≤ max(5× eager, 下限)、无 NaN、常数行等于 bias |
| `bench.py` | 性能与精度对照（真值取 float64），精度门不过则退出码 1 |
| `stability.py` | 单遍 vs 两遍数值稳定性扫描 |
| `bench_results.json` | GTX 1650 表的原始数据 |
| `three_way/` | T4 三方对照 `bench3.py`、端到端 `bench_e2e.py` 及其使用的 wrapper / Triton 副本 |
| `results_t4/t4_three_way_and_e2e.txt` | T4 运行的控制台输出（文首数字的来源） |
| `check_readme_numbers.py` | 从上面两个结果文件重算本 README 的数字，`--check` 不一致即失败 |
| `tests/` | CPU 测试（无 GPU） |

## 维护记录

**2026-11-03 审阅修复：** eps / 整型标量参数按内核签名的 4 字节类型传入（`fused_ln_cuda._scalar_args`）；
`bench.py`、`three_way/bench3.py` 加精度门；`bench_e2e.py` 加全局预热、轮换顺序、每配置 3 轮；
新增 README 数字核对与 Windows CI。已提交的结果文件与内核未改动（测试会核对其 Git blob）。
已提交的结果由修改前的脚本产生；重跑前文中的误差列、端到端 1.609× 都应视为待复核。

**2026-10-12 输入边界更新：** 两个维护入口（根目录与 `three_way/`）在
CUDA 原始指针路径之前检查设备、残差 dtype/原始形状，以及 weight/bias
的一维向量长度。非法输入显式拒绝，`python -O` 不会绕过检查。CuPy 仅在
实际 CUDA 路径导入，并要求当前 CuPy device 与输入 device 相同；不自动切换
context。现有 module/function cache 固定在本进程首次合法 CUDA device，
后续切设备显式拒绝，需新进程使用另一设备。合法 CPU fallback 的广播、dtype 提升与可选 affine
语义保留。离线 CPU/metadata 探针不能证明 GPU 正确性或性能。
[边界与验收](docs/input-guards.md) · [历史 wrapper 来源](historical/4af3e68/source.json)
