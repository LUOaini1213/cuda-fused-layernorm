"""CPU checks for the accuracy gate and for the eps-argument bug it is meant to catch.

No GPU is used. ``kernel_emulator`` re-implements the two-pass kernel math in
float32 and receives eps exactly as the CUDA kernel would read it from the
launch arguments. These tests prove the gate's logic and its wiring into
bench.py / three_way/bench3.py; they do not prove anything about real GPU runs.
"""
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy
import torch
import torch.nn.functional as F

from test_input_guards import ROOT, WRAPPERS, cupy_metadata, inputs, kernel_scalar_params, tensor

sys.path.insert(0, str(ROOT))
import accuracy_gate as G  # noqa: E402

EPS = 1e-5


def load(path, name, extra_modules=None):
    saved = {}
    for key, module in (extra_modules or {}).items():
        saved[key] = sys.modules.get(key)
        sys.modules[key] = module
    try:
        spec = importlib.util.spec_from_file_location(name, ROOT / path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for key, previous in saved.items():
            if previous is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = previous
    return module


def kernel_reads(args, kernel_file="kernel.cu"):
    """Decode scalar launch args the way CuPy packs them and the kernel reads them.

    CuPy RawKernel packs a Python float as float64 and a Python int as int64;
    NumPy scalars keep their own dtype. ``fused_add_ln_2pass`` reads 4 bytes
    per scalar parameter (int / float).
    """
    c_types = kernel_scalar_params(kernel_file)["fused_add_ln_2pass"]
    decoded = []
    for value, c_type in zip(args, c_types):
        if isinstance(value, numpy.generic):
            raw = value.tobytes()
        elif isinstance(value, float):
            raw = numpy.float64(value).tobytes()
        elif isinstance(value, int):
            raw = numpy.int64(value).tobytes()
        else:
            raise TypeError(type(value))
        decoded.append(struct.unpack("<f" if c_type == "float" else "<i", raw[:4])[0])
    return decoded


def launch_scalars(wrapper):
    """The four scalar arguments the wrapper hands to the (mocked) kernel."""
    with patch.object(wrapper.torch, "empty_like", side_effect=lambda x: tensor(x.shape, x.dtype, str(x.device), "output")), \
            patch.object(wrapper, "_as_cupy", side_effect=lambda x: x.name), \
            patch.object(wrapper, "_fn", return_value=MagicMock()) as function, \
            patch.object(wrapper, "_cupy", return_value=cupy_metadata()), \
            patch.object(wrapper, "_CUDA_DEVICE", None):
        wrapper.fused_add_layernorm(**inputs(), eps=EPS)
        return function.return_value.call_args.args[2][-4:]


def kernel_emulator(eps_seen):
    """float32 re-implementation of fused_add_ln_2pass with the eps the kernel saw."""
    def fn(x, r, w, b, *args, **kwargs):
        s = x.float() + r.float()
        n = s.shape[-1]
        mean = s.sum(-1, keepdim=True) / n
        d = s - mean
        var = (d * d).sum(-1, keepdim=True) / n
        rstd = torch.rsqrt(var + torch.tensor(eps_seen, dtype=torch.float32))
        return (d * rstd * w.float() + b.float()).to(x.dtype), s.to(x.dtype)
    return fn


def max_err(fn, rows=512, D=128, dtype=torch.float32):
    torch.manual_seed(0)
    x, r = torch.randn(rows, D, dtype=dtype), torch.randn(rows, D, dtype=dtype)
    w, b = torch.randn(D, dtype=dtype), torch.randn(D, dtype=dtype)
    ref = F.layer_norm(x.double() + r.double(), (D,), w.double(), b.double(), EPS)
    return (fn(x, r, w, b)[0].double() - ref).abs().max().item()


def eager(x, r, w, b, *args, **kwargs):
    s = x + r
    return F.layer_norm(s, (s.shape[-1],), w, b, EPS), s


class GateRuleTests(unittest.TestCase):
    def test_committed_gtx1650_fp32_errors_fail_the_gate(self):
        rows = json.loads((ROOT / "bench_results.json").read_text(encoding="utf-8"))
        fp32 = [r for r in rows if r["dtype"] == "fp32"]
        self.assertEqual(len(fp32), 6)
        for row in fp32:
            errs = {k: row["err_" + k] for k in ("eager", "2pass", "1pass")}
            with self.subTest(case=row["case"]):
                self.assertEqual(len(G.error_failures(errs, "fp32")), 2)

    def test_committed_gtx1650_fp16_errors_pass_the_gate(self):
        rows = json.loads((ROOT / "bench_results.json").read_text(encoding="utf-8"))
        for row in [r for r in rows if r["dtype"] == "fp16"]:
            errs = {k: row["err_" + k] for k in ("eager", "2pass", "1pass")}
            self.assertEqual(G.error_failures(errs, "fp16"), [], row["case"])

    def test_bound_floor_and_nan(self):
        self.assertEqual(G.error_failures({"eager": 1e-7, "k": 9e-6}, "fp32"), [])
        self.assertEqual(len(G.error_failures({"eager": 1e-7, "k": 1.1e-5}, "fp32")), 1)
        self.assertEqual(G.error_failures({"eager": 1e-5, "k": 4.9e-5}, "fp32"), [])
        self.assertEqual(len(G.error_failures({"eager": 1e-5, "k": 5.1e-5}, "fp32")), 1)
        self.assertEqual(len(G.error_failures({"eager": 1e-6, "k": math.nan}, "fp32")), 1)
        self.assertEqual(len(G.error_failures({"eager": math.nan, "k": 1e-6}, "fp32")), 1)

    def test_constant_row_rule_on_cpu(self):
        for dtype in (torch.float32, torch.float16):
            self.assertEqual(G.constant_row_failures(eager, 128, dtype, "cpu"), [])
        nan_out = lambda x, r, w, b: (torch.full_like(x, math.nan), x + r)
        self.assertEqual(len(G.constant_row_failures(nan_out, 128, torch.float32, "cpu")), 1)
        shifted = lambda x, r, w, b: (eager(x, r, w, b)[0] + 1e-3, x + r)
        self.assertEqual(len(G.constant_row_failures(shifted, 128, torch.float32, "cpu")), 1)


class EpsBugReproductionTests(unittest.TestCase):
    """The wrapper's scalars, decoded as the kernel reads them, drive the emulator."""

    def test_current_wrappers_deliver_eps_and_pass_the_gate(self):
        for wrapper in WRAPPERS:
            stride, n, eps_seen, has_res = kernel_reads(launch_scalars(wrapper))
            with self.subTest(entry=wrapper.__name__):
                self.assertEqual((stride, n, has_res), (3, 3, 1))
                self.assertAlmostEqual(eps_seen, EPS, delta=1e-12)
                fn = kernel_emulator(eps_seen)
                errs = {"eager": max_err(eager), "2pass": max_err(fn)}
                self.assertEqual(G.error_failures(errs, "fp32"), [])
                self.assertEqual(G.constant_row_failures(fn, 128, torch.float32, "cpu"), [])

    def test_python_float_eps_is_caught_by_both_gate_rules(self):
        # The pre-2026-11-03 launch tuple: (stride, n, float(eps), int(has_res)).
        stride, n, eps_seen, has_res = kernel_reads((3, 3, float(EPS), 1))
        self.assertEqual((stride, n, has_res), (3, 3, 1))   # ints survive the truncation
        self.assertLess(eps_seen, 0)
        fn = kernel_emulator(eps_seen)
        errs = {"eager": max_err(eager), "2pass": max_err(fn)}
        # Same signature as the committed data: ~1e-6 eager vs a few 1e-5 fused.
        self.assertLess(errs["eager"], 5e-6)
        self.assertGreater(errs["2pass"], 1e-5)
        self.assertEqual(len(G.error_failures(errs, "fp32")), 1)
        failures = G.constant_row_failures(fn, 128, torch.float32, "cpu")
        self.assertEqual(len(failures), 1)
        self.assertIn("NaN", failures[0])


class ReadmeClaimTests(unittest.TestCase):
    def test_emulated_eps_bug_error_range_quoted_in_readme(self):
        lost = kernel_reads((3, 3, float(EPS), 1))[2]
        errs = [max_err(kernel_emulator(lost), D=D) for D in (32, 128, 1024)]
        # README: "也得到 2.7e−5 – 3.1e−5 的误差"
        self.assertTrue(2.5e-5 <= min(errs) <= 2.9e-5, errs)
        self.assertTrue(2.9e-5 <= max(errs) <= 3.3e-5, errs)
        self.assertIn("2.7e−5 – 3.1e−5", (ROOT / "README.md").read_text(encoding="utf-8"))

    def test_launch_heuristics_quoted_in_readme(self):
        # README: both sides use fixed heuristics; at D=128 Triton 128 threads/row, CUDA 32.
        triton_src = (ROOT / "three_way/fused_ln_triton.py").read_text(encoding="utf-8")
        self.assertNotIn("autotune", triton_src)
        self.assertNotIn("triton.Config", triton_src)
        self.assertIn("num_warps = 4 if block <= 512 else 8", triton_src)
        for wrapper in WRAPPERS:
            self.assertEqual(wrapper._threads_for(128), 32)
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("at D=128 Triton uses 128 threads per row, this kernel 32", readme)
        self.assertIn("D=128 时每行 32 线程", readme)
        self.assertNotIn("自动调优", readme)


class BenchWiringTests(unittest.TestCase):
    """bench.py / bench3.py run() must record gate failures per shape."""

    def drive(self, module, impls_patch):
        with patch.object(module, "DEVICE", "cpu"), \
                patch.object(module, "time_one", return_value=1.0), \
                patch.object(module, "CASES", [c for c in module.CASES if c[1] <= 8192 and c[2] <= 128][:2]), \
                impls_patch, patch("builtins.print"):
            return module.run(torch.float32, "fp32")

    def test_bench_py_gate(self):
        bench = load("bench.py", "bench_under_test", {"fused_ln_cuda": WRAPPERS[0]})
        for eps_seen, expect_fail in [(numpy.float32(EPS), False), (kernel_reads((3, 3, float(EPS), 1))[2], True)]:
            rows = self.drive(bench, patch.object(bench.F, "fused_add_layernorm", kernel_emulator(eps_seen)))
            with self.subTest(eps_seen=float(eps_seen)):
                self.assertEqual(len(rows), 2)
                self.assertEqual(all(r["gate_failures"] for r in rows), expect_fail)
                self.assertEqual(any(r["gate_failures"] for r in rows), expect_fail)
                with patch("builtins.print"):
                    code = G.report([m for r in rows for m in r["gate_failures"]])
                self.assertEqual(code, 1 if expect_fail else 0)
                if expect_fail:
                    joined = " ".join(m for r in rows for m in r["gate_failures"])
                    self.assertIn("constant row", joined)
                    self.assertIn("max|err|", joined)

    def test_bench3_gate(self):
        fake_triton = types.ModuleType("fused_ln_triton")
        fake_triton.fused_add_layernorm = eager
        bench3 = load("three_way/bench3.py", "bench3_under_test",
                      {"fused_ln_cuda": WRAPPERS[1], "fused_ln_triton": fake_triton})
        for eps_seen, expect_fail in [(numpy.float32(EPS), False), (kernel_reads((3, 3, float(EPS), 1))[2], True)]:
            rows = self.drive(bench3, patch.object(bench3.CUDA, "fused_add_layernorm", kernel_emulator(eps_seen)))
            with self.subTest(eps_seen=float(eps_seen)):
                self.assertEqual(all(r["gate_failures"] for r in rows), expect_fail)
                self.assertEqual(any(r["gate_failures"] for r in rows), expect_fail)
                if expect_fail:
                    joined = " ".join(m for r in rows for m in r["gate_failures"])
                    self.assertIn("cuda2", joined)
                    self.assertIn("constant row", joined)
                    self.assertIn("max|err|", joined)
                    self.assertNotIn("triton", joined)

    def test_report_exit_codes(self):
        with patch("builtins.print"):
            self.assertEqual(G.report([]), 0)
            self.assertEqual(G.report(["x"]), 1)


class E2EOrderTests(unittest.TestCase):
    def test_every_mode_is_timed_first_in_some_round(self):
        e2e = load("three_way/bench_e2e.py", "bench_e2e_under_test", {"fused_ln_cuda": WRAPPERS[1]})
        modes = ["eager", "triton", "cuda"]
        firsts = {e2e.rotated(modes, k)[0] for k in range(e2e.REPEATS)}
        self.assertEqual(firsts, set(modes))
        self.assertGreaterEqual(e2e.REPEATS, 3)
        self.assertTrue(callable(e2e.warmup))


if __name__ == "__main__":
    unittest.main()
