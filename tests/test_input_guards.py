"""Actual wrapper entries; CPU numerics and metadata-only fake CUDA backends.

No CUDA tensor/device is created or probed. Fake calls prove routing boundaries,
not GPU correctness, stream/device-context compatibility, or performance.
"""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
BASE = "4af3e689dc912754061d247991b3f73e77d60a89"
LEGACY_SHA = "ed5389417ef8b94e4ad2c09e63fa3c54dbef326e8728164ccef90c843ff9b9e2"
PROTECTED = ("bench_results.json", "results_t4/t4_three_way_and_e2e.txt", "kernel.cu",
             "three_way/kernel.cu", "bench.py", "three_way/bench3.py", "three_way/bench_e2e.py",
             "three_way/fused_ln_triton.py")


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


WRAPPERS = [load("fused_ln_cuda.py", "guarded_root"),
            load("three_way/fused_ln_cuda.py", "guarded_three_way")]


def tensor(shape=(2, 3), dtype=torch.float32, device="cuda:0", name="x"):
    """Tensor-spec metadata mock, never a CUDA tensor or allocation."""
    value = MagicMock(spec=torch.Tensor)
    value.shape, value.dtype, value.device = shape, dtype, torch.device(device)
    value.is_cuda, value.name = device.startswith("cuda"), name
    value.dim.return_value = len(shape)
    value.numel.return_value = math.prod(shape)
    value.contiguous.return_value = value
    value.is_contiguous.return_value = True
    value.stride.return_value = shape[-1] if shape else 0

    def view(*dimensions):
        if len(dimensions) == 1 and isinstance(dimensions[0], tuple):
            dimensions = dimensions[0]
        if -1 in dimensions:
            dimensions = (math.prod(shape) // dimensions[-1], dimensions[-1])
        return tensor(dimensions, dtype, device, name)
    value.view.side_effect = view
    return value


def inputs(**changes):
    return {"x": tensor(), "residual": tensor(name="residual"),
            "weight": tensor((3,), name="weight"), "bias": tensor((3,), name="bias"), **changes}


def cupy_metadata(device=0):
    """A fake context query, with spies for forbidden context switches/wrapping."""
    cp = MagicMock()
    cp.cuda.runtime.getDevice.return_value = device
    return cp


def on_device(device):
    return {"x": tensor(device=device), "residual": tensor(device=device),
            "weight": tensor((3,), device=device), "bias": tensor((3,), device=device)}


class MetadataGuardTests(unittest.TestCase):
    def refuse(self, values, message, error=ValueError):
        for wrapper in WRAPPERS:
            with self.subTest(entry=wrapper.__name__), \
                    patch.object(wrapper.torch, "empty_like", side_effect=AssertionError("allocation reached")) as allocate, \
                    patch.object(wrapper, "_cupy", side_effect=AssertionError("CuPy reached")) as cupy, \
                    patch.object(wrapper, "_fn", side_effect=AssertionError("kernel reached")) as kernel, \
                    patch.object(wrapper, "_as_cupy", side_effect=AssertionError("pointer reached")) as pointer, \
                    patch.object(torch.cuda, "current_device", side_effect=AssertionError("device probe reached")):
                with self.assertRaisesRegex(error, message):
                    wrapper.fused_add_layernorm(**values)
                allocate.assert_not_called()
                cupy.assert_not_called()
                kernel.assert_not_called()
                pointer.assert_not_called()

    def test_cpu_and_other_gpu_affine_pointers_rejected_before_allocation(self):
        for name in ["weight", "bias"]:
            for device in ["cpu", "cuda:1"]:
                self.refuse(inputs(**{name: tensor((3,), device=device, name=name)}), name + " must be on the same CUDA device")

    def test_cpu_and_other_gpu_residual_rejected(self):
        for device in ["cpu", "cuda:1"]:
            self.refuse(inputs(residual=tensor(device=device)), "residual must be on the same CUDA device")

    def test_residual_dtype_mismatch_rejected(self):
        self.refuse(inputs(residual=tensor(dtype=torch.float16)), "residual dtype must match")

    def test_equal_numel_different_original_shape_is_not_accepted(self):
        self.refuse(inputs(residual=tensor((3, 2))), "original shape exactly")

    def test_broadcast_and_short_residual_cannot_reach_raw_pointer_path(self):
        for shape in [(1, 3), (3,), (2, 1), (4, 3)]:
            self.refuse(inputs(residual=tensor(shape)), "original shape exactly")

    def test_affine_vectors_require_exact_last_dimension(self):
        for name in ["weight", "bias"]:
            for shape in [(2,), (4,), (1, 3), (2, 3), ()]:
                self.refuse(inputs(**{name: tensor(shape)}), name + " must have shape")

    def test_nontensor_cuda_residual_rejected(self):
        self.refuse(inputs(residual=1.0), "CUDA residual must be a tensor", TypeError)

    def test_scalar_x_rejected_with_explicit_error(self):
        self.refuse(inputs(x=tensor(())), "at least one dimension")

    def test_valid_metadata_preserves_existing_launch_arguments(self):
        for wrapper in WRAPPERS:
            for residual in [tensor(name="residual"), None]:
                with self.subTest(entry=wrapper.__name__, residual=residual is not None), \
                        patch.object(wrapper.torch, "empty_like", side_effect=lambda x: tensor(x.shape, x.dtype, str(x.device), "output")), \
                        patch.object(wrapper, "_as_cupy", side_effect=lambda x: x.name) as pointer, \
                        patch.object(wrapper, "_fn", return_value=MagicMock()) as function, \
                        patch.object(wrapper, "_cupy", return_value=cupy_metadata()), \
                        patch.object(wrapper, "_CUDA_DEVICE", None):
                    result, total = wrapper.fused_add_layernorm(**inputs(residual=residual))
                    function.assert_called_once_with("2pass", "float")
                    grid, threads, args = function.return_value.call_args.args
                    self.assertEqual(grid, (2,))
                    self.assertEqual(threads, (32,))
                    self.assertEqual(args[-4:], (3, 3, 1e-5, int(residual is not None)))
                    self.assertEqual(pointer.call_count, 6)
                    self.assertEqual(result.shape, (2, 3))
                    self.assertEqual(total.shape, (2, 3))

    def test_direct_pointer_adapter_refuses_cpu_before_cupy(self):
        for wrapper in WRAPPERS:
            for value in [torch.ones(3, device="cpu"), tensor(dtype=torch.float64), tensor()]:
                if not isinstance(value, torch.Tensor):
                    raise AssertionError("fixture must satisfy Tensor metadata")
                if isinstance(value, MagicMock) and value.dtype == torch.float32:
                    value.is_contiguous.return_value = False
                with self.subTest(entry=wrapper.__name__), patch.object(wrapper, "_cupy", side_effect=AssertionError("CuPy reached")):
                    with self.assertRaisesRegex(ValueError, "contiguous supported CUDA tensor"):
                        wrapper._as_cupy(value)

    def test_same_input_device_but_wrong_cupy_context_rejected_without_switch(self):
        for wrapper in WRAPPERS:
            for tensor_device, current in [("cuda:0", 1), ("cuda:1", 0)]:
                cp = cupy_metadata(current)
                with self.subTest(entry=wrapper.__name__, tensor_device=tensor_device), \
                        patch.object(wrapper, "_cupy", return_value=cp), patch.object(wrapper, "_CUDA_DEVICE", None), \
                        patch.object(wrapper.torch, "empty_like", side_effect=AssertionError("allocation reached")) as allocation, \
                        patch.object(wrapper, "_fn", side_effect=AssertionError("kernel reached")) as function, \
                        patch.object(wrapper, "_as_cupy", side_effect=AssertionError("pointer reached")) as pointer:
                    with self.assertRaisesRegex(ValueError, "CuPy current device must match"):
                        wrapper.fused_add_layernorm(**on_device(tensor_device))
                    self.assertIsNone(wrapper._CUDA_DEVICE)  # A refused call cannot bind the cache.
                    allocation.assert_not_called(); function.assert_not_called(); pointer.assert_not_called()
                    cp.cuda.runtime.setDevice.assert_not_called(); cp.cuda.Device.assert_not_called()

    def test_cached_device_affinity_rejects_later_context_change(self):
        for wrapper in WRAPPERS:
            cp = cupy_metadata(1)
            with self.subTest(entry=wrapper.__name__), patch.object(wrapper, "_cupy", return_value=cp), \
                    patch.object(wrapper, "_CUDA_DEVICE", 0), \
                    patch.object(wrapper.torch, "empty_like", side_effect=AssertionError("allocation reached")), \
                    patch.object(wrapper, "_fn", side_effect=AssertionError("kernel reached")):
                with self.assertRaisesRegex(ValueError, "cached CUDA functions belong to another device"):
                    wrapper.fused_add_layernorm(**on_device("cuda:1"))
                self.assertEqual(wrapper._CUDA_DEVICE, 0)
                cp.cuda.runtime.setDevice.assert_not_called()

    def test_direct_pointer_context_checked_before_borrowing_memory(self):
        for wrapper in WRAPPERS:
            value, cp = tensor(device="cuda:1"), cupy_metadata(0)
            with self.subTest(entry=wrapper.__name__), patch.object(wrapper, "_cupy", return_value=cp), \
                    patch.object(wrapper, "_CUDA_DEVICE", None):
                with self.assertRaisesRegex(ValueError, "CuPy current device must match"):
                    wrapper._as_cupy(value)
                value.data_ptr.assert_not_called()
                cp.cuda.UnownedMemory.assert_not_called()
                cp.cuda.runtime.setDevice.assert_not_called()

    def test_context_binding_accepts_same_device_and_does_not_switch_it(self):
        for wrapper in WRAPPERS:
            cp = cupy_metadata(0)
            with self.subTest(entry=wrapper.__name__), patch.object(wrapper, "_cupy", return_value=cp), \
                    patch.object(wrapper, "_CUDA_DEVICE", None):
                self.assertIs(wrapper._cuda_for_tensor(tensor()), cp)
                self.assertIs(wrapper._cuda_for_tensor(tensor()), cp)
                self.assertEqual(wrapper._CUDA_DEVICE, 0)
                cp.cuda.runtime.setDevice.assert_not_called()


class CPUFallbackTests(unittest.TestCase):
    def check(self, x, residual, weight, bias):
        summed = x if residual is None else x + residual
        expected = F.layer_norm(summed, (summed.shape[-1],), weight, bias)
        for wrapper in WRAPPERS:
            with self.subTest(entry=wrapper.__name__), \
                    patch.object(wrapper, "_cupy", side_effect=AssertionError("CPU imported CuPy")), \
                    patch.object(wrapper, "_fn", side_effect=AssertionError("CPU launched kernel")), \
                    patch.object(torch.cuda, "is_available", side_effect=AssertionError("CPU probed GPU")):
                actual, total = wrapper.fused_add_layernorm(x, residual, weight, bias)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(total, summed, rtol=0, atol=0)

    def test_regular_cpu_residual_and_no_residual(self):
        x = torch.arange(6, device="cpu", dtype=torch.float32).view(2, 3)
        weight, bias = torch.ones(3, device="cpu"), torch.zeros(3, device="cpu")
        for residual in [torch.ones_like(x), None]:
            self.check(x, residual, weight, bias)

    def test_cpu_broadcasting_and_promoted_residual_preserved(self):
        x = torch.tensor([[1.], [2.]], device="cpu")
        residual = torch.tensor([[1., 2., 3.]], device="cpu")
        self.check(x, residual, torch.ones(3, device="cpu"), torch.zeros(3, device="cpu"))
        self.check(x.half(), residual.float(), torch.ones(3, device="cpu"), torch.zeros(3, device="cpu"))

    def test_cpu_optional_affine_and_mixed_parameter_dtype_preserved(self):
        x = torch.arange(6, device="cpu", dtype=torch.float32).view(2, 3)
        self.check(x, None, None, None)
        self.check(x.half(), None, torch.ones(3, device="cpu"), torch.zeros(3, device="cpu"))

    def test_large_last_dimension_and_float64_cpu_fallback(self):
        for size, dtype in [(1025, torch.float32), (3, torch.float64)]:
            self.check(torch.arange(size, device="cpu", dtype=dtype).view(1, size), None,
                       torch.ones(size, device="cpu", dtype=dtype), torch.zeros(size, device="cpu", dtype=dtype))

    def test_empty_cpu_fallback(self):
        self.check(torch.empty((0, 3), device="cpu"), None, torch.ones(3, device="cpu"), torch.zeros(3, device="cpu"))


class HistoricalTests(unittest.TestCase):
    def test_historical_wrapper_matches_both_recorded_git_paths(self):
        frozen = (ROOT / "historical/4af3e68/fused_ln_cuda.py.txt").read_bytes()
        self.assertEqual(hashlib.sha256(frozen).hexdigest(), LEGACY_SHA)
        manifest = json.loads((ROOT / "historical/4af3e68/source.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["source_commit"], BASE)
        self.assertEqual(manifest["snapshot_sha256"], LEGACY_SHA)
        self.assertEqual(set(manifest["unchanged_git_blobs"]), set(PROTECTED))
        for path in ["fused_ln_cuda.py", "three_way/fused_ln_cuda.py"]:
            self.assertEqual(subprocess.check_output(["git", "show", BASE + ":" + path], cwd=ROOT), frozen)

    def test_kernels_benchmarks_and_results_are_unchanged(self):
        manifest = json.loads((ROOT / "historical/4af3e68/source.json").read_text(encoding="utf-8"))
        for path in PROTECTED:
            self.assertEqual(subprocess.check_output(["git", "rev-parse", BASE + ":" + path], cwd=ROOT, text=True).strip(), manifest["unchanged_git_blobs"][path])
            checked = subprocess.run(["git", "diff", "--exit-code", BASE, "--", path], cwd=ROOT, capture_output=True)
            self.assertEqual(checked.returncode, 0, path)


if __name__ == "__main__":
    unittest.main()
