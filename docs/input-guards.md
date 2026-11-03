# Raw-pointer input boundary

Both maintained wrappers now validate assumptions before their fused raw-pointer
path reshapes tensors, allocates outputs, imports CuPy or dispatches a function.
The kernel code and launch arguments for valid metadata were unchanged by that
2026-10-12 change. This is an input-reliability change, not a GPU numerical or
performance result.

**2026-11-03 update.** The scalar launch arguments were later found to be wrong
in type: `float(eps)` was packed by CuPy as float64 while the kernel reads a
4-byte `const float`, so the kernel saw eps of about -1.37e-33. The wrappers now
pass `np.int32` / `np.float32` scalars (`_scalar_args`), and the suite checks
each scalar's type against the `kernel.cu` signature. The hashes in
`input-guard-checks.json` describe the 2026-10-12 wrapper and are kept as that
dated record.

The raw path requires a nonempty CUDA input with a supported float16/float32
dtype, last dimension 1–1024, and matching weight/bias dtype. Both affine tensors
must be on x's device with exact shape `(D,)`. A residual must be a tensor on
that device with x's dtype and **original shape**, not merely the same element
count after flattening. Violations raise explicit ValueError/TypeError before
pointer conversion. Checks do not use assert and remain effective under `-O`.
The direct pointer adapter additionally refuses CPU, unsupported-dtype and
noncontiguous tensors before CuPy.

Input device agreement alone is insufficient: CuPy's current device must also
match x's concrete device index before allocation, and each direct pointer
adapter checks its tensor's device before borrowing memory. This implementation
does not switch contexts. The existing single module/function cache is bound
under a lock to the first accepted device in the process; subsequent device
changes fail closed rather than reusing another device's cached handles. Use a
fresh process for another device. Module construction also observes this binding.
Offline fake-CuPy tests cover wrong-current-device rejection, no `setDevice` /
`Device` calls, no binding on rejection, GPU0-compatible metadata, cache-affinity
refusal, and adapter refusal before `data_ptr` / memory wrapping.

Paths outside that raw candidate still use the original PyTorch fallback.
CPU broadcasting, residual dtype promotion, mixed affine parameter dtype and
optional weight/bias remain supported where PyTorch supports them. CuPy is
loaded lazily, so CPU fallback needs only PyTorch. CUDA execution still requires
a compatible CuPy installation; this change does not install one. Empty input
uses the fallback instead of attempting a zero-grid raw launch.

The old actual entrypoint was tested with metadata mocks and a fake pointer/
dispatch backend. CPU weight, residual dtype mismatch, equal-numel but different
original residual shape, and short weight all reached that backend. No real
CUDA tensor, device probe, GPU allocation, CuPy runtime, NVRTC compilation or
kernel was used. The regression suite exercises both current entrypoints and
checks that invalid metadata is rejected before those boundaries. Paired valid
metadata checks retain the original grid/thread/stride/eps/residual values
(since 2026-11-03 as 4-byte NumPy scalars, see above).
Real tiny CPU tensors compare fallback results exactly against PyTorch.

These tests establish routing, explicit device/context refusals and CPU semantics. They do not establish GPU
memory safety in arbitrary execution contexts, multi-GPU/cache or stream
compatibility, new GPU numerical behavior, or performance. There was no GPU
rerun; guard/import overhead is unmeasured. Historical single-pass variance
cancellation/NaN limits and the T4 CUDA/Triton losses remain unchanged.

The last recorded wrapper source is frozen as a non-entry-point `.py.txt` from
`4af3e689dc912754061d247991b3f73e77d60a89`. Both original wrapper paths had exactly
those bytes (SHA256 `ed5389417ef8b94e4ad2c09e63fa3c54dbef326e8728164ccef90c843ff9b9e2`).
It is not imported/executed and is only historical source evidence. New wrappers
must not be attributed to the old measurements. Tests compare the snapshot
against both recorded Git blobs and protect the existing kernel and result
paths. The three benchmark drivers were edited on 2026-11-03 (accuracy gate,
end-to-end warm-up); their recorded blobs are still verified, but they are no
longer required to match. No old measurement is regenerated or rewritten.

Run locally with an existing CPU PyTorch installation:

```bash
python -B -m unittest discover -s tests -v
python -O -B -m unittest discover -s tests -v
```

The CPU workflow pins Torch 2.7.1 on Python 3.11 and fetches history for the
source audit. It first passed on GitHub Actions on 2026-10-12 (run 38204644101,
the merge of PR #1); since 2026-11-03 it runs on ubuntu-latest and windows-latest.
The 2026-10-12 local run used Torch 2.13.0+cpu/Python 3.13.
Actual counts, source fingerprints and audit boundaries are in
[input-guard-checks.json](input-guard-checks.json).
