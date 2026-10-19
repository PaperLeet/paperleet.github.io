# StrideTorch 0.81 candidate: implementation and initial distribution checks

The local 0.81 candidate integrates the packed-row optimization selected in the
[repeated WASM qualification](torch-runtime-matmul-packed-rows-qualification.md).
It is **not yet adopted on main or published**. Current PaperLeet application,
browser export/Worker and immutable-asset qualification remain outstanding.

Eligible 2D WASM matrix products now present contiguous reduction rows to the
existing `einsum` kernel. The qualified shape/dtype/exceptional-range guards,
batched implementation, native CPU routing and output layout contract remain
unchanged. Existing `import torch` aliases and CLI entry points stay available.
Packing can introduce temporary operand copies; the qualification explicitly
reports the individual-kernel memory cost as well as full-call peaks.

The final `_matmul.py` differs from the measured candidate only in comments and
formatting: their Python ASTs are identical. The final `__init__.py` changes only
the version from 0.80 to 0.81; normalizing that literal gives identical ASTs.
The other 63 runtime/license resources match byte-for-byte. The previous
three-process Node/Chrome numerical and timing evidence therefore applies to
this computational expression; it is not a fresh final-version browser export
or application check.

Initial distribution checks pass:

- Public and private application wheels and source archives build locally.
  Both wheels rebuild from their actual source archives. All six artifacts
  retain the exact 65 final-version resource hashes.
- The standalone package keeps Apache-2.0 metadata, NumPy as its only runtime
  dependency and the existing CLI. It owns no top-level `torch` or `paperleet`
  package. The application uses its separate private namespace.
- Both actual wheels install together in an isolated target. Public/private
  HVP, unchanged caller gradients, nested alias restoration with native Torch,
  and forced-WASM forward/two-VJP smoke checks pass.
- The installed 0.81 public wheel passes **12225** full SDK tests on NumPy 2.5.4.
  It passes **12222**, with the expected **3 skips**, on NumPy 1.26.4.
  Parent and child processes explicitly verify the installed 0.81 path.

The first installed-SDK harness (358) only inserted the installation target in
the parent process. Two subprocess import tests consequently failed in the
native environment; the minimum environment could instead find an older
installed package. Both original results are preserved. The fresh 359 harness
propagates the installation target through `PYTHONPATH` and verifies the child's
version and exact path before the unchanged full suites. Both corrected suites
pass without test or tolerance changes.

Next, qualify current application CPU/catalog tests, generate and verify final
browser assets, run the actual WASM catalog/Worker budgets and required frontend
build checks, reconcile moving main, then perform guarded adoption. No CI,
tag, package publication or deployment is initiated by this candidate stage.

[Machine-readable package/core result](../../packages/stridetorch/benchmarks/results/candidate-0.81-package-core.json).
The [raw package/core archive](../../packages/stridetorch/benchmarks/results/candidate-0.81-package-core-harness.tar.gz)
contains 1153 verified files, 5,016,121 bytes, SHA-256
`a39733729c36bb4fbf17c28e2159d817242d5b4cb4902f2359bf5207c291f64e`.
It includes actual distributions, rebuilt wheels, installed sources, scripts,
test outputs and the original failure. External environment paths require
adaptation for replay. The runtime version alone does not imply completion
of application integration or the broader independent-library objective.
