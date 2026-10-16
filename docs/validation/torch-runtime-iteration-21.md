# Torch runtime 0.21 candidate: shape contracts and graphless views

This is an unpublished candidate on the development branch. The refinement below supersedes the initial microbenchmark costs. Main remains on
0.20.0. Complete catalog, exported application and remote adoption gates are
pending; the measurements below do not establish an application speedup.

## Behavior

Shape methods and their functional entries validate integer dimensions, ranges,
permutation rank/duplicates, expansion sizes and scalar/empty shapes. Singleton
strides and Python object identity follow the tested native behavior. Empty JVP
tangents retain their own layout. A scalar/one-dimensional `.T` is a distinct
view; unsupported deprecated cases retain warnings.

A graphless view follows the ultimate Tensor base's gradient flag. Reporting
`requires_grad=True` does not create a reverse edge. Explicit
`requires_grad_()` promotes a leaf accumulator, and changing a base flag never
invents a missing connection. Stopped view inputs neither accumulate gradients
nor invoke hooks; other inputs to a joint VJP retain their gradients.

Nonoverlapping singleton expansions remain valid checked `copy_` destinations.
Expansion results retain a read-only public descriptor; duplicate-address
writes remain rejected before changing data or storage versions.

## Local validation

There are 1,333 new native-reference cases and 42 independent cases. The latter
include coordinate maps, adjoints, finite directional derivatives, VJP/JVP,
checkpoint/resume, graphless view flags, joint gradients/hooks and alias writes.

The pre-optimization source passed 22,644 infrastructure tests with one optional
Docker skip; it also passed 719 contracts in actual Node/Pyodide and Chrome
Workers and 739 installed-wheel tests on Python 3.10.19/NumPy 1.26.4 without
native Torch. Its wheel/sdist source and license bytes matched the snapshots.
The optimized source passes 22,644 infrastructure tests with one optional Docker
skip and 17 warnings (211.04s), plus the 3,185 focused shape/layout/copy/import
tests. Fresh actual Node/Pyodide and Chrome Worker runs each pass 719 contracts;
the independently installed minimum-version wheel passes 739 tests. Source and
license hashes match the optimized timing snapshot, browser VM and wheel/sdist.
The maintained CPU and WASM benchmark entry points each execute all 17 cases.
The current runtime remains 40 Python modules and two license resources.

The first full dynamic-view snapshot had one failing singleton-expansion copy
case (22,642 passed). Its log and source snapshot are retained. The writeable
ndarray backing fix restored checked copying without enabling overlapping writes.
Earlier fixture mistakes, empty tangent layout failures and no-grad gradient-edge
mismatches also remain in the evidence; no numerical tolerance was relaxed.

## Controlled comparison against 0.20

Both phases freeze the final 0.20 source and candidate source, Python/JS harnesses
and license files. Each phase runs sixteen fresh processes sequentially: two
runtimes, two dtypes, and baseline/candidate/candidate/baseline order. Each of the
seventeen cases retains eleven timing batches (1,200 calls, or 120 for reverse
and JVP) after three warmups. Inputs are rehashed before and after every process.
All processes completed successfully. The machine is shared; requested thread
limits do not establish the actual BLAS thread count.

The initial candidate roughly doubled expansion latency and substantially slowed
permutation/transpose. The optimized candidate moves inverse-permutation work
into the pullback, avoids a second permutation validation in transpose, and uses
NumPy's bounds-checked contiguous-buffer descriptor for singleton stride changes.
All initial samples and regressions are retained.

The optimized phase has these relative time changes, across float32/float64:

| Workload | CPU | WASM |
| --- | --- | --- |
| Transpose | 8–12% faster | 9–10% faster |
| Permute | 21–26% slower | 19–22% slower |
| Squeeze | 23–32% slower | about 29% slower |
| Flatten | about 25% slower | 24–25% slower |
| Expand | 51–55% slower | about 58% slower |
| Bare storage view controls | 10–15% slower | 10–12% slower |
| Shape-chain matching JVP | 18–20% slower | 17–18% slower |

Complete shape-chain reverse passes are near parity. Common construction and
reverse controls retain all samples, including noise and regressions. Validation
and native view metadata still have a measurable cost. This is a compatibility
candidate awaiting broader integration evidence, with no overall speedup claim.

The maintained `bench_shape_ops.py` is byte-identical to the measured Python
harness. CPU and WASM wrappers accept `--benchmark shape_ops`. Timings cover only
mutually successful values, aliases and derivatives; corrected scalar/empty and
no-grad semantics are excluded. There is no native-Torch throughput, whole-site
latency, cold-start or peak-memory claim.

## Integration boundary

Ordinary `import torch` remains supported through `paperleet-torch script.py`
or a host call to `install_as_torch()` before loading user code. The independent
library remains Apache-2.0, NumPy-only, and separate from native PyTorch's import
installation. No package has been published to PyPI. No 0.21 browser catalog has
been exported or published, and all historical published assets remain untouched.

Machine-readable local evidence is in
[`iteration-21.json`](../../packages/paperleet-torch/benchmarks/results/iteration-21.json);
source snapshots, harnesses, all timing samples and rejected logs are in
[`iteration-21-harness.tar.gz`](../../packages/paperleet-torch/benchmarks/results/iteration-21-harness.tar.gz).

## Refined metadata path and direct gradient flags

A dedicated single-input metadata result avoids general multi-input graph setup.
Expansion descriptors use the already validated geometry without a second NumPy
broadcast pass. Direct `requires_grad` assignment now has native bool/leaf rules,
including explicit promotion and restrictions after backward graph release.

Two additional balanced sixteen-process phases retain all samples. The final
property phase reduces CPU bare-view latency by25–28% and WASM by22–27%; expansion
latency falls15–18% on CPU and13–14% on WASM versus0.20. Shape-chain JVP is near
parity or faster. Ordinary reverse controls retain1–3% CPU and4–6% WASM costs.
These remain local microbenchmarks with the same limits described above.

The refined source passes3,547 focused contracts,721 each in real Node/Chrome,
and741 minimum-wheel cases. Direct assignment audit differences fall14/16 to0/16.
Native catalog passes78problems/10,218executions/3repeats/342mutants,184.04s.
The full infrastructure run passes22,666 numerical/other cases and fails one
structural AST check that mistook propertygetter/setter for method shadowing.
The check now allows one getter/setter/deleter per property and still rejects
repeated definitions of each kind; all39 contracts pass after the correction.
A fresh complete merged run remains pending. Only `_api.py` indentation differs
between final timed and runtime snapshots; their ASTs match and all other source
and license bytes are identical.

[Refinement evidence](../../packages/paperleet-torch/benchmarks/results/iteration-21-view-path.json)
and [source/harness archive](../../packages/paperleet-torch/benchmarks/results/iteration-21-view-path-harness.tar.gz)
retain both phases and the rejected structural run.
