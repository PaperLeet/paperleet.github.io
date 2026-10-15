# Lightweight torch 0.12 validation

This milestone fixes sequence dtype inference and scalar storage, exposes
`torch.promote_types`, and reduces common small-sequence/NumPy-scalar allocation
costs. NumPy remains the only runtime dependency; ordinary torch import aliases
and PaperLeet's isolated Worker integration remain available.

## Construction and promotion contracts

Previously `[np.float64(...)]` became float32 and `[np.int16(...)]` became int64.
Sequences now recursively infer Python/NumPy scalar and single-value Tensor
dtypes, preserving NumPy precision/widths. A float64 value `1 + 2**-40` and its
analytical gradient survive construction exactly. Copying a sequence owns data
and detaches source Tensor history; `as_tensor` of a sequence also owns a copy.
`new_tensor` inherits its owner's dtype unless overridden.

`promote_types(type1, type2)` follows supported Torch category/width rules, rather
than NumPy's stronger integer-to-float widening. Both positional and native
`type1=`/`type2=` arguments work; missing, duplicate or invalid arguments fail.
Native references cover all 196 supported dtype pairs, including wider unsigned
promotion boundaries. Floating promotion with wider unsigned types is accepted;
unsupported integer/complex combinations fail. Ordered sequence inference keeps
native maximal-complex behavior while still checking subsequent storage values.
Ordinary float16/integer/unsigned aliases and `half`, `short`, `char`, `cfloat`,
`cdouble` expose storage types without promising every kernel/AD rule for them.

Scalar storage follows Python numeric protocols and range checks. In particular,
checked sequence float→integer storage differs from ndarray casts: non-finite or
out-of-range sequence values fail, permitted negative integer→unsigned values
wrap, and NumPy scalar integer conversion honors the runtime's index protocol.
NumPy uint64 scalar→uint64 storage raises native TypeError, whereas explicit
uint64 Python integers and direct uint64 array imports work. Explicit array
float→integer casts retain their separately documented platform-dependent bounds.
Factory conversions isolate and restore NumPy floating-point error settings.

First-position zero-dimensional ndarrays, ragged input and scalar Tensor shape
rules are checked independently of NumPy's array-shape inference. Nested arrays
are copied as sequence values, including negative-stride/foreign-endian arrays;
direct ndarray imports retain 0.11's strict layout/dtype validation and independent
metadata descriptors. Those array import contracts have not been weakened.

Small homogeneous builtin float/int/bool lists and tuples allocate directly.
Primitive vectors/rows infer/check in batches, while other nested/mixed inputs
retain ordered recursive handling. NumPy numeric scalars create private array
descriptors without an extra copy; even a tested float64 subclass overriding its
array protocol retains independent scalar storage. No unsafe dtype/layout cache
is introduced.

The [upstream construction source](https://github.com/pytorch/pytorch/blob/main/torch/csrc/utils/tensor_new.cpp)
and [scalar storage rules](https://github.com/pytorch/pytorch/blob/main/torch/csrc/utils/python_scalars.h)
guide independent Torch 2.14.1 checks. Global default-dtype changes, broader array
protocol inference, the full legacy Tensor constructor and general higher-order
reverse graphs remain separate work. Default floating construction stays float32.

## Performance protocol

Baseline is released 0.11 commit `29201db58be1d81258c5f1434b22a62f5b4c0824`.
The baseline's 32 Python files and candidate's 33 files have frozen hashes, plus
two separately hashed license resources. The maintained benchmark harnesses are
also hashed and verified after measurement. All local correctness/build jobs are
terminal before timing; timing jobs run sequentially without source/harness edits.

CPU uses Python 3.12.11/NumPy 2.5.4 on macOS arm64/Apple Accelerate. OMP/MKL/
OpenBLAS/vecLib controls precede NumPy imports, and the driver selects/queries
Accelerate's single-thread policy before/after. CPU uses baseline→candidate→
candidate→baseline for each dtype and suite: 24 fresh processes. There are 12
fresh Node WASM processes and four separate memory processes. Real WASM uses
Pyodide 0.29.4/NumPy 2.2.5. Chrome 155 uses the same balanced order, starting an
entirely fresh browser process and Worker for each of eight samples.

There are 11 batches and three warmups. Core cases use 120 calls per batch,
hook controls 1,200, and construction 1,200. Small scalar/list/identity cases use
12,000, NumPy export/protocol cases 120,000, and 4,096-element cases 30. Every
result retains its actual counts and all samples. Fixture construction/imports
are outside timing; complete operation/allocation/temporary cleanup is included.
The 24 construction cases check values and dtype before timing and report equal
shape/dtype contracts across versions. Builtin defaults do not change with the
separate array fixture dtype.

The initial per-element implementation was much slower: a 4,096-integer CPU
construction took about 2.17 ms versus 0.069 ms baseline in that exploratory
protocol. Batch paths reduced this to about 0.101 ms. Both exploratory CPU phases
and their 14-case harness remain separate from final 24-case measurements. Verified
reverse patches reconstruct both 33-file intermediate sources; their temporary
0.11 version metadata does not identify released 0.11 code.

## Results and costs

Chrome process-median ranges, microseconds per complete call:

| Case | 0.11 float32 fixture | 0.12 float32 fixture | 0.11 float64 fixture | 0.12 float64 fixture |
| --- | ---: | ---: | ---: | ---: |
| Small integer list | 7.90–8.00 | 5.53–5.56 | 7.90–7.91 | 5.53–5.63 |
| Small boolean list | 6.08–6.16 | 5.46–5.48 | 6.12–6.17 | 5.45–5.51 |
| NumPy scalar | 5.63–5.69 | 5.15–5.22 | 5.67–5.69 | 5.07–5.29 |
| Small float list | 5.34–5.38 | 5.44 | 5.32–5.34 | 5.43–5.57 |
| Python float list, 4,096 | 163.3–166.7 | 213.3–216.7 | 163.3 | 206.7 |
| Python integer list, 4,096 | 196.7 | 276.7–283.3 | 196.7 | 276.7–283.3 |
| Explicit float list, 4,096 | 163.3 | 210.0–213.3 | 160.0–163.3 | 206.7–210.0 |
| MLP training | 144.2–145.8 | 140.0–141.7 | 137.5–139.2 | 137.5–140.0 |

Small integer construction is about 29–30% faster in Chrome, boolean construction
about 11%, and NumPy scalar construction about 8–9%. CPU small integers improve
about 11%, booleans 7–9% and NumPy scalars 6–9%. Small float-list Chrome means
increase about 1.5%/3.1%; ordinary scalar paths are not a general speedup claim.

Checked larger Python sequences still cost more. Chrome's 4,096-float construction
increases about 27–30%, integers about 42%, and explicitly typed floats about 29%.
CPU large floats increase about 29–32% and integers 44–47%. These costs include
the stronger type/range/protocol work and remain optimization targets. Final
comparisons use identical valid values/dtypes, rather than comparing corrected
float64 results with an incorrectly narrowed baseline result.

Chrome attention changes −0.4%/−0.2%, MLP −2.9%/+0.3%, AdamW −1.6%/+1.5%, graph
backward −0.6%/+0.8% and retained backward −0.5%/+2.7% (float32/float64 mean ratios).
CPU MLP is about 1% faster and attention/AdamW stay near parity. These controls
do not establish a general kernel/optimizer throughput improvement. The roughly
100-µs AdamW excursions seen in earlier milestones do not recur in this series;
their cause remains unresolved. No sample is discarded. Zero-clock-resolution
identity controls carry no percentage claim.

The retained-loss memory fixture is byte-identical to 0.11: float32/float64 retain
54,570/86,550 traced bytes after default backward, with peaks 668,690/1,290,502.
It traces 16 stages over 64×128 inputs allocated before tracing. Explicit-retention
results also match; this is not browser RSS or a claim about every graph workload.

## Validation and independent distribution

Final source passes **4,733 tests**, one optional Docker skip and seven native
warnings in 162.21 seconds. This includes all 71 exported NumPy reference/learner
suites and all 312 mutants, with unsupported APIs excluded from mutation kills.
The construction-focused run passes **2,045 tests**, including native scalar/type
matrices and seven independent NumPy-only workflows.

Real pinned WASM passes **108 lifecycle contracts**. The maintained gate passes
seven complete reference/mutant/learner suites. Seven fresh Chrome Workers pass
54 original tests, catch all 37 representative mutants and run all 108 contracts.
Version 0.12.0, actual import location, every exported source hash and LICENSE/
NOTICE are checked. Web checks and production build pass.

The standalone wheel installed outside the checkout passes **128 tests** on
NumPy 1.26.4 without native Torch. It is **110,606 bytes**, SHA-256
`f894800689cf9475dfe9c99672f8bb6e5aaec464bf4e7339451748d445299f19`.
All 33 wheel, installed and sdist source hashes match timed source. Both canonical
Apache-2.0 LICENSE/NOTICE copies are verified in the sdist, wheel and browser
profiles; application private-vendor distribution checks also pass.

All 4,222 published hashed assets survive unchanged. Only 423 untracked,
unpublished, unreachable intermediate assets were pruned. Earlier 4,720/4,729/
4,732-test phases remain separately labeled. Full samples, source/resource/
harness hashes, audits, reconstruction proofs and check receipts are in
[iteration-12.json](../../packages/paperleet-torch/benchmarks/results/iteration-12.json).
The reverse patches reproduce the [initial prototype](../../packages/paperleet-torch/benchmarks/results/iteration-12-prototype.patch)
and [batch prototype](../../packages/paperleet-torch/benchmarks/results/iteration-12-optimized-prototype.patch).
Full local logs and frozen artifacts remain under
`paperleet-wasm-evaluation-20261015/torch-iteration-12`.

The independent-library/core-coverage goal remains active. Further work includes
large-sequence costs, consistent promotion in other core operators, remaining
view/gradient lifecycle bounds and broader NN coverage. Resident WebGPU remains
an evaluated optional backend; this milestone adds no GPU execution mode and
makes no PyPI publication.

## Integration with current main

The focused implementation is rebased onto
`18e33077a0673528b40247b1d9668885d70020e1`. Upstream changes since the original
validation base affect frontend discovery, quotations, navigation and browser
bundle cache recovery. Numerical library/application sources, curriculum,
Python tests, exported data and the maintained WASM gate are unchanged. All 33
timed source hashes still match. The combined frontend check suite and production
build pass again after integration, including cached-bundle recovery checks.

## Remote quality, deployment and exact live verification

Release commit `9b1822ee0e307e5b1d5e5cafc160514bd54a3a9b` is adopted on main.
Its [quality run](https://github.com/PaperLeet/paperleet.github.io/actions/runs/38799534042)
and [Pages run](https://github.com/PaperLeet/paperleet.github.io/actions/runs/38799533989)
both succeed. The hosted full suite passes 4,733 tests (one optional Docker skip,
seven warnings) in 475.51 seconds; native CI checks all 71 modules with three
repetitions, 9,372 executions and all 312 mutants in 400.67 seconds. The seven
maintained pinned WASM suites pass on the same commit.

An independent verifier compares the live `https://paperleet.com/` compact index,
all 71 starters and every referenced manifest against expected artifacts frozen
from this exact release. Both runtime bundles match byte-for-byte: autograd is
339,988 bytes / SHA-256
`244c25bc1c00c4682f678c88404552fa7249e90d51f608cb69c3bca79c8e4c64`;
legacy is 35,895 bytes / SHA-256
`5ef028c8b739a20fe7a83c0861ea279eaaf8416519111e1ddf0394befa359fe8`.
Version 0.12.0 and canonical Apache-2.0 LICENSE/NOTICE match in both profiles.
Local receipts, frozen expectations and verifier are retained in iteration 12.
