# Lightweight torch 0.11 validation

This milestone validates NumPy import/storage boundaries, isolates mutable array
metadata and reduces builtin scalar/small-float-sequence construction overhead.
The NumPy-only CPython/Pyodide library retains existing torch aliases and bounded
reverse/forward differentiation contracts.

## Behavior and boundaries

`tensor`, `as_tensor`, `from_numpy` and `Tensor.new_tensor` check external NumPy
types/layouts before casts or copies. String/bytes/object/date/time/record/void
arrays and NumPy longdouble/clongdouble descriptors are rejected. Storage accepts
bool, ordinary signed/unsigned integers, float16/32/64 and complex64/128. This does
not extend every kernel or differentiation rule to every storage dtype. Actual
Torch 2.14.1 executions verify accepted/rejected types, including unsigned types
not exhaustively listed on the public import page. Diagnostics identify the
NumPy scalar type even when an extended descriptor prints as float64.

Foreign byte order, negative strides and strides not divisible by element size
raise ValueError. Checks include singleton, empty, complex and multidimensional
layouts whose contiguous flags alone would incorrectly permit invalid strides.
Legal Fortran/transposed/strided, zero-stride broadcast, unaligned, scalar and
empty arrays retain their values. Validation reads current metadata on every
import; no array validation cache can conceal subsequent dtype/stride changes.
The stride predicate combines sign/alignment bits for supported power-of-two
sizes, reducing generator overhead without dropping checks.

Copied factories own independent data; sharing imports retain data pointers.
Source/exported ndarray shape, dtype, strides and writeability descriptors are
independent from the Tensor. This fixes a demonstrated case where changing a
NumPy descriptor could reshape a Tensor, turn its float32 dtype into Unicode or
make a previously writable Tensor fail. `numpy()` and the NumPy array protocol
export views with separate metadata, without copying data. Internal result,
scalar and copied arrays already have private descriptors and avoid redundant
views. The private path still validates numerical dtype before storage tracking.

Read-only imports warn once across import APIs; copying factories do not warn.
Writes through initially read-only backing storage remain outside the supported
mutation contract. The NumPy-oriented Tensor constructor/internal views retain
their layout conveniences; the legacy positional shape allocator is not provided.
`new_tensor` inherits/overrides its owner's dtype and copies into an independent
leaf, detaching source history. Model dtype failures precede state changes.

Builtin float/int/bool construction and small homogeneous builtin-float
lists/tuples avoid an intermediate array/copy. Mixed, nested and large sequences
retain the generic path. Large default Python integers use native ValueError
behavior; weak uint64 operator scalars instead narrow according to the receiving
integer dtype. Native add/subtract/multiply checks cover both operand orders.

Factory floating casts isolate NumPy error settings and restore the caller's
settings. Defined finite float→integer values truncate toward zero. An initial
minimum-NumPy test incorrectly required one out-of-range integer sentinel;
NumPy 1.26 ARM and NumPy 2.5/native reference differed. The failed expectation is
preserved. The [official conversion contract](https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.to.html)
explicitly leaves out-of-range float→integer results undefined/platform dependent.
Tests compare defined values and error-mode restoration without inventing a
portable sentinel; no implementation change emulates one undefined output.

The [tensor](https://docs.pytorch.org/docs/2.14/generated/torch.tensor.html),
[as_tensor](https://docs.pytorch.org/docs/2.14/generated/torch.as_tensor.html) and
[from_numpy](https://docs.pytorch.org/docs/2.14/generated/torch.from_numpy.html)
interfaces guide native-reference checks. Python sequence promotion with NumPy
scalars, broader array-protocol inference, general higher-order reverse graphs
and the full legacy constructor remain separate compatibility work.

## Performance protocol

Baseline is released 0.10 commit `ac716bac9bfcd50e0d3ab530dc72deb264f50b2d`.
The initial 0.11 candidate `f105ad3617fe016b5788b067a9e5fe6d0d9f7265` and revised
metadata-safe source are measured under the same final protocol. Each variant
has 32 frozen Python hashes and two license-resource hashes. Correctness/build
producers are terminal before timing; all timing processes run sequentially.

CPU uses Python 3.12.11/NumPy 2.5.4, macOS arm64/Apple Accelerate. The maintained
CPU driver sets OMP/MKL/OpenBLAS/vecLib controls before importing NumPy and selects
Accelerate's single-thread policy, querying it before/after each process.
CPU order is baseline→initial→revised→revised→initial→baseline per dtype/suite.
There are 36 CPU reports and 18 fresh Node WASM reports. WASM uses Pyodide
0.29.4/NumPy 2.2.5. Chrome 155 runs the same balanced order with an entirely fresh
browser process and Worker per sample: 12 processes, two per variant/dtype.
Loaded source/resource hashes are checked in each report.

All suites use 11 batches and three warmups. Core cases use 120 calls per batch,
callback controls 1,200, and construction 1,200. Small scalar/list/identity cases
use 12,000; the very small numpy/export protocol cases use 120,000. Reports retain
actual counts and every sample. Import/fixture construction is outside timings;
complete operation/allocation/temporary cleanup is included.

The first candidate's separate 24 CPU, 12 Node and eight Chrome results remain
in the raw record. They are not pooled with final three-variant measurements or
relabeled as revised-source results. A verified reverse patch reconstructs all
32 intermediate files from the final source.

## Results and costs

Chrome process-median ranges, microseconds per complete call:

| Case | 0.10 float32 | Revised 0.11 float32 | 0.10 float64 | Revised 0.11 float64 |
| --- | ---: | ---: | ---: | ---: |
| Builtin float | 5.39–5.43 | 4.71–4.79 | 5.41–5.49 | 4.74–4.78 |
| Small float list | 5.55–5.66 | 5.18–5.27 | 5.54–5.65 | 5.21–5.24 |
| NumPy copy, 64×128 | 5.25–5.42 | 6.33–6.42 | 5.83–6.08 | 7.00–7.17 |
| NumPy shared import | 2.42 | 3.42 | 2.33–2.42 | 3.25–3.33 |
| NumPy cast | 7.00–7.08 | 10.17 | 6.75–6.83 | 9.67–9.83 |
| new_tensor | 2.08–2.17 | 3.42–3.50 | 2.58–2.67 | 3.92 |
| numpy export | 0.051–0.057 | 0.121–0.123 | 0.052–0.056 | 0.124 |
| MLP training | 136.7–141.7 | 140.0–141.7 | 132.5–134.2 | 136.7–140.0 |

Builtin scalar construction is approximately 11–16% faster and the small float
list about 7% faster than 0.10. The initial candidate's list path was 7.5–7.7 µs;
the revised path is about 31% faster than that candidate. Existing-Tensor
as_tensor identity remains near parity. Metadata export adds roughly 0.07 µs in
Chrome, copying a descriptor while retaining shared data.

Checked imports/casts still cost more: shared import adds roughly 0.9–1.0 µs,
array copy about 1.1 µs, and dtype-changing import about 3 µs. Those checks and
NumPy error-state isolation are retained for correctness. CPU shared import
adds about 0.3 µs; small-list construction improves to roughly 2.2 µs versus the
initial candidate's 3.0–3.2 µs. CPU scalar means improve about 9–17%, with visible
process ranges. A float64 copy process reaches 5.89 µs versus another at 3.51 µs;
all samples remain, rather than selecting the faster run.

Chrome attention changes about +0.5%/+0.3% and MLP training +1.2%/+3.7%
(float32/float64 process-mean ratios). Plain graph backward is +1.2%/+0.7%,
retained-gradient backward +2.5%/+2.4%, and leaf transformation callbacks
+5.2%/+4.4%. CPU MLP means are about 2% slower; float64 graph/retention controls
are about 7.6%/8.8% slower in this series. These are real costs of richer semantics,
not evidence of a general kernel speedup. Sub-clock-resolution identity conversion
samples carry no percentage claim.

Float64 Chrome AdamW remains variable: baseline 301.7–310.8 µs, revised 306.7 and
406.7 µs (a 16.5% averaged increase). Similar roughly 100-µs excursions previously
occurred in baseline and initial-candidate processes, including the first 0.11
series. The cause remains undetermined. No sample is discarded and no broad
optimizer improvement is claimed.

The retained-loss memory fixture is byte-identical to 0.10: after default backward,
float32/float64 retain 54,570/86,550 traced bytes with peaks 668,690/1,290,502.
It uses 16 stages on 64×128 inputs, excludes input allocation and does not call
retain_grad; it is not browser RSS or a measurement of every view/hook workload.

## Validation and distribution

The final implementation passes **2,646 tests**, one optional Docker skip and six
native warnings, 146.60 seconds, after integration with upstream SimCLR. This
includes every one of the 71 exported NumPy-adapter reference/learner suites and
all 312 mutants, with actual test
failures required rather than unsupported APIs. New coverage includes 260 native
input/metadata/sequence cases and seven independent NumPy-only workflows.
Focused lifecycle/boundary checks pass 465 tests.

Minimum NumPy 1.26.4 source checks pass 121 standalone tests without native Torch;
actual pinned WASM passes 101 lifecycle contracts with exact 32 source/two resource
hashes. The final installed wheel outside the checkout also passes **121 tests**,
is **106,668 bytes**, SHA-256
`c0583f0abc89acd31c909df618f219bcfd32c9b825fd09d4aa41b7347b39fbc6`.
All 32 wheel/installed source hashes match the timed revised source. Both wheel
and sdist retain canonical Apache-2.0 LICENSE/NOTICE, as do private application
copies and both browser profiles. The broader application license is unchanged.

The maintained real WASM gate executes seven complete representative original/
reference/mutant suites plus 101 contracts. Seven fresh Chrome Workers pass 54
original tests, catch all 37 mutants and run those 101 contracts, checking version
0.11.0, actual import location, every exported library source hash and notices.
Web checks and production build pass. The unchanged original native curricula
also have the preceding 0.10 exact-commit quality gate: 70 modules, 9,129 test
executions, three repeats and all 304 mutants caught. Its 70 numerical/curriculum
sources are unchanged upstream; added SimCLR separately passes 243 native
executions over three repeats, catching all eight mutants. Integration preserves
all 4,078 published hashed assets and all 32 timed library source hashes. The
earlier 2,645-test/six-Worker preintegration evidence remains separately labeled
in the raw record.

The 0.10 exact-commit [quality](https://github.com/PaperLeet/paperleet.github.io/actions/runs/38779719998)
and [Pages](https://github.com/PaperLeet/paperleet.github.io/actions/runs/38779720016)
runs succeed; its live 70-module starters/manifests and both runtime bytes/notices
were verified before this work. Complete raw samples, source hashes, audit inputs,
validation counts and exact local browser harnesses are in
[iteration-11.json](../../packages/stridetorch/benchmarks/results/iteration-11.json).
The [intermediate reverse patch](../../packages/stridetorch/benchmarks/results/iteration-11-intermediate.patch)
reproduces the first candidate. Full logs, previous expectations and frozen source
variants remain under `paperleet-wasm-evaluation-20261015/torch-iteration-11`.

The independent-library/core-efficiency goal remains active. Further work includes
sequence promotion, gradient payloads through metadata-only reverse paths,
remaining callback/retention costs, unexplained AdamW variation and broader NN
coverage. Resident WebGPU work remains an evaluated optional backend rather than
a supported execution mode; no PyPI publication is made.

## Main adoption and deployed verification

Commit `29201db58be1d81258c5f1434b22a62f5b4c0824` is integrated into main.
Its exact-commit [quality run](https://github.com/PaperLeet/paperleet.github.io/actions/runs/38792197827)
succeeds: 2,646 tests, one optional Docker skip and six warnings in 460.13 seconds;
71 original native modules execute 9,372 tests across three repeats and catch all
312 mutants in 392.94 seconds. All seven pinned WASM suites pass.

Concurrent application work superseded that commit's Pages job. The successful
[Pages run](https://github.com/PaperLeet/paperleet.github.io/actions/runs/38792526334)
deploys descendant `f3cbabbf3660a59bc05640f2c382e37cb1280a9b`. The library,
curriculum, browser sources and exported data are byte-identical between those
commits. Live paperleet.com verification matches all 71 starters/manifests and
both runtime files exactly, including version 0.11.0 and Apache-2.0/NOTICE:
autograd SHA-256 `3ef4e42e19b37a0a2d86e11a7519b548611a8f3c2fe0f3acb0e719660805d123`
(328,467 bytes) and schedules
`f277c45b7b84863740bebc979e22e2f5c1f6b1fa85516e03a7c87626a84e406d` (35,895 bytes).
Exact logs, descendant-tree audit and frozen live expectations remain with the
iteration-11 local evidence. This closes that release's remote verification;
the independent-library development goal remains active.
