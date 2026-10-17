# Torch runtime iteration 55: default floating dtype and half random streams

Version: 0.55.0. Baseline numerical main:
`ef5f03dfd92b4ee4597525ab7230e511eb966703` (0.54.0).
Final pre-adoption main: `398abd0833775012c775f319168f718d53c3eca8`.

The adapter previously fixed Python floating inference, factories, parameter
initialization and Adam step counters at float32. `get_default_dtype` and
`set_default_dtype` now provide a distribution-local floating policy, initially
float32, accepting the exported float16/32/64 dtype constants. Invalid settings
are atomic. The policy covers Python float/complex inference, floating factories,
wrapped scalars, integer true division/transcendentals, subsequent neural-network
parameters/buffers and unfused CPU Adam/AdamW scalar step initialization.

Explicit dtype, NumPy storage precision, existing tensors/models, like/new
factories, saved differentiation tapes and checkpoint dtype metadata retain their
own rules. Parameterless normalization and pretrained embeddings do not reject
float32/64 data because an unused default is float16. Namespaces from the same
modern distribution share the setting; public/private/native installations stay
independent. `as_torch()` restores imports, not dtype state.

Real half random factories now implement the PyTorch 2.14.1 CPU opmath profile:
scalar normal caches, float32 normal blocks, incomplete tail RNG consumption and
uniform endpoint wrapping after rounding to half. Subsequent float64/float32
draws verify stream position. Half neural-network kernels and forward AD are
still unsupported. NumPy cannot represent complex32; explicit real/complex
operands can absorb a wrapped complex Number when the resulting storage is
representable. Native `full` infers complex64 for a complex fill with default half.
Mixed sequences can promote a Python complex Number's inferred complex-half
category into NumPy/Tensor float32/64 or complex64/128. Inference therefore
checks the final promoted storage type rather than rejecting an intermediate
category. A wider ordered/nested corpus and regression tests cover this boundary.

## Validation

- **40,497 CPU tests passed, 1 skipped** across the repository and library.
- **6,867 contracts each** in Node Pyodide and a Chrome Worker, **51 suites**,
  checking all **58 resources** (56 Python modules plus LICENSE/NOTICE).
- **6,880 installed-wheel tests** on Python 3.10.19 / NumPy 1.26.4, without native
  Torch; public/private distribution identities, coinstallation, independent
  uninstall, dtype-policy isolation and fresh-process restore all passed.
- **171 new default-dtype tests**, covering factories, promotion, 21 module
  initializers, optimizer/checkpoint lifecycle, changing the default after JVP,
  alias state, invalid settings, parameterless models and a frozen native half
  normal-tail/stream-position regression.
- Native differential: **1,701 cases**; **1,664 numerically compared cases /
  1,962 arrays** passed strict dtype/shape checks and per-dtype tolerances.
  **37 complex-half cases** are explicitly classified as unavailable storage or
  native kernels and are not numeric passes. Float16 rtol/atol are 3e-3/2e-3,
  float32/complex64 5e-5/8e-6, float64/complex128 3e-10/3e-11. The worst observed
  absolute errors were 0.00048828125 (half), 1.38615e-6 (float32) and 9.16e-16
  (float64); complex64/128 outputs in the corpus were exact. Ninety half random
  scenarios check **180 subsequent RNG arrays**: float32 draws bitwise equal and
  float64 draws within 3e-14. No universal cross-platform bitwise claim.
- Fresh exported WASM catalog: **78/78 problems**, **342 mutants**, all contracts
  passed and no browser budget exceeded. The previous pure-native catalog result
  is reused after checking all **1,152 unchanged native input hashes**.
- Five final frontend checks/builds passed. The seven incoming upstream UI files
  from `398abd0833775012c775f319168f718d53c3eca8` (inline reply quoting) were
  preserved by fast-forward and frontend checks rerun. Numeric source, test and
  export identities remained unchanged; earlier frontend logs are retained.
- **11,651 previous immutable assets** are unchanged; 158 new reachable assets
  and three indices select the new runtime. Seven canonical release policies are
  unchanged. No dependencies or pinned Pyodide runtime were changed.

## Performance against 0.54

Twelve fresh child processes (four per CPU/Node/Chrome), each validating before
timing 60 encoder flows across three shapes, two normalization orders, two explicit dtypes,
forward, reverse, JVP, mixed AD and four SGD updates. Matching manual equations
isolate core cost; standard candidate layers are recorded separately. Each child
alternates variant order, uses five samples and checks source/benchmark hashes.
Training restore/copy is outside the timed region. Runtime order alternates and
thread limits are two. These measure common explicit-dtype workloads, not every
factory or implicit-default workload, startup, peak RSS or native/GPU speed.

Positive values mean time increased. Each child contributes the median of six
matched shape/normalization ratios; the table shows the median/range of four children.

| Runtime | Dtype | Flow | Core time change | Process median range |
| --- | --- | --- | --- | --- |
| cpu | float32 | forward | 0.6% | -0.7% to 1.6% |
| cpu | float32 | reverse | -0.2% | -2.4% to 5.5% |
| cpu | float32 | jvp | -1.1% | -1.9% to 2.5% |
| cpu | float32 | mixed | 0.7% | -0.5% to 1.2% |
| cpu | float32 | training4 | -0.1% | -1.9% to 0.6% |
| cpu | float64 | forward | 2.2% | -0.6% to 7.7% |
| cpu | float64 | reverse | -0.4% | -2.4% to 0.7% |
| cpu | float64 | jvp | 1.0% | -0.3% to 1.7% |
| cpu | float64 | mixed | -0.7% | -1.5% to 1.0% |
| cpu | float64 | training4 | -0.8% | -3.4% to 2.5% |
| node | float32 | forward | -0.1% | -0.4% to 0.6% |
| node | float32 | reverse | -0.2% | -0.6% to 0.3% |
| node | float32 | jvp | 0.1% | 0.1% to 1.2% |
| node | float32 | mixed | 0.0% | -0.9% to 0.8% |
| node | float32 | training4 | -0.2% | -0.6% to 0.2% |
| node | float64 | forward | 0.5% | -0.8% to 2.6% |
| node | float64 | reverse | 0.1% | -0.2% to 1.3% |
| node | float64 | jvp | 0.2% | -0.5% to 1.5% |
| node | float64 | mixed | -0.1% | -3.2% to 0.4% |
| node | float64 | training4 | 0.1% | -1.5% to 1.1% |
| chrome | float32 | forward | -0.0% | -0.6% to 0.0% |
| chrome | float32 | reverse | -0.2% | -0.6% to 0.0% |
| chrome | float32 | jvp | 0.3% | 0.0% to 1.1% |
| chrome | float32 | mixed | -0.4% | -0.6% to 0.1% |
| chrome | float32 | training4 | 0.0% | -0.0% to 0.3% |
| chrome | float64 | forward | -0.0% | -0.0% to 0.6% |
| chrome | float64 | reverse | -0.1% | -0.4% to 0.4% |
| chrome | float64 | jvp | 0.6% | 0.3% to 0.8% |
| chrome | float64 | mixed | -0.1% | -0.3% to 0.5% |
| chrome | float64 | training4 | -0.6% | -1.3% to 0.0% |

Small differences near zero and coarse browser timer effects should be treated
as measurement noise. This iteration expands API/lifecycle coverage; it does
not claim a universal kernel speedup or add a WebGPU execution backend. The 0.53
float32 mixed-AD qualification and dense CPU/WASM boundaries continue to apply.
The raw `Tensor` constructor retains NumPy-oriented storage inference.

## Reproducibility and release scope

Machine-readable result:
`packages/stridetorch/benchmarks/results/iteration-55.json`.
Frozen source/tests, native arrays, wheel artifacts, raw timing and terminal
receipts: `packages/stridetorch/benchmarks/results/iteration-55-harness.tar.gz`
(8847989 bytes, SHA-256 `a296c8b239f7c4533b194984e1a05a20d5d20a157e186874e9d4dc4f6746bf44`).
The `before-promoted-complex-fix` folder preserves earlier successful runs and
unpublished assets before that additional boundary was found. Final counts,
source identities and timings use the independently rebuilt final candidate.

All work ran locally. No tag, workflow dispatch, release, package upload or
deployment was initiated. The independent-library efficiency/core-coverage/backend
goal remains active.
