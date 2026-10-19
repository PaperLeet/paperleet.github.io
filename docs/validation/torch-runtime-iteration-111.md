# Torch runtime iterations 111–122: precision isolation and repository integration

The 0.79 development implementation is now integrated into the local working
branch. Its runtime is byte-identical to iteration 114: 65 resources, including
the new vector-norm and scoped Tensor-pullback modules. This checkpoint does not
claim main adoption, complete application qualification, overall performance
qualification, or a new GPU backend. No CI, tag, publication or deployment was
triggered.

## Implementation

`torch.linalg.vector_norm` adds real reverse AD, direct/functional JVP and
functional HVP, with dimension, order, dtype, empty-input and strided-input
contracts. Complex/half inference and their declared AD limits remain explicit.
Supported norm HVP graphs use three reverse traversals and one user callback.
Unrecorded existing graphs keep their prior HVP path. Public higher-order output
graphs remain unsupported.

Logical-zero cotangents are distinct from unused derivatives and materialized
zero arrays. Their declared propagation rules retain saved-storage version
checks, hook behavior, pruning and lifetime checks. The scoped recording context
owns its temporary Tensor rules and releases them after success or failure;
ordinary Tensor/node instances do not acquire recording maps.

Iteration 107's global double-exponent power change was too broad. The independent
power matrix exposed 132 newly different rows relative to the preceding
prototype. Iteration 114 confines that opmath to the private norm-HVP reverse
traversals and restores the prior public float32 storage arithmetic. A `finally`
block restores the context's reverse-phase flag. Float32/64 square, cube,
square-root and reciprocal specializations reuse one output buffer. The float64
reciprocal-square-root path now preserves negative zero and negative-infinity
behavior. NumPy scalar exponents no longer promote half result storage.

The dispatcher also avoids constructing empty option dictionaries and repeated
empty implementation-keyword sets for its common no-option calls. Earlier
iteration 95 measurements remain diagnostic evidence; this checkpoint makes no
new complete-version speed claim.

## Verification

All explicit Python invocations used `-B` and disabled bytecode writes. Functional
tests ran without owned latency or catalog-budget measurements in progress.

| Scope | Observed result |
| --- | --- |
| Integrated repository SDK, NumPy 2.5.4 | 12,105 passed; no skips or failures |
| Installed standalone wheel, NumPy 1.26.4 | 12,102 passed, 3 existing high-rank skips; no failures |
| Actual Node/Pyodide SDK | 12,092 passed; no skips or failures |
| Actual Chrome Worker/Pyodide SDK | 12,092 passed; no skips or failures |
| Primary norm controls, each native version | 780 cases, 3,660 matching stages |
| Norm compositions, each native version | 320 cases, 1,600 matching stages |
| Declared-zero chains, each native version | 576 matching cases |
| Expanded norm HVP, CPU, each native version | 4,480 matching cases |
| Expanded norm HVP, each WASM runtime/native version | 4,479 matching cases; 1 retained point difference |
| Original matrix controls, pinned Torch 2.6 | 11,079 matches and 262 explicit boundaries out of 11,341 |
| Original mixed-affine controls, pinned Torch 2.6 | 55,770 matches and 2,550 explicit boundaries out of 58,320 |
| Independent scalar-power matrix, each runtime/native version | 14,378 matches and 238 retained point differences out of 14,616 |
| Real native Torch 2.14 coinstallation | 13 standalone contracts and nested alias/cache restoration passed |

The matrix/mixed controls preserve all 69,661 candidate rows from the previously
qualified 0.78 runtime. Fresh native 2.14 observations contain 884 matrix and
9,192 mixed-affine point differences from this pinned-2.6 behavior, including
changed empty mixed-dtype `addmv` acceptance. These are preexisting version
differences. The native-2.14 raw comparison gates remain failed; this is not a
claim of compatibility with both versions' conflicting contracts.

The independent power probe covers float16/32/64, lengths 0/1/6/8/17/256,
contiguous/strided/transposed layouts, ordinary/tiny/large/nonfinite inputs,
21 scalar exponents and forward/VJP/direct-JVP/functional-JVP/HVP channels.
It verifies that caller data remains unchanged. Comparisons retain exact
NaN/Inf and zero-sign classes, dtype/shape checks, declared relative tolerances
and four least-subnormal absolute steps. The same unchanged probe was rerun
against the actual 0.78 resource snapshot: it had 644 differing rows. The new
implementation fixes 406 and introduces no newly differing row relative to
0.78. All remaining 238 cases are retained. CPU, minimum NumPy, Node and Chrome
have the same new comparison counts.

## Numerical limits remain visible

The WASM HVP difference is still
`float32/finite_large/2.9/sin_out`. Iteration 110 identified a one-ULP difference
in the final norm power before applying sine. The rounded accumulator and
reciprocal exponent are identical. Native equivalent power expressions and a
separately labeled native counterfactual reproduce the WASM phase and every HVP
entry. Those diagnostics neither replace the original references nor make the
original point gate pass. Iterations 108 and 116 preserve their terminal failed
supervisor results.

Native scalar power can itself choose different exponent precision in scalar
and vectorized paths; near-integer exponents on negative bases expose this as
finite versus NaN output. The implementation avoids applying one host scalar
policy globally to all public storage operations. The relevant primary source
is [Torch 2.6 PowKernel.cpp](https://github.com/pytorch/pytorch/blob/v2.6.0/aten/src/ATen/native/cpu/PowKernel.cpp).
No platform-wide bitwise equivalence or unrestricted exceptional-power
compatibility is asserted.

## Independent packaging and aliases

Both public and private wheels contain the same 65 runtime resources. Both
sdists rebuild wheels with those exact bytes. Public metadata keeps Apache-2.0,
Python >=3.10 and NumPy >=1.26, without installing a top-level `torch` package or
requiring native Torch/PaperLeet. The private application wheel retains its
separate `paperleet._vendor.torch` namespace.

A fresh minimum environment initially contains neither native Torch nor
PaperLeet. Its complete installed-wheel suite passes with the three documented
NumPy-1.26 rank limits. After installing the private wheel, both namespaces and
their explicit aliases are checked against their installed resource bytes.
A separate environment with actual native Torch 2.14 verifies nested public
and private aliases, then restores every previously cached native `torch.*`
module by identity. The launcher/import contracts also pass there.

Two harness failures are retained. Iteration 117 reused `sys.argv[1]` after the
build backend mutated it; iteration 118 captures the output path before either
backend call. Iteration 118 then incorrectly compared a yielded alias namespace
with another factory instance. Iteration 121 checks the yielded namespaces,
their Tensor classes, and the complete restored native module cache. It
completes the combined package proof without changing library source or
rewriting the original failed results.

## Evidence and remaining adoption work

Detailed frozen inputs, raw results and receipts live under
`/Users/david/Developer/paperleet-wasm-evaluation-20261015/torch-iteration-{111..122}`.
The selected source is iteration 114; iteration 119 renews the broad controls,
120 records the repository copy and integrated SDK run, 121 completes packaging,
and 122 compares scalar powers against the actual 0.78 baseline.

Full PaperLeet CPU/application tests, three native catalog repetitions,
whole-catalog WASM budgets, regenerated judge exports, frontend/build/worker and
nested-route checks, lifetime controls and uncontended complete-version
performance measurements still need renewal for these exact bytes. Older 0.78
application/performance evidence is not reused as proof of 0.79. Main adoption
still requires current upstream reconciliation, a fresh archive/audit and the
unchanged guarded ordinary-push path. The user-owned primary checkout and its
environment remain untouched. The larger core-API and WebGPU objective remains
active.
