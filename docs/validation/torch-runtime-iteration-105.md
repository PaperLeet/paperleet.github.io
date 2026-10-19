# Iterations 105–110: native-order HVP and an identified WASM rounding variance

The previous turn exposed 51 extreme-float32 HVP discrepancies. This revision
replaces their shared algorithmic cause for supported vector-norm graphs and
qualifies the new source on CPU, minimum NumPy and actual WASM runtimes. One
WASM point comparison remains different; a controlled native experiment now
isolates its cause to one ULP in a norm primitive. That raw failure is preserved.
The continuing goal remains active and canonical production code is unchanged.

## Scoped Tensor pullbacks instead of more singular-point patches

Native functional HVP constructs a Jacobian, differentiates it with a zero
dummy vector, then differentiates with respect to that vector. The old
reverse-over-forward method can be mathematically equivalent while evaluating
float32 intermediate underflow, overflow and cancellation differently.

The isolated `_higher_order.py` records Tensor-valued pullbacks only inside one
HVP context. A temporary Tensor subtype records elementwise, shape and sum
operations. Rules and Tensor owners live in a context map, adding no fields to
ordinary graph nodes. Supported vector-norm primal graphs use three reverse
traversals in the native order. Existing unrecorded graphs retain their previous
path without calling the user's function again. Public general `create_graph`
and differentiable HVP outputs remain unsupported.

The context preserves saved-version checks, target pruning and the distinction
between logical zero, numerical zero and an unused edge. It clears its temporary
rules and owners on success or exception. Caller-owned coefficients keep their
original class and gradient state; only fresh internal tensors are wrapped.
Nine additional checks cover one callback, caller ownership, the old matrix
fallback, state restoration, public mutation rejection, saved alias versions,
zero curvature and 100 repetitions with garbage collection disabled.

An initial mutation test expected the later saved-version error, but the existing
public JVP guard correctly rejected mutation earlier. Its seven-pass/one-failure
result is preserved. The assertion was corrected and a separate actual alias
mutation test exercises the new Tensor pullback's saved-version check.

## Scalar-power precision and one private result buffer

The first three-reverse prototype matches 4,476 of 4,480 HVP cases per native
version. The last four involve scalar powers at tiny float32 ReLU inputs.
The native CPU
[scalar-power kernel](https://raw.githubusercontent.com/pytorch/pytorch/v2.6.0/aten/src/ATen/native/cpu/PowKernel.cpp)
uses a double scalar exponent and has specialized square, cube, square-root,
reciprocal-square-root and reciprocal expressions. The NumPy-only helper now
follows that scalar opmath, writes the buffered double-power loop into a
float32 result, and reuses one private result buffer for multi-step cases.
It does not retain a full double-sized output or mutate input storage.

Iteration 107 also prevents the recording helper from reclassifying a
caller-owned Tensor coefficient. Its final source has 65 resources and remains
an isolated 0.79 candidate; the canonical 0.78 checkout is untouched.

## Completed CPU and installed minimum-version results

| Verification on the selected iteration-107 source | Result |
| --- | --- |
| Existing CPU SDK | 11,982 passed, zero skips |
| Norm/zero/scoped-reverse targeted tests | 104 passed, zero skips |
| Primary norm observations, per native version | 780 cases, all 3,660 stages match |
| Original composition observations, per native version | All 1,600 stages match |
| Logical-zero chains, per native version | All 576 match |
| Expanded HVP matrix, per native version | All 4,480 match |

Native references are Torch 2.6.0 and 2.14.1. Path observations prove all 4,480
expanded cases actually select three reverse traversals, with no missing rule
and no leaked context. This is not a legacy fallback result.

A local 0.79 wheel is installed in a fresh Python 3.10.19 / NumPy 1.26.4
environment without native Torch. All 65 installed runtime resources are
byte-identical to the source snapshot and wheel entries. The complete installed
suite passes 12,096 tests and skips three existing rank-34/35/64 cases because
NumPy 1.26 cannot construct those ranks. There are no test failures. The same
installed wheel also matches all 4,480 HVP cases against both native references,
again selecting the new path every time.

The wheel SHA-256 is
`b068f792efd8c2e8dd3fb05abf4e9bbaed5ecf7b9f5473a3f904793b690c5357`.
Local wheel/sdist construction is not publication. The candidate standalone
tests now explicitly pass `-B` to child interpreters, including isolated mode;
the final source distribution/rebuild proof must be renewed after that edit.
Private PaperLeet packaging and real native-Torch co-installation still need
their new-version qualification.

## Actual Node and Chrome results retain one point difference

Node/Pyodide and real Chrome each pass all 12,086 SDK/targeted tests with no
skips. Both reproduce the primary norm, composition and zero-chain results.
All 4,480 expanded HVPs execute the new path. Their numerical comparison matches
4,479 cases and differs on the same one case against each native reference:
`float32/finite_large/2.9/sin_out`.

The source-bound supervisors correctly exit nonzero because these counts differ
from the CPU counts. Their progress, logs and raw JSON remain unchanged. Test
counts alone do not convert this result into a complete WASM numerical pass.

## Causal precision study

Iteration 110 records every norm arithmetic stage on CPU, minimum NumPy and
Node/Pyodide. The powered values and float32 accumulated sums are bit-identical.
Only the first final power differs:

| Result | First norm value | Float32 bits |
| --- | ---: | --- |
| Native CPU and NumPy CPU | 12,699,981,824 | `0x503d3e96` |
| WASM | 12,699,980,800 | `0x503d3e95` |

The difference is one ULP, 1,024 at that magnitude, or approximately
`8.063e-8` relative. A 100-digit calculation of the final power using its already
rounded accumulator and exponent gives `12699981308.696910...`, just below the
midpoint. The native and WASM results differ from that value by approximately
0.50323 and 0.49677 ULP respectively. Sine amplifies the phase difference.

Two independent native experiments strengthen this diagnosis:

- On both Torch versions, the equivalent expression using scalar Tensor powers
  and a sum produces exactly the WASM norm and sine result.
- In a separately labeled counterfactual, pinning only the temporary native
  norm's forward value to the observed WASM value reproduces the actual WASM
  primal and every HVP entry. Original reference data and production mutation
  rules are not replaced by this diagnostic.

The native
[numerical-accuracy documentation](https://docs.pytorch.org/docs/2.6/notes/numerical_accuracy.html)
does not promise bitwise identity across equivalent floating-point expressions
or platforms. This evidence identifies a backend rounding variance rather than
an unexplained Tensor pullback error. The original composed point comparison
still fails; no case or tolerance has been removed, and no unconditional pass
claim is made from the diagnosis.

Iteration 109's all-double norm-power hypothesis reproduces the same point
difference on CPU, so it is not selected. The current candidate remains the
iteration-107 source, as installed and mounted in iteration 108.

## Remaining work before main adoption

Broaden direct scalar-power controls across sizes/layouts and renew the existing
native/mixed-AD oracle suites, since scalar-power opmath also affects ordinary
Tensor operations. Assess the identified variance against an explicit numerical
accuracy contract while preserving its raw point evidence. Renew public/private
wheel and sdist rebuild proofs, aliases/co-installation, full application tests,
native/WASM catalog budgets, exports, frontend/worker checks and lifetime tests.
Run fresh broad performance comparisons for this exact 65-resource source.

The old full-version CPU regression and original control anomalies remain part
of the performance evidence; this numerical work does not close them. Reconcile
current main and perform fresh guarded ordinary adoption only after the selected
version is mature. No GitHub CI, push, tag, release, publication or deployment
was triggered. CPU/WASM/core-coverage/WebGPU work remains active.
