# Torch runtime iteration 52 — MHA development preflight

This feature branch adds `nn.MultiheadAttention` and
`nn.functional.multi_head_attention_forward`. It is a development checkpoint:
package metadata still says 0.51.0, main remains at the prior adopted version,
and final browser exports/main adoption are pending. No release or online
deployment has been initiated. The broader library/core-coverage/backend goal
remains active. Project renaming awaits the user's selection.

## Behavior and compatibility

The math path supports real float32/float64, batched sequence-first/batch-first
and unbatched inputs, self/shared-KV/cross attention, separate K/V widths,
boolean and additive 2D/3D attention masks, key padding, averaged/per-head
weights, bias K/V and zero tokens, functional static K/V, dropout and evaluation.
Self-attention packs Q/K/V into one projection; shared K/V packs those two.
Output projection uses the native NonDynamicallyQuantizableLinear namespace,
genuine initialization writes and parameter versions. State dictionaries, hooks,
conversion, independent pickle/deepcopy and optimizer/checkpoint continuation
are tested. Input/parameter/mask gradients and JVP/mixed AD compose through the
existing engine, with saved-version guards.

Native reference execution explicitly disables MHA fastpath and selects SDPA
MATH on Torch 2.14.1. It does not claim fused inference equivalence. The native
probe retains backend differences: fastpath may ignore a causal hint and may
produce NaNs for fully masked rows even without returned weights. This library
follows math fallback behavior: causal hints require explicit masks; returned
weights preserve fully-masked softmax NaNs; no-weight SDPA returns zero attention
before the output projection, so nonzero output bias remains. Nonempty finite
mask cases are used in the full mixed-native differential, and fully-masked
values/first derivatives are checked separately against independent equations.
No new fully-masked native mixed-derivative equivalence is asserted.

Returned weights require a matching final additive-mask dtype, while SDPA also
accepts float32 masks with float64 input. Boolean masks mean True=blocked.
Training follows the library's dropout distribution/stream; native seeded dropout
bits are not promised. Invalid probabilities and non-positive/non-integer or
boolean dimensions are rejected eagerly; some native constructors defer or
accept these inputs. No nested tensors, GPU/Flash kernel or quantization support
is added by this change.

## Validation

| Check | Observed result |
|---|---|
| Complete CPU repository/library suite | 40,031 passed, 1 skipped; 231.82s |
| New independent NumPy MHA tests | 218 passed |
| Separate native processes | 192 differential cases + 16 initialization cases; 4,107 arrays |
| Node / Pyodide 0.29.4 / NumPy 2.2.5 | 6,402 contracts, 49 suites |
| Actual Chrome 155.0.8059.82 Worker | Same 6,402 contracts and exact source/test hashes |
| Installed Python 3.10.19 / NumPy 1.26.4 wheel | 6,415 passed, native Torch absent |
| Public/private coinstall and independent uninstall | Exact 54 Python modules plus LICENSE/NOTICE; MHA primal/JVP/mixed reverse runs in both namespaces |
| Release policy | Seven canonical files byte-identical |

The independent NumPy reference uses separate operand projections and explicit
head contractions. Directional finite differences validate each input,
parameter and floating mask VJP; JVPs are independently differenced, and nested
finite differences check reverse-over-JVP inputs, parameters and directions.
The output bias is correctly unused by a pure input-JVP loss. A stochastic
embedding/LayerNorm/MHA/MLP encoder trains, restores AdamW/scheduler state and
continues with identical parameters. Forward/reverse dropout replay, empty
static sources, fully masked rows and mutation guards have dedicated checks.

Native differential tolerances are rtol 8e-5 / atol 8e-6 for float32 and
2e-10 / 2e-11 for float64. All array schemas, dtypes, shapes, None gradients
and initialization versions agree. The largest observed absolute error is
9.54e-7 across gradients/mixed derivatives; initialization differs by at most
5.96e-8. Native arrays and source-identified script are archived separately from
the adapter arrays; no Torch installation is needed for portable tests.

## Projection packing measurements

Positive percentages mean less time than the same library's three-independent-
projection control. This is a mechanism comparison, not a previous-version or
native Torch comparison. Three self-attention shapes and both dtypes produce
24 flows per runtime. Four fresh processes per runtime run sequentially with
alternating runtime/method order, five samples and four calls per sample (one
four-step encoder SGD flow for training). Numerical-library thread caps are two.
Parameter restoration and diagnostic array copies are outside training timing;
optimizer construction is included. Values, all input/parameter gradients,
JVPs and matching four-update encoder parameters are checked before timing.

| Runtime | Flow | Median reduction | Range of process medians |
|---|---|---:|---:|
| cpu | forward | 9.5% | 8.3% to 13.4% |
| cpu | reverse | 13.7% | 9.2% to 14.2% |
| cpu | jvp | 12.3% | 10.0% to 13.8% |
| cpu | training4 | 5.2% | 4.5% to 9.7% |
| node | forward | 7.1% | 5.2% to 9.5% |
| node | reverse | 3.5% | 1.9% to 6.8% |
| node | jvp | 4.7% | 2.3% to 6.5% |
| node | training4 | 0.9% | -1.2% to 1.6% |
| chrome | forward | 3.7% | 2.2% to 5.8% |
| chrome | reverse | 3.4% | 3.1% to 3.8% |
| chrome | jvp | 4.2% | 1.7% to 5.5% |
| chrome | training4 | 1.9% | 1.4% to 2.4% |

All per-shape samples and slower cases are retained. Small differences can be
host noise; these observations establish no universal speed or peak-memory
improvement. Initial concurrent/restore-including pilots are preserved separately.

## Evidence and remaining integration

Machine-readable results and the matching raw source/tests, native arrays,
benchmarks, logs, wheels and receipts are stored in
`packages/stridetorch/benchmarks/results/iteration-52-mha-preflight.json` and
`iteration-52-mha-preflight-harness.tar.gz`. The archive is SHA-256 identified.
It excludes virtual environments, credentials and runtime binaries.

Before promoting this candidate, advance the version, regenerate active exports
without replacing published immutable payloads, verify the complete 78-problem
exported WASM catalog, and run frontend/build gates. Then inspect current remote
main and active quality/Pages chains before an ordinary non-force main push.
No CI trigger, tag, workflow dispatch, release, PyPI upload or deployment is
authorized by this checkpoint. CPU/WASM optimization and broader core coverage,
including Transformer composition and optional GPU execution, remain ongoing.
