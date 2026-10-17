# Torch runtime iteration 52 — Multihead attention

Version 0.52.0 adds `torch.nn.MultiheadAttention` and
`torch.nn.functional.multi_head_attention_forward` to the independent NumPy
CPU/Pyodide WASM library and the privately vendored PaperLeet runtime. It keeps
the existing implementation of tensor operations, AD, SDPA and module lifecycle.
The broader independent-library efficiency/core-coverage/backend goal remains
active; this milestone does not establish complete PyTorch compatibility.

## Behavior and implementation

The module supports real float32/float64; sequence-first, batch-first and
unbatched inputs; shared/self/cross attention and independent K/V feature widths;
boolean or additive 2D/3D attention masks and padding; averaged or per-head
weights; bias K/V and zero tokens; dropout/evaluation; and functional static K/V.
Self-attention uses one packed Q/K/V projection, while shared K/V uses one packed
K/V projection. Other cases use separate projections. The output projection has
the native NonDynamicallyQuantizableLinear type/namespace and genuine reset
writes, matching initialization versions. None-registered packed/separate
parameters and state keys match the native structure.

Input/parameter/mask VJPs, JVP and reverse-over-JVP compose through the existing
engine. Module state, hooks, pickle/deepcopy, dtype conversion, AdamW/SGD and
scheduler/checkpoint continuation are exercised. Mutation guards cover saved
inputs/parameters. No native Torch binary is installed into the browser.

Native comparisons target Torch 2.14.1 CPU MHA with its inference fastpath
disabled and SDPA MATH selected. Boolean MHA masks use True=blocked. A causal
hint requires an explicit mask; without padding and returned weights the SDPA
causal hint is used, otherwise the explicit mask is applied. Returned weights
require the final additive mask dtype to match the query. No-weight SDPA also
accepts float32 masks for float64 queries. Fully masked returned-weight rows
preserve softmax NaNs; no-weight SDPA produces zero attention before output
projection, so a nonzero output bias remains. Native fused inference kernels
have different corners and are not emulated.

Dropout uses this library's distribution/replay stream, not native seeded bits.
Positive non-boolean integer dimensions and probabilities in [0,1] are validated
eagerly. Native constructors can defer/accept some invalid inputs that this
library rejects. No nested tensors, quantization, FlashAttention or GPU kernel
support is claimed. Fully masked native mixed-derivative equivalence is not
newly claimed; the stable SDPA rule and prior documented limitations remain.

## Final-source local verification

| Gate | Result |
|---|---|
| Full repository/library CPU suite | 40,031 passed, 1 skipped; 262.74s |
| New independent MHA contracts | 218 tests |
| Separate native MHA oracle | 192 numerical cases + 16 initialization cases, 4,107 arrays |
| Actual Node / Pyodide 0.29.4 / NumPy 2.2.5 | 6,402 contracts, 49 suites |
| Actual Chrome 155.0.8059.82 Worker | Same 6,402 contracts and source/test hashes |
| Installed Python 3.10.19 / NumPy 1.26.4 wheel | 6,415 passed; native Torch absent |
| Both distributions / independent uninstall | Exact 56 source/license resources; MHA primal/JVP/mixed reverse in both namespaces |
| Original native PaperLeet catalog | 78 problems, 10,218 executions, 342 mutants, 3 repeats; 468.378s / 600s |
| Final exported WASM catalog | 78/78 problems, 342 mutants, 6,402 contracts; 228.674s |
| Frontend / default and Pages-profile builds | Five successful local receipts |
| Immutable browser resources | All 11,177 previous assets preserved; 158 new reachable assets; 11,335 total |
| Release policy | Seven canonical files unchanged |

Independent NumPy equations use separate operand projections and explicit head
contractions. Directional differences check each input/parameter/additive-mask
VJP; input JVPs and reverse-over-JVP derivatives are independently differenced.
The output bias is correctly unused by a pure input-JVP loss. A stochastic
embedding/LayerNorm/MHA/MLP encoder trains, restores model/AdamW/scheduler state
and continues identically. Masks, functional static and empty sources, returned
weights, alias/shared gradients, dropout replay and saved-version failures have
dedicated portable tests.

The final 0.52 adapter reruns against frozen native MHA arrays and exact fixture
script bytes. Every array schema, shape, dtype, None gradient and initialization
version agrees. Tolerances are rtol 8e-5 / atol 8e-6 for float32 and 2e-10 / 2e-11
for float64. The largest observed absolute error is 9.54e-7. The original native
PaperLeet catalog receipt is reused after rechecking all 1,152 native
catalog/runner/test/configuration identities; the final exported WASM catalog
is freshly executed. These are distinct oracle and application checks.

## Projection packing performance

The [development preflight](torch-runtime-iteration-52-preflight.md) retains
four fresh processes per runtime, five samples and alternating runtime/method
order. Three self-attention shapes and both dtypes produce 24 flows per runtime.
Forward/reverse/JVP and a four-step LayerNorm/MHA/MLP encoder SGD flow are measured
against a same-library three-independent-projection control. Values, all input/
parameter gradients, JVPs and matching updated parameters are checked first.
Training timing excludes parameter restoration and diagnostic copies; optimizer
construction is included. Thread caps are two; the host is shared.

The mathematical source and benchmark bytes are identical to that preflight.
Only `__init__.py` changes its version from 0.51.0 to 0.52.0; this exact one-line
change and every other engine hash are checked by `preflight-reuse.json`.
The final-source CPU/WASM/package/catalog gates above rerun. The old timing
evidence is explicitly reused rather than relabeled as a fresh 0.52 timing run.

Positive percentages below mean lower time than the projection control. They
are medians of per-process six-case medians. All per-shape samples and slower
rows are retained; no comparison with a previous library release, native Torch,
cold start, exclusive host capacity or peak RSS is asserted.

| Runtime | Flow | Median reduction | Range of process medians |
|---|---|---:|---:|
| NumPy CPU | forward | 9.5% | 8.3% to 13.4% |
| NumPy CPU | reverse | 13.7% | 9.2% to 14.2% |
| NumPy CPU | jvp | 12.3% | 10.0% to 13.8% |
| NumPy CPU | training4 | 5.2% | 4.5% to 9.7% |
| Node/Pyodide | forward | 7.1% | 5.2% to 9.5% |
| Node/Pyodide | reverse | 3.5% | 1.9% to 6.8% |
| Node/Pyodide | jvp | 4.7% | 2.3% to 6.5% |
| Node/Pyodide | training4 | 0.9% | -1.2% to 1.6% |
| Chrome Worker | forward | 3.7% | 2.2% to 5.8% |
| Chrome Worker | reverse | 3.4% | 3.1% to 3.8% |
| Chrome Worker | jvp | 4.2% | 1.7% to 5.5% |
| Chrome Worker | training4 | 1.9% | 1.4% to 2.4% |

Browser training gains remain small and include regressions: Node process
medians range from -1.2% to 1.6%, and individual Chrome shapes can be slower.
This does not demonstrate universal browser-training improvement.

## Source, distribution and export identity

The package contains 54 Python modules plus LICENSE/NOTICE. Relative to 0.51,
53 resources are unchanged; `__init__.py` and `_api.py` change and `_multihead.py`
is added. Sorted concatenated source/license bytes change from
656,717 to 674,834;
local concatenated gzip changes from 151,889 to
155,674. These are source estimates,
not transferred bytes or measured memory. All 14 pinned Pyodide files and the
npm lock are rechecked. Existing dependency versions remain unchanged.

The 161 active catalog/index/detail/manifest/runtime export identities match the
final source. Prior immutable payloads remain exact and all newly added payloads
are reachable. Public/private distributions have separate installed paths and
identical numerical source bytes. Source/wheel/sdist identity, minimum-runtime
tests and independent uninstall are checked in isolated environments.

Concurrent main update `cd48bff0c0ac2933bb765ba40e458560a6a974d8` is preserved
as an ordinary merge. All 16 incoming solution-workspace/interview-source UI
files remain exact upstream Git blobs. Numerical source, native catalog inputs,
tests, WASM driver and all active export identities are unchanged. The full CPU
receipt and numerical/package/catalog evidence therefore retain their exact
inputs. Five frontend/build checks rerun on the merged UI, including its new
solution-panel/source-preview checks. Both earlier and refreshed receipts are
archived; `upstream_syncs` records every path hash.

`packages/stridetorch/benchmarks/results/iteration-52.json` and
`iteration-52-integration.json` retain machine-readable results; three matching
raw harness archives retain final sources/tests, native arrays, commands, logs,
package artifacts and receipts. The earlier immutable preflight archive retains
the original native probes and all performance samples. No CI, tag, workflow
dispatch, release, PyPI upload or deployment is initiated by the local gates.

Transformer composition, broader core coverage, further numerical and buffer
optimization and measurement of optional persistent GPU execution remain work
for the overall goal. Project renaming awaits the user's selection.
