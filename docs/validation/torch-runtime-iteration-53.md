# Torch runtime iteration 53 — Dense Transformer and LayerNorm precision

Version 0.53.0 adds Transformer encoder/decoder layers, cloned stacks and the
complete encoder-decoder module to the independent NumPy CPU/Pyodide WASM
library and the privately vendored PaperLeet runtime. It also adds genuine
Linear reset initialization and improves float32 LayerNorm mixed-AD precision.
The broader independent-library efficiency/core-coverage/backend goal remains
active. Full PyTorch, NestedTensor, fused Transformer kernels and GPU execution
are not claimed by this revision.

## API and numerical behavior

`TransformerEncoderLayer`, `TransformerDecoderLayer`, `TransformerEncoder`,
`TransformerDecoder` and `Transformer` support real float32/64, batch-first and
sequence-first layouts, unbatched inputs, pre/post norm, ReLU/GELU/custom
callables, biasless or zero-width feedforward networks, self/cross attention,
boolean/additive attention and padding masks, causal hints/automatic square-mask
detection, optional final norm and custom full-model encoders/decoders.
`Transformer.generate_square_subsequent_mask` supports Boolean/floating/complex
mask generation; attention computation itself remains real float32/64.

Stacks clone layers and parameters independently. State keys, parameter order,
initialization values/versions and cloning follow the native structure. Linear
now exposes `reset_parameters()` and allocates before genuine initialization
writes. LayerNorm initialization likewise increments real storage versions.
MultiheadAttention's special output Linear reuses that corrected implementation.
Known functional ReLU/GELU activations are stored as stable module functions so
string/functional/module activation configurations survive pickle/deepcopy.

The dense backend records `use_nested_tensor=False` even if the optimization
preference `enable_nested_tensor=True` is accepted. `mask_check` is also a dense
optimization preference; validation remains active. Correct causal hints remain
the caller's responsibility. Nonnegative non-boolean integer counts/widths are
checked eagerly. Zero-depth stacks construct but fail on forward with IndexError,
matching the native dense path. Non-native optimizations and invalid-constructor
acceptance differences are kept explicit rather than advertised as full API
equivalence.

Float32 LayerNorm statistics use float64 accumulation before rounding back.
Finite input/result JVPs with finite epsilon at least float32 machine epsilon
also use float64 tangent intermediates and a wider reverse tape, then return
float32. The native LayerNorm convention of frozen mean/inverse statistics in
mixed AD is preserved. Float64 execution keeps the prior formula; tiny,
non-positive epsilon and non-finite cases retain the original tangent formula.
This adds temporary/intermediate storage; no peak-memory reduction is claimed.

## Accuracy qualification

Pinned Torch 2.14.1 CPU uses MHA fastpath disabled and SDPA MATH. The corpus
includes 480 numerical configurations and 160 initialization configurations:
five module types, both dtypes, three layouts, pre/post norm, ReLU/GELU,
unmasked/Boolean/causal/differentiable additive masks, and bias variants.
Outputs, every input/parameter/mask gradient, JVPs, reverse-over-JVP and
initialization produce 36,850 arrays. State schemas, shapes, dtypes, None
gradients and initialization versions match. Float32 elementwise tolerances
are rtol 5e-5 / atol 8e-6; float64 uses 3e-10 / 3e-11.

There are 5 float32 mixed arrays outside
those direct elementwise native tolerances; they remain classified in the raw
strict-comparison record. Outputs, ordinary VJPs, JVPs, float64 mixed AD and
initialization all meet the elementwise criteria. No claim that all float32
mixed arrays meet the direct criterion is made.

A second 240-case oracle promotes exactly the same representable float32 model
values, inputs, seeds, directions and masks to float64. Its 16,536 arrays are
compared elementwise between native and adapter. In addition, all 8,196 float32
mixed arrays from each backend must satisfy
`max_abs(fp32 - native_fp64) / max(1, max_abs(native_fp64)) <= 1e-5`.
Worst normalized errors are 7.23e-06 for native and
8.74e-06 for the adapter. Both satisfy that bound.
These are separate, documented normwise floating-point criteria, not a relaxed
claim of direct elementwise equivalence.

Before the precision change, the adapter's worst normalized error was about
1.60e-5 and failed that bound. Statistics-only, tangent-only and projected-form
experiments are retained; the combined wider calculation reduces the worst
error to about 8.74e-6. A small frozen native-float64 regression case runs on
CPU and actual WASM, including the public float32 output/gradient dtype.
The kernel tests do not require a native Torch installation.

## Final local validation

| Gate | Result |
|---|---|
| Complete repository/library CPU suite | 40,265 passed, 1 skipped; 271.87s |
| New independent Transformer suite | 232 tests |
| Automatically added namespace checks | One SDK and one repository case |
| Actual Node / Pyodide 0.29.4 / NumPy 2.2.5 | 6,635 contracts, 50 suites |
| Actual Chrome 155.0.8059.82 Worker | Same 6,635 contracts and source/test identities |
| Python 3.10.19 / NumPy 1.26.4 installed wheel | 6,648 passed; native Torch absent |
| Public/private coinstall and independent uninstall | Exact 57 source/license resources; full Transformer mixed AD in both namespaces |
| Original native PaperLeet catalog | 78 problems, 10,218 executions, 342 mutants, 3 repeats; 468.378s / 600s |
| Final exported WASM catalog | 78/78 problems, 342 mutants, 6,635 contracts; 250.232s |
| Frontend / default and Pages-profile builds | Five successful local receipts |
| Previous immutable payloads | All 11,335 preserved; 158 new reachable payloads; 11,493 total |
| Release policy | Seven canonical files unchanged |

Independent NumPy equations check dense attention, residuals, normalization and
feedforward composition. Directional differences exercise all inputs/parameters
together; full native array checks validate each gradient separately. Floating
mask VJPs/JVPs, custom activation, zero-width/biasless paths, causal detection,
activation serialization, independent clones, saved-version rejection and
stochastic model/AdamW/scheduler checkpoint continuation have portable tests.

The first WASM runs passed every pytest case but the old expected-count guard
was one low because the new namespace adds an automatically parametrized check.
Those logs are retained; the count was corrected and the Node/Chrome gates
rerun successfully. Existing successful builds were reused after source hash
checks. All 1,152 original native catalog/runner/test/configuration inputs are
byte-identical, so that native catalog receipt is reused; the final exported
WASM catalog above is freshly executed.

## Performance and cost

Four fresh processes per runtime run sequentially with alternating runtime and
method order, five samples and four calls per sample (one four-step training
flow). Three encoder shapes, both dtypes and pre/post norm produce 60 flows.
Actual 0.52 and candidate sources use identical manual encoder equations to
isolate core precision changes. The candidate standard EncoderLayer is measured
separately against that same baseline. Values, first gradients, JVP/mixed
gradients and matching four-SGD-step parameters are checked before timing.
Parameter restoration and diagnostic copies are outside training timing;
optimizer construction is included. CPU numerical-library threads are capped
at two. All source and benchmark bytes are SHA-256 identified.

Positive values mean **more time** than 0.52; negative values mean less time.
Each entry is a median of four process medians across six shape/norm cases.

| Runtime | Dtype | Flow | Core time change | Process median range | Standard layer time change |
|---|---|---|---:|---:|---:|
| NumPy CPU | float32 | forward | -0.9% | -2.8% to -0.7% | -1.3% |
| NumPy CPU | float32 | reverse | -0.3% | -0.7% to 0.1% | -1.0% |
| NumPy CPU | float32 | jvp | 4.8% | 4.6% to 4.9% | 4.2% |
| NumPy CPU | float32 | mixed | 3.7% | 2.5% to 4.7% | 4.4% |
| NumPy CPU | float32 | training4 | -2.2% | -2.5% to -1.0% | -1.5% |
| NumPy CPU | float64 | forward | 0.2% | -0.2% to 0.7% | 0.5% |
| NumPy CPU | float64 | reverse | 0.3% | -0.6% to 1.8% | -0.4% |
| NumPy CPU | float64 | jvp | -1.3% | -2.2% to -0.1% | -0.8% |
| NumPy CPU | float64 | mixed | 0.0% | -0.6% to 0.9% | -0.2% |
| NumPy CPU | float64 | training4 | -0.0% | -0.3% to 0.2% | -0.7% |
| Node/Pyodide | float32 | forward | -1.1% | -2.5% to -0.9% | -1.1% |
| Node/Pyodide | float32 | reverse | -0.8% | -1.3% to -0.5% | -0.6% |
| Node/Pyodide | float32 | jvp | 3.1% | 2.6% to 3.9% | 2.5% |
| Node/Pyodide | float32 | mixed | 2.2% | 1.7% to 2.8% | 1.7% |
| Node/Pyodide | float32 | training4 | -0.6% | -1.0% to -0.1% | -0.6% |
| Node/Pyodide | float64 | forward | -0.2% | -0.6% to 0.2% | 0.2% |
| Node/Pyodide | float64 | reverse | -0.1% | -0.2% to 0.2% | 0.2% |
| Node/Pyodide | float64 | jvp | 0.1% | -0.2% to 0.6% | 0.2% |
| Node/Pyodide | float64 | mixed | 0.1% | -0.3% to 0.7% | -0.1% |
| Node/Pyodide | float64 | training4 | -0.4% | -0.6% to 0.1% | -0.8% |
| Chrome Worker | float32 | forward | -0.4% | -1.3% to 0.0% | 0.0% |
| Chrome Worker | float32 | reverse | -0.4% | -2.0% to -0.0% | -0.5% |
| Chrome Worker | float32 | jvp | 3.0% | 2.5% to 3.8% | 2.6% |
| Chrome Worker | float32 | mixed | 2.2% | 1.7% to 2.8% | 2.3% |
| Chrome Worker | float32 | training4 | -0.5% | -0.9% to 0.0% | -0.3% |
| Chrome Worker | float64 | forward | 0.0% | -0.0% to 0.7% | -0.0% |
| Chrome Worker | float64 | reverse | -0.1% | -0.3% to 0.6% | -0.1% |
| Chrome Worker | float64 | jvp | 0.0% | -0.8% to 0.0% | -0.2% |
| Chrome Worker | float64 | mixed | 0.3% | -0.6% to 1.3% | 0.0% |
| Chrome Worker | float64 | training4 | -0.3% | -0.5% to -0.0% | -0.4% |

Float32 JVP and mixed AD become about 3%–5% and 2%–4% slower in the measured
core flows, respectively. Float32 four-step training medians improve modestly,
about 2.2% on CPU and 0.5%–0.6% on WASM. Float64 control results are near noise.
The precision benefit therefore has a measured forward-AD cost. Individual
regressions, timer quantization on small browser cases and every raw sample
remain visible. These shared-host observations establish no universal speed,
native Torch/GPU comparison, cold-start or peak-RSS improvement.

## Source and integration identity

The package contains 55 Python modules plus LICENSE/NOTICE. Relative to 0.52,
50 resources are unchanged, six change and `_transformer.py` is added.
Sorted concatenated source/license bytes change from
674,834 to 688,507;
local concatenated gzip changes from 155,674
to 158,623. These are source estimates,
not transferred bytes or measured memory. All 14 pinned Pyodide files and the
npm lock are rechecked. Dependency versions are unchanged.

Concurrent main updates through `0ef753cc0c194edf66b4aaf759ec7ee169511f41` are
preserved by fast-forward before the local feature commit. All 25 latest incoming
community-period/search-preview/company-comparison UI files remain exact upstream Git blobs.
Numerical source, tests, native catalog inputs and active exports are unchanged;
final frontend/build checks include the synchronized UI and its new checks.
The 161 active export identities match the final runtime source, and all new
immutable assets are reachable.

Machine results are in `packages/stridetorch/benchmarks/results/iteration-53.json`
and `iteration-53-integration.json`; five raw harness archives retain sources,
tests, oracle arrays, precision experiments, package artifacts, commands, logs
and timing samples. The inspected native Torch implementation is not included
in those archives. No CI trigger, tag, workflow dispatch, release, package upload
or online deployment is initiated by these local checks. Project renaming,
broader core coverage, further allocation/kernel work and an optional persistent
GPU execution chain remain part of the ongoing goal.
