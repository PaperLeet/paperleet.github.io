# StrideTorch 0.39: region-local average pooling

Version 0.39 adds `nn.AvgPool2d`, `nn.AdaptiveAvgPool2d` and functional
`avg_pool2d`, and replaces the existing adaptive average/area interpolation
implementation. The package name remains unchanged while browser-oriented
naming is under discussion. Main adoption and publication are separate steps;
this record does not establish that the candidate has been adopted.

## Numerical behavior and derivatives

The former weighted matrix implementation could multiply unrelated NaN/Inf
values by zero and contaminate other regions. Scaling samples before summation
also avoided finite overflow that native pooling preserves. The original probe
found six IEEE-class mismatches across nine cases. Ordered local reductions
preserve region membership and sum before dividing. Adaptive division follows
height then width; fixed pooling uses its actual divisor, including padding,
ceil tails and positive/negative `divisor_override` values.

Float32/float64 support reverse AD, JVP and reverse-over-forward. Float16
inference uses float32 accumulation; fixed pooling also supports int64
inference with truncation toward zero. Bool, unsupported integer types,
complex and half AD retain explicit unsupported boundaries. This does not add
general higher-order reverse AD or bicubic JVP.

The global adaptive 1x1 specialization follows mean's stateless backward rule.
Other adaptive outputs save input storage versions; modifying an output does
not invalidate backward. Tangent tapes save tangent versions without creating
a false reverse edge to the primal. Zero-sized adaptive outputs work in forward
evaluation and reject backward; empty batches and native empty-channel cases
are covered independently.

## Kernel selection

Exact partitions use strided additions for small blocks and ordered packed
reductions for larger blocks. Other regions use bounded packing, tiling,
streaming or scanning. A conservative temporary-workspace estimate is 32 MiB;
this is separate from output allocation and total process memory. Geometry
caches retain only bounded immutable index/weight arrays, never user tensors.

A first candidate's matrix adjoint failed the expanded random oracle: one
near-zero float32 gradient differed by 3.65e-7. This failure remains preserved
and the test tolerance was not relaxed. Adaptive adjoints now invert region
membership, gather contributions in native output order and accumulate them
without separable matrix reassociation. Three reproducible cancellation seeds
now match the native reference exactly. Nonfinite seeds affect only their
actual input regions. Large inverse gathers fall back to bounded ordered
scatter/loops. A preallocation lower bound rejects impossible inverse gathers
before constructing full-length indices. A float32 `1x1x2000000x1` backward
probe recorded source hashes for both revisions. Across two repeats, tracked
peak allocation fell from 48,007,128–48,008,368 to 16,006,320–16,006,400 bytes,
with identical gradients. These are tracked allocation values, not whole-process
RSS or WASM memory measurements; incidental shared-host timings are not an
end-to-end speed claim. See `adjoint_memory` in the core report.

Forward matrix shortcuts inspect live input values on every call and exclude
planes with mixed signs, nonfinite values or unsafe exponent ranges. This
also covers modifications through external NumPy aliases. Mixed-sign regions
use ordered reductions even when individual sample magnitudes look safe.

## Verified local checks

- Focused average-pooling, spatial and graph-lifecycle checks: **1,938 passes**.
- Actual Node Pyodide and Chrome Worker: **2,154 checks in 38 suites each**,
  Pyodide 0.29.4 / NumPy 2.2.5, without native Torch in the runtime.
- Isolated Python 3.10 / NumPy 1.26 wheel: **2,167 passes**. Public/private
  wheels and source archives match all **49** source/license resources.
  Independent namespace aliases, Module pickle, VJP/JVP and uninstall ownership
  checks pass.
- Two expanded random oracles cover **720 trials**, five forward choices and
  five backward choices, including **8,575 native gradient comparisons**.
  Inputs include integer-valued and random floating values, half/long inference,
  nonfinite values and finite overflow. Invalid native cases remain invalid.
- Export identity retains all **9,123** prior immutable payloads byte-for-byte;
  all **158** added payloads are reachable. Private package identity matches
  the current application catalog and server.

The final full Python suite passed **34,887 checks,
1 optional skip**, in 302.99s. Full frontend checks and
both typed builds pass. The original native catalog passed **78 problems,
10,218 executions, 342 mutants, 3 repeats**, in 152.94s with the
unchanged 600s budget. Actual exported WASM passed **78/78 modules and 342
mutants**, in 309743ms, with concurrency 2 and no budget exceeded.
Its attention runtime also ran all **2,154 independent checks in 38 suites**.
Earlier freezes (34,719 and 34,879 full passes) are retained separately and do
not replace final-source evidence.

## Performance and backend scope

Final measurements retain regressions as well as wins. Existing adaptive/area
and CNN flows compare the same APIs with 0.38; new fixed-pooling/IEEE flows
compare automatic dispatch with the same version's generic scan and loop.
Each runtime has two dtypes and two fresh children, nine ABBA/BAAB blocks,
warmups, GC enabled and equal loop counts. Every block checks outputs, dtypes,
gradients and actual updated SGD/momentum parameters where applicable. Negative
time reductions mean slower; the shared host and effective vendor thread
counts remain uncontrolled. These are not native-Torch or cross-runtime
throughput comparisons. Overlapping/source-changing pilots are retained but
excluded from the final measurements.

| Complete averaging workflow | CPU time reduction | WASM time reduction |
| --- | ---: | ---: |
| legacy-global-inference | 11.0 to 18.8% | 8.7 to 19.1% |
| legacy-global-backward | 9.8 to 16.1% | -8.1 to 9.8% |
| legacy-blocks-inference | -64.4 to -58.0% | -41.0 to -27.6% |
| legacy-blocks-backward | -21.5 to -6.4% | -10.9 to -4.6% |
| legacy-irregular-inference | -343.3 to -235.6% | -299.3 to -283.7% |
| legacy-irregular-backward | -279.3 to -256.8% | -208.6 to -191.2% |
| legacy-upsample-inference | -176.8 to -167.3% | -157.3 to -135.7% |
| legacy-upsample-backward | -124.3 to -109.3% | -96.9 to -87.4% |
| legacy-identity-inference | 77.0 to 81.8% | 83.4 to 85.7% |
| legacy-identity-backward | 33.5 to 38.7% | 52.2 to 56.6% |
| legacy-area-inference | -346.9 to -241.2% | -287.0 to -274.4% |
| legacy-area-backward | -301.3 to -255.6% | -220.6 to -211.8% |
| legacy-cnn-tiny-inference | 2.8 to 11.0% | 0.9 to 5.8% |
| legacy-cnn-tiny-sgd | -0.6 to 4.7% | -1.0 to 10.0% |
| legacy-cnn-image-inference | -0.2 to 6.8% | -2.4 to 3.4% |
| legacy-cnn-image-sgd | -2.2 to 8.9% | 4.1 to 8.5% |
| new-avg-block-inference | 51.7 to 55.9% | 47.0 to 49.4% |
| new-avg-block-backward | 65.9 to 67.3% | 65.0 to 66.0% |
| new-avg-block-jvp | 62.7 to 65.9% | 61.4 to 63.0% |
| new-avg-overlap-inference | 1.6 to 6.2% | 1.8 to 6.4% |
| new-avg-overlap-backward | 33.4 to 36.8% | 39.3 to 42.0% |
| new-avg-overlap-jvp | 25.4 to 30.4% | 30.0 to 38.8% |
| new-avg-ceil-inference | 24.6 to 28.9% | 25.8 to 28.5% |
| new-avg-ceil-backward | 27.0 to 36.8% | 33.3 to 37.4% |
| new-avg-ceil-jvp | 25.6 to 31.9% | 30.4 to 36.3% |
| new-avg-divisor-inference | 22.6 to 29.6% | 30.5 to 34.1% |
| new-avg-divisor-backward | 23.7 to 28.4% | 26.9 to 33.2% |
| new-avg-divisor-jvp | 26.0 to 29.7% | 28.6 to 33.7% |
| new-ieee-nonfinite-inference | 67.5 to 68.4% | 66.3 to 67.7% |
| new-ieee-overflow-inference | 65.2 to 66.6% | 62.4 to 66.4% |
| new-ieee-nonfinite-backward | 31.6 to 36.4% | 29.3 to 37.3% |
| new-ieee-overflow-backward | 28.0 to 34.3% | 33.2 to 37.5% |
| new-cnn-tiny-inference | 13.5 to 23.8% | 17.6 to 20.0% |
| new-cnn-tiny-sgd | 5.5 to 17.9% | 13.3 to 20.1% |
| new-cnn-image-inference | 11.8 to 20.4% | 11.1 to 20.5% |
| new-cnn-image-sgd | 19.6 to 28.9% | 21.8 to 26.7% |

## Existing numerical workflow controls

Twenty-one existing inverse/solve/JVP/SGD/Schedules workflows compare 0.38 with
0.39 using the same public APIs and optional-feature lookup. Raw paired samples
include absolute microsecond times.

| Existing workflow | CPU time reduction | WASM time reduction |
| --- | ---: | ---: |
| tiny-inference | -2.3 to 0.2% | -0.7 to 4.6% |
| small-inference | -1.1 to 3.3% | -4.2 to -0.0% |
| large-inference | -2.2 to 0.9% | 0.3 to 1.7% |
| upper-inference | -6.0 to 1.3% | -1.8 to 1.9% |
| batch-inference | -1.9 to 3.8% | -1.8 to 0.8% |
| tiny-backward | -18.2 to 3.1% | -11.0 to 3.9% |
| small-backward | -3.2 to 3.3% | -1.1 to 1.8% |
| large-backward | -3.2 to 0.7% | -0.6 to 0.6% |
| sgd-factor | -2.8 to 3.7% | -1.9 to 6.9% |
| optional-sgd | -7.2 to 2.1% | -4.9 to 2.5% |
| spd-chain | -1.0 to 4.1% | -1.3 to 1.9% |
| jvp | -18.0 to 4.5% | -0.7 to 4.9% |
| large-jvp | -2.5 to 0.5% | -0.7 to 1.8% |
| jvp-backward | -3.4 to -0.2% | -0.1 to 3.2% |
| large-jvp-backward | -0.6 to 0.7% | -1.1 to 0.6% |
| fixed-factor-jvp | -0.9 to 0.2% | -1.7 to 0.8% |
| legacy-inference | -0.4 to 5.9% | -0.7 to 0.9% |
| legacy-optional-inference | -0.4 to 1.1% | -0.2 to 0.9% |
| generic-control | -1.4 to 3.3% | -0.2 to 4.5% |
| cholesky-control | -6.3 to 5.7% | -3.9 to 1.5% |
| unary-control | -3.0 to 0.6% | -9.2 to -2.0% |

A diagnostic follow-up retained all original results and repeated tiny backward,
JVP, generic solve and unary controls with 100 calls/sample, 15 paired blocks,
explicit GC collection outside each block and normal/reversed package
initialization. Eight fresh children cover both dtypes and CPU/WASM. Their
median differences span -1.4% to +2.2%; the larger original outliers did not
reproduce consistently under this sampling. This does not establish the same
bound for every other workflow, and does not replace the original measurements.
See `diagnostic_existing_flow_followup` in the core report.

Sixteen fresh alternating processes measure cold namespace imports, reporting
NumPy included/preloaded separately. Namespace/source growth is separate from
network download, browser cold start and retained memory.

This iteration uses NumPy CPU and Pyodide WASM. WebGPU still requires resident
tensor storage, transfer semantics, compatible gradients and a demonstrated
whole-flow advantage before it can become a backend. No GitHub CI, tag,
workflow dispatch, package upload or online publication was performed.

Reproducible source identities, raw timing samples, both random-oracle drivers,
package/runtime harnesses, retained prototype failures and local application
receipts are archived in `iteration-39.json`, `iteration-39-integration.json`
and their SHA-256 identified harness archives. Main adoption requires a fresh
remote/policy check and a separate verification record.

Final synchronization preserves all ten incoming frontend/style/check/documentation
files through `33eacaa33ecfaeec0df638a4664ac0ae98938a73` byte-for-byte. The complete current
frontend checks and both typed builds pass again. All 49 computational/license
resources and 161 active exports retain their full-test identities. Python,
package,problem and catalog inputs did not change. See `iteration-39-main-sync.json`.

A later synchronization preserves six sequential-study UI/check/style/documentation
files through `c6faaed857c05c8a3f5248d4253f41dff72d8807`. The affected preparation-guide interaction suite and
both typed builds pass on this final tree. All 49 runtime/license resources and
161 active exports remain byte-identical. See `iteration-39-main-sync-final.json`.
