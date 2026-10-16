# Lightweight torch 0.19: grouped convolution

This iteration adds `nn.Conv1d/2d/3d`, functional `conv1d/2d/3d` and standard
`torch.nn.modules.conv` import aliases. The independent library continues to use
NumPy alone, ordinary torch imports through explicit host aliases/the script
launcher, Apache-2.0 resources and disjoint application/private package paths.

## Contract

Real float32/float64 inputs and matching weights/bias support batched/unbatched
layouts, groups/depthwise, stride/dilation, tuple/integer padding and same/valid.
Modules add zeros/reflect/replicate/circular padding and the common registered
parameter, dtype, initialization and in-memory checkpoint lifecycle. Same padding
requires unit stride. Reflect pads are smaller than the input axis; circular
pads cannot wrap more than once. General transpose convolution, float16/complex
kernels, GPU execution and higher-order reverse AD remain unsupported.

A shared analytical VJP computes only active input/weight/bias gradients.
Input and weight versions remain guarded even on bias-only tapes, without
retaining their original Tensor objects when neither array is required.
Nonzero-padding pullbacks accumulate duplicate source positions. Empty batches
and the native zero-input-channel output convention have explicit contracts.
Bilinear forward AD retains both convolution terms, including absent tangents
represented by explicit zeros: this preserves zero reverse dependencies and
zero-times-infinity/NaN behavior. Bias primals do not acquire tangent gradients.

## Kernel choice and workspace

Small/pointwise inputs use zero-copy window metadata and direct contractions.
Larger finite inputs use packed matrix products if the estimated logical NumPy
workspace fits 32 MiB. The estimate includes packed inputs, batch weight
contributions, output intermediates and padded input gradients; it is not a
NumPy/BLAS peak-memory bound. Inputs are repacked during backward rather than
retaining expanded im2col on the tape. Nonfinite operands/seeds retain direct
products. Forced direct/matrix/auto private test modes exercise all dispatch paths.

## Correctness and retained failures

The focused collection passes **1389 tests**: 1182 native-reference cases and
207 NumPy-only contracts, each covering direct/matrix/auto. Tests include scalar
convolution/gradient formulas, primal and JVP-reverse finite differences,
noncontiguous layouts, groups, padding adjoints, dtype and empty boundaries,
selective mutation guards, object lifetime, workspace fallback and complete
Conv/GroupNorm/SiLU/CE/AdamW/scheduler checkpoint-resume training. Separate native
probes retain geometry, padding, empty and nonfinite/partial-JVP observations.
Node/Pyodide 0.29.4 and Chrome 155.0.8059.82 Worker each pass **622 contracts**
against in-VM source/test hashes using NumPy 2.2.5.

Initial native tests retained 12 failures because numeric module padding metadata
used a list instead of native's tuple. This was corrected without numerical
changes; string same-padding retains native's list. The first all-kernel suite
retained 3 failures because a new contiguity test called the existing unsupported
Tensor.stride method. It now checks NumPy storage strides against native strides;
the outputs were already contiguous. Tensor.stride is not added by this release.
The raw prototype's first orchestrator failed on a malformed larger-case fixture;
the corrected distinct run retains per-sample logs. Original failed inputs/logs
are archived. No mathematical threshold was relaxed.

## Measured production path

The table is percent elapsed-time reduction for production auto dispatch versus
same-source direct convolution, including Tensor validation/allocation, packing,
graph construction, squared-mean loss and active reverse passes. Forward-only,
weight-only and bias-only cases are labeled separately. Each CPU/WASM and dtype
series uses four sequential fresh processes in direct/auto/auto/direct order,
nine batches, twenty base iterations and three warmups; tiny cases multiply
iterations by four, larger cases divide by eight. All raw samples/process medians
and frozen source/harness hashes are retained. CPU uses NumPy 2.5.4, WASM uses
Pyodide 0.29.4/NumPy 2.2.5. Thread-limit environment variables request two CPU
threads; this does not prove BLAS's actual thread count. These are local shared
machine measurements, with no claim about native Torch speed, browser startup,
whole-application latency or peak memory. Negative values mean slower.

| Case | CPU float32 | CPU float64 | WASM float32 | WASM float64 |
| --- | ---: | ---: | ---: | ---: |
| tiny_1d | -0.8% | -0.4% | 1.9% | 0.1% |
| pointwise_2d | 0.4% | 2.7% | 2.7% | -0.7% |
| spatial_2d | 86.1% | 86.9% | 54.1% | 55.5% |
| depthwise_2d | 10.4% | 27.4% | -0.6% | -3.4% |
| dilated_1d | 37.5% | 36.6% | 21.6% | 21.7% |
| volume_3d | 83.9% | 84.3% | 52.6% | 53.4% |
| larger_2d | 94.3% | 94.4% | 55.8% | 46.1% |
| spatial_forward | 88.5% | 88.2% | 69.7% | 70.0% |
| spatial_weight | 88.9% | 88.6% | 56.3% | 56.1% |
| spatial_bias | 83.8% | 83.4% | 64.9% | 65.2% |

Tiny and pointwise dispatches retain direct kernels; their small differences
are process variance/dispatch cost. WASM depthwise results retain the observed
0.6–3.4% regression; matrix dispatch is not a universal improvement. Spatial and
3D cases provide the strongest measured benefit. No previous-release comparison
is possible because 0.18 did not provide these convolution operations.

The separate raw NumPy prototype experiment checks packing plus forward/dX/dW/db
but excludes Tensor graphs and reuses packed inputs within a training call.
Production repacks during backward. Its faster percentages must not be confused
with the complete Tensor measurements above. The first failed prototype has no
final report; only actually emitted evidence is retained.

## Release gates

After safe rebase onto upstream `6f22f674`, the new GroupNorm course joins
this release's complete **78-problem** catalog. Library, benchmark and independent
contract source hashes remain invariant across the rebase; the curriculum and UI
changes are retained. The merged infrastructure suite passes **19783 tests,
1 optional Docker skip, 10 warnings**, 217.79s. Native catalog passes
**78 problems /10218 executions /3 repeats /342 mutants**, 143.196s.
Actual full WASM passes **78/78**, 428.190s, with no browser
budget excess. These durations are correctness observations, not comparative
performance measurements. All local producer handles were observed terminal.

The standalone wheel passes **642 tests** on isolated Python3.10.19 / NumPy1.26.4
without native Torch. Both standalone/private wheels and sdists have byte-identical
39 Python modules plus Apache LICENSE/NOTICE, matching Node/Chrome VM snapshots;
installed paths are disjoint. The private package's source/catalog build inputs
are unchanged by the upstream curriculum/UI merge. All frontend checks, production
build and tested-main Pages policy checks pass after rebase. Initial JS graph is
404491 raw /130131 gzip bytes, including upstream UI changes.

Export preserves all **5808 published hashed assets** byte-identically. All158
new assets are reachable. The autograd runtime is
`runtime-423e0a03aaaf78c687ea.json`; schedules use
`runtime-92c4687d6653d39d994d.json`. Autograd transfer size is430467raw /98993gzip
bytes, +19310raw /4339gzip versus0.18 under identical gzip settings. No cold-start
or peak-memory result is inferred.

[Complete evidence](../../packages/stridetorch/benchmarks/results/iteration-19.json)
retains all gates, probes, raw prototype versus production measurements and
failures. `iteration-19-harness.tar.gz` archives387 runner/source/test/log files.
The original timing parent handle became unavailable across a context transition;
16 child exit0 receipts, the success report, frozen hashes and the absence of
owned timing processes were independently observed. No parent exit receipt is
invented. The fully verified0.18 predecessor's exact main/Quality/Pages/live proof
is carried in `iteration-18-remote-adoption.json`. Exact remote0.19 adoption follows
publication; local tests alone do not establish deployment.
