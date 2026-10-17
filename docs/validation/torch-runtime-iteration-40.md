# StrideTorch 0.40: cached ordered average-pooling plans

Version 0.40 reduces repeated geometry work in 0.39's ordered kernels. Public
operators and supported numerical boundaries are unchanged; no GPU backend is
added. The independent package keeps its current name while browser-oriented
naming is under discussion. Main adoption has its own verification record;
package publication remains separate.

## Numerical behavior, ownership and bounded plans

Plans contain only readonly indices, masks and dtype-specific divisors. They
never retain user tensors, input arrays, gradient seeds or computed values.
The LRU holds at most 64 plans, each with at most 64 KiB of array data: 4 MiB
of array data across this new cache, excluding Python metadata. Existing
geometry/matrix caches retain their separate capacities.

Contiguous input can use flat spatial gathering; other layouts keep coordinate
gathering without copying the entire input to contiguous storage. Flat indices
are optional when they would exceed the per-plan byte cap. Large plans retain
the earlier bounded coordinate/tiled/streaming/scan paths, and huge inverse
indices still have a preallocation guard. The separate 32 MiB packing estimate
is not a total-process or output-allocation limit.

Both gathers own their new storage. Ordered accumulation, sum before scaling,
height-then-width division, local NaN/Inf behavior and cancellation residuals
are preserved. Mixed-sign planes still avoid matrix reassociation. The output
copies the final accumulated values so it retains no packed prefixes; final
+0 preserves native zero signs. Identity pullbacks skip redundant copying and
divisions by one. Storage-version and alias semantics remain covered.

## Final local validation

- Targeted native/independent/spatial/graph checks: **2,027 passes**.
- Full Python suite: **34,976 passes, 1 optional skip**,
  269.48s.
- Actual Node Pyodide and Chrome Worker: **2,243 checks in 39 suites each**,
  Pyodide 0.29.4 / NumPy 2.2.5, without native Torch in the browser runtime.
- Isolated Python 3.10 / NumPy 1.26 wheel: **2,256 passes**. Public/private
  wheels and source archives match all **49** source/license resources.
  Both namespaces retain cached-pooling ownership and independently survive
  the other's uninstall. An added test initially assumed two namespaces even
  after uninstall; its assertion was corrected, with no runtime/artifact change.
  The first failure is retained in the harness archive.
- Native random oracles: **720 trials / 8,575 gradient comparisons**, five
  forward and five backward paths, including dtype/geometry/IEEE boundaries.
- Original native catalog: **78 problems, 10,218 executions, 342 mutants,
  3 repeats**, 130.56s, unchanged 600s budget.
- Actual exported WASM: **78/78 modules, 342 mutants**, 316554ms,
  concurrency 2, no budget exceeded. Its attention runtime also executes the
  **2,243 independent contracts/39 suites** against the actual exported code.
- All **9,281** prior immutable payloads remain byte-identical, all **158** new
  payloads are reachable, and **161** active files are SHA-256 frozen. The
  smaller Schedules profile remains six resources. Full frontend checks and
  both typed builds pass on the recorded integration input.

Source comparison finds **47 byte-identical shared resources**; only version
metadata and average-pooling implementation change. Public API bytes and the
rest of the averaging AST are unchanged outside forward/backward dispatch,
three plan helpers, three globals and the namedtuple import. This supports,
but does not replace, the regression checks.

## Complete-flow timings against 0.39

All 36 averaging/area/CNN flows compare supported 0.39 APIs with 0.40. Eight
sequential fresh children cover two dtypes and two processes per CPU/WASM,
nine ABBA/BAAB blocks, warmups, GC enabled and equal call counts. Each block
checks outputs, gradients, dtypes and actual updated SGD/momentum parameters.
Negative time reductions mean slower. These measurements do not establish
native-Torch or cross-runtime throughput, nor recovery of every earlier
regression against 0.38's numerically incorrect weighted path.

| Complete flow | CPU time reduction | WASM time reduction |
| --- | ---: | ---: |
| legacy-global-inference | -7.8 to 8.6% | 2.6 to 5.1% |
| legacy-global-backward | -1.9 to 6.3% | -1.5 to 10.7% |
| legacy-blocks-inference | -2.9 to 1.7% | -0.4 to 11.8% |
| legacy-blocks-backward | -5.7 to 2.9% | -2.3 to 10.1% |
| legacy-irregular-inference | 25.4 to 35.5% | 18.6 to 27.8% |
| legacy-irregular-backward | 14.8 to 27.4% | 13.0 to 18.8% |
| legacy-upsample-inference | 46.3 to 53.6% | 38.4 to 40.4% |
| legacy-upsample-backward | 27.0 to 38.7% | 26.1 to 39.2% |
| legacy-identity-inference | -14.5 to 5.9% | -6.0 to 1.7% |
| legacy-identity-backward | -2.7 to 8.5% | 0.3 to 11.9% |
| legacy-area-inference | 22.8 to 30.4% | 11.8 to 17.3% |
| legacy-area-backward | 22.9 to 30.8% | 14.0 to 20.5% |
| legacy-cnn-tiny-inference | -3.2 to 6.2% | -6.4 to 18.4% |
| legacy-cnn-tiny-sgd | -3.7 to 6.1% | -5.2 to 2.8% |
| legacy-cnn-image-inference | -7.0 to 4.2% | -0.9 to 2.5% |
| legacy-cnn-image-sgd | -1.3 to 4.2% | -1.8 to 5.4% |
| new-avg-block-inference | -1.2 to 4.1% | -9.8 to 6.9% |
| new-avg-block-backward | -1.4 to 9.2% | -4.0 to 0.5% |
| new-avg-block-jvp | -1.0 to 2.5% | -4.3 to 6.3% |
| new-avg-overlap-inference | -5.1 to 3.2% | -2.4 to 2.4% |
| new-avg-overlap-backward | -0.8 to 3.8% | -1.5 to 10.2% |
| new-avg-overlap-jvp | -1.7 to 2.4% | -1.4 to 3.8% |
| new-avg-ceil-inference | -2.8 to -1.7% | -3.6 to 2.3% |
| new-avg-ceil-backward | -3.7 to 5.8% | -6.8 to 9.3% |
| new-avg-ceil-jvp | 0.3 to 6.6% | -4.1 to 0.6% |
| new-avg-divisor-inference | -8.0 to 2.3% | -10.5 to 2.4% |
| new-avg-divisor-backward | -0.4 to 4.9% | -2.3 to 2.4% |
| new-avg-divisor-jvp | -0.6 to 6.0% | 0.8 to 8.2% |
| new-ieee-nonfinite-inference | 42.5 to 45.0% | 38.8 to 42.2% |
| new-ieee-overflow-inference | 41.0 to 43.6% | 37.4 to 43.3% |
| new-ieee-nonfinite-backward | 23.4 to 32.7% | 27.3 to 31.6% |
| new-ieee-overflow-backward | 28.7 to 31.8% | 19.6 to 30.0% |
| new-cnn-tiny-inference | -12.7 to 2.3% | -11.3 to 13.6% |
| new-cnn-tiny-sgd | -6.6 to 0.3% | 0.3 to 3.9% |
| new-cnn-image-inference | -1.7 to 1.4% | -3.1 to 1.6% |
| new-cnn-image-sgd | -1.4 to 8.1% | -1.1 to 12.0% |

## Other existing numerical controls

Twenty-one old solve/inverse/JVP/SGD/Schedules flows use the same APIs and
optional-feature lookup on both sides. Raw reports retain absolute times and
all paired blocks, including outliers.

| Existing flow | CPU time reduction | WASM time reduction |
| --- | ---: | ---: |
| tiny-inference | -0.1 to 1.6% | -0.1 to 0.7% |
| small-inference | -2.2 to 0.8% | -1.0 to 1.5% |
| large-inference | 0.1 to 1.2% | -0.4 to 0.8% |
| upper-inference | -0.5 to 1.4% | -0.4 to 0.1% |
| batch-inference | -0.2 to 0.3% | 0.1 to 0.6% |
| tiny-backward | -1.0 to -0.5% | 0.1 to 0.5% |
| small-backward | -0.3 to 1.0% | 0.0 to 0.4% |
| large-backward | -1.8 to 1.6% | -0.4 to 0.8% |
| sgd-factor | -1.6 to 0.6% | -1.0 to -0.2% |
| optional-sgd | -2.3 to -0.4% | -1.0 to 1.4% |
| spd-chain | -0.7 to 0.7% | -0.1 to 1.5% |
| jvp | -7.5 to 3.0% | -2.6 to 1.6% |
| large-jvp | -0.2 to 1.0% | -0.2 to 0.5% |
| jvp-backward | -20.7 to -0.4% | -0.1 to 0.2% |
| large-jvp-backward | -1.8 to 1.1% | -0.0 to 1.0% |
| fixed-factor-jvp | -1.7 to -0.2% | -1.3 to 0.6% |
| legacy-inference | -0.5 to 0.5% | -1.8 to 1.5% |
| legacy-optional-inference | -0.9 to 1.1% | -1.7 to -0.1% |
| generic-control | -0.5 to 5.5% | 0.3 to 1.6% |
| cholesky-control | -4.6 to -0.2% | -0.2 to 8.1% |
| unary-control | -2.7 to 0.4% | -2.3 to 0.7% |

Prespecified longer-sample four-flow diagnostics and separate identical-source
A/A calibration retain all original measurements. Normal/reversed namespace
initialization and explicit GC collection outside paired blocks help inspect
sampling effects; they do not replace the main reports or establish a uniform
latency bound. Reverse-through-JVP also has a separate CPU follow-up. Shared
host activity and effective vendor thread counts remain uncontrolled. See
`diagnostic_existing_flow_followup` and `diagnostic_fairness` in the core report.

Sixteen alternating fresh CPU processes record cold namespace imports, with
NumPy included/preloaded reported separately. Raw source and gzip estimates
are separate from wire size, browser cold start and retained memory.

## Tracked allocation samples

Two warmed repeats cover each dtype and case. Forward samples include output;
the huge-adjoint case includes forward/backward. Persistent prewarmed cache
metadata is excluded. These are not process RSS, WASM memory or throughput.

| Case / dtype | 0.39 tracked peak bytes | 0.40 tracked peak bytes |
| --- | ---: | ---: |
| irregular / float32 | 49,295–49,295 | 29,400–29,400 |
| upsample / float32 | 15,400–15,400 | 8,912–8,912 |
| coordinate / float32 | 216,490–216,490 | 194,424–194,424 |
| huge-adjoint / float32 | 16,007,720–16,007,983 | 16,007,696–16,007,720 |
| irregular / float64 | 93,475–93,475 | 51,000–51,000 |
| upsample / float64 | 20,888–20,888 | 14,288–14,288 |
| coordinate / float64 | 315,332–315,332 | 263,424–263,424 |
| huge-adjoint / float64 | 32,007,680–32,007,696 | 32,007,680–32,007,696 |

The existing huge-narrow-input allocation bound remains verified. The new
cache's separately capped metadata is a deliberate memory/latency tradeoff.

## Reproduction and publication boundary

Both earlier prototypes, all pilot/final timing samples, random-oracle drivers,
exact source/test identities, package/runtime harnesses and retained failures
are in `iteration-40.json`, `iteration-40-integration.json` and their SHA-256
identified harness archives. See any later main-synchronization and adoption
records for their exact incoming files and verified commits.

This iteration uses NumPy CPU/Pyodide WASM. WebGPU still needs resident storage,
explicit transfers, compatible gradients and a measured whole-flow benefit
before it can become a backend. No GitHub CI, tag, workflow dispatch, package
upload or online publication was performed.

## Latest main synchronization

The complete frontend checks and both typed builds pass on
`7060b745fd4217daf8430760bd2e001b75a7dd87`. Its six-file validated snapshot is retained;
overlapping files updated by a later validated commit are explicitly identified
in `iteration-40-main-sync.json`.

Twelve later feed-outline frontend/style/check/documentation files through
`69513e870ea801f4bf5a7cf3e88f563f26659b33` are retained byte-for-byte. Package dependencies
and lockfile remain unchanged. Five affected interaction/reading suites and
both typed builds pass on this final tree. All 49 runtime/license resources
and 161 active exports preserve their full numerical/WASM-test identities.
See `iteration-40-main-sync-final.json`.
