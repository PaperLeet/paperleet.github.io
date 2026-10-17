# Torch runtime iteration 45: volumetric average pooling

## Result and scope

Version 0.45 adds `nn.AvgPool3d`, `nn.AdaptiveAvgPool3d` and their functional entry points to the independent NumPy/Pyodide package and the PaperLeet vendored/browser runtimes. Existing `torch` aliases and module pickling remain available. This iteration retains the current package name; the proposed BrowserTorch rename has not been selected.

Fixed pooling supports stride, padding, ceil mode, padding inclusion and signed nonzero integer divisor overrides. Adaptive pooling supports scalar/triple output sizes, per-axis `None`, zero-sized forward outputs and the native empty-global specialization. Float32/float64 reverse AD, JVP and reverse-over-forward are verified. Fixed int64 inference and adaptive float16 inference are covered; native CPU fixed float16 pooling is explicitly unsupported.

## Computation and memory

Small disjoint blocks use strided additions and broadcast adjoints. Other regions use ordered packing, recursive output/plane tiling or bounded single-region streaming; reverse paths use ordered loop/scatter blocks. Non-contiguous gradient seeds are gathered by block, avoiding an implicit full-seed reshape copy. Adaptive division follows depth, height and width order. Masks prevent invalid/padded samples and nonfinite gradient seeds from contaminating unrelated regions.

Up to 32 small geometries cache readonly coordinate/count arrays and optional flat gather indices/masks. Each gather plan is at most 128 KiB. Contiguous input gathers directly into its accumulation buffer; other layouts retain coordinate gathering. No cache holds input values, tensors or user storage. The 32 MiB packing estimate excludes input/output allocations and is not a total-process or measured peak-memory claim.

Global 1×1×1 pooling shares the stateless mean rule and canonicalizes Fortran leading axes to native contiguous output metadata while preserving unambiguous channels-last output. An optional output formatter is the only arithmetic-helper change shared with existing 1D/2D averages.

## Verification

| Check | Observed result |
|---|---|
| Full CPU repository/package tests | 38,329 passed, 1 skipped, 494.57 s |
| New 3D contracts | 724 passed; four forward/backward route pairs |
| Installed native PyTorch 2.14.1 | 151 cases × four adapter routes; 2,892 array comparisons |
| Node/Pyodide and actual Chrome Worker | 4,701 independent contracts each, 43 suites |
| Minimum installed public wheel | Python 3.10.19 / NumPy 1.26.4, 4,714 tests, native Torch absent |
| Public/private ownership | Both namespaces, aliases, pickle/checkpoints, VJP/JVP and independent uninstall survival |
| Native PaperLeet catalog | 78 problems, 10,218 executions, 342 mutants, three repeats; 417.741 s < 600 s |
| Exported browser/WASM catalog | 78/78, 342 mutants, 4,701 independent contracts; 341.699 s |
| Frontend | Driver/policy checks, complete npm check, normal build and current Pages-profile build |
| Identity | 51 source/license resources; 161 active export hashes; 10,071 prior immutable payloads preserved and 158 reachable payloads added |

Native comparisons include CDHW/NCDHW, C/F/strided/channels-last input and seed layouts, local NaN/Inf, empty global input, values, exact output strides, VJP, JVP and reverse-tangent gradients. Non-singleton leaf-gradient strides agree. Four channel-singleton cases retain imported singleton gradient strides that differ from native canonical singleton strides; all indexed values agree. Every raw difference is recorded rather than counted as exact metadata equality.

A deliberately ill-conditioned global float32 example `[-0,-0,1e20,-1e20,1,2,3,4]` yields 0 from native SIMD mean and 1.25 from NumPy mean, while fixed ordered pooling yields 1.25. Global pooling follows this library's mean contract; there is no bitwise-equivalence promise across numerical backends for cancellation-dominated sums. Ordinary native comparisons use explicit dtype-appropriate tolerances.

## Complete-flow timings

Baseline: supported 0.44 2D spatial pooling followed by 1D depth pooling, including reshapes/permutations. Candidate: direct 0.45 3D pooling. A same-source 0.45 composition is retained as a control. The floating composition is mathematically equivalent on the checked finite random inputs; it is not a bitwise replacement for ordered 3D reduction on exceptional or cancellation-dominated values.

Four fresh processes per runtime, alternating variant/runtime order, five samples per case, 12 ordinary calls or three training steps per sample. All 42 flows validate values/gradients or one complete CNN SGD update before timing. Every sample and child median is retained. Shared-host results describe these workloads, not native Torch/GPU speed or cross-runtime comparisons.

Percent reduction in elapsed time versus the 0.44 composition; negative values mean the direct 3D operation is slower. The training flow is Conv3d → SiLU → AvgPool3d → global adaptive average → Linear → loss/backward/SGD.

| Case | Dtype | CPU | Node/WASM | Chrome Worker/WASM |
|---|---|---:|---:|---:|
| fixed / forward | float32 | 46.3% | 44.4% | 41.3% |
| fixed / forward | float64 | 40.8% | 32.6% | 37.5% |
| fixed / backward | float32 | 31.7% | 29.5% | 35.6% |
| fixed / backward | float64 | 34.2% | 34.6% | 33.3% |
| global / forward | float32 | 59.9% | 59.0% | 57.1% |
| global / forward | float64 | 59.1% | 59.0% | 66.7% |
| global / backward | float32 | 43.5% | 41.3% | 40.9% |
| global / backward | float64 | 43.0% | 42.3% | 42.9% |
| irregular / backward | float32 | 8.1% | 7.4% | -3.2% |
| irregular / backward | float64 | 7.8% | 21.5% | 12.2% |
| upsample / backward | float32 | -47.2% | -44.7% | -46.1% |
| upsample / backward | float64 | -41.3% | -48.6% | -45.1% |
| overlap / forward | float32 | -92.6% | -129.7% | -111.5% |
| overlap / forward | float64 | -77.0% | -128.3% | -116.1% |
| overlap / backward | float32 | -130.4% | -127.0% | -113.1% |
| overlap / backward | float64 | -122.7% | -125.4% | -113.1% |
| cnn / training | float32 | 12.6% | 12.7% | 12.5% |
| cnn / training | float64 | 12.1% | 6.2% | 14.0% |

The new API is useful beyond speed: native-order volumetric reductions and derivatives can be expressed directly. Overlapping fixed windows and some adaptive adjoints can remain slower than the reassociated composition. These are explicit follow-up optimization targets; the evidence does not support a universal speedup claim.

## Source footprint

| Measure | 0.44 | 0.45 |
|---|---:|---:|
| Source/license bytes | 590,273 | 607,595 |
| Local concatenated gzip estimate | 137,747 | 141,441 |

These source estimates do not measure wheel size, network transfer, peak memory or GPU storage. Forty-seven prior source/license resources are unchanged; the existing init/API/mean-helper files change and one 3D average module is added.

## Reproducibility and integration

The preliminary uncached implementation passed its focused/Node/Chrome/minimum-wheel checks. Its owned full CPU test process was deliberately stopped before source optimization, with its process tree, partial log and nonzero receipt retained in `before-index-plans`; observation timeouts were not treated as termination. The optimized source was frozen anew, and every adapter-dependent gate and full-flow timing was run on the final hashes. The native original-catalog run continued because it uses native Torch and unchanged native problem inputs.

Repository benchmark: [`bench_average_pooling_3d.py`](../../packages/stridetorch/benchmarks/bench_average_pooling_3d.py). Machine-readable source/integration reports and four raw harness archives are in [`benchmarks/results`](../../packages/stridetorch/benchmarks/results/). Reports include exact source/test/export/dependency hashes and installed distribution identities. Native comparison arrays and all timing samples are retained in the archives.

Seven release-policy files remain byte-identical to canonical commit `13d2782a494415ba72bc604a0ef7e4c4f875ba25`. Validation is local. No tag, workflow dispatch, release, package upload or online deployment is initiated by this iteration. The broader independent-library/core-efficiency/backend goal remains active.

## Upstream UI integration

Preserved all 16 web/docs changes from `cc1da1421eb100e362de5334180365305499dee2` through `11851e524fcaa83ba3f31416035a56b55b56156f` and reran the complete frontend check and both builds. Scope: 16 web/docs files only: submitted-source readers and optional affiliation verification. All 51 runtime resources,43 math test suites,161 active exports,maintained WASM driver,Python SDK/server/catalog and seven release policies unchanged; preserve existing mathematical/package/native/WASM/performance evidence and rerun complete frontend plus both builds after local rebase.

Preserved all 6 web/docs changes from `11851e524fcaa83ba3f31416035a56b55b56156f` through `8f5ce64dc017a4c5a363ef3d3cb083ab58b7fb16` and reran the complete frontend check and both builds. Scope: Six web files only: searchable test details preserving filters/run boundaries. Mathematical source/tests, SDK/native inputs, exports and release policies unchanged. Repeat complete frontend and both builds on the combined source.

The runtime/test/export/SDK/catalog/release-policy inputs to the CPU, native, installed-package, exported-WASM and performance evidence remain unchanged. Historical upstream hashes are checked against their Git blobs; current files match the latest union. Every original/latest frontend receipt is retained. Existing manually dispatched quality/Pages chains are observed read-only; remote main is held until they are terminal.

## Main adoption

Validated source `7f98aebde57b6afc3486d0d55baef9956eba2379` was adopted on remote main by ordinary non-force push. The remote SHA was verified exactly; zero Actions runs were observed for this source SHA. All 51 source/license resources, 161 active export identities, 10,229 immutable payloads and seven canonical release-policy files were rechecked. The separate proof commit is verified outside the repository after its push. No tag,dispatch,release,PyPI upload or online deployment was initiated by this work.
