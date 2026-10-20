# Torch runtime iteration 46: ordered inverse pooling adjoints

## Result and bounds

Version 0.46 optimizes fixed/adaptive 3D average-pooling backward. Seeds are scaled once, then gathered by input position and accumulated in native output Z/Y/X order. Public module/function signatures, mutation guards, float32/float64 VJP/JVP/reverse-over-forward, aliases and checkpoint behavior remain available.

The direct baseline is adopted 0.45 source 7f98aebde57b6afc3486d0d55baef9956eba2379, with proof/main 8bfd8c5f74f493011b8f07beb44526054a555e5e. Forty-nine of 51 source/license resources are unchanged; only init/version and the 3D average implementation change.

A separate 32-entry inverse-geometry cache holds readonly arrays and bounded immutable coordinate-byte keys. Each optional gather-index/mask plan is at most 128 KiB. No cache stores tensors or seed values. Admission checks input-axis construction, the full scaled seed copy and packing estimates before allocation. Nominal inverse work over twice scatter work retains the previous adjoint; forced-inverse tests cover those shapes independently. Selected inverse calls avoid allocating a discarded zero-gradient buffer; exact disjoint blocks fill an empty result directly.

The 32 MiB workspace estimate excludes required input/output storage and resident cache metadata. It is not a measured total-process-memory limit. Oversized plans retain bounded loop/scatter paths; large-axis rejection is tested before coordinate allocation.

## Verification

| Check | Observed result |
|---|---|
| Full CPU repository/package tests | 38,530 passed, 1 skipped; 412.68 s |
| Targeted 3D pooling | 925 tests; five forward/backward route pairs |
| Native PyTorch 2.14.1 | 155 cases, five adapter routes, 3,715 arrays |
| Node/Pyodide and actual Chrome Worker | 4,902 independent contracts each; 43 suites |
| Installed public wheel | Python 3.10.19 / NumPy 1.26.4; 4,915 tests; native Torch absent |
| Distribution ownership | All 51 resources identical; coinstallation, aliases, checkpoint resume and independent uninstall survival |
| Native original catalog | 78 problems, 10,218 executions, 342 mutants, three repeats; 226.830 s < 600 s |
| Exported WASM application | 78/78, 342 mutants, 4,902 independent contracts; 349.949 s |
| Frontend | Complete npm check, driver/policy checks, normal and current Pages-profile builds |
| Identities | 161 active export hashes; 10,229 prior immutable payloads preserved, 158 reachable payloads added |

Native comparisons include CDHW/NCDHW, contiguous/Fortran/strided/channels-last inputs and seeds, local NaN/Inf, empty global input, dtype/shape boundaries, VJP, JVP and reverse-tangent differentiation. New cancellation-sensitive gradient seeds are checked against native Torch and the prior scatter order. Exact output strides and non-singleton leaf-gradient strides agree. Existing singleton-gradient-stride and ill-conditioned global-mean differences remain scoped as in 0.45; no cross-platform bitwise promise is added. The native catalog uses CI-style OMP/MKL/OpenBLAS/Accelerate thread caps of two. Its elapsed time is a budget check, not a backend speed comparison.

## Complete-flow timings

Direct 0.45 is compared with 0.46 and an independent same-source inverse-disabled control. Four fresh processes per CPU/Node/Chrome runtime alternate order. Five samples per flow contain 12 ordinary calls or three training steps. All 46 flows check values/gradients or a complete SGD update and record selected/fallback inverse calls before timing.

Percent elapsed-time reduction versus direct 0.45; negative means slower. CNN flows contain Conv3d, SiLU, fixed downsampling, adaptive pooling, Linear, loss/backward and SGD.

| Flow | Dtype | CPU | Node/WASM | Chrome/WASM |
|---|---|---:|---:|---:|
| overlap / backward | float32 | 35.4% | 23.8% | 30.3% |
| overlap / backward | float64 | 41.3% | 29.7% | 32.8% |
| upsample / backward | float32 | 33.7% | 25.7% | 28.0% |
| upsample / backward | float64 | 34.3% | 26.5% | 29.3% |
| irregular / backward | float32 | -0.4% | -4.3% | -5.1% |
| irregular / backward | float64 | -0.6% | -0.9% | -1.6% |
| fixed / backward | float32 | 0.1% | -2.0% | -0.0% |
| fixed / backward | float64 | 0.3% | -0.4% | 4.9% |
| cnn_upsample / training | float32 | 10.1% | 6.4% | 6.5% |
| cnn_upsample / training | float64 | 8.7% | 3.4% | 7.0% |
| cnn_irregular / training | float32 | -2.7% | 0.2% | 0.0% |
| cnn_irregular / training | float64 | 7.3% | 1.7% | 1.0% |
| cnn / training | float32 | 0.5% | -2.5% | 0.0% |
| cnn / training | float64 | 6.3% | 1.7% | 6.7% |

Every sample, child median and control is retained. Unselected paths/general CNNs have no uniform speedup; fallback overhead and shared-host/GC variation remain visible. Results support the measured overlapping/upsampling adjoints, not native Torch/GPU or cross-runtime performance claims.

## Footprint and reproduction

| Measure | 0.45 | 0.46 |
|---|---:|---:|
| Source/license bytes | 607,595 | 610,370 |
| Local concatenated gzip estimate | 141,441 | 142,106 |

These are source-concatenation estimates, not wheel size, network transfer or peak memory. Repository tool: [bench_average_pooling_inverse.py](../../packages/torchwasm/benchmarks/bench_average_pooling_inverse.py), importing the neighboring bench_average_pooling_3d.py helper. Pass baseline/candidate source directories, output path and optional samples/iterations/reverse order. [Machine-readable reports and four harness archives](../../packages/torchwasm/benchmarks/results/) retain exact source/test/export/dependency/distribution identities, native arrays and all timing samples.

The numerically correct but slower per-plane prototype and initial packed versions remain in the local prototype-stage archive. All formal gates/timings are repeated on the final 0.46 identities. Seven release-policy files remain byte-identical to canonical commit 13d2782a494415ba72bc604a0ef7e4c4f875ba25. This work initiates no tag, workflow dispatch, release, package upload or online deployment. Broader independent-library/core-coverage/backend work remains active.

## Main adoption

Validated source `8e97eeeabde1f7621e7b83aa7bc5c300b6fd9d91` was adopted on remote main by ordinary non-force push. The remote SHA was verified exactly; zero Actions runs were observed for this source SHA. All 51 source/license resources, 161 active export identities, 10,387 immutable payloads and seven canonical release-policy files were rechecked. The separate proof commit is verified outside the repository after its push. No tag,dispatch,release,PyPI upload or online deployment was initiated by this work.
