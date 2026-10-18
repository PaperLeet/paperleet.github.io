# Iteration 79: vector APIs and lower matmul inference overhead

Version 0.77 adds `torch.mv`, `Tensor.mv`, `Tensor.dot` and `Tensor.matmul`, with native Tensor-only binding and keyword names for root `dot`/`matmul`. Existing `import torch` aliases and installed launchers expose the same APIs. Normal CPU/WASM numerical kernels, including the qualified batched matmul and SDPA engines, retain their 0.76 bytes. This iteration reduces Python graph setup for ordinary inference and handles empty products from metadata.

## Contracts and gradients

`dot` requires two vectors, `mv` a matrix and vector, and `matmul` supports vector promotion and batch broadcasting. Root `out=None` is accepted; actual output buffers remain explicitly unsupported and methods reject `out`. Foreign operands follow Python reflected-operator handling. NumPy reflected matmul inference requires detached inputs when gradients are enabled. Real float32/float64 paths cover reverse gradients, JVP, functional reverse-based JVP and HVP. Existing complex-autograd and nonleaf mutation boundaries remain explicit.

Native Torch 2.6 empty products dispatch differently depending on rank, stride, gradient flags and broadcast/folding. Matching those routes requires shape/dtype rules, including mixed-dtype empty results and the correct saved-input storage guards. The implementation does not expand or cast source arrays for those zero results. Empty output element strides are contiguous. Floating gradients cannot silently disappear into an integer/bool result. An LRU cache stores only plain integer geometry, with a regression proving that user dtype metadata is released.

The 680 new public regressions include independent bilinear Jacobian-column, VJP and HVP expectations, literal native controls, strict signatures, dtype/error behavior, alias mutations, lifetime checks and installed aliases. Final CPU library suite: 11876 passed, 1 warning in 22.43s. Final Node and real Chrome Workers each pass 11,863 library tests plus 11,341 native controls examined: 11,079 matched cases and 262 explicit existing unsupported boundaries (180 nonleaf mutations, 79 complex-AD cases, three output buffers). Unsupported cases are reported separately rather than counted as matches. Pyodide is pinned to 0.29.4 with NumPy 2.2.5.

The independent wheel on Python 3.10.19 / NumPy 1.26.4 passes 11,873 tests with three existing rank-limit skips. Public/private wheels contain the exact 62 selected source/notice resources, both sdists are actually rebuilt, and co-installed namespaces plus unchanged `import torch` CLI execution pass without native Torch. No artifacts are published.

## Warm latency

Each runtime uses three fresh processes, 98 flows, three rotated batches per flow and identical-iteration A0/A1/B comparisons. A0/A1 independently import the same 0.76 baseline. Values and full gradients are checked before calibration; batches target 100 ms and every recorded batch exceeds 20 ms. The 78 existing flows are retained and 20 tiny/empty flows added. The `mv16` flow measures the existing `@` route so the baseline remains runnable; these are not direct timings of a previously absent API.

Ratios below compare 0.77 elapsed time with 0.76. Group values are geometric means of per-flow median process ratios, not whole-application speedups.

| Runtime | Tiny inference | Empty broadcast backward | PaperLeet Attention/GQA | A/A control |
|---|---:|---:|---:|---:|
| cpu | 0.683 | 0.438 | 0.998 | 1.001 |
| node | 0.661 | 0.489 | 0.999 | 1.000 |
| chrome | 0.669 | 0.474 | 0.998 | 1.001 |

All flow/process ratios and raw timings are retained. Stable regressions above five percent across every process are listed here:

- node: `float32_empty_mv_backward`, median ratio 1.083; process ratios 1.083, 1.055, 1.116.
- node: `float64_empty_mv_backward`, median ratio 1.061; process ratios 1.061, 1.061, 1.121.
- chrome: `float64_empty_mv_backward`, median ratio 1.078; process ratios 1.096, 1.078, 1.057.

Small backward calls also carry strict dispatch/stride bookkeeping costs; speed improvements are scoped to the measured paths. These results exclude cold import, native-Torch/GPU comparisons, transfers and whole-application throughput. The benchmark and original PaperLeet fixtures are retained for reproduction.

## Allocation evidence

For A=(32,1,0,128) float32 with gradients and B=(1,8,128,256) float64 without gradients, native Torch 2.6 retains a separate expanded 64 MiB saved matrix. The candidate probe traces 5,457 bytes of peak new CPython allocation and releases original B. Native logical retained storage and CPython traced allocations are distinct metrics; no RSS or physical VRAM comparison is claimed.

## Application integration

Full native application/library suite: 45505 passed, 1 skipped, 34 warnings in 270.44s (0:04:30).

The native catalog reference/mutation/determinism run retains three repetitions and every per-module CPU budget under the aggregate 600-second deadline. All 89 maintained browser modules and 406 mutants pass their actual WASM budgets. Complete frontend checks, release-policy checks, default and current Pages path-routing profile builds, asset-retention checks and exact worker regeneration pass. A fresh real Chrome session loads the nested Attention page, edits its canonical solution through CodeMirror and passes 7/7 actual Worker/Pyodide tests plus explicit mm/bmm/mv/dot/matmul assertions. Only health and anonymous-account responses are trusted fixtures; all other external requests, including writes, are blocked. No live API/backend integration is claimed.

The full WASM catalog uses concurrency 1. A separate native test process was observed on this host during that application run and left untouched. Those results establish actual per-module budget compliance; they are not a whole-catalog speed comparison with an earlier run or a causal host-load measurement. The separately completed three-process latency records retain their own controls.

The export preserves all 15,319 previous immutable resources and adds 180 reachable assets with 183 live export/index identities. Core numerical execution retains its original base `e64e88956d796f2d8bd06d3854257575fa4e9856` and frozen candidate bytes. Application qualification executes on `b4b06237893fbdfe70efd3afbf0ee38b8b7cb4fe`, rebased onto `f2540530bfc7ccf71d3ac9b0a77eafd6052560f5` with the 24 reviewed external UI paths preserved. Numerical runs are not relabeled after rebasing.

This is local qualification; main adoption is recorded separately when verified. Experimental WebGPU code is unchanged and does not provide a Torch GPU device. No CI, tag, publication or deployment was initiated.

## Concurrent notification UI integration

Main advanced by three notification/inbox/admin commits to `1535c6bc492726119ef10afc850bf7b52bcc0cfd` while the original application gates ran. The 16 reviewed external paths change UI/docs and add one frontend check; dependencies, other scripts, compute/application inputs and seven release-policy files are unchanged. The locally qualified source/export commits were preserved at a checkpoint and rebased without conflicts. All 62 source resources, 79 library tests, 1,426 application inputs, 183 live exports and 15,319 prior immutable resources retain their qualified identities.

Complete frontend checks including the new review-notification check, both default/current Pages builds, asset-retention and release checks, exact worker regeneration and a fresh nested-page real Chrome judge run all pass on `fc128d125e6ac2f09eb5d47840685a0cb549d4a9`. The latter again passes 7/7 Attention tests and the explicit new API assertions. Numerical CPU/WASM and latency records retain their original execution bases and frozen bytes; no rerun on the rebased parent is claimed. No CI, tag, publication or deployment was initiated.

[Machine record](../../packages/stridetorch/benchmarks/results/iteration-79.json). [Raw evidence](../../packages/stridetorch/benchmarks/results/iteration-79-raw.tar.gz): 6,128,847 bytes, SHA-256 `2f95dfc24e6795212fb31d91154b8f9ca686668329f003bd4f6e07840d48c3b3`.

## Main adoption

Version 0.77.0 was adopted on main at `3941149119e06d2db62d4f00ede7d603d9732ed3` through the unchanged guarded ordinary-push wrapper. Git/GitHub refs, the clean source tree and 2,442 qualified Git blobs were verified. All 15,319 prior immutable blob identities and modes are preserved. The source commit had zero Actions runs at verification. No CI, tag, publication or deployment was initiated. [Adoption receipt](../../packages/stridetorch/benchmarks/results/iteration-79-main-adoption.json).
