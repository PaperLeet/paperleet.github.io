# Iteration 76: large 2D matmul on NumPy/WASM

Version 0.74 keeps the synchronous NumPy/Pyodide runtime and optimizes large float32/float64 matrix products and their two pullbacks. Native CPU keeps NumPy matmul. No supported WebGPU device or new binary/runtime dependency is introduced.

## Dispatch and storage

Both operands must be 2D, have equal real float32/float64 dtypes and matching inner dimensions. Output dimensions must be at least 32, the reduction dimension at least 64, work at least 1,048,576 multiply-add terms, and the captured left input at least 4,096 elements. All other inputs retain the original operation. Nonzero magnitudes must be between `sqrt(finfo.tiny)` and `sqrt(finfo.max / K) / 2`; NaN, infinity and exceptional finite ranges fall back. This range guard does not eliminate cancellation or backend roundoff.

The implementation uses direct `einsum` with its default traversal, then normalizes output to C layout. Explicit `order="C"` was an unselected attempt: it disabled the fast WASM traversal despite the output already being C-contiguous. Both failed/unselected attempts and their diagnostics remain in the raw evidence. Opposite-operand live reads, minimal saved-value guards and zero signs remain qualified. Inputs and packed operands are not cached.

## Numeric and package controls

* Forty native Torch 2.6 fixtures cover four large shapes, five supported layouts and two dtypes. Each checks forward, both VJPs, JVP, both HVP outputs and scalar loss; 120 independent scalar forward entries also use `math.fsum`. The new contraction executes 13 times per fixture in the combined derivative workflow.
* Final-version Node/Pyodide reruns all 40 fixtures and 73 new regressions. Chrome/Pyodide runs the complete 75-suite SDK: 8,911 passed. The complete native application and standalone suites report 42550 passed, 1 skipped, 34 warnings in 260.19s (0:04:20).
* A fresh Python 3.10.19 / NumPy 1.26.4 environment installs the standalone wheel with no Torch or PaperLeet installed: 8924 passed, 41 warnings in 21.64s. Both built wheels contain the exact 62 selected source/notice resources in their respective public/private namespaces; explicit `import torch` aliases work after co-installation.

A repeated-row float32 HVP fixture exposes composed-reduction rounding: maximum absolute errors against float64 controls were about 0.00190 for native Torch/unchanged adapter and 0.00228 for the candidate. The HVP regressions use a separate `atol=0.002, rtol=0.0003` budget; other new comparisons retain their own stricter budgets. This is not a bit-identical floating-point claim.

## Warm-call performance

Four fresh processes each in Node/Pyodide, Chrome/Pyodide and native CPU compare baseline A0, an independent baseline import A1, and candidate B. Thirty-four flows per process cover both dtypes, forward/full forward-VJP calls, layouts and fallback controls. Batches target 50 ms, three rotated samples per variant. All samples are retained: 3 of 3,672 batches were below 20 ms. The original candidate carried version 0.73.0; selected 0.74.0 changes only that version literal, with all compute sources byte-identical.

Ratios below are medians of four per-process B/A0 ratios in Chrome; lower is faster. Full data, including A/A drift, ranges and all native controls, is in the machine record.

| Flow | float32 B/A0 | float64 B/A0 |
| --- | ---: | ---: |
| 256-square forward | 0.3931 | 0.4015 |
| 256-square forward + VJP | 0.3244 | 0.3409 |
| 512-square forward | 0.3457 | 0.2538 |
| 256-square Fortran layout | 0.3939 | 0.3941 |
| 256-square strided | 0.4901 | 0.4862 |
| 128×64×128 forward | 0.7223 | 0.7169 |
| 4×5×6 forward | 1.0032 | 1.0216 |

Small WASM calls show approximately 1–2% overhead in these measurements. A focused four-process CPU follow-up covers six flows with nine longer rotated batches: median B/A0 ratios range from 0.9955 to 1.0006. These controls did not reproduce the first-pass 4.5% CPU outlier. No universal speedup, native-Torch latency comparison, GPU routing threshold, cold-load, transfer or full-application throughput claim is made.

## Dispatch-boundary follow-up

Six additional shapes include `(32,128,256)`, `(256,128,32)`, `(32,1024,32)`, `(64,64,256)`, the inactive left-size boundary `(32,64,512)`, and the short-reduction forward fallback `(128,32,256)`. Thirty-six float32/float64/layout fixtures use closed-form rank-one polynomial forward/VJP/JVP/HVP controls with `math.fsum` scalar factors. Native Torch 2.6 and both selected-version WASM runtimes independently pass those same 36 fixtures. These are additional controls outside the maintained SDK inventory.

Eight more fresh WASM processes measure 24 forward/forward-VJP flows per process, with 0 of 1,728 batches below 20 ms. Active-forward boundary median B/A0 ratios range from approximately 0.497 to 0.887 across the two runtimes; inactive forward controls retain approximately 1–2% overhead. The original dispatch is retained. The short-reduction case can still accelerate a separately eligible pullback. Every boundary sample, range and A/A control is retained.

## Tracked allocation

In Node/Pyodide, 512-square float32 forward incremental traced peak increased from 1,049,932 to 1,345,400 bytes; forward/VJP increased from 3,149,212 to 3,445,240 bytes. The extra peak comes from bounded range scans and optional output-layout normalization. After 13 discarded calls, both baseline and candidate retained the same 188 traced harness bytes; weak references confirmed the source buffers were collectable. This measures tracked Python/NumPy allocation deltas, not total RSS or physical WASM heap.

## Application and release boundaries

All local application gates passed on `e57526de68607bba905cda0703a8b9d727d8a3b3` with the 0.74 selected source. Real exported WASM passed 86/86 catalog modules and 386 mutants, plus the complete frontend checks, policy checks and both build profiles. The 14,779 prior immutable assets remain byte-identical; 174 new content-addressed assets are reachable from the live export. Release policies remain unchanged. No CI, tag, package publication or deployment was initiated.

Baseline numeric/performance sources were frozen at `2f3faf32afffb086b44c20e490ec01dde8221e44`. The unrelated seven-file UI advance to `e57526de68607bba905cda0703a8b9d727d8a3b3` was reviewed and preserved before application validation; all core sources were unchanged by that advance.

[Machine record](../../packages/stridetorch/benchmarks/results/iteration-76.json).

[Raw evidence](../../packages/stridetorch/benchmarks/results/iteration-76-raw.tar.gz) — 30,646,649 bytes; SHA-256 `ec28d6bc1359935d2d8e2eaae5f25cd2dd7d357b372b44edbf433e6d5f4de297`.

## Main adoption

Version 0.74.0 was adopted on main at `76dd285a376cca3586190f457cd554467e6bde8e` through the unchanged guarded ordinary-push wrapper. The remote ref, clean worktree and 2,341 qualified Git blobs were verified, including exact selected source and live export identities. All 14,779 prior immutable asset tree entries retain their Git object identities and modes. The source commit had zero Actions runs at verification. No CI, tag, package publication or deployment was initiated. [Adoption receipt](../../packages/stridetorch/benchmarks/results/iteration-76-main-adoption.json).
