# Iteration 77: batched matmul and fused SDPA on NumPy/WASM

Version 0.75 extends the measured synchronous WASM matrix path to batch/broadcast matmul and fused SDPA products. Native CPU retains NumPy matmul. The runtime has no new binary dependency or supported GPU device.

## Qualified dispatch and data lifetime

The batched path requires real matching float32/float64 operands with valid broadcast shapes, output core dimensions at least 32, reduction dimension at least 64, at least 262,144 multiply-add terms per core, and at least 1,048,576 across all output batches. The existing captured-left-size gate of 4,096 elements remains. Exceptional ranges retain the original operation, using the 0.74 nonzero-magnitude limits. Higher-rank inputs above 34 retain matmul: the shape-only broadcast helper failed beyond 32 batch axes on tested NumPy builds even when matmul worked. Regression controls cover ranks through 64 where the installed NumPy can construct them.

A direct ellipsis contraction was rejected: Fortran layout could be 3–5 times slower and small-core batches also regressed. The selected route presents contiguous reduction rows to einsum, while retaining zero-stride batch sharing. Range scans inspect unique values; input packing does not expand broadcast repeats. Output storage remains C-contiguous. Grouped SDPA keeps unrepeated K/V heads. The small-core fallback remains selected by the measured per-core work threshold.

Cold-cache probes found an additional existing 2D issue: passing a payload-bearing dtype directly to `np.finfo` could retain its metadata through NumPy’s global cache after tensors were freed. Both 2D and batch limits queries now use canonical scalar types. Four lifetime regressions cover cold float32/float64 limits caches.

## Native controls and contracts

Thirty full-array fixtures use native Torch 2.6 with the explicit MATH SDPA backend. They cover batch matmul, single-sided/cross broadcasting, actual Q/K head interleaving, Fortran/strided/zero-stride layouts, empty batches, masked grouped SDPA and dropout-one. Each compares forward, VJPs, JVP, HVP, scalar loss, shape/dtype and exact regenerated input hashes. Selected-version Node and Chrome both pass these controls and 9,002 SDK contracts. Actual Tensor output contiguity is checked before serialization. The original native-generation program remains preserved separately from later validation-only assertions.

The minimum wheel environment uses Python 3.10.19 and NumPy 1.26.4 without native Torch or PaperLeet installed: 9012 passed, 3 skipped, 41 warnings in 39.60s. Three high-rank cases are skipped because NumPy 1.26 cannot construct those arrays; they execute on the newer CPU/WASM builds. Public and private installed namespaces contain the exact 62 selected source/notice resources, and explicit unchanged `import torch` aliases work after co-installation.

The wheel/sdist qualification preceded a final README-only addition. Executable resources and version match the selected source exactly; final README bytes are not claimed to match that earlier build metadata. No artifact was published.

## Public-call performance

Four fresh processes per native CPU, Node/Pyodide and Chrome/Pyodide compare baseline A0, independent baseline import A1, and candidate B. Every process checks values then measures 78 float32/float64 flows: public Tensor forward/full VJP calls, fused SDPA forward/VJP/JVP, and the actual unchanged PaperLeet attention/GQA functions. Fixtures keep their original imports and bodies; each namespace is bound with `as_torch()` during setup. Packing and range scans are timed. Batches target 50 ms with three rotated samples; all 8,424 batches exceed 20 ms.

Measured candidate compute sources are byte-identical to selected 0.75; only its version literal changes from the prototype’s 0.74. Absolute timings exclude import, worker/bridge transfer, native Torch, GPU and whole-app throughput. The successful first native process is retained as cycle 0, not rerun.

Chrome medians of four per-process B/A0 ratios follow; lower is faster. Every sample, range and A/A control is retained in the machine record.

| Workflow | float32 B/A0 | float64 B/A0 |
| --- | ---: | ---: |
| SDPA 128 forward | 0.7041 | 0.6435 |
| SDPA 128 forward + VJP | 0.6584 | 0.5728 |
| SDPA 128 JVP | 0.7015 | 0.6170 |
| Masked GQA 128 forward + VJP | 0.6532 | 0.6192 |
| PaperLeet attention forward + VJP | 0.6651 | 0.5705 |
| PaperLeet GQA forward + VJP | 0.6673 | 0.6348 |
| Fortran batch forward | 0.3926 | 0.2919 |
| Tiny SDPA forward | 1.0129 | 1.0130 |
| Existing 256-square 2D control | 1.0000 | 0.9991 |

Native CPU median ratios stay within 2.5% of baseline across the measured flows. Small SDPA has approximately 1–2% WASM overhead; existing 2D controls remain near baseline. These results do not define a universal routing threshold or native-Torch/GPU comparison.

## Tracked allocation

Packing adds temporary memory. In Node/Pyodide, 128-length float32 SDPA forward/VJP incremental traced peak rises from 3,150,064 to 4,036,323 bytes; the PaperLeet attention flow rises from 3,287,409 to 3,781,288 bytes. The Fortran batch forward/VJP peak rises from 789,972 to 1,413,687 bytes. These are operation allocation deltas, not total RSS or physical WASM heap.

All 24 native and 24 real-WASM allocation/lifetime checks pass. After 13 discarded calls, residual tracked deltas remain small and actual source ndarray owners are collectable. Zero-stride regression spies check both unique-value scans and unexpanded packed batch views. These checks are separate from latency runs.

## Application validation and release boundary

Complete native application/library suite: 42644 passed, 1 skipped, 34 warnings in 340.93s (0:05:40).

Real exported WASM passed 89/89 catalog modules and 406 mutants, plus the complete frontend checks, policy checks and both build profiles. All 14,959 prior immutable assets remain byte-identical; 180 new content-addressed assets are reachable from the live export. Release policies remain unchanged. No CI, tag, package publication or deployment was initiated.

Performance baseline was frozen at `f2d91d09e1559d287f709094774b3d6e95e234d5`. The unrelated 95-path main advance to `7e7b64a5a5cfd6f296b76a7f6e740fd2f0900889` adds BERT, GAN and Dropout exercises and UI changes; core sources and benchmarked PaperLeet function bodies were unchanged. Application validation uses the updated 89-module catalog, preserving all external changes.

[Machine record](../../packages/torchwasm/benchmarks/results/iteration-77.json).

[Raw evidence](../../packages/torchwasm/benchmarks/results/iteration-77-raw.tar.gz) — 39,611,534 bytes; SHA-256 `da5b7f549bd1a1a85485780ef1d8d966561f24a721ba40eea9df05f31486ab30`.
