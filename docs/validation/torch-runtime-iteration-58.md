# Torch runtime iteration 58 — bounded direct pairwise distances

Version 0.58.0 broadens `torch.cdist`, `torch.pdist` and the ordinary
`torch.nn.functional.pdist` alias while replacing retained point-pair-by-feature
difference arrays with bounded recomputation and shared first-order pullbacks.
The broader independent-library/core-efficiency/backend objective remains active.

## Behavior and algorithms

Real float32/64 Minkowski distances support nonnegative orders, including zero,
fractional orders and infinity. Cdist broadcasts batch dimensions; pdist emits
the condensed upper triangle. The mathematical API is compared with the
[PyTorch 2.14 cdist documentation](https://docs.pytorch.org/docs/2.14/generated/torch.cdist.html),
and observed native CPU scalar/IEEE/gradient conventions are separately tested.
No upstream implementation source is vendored.

Point, batch and feature tiles use an 8 MiB work-array planning budget. The bound
excludes input/output storage, requested input gradients and runtime overhead;
it is not a process RSS or browser heap limit. Small cdist geometries and tiny
pdist inputs bypass tile scheduling when the same byte model proves they fit.
The forward tape saves required input arrays, output distances and metadata;
non-leaf Tensor objects need not be retained. Pair directions are recomputed once
per block/backward traversal, shared by active inputs, and broadcast gradients
accumulate directly into original input shapes. Pdist visits upper-triangle
blocks and shares one input-gradient buffer. No result or gradient cache is added.

Outputs remain limited to 16 million elements; pair-feature work is capped at
512 million. A 128-by-129 point problem with 1,024 features now runs and
differentiates under the bounded workspace; the old full-difference allocation
limit rejected that workload. Output/work limits reject oversized requests before
allocation. Zero-coordinate, zero-distance and infinity-tie gradients, storage
versions, dtype conversions, no_grad and graph release/retention are checked.

Cdist still validates all three native compute-mode names and deliberately keeps
the library's existing stable direct-difference numerical policy. It does not
implement native matrix-multiplication dispatch/rounding or its AD behavior.
Half/complex kernels, forward/higher derivatives and GPU execution remain outside
this contract. Existing numerical qualifications from iterations 53 and 56 remain.

## Verification

| Gate | Result |
| --- | --- |
| Full repository CPU tests, including SDK | 40,815 passed, 1 skipped |
| Node Pyodide 0.29.4 / NumPy 2.2.5 | 7,185 passed |
| Chrome 155.0.8059.82 worker, same Pyodide | 7,185 passed |
| Public wheel, Python 3.10.19 / NumPy 1.26.4 | 7,198 passed |
| Public/private wheel identity, separate uninstall, fresh pickle/alias processes | Passed |
| Installed public/private `import torch` and `from torch.nn.functional import pdist` | Values, gradients and restoration passed |
| Exported PaperLeet catalog | 78/78 modules, 342 mutants; all original browser budgets passed |
| Independent contracts inside exported attention runtime | 7,185 tests / 53 suites |
| Frontend | Full suite through deb2c6c9 passed; affected writing/preview checks and both builds passed on final main |
| Source/license resources across source, packages and WASM | 60 (58 Python modules plus LICENSE/NOTICE) |
| Immutable payloads retained from previous main | 12,125 unchanged; 158 new reachable payloads |
| Active exports/indices | 161 identities checked |
| Canonical release policies | Seven byte-identical to `13d2782a494415ba72bc604a0ef7e4c4f875ba25` |

The 168 new standalone tests include scalar formulas, scalar finite differences,
broadcast pair-loop gradient oracles, empty shapes, nonfinite conventions,
infinity ties, mutation/version guards, non-leaf collection, saved-array release,
retained seeds, newly accepted workloads, memory/work guards and complete AdamW
training/checkpoint continuation. Existing native distance/model-conversion tests
remain; their former valid-order rejection assertions were updated for the new
coverage. The changed root test is separately hashed and preserved in the archive.
All 1,152 pure-native catalog input hashes were rechecked before reusing that
catalog qualification. The initial package/CPU/frontend gates finish before the full WASM budget run.
Later disjoint UI commits receive separate checks after that run. No heavy
qualification overlaps formal timing.

## Differential qualification

Ninety-six frozen logical source/seed arrays drive the same 864 distance cases
on CPU, Node WASM and Chrome WASM, against installed native Torch
2.14.1. Each runtime has the same classification:

| Classification | Cases |
| --- | ---: |
| Native per-element value/gradient comparisons | 720 |
| Separately qualified fractional-gradient cases | 8 |
| Native forward plus independent empty-pdist backward contract | 64 |
| Matching negative-stride construction errors | 72 |

The 728 ordinary/norm-qualified rows contain 1,852 numeric arrays. Dtype/shape
match exactly. Finite/NaN/positive-infinity/negative-infinity masks are checked
separately; ordinary per-element rtol/atol are float32 `3e-5/3e-6` and float64
`3e-12/3e-12`. Both default and deliberately tiny 16 KiB workspaces are exercised.
A further 60 scalar-control cases have 32 numeric matches and 28 matching error
categories, including NumPy scalars, scalar Tensor restrictions, complex-real
conversion, malformed inputs and negative/NaN orders.

Eight cases involve cancellation in fractional distance gradients of longer
vectors. They fail the stated native per-element bound and are not counted as
per-element native passes. Both implementations independently meet scalar
compensated FP64 references built from the exact frozen inputs/seeds, using
math.sqrt/math.fsum; normalized L2 and maximum-component errors are bounded by
`3e-6` for float32 or `2e-12` for float64. Very small float32 orders that round to
zero model the observed native underflowed-order kernel-gradient convention,
not the real-arithmetic norm at that positive order. Per-case errors and formulas
are retained in the qualification JSON; tolerances were not broadened globally.

Native Torch reproducibly exited with code 139 on empty `pdist` input during
fractional backward (the trace ends at float32 p=0.5, shape `(0,7)`). The same
unsafe empty geometry at other nonzero orders was not executed again. Those 64
rows compare native forward only; the adapter's empty gradient is independently
checked for exact shape/dtype and zero elements. Raw progress traces and observed exit-code evidence are retained; no native C stack trace was available.
They are not native backward passes and do not imply that every skipped row was
individually observed to crash.

## Performance and allocation

Baseline is validated 0.57.0 (`dff3444dcd784eb41b1c4e5532cef1e66c2023c7`), candidate 0.58.0.
Four fresh processes per CPU/Node/Chrome runtime, five samples per case, alternate
variant order; shared batches are calibrated outside timing to at least 20 ms.
Input/seed conversion, correctness, warmup and GC are outside timing. Kernel and
graph construction, optional weighted backward, gradient clearing and result
release are included. There are 44 matched legacy-supported rows and 18
candidate-only new-order/larger-workload rows per process. Unsupported baseline
workloads have no speed ratio. No native Torch/GPU throughput is measured.

Representative float32 L2 rows below show candidate latency / time ratio, followed
by the full ratio range across four children. Below one means faster. Cdist
small/medium/large point-feature shapes are 4×5×7, 64×80×64 and 128×192×128;
broadcast uses batch (2,1)/(1,3), 16×20 points and 32 features. Pdist small,
medium and large shapes are 5×7, 64×64 and 128×128.

| Workload / shape / mode | CPU | Node WASM | Chrome WASM |
| --- | --- | --- | --- |
| cdist / small / forward | 7.6 μs / 1.056× (1.047–1.075) | 18.2 μs / 1.075× (1.058–1.092) | 17.9 μs / 1.073× (1.068–1.101) |
| cdist / small / backward | 42.8 μs / 1.046× (1.032–1.051) | 89.8 μs / 1.010× (0.994–1.025) | 90.4 μs / 1.009× (0.993–1.032) |
| cdist / medium / forward | 229.6 μs / 1.191× (1.190–1.196) | 542.6 μs / 1.319× (1.291–1.324) | 518.7 μs / 1.318× (1.313–1.388) |
| cdist / medium / backward | 742.4 μs / 1.055× (1.052–1.058) | 1732.4 μs / 0.941× (0.925–0.945) | 1750.0 μs / 0.937× (0.933–0.946) |
| cdist / large / forward | 1303.6 μs / 1.172× (1.153–1.192) | 4262.9 μs / 1.280× (1.278–1.292) | 4331.3 μs / 1.287× (1.258–1.307) |
| cdist / large / backward | 5227.8 μs / 0.903× (0.869–0.912) | 14308.4 μs / 0.855× (0.850–0.860) | 14400.0 μs / 0.856× (0.851–0.885) |
| cdist / broadcast / forward | 48.3 μs / 1.094× (1.089–1.102) | 124.8 μs / 1.162× (1.152–1.168) | 126.6 μs / 1.163× (1.127–1.167) |
| cdist / broadcast / backward | 194.6 μs / 0.973× (0.971–0.981) | 453.0 μs / 0.916× (0.915–0.931) | 453.1 μs / 0.900× (0.894–0.910) |
| pdist / small / forward | 11.6 μs / 1.145× (1.129–1.173) | 24.2 μs / 1.154× (1.116–1.170) | 25.7 μs / 1.151× (1.145–1.156) |
| pdist / small / backward | 41.6 μs / 1.081× (1.066–1.098) | 84.2 μs / 1.068× (1.063–1.101) | 90.4 μs / 1.079× (1.067–1.087) |
| pdist / medium / forward | 199.7 μs / 1.868× (1.865–1.871) | 436.7 μs / 2.033× (1.982–2.044) | 455.1 μs / 1.967× (1.937–2.007) |
| pdist / medium / backward | 704.5 μs / 0.492× (0.488–0.494) | 1535.7 μs / 0.523× (0.512–0.536) | 1584.4 μs / 0.484× (0.463–0.503) |
| pdist / large / forward | 669.1 μs / 1.138× (1.127–1.146) | 2044.6 μs / 1.508× (1.488–1.554) | 2012.5 μs / 1.483× (1.423–1.500) |
| pdist / large / backward | 2957.9 μs / 0.269× (0.268–0.270) | 6990.5 μs / 0.309× (0.305–0.310) | 6937.5 μs / 0.287× (0.280–0.293) |

Small-call validation and bounded packing have costs, and some forward workloads
are slower. The complete table includes these regressions, L1 controls, new-order
latencies and large feature-workload timings; the release is not an all-path
speedup claim. In particular, medium pdist forward remains a follow-up optimization
target even where its backward benefits from block reduction.

The CPU allocation comparison below creates inputs before tracing and uses
float32 cdist 128×192×128 / pdist 128×128. Values are decimal MB. Forward retained
means newly tracked allocations still live after the call; pre-existing input
storage is excluded in both variants. Backward starts a separate trace. These are
not total tape memory, process RSS or WASM heap high-water marks.

| Factory | New forward-retained MB, old → new | Forward peak MB, old → new | Backward peak MB, old → new |
| --- | --- | --- | --- |
| cdist | 12.687 → 0.104 | 25.366 → 1.149 | 25.237 → 1.230 |
| pdist | 4.328 → 0.038 | 8.521 → 1.079 | 8.459 → 1.142 |

## Integration and reproduction

Main UI work through `86bbf5893077c55c6264ec7f8ce03a294b112bb2` was preserved. The complete frontend
suite and both builds passed at `deb2c6c96bc65a237c3a7c848b710002cde46439`.
A subsequent disjoint Markdown-editor commit was checked with sixteen affected
writing, section-navigation, preview and reading tests plus driver/policy checks
and both local builds; their twenty receipts identify the final main exactly. Naming remains StrideTorch pending a separate choice.
No tag, workflow dispatch, release, package upload or deployment is initiated.
Main adoption is recorded separately after local verification; the broader goal
remains active.

- Result: `packages/stridetorch/benchmarks/results/iteration-58.json`
- Benchmark: `packages/stridetorch/benchmarks/bench_distances.py`
- Raw archive: `packages/stridetorch/benchmarks/results/iteration-58-harness.tar.gz`
- Archive bytes: 12,100,006
- Archive SHA-256: `22df19138019c8109c1ec7178ee13dbf709c0ba7323bbd300d3e460cec5e483f`

The archive contains frozen code/tests/inputs, native arrays and crash traces,
CPU/Node/Chrome qualification reports and replay harnesses, package and alias
proofs, full catalog/frontend receipts, pilots and all twelve final benchmark
children. Environment directories, caches and third-party implementation source
are excluded. GPU/resident-storage work remains separate from this dense CPU/WASM
milestone.
