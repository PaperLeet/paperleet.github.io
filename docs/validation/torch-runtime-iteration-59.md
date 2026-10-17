# Torch runtime iteration 0.59 — packed medium pdist forwards

Status: validated locally; main adoption is recorded separately below when verified.

## Change and scope

`pdist` now calculates the condensed upper triangle directly when its packed forward work fits the existing 8 MiB workspace model. Medium inputs avoid redundant lower-triangle/diagonal calculations and tile packing. Backward retains the 0.58 bounded block reductions rather than scattering one gradient per packed pair. Larger inputs retain the prior path. No differences or pair indices are retained in the gradient tape; no result cache is added. Mutation guards and required input/output lifetimes are unchanged.

This is a scheduling optimization over 0.58, not a new numerical formula or GPU backend. Float32/64 nonnegative Minkowski orders and first-order reverse AD keep the documented contract. The workspace model excludes inputs, outputs, requested gradients and general runtime overhead; it is not a process RSS limit. Outputs remain capped at 16 million elements and pair-feature work at 512 million.

## Performance

Four fresh child processes per runtime (12 total), five samples per row, alternating variant order, common batches calibrated to at least 20 ms. All 62 rows are matched against the adopted 0.58 source, including non-Euclidean orders and feature-heavy cdist. Input conversion/seeds are outside timing. Graph construction, kernels, optional weighted backward, gradient clearing and result destruction are inside. “Backward” below means the complete forward plus weighted backward. Native Torch performance, cold loading and GPUs are not measured.

| float32 L2 workflow | CPU ratio | Node WASM ratio | Chrome Worker ratio |
|---|---:|---:|---:|
| pdist [64, 64] forward | 0.593× | 0.595× | 0.607× |
| pdist [64, 64] backward | 0.884× | 0.888× | 0.887× |
| pdist [128, 128] forward | 1.001× | 0.998× | 1.003× |
| pdist [128, 128] backward | 1.004× | 0.999× | 1.004× |
| cdist [128, 128] forward | 1.002× | 1.001× | 1.000× |
| cdist [128, 128] backward | 1.004× | 1.002× | 1.000× |

Ratios are candidate/0.58; lower is faster. Full per-process ratio ranges, all float64/order/workload rows and absolute microseconds are in the machine result. The medium optimization does not remove the earlier large-pdist forward regression relative to 0.57; that larger workload still uses 0.58 tiling. Small timing changes on unchanged paths are not claimed as algorithmic gains. One unchanged Node large-pdist full-step process produced a 0.585 ratio; its raw samples remain included in the reported range rather than being discarded.

## Tracked CPU allocations

These allocation rows use float32 inputs. Inputs are allocated before CPU tracemalloc starts. These are incremental tracked allocations, not total tape, RSS or WASM heap measurements. Forward-retained/peak and separate backward peak are recorded.

| workflow | variant | new retained bytes | forward peak bytes | backward peak bytes |
|---|---|---:|---:|---:|
| cdist [128, 128] | baseline | 104,176 | 1,149,816 | 1,231,212 |
| pdist [64, 64] | baseline | 13,320 | 1,109,892 | 1,172,313 |
| pdist [128, 128] | baseline | 36,848 | 1,077,840 | 1,142,124 |
| cdist [128, 128] | candidate | 103,504 | 1,149,144 | 1,229,900 |
| pdist [64, 64] | candidate | 12,791 | 1,071,431 | 1,172,449 |
| pdist [128, 128] | candidate | 36,848 | 1,077,840 | 1,142,124 |

## Numerical qualification

CPU, Node Pyodide and Chrome Worker each replay the same 112 frozen input/seed arrays across 1008 cases: float32/64, cdist/pdist, nine orders, 14 geometries and two workspace budgets (8 MiB and 16 KiB). New 64×64 and 65×64 pdist inputs straddle the float64 packed-work boundary. Positive/negative strides, empty dimensions, broadcasting, nonfinite values and infinity ties remain covered.

- 850 cases pass native per-element value and gradient tolerances, including exact shape/dtype and NaN/signed-infinity masks.
- 22 cases with fractional or float32-underflowed orders fail the strict per-element comparison after cancellation. Both native and adapter are separately qualified against independent Python scalar FP64 formulas using `math.sqrt`/`math.fsum`, normalized L2 and Linf error ≤3e-6 (float32) or ≤2e-12 (float64). These are not counted as per-element native passes.
- 64 empty-pdist cases compare native forward and an independent exact empty-gradient contract. Native nonzero-order empty-row backward previously caused two native-process crashes in 0.58 and is deliberately not reexecuted here. The prior crash receipts/logs and opt-in minimal reproduction are preserved; no new crash is claimed.
- 72 negative-stride constructor cases match native ValueError.
- 2212 numerical arrays are compared across the ordinary and norm-qualified cases. A separate 60-control matrix has 32 numerical and 28 matching-error cases.

For float32 p=1e-50, the scalar oracle explicitly models native underflow-to-zero forward order and its original-order kernel-gradient convention, not a mathematical real-order limit. For cdist infinity order NaN components are ignored in forward reduction; pdist propagates them. Existing IEEE gradient conventions remain.

## Integration and package checks

- Full CPU: 40,845 passed, one skipped.
- Independent public library: 7215 contracts each in Node/Pyodide and actual headless Chrome Worker; 53 suite hashes and all 60 source/license resource hashes match.
- Installed wheel: 7228 tests on Python 3.10.19 / NumPy 1.26.4, with no native Torch installed. Wheel/sdist payloads match the frozen source. Public/private co-installation, independent uninstallation, aliases, serialization and fresh-process loads pass.
- Public/private `import torch` aliases, functional pdist and gradients restore previous module identities on exit. Native Torch default dtype and identity remain isolated.
- PaperLeet exported catalog: all 78 modules, 342 mutants and 7215 embedded independent contracts pass with original browser budgets; no budget override.
- All 1152 native catalog inputs remain hash-identical, so their established native evidence is reused.
- Complete frontend check and full CPU suite passed on 644a8fee; numerical sources, Python tests, catalog inputs and browser drivers remain unchanged through disjoint UI fast-forwards 285dcbdc and 5ecb3ef9. On final integration base 5ecb3ef9, 20 affected practice/experience/deferred/navigation/overview checks plus WASM driver, Pages policy and both normal/observed Pages-profile builds pass (24 receipts). This does not claim a second complete frontend or CPU suite on the new UI commit. An intermediate Pages-profile build at 285dcbdc exceeded the existing raw JS budget by 148 bytes; that failure is preserved. Upstream 5ecb3ef9 defers optional practice metadata and resolves the integration budget: final Pages initial JS is 405725 bytes raw / 131559 bytes gzip, within the unchanged 409600 / 133120 limits. No local UI changes are included in the final numeric iteration.
- 12,283 previously published immutable payloads remain byte-identical; 158 new reachable assets, 161 live identities. No older asset is removed.

## Limits and release policy

All cdist compute-mode names continue to use direct differences. Native matrix-dispatch rounding, distance forward/higher AD, half/complex distance kernels and GPU execution remain outside this contract. Existing documented half NN/AD, complex AD and 0.53/0.56 numerical qualifications still apply. No broad full-PyTorch compatibility or universal speedup claim is made.

Seven release-policy files are byte-identical to canonical commit `13d2782a494415ba72bc604a0ef7e4c4f875ba25`. No tag, workflow dispatch, release, PyPI upload or deployment is initiated. Ordinary main adoption is attempted only after checking remote identity and active quality/Pages chains, including a pending workflow_run delivery gap. The broader independent-library/core-coverage/backend goal remains active.

## Reproduction evidence

Numerical baseline: `644a8feeeca50c5a70d11a000d1d9d184f854fc9` (0.58); integration baseline: `5ecb3ef9ce5ed41f6d17113b19472e3647f5028f`. All source/test/input hashes, per-runtime qualifications, timing samples, package receipts, catalog receipts and harnesses are archived.

- Machine result: `packages/stridetorch/benchmarks/results/iteration-59.json`.
- Raw archive: `packages/stridetorch/benchmarks/results/iteration-59-harness.tar.gz`.
- Archive bytes: 22434792.
- Archive SHA-256: `0e2c6b795f81b8b62b2570486d5359168217c73d21ddc3bbb34eca5689a22b59`.

Runtime versions: Apple M4 arm64; native Torch 2.14.1 / NumPy 2.5.4 / Python 3.12.11; Pyodide 0.29.4 / NumPy 2.2.5; Chrome 155.0.8059.82. All locally owned validation and timing children were observed terminal before closure.
