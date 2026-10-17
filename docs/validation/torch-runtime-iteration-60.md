# Torch runtime iteration 0.60 — triangular forward distance tiles

Status: validated locally; main adoption is recorded separately below when verified.

## Change and scope

Larger `pdist` forwards compute packed pairs only on diagonal tiles. Off-diagonal tiles use broadcast differences and write directly into their condensed output positions. Every computed pair belongs to the output; lower-triangle and diagonal self-pairs are not evaluated. Medium inputs keep the 0.59 whole-packed fast path.

Forward uses a separate two-feature-work-array model; diagonal differences explicitly overwrite one gathered copy. Backward retains the existing eight-work-array point/batch/feature plan and accumulation order. Feature chunk boundaries, mutation guards, required input/output lifetimes and first-order VJP contracts remain unchanged. No pair indices or differences are saved in the graph, and no result or gradient cache is added.

Both planners retain the 8 MiB private workspace model, excluding inputs, outputs, requested gradients and general runtime overhead. It is not a process RSS or WASM heap limit. Forward can use a larger bounded working set than 0.59; the allocation measurements below quantify that tradeoff. Output and pair-feature work limits remain 16 million and 512 million.

## Performance

Four fresh children per runtime (12 total), five samples per row, alternating variant order, common batches calibrated to at least 20 ms. All 78 rows are matched against 0.59, including L1/L2 cdist, L2 pdist across seven geometries, non-Euclidean orders and feature-heavy cdist. Inputs, conversion and seeds are outside timing. Graph construction, kernels, optional weighted backward, gradient clearing and result destruction are inside. “Backward” means the complete forward plus weighted backward. No native Torch, GPU, cold-load or universal throughput claim.

| L2 pdist input / mode | dtype | CPU ratio | Node WASM ratio | Chrome Worker ratio |
|---|---|---:|---:|---:|
| [64, 64] forward | float32 | 0.998× | 0.999× | 1.008× |
| [64, 64] backward | float32 | 1.004× | 1.029× | 1.007× |
| [128, 128] forward | float32 | 0.913× | 0.805× | 0.813× |
| [128, 128] backward | float32 | 0.969× | 0.953× | 0.951× |
| [256, 64] forward | float32 | 0.837× | 0.873× | 0.868× |
| [256, 64] backward | float32 | 0.961× | 0.964× | 0.962× |
| [64, 512] forward | float32 | 0.893× | 0.766× | 0.764× |
| [64, 512] backward | float32 | 0.986× | 0.928× | 0.940× |
| [128, 512] forward | float32 | 0.907× | 0.843× | 0.832× |
| [128, 512] backward | float32 | 0.989× | 0.970× | 0.959× |
| [17, 4097] forward | float32 | 0.808× | 0.682× | 0.686× |
| [17, 4097] backward | float32 | 0.944× | 0.936× | 0.930× |
| [64, 64] forward | float64 | 1.023× | 1.018× | 0.998× |
| [64, 64] backward | float64 | 0.994× | 0.996× | 1.000× |
| [128, 128] forward | float64 | 0.697× | 0.825× | 0.826× |
| [128, 128] backward | float64 | 0.944× | 0.927× | 0.948× |
| [256, 64] forward | float64 | 0.851× | 0.871× | 0.872× |
| [256, 64] backward | float64 | 0.960× | 0.964× | 0.974× |
| [64, 512] forward | float64 | 0.912× | 0.776× | 0.781× |
| [64, 512] backward | float64 | 0.981× | 0.938× | 0.939× |
| [128, 512] forward | float64 | 0.934× | 0.841× | 0.828× |
| [128, 512] backward | float64 | 0.996× | 0.947× | 0.945× |
| [17, 4097] forward | float64 | 0.762× | 0.663× | 0.651× |
| [17, 4097] backward | float64 | 0.960× | 0.946× | 0.918× |

Ratios are candidate/0.59; lower is faster. Absolute microseconds, all process-ratio ranges and every workload are in the machine result, including unfavorable or noisy measurements. Unchanged medium/cdist paths are controls; their small timing differences are not algorithmic gains. Previous forward regressions versus 0.57 are not claimed eliminated by this 0.59-relative comparison. Single-process prototype screens are archived separately and are not substituted for the final measurements.

## Tracked CPU allocations

Inputs are created before tracemalloc starts. These are incremental tracked forward-retained/peak allocations and a separate backward peak, excluding pre-existing inputs. They are not total tape memory, process RSS or WASM heap measurements.

| workflow | dtype | variant | new retained bytes | forward peak bytes | backward peak bytes |
|---|---|---|---:|---:|---:|
| cdist [128, 128] | float32 | baseline | 104,176 | 1,149,816 | 1,231,212 |
| pdist [64, 64] | float32 | baseline | 12,791 | 1,071,431 | 1,172,985 |
| pdist [128, 128] | float32 | baseline | 36,848 | 1,077,840 | 1,142,124 |
| pdist [128, 128] | float64 | baseline | 69,312 | 1,198,681 | 1,274,065 |
| cdist [128, 128] | float32 | candidate | 103,504 | 1,149,144 | 1,229,900 |
| pdist [64, 64] | float32 | candidate | 12,592 | 1,071,232 | 1,172,449 |
| pdist [128, 128] | float32 | candidate | 37,888 | 3,976,748 | 1,142,164 |
| pdist [128, 128] | float64 | candidate | 69,320 | 4,263,464 | 1,273,545 |

## Timing control follow-up

The original Node small float32 L1 cdist full-step median ratio is 1.117, with process range 1.002–1.347. It remains in the original result. Eight source-path identities, including the small _run prefix, are unchanged. Four additional fresh Node children ran 12 alternating paired batches of 512 calls per control; that small cdist ratio was 1.004, range 0.985–1.010. This is consistent with measurement variability, but does not replace the unfavorable original data or establish universal equal throughput. Original large-pdist full-step ranges also remain, including a CPU ratio above 1 in one child.

## Numerical qualification

CPU, Node Pyodide and Chrome Worker each replay 128 identical frozen input/seed arrays across 1152 distance cases: float32/64, cdist/pdist, nine orders, 16 geometries and two workspace budgets (8 MiB and 16 KiB). New geometries include 128×128 points/features and 17×257 feature tiles. Existing empty, batch-broadcast, stride, nonfinite, tie and 64/65-row boundary cases remain. All original 112 frozen arrays are identical to 0.59, and the classification of its 22 separately qualified cases is preserved.

- 980 cases pass strict native per-element value/gradient tolerances with exact shape/dtype and separate NaN/signed-infinity masks.
- 36 cases with fractional or float32-underflowed orders fail strict per-element comparison after cancellation. Both native and adapter are independently qualified against scalar compensated FP64 formulas using math.sqrt/math.fsum, normalized L2 and Linf error ≤3e-6 (float32) or ≤2e-12 (float64). These are not per-element native passes.
- 64 empty-pdist cases compare native forward and an independent exact empty-gradient contract. Native nonzero-order empty-row backward previously crashed two processes in 0.58 and is not reexecuted. Prior crash evidence and the opt-in minimal repro are preserved; no new crash is claimed.
- 72 negative-stride constructor cases match native ValueError.
- 2572 numerical arrays are compared across ordinary and norm-qualified cases. The separate 60-control matrix has 32 numerical and 28 matching-error cases.

The float32 p=1e-50 scalar reference explicitly models native underflow-to-zero forward order and its original-order kernel-gradient convention, not a mathematical real-order limit. Infinity-order pdist propagates NaN components while cdist ignores NaNs in its forward maximum; existing IEEE gradient conventions remain.

36 SDK cases were added: 71-feature scalar pair loops, exact condensed order and weighted L1 gradient across diagonal/feature boundaries, 4097-feature independent line geometry and weighted gradients, and large-path graph release. One fractional float32 71-feature fixture uses the documented norm contract after native also fails strict componentwise comparison. A separate CPU diagnostic verifies the candidate gradient is exactly equal to 0.59 for that fixture; normalized errors are about 1.3e-7 L2 / 2.3e-7 Linf versus compensated FP64.

## Integration and packaging

- Full CPU: 40,881 passed, one skipped.
- Node/Pyodide and actual headless Chrome Worker: 7251 independent contracts each, all 53 suite hashes and 60 source/license resource hashes match.
- Installed wheel: 7264 tests on Python 3.10.19 / NumPy 1.26.4 with no native Torch installed. Wheel/sdist bytes match the frozen source.
- Public/private co-installation, independent uninstallation, import torch aliases, functional pdist, gradients, serialization and fresh-process loads pass; prior module identities are restored. Native Torch dtype and identity remain isolated.
- PaperLeet exported catalog: 78/78 modules, 342 mutants and 7251 embedded independent contracts pass with original browser budgets and no overrides.
- A separate 1152 native catalog input files remain hash-identical; their established native evidence is reused.
- Complete frontend check, normal build, observed Pages-profile build, WASM driver and Pages-policy checks pass locally.
- Full CPU/frontend baseline: 8eac281c6c9efdafe3cf72f3704eb4a938619f6b; final UI integration base: 0383023463b26d6696843415e478e7a62e6d4e83. Numerical sources, Python tests, catalog inputs and browser drivers are unchanged through the disjoint UI sync. The final UI has 19 affected checks plus driver/policy/two builds (23 receipts). This does not claim a second complete CPU/frontend run on the new UI commit.
- 12,441 previously published immutable payloads remain byte-identical; 158 new reachable assets and 161 live identities. No previous asset is removed.

## Limits and release policy

Cdist compute-mode names retain stable direct differences, without native matrix-dispatch rounding. Distance forward/higher AD, half/complex distance kernels, output buffers and GPU execution remain outside this contract. Existing half NN/AD, complex AD and 0.53/0.56 numerical qualifications remain. This is a CPU/WASM scheduling optimization, not full native PyTorch compatibility.

Seven release-policy files remain byte-identical to canonical commit 13d2782a494415ba72bc604a0ef7e4c4f875ba25. No tag, workflow dispatch, release, PyPI upload or deployment is initiated. Ordinary main pushes require exact remote identity and no active quality/Pages chain or pending workflow_run delivery gap. The broader independent-library/core-coverage/backend goal remains active.

## Reproduction evidence

Numerical baseline: 8eac281c6c9efdafe3cf72f3704eb4a938619f6b (0.59); integration baseline: 0383023463b26d6696843415e478e7a62e6d4e83. Source/test/input hashes, runtime qualifications, all timing samples, package/catalog/build receipts and executable harnesses are archived.

- Machine result: packages/stridetorch/benchmarks/results/iteration-60.json.
- Raw archive: packages/stridetorch/benchmarks/results/iteration-60-harness.tar.gz.
- Archive bytes: 49175238.
- Archive SHA-256: 303fc0e49e5fb80945b5af7b09fdea5238f91ea3e833cee9db2f77b13296963c.

Runtime versions: Apple M4 arm64; native Torch 2.14.1 / NumPy 2.5.4 / Python 3.12.11; Pyodide 0.29.4 / NumPy 2.2.5; Chrome 155.0.8059.82. Locally owned validation/timing children were observed terminal before closure.
