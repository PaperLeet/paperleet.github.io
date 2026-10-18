# Torch runtime iteration 67 — joint callback lifetime and inverse temporaries

Version **0.67.0** reduces saved-payload lifetime and temporary work in existing
analytical reverse rules. It preserves the public API, CPU/WASM device policy,
saved-storage guards and supported first/mixed derivative boundaries. Baseline
main is `08642b250e21f715b608731cba90c817fee6b2c8`, whose numerical source is the adopted 0.66. Full CPU and WASM gates ran at
`0172e7b9aea1b1832e3b222c7d674cc156dab8c4`; later upstream changes affect only the
administrator/taxonomy UI. All numerical, Python/test and export hashes remain
identical through synchronization.

## Implementation

After a joint pullback is routed, the graph engine releases its local callback
reference before executing predecessor hooks. Previously that local reference
kept the consumed callback's saved payload alive despite releasing its graph
node. Retained graphs keep the callback through their requested traversals;
callback failures preserve valid retry behavior. Fifteen lifecycle tests cover
full/targeted backward, retained passes, exceptions and payload observations.
Twelve of those tests expose the earlier lifetime problem; three controls pass
on both implementations.

Joint nodes whose parent edges already contain no callbacks reuse that tuple
when releasing saved state. All-active, unfiltered joint VJPs use direct zipped
routing with the original length, required-gradient, dtype and shape checks.
Filtered/frozen inputs retain general routing. Empty version lists and parent
lists avoid unnecessary iterators; leaf work exits after hooks/gradient handling.
Four additional regressions cover repeated edges, mixed dtypes, hooks and
retained backward. The separate consumed-cotangent experiment in iteration 68
is not part of this version.

The existing large-matrix WASM inverse-JVP shared rule accumulates a private
matrix temporary in place rather than allocating a new array for its sum. The
buffer is a fresh matrix-product result, without mutating user inputs or seeds.
CPU and small/one-input dispatch remain unchanged.

## Current release-source validation

- Complete application infrastructure/native-comparison plus standalone suite:
  **41705 passed, 1 skipped, 34 warnings in 585.51s (0:09:45)**. The four-case increase from the earlier 41,701
  candidate run comes from the four newly cataloged browser-module test cases.
- Actual exported Pyodide **0.29.4 / NumPy 2.2.5**, concurrency one:
  **82 modules / 359 mutants**, all passing within their original browser budgets.
  The attention workload also runs **8,071 independent contracts / 62 suites**
  from exported runtime assets and exact current test hashes.
- Fresh versioned Node and Chrome workers independently pass the same **8,071**
  contracts; Chrome is **155.0.8059.82**. Their 61 source/license hashes match
  the released source exactly.
- Fresh standalone wheel/source archive and application wheel/source archive
  preserve all 61 resources. Python **3.10.19 / NumPy 1.26.4**, without native
  Torch, passes **8,084** installed tests. Both namespace aliases, separate
  uninstall ownership, ZIP/shared-storage checkpoints, resumed optimizer/scheduler
  updates and fresh-process lazy Transformer activation pickle are verified.
- Ten affected browser/export checks and both **Node 22.23.3** builds pass at
  the full numerical-validation baseline. After 35 disjoint upstream UI/build
  paths change, the complete current frontend suite, driver/release-policy
  checks and both Node22 build profiles pass again. The runtime, package,
  application-Python/test and export hashes remain pinned throughout the gates.
- All **13,563** prior immutable payloads are preserved. **166** new payloads
  are content-addressed and reachable; **169** active asset/index hashes cover
  the 82-module export. There are **13,729** immutable payloads in total.
- The seven release-policy files remain byte-identical to the reviewed upstream
  `4db2b71b8948eb4fe8156fc80a97461431c412d8`: tag/manual-only CI and exact
  current-main publication gates. No workflow dispatch, tag or deployment is
  initiated for this iteration.

## Performance provenance and limits

Twelve serial fresh CPU/Node/Chrome children each compare **170** matched complete
workflows: 100 determinant/log/graph, 44 inverse/JVP/matmul/complex and 26
GroupNorm/classification cases. Frozen A exactly matches the adopted 0.66 source.
Frozen B's numerical and graph files match versioned 0.67; its only difference
is the `__init__.py` version literal, changed from 0.66.0 to 0.67.0. Fresh package,
CPU and actual WASM gates use versioned 0.67. The original samples, labels and
hashes are retained; they are not relabeled as newly collected 0.67 latency.

Ratios above one are slower, and the table deliberately retains each group's
worst four-child median. These are selected library-flow results, without a
uniform speedup, native-Torch, import or complete-application throughput claim.

| Runtime | Group | Cases | Largest four-child median ratio B / A |
|---|---|---:|---:|
| cpu | detloggraph | 100 | 1.0552 |
| cpu | inverse | 44 | 1.0279 |
| cpu | joint | 26 | 1.0141 |
| node | detloggraph | 100 | 1.0318 |
| node | inverse | 44 | 1.0274 |
| node | joint | 26 | 1.0068 |
| chrome | detloggraph | 100 | 1.0171 |
| chrome | inverse | 44 | 1.0116 |
| chrome | joint | 26 | 1.0253 |

The original CPU complex128 32×32 slogdet forward outlier is approximately
1.0552 in the full series. Longer balanced A/A/B checks of selected outliers use
eight fresh CPU/Node children, rotated module load order and twelve common
batches of at least 50 ms. The CPU case is approximately 1.0020 against baseline
and 1.0004 against its same-source duplicate; the Node float64 64×64 slogdet
backward case is approximately 0.9973 and 1.0044. The original regressions and
controls remain archived. Alignment metadata is not treated as a causal finding.

CPU tracked allocations for complete soft-label loss/backward calls fall by
roughly 11–14% in the measured small/medium fixtures; the GroupNorm control is
unchanged. Ordinary CPU inverse allocation is unchanged because its dispatch
remains independent. Forcing the shared inverse route on CPU gives backward
peak ratios approximately 0.841–0.850 versus A. That is an explicitly forced
allocation proxy, **not a WASM heap/RSS or browser tape-size measurement**.

## Reproduction

The [public joint benchmark](../../packages/stridetorch/benchmarks/bench_joint.py)
uses the existing official GroupNorm/classification fixtures. Its fixture ASTs
match the frozen formal benchmark. For an idle-host timing comparison:

```sh
python packages/stridetorch/benchmarks/bench_joint.py \
  --baseline-source /path/to/0.66/stridetorch \
  --candidate-source packages/stridetorch/src/stridetorch \
  --output joint.json
```

Add `--validate-only` for untimed value/derivative checks; all 26 flows pass in
this checkout. Timed output records runtime/thread environment, every sample,
source and fixture hashes. Run timing separately from tests and other workloads.

The [machine record](../../packages/stridetorch/benchmarks/results/iteration-67.json)
and [raw harness](../../packages/stridetorch/benchmarks/results/iteration-67-harness.tar.gz)
retain original prototypes/failures, source/test snapshots, formal samples,
A/A controls, allocation diagnostics, frozen native comparisons, package artifacts,
current export guards and integration receipts. Identical payloads with matching
permissions/ownership share tar hardlinks; every logical payload hash is verified.
Filesystem timestamps are not comparison inputs; recorded sample/receipt times
remain unchanged.
The raw archive is
**20,562,245 bytes**, SHA-256 `80aa2c7475f7f01000be829085c5753602cf236085bd11be954c85cf623bf755`. Reproduction uses
the repository sources and the pinned Pyodide/runtime dependencies identified
by those manifests; local temporary environments/build outputs are excluded.

The independent-library efficiency, core API and backend objective remains active.

## Main adoption

Validated source `0be832d1a1ee468b90e4da2582d662e48c7a07b4` was adopted on remote main by an ordinary non-force
push, with zero observed Actions runs for that SHA. All 61 runtime resources,
169 active export/index identities, 13,729 immutable payloads and seven unchanged
upstream policy files were verified. A separate proof commit records this result.
No tag, workflow dispatch, release, package upload or deployment was initiated.
The overall independent-library efficiency/core/backend goal remains active.
