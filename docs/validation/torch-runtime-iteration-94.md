# Iteration 94: declared zero rules and complete Node factor observations

Two metadata counterexamples exposed an unsafe shortcut in the unqualified
logical-zero prototype. The isolated successor now requires an explicit
per-rule declaration. Three Node/WASM factor processes also completed and were
independently verified. Chrome timing remains live; the continuing goal stays
active. Production sources, main, GitHub CI, tags and publication are unchanged.

## Logical-zero propagation must respect each pullback

The previous prototype propagated a logical zero through every upstream node
without executing its numerical rule. A stdlib-only reproduction demonstrates
two incorrect consequences: a pullback returning `None` becomes an observable
zero gradient, and a pullback deliberately returning an ordinary zero array
is skipped, changing later IEEE zero-times-infinity behavior.

The successor defaults undeclared nodes to materialized cotangents. A separate
`ZeroPreservingNode` type opts into bypass only for explicitly declared rules;
its instances use the same storage size as ordinary nodes. The prototype
currently declares the norm's constant sign derivative, nonzero powers and
exponentials. Other rules retain their numerical pullbacks until independently
qualified. The joint-result helper can declare a validated joint rule too.
Versions, callbacks, target selection and graph release still precede bypass.

All twelve stdlib-only graph control checks pass, including both formerly
failing counterexamples. Targeted tests now include the distinction between an
unused upstream gradient and a materialized zero, and joint bypass uses an
explicitly declared test rule rather than assuming a particular public
operator uses a joint implementation. Syntax checks also pass. These are
metadata/source checks; the successor has not run numerical qualification.

The earlier qualification supervisor was verified idle with zero receipts,
snapshotted and terminated before execution because of these reproduced
semantic defects. Its original source/plans and terminal exit 143 are preserved.
This did not interrupt the active factor timing process and was not a timeout
restart. A new supervisor waits for that same known timing process to finish.

## Expanded numerical qualification waiting behind timing

The frozen successor qualification includes:

- The unchanged 780 isolated norm cases and 320 nonlinear composition cases
  against both native Torch versions.
- A new 288-case native/prototype matrix of logical-zero chains through powers,
  exponentials, zero norm, rounding, views, arithmetic, matrix products, Linear,
  fused products, reductions and hooks. Every discrepancy remains visible.
- Ninety-five targeted norm/graph tests and the 11,982 existing SDK contracts
  on CPU, plus the combined 12,077 tests on actual Node/Pyodide and Chrome.

The new chain controls will be executed independently on native Torch 2.6 and
2.14 only after timing terminates, then bound by source and result hashes for
the WASM runs. Existing native fixtures remain bound to their original probe
hashes. The queue checks discovery counts as well as test exit codes. This
expands verification; it does not claim all zero rules or the original twenty
norm-composition findings are repaired. Minimum-NumPy, package, application and
performance qualification still remain separate requirements.

## Three completed Node/WASM factor processes

Independent verification matches all 54 flow records to their durable logs and
successful process receipts. It recomputes every ratio and checks source plans,
equal counts, Williams positions/ordered pairs, calibration and measured minima.

| Node process | Minimum measured chunk | Maximum A/A drift | A/A cases above 5% |
| --- | ---: | ---: | ---: |
| 0 | 86.40 ms | 1.49% | 0 |
| 1 | 83.62 ms | 1.36% | 0 |
| 2 | 99.18 ms | 0.96% | 0 |

| Batched forward, median across processes | Current B / A0 | Out-of-place fast C / A0 | Raw paired B / C |
| --- | ---: | ---: | ---: |
| float32 | 0.99220 | 0.99280 | 1.00005 |
| float64 | 0.98939 | 0.99153 | 0.99956 |

Lower is better. Paired B/C is calculated from individual round ratios before
aggregation. These observations do not establish a meaningful advantage for
switching the in-place bias operation to a newly allocated output.

The factor baseline still uses the legacy `_nn.py` with the rest of 0.78 held
fixed. It does not isolate the entire 0.77-to-0.78 change. The original roughly
6.7% full-version CPU regression and six original control anomalies remain
unresolved. The general operator registry is an unisolated source difference,
not a demonstrated cause. All three Chrome processes and a full-source
comparison are still needed before a production performance decision.

## Evidence and next actions

`torch-iteration-94` in the task evidence contains the negative controls,
supersession receipt, declared-rule source overlay, twelve metadata checks,
new chain probe, frozen qualification plan and waiting supervisor. It also
contains the independently verified Node factor observations. Raw timing
results and the continuing timing supervisor remain in `torch-iteration-92`.

Finish Chrome timing, execute the expanded numerical qualification and repair
its findings. Explain the full-version performance regression, select changes
from matching evidence, and renew the full library/package/application checks.
Reconcile current main and perform fresh guarded adoption without CI or release
only after qualification. Core coverage and CPU/WASM/WebGPU work continue.
