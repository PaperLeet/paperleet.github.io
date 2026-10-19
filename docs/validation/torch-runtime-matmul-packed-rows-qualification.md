# Packed-row WASM matrix candidate: repeated timing and memory qualification

The isolated packed-row candidate is selected for the next integration stage.
It passes the scoped numerical, repeat-process performance and memory-lifetime
checks below. This record does **not** claim package/application qualification
or adoption on main. The supported production version remains 0.80 at this stage.

Only `_matmul.py` differs from the 65-resource 0.80 baseline. The exact candidate
is unchanged from the [initial investigation](torch-runtime-matmul-packed-rows-candidate.md).
Eligible 2D WASM products pack contiguous reduction rows; existing shape, dtype
and exceptional-range guards, normal native CPU routing and batched kernels
remain unchanged. Existing aliases and public APIs are unchanged.

## Six-process comparison

The campaign covers 27 full public calls: the previous 18 dense/layout flows,
six rectangular/threshold cases, left-only and right-only gradients, and a
forward-only case. All return independent CPU snapshots. Read-only inputs are
physically shared among independent A0/A1 baseline imports and candidate B.
Native Torch 2.6 generates all 77 expected outputs; both runtime preflights pass
before measurement. Every batch's last output is checked outside the clock.

Three fresh Node processes and three fresh headless Chrome 155.0.8059.82
processes use Pyodide 0.29.4 / NumPy 2.2.5 on the same machine. Every flow uses
all six implementation orders followed by their reverse, for 12 measured rounds.
Calibration reaches at least 100 ms. Measured batches target 45 ms; observed
minima are **34.519 ms in Node** and **38.5 ms in Chrome**, so the target is not
uniformly reached. GC is disabled during calibration/measurement and restored;
input fingerprints and NumPy error state remain unchanged.

All **162** flow/process A/A controls stay within the preregistered **±5%**.
No process, median or flow shows candidate/baseline greater than **1.10**.
All raw samples are retained, with no favorable reruns or sample exclusions.

The table reports ranges of the three-process median paired B/A0 ratios.
These are measurements for the stated flows, not device-independent promises.

| Flow group | Node B/A0 | Chrome B/A0 |
| --- | ---: | ---: |
| 128/256 strided, both real dtypes | 0.550–0.617 | 0.553–0.622 |
| 128/256 broadcast rows, both real dtypes | 0.817–0.928 | 0.836–0.938 |
| Dense 128/256 C/F plus 512 float32 C | 0.901–0.941 | 0.907–0.961 |
| Additional rectangles/thresholds/gradient modes | 0.704–1.001 | 0.712–1.000 |

This supports approximately **38–45% less time for the measured strided flows**,
**6–18% for broadcast rows**, and **4–10% for dense C/F flows**. The unchanged
below-threshold routes remain near parity. Small improvements in individual
gradient modes should not be read as precise effects beyond the controls.

## Numerical and lifetime checks

The exact candidate inherits the unchanged four-suite native gates: **3025**
tests on NumPy 2.5.4, **3022 passed / 3 skipped** on NumPy 1.26.4. The full
**12212-test** Node SDK gate remains valid by resource identity. A fresh complete
Chrome SDK gate now also passes **12212** tests against that exact candidate.
No test or numerical tolerance is relaxed.

Both native NumPy environments separately profile full public calls with the
exact WASM kernel routing forced. After 30 repeated calls per implementation and
flow, all observed contiguous-copy and output owners are released. Fresh actual
Node/WASM instrumentation checks **2430** full calls with the same result: zero
remaining observed copy or output owners. Its largest simultaneously live
contiguity-copy payload is **2 MiB**. These counters include output normalization
as well as packed inputs; they are not total process-memory measurements.

Full-call traced peak changes are small (at most **3482 bytes** on NumPy 2.5.4
and **2706 bytes** on 1.26.4), because returning owned snapshots already dominates
these call peaks. This must not hide the individual-kernel cost. Separate
77-product profiling finds a largest additional traced kernel peak of
**1,835,306 bytes** (about **1.75 MiB**) on NumPy 2.5.4, and **1,802,562 bytes**
on 1.26.4, for the right VJP of a 512-cube float32 product. This temporary memory
is a real tradeoff for faster contiguous reductions.

CPython `tracemalloc` is not browser RSS. The actual WASM run's heap capacity
increases from **43,581,440** to **62,783,488** bytes while exercising changing
shapes and implementations. That retained capacity is a high-water allocation;
it cannot be attributed entirely to this candidate or interpreted as live-array
bytes. Only the recorded weak-reference checks establish the stated owner release.

The first memory harness (348, supervisor 351) accesses `_tensor` before
materializing the library's lazy `torch` namespace. It aborts before memory
measurement. The frozen 352/353 replacement fixes the harness loading order;
the numerical candidate does not change. Original scripts and the failed log
are retained alongside all successful results.

## Integration still required

Finalize the next candidate version and comments, verify equivalence to this
qualified expression, then run distribution, current PaperLeet application,
browser Worker/export and immutable-asset checks. Reconcile moving main and use
the existing guarded adoption process only after those checks pass. WebGPU
remains experimental and disabled as a torch backend; its earlier 0.80 timing
must not silently become a measurement of a newer CPU baseline.

The [machine report](../../packages/stridetorch/benchmarks/results/matmul-packed-rows-qualification.json)
retains every flow ratio, process identity, memory scope and source hash. The
[raw harness archive](../../packages/stridetorch/benchmarks/results/matmul-packed-rows-qualification-harness.tar.gz)
contains **363** verified files, **1,195,489** bytes, SHA-256
`4c9a24ef5a0cd7645f45b1361700926bf74093885356e50c17ea7926799d0f77`.
The archive includes the referenced 342 SDK payloads; prior native targeted
evidence remains in the linked initial candidate archive. External runtime
dependencies and original paths require adaptation for replay. This stage
does not trigger CI, publish a package, create a tag or deploy the application.
