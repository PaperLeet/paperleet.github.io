# Iteration 91: norm AD repairs and matched-GC Linear calibration

The independent norm implementation has made concrete progress, but remains
outside production: broader composition tests still identify unresolved cases.
The Linear diagnostic campaign has resumed with corrected calibration after a
real minimum-duration failure. The continuing library goal remains active;
production sources, main, CI, tags and publication are unchanged.

## Norm observations and implementation fixes

Native Torch's lazy ZeroTensor needs materialization before numerical export.
The original probe's direct NumPy conversion read uninitialized backing values
in native 2.6 and raised a ZeroTensor error in 2.14. A direct native diagnostic
confirms that `clone().numpy()` materializes logical zeros in 2.6, while
`numpy(force=True)` alone does not. The corrected probe clones detached outputs
before export and binds each native result to the probe source hash. Initial
observations and code are preserved separately rather than silently replaced.

After correcting that probe, both native versions exposed 92 genuine stage
differences in the isolated norm suite. The prototype now distinguishes direct
JVP from reverse-based functional JVP at singular denominators and preserves
native IEEE multiplication order for nonfinite fractional norms. Functional
HVP uses guarded zero-coordinate rules while retaining NaNs where native does.
These changes do not alter direct JVP's reverse behavior at a zero norm.

| Isolated norm validation | Cases | Matched stages against each native version | Independent tests |
| --- | ---: | ---: | ---: |
| CPU prototype | 780 | 3,660; zero differences | 80 passed |
| Node/Pyodide | 780 | 3,660; zero differences | 80 passed |
| Real Chrome Worker | 780 | 3,660; zero differences | 80 passed |

Each comparison uses independently executed native Torch 2.6 and 2.14 results.
The 3,660 stages include 160 matched expected error stages. The tests add checks
for zero versus NaN functional derivatives, finite differences on smooth
coordinates, and preservation of direct-versus-functional AD behavior. The
actual WASM runs use the same frozen implementation/probes and verify every
mounted input hash. This establishes the isolated suite, not complete norm or
library qualification.

## Broader cases that remain open

A separate 320-case CPU suite composes the norm with square/sine outputs,
exponential/squared inputs, and a normalization expression. It compares forward,
reverse, direct JVP and its reverse graph, differentiable functional JVP and
its reverse graph, and HVP. Against each native version, 1,580 of 1,600 stages
match; **20 still differ**.

The remaining findings involve singular composed HVPs and zero cotangents
propagating through nonfinite upstream values. For example, a native lazy zero
cotangent through a squared infinite input stays zero, whereas the current
materialized NumPy-zero pullback can evaluate zero times infinity and yield
NaN. A standalone norm's correct HVP does not establish correctness of the
normalization chain. These cases are retained for repair, not filtered or
marked as passing. The composition suite still needs actual WASM qualification,
and minimum-NumPy/package/full-library qualification remains outstanding.

## Linear experiment failure and recovery

The prior balanced v2 CPU experiment exited with an assertion after a
float64 medium-backward chunk measured **29.35 ms**, below its unchanged 40 ms
minimum. Its ten completed raw flow records remain preserved; the failing flow
did not reach the old raw-row flush. That process is terminal. Its downstream
WASM and norm waiters also terminated on the failed prerequisite, and the last
WASM waiter recorded that it ran no job. These are observed terminal failures,
not observation timeouts or permission blockers.

Code inspection found a measurement mismatch: calibration ran with GC enabled,
then measurement disabled GC. The fresh v3 campaign in the task's
`torch-iteration-92` directory calibrates and measures with GC disabled, restores
the caller's GC state afterward, and checks two common-count preflight passes
across every label. Calibration requires at least 100 ms in its final preflight;
the measured-chunk minimum remains 40 ms. An inadequate estimate increases the
same iteration count for all labels. Unstable calibration fails explicitly.

The 18 flows, shared physical buffers, three batches and ten Williams orders
remain. Failed measurement rows now flush before rejection; calibration
history is preserved too. Deterministic calibration checks verify expansion of
a short estimate, equal counts across labels, and rejection of instability.

CPU, Node and Chrome numerical validation passed all 18 flows. A single
supervisor now runs three fresh timing processes per runtime sequentially,
with explicit failure progress instead of a chain of stale waiting records.
The first two CPU flows have minimum measured chunks of approximately 99 ms
and 149 ms. Complete timings, control review and source selection are pending.

## Remaining work

Resolve the 20 norm composition differences without narrowing the suite, then
qualify the selected implementation across actual runtimes and packaging.
Complete the new Linear factor campaign, resolve the repeated CPU slowdown and
the six original A/A anomalies, and requalify any production change through the
full library/application/performance checks. Reconcile current main and use
fresh guarded adoption only after that evidence is complete. No GitHub CI or
release is requested as part of this work.
