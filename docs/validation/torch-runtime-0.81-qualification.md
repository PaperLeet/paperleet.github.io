# StrideTorch 0.81 qualification

Version 0.81 completes local qualification of the packed-row WASM matrix path.
The current integration source is `e9d3fc54d33a058877181f06fa5bdb4714fe0c26`,
including upstream main through `f5c0e42016a5f2da2969c6263379e44d00fc0c7a`.
The main-adoption receipt is recorded separately after the guarded push. No CI
workflow, tag, package publication or online deployment is requested by this
qualification. BrowserTorch remains a naming proposal; the package is stridetorch.

## Implementation and performance scope

Eligible 2D WASM products pack both reduction rows before the existing NumPy
`einsum('ik,jk->ij', ..., optimize=False)` contraction, then normalize the result
to C order. Existing shape, dtype and exceptional-range guards remain in place.
Native CPU BLAS routing, batched kernels and public APIs are unchanged. The
final expression has the same AST as the measured candidate; version/comment
edits and all 65 runtime/license hashes are bound in experiment 360.

The [matrix qualification](torch-runtime-matmul-packed-rows-qualification.md)
retains 27 complete forward/selected-gradient/owned-copy flows, with native
Torch 2.6 oracles for 77 outputs. Three fresh Node and three fresh Chrome
processes each run 12 balanced rounds with identical read-only operands.
All **162 A/A controls** remain within 5%; no per-process flow or process median
exceeds the predefined 10% regression threshold. No sample/process was excluded.

| Measured full flow | Node candidate/baseline median | Chrome candidate/baseline median |
| --- | --- | --- |
| Strided operands | 0.550–0.617 | 0.553–0.622 |
| Broadcast operands | 0.817–0.928 | 0.836–0.938 |
| Dense C/F 128/256, plus 512 float32 | 0.901–0.941 | 0.907–0.961 |

These are finite real dyadic workloads on one machine, against the frozen 0.80
matrix implementation. They do not establish universal speedups. The 45 ms
measured-batch target was not uniformly reached: minima were 34.519 ms in Node
and 38.5 ms in Chrome. All short batches remain in the report.

Packing introduces temporary copies. Individual-kernel native tracing observes
up to **1,835,306 extra bytes** (about 1.75 MiB), whereas full-call snapshots mask
much of that cost. Actual WASM instrumentation observes up to **2 MiB** of live
contiguity-copy payload across 2,430 full calls; observed copy/output owners
return to zero. WASM heap capacity grows from 43,581,440 to 62,783,488 bytes and
retains its high-water allocation. This is neither browser RSS nor live usage.

## Correctness, distribution and application checks

| Gate | Evidence and result |
| --- | --- |
| Installed standalone SDK, NumPy 2.5.4 | 12,225 passed; experiment 359 |
| Installed standalone SDK, NumPy 1.26.4 | 12,222 passed, 3 existing skips; experiment 359 |
| Actual Node and Chrome Pyodide SDK | 12,212 passed in each, experiments 342/349; candidate expression bound to final source by AST/resource identity |
| Public/private distributions | Both wheels and sdists, both wheels rebuilt from sdists; 65 exact resources, experiment 357 |
| Final public metadata | Fresh wheel/sdist/rebuilt wheel, exact current README, Apache-2.0 and NumPy-only dependency, experiment 366 |
| Final package installation | Actual refreshed public wheel plus verified current private wheel in a new target, both NumPy environments; experiment 376 |
| Full application CPU tests | **45,854 passed, 1 optional Docker skip**, no failures/errors; experiment 361 |
| Native catalog | **89 modules × 3 repetitions**, 12,360 records, passed; experiment 365 |
| Real Node/Pyodide WASM catalog | **89/89 modules, 406 mutants**, no browser-budget exceedances; experiment 365 |
| Immutable browser export | All 16,047 prior JSON assets preserved; 180 new assets reachable, 183 active asset/index identities; experiments 365/374 |
| Full frontend check | All **202 commands** at `e27904f1`, experiment 372 |
| Latest library-filter delta | **15 focused commands**, complete default/Pages builds and real Worker on `e9d3fc54`, experiment 374 |
| Generated Worker and Pages assets | Exact generated-byte proof, asset/retention/release-policy checks pass; experiment 374 |
| Real Chrome nested Worker | Attention 7/7 plus top-level matrix, norm/JVP/HVP, caller-gradient, callback-branch and new packed-route assertions; experiment 374 |

Native Torch 2.6 remains the compatibility target; newer versions differ on
some empty/mixed contracts. The full WASM catalog ran in Node with real Pyodide
0.29.4/NumPy 2.2.5, not all 89 modules in Chrome. Its total was 746,885 ms at
concurrency one. An earlier 0.80 run was faster in wall time, so there is **no
whole-catalog speedup claim**; the controlled matrix comparison supports only
the scoped gains above.

The real Chrome 155 Worker loads the actual exported 0.81 runtime. Its packed
128×64 by 64×128 probe records exactly three `ik,jk->ij` calls (forward and both
VJPs), validates all three sums, and restores the NumPy spy in `finally`.
The Python harness executes the entire learner source, including these probes.
No page errors were recorded. The scoped matrix/norm probes are additional
assertions, not an inflation of the attention exercise's seven-test count.

### Carrying qualification across upstream UI merges

CPU and catalog runs bind the application source at `566dc4aa`. Subsequent
upstream merges change UI/docs and append interview-guide follow-ups; they
change no backend, numerical implementation, problems, judge, Worker or WASM
driver. Experiment 372 verifies unchanged mathematical inputs and append-only
guide data (89 identities, 267 core prompts, 40 previous follow-ups retained,
10 added). It runs the full 202 frontend commands and both builds.

The final six-file `f5c0e42` delta affects library filters/focus and appends only
`.paper-library` CSS. Experiment 374 records all changed paths, verifies the
other 2,348 prior static inputs unchanged, runs all 15 affected library/favorites/
workspace/reading/concept/path checks and both complete builds. This is **202
previous full checks plus 15 current focused checks**, not a fresh full 203.
Experiment 375 rechecks all 2,353 current static hashes and receipts before
the documentation-only metadata refresh. The current private wheel's 102
application/runtime files match the source; no stale application snapshot is
substituted. Full installed-SDK results carry across the metadata refresh by
identity of all 65 runtime resources.

## Retained failures and resource recovery

The package/core [candidate record](torch-runtime-0.81-candidate.md) retains the
initial SDK child-path error and its corrected full suites. Allocation-harness
348/351 failed before measurement because a lazy namespace was not materialized;
fresh 352/353 supplied the measurement. Neither changes numerical tolerances.

Application attempts preserve their original diagnostics. Experiment 363's
temporary backup and 367's frontend check ran out of disk; neither is counted
as a passing gate. Experiment 365 completed catalog/build gates but its browser
fixture waited for a CodeMirror line outside the virtualized viewport before
starting the judge. Fresh 370 fixes the observable end marker and stale version
assertion, and all stages pass; its final result formatter then raises NameError.
Experiment 371 verifies those original completed receipts. Fresh 372 and 374
produce complete successful results with the corrected formatter.

Experiments 364/369 recover space by verifying thin zstd backups of generated
pytest temporary trees. Each payload is either stored in the archive or mapped
to a verified immutable Git blob with path, commit and SHA-256; manifests retain
metadata/symlinks. Restoration requires those recorded Git objects. Test logs,
JUnit, failed attempts and qualified environments remain. Derived `web/dist`
was removed only after checking it was untracked and was rebuilt. Historical
immutable browser assets were not removed. The partial 283 MB failed backup
remains in the evaluation workspace; the portable qualification archive records
its identity instead of embedding it. No numerical failure was hidden or gate
waived to address disk pressure.

## Remaining work

This milestone does not complete independent-library readiness. Existing
norm-HVP costs, broader core API coverage, storage/autograd semantics and
performance across more shapes remain active. The 0.80 raw norm/sine HVP point
corpus still has 4,479 matches and one failure: a one-ULP primal norm difference
can cause a large downstream phase/HVP error. It is not described as a one-ULP
HVP error. Public differentiable HVP outputs and
`autograd.grad(create_graph=True)` remain unsupported.

The [WebGPU whole-call prototype](torch-runtime-webgpu-whole-call-0.80.md)
includes upload, both explicit VJPs and owned CPU readback, compared with 0.80.
It does not measure the new 0.81 baseline. Small workloads were slower and the
opaque learner Worker cannot expose the tested host GPU path. The production
torch GPU backend stays disabled; no global GPU cutoff is inferred.

[Aggregate result](../../packages/stridetorch/benchmarks/results/qualification-0.81.json)
and its linked raw archive retain exact source/evidence identities. Earlier
candidate reports describe their historical stage; this record supersedes their
then-pending local application gates.

The [raw application/distribution archive](../../packages/stridetorch/benchmarks/results/qualification-0.81-harness.tar.gz)
contains 817 individually verified files, 18,419,933 bytes, SHA-256
`b3fbac64965b071a4f29e5c02566ca2ceb999298216b23f1c65c2b07450597de`.
Its external-evidence manifest links the existing core/performance/GPU archives,
records the retained partial-backup identity and explains replay dependencies.

## Final comment-search main reconciliation

Upstream main advanced to `3aa2c298f4d82192c7c6191d8319d659acadabca` during
the first upload retry. That ordinary transfer was stopped after verifying the
expected remote was obsolete; no workflow was cancelled. The new shared search,
reply and card UI is merged at `09c5d6b9a581b0e153f2e3766f65b670e7cdb6bb`.
Experiment 385 passes the **entire current 204-command frontend check**, both
complete builds, Pages/retention/generated-Worker gates and the actual Chrome
Worker with the packed-route assertions. Core/catalog inputs and all 183 export
identities remain unchanged; final public metadata from 366 is still current.
The earlier 202-plus-15 scope above remains an accurate historical stage.

Experiment 384 failed before frontend checks because its preparation omitted
the catalog-count context field. Fresh 385 supplies 89 to the unchanged verifier;
the original failure is retained. The first upload failed with curl 55 and left
main unchanged. Both transport attempts, audits, exact grants, the corrected
204-command run and source manifests are in the
[main-sync archive](../../packages/stridetorch/benchmarks/results/qualification-0.81-main-sync-harness.tar.gz): 86 verified files,
3,744,761 bytes, SHA-256 `ee6299ffe7f736ed623447f9bbb51134dfcd16560192464cdb80e86b658fe27d`.
[Machine-readable reconciliation](../../packages/stridetorch/benchmarks/results/qualification-0.81-main-sync.json).

## Append-only study-data follow-up

Main subsequently advanced to `9104986dfd91734d7a7097e86685a6b2727f842f`.
Only the guide JSON, its existing check and its documentation change. All 89
module identities, 267 core prompts, previous 50 follow-ups and root metadata
are preserved; seven guides gain two follow-ups each (64 total). Experiment
396 records that parsed-object proof. No production UI, backend, problem, judge
or Worker code changes.

At `daa6c4d90f152e857c1d0f4a627f85debe25a2f3`, experiment 398 passes both current
data checks, both complete build profiles, Pages assets/retention/generated
Worker checks and the actual Chrome Worker. The full 204-command source-code
qualification remains experiment 385; this is not a claim of a fresh full 204
on the data-only delta. Canonical 0.81 resources, installed metadata and export
identities remain unchanged. An early launch (397) started before preparation
finished and failed before any gate; its original error is retained. Fresh
398 waits for the prepared frozen plan and passes all gates unchanged.

The 388 upload reached a terminal ordinary ref-lock rejection after 590 seconds
because main had advanced; no new content was overwritten. A subsequent
read-only Git negotiation acknowledges the already received `de22a49f` commit.
Future attempts can use standard `push.negotiate=true` through the same native
wrapper and release guards, reusing objects without a force push.
[Data-follow-up evidence](../../packages/stridetorch/benchmarks/results/qualification-0.81-data-followup.json)
and [raw archive](../../packages/stridetorch/benchmarks/results/qualification-0.81-data-followup-harness.tar.gz): 245 verified files
(151 unique payloads, tar hard links for identical bytes),
2,562,527 bytes, SHA-256 `cf6a15fa23532a2ebec0eee55c73f00ac91638fb9a18494cd9ac5072bc39f519`.

## Verified main adoption

Main adopted 0.81 at **`ffa9d1912abd176bace4767fdb6cb6c97adcb4bc`**. The remote Git commit
root and runtime subtree match the verified local commit. Every one of the 65
runtime/license blobs and all 183 active export identities match the qualified
hashes. The current code baseline passed all 204 frontend commands; the later
append-only data delta passed its two checks, both builds and the real Worker.

GitHub reports zero workflow runs for the adopted commit at the recorded
verification time. No CI/manual workflow, tag, package publication or deployment
was initiated. Pages remains legacy `gh-pages /`. The successful ordinary
native push used the unchanged mandatory wrapper and fresh full source/release
guards, with standard object negotiation; it completed in about 10.13 seconds.
The known raw norm/sine WASM discrepancy remains and the production GPU backend
and experimental terminal-HVP change remain disabled. The broader library goal
continues.

[Remote adoption receipt](../../packages/stridetorch/benchmarks/results/qualification-0.81-main-adoption.json)
and [raw audit/remote-tree archive](../../packages/stridetorch/benchmarks/results/qualification-0.81-main-adoption-harness.tar.gz):
24 verified files, 956,733 bytes, SHA-256
`d94a915a9a5b8f9d86c82e280216ad7ca0a3cc222f2c6248b6c83544c9aacab7`.
