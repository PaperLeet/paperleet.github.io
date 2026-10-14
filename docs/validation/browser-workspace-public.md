# Browser workspace and learning persistence

Validated on 2026-10-14 against production Pages and hosted builds in the active split public repository. The user story is editing a module, running its original CPU/WASM suite, retaining source and learning results through refresh, and continuing practice without writing synthetic account scores.

## Baseline and correction

The split public snapshot at `637b089` retained whole-workspace localStorage writes and in-memory hosted browser feedback. Two tabs opened from the same snapshot reproduced source loss: editing Attention and then Schedules caused Attention to revert to its starter after reload. Previously reviewed browser workspace and learning changes were transferred from the retained combined checkout's committed source, preserving the split edition guards.

Local and anonymous workspaces now use per-module IndexedDB transactions with compare-and-swap versions. A transaction writes the canonical draft and recoverable source together. Same-module conflicts retain both sources; explicitly selecting a version preserves the source it displaces. A late report updates result history without restoring old submission source over the editor. Legacy local/anonymous caches are migrated only where a module has no new record. Existing legacy caches remain as a fallback copy.

Hosted browser learning is a separate owner- and suite-scoped store. It restores the latest report and earlier accepted result, unlocks Notes and informs local recommendations. Its projection never enters account draft synchronization, official progress, statistics or rankings. Native completion dismisses only the observed browser report. Clearing browser storage loses local records; incomplete writes and abrupt tab termination are not a cross-refresh durability guarantee. It is not a cross-device sync or full execution-history service.

## Browser evidence

| Boundary | Observed result |
| --- | --- |
| Old-origin migration | Existing RoPE reference source downloaded byte-for-byte after migration; prior acceptance still unlocked Notes. |
| Independent modules | Attention and Schedules edited in separate tabs; both exact source downloads survived reload. |
| Same-module conflict | Current source remained in the editor while the competing source was reviewable. Choosing the other version, restoring the first and resolving the conflict retained both recoverable alternatives. |
| Transaction failure | A fault only in the backup store rolled back the canonical source write. A separate observer saw the prior source; the edited source remained downloadable. Restoring storage and retrying persisted the exact edited source. |
| Small screen | Recovery UI at 390 × 844 had document width 390, with horizontally scrollable source blocks. |
| Optional notifications | A BroadcastChannel constructor throwing SecurityError did not blank the page or prevent saving, reload recovery or RoPE's 10/10 browser suite. Unit checks additionally cover send/close failures and disposed handlers. |
| Hosted refresh | Guest Attention 7/7 and Alice Schedules 8/8 remained reviewable after refresh. |
| Owner changes | Alice did not inherit the guest Attention pass. Bob did not inherit Alice's Schedules source or pass. Returning to the guest restored the guest's Notes. |
| Learning storage failure | Attention starter truthfully returned 0/7 with a visible retry warning. Exact source download remained available. After storage recovery/retry, refresh retained 0/7 while the earlier accepted result still unlocked Notes. |
| Native authority | Real temporary private-account SQLite audit: 0 practice runs, 0 accepted runs, 1 account draft; request audit: 0 native judge POSTs. Browser learning never fabricated a server result. |

The hosted fixture used the actual private account/draft handlers on loopback with disposable Alice/Bob accounts. Native execution endpoints were blocked. Credentials were kept in an ignored, permission-restricted fixture, not in this record. This does not assert production deployment or Docker judge validation.

## Runtime initialization regression

A real Schedules run reached the harness with pytest present but `pluggy` missing. The package loader's resolved promise alone was insufficient evidence that dependencies were installed. Initialization now names the complete dependency closure from the pinned Pyodide lock, verifies every loaded package and imports NumPy/pytest before starting the harness. One bounded retry reloads any missing dependency even if its parent was marked loaded; persistent errors retain fallback and diagnostics without extending existing deadlines.

A fresh Pages origin deliberately returned 503 for the first pluggy wheel request. Its next request returned 200, and the same user run completed Schedules 8/8 without manual retry or any API request. The final Pages runtime also completed Attention 7/7 and LoRA 6/6, including maintained reference/mutant calibration and the original gradient/training tests. No numerical tolerances, test counts, references, mutants or experiments were reduced.

## Automated checks and scope

`npm run check` passes, including new real IndexedDB transaction tests through fake-indexeddb, owner/suite/version isolation, failed-write queues, recovery/disposal guards, optional notifications, dependency closure and partial package load recovery. TypeScript and both production editions build successfully.

The Pages initial graph is 386,383 bytes raw / 124,100 bytes gzip; the hosted graph is 390,684 / 125,366. Both retain the existing 400 KiB raw / 130 KiB gzip limits, with the learning controller deferred. The numerical Python core and maintained test bundles were unchanged in this iteration; this iteration did not repeat the preceding full native curriculum CI. Changes were verified locally, not pushed or deployed.
