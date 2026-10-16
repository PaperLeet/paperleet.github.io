# Browser run history

The workspace saved browser results but kept calibration and timing information
only in the current judge hook. Refresh or module navigation could therefore
retain acceptance while losing the details needed to understand the run's cost.

New browser reports carry a versioned `browser_run` record: reference/mutant
counts, whether that run reused calibration, and available phase measurements.
It stays attached to the report and its SHA-256 source identity through the
existing guest and owner-scoped browser storage. Rendering selects details from
the displayed report; live-only fallback data must refer to that same report.

The coordinator records its checked calibration, overriding any advisory fields
inside the worker report. Stored history contains bounded scalar data. It carries
no calibration proof key or environment token and cannot populate the page-local
proof cache. Fresh-page runs still perform the full maintained calibration.

The report parser ignores malformed optional history while preserving usable
results. Native reports and reports without source identity do not acquire this
browser record. Existing reports remain readable without invented timings.
Historical end-to-end measurements allow setup before the worker deadline;
worker measurement validation and judge deadlines retain their existing limits.

## Verification

Focused checks cover report/guest recovery, owner and suite isolation, source
identities, malformed and oversized advisory fields, native/legacy compatibility,
measurement consistency, coordinator-owned counts and bilingual restored text.
Existing opaque-worker, proof expiry, captured-source and failure-eviction checks
also pass. Complete frontend checks and all three production editions pass the
existing initial-asset caps; exact edition measurements are recorded with the PR validation.

Actual standalone production UI on runtime 0.20 starts from the existing tagged
958-byte GroupNorm draft. A new 11/11 run records 11 reference tests and six
mutants, with the entire displayed details block retained after reload. A second
fresh-page run calibrates again. A detached-statistics learner fails 7/11 while
reusing the current page's valid calibration; that failure, source and zero-cost
calibration timing recover exactly after reload. Restoring the original draft
then runs full fresh calibration and passes 11/11.

RMSNorm → GroupNorm navigation preserves the final details exactly. Chinese and
English display the same measurements. Downloaded source remains byte-identical
to the initial draft, and no console errors are observed. These are guest UI and
synthetic parser checks; no account or formal submission records are created.

React review retains the existing native disclosure and focus behavior, adds no
effects or duplicated state, and keeps historical parsing in a small shared module
so the compatibility cache remains deferred with the judge.
