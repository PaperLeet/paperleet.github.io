# Parallel complete quality checks

Long serial quality runs can finish after main advances, causing the existing
exact-main Pages guard to skip a release. Independent infrastructure tests now
run in a required job alongside the unchanged native reference/mutation/budget
command and frontend checks. The native command retains three repetitions and
the 600-second aggregate budget. Existing thread counts and per-module limits
remain in place.

The complete WASM catalog runs at most two child processes concurrently in CI.
Each module still gets its own process and Python VM, complete calibration,
all maintained tests and every mutant. Child output limits and deadlines are
unchanged. Completed rows are retained in catalog order, and failed children
produce failed rows while the remaining modules continue. Duplicate input IDs,
mismatched/partial results and invalid concurrency settings fail validation.

Local execution defaults to one process. Maintainers can explicitly use
`PAPERLEET_WASM_CONCURRENCY=2 npm run check:browser-wasm` from `web/`; values other
than 1 or 2 are rejected. CI continues to select the complete catalog.

Both the pilot shell and Pages release resolver now require infrastructure,
native/frontend and complete WASM success. Missing, skipped, failed, cancelled,
in-progress or duplicated required jobs cannot publish. First-party provenance,
exact-main revision checks, authoritative latest attempts and the final
pre-deploy recheck retain their existing behavior.

Verification covers bounded admission with deliberately held completions,
out-of-order completion, failure continuation, exact-once ordered coverage and
malformed child results. Release tests exercise the actual pilot shell with
every upstream job unsuccessful or absent, in addition to resolver fixtures and
the existing stale-main/rerun guards. Complete frontend and real full-catalog
WASM checks validate the implementation; compare per-module reference hashes,
bundle/runtime URLs, packages, test/mutant counts and budgets with the previous
successful source. Cloud correctness and elapsed duration remain part of the
PR review; local elapsed time is not a controlled comparison with GitHub runners.

On source `1305e09d` with this change, complete frontend checks pass and the real
two-process run passes all 78 modules, 619 learner tests and 342 mutants without
browser-budget excess. Every module's reference hash, bundle/runtime URL,
backend, package list, test/mutant count and budget matches successful main run
38893179169. The seven executable/configuration files remain byte-identical
through validation. The infrastructure-required resolver test first fails on
the previous gate, then passes with the added required job.
