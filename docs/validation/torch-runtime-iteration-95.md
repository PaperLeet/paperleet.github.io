# Iteration 95: full-source dispatcher diagnostic

An isolated dispatcher variant removes the empty options dictionary on the
common path and computes the unsupported option names at registration time.
This is a candidate for measurement, not a demonstrated explanation of the
earlier full-version CPU regression. Production sources remain unchanged.

## Preserved behavior and version-specific metadata

Stdlib-only tests load the actual old and candidate registry definitions with
stub tensor/state dependencies. Thirty-three call observations match on Python
3.9.6 and 3.12.11, including repeated calls through the pattern cache, binding
errors, unsupported options, implementation-owned options and dynamic AD guards.
The checks also verify the 32-pattern cache bound, release of payload-bearing
keyword subclasses, registration validation and serialization references.

For 100 synthetic no-option registrations, the existing registry retains 100
distinct empty frozensets on Python 3.12, totaling 21,600 shallow bytes. The
Python 3.9 run retains one, totaling 216 shallow bytes. The candidate captures
none in these operator closures. These are specific metadata observations,
not total process memory, browser measurements or a speed result. The earlier
assumption that all Python versions retain 100 was rejected by the 3.9 run.

## Matching complete-version comparison

The next independent numerical queue uses these five labels:

| Label | Runtime source |
| --- | --- |
| A0, A1 | Independent imports of complete 0.77, 62 resources |
| B | Complete 0.78, 63 resources |
| C | Complete 0.78 with only the legacy 0.77 `_nn.py` |
| D | Complete 0.78 with only the candidate `_registry.py` |

Source hashes prove C and D differ from B in exactly those files, and C's
Linear source matches 0.77. Every current canonical runtime resource and all
79 existing SDK test files used by this queue are also frozen by hashes.

The queue first runs the 11,982 existing SDK contracts against D, then validates
values, gradients, dtype, shape and physical input sharing for all five labels.
It measures twelve Linear flows across three fresh CPU processes: float32 and
float64 tiny forward, medium forward/backward, batched forward/backward and
batched forward without bias. This is a focused diagnosis; it does not replace
the full 124-flow application/performance verification.

Each process uses three blocks of ten Williams orders, balancing positions and
ordered predecessors. Calibration and measurement both disable GC, every label
uses the same iteration count, preflight requires 100 ms and actual measurements
retain the 40 ms minimum. Raw rows are flushed before any timing rejection.
The final verifier recomputes each aggregation and paired comparison from logs,
and retains all control anomalies. Stdlib-only sampler checks cover 100 order
seeds and a deliberately underestimated calibration.

## Sequencing and remaining work

The queue waits for the known iteration-94 supervisor to terminate and records
its terminal result before numerical execution. Iteration 94 itself waits for
the current iteration-92 Chrome timing process. This prevents competing owned
numerical work from contaminating timings. The dispatcher experiment uses its
own 0.78 snapshots and does not validate the separate norm/graph prototype.

At preparation time CPU and Node factor runs and two Chrome runs were terminal;
the third Chrome run remained live. The full factor campaign, expanded norm
qualification and this dispatcher diagnostic still need their terminal results
reviewed. The original full-version CPU regression and six original control
anomalies remain unresolved. No production optimization is selected yet.

The evidence directory `torch-iteration-95` contains both Python contract
records, four source snapshots, the frozen 260-input experiment plan, sampler
checks and the sequential supervisor. No main adoption, GitHub CI, tag, release,
publication or deployment was performed. The continuing core/API/backend goal
remains active; BrowserTorch remains a naming proposal.

## Execution checkpoint

The dispatcher candidate subsequently passed all 11,982 existing SDK tests on
CPU with no skips. All twelve five-label numerical validation flows passed.
The first of three full-source timing processes then started. The performance
result remains pending; neither contract compatibility nor those validation
flows establish a speed improvement.
