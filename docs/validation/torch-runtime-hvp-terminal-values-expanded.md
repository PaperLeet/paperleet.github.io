# Expanded terminal-HVP experiment: correctness and memory pass, timing inconclusive

The isolated terminal-HVP candidate still differs from qualified 0.81 in only
one computation: its final private Tensor reverse traversal runs under
`no_grad()`. The public API already returns detached HVP outputs. The canonical
runtime has not changed, no new release version is assigned and the candidate
is not adopted on main. No CI, tag, publication or deployment is requested.

## Functional scope

Experiment 407 retains the original 14 pilot cases and adds 18 full calls,
covering tuple inputs, a captured coefficient, axis reductions, normalization,
strided inputs, gradient barriers, discarded norm branches and zero-order norms.
The 32 cases use an independent actual Torch 2.6 oracle. Both native NumPy
environments match that reference, with exact baseline/candidate output bytes,
preserved caller buffers and unchanged inputs. Every measured process repeats
those preflight checks before timing its cases.

Actual Chrome 155/Pyodide 0.29.4 passes all 12,212 SDK tests on the candidate
(408), complementing the earlier actual Node SDK and both native SDK suites.
Experiment 409 executes the unchanged expanded 4,480-case norm-HVP corpus in
four fresh contexts: baseline and candidate in Node and in Chrome. Each
candidate raw JSON observation is byte-identical to its corresponding baseline.
All 4,480 cases select the Tensor reverse path with no missing rule or leaked
context. The original native comparison remains **4,479 matches and one
difference** against each preserved Torch reference. The retained
`float32/finite_large/2.9/sin_out` phase discrepancy is not repaired or hidden.

## Ownership correction and memory

The original 381/391 pilot returned `.numpy()` views. Its former description as
independent owned copies was inaccurate; the
[pilot record](torch-runtime-hvp-terminal-values-pilot.md) now corrects that
description without changing frozen results. Experiment 407/410 explicitly
uses `.detach().cpu().numpy().copy()` and verifies `OWNDATA` and a null base.
The expanded timing scope includes these independent output copies and is
distinct from the old view-returning pilot.

Separate memory checks pass all 32 cases in NumPy 2.5.4, NumPy 1.26.4, actual
Node WASM and actual Chrome WASM. Per environment, 1,920 complete calls are
instrumented. After collection, all observed temporary Tensors, contexts and
Tensor-backed array owners return to zero; fixed input/gradient arrays are
explicitly excluded. Caller buffers and subsequent gradient mode remain valid.

For the 64-element float32 norm-square case, node construction falls from 70 to
42 per call. NumPy 2.5.4's median traced peak falls from 155,175 to 100,347 bytes.
Native tracing is performed separately from owner instrumentation, following
explicit garbage collection; these are traced allocation peaks, not RSS or a
claim of immediate release without collection. Some control peaks increase
slightly: the largest observed extra is 704 bytes on NumPy 2.5.4 and 728 bytes
on 1.26.4. WASM heap capacity is recorded separately as retained high-water
allocation, not live object usage.

## First repeated timing campaign remains inconclusive

Experiments 410/412 run all **12 planned fresh processes**, three each in the
two native NumPy environments, Node and Chrome. Each measures all 32 cases in
12 balanced implementation orders, with independent A0/A1 baseline imports,
the same read-only operands and candidate B. GC is disabled during calibration
and batches and restored afterward. Workload and implementation orders rotate
across cycles. Every calibration attempt, batch and process is retained.

The predefined targets were 100 ms calibration, 60 ms measured batches, an
8,192-call cap, 20 ms minimum batch, 5% A/A tolerance and 10% candidate-regression
threshold. The observed minimum batch is 41.7 ms; the 60 ms target is not
uniformly attained. No process/case exceeds the candidate's 10% regression
threshold. However, **five of 384 A/A controls fail**, so the overall performance
qualification is **not passed**:

| Runtime / cycle | Case | A1/A0 |
| --- | --- | --- |
| NumPy 2.5.4 / 1 | axis, float32, 64 | 0.93839 |
| NumPy 2.5.4 / 1 | captured coefficient, float64, 64 | 0.94919 |
| NumPy 2.5.4 / 2 | gradient barrier, float32, 64 | 1.06446 |
| Node WASM / 1 | norm-sine, float64, 1,024 | 1.05681 |
| Node WASM / 2 | gradient barrier, float32, 64 | 1.05113 |

The raw data contain substantial within-case baseline shifts. One Node baseline
batch ranges from about 51 to 357 ms. Paired-block diagnostics can differ from
ratios of separate medians; they are diagnostic only and do not retroactively
turn this campaign green. A paired diagnostic also finds an additional marginal
NumPy 2.5.4 tuple-case deviation. Numerical parity, lower construction counts and
scoped memory reductions do not establish complete latency qualification.

## Prespecified precision follow-up

Before executing another measurement, 415/416 freeze one six-process follow-up
for the two inconclusive runtime groups: NumPy 2.5.4 and Node, three fresh
processes each. **Every one of the same 32 cases remains**, with the same source,
oracle, read-only operands, owned outputs and 12 balanced orders. It adds 200 ms
warm-up per implementation, raises calibration to 250 ms and the batch target
to 150 ms, and requires a 100 ms minimum batch. Both the original ratio of
medians and the median paired-block ratio must satisfy the unchanged 5% A/A
and 10% regression bounds. There are no individual-case replacement runs.

The complete first campaign remains failed/inconclusive and preserved. The
follow-up is a separately identified precision experiment, not deletion or
relabeling of those five failures. All three original NumPy 1.26.4 and Chrome
processes already pass their original controls. The follow-up is now complete and remains inconclusive (see below). The
terminal-HVP candidate is not adopted; release packaging and application/export
qualification for this optimization have not been performed.
The independent-library objective, further core APIs and GPU investigation
continue; production GPU support stays disabled.

[Machine-readable first campaign](../../packages/torchwasm/benchmarks/results/hvp-terminal-expanded-first-campaign.json)
and [verified raw archive](../../packages/torchwasm/benchmarks/results/hvp-terminal-expanded-first-campaign-harness.tar.gz):
231 files (160 unique payloads, internal
tar hard links for identical bytes), 1,294,084 bytes, SHA-256
`0e95b66a92a207648dd874d423117735ada33dd1b72e136b62bbd1a6beb8e39b`. The replay-dependency manifest links the retained original
native oracles in the existing 0.81 data-follow-up archive; external runtime
dependencies and original absolute paths need adaptation. The precision source
and plans are included before any precision-run output exists.

## Completed precision follow-up: retain failed gates, no adoption

All six prespecified processes have complete 32-case primary records. The
original supervisor stopped after its second process: Node's primary JSON
contains all 32 cases, but its stdout mirror contains only the first 12.
Experiment 421 verifies every primary batch, recomputed median, both ratios,
source identity and the matching stdout prefix. That failed mirror check is
preserved. A fresh driver adds a per-case ledger and synchronous host writes;
423 runs only the four previously unrun processes. All four durable ledgers and
stdout mirrors match their respective primary records. No completed measurement
is replaced or repeated.

Across all 192 precision cases, 12 A/A controls fail (all in Node). Two first-cycle
batches fall below the unchanged 100 ms floor: 97.854 ms on CPU and 99.643 ms in
Node. CPU cycle 2's float64 zero-norm case also exceeds the 10% candidate
regression limit using the ratio of medians: 1.17359, while its paired-block
ratio is 1.02126. Both estimators were required; the paired diagnostic does not
waive that failure. The result remains **inconclusive**, and the canonical
runtime retains its qualified 0.81 HVP implementation. Numerical and scoped
memory evidence remain useful, but do not establish a complete latency benefit.

[Machine-readable precision results](../../packages/torchwasm/benchmarks/results/hvp-terminal-precision-followup.json) and [verified raw archive](../../packages/torchwasm/benchmarks/results/hvp-terminal-precision-followup-harness.tar.gz): 48 files, 991,560 bytes, SHA-256 `8aa3b3ac0456db5f5e43f8b2acd95b4c53bb0bb589c62eeba4b9a0441ff132d1`. The replay manifest links the earlier archive for frozen implementations, workload and independent oracle inputs.
