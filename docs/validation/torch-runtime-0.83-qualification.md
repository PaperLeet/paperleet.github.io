# Torch runtime 0.83 qualification

Version 0.83 releases intermediate NumPy arrays that multiplication and direct
division do not need for an active first-order pullback. The graph retains shape
metadata and the operand values required by each derivative. Both-active
operands, mutation guards, saved-graph consumption and higher-order behavior
keep their existing contracts. Ten regression tests cover unused-array release
and required-array retention. PaperLeet and the independent NumPy-backed package
share the same 66-resource runtime. The package remains StrideTorch pending a
naming decision; BrowserTorch is a recommendation, not an adopted rename.

## Memory and latency evidence

The preplanned owned-flow campaign measures complete calls, including input
copies and owned result/gradient copies, across native NumPy 2.5.4, native NumPy
1.26.4, Node WASM and a Chrome Worker. Twenty-four flows cover two dtypes,
scalar and tensor chains, both-active operands, division and add/HVP controls.
Twelve fresh timing processes produce 96 aggregate contexts. All predefined
aggregate latency and allocation gates pass. Native Torch 2.6 checks all 24
flows; baseline and candidate results are bitwise identical.

For a representative 256-by-256 float64 scalar chain of length 32, median
traced allocation peaks fall from 18,935,096 to 3,171,256 bytes on NumPy 2.5,
18,945,704 to 3,183,156 on NumPy 1.26, and 18,905,204 to 3,158,588 in each
WASM runtime: about 83% less peak allocation. An actual NumPy-buffer tracing
probe passes in all four runtimes. These are allocation peaks, not process RSS.
All observed Tensor-backed array owners are released after complete calls.

This is not a universal latency improvement. The largest aggregate CPU cost is
about 7.9% for a float32 scalar chain on NumPy 2.5. One individual Node add-control
context remains flagged: its unpaired candidate ratio is 1.1771 and paired
ratio is 1.0197, with paired A/A outside the individual 5% band. The campaign's
three-process aggregate rule was frozen before execution; the original
individual result is retained rather than relabeled as passing. Allocation and
owner instrumentation run separately from ordinary uninstrumented timings.

The measured candidate retains a 0.82 package-version literal. Its 66 resources
match canonical 0.83 exactly after changing only that literal. The
[owned-flow report](torch-operand-lifetime-owned-flow.md) and its raw archive
preserve the plans, twelve processes, allocation samples and native controls.

## Functional and packaging evidence

- The standard application/SDK composite passes 45,942 tests, with one existing
  optional Docker skip and no failures or errors. The SDK portion passes 12,308.
- The installed public wheel on Python 3.10 / NumPy 1.26.4 passes 12,305 SDK
  tests with three existing skips. All 93 SDK source files are included in the
  public sdist and checked against the qualified sources.
- Actual Pyodide 0.29.4 / NumPy 2.2.5 passes 12,295 SDK tests in both Node and a
  Chrome Worker, with no failures or skips. These runs assert version 0.83.
  The independent native Torch 2.6 clipping oracle also passes 478 scenarios.
- Six current public/private wheel, sdist and rebuilt-wheel artifacts contain
  identical 66-resource runtimes. Both native NumPy environments pass
  coinstallation, HVP, clipping, packed-matmul and import-alias restoration
  checks. Private application source/resource bytes are checked in the wheels.
  Apache-2.0, Python >=3.10 and the sole runtime dependency NumPy >=1.26 remain.

The README changed after the full application run, from a development notice
to the qualified 0.83 description. This package-metadata change is explicitly
recorded, not treated as a documentation-only delta. Six fresh artifacts embed
the exact new README; fresh packaging/import tests pass all 12 cases. All 66
runtime resources, 93 SDK sources and 183 active exports remain unchanged.

## PaperLeet integration evidence

The native catalog passes 89 modules with three repetitions and 12,360 test
records. Actual WASM passes all 89 modules and 406 mutants within the existing
browser budgets. Export verification preserves 16,407 existing immutable
assets; all 180 new assets are reachable, and the 183 active export/index
identities are recorded.

The full frontend check, custom-case and WASM-driver checks, both build
profiles, Pages asset checks and exact generated Worker checks pass on the
first merged application baseline. A subsequent main merge changes only
study-followup documentation, its frontend checker and preparation-guide data.
It preserves all qualified package inputs and active exports. Targeted
study-followup checks, both builds, Pages asset checks and an actual nested-page
Chrome Worker run pass again on that merged head. The Worker verifies 62
autograd-runtime resource bytes and executes seven attention/reference and
runtime probes, including the new unused-array release assertions. All seven
pass, with no page errors.

A second frontend-only merge adds shared comment-draft post quotation in nine
frontend/documentation files. Post-quotation, community-card action,
reply-quotation and interview-browse regression checks pass. Both build
profiles, Pages asset/retention checks, exact Worker generation and the actual
seven-test nested-page Worker run pass again. All current package inputs and
183 active exports remain byte-identical. The first guarded adoption attempt
was rejected before any push because remote main advanced to this change;
its failed audit and log are preserved separately.

The CPU suite precedes these frontend-only merges. Explicit source-equivalence
records bind its 1,655 inputs to the application qualification and identify the
separately tested README metadata change. This does not claim that the entire
CPU suite was rerun after a frontend-only merge.

## Preserved failures and limits

Original failed experiments remain in their evidence. A reference harness
initially overwrote duplicate-leaf gradient sentinels. Another WASM invocation
included host-only subprocess tests: 12,295 tests passed, but three unsupported
subprocess cases failed, so that invocation remains failed. A fresh explicit
browser contract collection and the final canonical Node/Chrome runs are
reported separately. No failed invocation is relabeled as passing.

Six existing mixed float64-scalar/float32 nonlinear-cotangent contexts remain
flagged against native Torch's strict reference tolerance. Forward values match
bitwise, and binary pullbacks match when given identical native cotangents.
This is not a claim of full PyTorch numerical parity.

The earlier terminal-HVP optimization and the later parent-shape reuse
prototype are excluded. Experimental WebGPU RPC remains explicit and
production WebGPU remains disabled. The current paired CPU/GPU comparison
finds overhead at small matrices and gains at 256/512 sizes; it does not
establish transparent GPU Tensor storage or autograd support. Existing RPC
cancellation, ownership and cleanup controls should be reused for further work.

[Machine-readable qualification](../../packages/stridetorch/benchmarks/results/qualification-0.83.json)
and its [verified raw archive](../../packages/stridetorch/benchmarks/results/qualification-0.83-harness.tar.gz)
contain the source/export identities, current artifacts, original logs, scoped
merge/metadata records and replay dependencies. The archive contains 763 files,
587 unique payloads and SHA-256
`9282d423e72f9433cd4e2fa5a7ad49cbb2b55ac1356ee0c438ecb106823f7cc5`.
The [development record](torch-runtime-0.83-development.md) remains historical.

Main adopted commit `7e4e8a500f3cb3a875964be08c03254d9c126d57`. The
[remote adoption receipt](../../packages/stridetorch/benchmarks/results/qualification-0.83-main-adoption.json)
verifies all 66 runtime resources, 183 active export identities and seven
release-policy files from that same remote commit. It observes zero workflow
runs for the adopted SHA and confirms Pages still serves `gh-pages` rather
than main. Its verified raw archive preserves the late frontend checks, the
rejected stale-remote attempt and the successful guarded ordinary push.
No tag, workflow dispatch, publication or deployment was requested. The
continuing runtime improvement goal remains active.
