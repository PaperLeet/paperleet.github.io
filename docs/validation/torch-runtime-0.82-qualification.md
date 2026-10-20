# Torch runtime 0.82 qualification

Version 0.82 adds `torch.nn.utils.clip_grad_norm_`, `clip_grad_value_`, the
deprecated `clip_grad_norm` alias and the ordinary `torch.nn.utils.clip_grad`
import path. PaperLeet's generated runtime uses the same source as the public
NumPy-backed package. The package remains named StrideTorch pending a naming
decision; BrowserTorch is a proposal, not an adopted rename.

Clipping preserves existing gradient buffers, non-contiguous storage and
gradient history. A nonfinite norm is rejected before writes; value conversion
precedes consuming a parameter generator. The foreach hints select compatible
per-gradient NumPy operations, without claiming a fused or GPU implementation.
Multiplication and direct division now guard only operands needed by an active
pullback, while preserving saved-graph consumption checks. SDPA keeps its
existing conservative input-mutation contract.

## Functional and package evidence

- The standard application/SDK composite passes 45,932 tests, with one existing
  optional Docker skip and no failures or errors. The native SDK separately
  passes 12,298 tests on NumPy 2.5.4 and 12,295 with three existing skips on
  NumPy 1.26.4.
- Actual Pyodide 0.29.4 / NumPy 2.2.5 passes 12,285 SDK tests in both Node and a
  Chrome Worker. Its isolated snapshot retains a 0.81 version literal; only that
  literal differs from canonical 0.82. The final application Worker separately
  checks version 0.82 and every exported autograd-runtime resource byte.
- An independent native Torch 2.6 oracle checks 478 clipping scenarios across
  native and WASM environments. Another 84 mixed-precision cases pass in each
  native NumPy environment and actual Node WASM. Thirty-six independent native
  controls validate binary-operation mutation guards.
- Six public/private wheel, sdist and rebuilt-wheel artifacts contain identical
  66-resource runtimes. Both native NumPy environments pass coinstallation,
  clipping, HVP, packed-matmul and import-alias restoration checks. All current
  private application source/resource bytes, including daily challenges, are
  checked in the built wheels. Apache-2.0 and NumPy >=1.26 remain unchanged.

The secondary Torch 2.14.1 comparison has three unsupported-complex-clipping
error-class differences from the pinned Torch 2.6 behavior, with no numerical
or mutation differences. This is not a claim of full PyTorch compatibility.

## PaperLeet integration evidence

The native catalog passes 89 modules with three repetitions and 12,360 test
records. Actual WASM passes all 89 modules and 406 mutants without exceeding
the existing browser budgets. Export verification preserves 16,227 existing
immutable assets; all 180 new assets are reachable and the 183 active export
and index identities are recorded.

The full frontend check passes at the first merged application baseline.
Two later frontend-only main merges leave the computation, package inputs,
problem definitions and active exports unchanged. Their catalog/pagination and
inbox changes receive targeted regression checks. Both build profiles, Pages
asset checks, exact generated Worker regeneration and an actual nested-page
Chrome Worker run pass on the latest merged head. The Worker executes the
attention reference plus clipping, HVP, JVP and matrix probes: seven tests pass
with no page errors.

The CPU composite precedes a main merge adding daily challenge Python/JSON
resources and package-data metadata. An explicit source-equivalence record
checks all previous CPU inputs, normalizes only that package-data addition,
and identifies the two new backend files. Fresh packaging/import checks,
installed calendar smoke checks and the full frontend challenge checks cover
that addition. Later merges change only frontend files and documentation.

## Preserved failures and limits

Original failed invocations remain in the evidence. `pytest tests problems`
had 721 unimplemented workspace-exercise setup errors; the standard composite
subsequently passes. One application driver used the wrong working directory;
the corrected continuation passes its previously unexecuted checks. A later
browser driver waited for an offscreen CodeMirror DOM marker before launching
the Worker; a fresh driver uses an end-of-source marker and checks exact
runtime identity. Completed results were retained, not rerun to select passes.

The terminal-HVP latency campaign remains inconclusive and its candidate is
excluded. Operand-lifetime and JSPI/WebGPU experiments are separate; production
WebGPU remains disabled. This release makes no new latency speedup claim.

[Machine-readable qualification](../../packages/torchwasm/benchmarks/results/qualification-0.82.json)
and its [verified raw archive](../../packages/torchwasm/benchmarks/results/qualification-0.82-harness.tar.gz)
contain the current source/export identities, build artifacts, logs, failures,
source-equivalence records and replay dependencies. The archive has 661 files,
628 unique payloads and SHA-256
`e867f2d80b959bf21fbc3a9a634321ee2ffaa53a0eb2a8251f30599e2725c1e9`.
The earlier [development record](torch-runtime-0.82-development.md) retains the
full independent oracle and SDK evidence.

Main adopted commit `905f4da206a6168838efabeae147ec5af55b3fec`. The
[remote adoption receipt](../../packages/torchwasm/benchmarks/results/qualification-0.82-main-adoption.json)
verifies all 66 runtime resources, 183 active export identities and seven
release-policy files from that same remote commit. It observes zero workflow
runs for the adopted SHA and confirms Pages still serves `gh-pages` rather
than main. The ordinary guarded push used normal object negotiation; no tag,
CI dispatch, publication or deployment was requested. The continuing runtime
improvement goal remains active.
