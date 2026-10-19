# StrideTorch 0.80 qualification

The 0.80 candidate is applied on `codex/torch-transformer`, after merging upstream
main through `ec4370149d7e19fdb9368262965e442f23b0f3e9`. Standalone correctness,
installed-package ownership and the measured CPU/WASM performance controls have
completed. The full application CPU suite passed; remaining current PaperLeet integration
gates are in progress. This
candidate has not been adopted on remote main, published, tagged or deployed;
no GitHub workflow was dispatched. BrowserTorch remains a naming proposal.

This version record is distinct from the earlier numbered iteration-80 report,
which describes an older affine-product development checkpoint.

## Implementation and scope

The selected source is experiment 284, with the package version updated to
0.80.0 and a stale vector-norm module description corrected. Its 65 runtime
resources match the staged public/private package manifest exactly. NumPy remains
the only runtime dependency; ordinary `import torch` scripts retain the launcher
and explicit host alias entry points. Apache-2.0 and native Torch coexistence
remain unchanged.

- Captured Tensor numerators retain reverse history through division. Functional
  JVP/HVP respect graph disconnections through `no_grad()` and views, retain caller
  gradients, and allow the same callback to alternate between polynomial and norm
  branches. There are 120 new public regression cases across six test files.
- Scoped Tensor pullbacks use ordinary Tensor objects, visit the actual output's
  ancestors, and release temporary owners after both success and failure.
  Unrecorded graphs retain their previous HVP route. Public
  `autograd.grad(create_graph=True)` and differentiable HVP outputs remain
  unsupported.
- Arithmetic sets up numerical reverse nodes and Tensor descriptors together.
  Numerical callbacks are elided only during the private Tensor reverse pass.
  Dtype descriptors and an already validated first traversal order are reused
  within an invocation; there is no cross-call Tensor or graph cache.
- Nonempty nonscalar C/F-contiguous real scalar powers allocate their result in
  the first ufunc. Scalar, empty and other strided layouts retain the explicit
  buffer route. Earlier unrestricted allocation changed broadcast output strides
  and was rejected; the selected route passed 384 layout controls across four
  runtimes, including values, shape, dtype, strides, storage offset, ownership
  and input nonmutation.

## Standalone correctness and installation

| Check | Result | Scope |
| --- | --- | --- |
| CPython SDK | 12,225 passed | NumPy 2.5.4 |
| Minimum environment SDK | 12,222 passed, 3 skipped | Python 3.10 / NumPy 1.26.4; existing rank limits |
| Actual Node Pyodide SDK | 12,212 passed | Pyodide 0.29.4 / NumPy 2.2.5 |
| Actual Chrome Pyodide SDK | 12,212 passed | Chrome 155.0.8059.82, same frozen source |
| Public/private packaging | Passed | Wheels, sdists and wheels rebuilt from each sdist; all 65 resource hashes |
| Installed ownership | Passed | Actual coinstallation and uninstall/reinstall in both directions |
| Native Torch coexistence | 13 standalone tests passed | Actual installed candidate packages; nested aliases restore cached native modules |

The initial native dependency installation could not resolve an offline cache.
That failed attempt is retained. Its correction uses a fresh environment with
the actual candidate wheels installed there and an explicit path to a previously
qualified, read-only Torch 2.14.1 dependency environment. Candidate import paths
and hashes are checked; 130 files in the earlier installed public/private
packages remain unchanged. This is dependency reuse, not a claim that the
initial offline installation succeeded.

Native Torch 2.6 is the compatibility target. Newer native versions differ on
some empty and mixed-dtype matrix contracts. Matching the SDK does not establish
universal native parity.

The final current-source rebuild also verifies all packaged application Python
files and `interview_catalog.json`, rather than reusing the earlier application
snapshot. Both wheels and both sdists rebuilt into wheels match the 65 resources;
actual coinstallation in a separate target passes HVP/caller-gradient and nested
native-alias restoration checks. Generated `egg-info` is excluded from frozen
source inputs; an initial overly broad snapshot and its failure remain retained.

The raw WASM/native HVP point gate still records **4,479 matches and one failure
out of 4,480 cases**. In the retained large finite float32 p=2.9 case, the norm
differs by one ULP before sine. That phase difference can cause a large downstream
HVP difference. Native equivalent expressions and a controlled replay reproduce
the WASM path. The original failure and tolerances are preserved; the downstream
error is not described as merely one ULP.

## Measured performance

The baseline is the frozen 0.77 tree at
`4d032047aa9f5db71f3b4dfc5eac0db2e04855a9`, not the currently moving remote
main. A0/A1 independently load identical baseline code; B is source 272 and C
is selected source 284. Each CPU, Node WASM and Chrome WASM runtime used three
fresh processes for each workload family, balanced four-implementation orders,
the same read-only operands/seeds, output preflight checks and disabled GC during
calibration/measurement. No timing process was excluded or replaced.

There are 24 ordinary flows per runtime/process and 12 HVP flows, each covering
float32 and float64. Every ordinary flow and all eight preexisting HVP flows
stayed at or below 1.10 C/A0 in every process. All A/A controls stayed within 5%.
The new norm flows are reported separately because 0.77 has no vector-norm API;
their baseline is an equivalent expression.

| Workload | CPU C/A0 median | Node WASM C/A0 median | Chrome WASM C/A0 median |
| --- | --- | --- | --- |
| 2,048-element power forward, float32 | 0.387 | 0.496 | 0.502 |
| 2,048-element power forward, float64 | 0.335 | 0.300 | 0.304 |
| Worst ordinary-flow median | 1.053 | 1.056 | 1.057 |
| Worst existing-HVP median | 1.086 | 1.044 | 1.045 |

Ratios below one mean lower elapsed time. CPU polynomial HVP medians remain
1.077/1.086 for float32/64, although descriptor reuse improves them by about 3%
relative to source 272. New norm HVPs cost roughly 2.8–4.5 times the baseline
equivalent expression. They are not advertised as faster, and reducing that
cost remains active work. These measurements qualify these workloads, not
library-wide performance or arbitrary shapes/hardware.

All per-process ratios and evidence hashes are in
[`qualification-0.80.json`](../../packages/stridetorch/benchmarks/results/qualification-0.80.json).
Earlier rejected source-251 layout changes, source-272 timing regressions,
control drift and failed harness attempts remain in the experiment archive.

## WebGPU experiment

A matched workload provider now executes the real CPU library's forward matmul
and analytical VJPs, with owned NumPy copies of all three results. The explicit
GPU program computes the same forward/left-VJP/right-VJP results, also reading
all three back. Cases include a rectangular matrix and 64/256/512 square
matrices, both cold inputs and resident handles. Read-only inputs, repeated
calls and output independence are checked against native Torch 2.6 and 2.14.

The real Chrome fixture completed 48 whole forward/VJP calls through the opaque
Python Worker and a bounded host GPU service. It submitted/completed 42 GPU
programs via 120 RPC requests and returned to zero live handles/allocations;
peak tracked GPU allocation was 7,340,096 bytes. The adapter reported Apple
Metal 3 with `fallback=false`; this does not establish results for other devices.
The opaque worker retains
origin `null`, no exposed WebGPU and the existing sandbox. The 512 case is split
into two programs to fit the unchanged aggregate work limit. No quota is raised.

These are correctness, transfer and lifetime controls. No matched latency has
been measured, and no production torch GPU backend or automatic autograd/device
routing is enabled. Whole-call timing against the improved CPU implementation,
including cold/resident transfers and readback, is required before selecting a
GPU route. A previous fixture's secure-context hashing error was found before
GPU execution; its replacement verifies payload hashes inside Pyodide.

## Application integration still required

The full application CPU suite completed with 45,854 passed, one optional Docker
submission test skipped and no failures/errors (45,855 collected). Source/test
hashes were checked before and after execution against merge head
`872535b4b82d38238d0e0d0b118ca42b954aa21a`. Three native catalog repeats then
passed across all 89 modules: 12,360 test records in 153.22 seconds, with no
issues. Export verification preserves 15,874 existing immutable resources and
checks 180 new reachable payloads plus 183 active identities.

The initial WASM child could not start because an npm-cache Node executable had
been removed. That terminal failure is retained; no WASM test ran in it. The
continuation preflights installed Node 22.22.1 and incorporates upstream's
11 frontend/docs changes at merge head
`a7cfbbde2d9c148821f4bd47f9d909003e138f1b`. All 1,584 non-frontend inputs are
byte-identical, so the passing CPU/native proofs carry forward. Current frontend
and browser execution are checked anew. The full WASM catalog then passed all
89 modules and 406 mutants in 488.487 seconds, with no budget exceeded.

Frontend qualification exposed an unguarded browser-location read during static
Community rendering; a one-line guard follows the neighboring component's
existing behavior. Two older discovery/publication-period fixtures still expected
automatic page clamping, conflicting with upstream's documented explicit recovery
feature. Their responses now return empty items for expired offsets, assert the
requested page remains unchanged without another automatic request, then activate
the real recovery control and retain their original scope/pagination assertions.
Both targeted checks and the existing navigation/static-render check pass.
Original failures are retained. No production automatic redirect is restored.

Upstream's subsequent column-sorting change is merged at
`4915226e9ec885c76f63689cdb612d8672fcbe9a`. All non-frontend and installed-package
source inputs remain unchanged. A diagnostic sweep ran all 198 current frontend commands: 192 initially passed
and six failed. The six original checks now pass after synchronizing the mounted
interview fixture URL, resetting search on actual source-scope changes while
excluding directory ordering and irrelevant report-page numbers, distinguishing
explicit recovery scrolling from late response scrolling, and exercising the
current recovery button. The existing assertions for drafts, native fields,
selection, private code, filters and stale responses remain. The initial broad
search reset was rejected because it discarded company-ordering drafts; that
failed source and both failed focused results are retained.

The React checklist covers stable primitive effect dependencies, existing labeled
native controls, unchanged data requests and imports, and the existing typed
reset-scope prop. A subsequent complete-App check rejected the route-based reset during local
learning-state recovery. Search now uses the existing report/summary request
parameters as its primitive scope. This preserves pending query, native selection
and horizontal scroll when local ordering/practice filters change, while actual
evidence-source changes reset to the applied query. The six original checks and
the real App library/filter-recovery journey all pass (seven focused checks).
The rejected route-based source and full-App failure remain retained.
Full frontend/type/build verification is still pending.
Remaining gates are the complete frontend check, both build profiles, generated
Worker identity, retained
immutable assets and an actual nested-route browser submission with matrix,
norm and alternating-branch HVP assertions. Existing payloads are preserved.

After those checks, reconcile any newer upstream main changes, archive/audit
the exact evidence and use the existing guarded main-push procedure. No CI,
release or deployment is part of this local qualification. Broader core coverage,
independent-library readiness and GPU backend work remain active.
