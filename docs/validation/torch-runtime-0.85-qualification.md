# StrideTorch 0.85 CPU/WASM qualification

Version 0.85 removes index-sized boolean comparison buffers from integer bounds
validation shared by gather, selection, indexed addition and selection
pullbacks. The [index-bounds record](torch-index-bounds.md) retains the original
allocation measurements, complete-flow timings and three focused independent
Node repeats of an initial timing anomaly. The measured mathematical source is
unchanged; the runtime resource delta since that candidate is only the package
version literal.

| Verification | Result |
| --- | --- |
| Installed wheel SDK and native indexed comparisons, NumPy 2.5.4 / Torch 2.14.1 | 13,144 passed |
| Same, NumPy 1.26.4 / Torch 2.6.0 | 13,141 passed, 3 existing skips |
| Actual wheel in Node and Chrome Worker Pyodide | 12,466 passed each |
| Default exported CPU VM independent SDK | 12,455 passed across 93 suites |
| Full exported PaperLeet WASM catalog | 89 modules, 406 mutants caught, no budget violations |
| Public/private wheel, sdist and rebuilt wheel | Six artifacts inspected; both native NumPy baselines pass |
| Current frontend command coverage | All 222 checks passed |
| Local and Pages-profile builds | Passed |
| Generated Worker and actual nested problem page | Exact 63-resource identities; 7/7 tests and explicit bounds/gradient/JVP/HVP checks passed |

The native 89-module catalog result is reused only after checking all 1,427
problem, native runner and root-test inputs and all 89 report hashes. These
native PyTorch computations do not call the lightweight bounds predicate. The
fresh real WASM run executes the new adapter through the complete catalog.
This is not described as another complete native application run.

The first frontend run stopped at a stale favorite-pagination assertion. The
component now includes a page-total annotation inside its range status. The
repaired check asserts the exact result range and exact page total separately;
its other interactions remain covered. The unchanged 84-command successful
prefix and all 138 repaired/remaining commands are retained, with no omitted
check. A later build stopped with ENOSPC while copying static resources. Both
failed predecessors remain in the raw record.

The failed partial build was verified against canonical resources before
cleanup. Local build copies then used independent native APFS snapshots for
immutable assets: roughly 580 MB of logical resources initially added about
10 MB of physical storage. Only derived output links to those snapshots; the
original files have separate inodes. Every immutable output was verified
against its original bytes. Mutable assets and generated pages use ordinary
build behavior. This local storage accommodation changes no application or
product build script. Both complete build profiles and the real Chrome Worker
passed afterward.

The installed public package retains 78 resources, the private runtime 67,
default embedding 63 and schedules embedding six. Optional WebGPU imports and
exports remain covered by the wheel SDK; automatic GPU dispatch stays disabled.
The package name remains unchanged pending a separate naming decision.

The [summary](../../packages/stridetorch/benchmarks/results/qualification-0.85.json)
and [raw archive](../../packages/stridetorch/benchmarks/results/qualification-0.85-harness.tar.gz)
contain exact sources, all six artifact identities, installed/WASM results,
complete frontend coverage, both failed predecessors and recovery receipts,
export identities, generated code and actual Worker probes. The archive has
4,010 logical files and 2,822 unique payloads. SHA-256:

`94cab7f2767e82f00b707f66821829b43295a60eaaffa05cf7a5760046006042`

Core qualification uses source base `d8012b46` with separately hash-bound fresh
exports. Later main updates require their own integration qualification before
adoption. No Actions dispatch, tag, package publication or site deployment was
performed by this qualification.

Main subsequently advanced to `31f26cd4`, adding 24 frontend, font and search/
paging paths. Those changes are preserved, including new saved-page recovery
tests. The merge retains separate exact assertions for result range and page
total. All 78 public resources, 95 SDK files and 183 active exports remain
identical to core qualification. Twelve affected and related frontend checks,
both builds, immutable asset copies, generated Worker and actual nested Worker
pass on combined source base `8c196845`. The
[current-main integration record](../../packages/stridetorch/benchmarks/results/qualification-0.85-current-main.json)
and its raw archive preserve that later qualification separately.

The later `c6132c07` main update adds six interview-authoring paths. Source base
`650b48d6` preserves them and passes all 31 interview-related check commands,
including their readonly variants, both builds and the actual Worker. The
[follow-up record](../../packages/stridetorch/benchmarks/results/qualification-0.85-current-main-followup.json)
retains the initial runner argument-splitting failure and its correction.
Numerical sources, SDK and active exports remain unchanged.

Main then added five identity-choice paths at `3c61d129`. Source base `773be1f2`
preserves them; five affected identity/interview checks, both build profiles and
the actual Worker pass. The [next follow-up](../../packages/stridetorch/benchmarks/results/qualification-0.85-current-main-followup-2.json)
retains this qualification and the cumulative 31 frontend path identities.
