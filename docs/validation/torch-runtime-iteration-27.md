# StrideTorch 0.27: stored gradient layout and ownership

This iteration audits the first gradient allocation and retained non-leaf
accumulation against native PyTorch 2.14.1. The baseline is 0.26 at
`11434fb8fceaa765cad52802ac1355303a61211f`. The independent package remains
`stridetorch`, licensed Apache-2.0, with unchanged `import torch` aliases and
launcher behavior. No independent repository or package publication is implied.

## Contract and implementation

The native [default gradient layout rules](https://docs.pytorch.org/docs/2.14/autograd.html#default-gradient-layouts),
[accumulation implementation](https://github.com/pytorch/pytorch/blob/v2.14.1/torch/csrc/autograd/functions/accumulate_grad.h)
and [layout helper](https://github.com/pytorch/pytorch/blob/v2.14.1/torch/csrc/autograd/utils/grad_layout_contract.h)
distinguish stored leaf gradients from returned derivatives and retained
intermediate gradients. Before this change all first gradients were C copies.

- Dense non-overlapping leaf gradients follow parameter strides; this includes
  transposes, dense permutations, singleton dimensions and empty descriptors.
  Gapped and overlapping layouts fall back to C order. Conventional C/F layouts
  take direct copy paths. General dense descriptors are allocated only after
  proving a compact storage extent; no out-of-bounds temporary view is built.
- The stored buffer always owns separate bytes from user seeds, hook results
  and parameter storage. Native's reference-count-based internal buffer stealing
  is not emulated. Existing leaf gradients retain object/storage/layout and bump
  their shared mutation version when updated.
- Retained non-leaf gradients start as C copies, then accumulate into a new
  object. Holding the old gradient therefore preserves its values and version.
  This also corrects the previous in-place retention behavior. `autograd.grad`
  return layouts are not coerced to parameter layouts.
- Existing forward graphs after model dtype conversion retain their original
  gradient dtype. Saved-state guards, hook routing and requested-input behavior
  are unchanged. Sparse/complex gradients, general higher-order graph creation
  and comprehensive memory-format compatibility remain outside the contract.

## Independent validation

The new native suite covers six 3D permutations, gapped/offset/broadcast leaves,
scalar/singleton/empty layouts, float16/32/64, C/F/broadcast seeds, hook replacement,
post-hook layout observation, existing gradient identities/versions and dtype
conversion after forward. Independent NumPy-only tests add strided SGD training,
buffer reuse, half gradients, input/seed ownership and retained snapshots.

Focused tests pass256. A first draft incorrectly attempted unsupported half
optimizer training (six failures); the corrected tests retain the existing
float32/64 optimizer boundary and test half gradient ownership separately.
Python3.10.19 / NumPy1.26.4 installs the actual standalone wheel without native
Torch and passes853. Node and headless Chrome155 each execute the pinned
Pyodide0.29.4 / NumPy2.2.5 and pass841 tests across27 standalone suites. All45
runtime/license hashes agree with the frozen candidate. Public/private wheels
have disjoint paths and each survives uninstalling the other distribution.

A first broad pytest command accidentally included `problems` without its
implementation plan:27059 tests passed, with619 fixture setup errors for missing
workspace learner files and one optional Docker skip. Its log is retained. The
correct base command and official original-catalog CI are separate gates below.

## Timing and limitations

Fresh sequential ABBA children compare identical logical workloads on CPython
and actual WASM for float32/64. Nine batches follow three warmups. Small first
gradient calls use500 iterations, large copies60, reused graphs300 and full
square/sum/backward30. CPU requests two BLAS threads; actual BLAS threading is
unverified. No owned tests/builds overlap these timing runs; the shared host is
not otherwise controlled. These are steady-state latency observations, not
application, cold-start or peak-memory claims.

An initial trial used a general allocation path even for ordinary F layouts;
its timing and source are retained. An audit then found that NumPy's strong
integer scalar promoted the nominal float32 first-copy inputs to float64. Those
early first-copy samples are not valid float32 evidence. The final run uses a
Python integer divisor, asserts actual parameter/seed/gradient dtypes and records
their shapes/strides. Its unchanged square/backward inputs were already correctly
typed. The adopted source uses direct F copies and one owning allocation for
nonempty dense permutations. Final F-layout square/sum/backward reduces observed
time23.1–29.5% on CPU and10.3–14.9% on WASM,
because the intermediate gradient already follows the parameter layout.

Costs are explicit: small C first-gradient calls increase1.3–5.5%. For a large
C seed supplied to an F leaf, first allocation increases54.6–263.4%; dense permuted
leaf allocation also regresses. The old behavior copied contiguous bytes into
the wrong layout, so these are correctness costs rather than equivalent-layout
copy throughput comparisons. C square/backward increases1.0–2.3% on CPU and
3.2–6.3% on WASM in the final run. The earlier optimized WASM float32 run showed
17.6% regression, with candidate process medians0.0955/0.0759ms versus baseline
0.0724/0.0735ms; it is retained alongside the final run. Small/reused graph
samples vary substantially, including an unexplained CPU float32 improvement;
no reliable benefit is inferred from those cases. These observations do not
establish unchanged C workloads or a universal speedup. Further profiling should
separate descriptor checks, strided copying and full optimizer access.

## Application and release gates

The corrected complete base suite passes27059, with one optional Docker skip
and34 warnings in193.61s. Official `paperleet ci --budget 600` passes78 modules,
10218 test executions, three repeats and342 mutants in127.54s, with no issues.
These durations are budget gates, not controlled speed comparisons.

The integration merges latest main `f445118aa4b17f3029a8a076231516422c9d0ea0`,
preserving all16 upstream UI/docs/dependency files byte-for-byte. These merges
change no Python/runtime/test source, so the complete base/native source gates
remain applicable. Both actual application distributions contain the same45
core/license hashes as the independently installed wheel; public/private installed
paths remain disjoint. The current application catalog/server files match.

All7072 previously published immutable files remain unchanged. Export adds158
reachable immutable files and freezes161 current catalog/detail/manifest/runtime
files. Full actual exported WASM passes78/78 modules and342 mutants without any
calibration/run/total budget violations, using CI's two independent processes.
It takes289863ms on this runner; that duration is a budget gate, not a controlled
comparison with earlier versions. Attention's exported VM additionally passes all
27 maintained independent suites/841 contracts, with exact frozen suite hashes.
The canonical source, harness, driver and all161 export hashes remain unchanged.

The autograd payload is474542 raw bytes /109955 local gzip-estimate bytes,
containing39 package Python modules, LICENSE/NOTICE and four compatibility
facades. The schedules profile is37343 /11550 bytes, with three package modules,
the licenses and four facades. Gzip estimates do not assert network compression.

The first exported attempt used the script's default one-process configuration;
it was explicitly stopped to align with CI's configured two processes and its
partial log is retained. During the next attempt an external shared dependency
target lost Pyodide; later child startups failed to import it. Its failed full
report/log is retained. The owned worktree symlink was then replaced by its own
`npm ci` tree, leaving the original target untouched. All14 Pyodide package files
match the pre-interruption hashes exactly. The final full run succeeds using this
independent dependency tree and upstream's locked KaTeX0.18.2.

Complete frontend checks pass54 bilingual guides/162 prompts and the upstream
math-renderer security checks. Default and actual hosted Pages-profile builds
pass existing budgets: three initial JavaScript assets, respectively401875 raw /
129843 gzip and403239 raw /130323 gzip bytes. Actual profile settings come from
the observed repository variable. Logs retain ephemeral Vite startup/shutdown
messages; every complete command/assertion returns0. No application browser
visual QA is claimed.

[Integration evidence](../../packages/torchwasm/benchmarks/results/iteration-27-integration-gates.json)
and the212-file receipt archive retain the frozen exports, actual reports,
dependency interruption/recovery and process receipts. Main adoption uses a
non-force push after latest-ref revalidation. Exact remote Quality and eligible
Pages/live verification are tracked separately from these successful local gates.

## Actual remote adoption

Main adopted `f1b33b0be7b47bb9e06ff92ba60a5032fb8cea06` by non-force push.
Its actual [Quality38908465541](https://github.com/PaperLeet/paperleet.github.io/actions/runs/38908465541)
passes all four required jobs. The infrastructure log reports27059 passed/one
optional Docker skip in541.01s; native original CI passes78/10218/342/three repeats
in316.31s; exported WASM passes78/342/no budget violations in296387ms, including
the same27/841 independent suite hashes. These are runner budget/compatibility
gates rather than comparative performance evidence.

Main advanced before an eligible source-commit deployment was observed. Descendant
`9f18e510b8f72cf65222a2b94d9e70b9720f10d0` changes only ten UI/docs/build-script
files: all45 source/resource and161 frozen export hashes match the original
commit. Its actual [Quality38908787448](https://github.com/PaperLeet/paperleet.github.io/actions/runs/38908787448)
passes all four required jobs and
[Pages38910349850](https://github.com/PaperLeet/paperleet.github.io/actions/runs/38910349850)
completes the actual deployment steps successfully. Downloaded descendant reports
pass native78/10218/342/three repeats in416.25s and exported WASM78/342/no budget
violations in524859ms, with identical per-case source hashes, counts, packages,
budgets and27/841 standalone contracts.

[Remote adoption evidence](../../packages/torchwasm/benchmarks/results/iteration-27-remote-adoption.json)
and its14-file archive retain both runs, byte equivalence and actual live checks.
All78 decoded starters/current detail/manifest references and158 immutable payloads
match the frozen version, including both runtime byte sequences and LICENSE/NOTICE.
Version0.27 is positively verified on the site through that runtime-equivalent
descendant; the original commit's absent eligible deployment is kept distinct.
