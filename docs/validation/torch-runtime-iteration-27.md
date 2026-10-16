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

The original reference/mutant/determinism/budget catalog,
immutable export, exported WASM, latest application packaging/frontend checks and
non-force main adoption are tracked with their actual process receipts. Exact
remote Quality and eligible Pages/live verification remain separate gates.
