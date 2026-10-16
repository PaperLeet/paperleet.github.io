# Lightweight torch iteration 16: stateful batch normalization

Repository integration version `paperleet-torch` **0.16.0** adds `nn.BatchNorm1d/2d/3d`,
`nn.functional.batch_norm`, ordinary module aliases and the `_BatchNorm` base
alias. NumPy remains the only runtime dependency. This is an in-repository
integration; all local gates below passed before main publication. Exact remote
Quality/Pages/live checks follow publication and remain a separate proof.

## Behavior and independent oracles

Native Torch 2.14.1 probes established the state/graph contract before
implementation. Training uses biased variance for normalization and an unbiased
running-variance update. Cumulative averages, tracking changes, affine/frozen
operands, train/eval, empty batch/spatial dimensions, attempted-batch counters,
resets, dtype conversion and version-2/legacy checkpoint loading are checked.

Native automatic running-statistic updates do not bump their storage versions.
Explicit input/weight/statistic writes invalidate saved backward data; bias and
output storage are not saved by this kernel. Eval backward reads current running
buffers even if later training calls updated them. The implementation preserves
these observed behaviors, including selective guards for frozen inputs.

JVP input/weight/bias rules hold saved mean/inverse fixed on their reverse tape,
with an identity derivative for the centered source. An ordinary differentiable-
statistics composition produces a different native reverse-over-forward result;
the native graph audit records that discrepancy and the matching rule. Running
buffer forward tangents are ignored by native BN, with strict output independence
still enforced. The functional API supports this behavior. A tracked module JVP
rejects captured batch-counter mutation before side effects, matching native.

Native-reference tests cover four ranks, float32/float64, all tangent inputs,
retained seeds, strided/Fortran layouts, state and mutation boundaries, and complete
Adam MLP training/save/rebuild/load/resume/eval. NumPy-only contracts separately
check analytic/scalar statistics, first-order finite differences, fixed-statistic
reverse-over-forward finite differences, aliases and checkpoint continuation.
Direct differentiable non-leaf edits remain an existing library boundary;
detached aliases exercise the same shared storage counters in mutation tests.
Float16/complex kernels and broader native backend variants remain unsupported.

## Float32 numerical policy

Ordinary conditioned inputs keep the strict native-reference tolerance. The
native CPU fast path expands normalization into channel linear/constant terms;
large offsets with small spread can consequently lose substantial output
precision. See the [primary CPU source](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cpu/batch_norm_kernel.cpp).

This library keeps centered subtraction and selectively recomputes finite
ill-conditioned float32 channels using float64 statistics. The channel-only
inverse is evaluated in float64 and cast once to storage dtype. Independent
float64 centered mathematics validates both output and gradients; generic
native comparison tolerances were not loosened. Three layouts include mixed
sensitive and ordinary channels, with input preservation and running statistics.
The raw native output differences are preserved in the measurement record.
Same-dtype reduction overflow remains possible and its observed non-finite
classes are tested. This policy does not promise bitwise backend equivalence.

## Cost measurement

The reference is a mathematically equivalent **first-order Tensor composition
in this same library**, not native Torch or a previous release that lacked BN.
Its JVP reverse tape is not the native BN rule and is not used as that oracle.
All correctness/build producers terminated before the timing series. Source
and harness were frozen, hashed before/after each process, and verified against
the Chrome and installed-wheel runtime bytes. CPU and Node/WASM are separate
series, each with float32 and float64 balanced composite/joint/joint/composite
fresh processes. All 16 process reports and all nine batch samples remain.

| Workload | CPU float32 | CPU float64 | WASM float32 | WASM float64 |
| --- | ---: | ---: | ---: | ---: |
| 8×16 train backward | 30.8% | 35.5% | 32.5% | 37.5% |
| 256×128 train backward | 19.3% | 17.8% | 18.8% | 16.1% |
| 8×16×16×16 train backward | 20.1% | 23.2% | 25.6% | 24.9% |
| 256×128 eval backward | 2.6% | **−4.3%** | 0.3% | −0.4% |
| affine-only backward | 11.0% | 11.0% | 8.3% | 12.6% |
| bias-only backward | 9.7% | 9.7% | 8.4% | 10.8% |
| training forward, no grad | 20.5% | 18.1% | 17.4% | 21.8% |

Values are local median time reductions relative to composition, with two
processes per implementation/dtype/runtime. Negative values are costs. CPU
float64 eval backward process medians are 0.17930/0.17902 ms versus reference
0.17093/0.17268 ms. The eval path recomputes from potentially changed running
buffers to preserve native semantics; the reference fixture's unchanged buffers
does not exercise that state change. No statistical/hardware, memory, browser
latency or application-wide speedup claim follows from these samples.

## Validation status

- New native-reference BN contracts: **304 passed**; independent NumPy-only BN
  contracts: **67 passed**; combined focused gate: **371 passed**.
- Complete Python infrastructure/package gate: **17,474 passed**, one optional
  Docker skip and seven warnings in 179.82 s.
- Real Pyodide 0.29.4 / NumPy 2.2.5: **273 standalone contracts passed** against
  exact current source, including all new BN contracts.
- Real Chrome 155.0.8059.82 Worker: **273 contracts passed**, with source/test
  hashes verified in the VM. Earlier 265-contract snapshots remain separate.
- Actual Python 3.10.19 / NumPy 1.26.4 installed wheel: **293 passed**, without
  native Torch. Both standalone/private-vendored wheels and source archives have
  the same 36 Python modules and LICENSE/NOTICE; installed paths are disjoint.
- All **5,338** previously published hashed assets remain byte-identical; all
  **156** candidate assets are reachable after source export.
- Full native reference/mutation/determinism catalog: **77 passed**, three
  repeats, 9,987 executions and all 336 mutants caught, 124.288 s.
- Full real WASM reference/mutation catalog: **77/77 passed**, 382.829 s,
  including the new 273 in-VM contracts. This local runner observed no browser
  budget excess. These concurrent correctness gates are not comparative timings.
- All 77 visualization datasets generated. Published/new asset identity and
  reachability were verified again after generation; no orphan cleanup needed.
- Final `npm run check` and production build passed on incoming `e73a8258`.
  One prior attempt hit a real 5 s Python syntax-preflight child timeout and
  terminated. The unchanged standalone check passed in about 0.55 s, followed by
  a complete successful rerun at the original limits. Failed log retained.
- All local release gates passed; exact remote adoption is pending publication.

The [evidence record](../../packages/paperleet-torch/benchmarks/results/iteration-16.json)
retains native semantic/range probes, initial failures, raw performance samples,
source identity, package and VM proofs. `iteration-16-harness.tar.gz` contains
the original experimental runners and prototype kernel. Full local evidence is
under `/Users/david/Developer/paperleet-wasm-evaluation-20261015/torch-iteration-16`.
Incoming UI work through `e73a8258` was fast-forwarded without changing Python,
library, problem, test, catalog or real-WASM harness inputs from main. Final
frontend checks include the new ordered interview-practice group.
