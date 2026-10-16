# Lightweight torch 0.18: weighted spatial classification

This iteration expands cross entropy beyond the old `[N,C]` / int64 `[N]`
contract and adds `nn.functional.nll_loss` / `nn.NLLLoss`. NumPy remains the only
runtime dependency. Standard `torch` aliases, the script launcher, Apache-2.0
resources and disjoint standalone/private package layouts are preserved.

## Supported objectives and gradients

Cross entropy accepts unbatched `[C]`, batched `[N,C]` and spatial `[N,C,...]`
float32/float64 inputs. Class-index targets use int64 and support class weights,
ignored labels, all three reductions and label smoothing. Native mean reduction
uses the sum of valid selected class weights; smoothing does not change that
denominator. Zero/negative weights and zero denominators are not silently fixed.
Negative/NaN smoothing follows the observed native unsmoothed branch, while
smoothing above1 raises. Nonfinite logit behavior and ignored rows are checked
against native values/gradient classes.

Probability targets have the input shape and support target and class-weight
VJPs, smoothing, mixed float32/float64 promotion and all three reductions. Their
mean uses the number of samples/spatial positions, not a class-weight sum. Native
permits unnormalized or negative target entries; the adapter does too. All8 dtype
triples across logits/target/weight were additionally audited through JVP and
reverse tapes at3 smoothing levels.

NLL exposes the weighted class-index linear objective without log-softmax. It
supports the same unbatched/spatial layouts, ignored labels and reductions.
Both loss modules register class weights as buffers with dtype conversion and
in-memory checkpoints. Class-index weights remain nondifferentiable under active
reverse recording; no-grad evaluation is allowed. Native has no class-weight JVP
for this path, so that boundary is explicit.

CE/NLL input JVPs and probability CE input/target/weight JVPs support
reverse-over-forward. The probability JVP uses the existing differentiable Tensor
composition; general first-order evaluation uses a shared raw NumPy pullback and computes
only active gradients. Finite unweighted class-index CE reuses its exponential/
partition and caches a constant analytical input derivative, with a single-edge
pullback and the native target guard. Matrix inputs avoid identity axis moves
and spatial/probability shape inference while retaining complete bounds checks. A weight-only probability tape does not retain the
original input/target Tensor objects; this is an object-lifetime result, not a
peak-memory measurement.

CE saves derived probabilities rather than the logits storage. Mutating logits
or the scalar loss does not invalidate the loss itself. Index targets/class
weights are guarded; probability guards depend on active gradients and whether
smoothing made a new target. NLL also guards its input version, matching native.
Saved-storage rebinding remains explicitly unsupported. Legacy `size_average` /
`reduce`, float16/complex kernels and zero-class probability JVP remain outside
this contract. Zero-class spatial index losses reject the invalid native layout.

## Independent validation and retained failures

- 640 new native-reference cases cover values/VJPs, retained seeds, arbitrary
  layouts, spatial/empty axes, weighting/smoothing, dtype promotion, selective
  mutation guards, nonfinite/zero denominators, JVP reverse tapes and a complete
  weighted classifier Adam checkpoint/resume workflow.
- 95 new NumPy/scalar contracts include independent log-sum-exp/weighted formulas,
  finite-difference checks for probability targets and weights, primal gradcheck,
  JVP reverse finite differences, standard imports, buffer lifecycle and training
  continuation.
- The focused new collection plus the existing browser-autograd and network-block tests passed
  **985 tests**. All8 mixed-dtype triples x3 smoothing levels passed the additional
  native full-JVP/reverse audit.
- Actual Node/Pyodide0.29.4 and Chrome155.0.8059.82 Worker: **415 contracts** each,
  NumPy2.2.5; source and test identities are checked inside the VM.
- Installed standalone wheel: Python3.10.19 / NumPy1.26.4, **435 tests**, without
  native Torch. The corrected portable fixture was rerun in this minimum
  environment. All38 Python modules and LICENSE/NOTICE match both standalone and
  private wheels/sdists and the real browser VM.

The initial native matrix retained36 failures:24 exposed float32 smoothing
promotion from a NumPy float64 scalar; the smoothing fraction now uses a Python
scalar and preserves Tensor dtype. Another12 came from native2.14.1/macOS NLL2D
backward rejecting Fortran gradient storage. Our original layout is compared to
an equivalent contiguous native oracle, and separately to independent scalar
mathematics; the native implementation failure is not reproduced as a feature.
A later3 failures established native's zero-class spatial index layout rejection.

The first real WASM run retained1 failure: a gradcheck fixture used NumPy's
default integer type, which is int32 on this WASM build. The fixture now requests
int64 explicitly. Both Node and Chrome were rerun; the initial full WASM parent
and its owned children were intentionally terminated for this concrete failure,
then the complete catalog restarted. The running native producer was allowed to
terminate before editing its test inputs. The initial full-native run retained
one obsolete test requiring smoothing rejection; that test now checks the still
unsupported legacy option. No numerical thresholds were relaxed for these fixes.

## Release gates and measurement

Final native infrastructure passed **18393 tests,1optional Docker skip,9warnings**,
239.74s. Native catalog passed **77 problems /9987 executions /3 repetitions /336 mutants**,
196.187s. Complete real WASM passed **77/77**,462.961s, no budget excess. All local
producers were terminal before the final32process benchmark. Final upstream/
frontend integration is checked separately after safe rebase. All5650 published immutable assets are
preserved;153 unreachable, untracked and unpublished initial candidate assets
were archived and pruned, retaining3 reused schedule assets.
Their elapsed times are correctness observations, not comparative speed claims.

The benchmark includes independent first-order value/VJP checks before timing.
It runs two separately labeled balanced fresh-process series after all local
correctness/build/audit producers terminate:

1. The0.18 joint implementation versus the supplied same-library Tensor
   composition across six index/spatial/smoothed/probability/NLL cases.
2. The published0.17 implementation versus0.18 on three unchanged two-dimensional,
   unweighted class-index workloads supported by both releases.

CPU and actual WASM are independent series. Source, licenses and harness files
are frozen and hashed before/after every process. All raw batches/process medians
are retained, including regressions. These are kernel measurements; no native
PyTorch, browser end-to-end, peak-memory or whole-application speedup is inferred.

## Held candidate and performance repair

The first correct unpublished implementation passed full gates but increased
common2D CE time by50-86% versus published0.17. It was held from main. The cause
was a second exponential, generic coefficient/projection arrays and repeated
shape/mask/axis work. A first cache repair and a later matrix/layout repair were
measured in separate fresh balanced process series; all rejected samples remain.
The final-source16process preflight put medium/large workloads within approx
-0.9% to+2.9% median time reduction. Tiny calls retained0.9-8.6% elapsed overhead.
Target version protection and weighted/nonfinite coefficient ordering remain.
The full final32process series below is separate from that preflight.

## Final measured source

All rows below use the final38-module implementation, NumPy2.5.4 on local CPU
and real Pyodide0.29.4/NumPy2.2.5 for WASM. Each role has2 fresh processes in
balanced order,9 batches per case,80base iterations and3warmups. Tiny cases
multiply iterations by4; vocabulary/spatial cases divide by4. Each process
validates values and active VJPs against independent float64 formulas first.
Source, notices and harness hashes remain invariant before/after every process.
All samples are retained. Positive cells mean lower elapsed time.

Published0.17 versus final0.18, identical common objectives:

| Case | CPU float32 | CPU float64 | WASM float32 | WASM float64 |
| --- | ---: | ---: | ---: | ---: |
| tiny_index | -2.7% | -2.4% | -7.7% | -2.7% |
| classifier_index | 3.9% | 3.9% | 6.7% | -1.0% |
| vocabulary_index | 7.0% | -0.1% | 5.1% | 3.7% |

Same-source joint kernel versus the supplied explicit Tensor composition
(**different comparison**, not a previous-release or native-Torch comparison):

| Case | CPU float32 | CPU float64 | WASM float32 | WASM float64 |
| --- | ---: | ---: | ---: | ---: |
| tiny_index | 63.0% | 63.1% | 60.8% | 62.0% |
| classifier_index | 49.0% | 45.4% | 41.3% | 40.4% |
| vocabulary_index | 45.5% | 42.9% | 39.0% | 48.6% |
| spatial_smooth | 15.3% | 13.0% | 15.6% | 12.2% |
| probability_trainable | 22.2% | 17.7% | 23.5% | 20.5% |
| spatial_nll | 21.2% | 22.2% | 18.6% | 20.3% |

The remaining tiny-call cost is retained; target mutation safety is not removed
to reproduce the old input/output-only guards. No inference about whole-application
latency follows from these microbenchmarks. Final browser runtime transfer size
is411157raw /94654gzip bytes, up14312raw /2726gzip from0.17 under identical gzip
settings; cold start and peak memory were not measured.

[Full evidence](../../packages/paperleet-torch/benchmarks/results/iteration-18.json)
retains final native/WASM/distribution/source proofs, all32 final process reports,
all earlier rejected samples/probes and archived unpublished-asset receipts.
`iteration-18-harness.tar.gz` contains original runners, failed fixtures, each
frozen implementation and archived pre-fastpath unpublished JSON payloads.
The predecessor0.17's exact remote proof is carried with this source iteration.

## Final upstream integration

Upstream `0e124d7f` contributes UI/curriculum and tested-main Pages gating.
No library/Python/problem/unit-test/browser-harness/immutable-data inputs changed;
only the interview curriculum JSON needed its own schema rerun(1passed). The
private application wheel/sdist were rebuilt and checked against the new catalog.
Complete frontend checks, production build and Pages policy validation passed
after rebase. Final initial JS graph:404456raw /130110gzip bytes; this includes
upstream UI changes.
[Integration evidence](../../packages/paperleet-torch/benchmarks/results/iteration-18-integration.json)
records input invariance, final distribution identities and all affected checks.

## Verified remote adoption

Commit `5321620fc58b014deaf0fa4bb57000fb9b2e1a71` passed exact-main Quality
38848025756 and the subsequent Pages38849647095 deployment. Remote native passed
18393 infrastructure cases and77 catalog modules/9987 executions/336 mutants;
actual WASM passed77/77 with415 independent contracts and no budget excess.
Live verification compared all77 starters/manifests/details and156 immutable
payloads, including both0.18 profiles and Apache notices, to frozen expected bytes.
The [remote proof](../../packages/paperleet-torch/benchmarks/results/iteration-18-remote-adoption.json)
retains job/step outcomes, downloaded reports, live hashes and observed receipts.
