# StrideTorch0.31 triangular systems

`torch.linalg.solve_triangular` adds structured dense left/right solves with
batch broadcasting, required `upper` and optional `unitriangular`. B is always a
matrix/batch, including shapes that generic solve would classify as vectors.
Only the selected triangle participates; unit diagonals ignore input values.
Float32/64 cover values, analytical reverse gradients, JVP and reverse through
both JVP channels. Complex64/128 cover inference. Out buffers, half factorization,
sparse systems and complex AD remain explicit unsupported cases.

The API follows [native triangular solve](https://docs.pytorch.org/docs/2.14/generated/torch.linalg.solve_triangular.html).
Values use column-major final axes for both solve sides, including empty and
singleton layouts. Backward solves the adjoint system once for both active
inputs, masks A's gradient and removes the unit diagonal gradient. A/output
mutation guards match native; B values are not retained. Non-leaf wrappers can
be collected while required arrays survive until backward, then are released.

Native JVP probes establish a material difference from generic solve: the outer
triangular coefficient remains differentiable when reversing the tangent.
The implementation keeps that edge and explicit zero paths. It does not reuse
generic solve's detached saved-LU rule. Small, empty, singleton and zero-batch
cases retain native values, gradients and layouts; singular diagonals propagate
nonfinite values instead of a generic singular-matrix exception.

## Numerical kernels and measured refinement

Tiny1x1/2x2 systems use direct formulas. Small unbatched systems use one local
NumPy solve. Larger/batched systems perform32-row block substitution, rectangular
NumPy matrix updates and small LAPACK solves, avoiding a full dense factorization.
The same algorithm runs on CPU and pinned WASM without SciPy or native Torch.

Initial full proofs passed30831 tests and1097 actual VM contracts, but complete
small SGD flows were3–14% slower and JVP19–24% slower than the prior projected
composition. A CPU B-only float32 process also showed a−28% anomaly, retained
without a causal attribution. Profiling showed repeated mask construction,
unnecessary B copies/output assignments and general batch parsing.

The final source returns local solves directly, bypasses broadcasting for equal
batch shapes and caches only small read-only shape masks. Sixteen masks bound
mask data to64KiB; Python object overhead is additional. Values, factors and
user tensors are never cached. A separate284-call measurement verifies the bound,
readonly boolean masks, input-array release and backward after cache churn.

## Final compatibility evidence

The refined source retains44 Python modules/two license resources.

- Complete root/independent suite:30879 passed, one optional Docker skip,
  34 warnings in192.27s. Focused native/independent triangular tests:919 passed.
- A fresh Python3.10.19/NumPy1.26.4 installation without native Torch passes1125
  tests. Wheel/sdist source and current README metadata match.
- Real Pyodide0.29.4/NumPy2.2.5 under Node and Chrome155 Worker each pass1113
  checks in31 standalone suites, with all46 final resource/source hashes equal.
- Native2.14.1 grids cover float32/64 gradients, complex inference, broadcasts,
  block boundaries, ignored NaNs, native return/gradient layouts, mutations and
  reverse through all four primal/direction inputs. Independent2x2 algebra,
  constructed large systems/adjoint seeds and real SGD/StepLR ZIP-resume flows
  provide mathematical checks without native Torch.
- Public/private wheels match all46 hashes, use disjoint installed paths and
  survive uninstalling either owner. Both namespaces execute aliases, checkpoint
  resume and triangular VJP/JVP/full tangent reverse. Private wheel/sdist also
  match the current application server/catalog.

The first native grid's80 failures were strict singleton-gradient stride
assertions. Native probes show the same1x1 parameter produces strides(4,2) after
sum/explicit-seed backward and(1,1) after square/matmul/triangular backward.
Native's [gradient layout contract](https://raw.githubusercontent.com/pytorch/pytorch/v2.14.0/torch/csrc/autograd/utils/grad_layout_contract.h)
compares non-singleton axes and permits different positive singleton strides.
The new assertion checks that actual contract and contiguity. Product allocation
and all previous gradient-layout tests remain unchanged. Four later scalar JVP
fixture failures mixed default float32 A with explicit float64 B; explicit fixture
dtypes correct them. Failed logs/probes remain archived.

## Complete-flow performance

The baseline is StrideTorch0.30 generic solve with the mathematically required
triu/tril projection and identity addition. The frozen-A batched B-only case
instead starts with a valid triangle and adds no baseline projection. These are
comparisons with the prior NumPy library composition, not native PyTorch speed
claims. Each operator retains its own native output layout.

Two fresh processes per runtime/dtype retain nine ABBA/BAAB blocks with GC enabled,
20/100 warmups and5/60/100 calls per sample depending on size. Values, float32/64
dtypes, gradients/JVP channels and actual SGD parameter states match. JVP compares
forward channels only; its retained reverse tape intentionally follows the two
different native operators. Baseline projection of primal/direction is inside the
timed work because0.30 triu/tril lack forward-mode rules. Timing processes run
sequentially after compatibility gates; shared-host and actual BLAS-thread-count
uncertainty remain explicit.

Final process-median latency reductions across float32/64:

| Complete flow | CPU | WASM |
| --- | --- | --- |
|2x2 solve/loss/backward |10–12% |8–11% |
|16x16 solve/loss/backward/real SGD |4–12% |11–12% |
|4 batches64x64, frozen A/B gradient |−0.1–8% |14–17% |
|256x256,4 RHS,loss/backward |39–46% |73% |
|256x256,64 RHS,right/unit,loss/backward |21–34% |34–35% |
|256x256,4 RHS,no-grad inference |54–55% |89–90% |

The small JVP is0.5–1.1% slower on CPU and approximately unchanged on WASM
(−0.2% to+0.8% reduction). Unchanged unary/attention controls fluctuate roughly
within±2%, providing no general speedup claim. Every initial/regressed sample and
both final processes remain available; no case is dropped to select only wins.

Sorted concatenated source/license data grows489518→497125 raw and
115361→116891 gzip (+1530 bytes). This is source footprint, not network transfer.
Four fresh CPU processes per variant/preload condition show namespace medians
79.50→80.09ms including NumPy and56.83→57.68ms with NumPy preloaded. Ordinary
namespace import still leaves zipfile unloaded. Earlier cold samples, including
a123.9ms baseline outlier, are retained separately from the refined phase.

[Core evidence](../../packages/torchwasm/benchmarks/results/iteration-31.json)
and its receipt archive preserve all source/proof phases, native contracts,
profiles, kernel selection,48 import/performance children and bounded-cache
measurements. Full application catalog/exported-WASM/frontend integration and
main/remote adoption remain pending. Version0.30 remains the last positively
verified site release.


## Complete application integration

The merged application reruns all30879 infrastructure/core tests with one optional
skip in218.65s because incoming0ac530d0 adds a preparation topic to the backend
catalog. Original native78/10218/342/three-repeat CI passes in142.34s. Actual
exported WASM passes all78/342 in207101ms at concurrency2,including1113 checks
in31 standalone suites with frozen hashes. No original browser budget is exceeded.
Catalog timings are compatibility/budget gates,not performance comparisons.

All23 upstream files through769915be are preserved byte-for-byte. The initial19
files add editor navigation,profile/achievement returns and visual-quantization
guides;later six-file overlap adds diagnostic/editor CSS and hosted-offline test
coverage. Python/library/original problem/judge inputs do not change in the later
follow-up. All7704 prior immutable assets remain unchanged;158 new reachable
payloads plus three indexes are frozen. Both runtime profiles retain version0.31
and Apache-2.0/NOTICE;the full runtime is514747 raw/119781 estimated gzip bytes,
and the schedule profile is37343/11550.

Full npm check passes60 maintained bilingual guides/180 prompts at0ac530d0.
After769915be,the new hosted-offline check and existing Python navigation,
workspace-action and source-location checks pass again with both builds. Latest
default/actual hosted Pages builds each contain three initial JS assets:
403879 raw/130426 gzip and405281 raw/130879 gzip bytes. Original size budgets and
new deferred-editor-style assertions remain intact. The condensed React review
preserves source/read-only/stale-navigation guards,native labels/buttons and
unconditional hooks;no manual visual QA is claimed.

The application wheel/sdist are rebuilt after the backend catalog change and
match all46 core/resource hashes and current server/catalog bytes. Source library
hashes remain the same as the refined core/NumPy-only/Chrome/performance proofs.
[Application evidence](../../packages/torchwasm/benchmarks/results/iteration-31-integration.json)
and its229-file archive retain original/follow-up frontend contexts,all process
receipts,exports and complete real VM results. Main adoption uses a non-force
push;remote Quality,actual Pages and live-byte checks are separate gates. Exact main source b3b56a19 subsequently passed all remote and live gates below.


## Final main synchronization

Pre-push revalidation finds newer main06a20555 and stops before pushing. Its six
files add case-insensitive library topic filtering with derived canonical option
labels,shared pure matching helpers and two relevant checks. No backend/catalog,
Python/library,original problem,dependency or lock input changes. The merge
preserves all29 upstream files and all46 library hashes/161 frozen exports.
Catalog-workbench/library-navigation checks and both complete builds pass again.
Latest default and hosted Pages builds retain three initial JS assets:
404162 raw/130579 gzip and405564 raw/131014 gzip bytes,within original budgets.

[Final synchronization evidence](../../packages/torchwasm/benchmarks/results/iteration-31-main-sync.json)
and its13-file archive preserve the guard,merge scope,static React review,
unchanged mathematical inputs and actual check/build receipts. The original full
Python/native/WASM proof remains applicable;no case,test or budget is removed.


## Exact main deployment verified

Main source `b3b56a19f403cc9fb7e1f0d9c4e109ecb66ac545` passes all four jobs in
[Quality38944777730](https://github.com/PaperLeet/paperleet.github.io/actions/runs/38944777730).
Native CI covers78 problems/10218 tests/342 mutants/three repeats in415.62s,
within the original600s limit. Infrastructure passes30879 with one optional
skip and37 warnings in418.90s. Real exported WASM passes78/342 in523003ms,
including1113 independent checks across31 suites, with no budget exceedances.

[Pages38946337720](https://github.com/PaperLeet/paperleet.github.io/actions/runs/38946337720)
actually executes and passes its deployment steps for that exact source. Live
verification decodes and compares all78 starters/details/manifests and checks
all158 immutable payloads byte-for-byte. Both profiles expose0.31.0 and the
expected Apache-2.0 license/NOTICE. All46 source resources and161 exported
files are verified against the pinned Git commit, not the newer0.32 worktree.

[Remote adoption evidence](../../packages/torchwasm/benchmarks/results/iteration-31-remote-adoption.json)
and its17-file archive retain source/export hashes, actual job/step results,
all three successful download receipts and live resource evidence. Remote runner
durations are budget gates; no cross-run performance improvement is inferred.
