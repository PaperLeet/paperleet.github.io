# Lightweight torch runtime development

The goal is an independently usable torch-compatible library with efficient
computation, broader core API coverage and validated CPython/browser behavior.
PaperLeet remains the integration workload, not the definition of all supported
behavior. Passing its current exercises alone does not establish library readiness.

## Version 0.2 milestone

- Preserve ordinary `torch` imports through the host alias or script launcher.
- Speed up basic slicing backward while retaining duplicate-index accumulation.
- Share the analytical attention pullback through a multi-input graph node, with
  no retained logits array or cross-backward gradient/seed cache.
- Preserve float32 attention outputs under NumPy 2.x instead of promoting them
  through a NumPy float64 scaling scalar.
- Fix Adam's per-parameter update counts, coupled weight decay and missing-gradient
  zeroing. Validate against independent native PyTorch, not another adapter path.
- Maintain reproducible CPython/WASM benchmarks and a separately labeled WebGPU
  feasibility probe with full upload/dispatch/readback timings and value checks.

## Version 0.3 milestone

- Registered model lifecycle: recursive train/eval/apply, shared-module and
  parameter/buffer traversal, four registered containers and Sequential slicing.
- In-memory state dictionaries with metadata, extra state, persistent buffers,
  strict key validation, native copying rules and retained optimizer references.
- Independent checkpoint copies preserve tensor storage views; copied model
  Parameters start without gradients. Native-reference and NumPy-only tests cover
  training, checkpoint restoration and continued updates under CPython and WASM.
- At this milestone, hooks, assign-based loading, model device/dtype conversion,
  file serialization and optimizer checkpoints were outside the contract.

## Version 0.4 milestone

- Common optimizer lifecycle for SGD/Adam/AdamW: parameter groups, named inputs,
  dynamic group addition, mutable learning rates, closures, in-memory snapshots
  and positional checkpoint remapping into current parameter objects.
- Lazy Adam moments and per-parameter counters, AMSGrad/maximization, native
  missing-gradient rules, reusable zeroed gradients and state mutation guards.
- Same-dtype Tensor conversion preserves identity, including during JVP, avoiding
  needless graph nodes. Raw NumPy updates remain independent learner oracles.
- Native-reference/standalone train-save-rebuild-load-resume validation and
  construction/update/zeroing/cast/memory benchmarks; unsupported execution modes
  and file serialization remain explicit boundaries.

## Version 0.5 milestone

- Recursive CPU model dtype/device conversion preserves Parameter and existing
  gradient identities, optimizer references and integer buffer rules.
- Tensor conversion overloads, memory-format/copy behavior, CPU device metadata,
  and explicit backend availability queries support unchanged CPU-selection code.
- Gradient modes support nested contexts, direct setters and recursive/generator
  decorators, with caller-mode restoration on suspension, return and exceptions.
- Linked storage-version counters retain mutation guards across model storage
  rebinding. Saved-self, saved-view and saved-output backward cases are compared
  with native Torch, including underflow and mixed-dtype errors. Old live graphs
  are operator-specific and are not an unrestricted compatibility promise.
- Allocation/hot-loop measurements check the cost of richer graph metadata and
  conversion semantics. CPU and real WASM tests include independent conversion,
  continued-training and in-memory checkpoint contracts.

## Version 0.6 milestone

- Twelve common learning-rate schedules share grouped initialization, last-rate
  inspection, optimizer-step tracking and in-memory continuation state.
- Linear warmup, cosine/step/polynomial decay, lambda/callable policies, chained
  and sequential composition and metric-driven plateau scheduling run through
  ordinary `torch.optim.lr_scheduler` imports.
- Native-reference tests cover rate trajectories, actual updates, Tensor-rate
  identity, manual rate changes, custom subclasses, dynamic groups, nested
  boundaries and save-rebuild-load-resume behavior. Independent NumPy-only
  training/checkpoint workflows are checked in real WASM.
- Scalar Tensor rates remain frozen CPU values. Cyclic/one-cycle/warm-restart
  policies, general metric-driven composition and executable function/file
  serialization are separate work.

## Version 0.7 milestone

- Twelve registered normalization, embedding, activation, shape and loss modules
  support complete attention/MLP model workflows and in-memory continuation.
- Shared analytical normalization VJPs honor active inputs; JVP and
  reverse-over-forward include affine tangents and native LayerNorm cache rules.
  Sensitive float32 rows use more accurate centering with documented differences
  from native CPU reduction order. Frozen bias-only tapes release unused inputs.
- GELU exact/tanh rules, distribution-faithful dropout with shared derivative
  masks, pretrained/padded embeddings, frequency scaling, max-norm renormalization
  and ordinary embedding JVP extend the functional and module APIs.
- Native full-gradient encoder training and independent NumPy-only stochastic
  AdamW/scheduler checkpoint workflows validate behavior beyond lesson coverage.

## Version 0.8 milestone

- Python-style missing-attribute behavior, generic model initialization,
  registered-name introspection and computed submodule access.
- Local forward pre/post callbacks support ordering, kwargs, input/output
  replacement, gradients, always-call exception cleanup and removable handles.
- Public checkpoint pre/post callbacks support metadata, format transformation,
  child traversal and shared incompatible-key editing during continuation.
- Weak handles, deep-copy behavior, lazy callback storage and an unhooked call
  path are validated with native-reference and independent NumPy/WASM workflows.

## Version 0.9 milestone

- Reverse graph metadata separates history from intermediate Tensor storage;
  saved-state release and explicit retention preserve non-leaf identity.
- Multiple roots/seeds, requested input pruning, named/duplicate/unused inputs,
  retained non-leaf gradients and selected backward inputs follow native routing.
- Operator-specific retention audits distinguish saved Tensor state from
  repeatable metadata-only VJPs, including pooling/resize specializations.
- Scalar integer views retain shared storage/version guards; reduction scalar,
  empty-axis and dtype handling and constant-spacing trapezoid improve core APIs.
- Independent CPU/WASM contracts and retained-loss memory/control benchmarks
  validate values, release behavior and performance costs.

## Version 0.10 milestone

- Tensor gradient callbacks support replacement/observation, phase snapshots,
  requested-input pruning, retained gradients, weak handles and no-grad scopes.
- Leaf post-accumulation callbacks support inline updates and gradient clearing;
  explicitly returned hook Tensor graphs survive at requested gradient targets.
- Callback metadata shares the optional non-leaf retention slot. Plain graphs
  allocate no hook dictionaries or additional graph-node slots, and backward
  skips derivative records used only by autograd.grad.
- CPU/no-grad copy_ supports broadcasting, casts, strided views, alias/version
  guards and overlap errors, with typed boundaries for differentiable mutation.
- Native and independent NumPy/WASM training workflows validate transformed
  gradients, inline SGD, reverse-over-forward and retained callback lifetimes.

## Version 0.11 milestone

The 0.11 milestone validates external NumPy inputs before copying/casting,
protects the numerical storage invariant, handles read-only warnings and reduces
redundant builtin scalar construction. Native checks distinguish default int64
construction from weak scalar operands; independent CPU/WASM workflows cover
ownership, independently mutable NumPy metadata, gradients and error-mode
restoration. Balanced CPU/WASM/Chrome measurements preserve both improvements
and import/checking costs; 71-module integration, standalone wheel and browser
contracts pass. Apache-2.0 resources remain part of every distribution profile.

## Version 0.12 milestone

Recursive sequence inference preserves NumPy scalar widths and precision, using
Torch category/width promotion and native scalar storage protocols. Public
promote_types and ordinary dtype aliases expose the same supported rules.
Primitive vectors/rows use batch checks, while small builtin sequences and NumPy
scalars avoid extra allocations. Native-reference matrices and standalone CPU/
WASM workflows cover precision, copies, gradients, bounds and invalid inputs.
Balanced CPU/WASM/Chrome performance, independent distribution and 71-module
integration gates pass; large-sequence checking costs remain explicit targets.

## Version 0.13 milestone

Mixed-dtype `where`, `cat` and `stack` use native Tensor/Number category priority
and all-input promotion; complex joins preserve imaginary values. Public
`result_type`, Tensor.where, concat/concatenate, keyword/axis aliases and
one-argument where indices improve ordinary torch code compatibility. Scalar
protocol/overflow handling is validated separately from constructor inference.
Where pullbacks retain condition data and shape metadata without branch Tensor
closures; condition-only version guards, release/retention and VJP/JVP workflows
are checked independently of course coverage. Fixed CPU fixture memory savings
and common-path latency costs are recorded alongside balanced CPU/WASM/Chrome
controls. ComplexHalf, real output buffers and remaining operator-wide promotion
are explicit boundaries.

## Version 0.14 milestone

Native category/width arithmetic promotion, checked scalar/alpha protocols,
functional aliases, floor/trunc division and reciprocal improve unchanged torch
code coverage. Reverse kernels preserve native numerical operation order;
undefined constant forward-tangent terms are omitted without losing explicit
zero tangents. A traversal-wide numerical scope keeps callbacks' mutable NumPy
error policy intact. Independent IEEE, dtype, lifecycle and CPU/WASM/Chrome
measurements retain their costs as well as gains. Native and full WASM quality
gates, main deployment, live assets and Python 3.10 isolation passed.

## Version 0.15 milestone

Full Flow Matching profiling identified linear forward contractions as the main
remaining WASM target. A bounded large/contiguous-feature path improves local
complete-suite median time by 27.0% in Node/WASM and 21.1% in Chrome Workers in
separate balanced fresh-process series. CPU BLAS and small/strided paths remain.
All eight timed full suites passed with recorded floating-point differences;
this is not a remote deadline guarantee. Large scalar-sum/native VJP/JVP and
saved-storage tests, independent wheel/source identity, and actual Python
3.10/NumPy 1.26 isolation supplement course coverage. All local infrastructure,
native catalog, full WASM catalog, visualization and frontend/build gates pass;
exact remote adoption checks follow publication. See the iteration 15 record.

## Version 0.16 milestone

BatchNorm1d/2d/3d and functional batch normalization add stateful channel
normalization, affine gradients, train/eval and tracking modes, cumulative
statistics, checkpoint/counter compatibility and standard module aliases.
Native VJP/JVP conventions, selective storage guards and complete Adam
train/checkpoint/resume workflows are covered alongside independent NumPy and
finite-difference contracts. Ill-conditioned float32 channels use a documented
centered numerical policy. Complete local native/WASM/application, packaging and
frontend gates pass. Balanced CPU/WASM costs retain the float64 eval-backward
overhead as well as training gains; exact remote adoption subsequently passed for the published0.16commit.

## Version 0.17 milestone

GroupNorm adds independent per-sample/channel-group normalization, shared active
VJPs, native full-statistics JVP reverse tapes, standard imports, selective
mutation guards and common model/checkpoint lifecycle. Native and independent
float64-math/finite-difference oracles cover strided inputs, empty batches,
float32 cancellation and training continuation. Full native/application/WASM,
Chrome Worker and isolated minimum-version wheel gates pass. Balanced same-source
CPU/WASM measurements preserve all samples; the larger runtime transfer size is
reported separately. See [validation](../../docs/validation/torch-runtime-iteration-17.md).

## Version 0.18 milestone

Weighted/smoothed/spatial cross entropy, probability targets and NLL add native
VJP/JVP and full checkpoint-resume coverage. Finite default CE caches its
analytical derivative and avoids generic spatial work. Rejected regressions and
final CPU/WASM measurements are retained. Exact main Quality, Pages and live
assets passed; see the iteration 18 remote adoption receipt.

## Version 0.19 milestone

Grouped 1D/2D/3D convolution adds batched/unbatched layouts, dilation, padding
modes, analytical active VJPs, bilinear JVP reverse tapes and module lifecycle.
Direct/packed-matrix/auto paths are tested against native and scalar mathematics.
A 32 MiB logical-workspace estimate bounds matrix dispatch and backward repacks
rather than retaining expanded activations. Actual WASM/Chrome and independent
CNN training supplement catalog coverage. Same-source CPU/WASM timing records
both spatial gains and depthwise regression. See the iteration 19 validation.

## Version 0.20 milestone

True storage-sharing view/view_as semantics separate from reshape, physical
stride/element-size queries and empty factory metadata improve ordinary Torch
code compatibility. Wrapper ownership preserves version guards and checkpoint
aliases. Native/scalar contracts and actual WASM/Chrome/minimum-version wheels
cover reverse and forward tapes. Four frozen CPU/WASM comparisons retain the
initial rejected regressions and the remaining validated view-checking costs.
Complete local native/catalog/WASM/distribution/frontend gates pass; exact remote
adoption follows publication. Evidence is tracked in
[iteration 20](../../docs/validation/torch-runtime-iteration-20.md).

## Version 0.21

Dimension validation and native shape/view identity rules cover functional and
method entries, scalar/empty cases, singleton strides, VJP and JVP. Graphless
views preserve base-dependent gradient flags without inventing gradient edges;
explicit `requires_grad_()` creates a leaf accumulator. Checked `copy_` retains
nonoverlapping singleton expansions. Source snapshots, rejected failures and two
initial CPU/WASM benchmark phases are retained alongside newer improvements.
A dedicated one-input metadata result path and validated expansion descriptors
remove the initial view/expansion latency regression. Direct gradient-flag
assignment follows native leaf rules; ordinary reverse microbenchmarks retain
small costs. Version 0.21 is adopted on main at `9eee6fa0` with all 78 exported WASM modules
within budget. Exact Quality passes; the runtime-identical curriculum descendant
`abbf721b` passes Quality/eligible Pages and all 78 live module/158 immutable
asset comparisons. See
[iteration 21](../../docs/validation/torch-runtime-iteration-21.md).

## Version 0.22 milestone

Shared real statistics validate dimension/type/range/duplicate rules, scalar and
empty reductions, fractional correction and unbiased overloads. Native values
and VJP/JVP cover insufficient degrees of freedom and the distinct empty-dimension
reverse convention. New `var_mean/std_mean` share moments; selective saved tensors
retain input and standard-deviation guards without guarding unsaved means or
variance outputs. Reverse-over-JVP and post-forward model dtype conversion are
checked. Float32 paired accumulators retain their native rounding distinction.
Independent directional derivatives, duality and continued optimizer updates
supplement native references. Local infrastructure passes 25,605 tests; all 78
native and exported WASM problems pass within their existing budgets. Independent
Node/Chrome contracts, minimum-dependency installed wheels and balanced CPU/WASM
comparisons pass. Remaining standard-deviation reverse costs are documented.
Main at `7eac78ab` passes exact remote Quality, including all 78 WASM modules.
Runtime-identical curriculum descendant `db6408fd` passes exact Quality and
eligible Pages publication; all 78 live modules/158 immutable comparisons pass.

## Version 0.23 milestone

Public operators retain at most 32 successful signature patterns per operator,
using positional arity and ordinary keyword names. Argument values, tensor
objects, gradients and execution modes are never cached. Unsupported options,
forward/reverse policies and numerical validation still run on every call;
uncached or excess patterns retain ordinary binding and error behavior. Keyword
name subclasses cannot retain attached user payloads through the cache.

Final infrastructure passes 25,612 tests; real Node/Pyodide and Chrome Worker
each pass 774 independent contracts. The installed wheel passes 794 tests on
Python 3.10/NumPy 1.26 without native Torch. Balanced CPU/WASM comparisons retain
all three phases, including an unstable middle phase and direct-method controls.
Original native catalog, export/history, latest-curriculum packaging/frontend and
all 78 exported WASM modules pass. Main `3cea3727` passes exact Quality, including
25,613 merged infrastructure tests and 78 WASM modules. Runtime-identical
`1e2a8a66` also passes Quality; both Pages triggers are superseded by later UI
work, so site publication is not claimed. A mature later version may supersede
this site rollout, with final eligible deployment/live verification still required.

## Version 0.24

Method/functional mean support explicit dtype with strict native reduction
overloads, integer/boolean floating output, existing complex forward values,
float32/float64 JVP and reverse differentiation of its tangent. Half sum/division
uses float32 opmath without explicitly copying the entire input; reverse preserves
output rounding and native post-forward model-conversion gradient dtypes.

Final infrastructure passes 25,928 tests; Node/Chrome each pass 784 independent
contracts and the NumPy 1.26/Python 3.10 installed wheel passes 804 without native
Torch. Four balanced CPU/WASM phases retain all 64 children, rejected input-copy
work and remaining small-call costs. Original native catalog, export/history and
latest-upstream frontend/packaging gates pass. All 78 exported WASM modules pass
within unchanged budgets. Later upstream UI changes at `1be5db7e` are preserved
and repeated frontend checks/build pass. Main `002b693e` passes exact remote
Quality: 25,928 infrastructure tests, all 78 native problems / 10,218 executions /
342 mutants / three repeats, and 78 WASM modules within unchanged budgets.
Pages eligibility was superseded by later UI work; deployment/live verification
is not claimed. Actual reports and terminal observer are retained in the 0.24
main Quality archive.

## Version 0.25 candidate

Real nansum/nanmean cover dtype, strict dimensions, VJP and float32/64 JVP with
reverse over the tangent. IEEE mask rules, half composite overflow, post-forward
model conversion and selective graph pruning match native references. All 2,982
supported cases in a 3,024-case audit agree; 42 empty real-to-complex cases retain
the explicit complex boundary. Node/Chrome each pass 806 independent contracts;
the minimum Python 3.10/NumPy 1.26 wheel passes 818 without native Torch. Fresh
public/private wheel coinstallation and uninstall ownership pass. Final merged
infrastructure passes 26,466 tests with one optional Docker skip. All 16 balanced
CPU/WASM timing children complete: small/casted nanmean improves 51–56% against
the matching where/sum composition, with ordinary controls and all regressions
retained. Original native catalog passes 78 / 10,218 executions / 342 mutants /
three repeats, and complete exported WASM passes 78/78 without budget violations.
The maintained exported-VM harness now discovers all independent NumPy-only
suites (25 / 806), excluding only host launcher/install tests. All 6,756 published
immutable assets stay unchanged; 158 reachable new assets are frozen. Latest
upstream's 34 UI/curriculum files remain exact; frontend and actual hosted Pages
profile builds pass. Main `22a0f289` passes exact Quality; runtime-identical
`90ac44ef` passes all four required jobs and actual Pages run38900518818. Live
starters/manifests/details, 158 immutable payloads, runtime25 and notices match.

## Version 0.26 milestone

Functional gather, strict indexing/sorting schemas, half scatter-add opmath and
gather/sort/topk JVP cover the next core gaps. Metadata/selected-index pullbacks
match native version guards, avoid unnecessary intermediate value retention,
and retain named JVP results. Cached result types, partial topk and validated
direct indexing improve measured CPU/WASM work. Four balanced phases retain all
64 children, including the initial gather regressions and remaining CPU float32
reverse-gather cost. Final infrastructure passes 26,803; Node/Chrome each821;
minimum NumPy-only wheel833 and both wheel ownership/uninstall gates pass.
Original native catalog passes78 /10,218 executions /342 mutants /three repeats;
export preserves all6,914 published immutable files and adds158 reachable assets.
All31 latest upstream UI/catalog/parallel-CI files remain exact, and frontend,
actual hosted profile and application packaging checks pass. Complete exported
two-process WASM passes78/78 without budget violations and includes all26/821
independent contracts. Main adopted11434fb8 by non-force push. Its actual remote
Quality38903517940 passed all four required jobs:26803 base tests,78 original
modules/342 mutants, and821 independent contracts in exported WASM. Pages built
successfully but its deploy action was skipped after main advanced. Runtime and
all161 frozen export hashes are identical on descendant3c85b178; its actual
Quality38904656048 and actual Pages38906085493 both passed. Frozen live checks
verify78 decoded starters/manifests/details and all158 immutable payload bytes,
including both runtime profiles and their licenses. Version0.26 is positively
verified on the site through that runtime-equivalent descendant; the original
source commit's skipped deploy remains separately recorded.

## Version 0.27 milestone

- First dense non-overlapping leaf gradients preserve parameter layout, with
  ordinary C/F fast paths and bounded general dense allocation. Gapped and
  overlapping leaves use C order. Scalars/singletons/empties are covered.
- Stored gradients own separate bytes from parameters, user seeds and hook
  returns. Existing leaf gradients keep their objects/layouts/versions semantics;
  retained non-leaf gradients start contiguous and subsequent updates replace
  their object, preserving earlier snapshots.
- Independent strided SGD and ownership tests run on minimum Python/NumPy and
  real Node/Chrome WASM. Complete base27059, minimum853 and WASM841 pass. Original
  native catalog78/10218/342/three repeats passes unchanged budgets.
- Dtype-asserted ABBA timings retain both wins and costs: full F-layout backward
  improves in these samples; external contiguous seed conversion to F/permuted
  layout costs more than the former incorrect C copy. Initial generic-layout
  and incorrectly labeled float32 first-copy trials remain in the audit trail.
  Latest-main f445118a UI/docs/dependency changes are preserved; owned npm
  dependencies isolate the runtime from external node_modules changes.
- All7072 published immutable files stay unchanged;158 reachable exports are
  added. Full exported WASM78/342 and27/841 independent contracts pass unchanged
  budgets. Latest application distributions, complete frontend/security checks
  and both actual build profiles pass. Main adoption uses non-force revalidation;
  exact remote Quality and eligible Pages/live proof remain separate gates.

Main adoptedf1b33b0b without force. Actual sourceQuality38908465541 passed all four
jobs. Runtime/export-identical descendant9f18e510 then passedQuality38908787448 and
actualPages38910349850. Frozen live verification succeeds for78 decoded starters,
current detail/manifest references and158 immutable payloads, including both
runtime profiles and licenses. Version0.27 is positively verified on the site;
source-commit deployment was superseded and is not claimed as published.

## Version 0.28 milestone

- Matrix `t/mT`, movement `movedim/moveaxis` and swap `swapaxes/swapdims` preserve
  ordinary views, storage versions and real VJP/JVP routing. Scalar aliases,
  keyword names, native scalar/sequence parsing and error priority are tested.
- Existing transpose rejects nonnative scalar axis forms and uses an inline
  path for ordinary valid Python integer axes. Inverse permutations use a
  bounded32-entry cache of integer tuples, with no numerical or graph state.
- Native and independent NumPy/CPU/WASM contracts cover complex inference,
  strided/singleton/empty views, multiple batch axes, dynamic seeds, argument
  ownership and retained graph/value release. Eager and per-node metadata trials
  were rejected using timing evidence; all trials and anomalies are preserved.
  Full final-source and application integration outcomes are recorded separately.
- Final core gate29402/one optional skip, minimum wheel893 and Node/Chrome881
  pass. Latest-main integration preserves33 upstream files, passes107 affected
  Python tests and all78/342 exported WASM cases with28/881 standalone contracts.
  Frontend/default/actual Pages-profile builds pass. Exact-main0fb0bdc5 then
  passesQuality38919796621 and actualPages38921426204;78 decoded starters and
  158 immutable live payloads match the frozen release. Version0.28 is positively
  verified on the site; remote receipts are archived separately.

## Version0.29 adopted

- Batched/broadcast left and right dense solve, native vector overload/strides,
  shared single-adjoint VJP and real JVP with native saved-factor reverse rules.
- Independent closed-form algebra, continued training/checkpoint and CPU/real
  WASM tests complement native value/gradient/layout/error/retention contracts.
- Initial/selective regressions are retained. Refined ndarray metadata, ordinary
  solve pullbacks and single/shared-node construction improve broader CPU/WASM
  controls and actual SGD flows, with some tiny solve-only regressions remaining.
  Refined core gates29853/one skip,minimal wheel916 and Node/Chrome904 pass;
  public/private wheels coinstall and survive independent uninstall. Full
  application adoption and actual live verification now pass for0.29.
- Local application gates now pass original78/10218/342/three-repeat CI and
  exported78/342/29suites904 WASM with unchanged budgets. All23 upstream UI/docs
  files and7388 immutable assets are preserved;latest targeted inbox/default/
  actual Pages-profile builds pass. Non-force main0d5975a0 passed all four
  Quality jobs;byte-identical descendant033dc1bc completed actual Pages deployment
  and78-module/158-payload live verification. An earlier overall-success Pages
  run skipped its deploy action and is retained as unpublished evidence.

## Version0.30 adopted

- Modern ZIP CPU `torch.save/load` for model, optimizer and scheduler data;
  native-format storage sharing/strides and14 dense dtypes, with independent
  native fixtures and bidirectional continued-training checks.
- Fixed data-global loading rejects arbitrary globals and cached pickle
  extensions;unknown modes/objects fail explicitly. No native Torch dependency.
- Final hardened core29960/one optional skip,NumPy-only wheel960,and actual
  Node/Chrome948 in30 suites pass;installed public/private ownership and native
  bidirectional resumed training pass. Source gzip grows5074 bytes,cold CPU
  namespace import about2ms;compute controls remain within about±2%. Small-tensor
  save metadata remains slower than native Torch.
- [Core evidence and measured scope](../../docs/validation/torch-runtime-iteration-30.md)
  and complete application evidence now pass original78/10218/342/three-repeat
  native CI and exported78/342/30suites948 WASM,with unchanged budgets.
- All15 upstream UI/docs/style/check files through9d35 and7546 existing immutable
  assets remain unchanged;latest profile-settings and both frontend builds pass.
  Non-force mainb1a1631a passes all four Quality jobs38935475320 and actual
  Pages38937107666;78 decoded modules/158 immutable assets and both runtime
  licenses/version0.30 match pinned source in live requests.
  [Exact remote evidence](../../docs/validation/torch-runtime-iteration-30.md)
  separates budget checks from comparative benchmarks.

## Version0.31 local integration verified

- Dense triangular left/right/broadcast/unit solves with native matrix-only B
  interpretation,F final axes,A/output guards and analytical real VJP/JVP.
  Reverse of JVP retains its native outer coefficient edge;complex inference
  and empty/singleton/zero-batch boundaries are covered.
-32-row block substitution accelerates larger CPU/WASM systems. Direct tiny
  formulas,local small solves,equal-batch metadata and bounded read-only masks
  reduce small-flow overhead without caching tensor values or factorizations.
- Final core30879/one optional skip,NumPy-only wheel1125 and Node/Chrome1113 in
  31 suites pass. Public/private ownership,independent uninstall and actual
  training/checkpoint resume pass. Earlier regressions/failed probes are retained.
- [Measured complete flows](../../docs/validation/torch-runtime-iteration-31.md)
  improve selected large/small workloads;CPU JVP remains about1% slower and
  controls are within about±2%. Source gzip grows1530 bytes. Complete application
  original application78/10218/342/three-repeat native CI and exported78/342/
  31suites1113 WASM now pass. Merged infrastructure30879,full frontend60 guides/
  180 prompts,and latest offline/navigation/workspace/source checks/builds pass.
- All23 incoming files through769915be and7704 prior immutable assets remain
  unchanged;backend catalog changes trigger a full infrastructure rerun and a
  rebuilt current application wheel. Non-force main adoption and actual remote
  deployment/live remain separately tracked.
- Final main06a20555 adds only topic-filter UI/helpers/checks/docs. Pre-push
  revalidation stops safely,then all29 incoming files and46 source/161 export
  hashes are preserved;targeted library checks and both builds pass again.

## Version0.31 exact main deployment verified

- Source b3b56a19 passes all four Quality38944777730 jobs; native78/10218/342/
  three repeats415.62s, infrastructure30879/one optional skip and exported
  WASM78/342/1113 independent checks pass within original budgets.
- Pages38946337720 actually deploys that source. Live78 decoded starters,
  details/manifests and158 immutable assets match; both profiles expose0.31.0
  with licenses. Pinned46 resources/161 exports and all receipts are retained in
  [remote evidence](benchmarks/results/iteration-31-remote-adoption.json).

## Version0.32 core and complete local integration validated

- Cholesky selected lower/upper factors,batched/strided/empty layouts,real
  float32/64 VJP/JVP and reverse through primal/direction;complex inference.
  Output-only saves match native A-mutation/output-guard behavior.
- Analytical tiny pullbacks,direct small factor solves and blocked large solves
  remove initial small-flow regressions. Selected tiny/large complete flows
  improve versus the first generic-LU0.32 implementation;upper float64 WASM
  gains little.0.31 has no Cholesky and is not a throughput baseline.
- Core31042/one optional skip,focused163,NumPy-only wheel1140 and actual
  Node/Chrome1128 checks in32 suites pass. Public/private ownership,aliases,
  checkpoint resume and independent uninstall pass. Source gzip grows1791bytes.
- [Measurements and adoption boundary](../../docs/validation/torch-runtime-iteration-32.md)
  retain all initial regressions/failed probes. Original native78/10218/342/
  three repeats146.10s and actual exported WASM78/342/1128 checks243138ms
  pass within original budgets. Full frontend60 guides/180 prompts and both
  builds pass;all7862 previous immutable assets remain unchanged and158 new
  assets are reachable. No frontend/backend/dependency/lock edits accompany
  this release. Non-force adoption and exact remote deployment are tracked
  separately from these complete local gates.

## Version0.32 remote failure;local-first iteration

- Quality38948582658 native/exported WASM jobs pass,but infrastructure fails4
  NaN exception cases;Pages38949910058 build/deploy are skipped.0.32 was not
  deployed;last verified live remains0.31. Exact failed evidence is retained.
-0.33 adds full-factor Cholesky solve AD and portable failure reporting;expanded
  browser tests catch silent invalid factors. Complete local validation and
  measured efficiency remain required. Ordinary pushes/PRs now start no CI; explicit user releases run all gates
  under the canonical release policy, preserving Actions quota.

## Version 0.33: full-factor Cholesky solve and quiet-WASM status

- Adds real Cholesky-solve VJP/JVP/full tangent reverse and complex inference;
  reuses graph-local triangular intermediates and associates narrow products.
- Quiet NumPy Cholesky/solve/inverse failures now produce explicit diagnostics.
  Finite native comparisons remain strict; nonfinite backend assumptions are
  documented and independently tested.
- Full merged core 31,655 passes; Chrome/Node 1,375 checks each;
  minimum Python/NumPy wheel 1,387 passes. Complete original native/WASM catalogs,
  current frontend/builds and immutable assets all pass unchanged gates.
- Initial regressions and final paired measurements are retained in the
  [validation record](../../docs/validation/torch-runtime-iteration-33.md).
  Main adoption is separate from online publication; no CI/tag/dispatch is
  triggered. The user will explicitly tag a release later.

## Version 0.34: fused narrow Cholesky JVP

- One analytical multi-input tangent pullback retains full native primal/direction
  reverses, broadcasting and mutation behavior. Conditional direction snapshots
  correct 12 overly restrictive 0.33 probe cases; unneeded X/T arrays release.
- Full core 32,043 passes; Node/Chrome 1,439 each; minimum installed
  wheel 1,451. Original native/exported WASM catalogs and application gates pass.
- [Measured paired results](../../docs/validation/torch-runtime-iteration-34.md)
  compare both 0.33 fused and 0.32 public composition workflows, including a full
  reverse-through-JVP workload. No uniform speedup or GPU support is claimed.
- No automatic CI, tag or online publication. Naming remains pending user choice.

## Version 0.35: reuse narrow Cholesky JVP intermediates

- Actual finite triangular factors reuse primal/tangent/adjoint solve arrays
  while retaining all four full-factor reverse channels and mutation semantics.
  Validated-primal dispatch and measured Emscripten/RHS scan policy address
  prototype regressions for small and frozen-factor workflows.
- Full core 32,235 passes; Node/Chrome 1,495 each; minimum installed wheel 1,507.
  Original native/exported WASM catalogs and full application gates pass.
- [Paired measurements and retained pilots](../../docs/validation/torch-runtime-iteration-35.md)
  include the older 0.32 composition baseline and frozen/no-reverse workflows.
  Main adoption is separate from online publication; no CI/tag/dispatch occurs.

## Version 0.36: inverse from an existing Cholesky factor

- Adds torch.cholesky_inverse and Tensor method, real full-factor VJP/native
  ordered JVP/both-channel reverse, complex inference with real diagonals,
  singular diagnostics, empty strides and native saved-value mutation behavior.
- Scalar/small LAPACK/CPU Gram/large WASM solve paths improve selected complete
  workflows; all initial regressions and remaining inference costs are retained
  in the [validation record](../../docs/validation/torch-runtime-iteration-36.md).
- Full core 32,463 passes; Node/Chrome 1,532 each; minimum wheel 1,544.
  Original native/exported WASM catalogs and full application gates pass.
  A bounded actual-state wait repairs an upstream mounted navigation check.
- Quiet main adoption is separate from publication. No CI, tag, dispatch or
  online release; browser-oriented naming remains pending user choice.

## Version 0.37: Python-compatible optional feature discovery

- Missing public namespace/Tensor/optimizer attributes follow AttributeError,
  hasattr and getattr-default rules while retaining typed unsupported catches.
  Standard import failures preserve actual browser fallback; real operation and
  descriptor errors remain visible. Both browser profiles share the behavior.
- The minimal Schedules export explicitly carries shared errors and starts
  independently of the full API. Dynamic attributes, copy/pickle, object
  lifetimes and both installed namespaces are validated across tested runtimes.
- Full core 32,549 passes; Node/Chrome 1,581 each; minimum wheel 1,594.
  Complete native/exported WASM catalogs and frontend/builds pass. No uniform
  numerical speedup is claimed; all controls and outlier follow-up are retained
  in the [validation record](../../docs/validation/torch-runtime-iteration-37.md).
- Quiet main adoption is separate from publication. No CI/tag/dispatch/online
  release. Actual missing pooling/transpose-convolution APIs remain future work.

## Version 0.38: native-compatible 2D max pooling and CNN paths

- Add functional/root max_pool2d, with-indices and MaxPool2d with native
  geometry, int64 indices, finite/NaN/zero selection, noncontiguous inputs,
  empty batches, VJP/JVP, saved-storage guards, imports and Module lifecycle.
- Use measured 2x2 blocks, bounded packing and scan fallback. Complete CNN
  inference/SGD checks compare values, gradients and updated parameters;
  existing numerical workflows have a separate 0.37 regression comparison.
- Full suite 33,167 passes; actual Node/Chrome 1,730 each;
  minimum wheel 1,743. Native/exported catalogs and frontend/builds pass.
  Raw controls, compatibility limits and measurements are in the
  [validation record](../../docs/validation/torch-runtime-iteration-38.md).
- Quiet main adoption remains separate from publication. No CI/tag/dispatch
  or online release. Broader pooling/convolution/normalization remains active.

## 0.39: local average pooling and ordered adjoints

- Add AvgPool2d/AdaptiveAvgPool2d, fixed average-pooling geometry and linear
  VJP/JVP rules; share the adaptive rule with area interpolation.
- Preserve local NaN/Inf behavior, sum-before-divide overflow and output-ordered
  gradient cancellation. Replace the failed separable matrix adjoint with
  bounded inverse-region gathering; retain scanning/scatter fallbacks.
- Full Python suite: 34,887 passes; actual Node/Chrome: 2,154 each;
  isolated wheel: 2,167. Native and actual exported WASM catalogs pass all
  78 problems; source and payload identities are preserved.
- Preserve measured regressions: irregular/upsampled adaptive and area calls
  can take roughly 2–4.4 times the old implementation's time. Next, optimize
  ordered packing and cached geometry without weakening cancellation or IEEE
  behavior. New fixed-pooling CNN training is 20–29% faster than this version's
  generic scan on the measured cases; this is a different comparison.
- A separate one-dimensional pooling prototype has native reference coverage;
  it requires independent contracts, runtime checks and complete-flow timing
  before integration. See the [validation record](../../docs/validation/torch-runtime-iteration-39.md).

## 0.40: cached ordered pooling geometry

- Cache readonly packed indices, masks and typed divisors with explicit plan
  count/byte limits. Use flat spatial gathering for contiguous inputs while
  retaining coordinate gathering and bounded fallbacks for other cases.
- Preserve 0.39 cancellation, IEEE classes, zero signs, native division order,
  live NumPy aliases and storage-version semantics. Never cache user arrays.
- Full Python suite: 34,976 passes; actual Node/Chrome: 2,243 each;
  isolated wheel: 2,256. Native and actual exported WASM catalogs pass all
  78 problems with the existing budgets.
- Final paired measurements against 0.39 improve irregular forward/backward
  flows by about 19–36% / 13–27%, and upsampling forward/backward by about
  38–54% / 26–39%. Keep per-case regressions and sampling diagnostics visible;
  these are not uniform speedup or cross-runtime throughput claims.
- Warm tracked allocation samples improve while the huge-adjoint bound is
  retained; the separately bounded plan metadata is a memory/latency tradeoff.
  See the [validation record](../../docs/validation/torch-runtime-iteration-40.md).

## 0.41: one-dimensional average pooling and direct graph nodes

- Add fixed/adaptive 1D average pooling, Modules and functional/root aliases for
  CL/NCL inputs. Reuse ordered kernels without extra Tensor view graph nodes.
- Preserve native constructor normalization, stride/padding/ceil behavior,
  dtype boundaries, empty length/channel cases, saved versions and linear JVP.
- Correct conventional strides for empty 1D/2D pooling outputs. Native layout
  probes extend the independent oracle rather than relying only on passing values.
- Validate region sums/transpose scatter, native values/VJP/JVP, full Conv1d
  training and checkpoint continuation, independent packaging and actual WASM.
- Measure 34 complete flows against the supported 0.40 composition in four fresh
  CPU/Node/Chrome processes each, retaining same-candidate composition controls,
  legacy workflows, import costs, every sample and regressions. See the
  [0.41 validation record](../../docs/validation/torch-runtime-iteration-41.md).
- Full local catalog/frontend and export-identity gates precede main adoption.
  GitHub Actions, tag releases, package upload and deployment remain user-controlled.

## 0.42: direct 1D max pooling, pair blocks and reachable scans

- Add CL/NCL max pooling, root/functional/Module aliases, indices, stride,
  padding, dilation and ceil mode with native dtype and linear-AD boundaries.
- Keep 1D/2D Modules as sibling types; preserve type checks, pickle, checkpoint
  state and native input/index/tangent version rules. Normalize empty outputs.
- Share a 1x2 block kernel with existing 2D pooling. Bound ordered scan loops
  to globally reachable offsets, retaining tie/NaN order under huge padding.
- Validate independent regions/scatter, native selection/metadata/parameters,
  four-route seeded 1D/2D geometry, million-tap examples and checkpointed Conv1d
  training across CPU, minimum packaging and actual Node/Chrome WASM.
- Measure 52 complete flows with candidate-composition controls plus a separate
  forced-scan study, legacy flows and imports; retain every sample/regression.
  See the [0.42 validation record](../../docs/validation/torch-runtime-iteration-42.md).
- Full original catalog, latest frontend, typed builds and immutable export
  identity gates precede main adoption; CI/tag/publishing remain user-controlled.

## 0.43: volumetric max pooling and native layout routes

- Add MaxPool3d and functional/root value/index aliases, CDHW/NCDHW geometry,
  integer/half inference, float32/64 VJP/JVP/reverse tangent and empty batches.
- Preserve ordered selection and flattened index storage, NDHWC value/index
  strides and native ambiguous-singleton tangent strides. Keep public MaxPool
  classes as siblings and share the existing graph/scatter machinery.
- Add cube/slab block kernels, bounded packing and reachable-tap 3D scans.
  Retain explicit resource/empty-sampling/AD dtype boundaries.
- Validate 37,450 CPU tests, 3,822 tests in Node and Chrome WASM each, 3,835
  minimum-environment wheel tests, 944 native-reference trials, original 78/342
  catalog flows and complete 62-flow performance cases in 12 fresh children.
- Preserve all 9,755 prior immutable payloads and all release rules. See
  [the 0.43 validation record](../../docs/validation/torch-runtime-iteration-43.md).
  The measured overlap-window regression motivates a separable numerical
  reduction path next, preserving native NaN/index/layout and derivative rules.
  Independent-library, broader core API and GPU/backend work remains active.

## 0.44: reuse spatial maxima across overlapping depth windows

- Reuse raw HW reductions and carry selected indices through an ordered depth
  scan, without adding Tensor view nodes or a depth-index gather intermediate.
- Select by nominal repeated-work reduction and reserve live intermediate and
  coordinate/mask/index space before packing within the remaining budget.
  Retain cube/slab paths, empty-batch allocation behavior and sampling boundaries.
- Validate 37,605 CPU tests, 3,977 Node/Chrome tests each, 3,990 minimum-wheel
  tests, 1,180 native trials, original 78/342 catalog checks and 76 complete
  performance flows in twelve fresh processes. Preserve 9,913 prior payloads.
- See [the 0.44 validation record](../../docs/validation/torch-runtime-iteration-44.md)
  for measured benefits, regressions and controls. Broader APIs, bounded memory
  work and GPU/backend evaluation remain active toward an independent library.

## 0.45: volumetric average pooling

- Fixed/adaptive 3D module and functional APIs cover float32/float64 VJP, JVP,
  reverse-over-forward, fixed int64 and adaptive half inference, native parameter
  and empty/global boundaries, aliases, pickle and complete Conv3d training.
- Ordered 3D sums, disjoint block kernels, bounded packing/streaming and adjoints;
  readonly geometry/gather plans retain no user values, and strided gradient seeds
  are read by block. Global mean canonicalizes native 3D output layout.
- Native differential, minimum-wheel, Node/actual Chrome, exported 78-problem
  catalog, frontend and four-process/runtime whole-flow evidence are retained.
  Overlapping windows and some adaptive adjoints remain optimization targets.
- See [the 0.45 validation record](../../docs/validation/torch-runtime-iteration-45.md).

## 0.46: ordered inverse 3D pooling adjoints

- Scale seeds once; gather inverse regions in native output order with bounded
  readonly caches, cost fallback and workspace admission before allocation.
- Avoid discarded gradient buffers and zero filling exact disjoint blocks. Keep
  cancellation, nonfinite locality, strided seeds, JVP and reverse-tangent rules.
- Complete CPU, installed-wheel, Node/Chrome, 78-problem application and whole-flow
  evidence is retained; unselected paths have no universal speedup.
- See [the 0.46 validation record](../../docs/validation/torch-runtime-iteration-46.md).

## Next implementation areas

1. Remaining model lifecycle: backward/global hooks, broader checkpoint object
   formats and expanded memory formats. Local forward and public checkpoint callbacks
   are covered by the 0.8 milestone above.
2. Optimizer extensions: remaining scheduling policies, common hooks and algorithms where
   numerical and performance evidence justify them; keep state compatibility and
   unknown-option errors explicit. Profile small-parameter update overhead.
3. Core tensor/autograd coverage: audit shape/dtype promotion, views and mutation,
   remaining retention corner cases, graph-node callbacks and forward-mode gaps.
   Profile first-gradient descriptor checks and strided copies after the0.27
   layout/ownership fix, including optimizer access and many small parameters.
   Add APIs
   with native value, VJP/JVP and boundary tests, independent of course coverage.
4. Neural-network coverage: expand attention options,
   further pooling/convolution options, with masked/empty/non-contiguous
   cases and numerical checks. Optimize exact array erf and continue auditing
   low-precision/statistic accumulation behavior across numerical runtimes.
5. Runtime efficiency: profile allocation/graph overhead and larger linear algebra
   workloads. Track import size, cold start, steady-state latency and memory as
   separate measurements. Retain baselines and record regressions, not just wins.
   The multi-input VJP removes the prior float64 logits-retention regression;
   larger attention sequences still need memory and throughput measurements.
6. Optional WebGPU backend: keep storage resident across operations, provide
   explicit device/transfer semantics, feature detection and unsupported-operation
   policy, and validate analytical gradients. Establish an end-to-end benefit
   before enabling it for PaperLeet. A WGSL matmul probe is not an implemented
   torch GPU backend; Python/JS transfer and graph execution remain required work.

## Stable integration and independent release gates

- The default upstream branch is currently `main` (`origin/HEAD`), corresponding
  to the user's request to adopt mature versions on the main/master branch.
  Prepare a focused, reproducible integration that excludes unrelated concurrent
  UI/backend edits; do not treat an uncommitted working tree as a released version.
- Both independent and private vendored wheel layouts work outside the checkout
  and have disjoint installed paths. No dependency on the application, catalog,
  native PyTorch or browser assets is allowed in the independent runtime.
- Maintain native-reference core API tests and a standalone test entry point;
  NumPy is the only runtime dependency. Check packaging metadata, license,
  documented scope, installation, CLI and import-alias behavior before publishing.
  The independent library uses Apache-2.0. Wheel metadata, both installed layouts
  and both browser profiles carry LICENSE/NOTICE; this scope does not license the
  rest of the application. Public package publication remains separate work.
- Run the complete application infrastructure suite, all configured original
  reference/mutant suites, exported-source checks and representative real WASM
  regressions. New backend behavior also needs real browser Worker evidence.
- Document supported API/dtype/gradient/device boundaries and performance under
  identified environments. Preserve typed unsupported errors and server fallback
  behavior; never silently pass off another backend's result as GPU execution.
- A stable repository version is an integration milestone. Independent-library
  readiness and optional GPU adoption remain separate goals requiring evidence.

## InstanceNorm milestone (0.47)

- Add batched/unbatched InstanceNorm1d/2d/3d, functional/root entries, native
  shared `_NormBase` state and optional bias. Preserve running snapshots,
  counters, mutation guards, empty boundaries, analytical VJP/JVP and mixed AD.
- Validate 182 focused tests, 118 native differential cases, 38,714 CPU passes,
  5,085 actual Node/Chrome contracts and all 78 exported PaperLeet problems.
- Compare 26 direct/composed flows across 12 fresh CPU/Node/Chrome processes,
  including complete convolution/SGD steps; retain every timing sample.
- Preserve 10,387 prior payloads and seven release-policy files. See the
  [iteration 47 record](../../docs/validation/torch-runtime-iteration-47.md).
  Broader core APIs, efficiency and optional persistent GPU backend work remain active.

## Transposed convolution milestone (0.48)

- Add grouped ConvTranspose1d/2d/3d, functional/root entries, `_ConvTransposeNd`,
  output-size resolution, native weight/reset state and 2D output layout rules.
- Use clipped scatter and bounded packed contraction, analytical VJPs, bilinear
  mixed AD and large-padding gather/tap fallbacks that retain native IEEE behavior.
  Correct existing regular-convolution empty output/tangent strides.
- Validate 515 transposed +207 regular convolution cases, 144×5 native numeric
  cases plus68 state/boundary cases, 39,229 CPU passes, 5,600 actual Node/Chrome
  contracts and all78 exported problems. Retain38 timing flows across12 fresh
  CPU/Node/Chrome children with direct/matrix controls and decoder SGD updates.
- Preserve10,545 prior payloads, latest upstream frontend work and seven canonical
  policy files. See the [iteration48 record](../../docs/validation/torch-runtime-iteration-48.md).
  Broader core APIs, efficiency and persistent GPU backend evaluation remain active.

## Bounded regular-convolution zero padding (0.49)

- Share clipped gather/scatter across regular and transposed convolution. Avoid
  full theoretical padded activations/adjoints; release expanded gather arrays
  after forward and regather only for weight VJPs. Preserve implicit-zero IEEE
  products, mixed AD, native metadata and all existing nonzero padding modes.
- Validate292 new tests plus722 convolution regressions,96×6 regular native
  cases,144×5 transposed cases and explicit IEEE/state oracles;39,521 CPU passes,
  5,892 actual Node/Chrome contracts and all78 exported problems.
- Retain32 performance flows across12 fresh CPU/Node/Chrome children, actual
  version baselines and padded/gather/tap controls; ordinary cases and full
  encoder/decoder SGD measurements keep regressions/noise visible.
- Preserve10,703 prior payloads, latest upstream frontend work and seven policies.
  See the [iteration49 record](../../docs/validation/torch-runtime-iteration-49.md).
  Broader core API, efficiency and persistent GPU-backend evaluation remain active.

## Math attention, grouped heads and forward AD (0.50)

- Add GQA without repeated K/V storage, attention dropout with one shared mask,
  exact JVPs and reverse-over-forward for operand/mask channels and shared views.
- Preserve fused VJPs, explicit dtype/shape policies and a bounded metadata cache.
- Validate 161 independent tests, 204 native math cases / 2,592 arrays, a native
  mixed-AD encoder and checkpoint continuation, complete CPU/Node/Chrome/catalog
  gates and 22 performance flows across 12 fresh children.
- Record the stable zero derivative for fully-masked rows and native mixed-AD NaN
  differences explicitly. Keep every ordinary/grouped/training timing and control.
- See the [iteration 50 record](../../docs/validation/torch-runtime-iteration-50.md).
  Multihead/Transformer modules, larger-sequence memory/throughput, further core
  coverage, measured attention validation/setup overhead and a persistent GPU
  backend remain active work.
