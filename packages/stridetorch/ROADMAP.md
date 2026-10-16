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
Quality/Pages/live verification is being followed separately. This is not yet
positive evidence of0.26 site publication.

## Version 0.27 candidate

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
  Immutable export/full exported WASM/latest-main adoption remain gates.

## Next implementation areas

1. Remaining model lifecycle: backward/global hooks, file checkpoint serialization
   and expanded memory formats. Local forward and public checkpoint callbacks
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
4. Neural-network coverage: expand instance normalization, attention options,
   transpose convolution and pooling, with masked/empty/non-contiguous
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
