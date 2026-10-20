# StrideTorch0.29 candidate: batched linear systems

This candidate expands the former unbatched-only `torch.linalg.solve` to dense
batched and broadcast systems, vector/matrix right-hand sides and `left=False`.
Initial performance regressions prompted the metadata/graph refinements below.
Local application checks are recorded separately from core proofs and actual
remote deployment/live verification. Version0.28 is the last runtime with a
completed exact-source remote deployment/live proof at this milestone.

## Compatibility and differentiation

The public signature uses native `A/B`, keyword-only `left/out`, strict Tensor
inputs and boolean `left`. Float32/64 differentiation and complex64/128 inference
are supported. Native vector classification is matched: a one-dimensional B, or
an exact A.shape[:-1] shape, is a vector; rank alone is insufficient. Right solves
reject vector B as native does. Broadcast inputs are views, and explicit matrix
promotion works under both NumPy1.26 and2.x. Singular matrices raise RuntimeError,
including empty right-hand-side matrices. Output strides follow native CPU
geometry, including scalar-sized axes, batched vectors and empty outputs.

The real VJP solves one adjoint system per traversal, sharing it when A and B
both need gradients. Broadcast dimensions reduce into original input shapes;
repeated seeds are recomputed rather than cached. Frozen-input cases use one
ordinary graph edge. Native saved-value guards include A and the output, while
B may change after forward. Detached aliases trigger the same storage guards.
Intermediate arrays release after nonretained backward. Existing module dtype
rebinding continues to use original saved arrays as before this change.

JVP implements dX=solve(A,dB-dA@X), or its right-sided form. Its reverse behavior
matches native's constant saved-factor rule: the outer coefficient is detached,
while X and residual retain reverse edges. A materialized zero coefficient
direction preserves native zero-versus-None gradients when only B is dual.
NumPy still refactorizes each solve; no reusable LU factorization is claimed.
Complex AD, half-precision factorization, sparse systems and out buffers retain
explicit unsupported boundaries. Lazy conjugate metadata is not implemented;
complex values are compared against native resolve_conj() results.

Native-reference tests cover values/strides, three input layouts, both sides,
broadcast reductions, all gradient-input combinations, dynamic retained seeds,
JVP/reverse-of-tangent and saved-value mutation policy. Independent tests use
closed-form2x2 inverses and residual identities, analytical derivatives and five
SGD steps with checkpoint restoration, rather than another adapter implementation.

## Measurements and candidate status

Three fresh-process ABBA phases preserve48 child receipts: CPython/Pyodide,
float32/64,2x2 and16x16 matrices, both or one input gradient, retained backward,
complete forward/square/mean/backward and no-grad forward. Each case uses100
warmups and nine batches; small cases use500 calls, larger derivative cases100.
Timing excludes import, which is recorded separately. CPU thread counts are
requested through environment variables; actual BLAS threading is not verified.
The host is shared, and results are not a general throughput claim.

The first broad implementation regresses complete small-system paths by roughly
20–50%, while no-grad forward more than doubles. Removing unnecessary broadcast
views, caching adjoint/transposed-output metadata and selecting ordinary graph
edges for frozen inputs reduces this cost. The final selective trial generally
saves about9–18% in double-input retained backward on WASM; one CPUfloat32 small
sample is0.9% slower and CPUfloat64 large samples show larger savings. Complete
forward/backward is still mostly slower, with final observed regressions up to
23.5%; no-grad forward is up to35.6% slower. These results are retained, not hidden
by the reduced number of mathematical solves. Those results motivated the subsequent refinement below and do not describe
the adopted refined source.

Initial native failures exposed vector result layout and the native JVP saved
factor/zero-edge rules, plus two test-fixture errors: complex lazy-conjugate NumPy
conversion and direct mutation of unsupported nonleaf tensors. Later a capability
attribute typo was corrected. All original failing logs and prototype sources
remain in the evidence archive; corrected tests use resolve_conj and detached
writes without expanding general in-place AD.

Core CPython, minimum wheel and real Node/Chrome proofs are recorded in the
[machine-readable candidate evidence](../../packages/torchwasm/benchmarks/results/iteration-29.json).
Full exported-catalog/frontend/application distribution checks and main adoption
are separate later gates, not implied by independent-package compatibility.


## Original candidate core gates

The original selective source passes29853 root/independent tests with one optional
Docker skip in194.91s. Focused solve/model-conversion/joint-pullback tests pass574.
The actual public wheel installs without native Torch in Python3.10.19 /
NumPy1.26.4 and passes916 tests. Pinned Pyodide0.29.4 / NumPy2.2.5 under Node and
real headless Chrome155 Worker each pass904 tests in29 suites. All45 core/license
hashes match the timed selective source, both WASM executions and installed wheel.
These gates establish local core compatibility. They do not waive the measured
complete-flow performance concern or substitute for application adoption gates.


## Metadata and graph refinement

The refined implementation validates array metadata directly, uses ndarray axis
methods, specializes ordinary result layouts and avoids column promotion for an
unbatched vector. Ordinary unbatched left-solve pullbacks avoid constructing
broadcast/reduction callbacks. All batched/right-side and native layout/guard
contracts remain. The general result constructor builds a single active edge
without temporary lists/generators; the joint constructor builds its shared
node once. No numeric values, gradients or seeds are added to a cache.

Five additional fresh-process ABBA phases preserve80 child receipts for metadata,
joint-node, single-edge, unbatched and vector trials. They use the original
14-case matrix-RHS recipe and retain every anomaly. The final fresh phase includes
large outliers, including26% slower CPUfloat64 small full-flow and25% slower
WASMfloat32 large single-input full-flow samples, alongside implausibly broad
float64 WASM wins up to64%. These are not used to infer a general speedup or a
particular cause of timing variance.

Two additional paired phases preserve16 fresh processes. Each loads two isolated
package namespaces in the same NumPy interpreter, alternates13 ABBA/BAAB blocks,
warms1000 calls and leaves GC enabled. The final phase uses200 calls per full-flow
sample and1500 per no-grad sample, includes vectors and real SGD updates, and
checks final input/output values and dtypes across implementations. It supplements
rather than replaces the original fresh-process evidence.

Final paired process medians show no-grad small-vector solve about7–11% faster,
LayerNorm about1.5–4.1%, attention2–3%, unary chains3.8–6.1% and no-grad exp27–29%.
Actual SGD update flows range from1.5% slower to11.9% faster, depending on runtime,
dtype and matrix size. Some tiny solve-only flows remain2–5.3% slower; no claim
that every operation or training run accelerates is made. Both numerical runtimes
and float32/64 are measured; actual BLAS threading remains unverified.

The candidate's broad library value is native batch/right/JVP coverage plus reduced
shared graph overhead. App adoption still requires complete final-source package,
exported-WASM, catalog and frontend gates; the coarse fresh-process outliers and
small paired regressions are retained for later performance work.


The refined final source separately passes29853 tests/one optional Docker skip
in185.67s, plus the2084 focused contracts. Its actual wheel/sdist match current
README metadata and45 core/license hashes. The NumPy-only Python3.10.19 /
NumPy1.26.4 installation passes916 tests; Node and real Chrome155 Worker each
pass904 in29 unchanged standalone suites. Both public and private wheels execute
batched right solve/VJP/JVP, matrix views, gradient retention and NaN workflows;
each remains usable after uninstalling the other. The private wheel/sdist also
match current application server/catalog bytes. Original proof folders remain
separate; the adopted refined proof is `proof-refined`, source `trial-vector`.

[Refinement evidence](../../packages/torchwasm/benchmarks/results/iteration-29-refinement.json)
and its receipt archive preserve all96 additional timing children, five source
trials, both paired recipes, original anomalies and exact final-source proofs.
The broader main-branch adoption gates are still pending at this core milestone.


## Local application integration

The original native catalog passes78 modules/10218 executions/342 mutants with
three repeats in131.68s. All original budgets remain unchanged. Exported WASM
passes all78 modules/342 mutants in273754ms at concurrency2, including904 checks
in29 standalone suites whose hashes match the independent Node/Chrome proof.
No browser budget is exceeded. Catalog durations are compatibility/budget gates,
not comparative performance evidence.

Integration preserves23 incoming upstream files byte-for-byte through96200d50:
21 community-affiliation UI/test files and two later inbox CSS/documentation files.
No upstream problem/test/judge/package/Python/catalog/lockfile input changes.
The complete29853-test core/infrastructure gate and native catalog therefore
retain exact mathematical inputs. All7388 prior immutable assets remain unchanged;
158 new reachable payloads and three mutable indexes are frozen for later live
checks. Both runtime profiles carry version0.29 LICENSE/NOTICE.

Complete frontend checks pass57 bilingual guides/171 prompts at123d. After the
CSS-only962 update, the targeted inbox check and both complete builds pass again.
Latest default and actual hosted Pages-profile builds each have three initial JS
assets:402639 raw/130068 gzip and404041 raw/130523 gzip bytes, within existing
budgets. Actual profile settings come from the revalidated repository variable.
The condensed React checklist preserves derived eligibility, semantic disabled
controls, existing owner/route/in-flight guards and deferred parent pages; no
manual visual UI review is claimed.

The initial targeted Inbox command used the repository root instead of its
required web working directory and failed module loading before assertions. That
failed log/receipt is retained; the corrected web-directory command passes with
no source fix. Full frontend and latest CSS follow-up contexts remain explicit.
Application wheel/sdist core/license hashes and current server/catalog bytes match
the same refined source after UI merges; installed public/private ownership and
independent uninstall proofs remain in the core refinement record.

[Integration evidence](../../packages/torchwasm/benchmarks/results/iteration-29-integration-gates.json)
and its222-file archive preserve actual reports, exports, dependency hashes,
all process receipts and the corrected-command history. Adoption uses a non-force
push after latest-main revalidation. Exact remote Quality, eligible actual Pages
deployment and live-byte verification remain separate pending gates.


## Remote adoption and actual live proof

Source main0d5975a0 passes all four Quality jobs in run38926381186. Infrastructure
passes29853/one optional skip in570.26s;native78/10218/342/three repeats passes
in231.33s and WASM78/342/904 independent checks passes in531004ms. Source Pages
is superseded. Equivalent descendant050c467a also passes Quality;its overall-success
Pages38929018902 skips `actions/deploy-pages`, so it does not prove publication.

Descendant033dc1bc passes Quality38929257323 and actual Pages38930642945,
including the deploy action. All45 frozen runtime/resource files and161 active
exports are byte-identical to0d5975a0 at both descendants;the seven changed files
are UI/styles/checks/docs. Its native gate passes in312.06s and WASM in527082ms.
Live requests verify78 decoded starters/details/manifest references,158 immutable
payloads and both runtime licenses/version0.29. This is the latest positively
verified site version;runner durations are budget gates, not speedup comparisons.

[Remote evidence](../../packages/torchwasm/benchmarks/results/iteration-29-remote-adoption.json)
and its27-file archive retain all three Quality/artifact contexts, the skipped
publication history, pinned Git/source/export equivalence and actual live receipts.
