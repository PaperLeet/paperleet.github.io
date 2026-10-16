# StrideTorch0.29 candidate: batched linear systems

This candidate expands the former unbatched-only `torch.linalg.solve` to dense
batched and broadcast systems, vector/matrix right-hand sides and `left=False`.
It is not yet adopted on main: initial performance measurements show regressions
in the complete small-matrix forward/backward path despite savings in shared
adjoint work. The verified stable runtime on main/site remains0.28.

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
by the reduced number of mathematical solves. The candidate needs further
forward-path/graph-allocation work before stable application adoption.

Initial native failures exposed vector result layout and the native JVP saved
factor/zero-edge rules, plus two test-fixture errors: complex lazy-conjugate NumPy
conversion and direct mutation of unsupported nonleaf tensors. Later a capability
attribute typo was corrected. All original failing logs and prototype sources
remain in the evidence archive; corrected tests use resolve_conj and detached
writes without expanding general in-place AD.

Core CPython, minimum wheel and real Node/Chrome proofs are recorded in the
[machine-readable candidate evidence](../../packages/stridetorch/benchmarks/results/iteration-29.json).
Full exported-catalog/frontend/application distribution checks and main adoption
are separate later gates, not implied by independent-package compatibility.


## Verified local core gates

The final selective source passes29853 root/independent tests with one optional
Docker skip in194.91s. Focused solve/model-conversion/joint-pullback tests pass574.
The actual public wheel installs without native Torch in Python3.10.19 /
NumPy1.26.4 and passes916 tests. Pinned Pyodide0.29.4 / NumPy2.2.5 under Node and
real headless Chrome155 Worker each pass904 tests in29 suites. All45 core/license
hashes match the timed selective source, both WASM executions and installed wheel.
These gates establish local core compatibility. They do not waive the measured
complete-flow performance concern or substitute for application adoption gates.
