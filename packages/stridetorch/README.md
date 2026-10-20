# StrideTorch

A lightweight, bounded `torch`-compatible Python library backed by NumPy.
It runs in CPython and Pyodide without installing native PyTorch. Analytical
reverse-mode gradients and a validated forward-mode subset come from PaperLeet's
existing browser adapter. Core API fixes and performance improvements are
developed here and checked against independent native PyTorch results.

Version **0.85** avoids index-sized comparison buffers when checking integer
indices. In the measured large indexed aggregation forward calls, traced peak
allocation falls about 95% on native NumPy and 85% in Pyodide. Empty inputs,
index views, guarded updates and analytical derivatives retain their behavior.
The [index-bounds record](../../docs/validation/torch-index-bounds.md) retains
complete-flow timings, unchanged-peak controls and an initial Node timing
anomaly that did not recur in three independent focused repeats.

Version **0.84** adds `index_select`, `index_add` and `scatter_add`, including
analytic gradients, float32/64 forward differentiation and guarded in-place
updates. Selection reverse allocates its zero accumulator directly; the
[indexed CPU/WASM record](../../docs/validation/torch-indexed-cpu-wasm.md)
records about 25%–50% lower traced peak allocation in the measured CPU cases.
That measurement is an allocation result, not a latency or WASM heap claim.

Version **0.83** releases unused operand arrays from ordinary multiplication
and direct-division pullbacks. A measured large scalar chain reduces traced
peak allocation by about 83%; the scoped latency costs are retained in the
[qualification record](../../docs/validation/torch-runtime-0.83-qualification.md).
Gradient arithmetic, saved-value guards and higher-order rules retain their
qualified 0.82 behavior. CPU/Pyodide package and current PaperLeet application
checks pass locally. Production WebGPU remains disabled.

Version **0.81** packs contiguous reduction rows for eligible 2D WASM matrix
products. Representative complete forward/gradient flows improve while native
CPU BLAS routing and public APIs retain their existing behavior. The
[0.81 qualification record](../../docs/validation/torch-runtime-0.81-qualification.md)
records measured scope, temporary-copy memory costs, installed-package checks
and local PaperLeet application/browser verification. WebGPU remains an
experimental developer prototype; it is not an enabled torch backend.

```sh
python -m pip install ./packages/stridetorch
stridetorch train.py --epochs 10
# Equivalent: python -m stridetorch train.py --epochs 10
```

Keep existing imports in `train.py`:

```python
import torch
from torch import nn
import torch.nn.functional as F

x = torch.tensor([2., 3.], requires_grad=True)
x.square().sum().backward()
print(x.grad.tolist())  # [4.0, 6.0]

layer = nn.Linear(2, 1)
optimizer = torch.optim.Adam(layer.parameters(), lr=0.01)
```

The launcher enables `torch` and its supported submodule aliases before running
the script, passes through arguments and exit codes, and allows imports from the
script's directory. For a Python or Pyodide host, activate aliases once before
loading existing user code:

```python
from stridetorch import install_as_torch
install_as_torch()

import torch
from torch.nn import Linear
import torch.nn.functional as F
```

This enables `torch.nn`, `torch.nn.functional`, `torch.optim`, `torch.linalg`,
`torch.autograd`, `torch.func`, `torch.testing`, `torch.random` and
`torch.serialization`. Plain
`import stridetorch` leaves native PyTorch imports untouched. Explicit usage
via `from stridetorch import torch` also remains available. To scope aliases:

```python
from stridetorch import as_torch

with as_torch():
    import torch
    from torch.nn.functional import linear
    # Existing torch imports now use the lightweight implementation.
# Pre-existing native torch modules are restored, including after exceptions.
```

Import aliases are process-wide: activate them before importing user code in an
isolated process or browser Worker. Activation replaces cached `torch.*` modules;
it cannot rewrite references already held by other code. Namespace objects share
gradient mode, storage tracking and RNG
state; they are not independent execution engines. Create a fresh Worker for each
untrusted submission, as PaperLeet does.

## Matrix and axis views

`Tensor.t()` / `torch.t` transpose a matrix and return a view for scalars or
vectors; ranks above two raise an error. `Tensor.mT` swaps the last two axes of
a matrix batch. As in native Torch, a scalar `.mT` returns the original object
with a deprecation warning, while a vector `.mT` is invalid. Neither operation
conjugates complex values.

`movedim` / `moveaxis` and `swapaxes` / `swapdims` are available as Tensor methods
and top-level Torch functions. Movement accepts two integers or two integer
sequences with equal lengths and unique normalized axes. Unmoved axes keep their
relative order. `swapaxes` uses `axis0/axis1` keywords; `swapdims` uses `dim0/dim1`.

```python
x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
assert x.movedim((0, 2), (2, 0)).shape == (4, 3, 2)
assert x.mT.shape == (2, 4, 3)
assert torch.swapaxes(x, axis0=0, axis1=2).shape == (4, 3, 2)
```

Ordinary primal views share storage and mutation versions with their input;
their reordered strides do not require a value copy. Real floating reverse-mode gradients,
float32/64 JVP and reverse differentiation of the tangent use the existing view
engine, including its tangent-layout materialization rules. Complex transposes are available for inference; complex differentiation
retains its explicit unsupported error. Axis lists and integer Tensor arguments
are converted to immutable metadata at forward time and are not retained by the
tape. Detached view writes still invalidate computations that saved those values.

## Optional feature detection

Torch module namespaces, Tensors and optimizers support ordinary Python feature checks:
`hasattr(value, name)` returns false for a missing symbol, and
`getattr(value, name, default)` returns the supplied default. Dynamic attributes
are honored; missing names are not cached. For example,
`getattr(torch.nn.functional, "conv_transpose2d", None)` can select your own
optional implementation when that operator is unavailable.
Both offline browser profiles share this behavior; the smaller Schedules
profile retains its explicit unsupported-autograd boundary.

A direct missing public lookup raises an `AttributeError` that also matches
`UnsupportedOperation`, preserving explicit backend fallback handlers. An
unavailable `from torch import ...` follows Python's `ImportError` behavior.
Existing errors from unsupported modes and real descriptor failures still
propagate; attribute existence alone does not establish dtype, gradient or
device support. The compatibility and validation scope is recorded in the
[0.37 report](../../docs/validation/torch-runtime-iteration-37.md).

## Max pooling

`torch.nn.MaxPool1d` and its functional entry points support CL/NCL inputs,
stride, padding, dilation, ceil mode, returned indices and float32/float64
reverse AD, JVP and reverse-over-forward. The root `torch.max_pool1d` returns
values; `torch.max_pool1d_with_indices` returns values and indices. The 1D path
shares numerical kernels without adding Tensor view graph nodes. MaxPool1d and
MaxPool2d remain sibling Module types, preserving dimensional type checks.
Float16 inference is supported. Integer 1D inference requires returned indices,
matching the targeted native CPU dispatch; value-only integer 1D pooling is an
explicit unsupported operation. Valid 1D kernels above 8,000,000 taps require
the server. Conventional contiguous strides are preserved for empty values
and indices.

`nn.MaxPool3d`, `nn.functional.max_pool3d` and `max_pool3d_with_indices`
support CDHW/NCDHW inputs, empty batches, stride, padding, dilation, ceil mode
and per-channel flattened int64 indices. The root `torch.max_pool3d` returns
values. Float16 and supported integer types work with either return form;
float32/float64 support reverse AD, JVP and reverse-over-forward. Positive
strided NumPy inputs with an NDHWC layout preserve native channels-last output
and index strides. JVP follows the native contiguous layout for ambiguous
singleton outputs. This does not add general `channels_last_3d` conversion APIs.
The three MaxPool Module classes are siblings. Three-dimensional packing uses
the same 32 MiB workspace guard, with specialized 1/2-edge blocks and an ordered
scan fallback. Valid kernels or outputs above 8,000,000 points and input spatial
indices beyond the runtime's pointer range require the server.

Overlapping 3D depth windows can reuse each HW reduction. Automatic dispatch
requires a substantial reduction in estimated repeated work and bounds the
live intermediate arrays before selecting this path. Small index/gradient
tapes retain their original route when fixed processing overhead dominates.
An ordered depth pass
carries the selected HW indices directly, preserving first-finite ties and
last-NaN routes without extra Tensor graph nodes. Each packed HW stage uses the
remaining temporary-workspace budget; larger stages use ordered scans. The
existing 3D paths handle geometries without useful reuse or enough workspace,
including the established empty-sampling derivative boundary. Empty batches
still avoid spatial coordinate grids.

`torch.nn.MaxPool2d`, `torch.nn.functional.max_pool2d` and
`max_pool2d_with_indices` support batched/unbatched 2D inputs, noncontiguous
arrays, empty batches, stride, padding, dilation and ceil mode. Indices are
int64 offsets within each input channel. Finite ties select the first sample;
multiple NaNs select the last NaN, including the corresponding gradient route.
The root `torch.max_pool2d` entry point returns values only.

Float32/float64 support reverse-mode gradients, JVP and reverse-over-forward.
Float16 and supported integer types also work for inference. Inputs and returned
indices retain native-style storage-version guards; detached output edits do
not invalidate the pool's backward rule. The existing general restrictions on
direct edits of differentiable non-leaf tensors still apply.

Specialized 1×2/2×2 pooling, window packing and scanning share these semantics.
Scanning visits only kernel offsets that can reach an input sample, retaining
their original order. Huge padding with a tiny input therefore avoids iterating
every nominal kernel position, including on the bounded-workspace fallback.
Window packing has a conservative 32 MiB temporary-workspace limit and falls
back to scanning beyond it. Extreme dilated/padded windows with no real input
sample reproduce native inference values/indices but reject derivatives
explicitly. See the [0.38 validation record](../../docs/validation/torch-runtime-iteration-38.md)
for measured complete CNN flows and the remaining compatibility boundaries.

## Average pooling

`nn.AvgPool1d`, `nn.AdaptiveAvgPool1d` and their functional/root entry points
accept CL/NCL input with float32/float64 reverse AD, JVP and
reverse-over-forward. They share the ordered 2D reduction kernels without
creating extra Tensor view nodes. Float16 inference and fixed int64 inference
are supported. Fixed 1D pooling implements stride, padding, ceil mode and
padding inclusion; divisor overrides are a 2D-only option. Global adaptive
length 1 follows mean's stateless backward, including NaN output and empty
gradient for empty input length. Other adaptive lengths preserve native
empty-channel and saved-input version rules.

`nn.AvgPool2d`, `nn.AdaptiveAvgPool2d` and their functional entry points support
float32/float64 reverse AD, JVP and reverse-over-forward. Area interpolation
shares the adaptive rule; bicubic forward AD retains its explicit unsupported
boundary. Float16 inference is supported, and fixed average pooling also accepts
int64 inference. Fixed pooling implements ceil mode, padding inclusion and
nonzero positive/negative divisor overrides.

NaN/Inf values and gradient seeds affect only their actual regions. Ordered
fallbacks retain overflow and ill-conditioned cancellation behavior. The global
1×1 specialization follows mean's stateless backward behavior; other adaptive
outputs retain native-style input version guards. Zero-sized adaptive outputs
are valid for forward evaluation and reject backward as native PyTorch does.
Empty average-pooling outputs preserve conventional contiguous element strides.

`nn.AvgPool3d`, `nn.AdaptiveAvgPool3d` and their functional entry points accept
CDHW/NCDHW inputs with float32/float64 reverse AD, JVP and reverse-over-forward.
Fixed 3D pooling implements stride, padding, ceil mode, padding inclusion and
nonzero integer divisor overrides, including int64 inference. Adaptive 3D pooling
also supports float16 inference and `None` output dimensions. Native CPU fixed
3D pooling does not support float16; this backend reports that boundary explicitly.
The input must be at least as large as the fixed kernel, even with padding.
Empty batches are supported without constructing coordinate grids.

Volumetric reductions preserve spatial accumulation order and divide adaptive
regions by depth, height and width in sequence. Small non-overlapping blocks use
strided additions; other regions use packing, tiling or ordered streaming under
a 32 MiB packing estimate. Adjoints use block broadcasting or bounded ordered
scatter, reading non-contiguous gradient seeds by blocks. This estimate excludes
input/output storage and is not a total-process memory limit.
Up to 32 small geometry plans cache readonly coordinates, divisors and optional
flat gather indices/masks (at most 128 KiB of gather arrays per plan). Contiguous
inputs gather directly into the accumulation buffer; the cache holds no input
values or Tensor storage.
The global 1×1×1 case follows mean, including empty-input NaNs and stateless
backward, with conventional or channels-last output layout. NumPy and native CPU
mean can differ on ill-conditioned cancellation inputs; bitwise equality is not
promised. Singleton leaf-gradient strides can also differ while their indexed
values and non-singleton strides agree.

Geometry and inverse-region caches are bounded and hold no user tensors. Finite
forward matrix paths check live values and require each plane to have no mixed
signs. Adaptive adjoints gather contributions in native output order rather
than reassociating them with matrix products. Larger or exceptional regions use bounded packing,
tiling or streaming. Oversized adjoints reject inverse-gather plans before
allocating full-length inverse indices. The packing estimate is separate from output allocations
and total process memory. Correct IEEE behavior adds overhead to some small
pooling calls; identity resizing, upsampling and complete CNN flows have their
own measured results in the [0.39 validation record](../../docs/validation/torch-runtime-iteration-39.md).

Version 0.40 additionally caches readonly packed indices, masks and typed
divisors: at most 64 plans, each holding at most 64 KiB of array data. Contiguous
inputs can use flat-index gathering; other layouts retain coordinate gathering
without copying the entire input into contiguous storage. Both paths preserve
ordered accumulation and return independent output storage. Oversized plans
keep the existing bounded fallbacks. See the
[0.40 validation record](../../docs/validation/torch-runtime-iteration-40.md).

Version 0.46 adds ordered inverse-region adjoints for fixed/adaptive 3D pooling.
Each output seed is scaled once before gathering contributions in native output
order. A separate 32-entry readonly inverse-geometry cache holds no tensors or
seed values; each optional gather-index/mask plan is at most 128 KiB. Cache keys
use bounded immutable coordinate bytes. Input-axis construction, the scaled seed
copy and packing estimates must fit before selecting this path. Shapes with
nominal inverse work over twice the scatter work retain the existing adjoint.
The estimate excludes required input/output storage and resident cache metadata;
it is not a measured process-memory limit. Selected inverse calls avoid a
previously discarded zero-gradient allocation, and complete disjoint blocks fill
their result directly. Float32/float64 cancellation-sensitive seeds, nonfinite
locality, strided seeds and reverse-over-forward retain their ordered rules.

## Linear systems

`torch.linalg.solve(A, B, *, left=True)` handles dense batched/broadcast systems,
vector or matrix right-hand sides and right-sided matrix systems (`left=False`).
Float32/64 values, real analytical VJP/JVP and complex64/128 inference are covered.
The shared VJP solves one adjoint system even when both inputs require gradients;
B is not saved, while mutations of A or the output invalidate backward.

Vector classification follows native Torch's exact shape rule; a vector B is
invalid for a right-sided solve. Outputs follow native CPU strides, including
batched vectors and empty shapes. The reverse tape of a JVP holds its outer
factorization constant, matching native saved-factor semantics. The NumPy backend
still factorizes each invocation. Complex AD, half factorization, sparse systems
and `out` buffers are explicit unsupported cases.

## Triangular systems

`torch.linalg.solve_triangular(A, B, *, upper, left=True, unitriangular=False)`
solves dense batched/broadcast triangular systems. B must be a matrix or matrix
batch; this API does not apply generic solve's vector-overload rule. Only the
selected triangle is used. With `unitriangular=True`, diagonal input values are
ignored and treated as ones, including uninitialized/NaN values there.

```python
a = torch.tensor([[2., .5], [99., 3.]], dtype=torch.float64, requires_grad=True)
b = torch.ones(2, 1, dtype=torch.float64)
x = torch.linalg.solve_triangular(a, b, upper=True)  # Ignores the lower99.
x.square().sum().backward()
```

Float32/64 support analytical VJP/JVP and reverse differentiation of the tangent;
complex64/128 support inference. Results use column-major final axes on both
left and right solves. Backward computes one adjoint solve for both active input
gradients, projects A's gradient onto its used triangle, and zeros the unit
diagonal's gradient. A and the output carry native mutation guards; B is not
saved. The JVP's outer triangular coefficient remains differentiable, matching
native Torch rather than generic solve's saved-LU behavior.

Larger systems use32-row block substitution with NumPy matrix updates and small
LAPACK calls; tiny systems use direct formulas and small unbatched systems use
one local solve. A bounded cache retains only small read-only triangular masks
(up to64KiB of structural mask data), never tensor values. Singular diagonals
propagate nonfinite values rather than
raising the generic singular-matrix exception. Output buffers, half factorization,
sparse systems and complex differentiation remain explicit unsupported cases.
Version0.31 passed local application integration and was adopted on main. Remote
Quality, deployment and live verification are tracked separately in its validation report.

Version0.29 passed application integration and actual Pages/live verification.
Refinements reduce graph/metadata overhead and improve broader inference and
training controls; some tiny solve-only flows remain a few percent slower. See the
[measurements and adoption boundary](../../docs/validation/torch-runtime-iteration-29.md).

## Cholesky factors

`torch.linalg.cholesky(A, *, upper=False)` factors positive-definite real or
Hermitian complex matrix batches. It reads only the selected triangle and returns
column-major final axes. Float32/64 support analytical reverse gradients, JVP
and reverse differentiation through the primal and direction of a JVP.
Complex64/128 support inference; complex differentiation, half factorization and
output buffers raise explicit unsupported errors.

```python
a = torch.tensor([[4., 1.], [1., 3.]], dtype=torch.float64, requires_grad=True)
factor = torch.linalg.cholesky(a)
factor.square().sum().backward()
```

Backward reuses the returned factor and triangular systems. It saves no input
array: changing A after forward is permitted, while changing the factor
invalidates backward. A failed factorization reports the first failing batch and
leading minor. Factorization and browser behavior are covered by the
[validation report](../../docs/validation/torch-runtime-iteration-32.md).

`torch.cholesky_solve(input, input2, upper=False)` solves from an existing
Cholesky factor instead of constructing the positive-definite coefficient again.
`input` is a matrix RHS and `input2` is the lower/upper factor; batches broadcast.
The equivalent method is `input.cholesky_solve(input2, upper=False)`.

```python
b = torch.ones(2, 1, dtype=torch.float64)
x = torch.cholesky_solve(b, torch.linalg.cholesky(a))
```

Float32/64 support VJP, JVP and reverse differentiation through both primal and
direction. Complex64/128 support inference. Primal solves read the selected
triangle; native differentiation extends the full factor and direction. For a
trainable SPD system, compose the factorization with `A = M @ M.T + I`.
Backward guards RHS, factor and output mutations, and shares one adjoint solve
across both input gradients. Narrow matrix RHS gradients associate products
through their columns to avoid a cubic factor-gradient multiplication. Finite
triangular factors needing gradients reuse one forward and one adjoint
intermediate. Finite narrow JVPs use one analytical multi-input pullback while
keeping all full-factor and direction gradients. Forward and adjoint products
through RHS columns are shared across the four pullbacks. Actual finite
triangular factors also reuse the first primal, tangent and adjoint solves; full
factor/direction gradients are preserved. Frozen-factor classification is
selected by runtime and RHS size to avoid scans that lose to native CPU BLAS. Unneeded primal/tangent arrays
are released; permitted direction edits use a snapshot, while required saved
values retain native mutation guards. These are graph-local values, not
cross-call caches. Output
buffers, half factorization and complex AD remain explicit unsupported cases.
Version 0.35 validation and integration are tracked in its
[validation report](../../docs/validation/torch-runtime-iteration-35.md).

`torch.cholesky_inverse(input, upper=False)` and the matching Tensor method invert
a matrix from its existing Cholesky factor. They read the selected factor triangle
and return column-major final axes, including empty batches. A zero diagonal
reports its first singular batch/element. Float32/64 provide full-factor VJP,
JVP and reverse differentiation through both JVP channels. Input/output saved
values retain native alias mutation guards; an unneeded direction source is not
guarded or copied. No factor, inverse or gradient is cached across calls.

The JVP preserves native PyTorch's ordered factor/inverse expression. For a
non-diagonal factor this can differ from the mathematical inverse-system JVP;
interchanging them would change compatibility, including its reverse graph.
Independent closed-form and native-reference tests distinguish these contracts.
Complex64/128 inference requires real factor diagonals. Non-real diagonals do not
form Cholesky factors and native POTRI results there vary with LAPACK blocking;
they raise an explicit unsupported error here. Half, complex AD and output
buffers retain explicit unsupported errors. CPU and WASM choose measured
factor-inversion paths; performance and adoption are tracked in the
[0.36 validation report](../../docs/validation/torch-runtime-iteration-36.md).

Factorization failures are checked across numerical runtimes. Cholesky verifies
its returned diagonal because some WASM NumPy builds return NaN factors without
raising. General solve/inverse use explicit singular-pivot diagnostics when the
runtime cannot report floating-point error flags; empty-RHS solves also check
the coefficient. Triangular solves retain their documented nonfinite-result
behavior for zero pivots. Native linalg makes no general cross-backend guarantee
for NaN inputs; StrideTorch consistently rejects a non-positive/NaN Cholesky
factor diagonal while continuing to ignore unused input triangles.

## Matrix inverses and forward AD

`torch.inverse(input)`, `torch.linalg.inv(A)` and `Tensor.inverse()` support
real float32/64 matrices and complex64/128 forward evaluation. The existing
`torch.linalg.inv(input=...)` spelling remains available. Outputs match native
column-major final axes, including batches and empty matrices. Use `reshape`
when flattening a layout that cannot be represented by `view`. General inverses
and JVP final products generate transposed output buffers directly, avoiding a
second full-matrix layout copy.

Real inputs support `torch.func.jvp` and reverse differentiation through both
returned channels. The inverse is computed once and reused by both reverse
paths. Ordinary inverse backward saves its output, so changing the original
input is allowed; changing the inverse result is guarded. Singleton inversion
uses the dtype reciprocal, avoiding unnecessary factorization and preserving
native extreme-scale outcomes. Finite-input restrictions, singular errors and
unsupported output buffers remain explicit; complex AD requires the server.

The JVP follows native tangent feature-stride copying: a direction with
different strides is copied when reverse propagation needs its values.
Otherwise it is protected in place. The returned primal's hooks and retained
gradient are separate from the private inverse used by the tangent graph.
Matrix multiplication now protects only operands needed by active derivatives;
its output is not saved by its own pullbacks.

## Stable log determinants

`torch.slogdet(input)`, `torch.linalg.slogdet(A)` and `Tensor.slogdet()` return
a `(sign, logabsdet)` named tuple. `torch.logdet(input)` and `Tensor.logdet()`
return the logarithm directly. They use a log factorization rather than taking
the logarithm of an already overflowed or underflowed determinant. Square
matrices, batches, strided views and empty matrices are supported.

Ordinary inputs reuse the log factorization's results without constructing
unused NaN/negative corrections. Single-matrix predicates avoid redundant
NumPy reductions. Special-value handling, saved-value guards and gradient
contracts retain the same validated behavior.

Float32/64 support first-order reverse gradients and real JVPs. Real slogdet's
sign is a differentiable graph output with an unused input-gradient edge:
backward through sign alone leaves the input gradient `None`. Both outputs
share saved-graph lifetime and the sign's mutation guard. Negative real logdet
returns NaN with zero derivative, matching the validated native behavior.
Singular log derivatives are undefined; the implementation preserves the
tested IEEE Inf/NaN results without replacing zero pivots. Complex64/128 support
forward evaluation only; singular complex logdet preserves its `-Inf` real
component and NaN imaginary component.

Reverse differentiation through these JVPs follows the validated PyTorch 2.14.1
saved-factor contract: it differentiates the direction and has no reverse edge
to the original matrix. This does not promise arbitrary higher derivatives or
parity with future native versions. Complex AD and output buffers remain
explicitly unsupported.

Ordinary invertible `det` backward uses a linear solve instead of an SVD.
For 1×1 matrices, the value is copied directly and the derivative is exactly
the incoming seed, including extreme finite scales and signed zero. Raw VJPs
retain native seed-view layouts; stored leaf gradients still own their buffers.
Factorization arithmetic is isolated from NumPy's floating-point error policy
on runtimes that report IEEE status flags; the caller's policy and callback
are restored. This also prevents valid finite matrices from raising a spurious
NumPy floating-point error.
Other singular, underflowed and overflowed determinants retain the existing
polynomial-cofactor path. That legacy singular-gradient contract can differ
from native PyTorch's first backward. `det` still requires finite real input.

`torch.func.jvp` supports `torch.det`, `torch.linalg.det` and `Tensor.det()`
for real float32/64 matrices, including batches and empty matrices. Both JVP
channels and reverse differentiation through them follow the validated native
PyTorch 2.14.1 saved-LU contract. Zero LU pivots use the native epsilon rule;
at singularity this JVP can be zero even when the polynomial differential is
not. The ordinary `det` API's existing cofactor extension remains separate.
The returned primal is an alias of the private determinant result: its hooks
and retained gradient are independent of the tangent's use of that private
result. Primal data mutations still trigger saved-value guards. Direction and
tangent mutations follow the validated native behavior. The implementation
shares one immutable matrix snapshot between the two reverse paths, releases
the solve workspace after forming the trace, and skips unrequested pullbacks.

## Gradient accumulation and storage

The first `.grad` for a non-overlapping dense leaf follows the parameter's
strides, including transposes, dense permutations and singleton/empty layouts.
Sliced layouts with gaps and overlapping/broadcast leaves receive C-contiguous
gradients. Existing leaf gradients retain their object, storage and layout during
accumulation; resetting `.grad = None` selects the first-allocation rule again.

Native accumulation can adopt different positive strides on singleton axes,
which do not change any addressable value. StrideTorch keeps its owned-buffer
allocation rule; meaningful non-singleton strides follow the same layout contract.

Gradient allocations own their bytes: changing a supplied backward seed or a
hook's returned tensor after backward cannot change the stored gradient.
Post-accumulation hooks observe the allocated layout. `retain_grad()` on a
non-leaf instead copies its first gradient in C order and replaces the gradient
object on subsequent accumulation, preserving previously held snapshots.
`autograd.grad` results keep their operator's derivative layout; this rule governs
stored `.grad` buffers rather than forcing every returned derivative to match a
parameter. Sparse/complex gradients and general higher-order graph construction
remain outside the supported contract.

## Indexing and value selection

`torch.index_select`, `torch.index_add` and `torch.scatter_add`, and their Tensor
methods, support dense computation and real analytic gradients. Float32/64 JVP,
tangent reverse and functional Jacobian/HVP are covered. Repeated half indices
follow the tested native CPU accumulation routes; int32/int64 indices are
accepted. Int32 scatter follows PyTorch 2.14 behavior; PyTorch 2.6 rejects it.

```python
features = torch.tensor([[1., 2.], [3., 4.], [5., 6.]], requires_grad=True)
destinations = torch.tensor([0, 0, 1])
grouped = torch.zeros(2, 2).index_add(0, destinations, features)
assert grouped.tolist() == [[4., 6.], [5., 6.]]
grouped.square().sum().backward()
```

`index_add_` and `scatter_add_` preserve destination storage and aliases for
non-differentiable updates, including valid `no_grad` leaf updates. They check
overlap and invalid indices before writing. Differentiable mutation, complex
autograd and output buffers remain explicit unsupported cases. Pullbacks retain
the index/source version guards needed by their active inputs rather than
keeping intermediate value arrays alive. Empty/scalar scatter gradients reduce
to the source shape where native autograd permits it.

`torch.gather` and `Tensor.gather` support dense analytic gradients, float32/64
JVP and reverse differentiation of the tangent. Repeated half gradients
accumulate in float32 before rounding. Int32/int64 indices, scalar and strided
inputs and native empty-index behavior are covered. Sparse gradients and output
buffers remain unsupported; a sparse flag is accepted for an unrecorded forward.

`sort/topk` return cached namedtuple types with `values` and `indices`. Their
pullbacks retain selected indices and shape/dtype metadata, permit input/output
mutation, and guard index mutation. Gather retains the native input/index guards.
Neither rule retains intermediate input values solely for its derivative.
JVP preserves named outputs, including integer zero tangents for indices.

Small-K selection uses partitioning for sufficiently wide axes; other cases
retain full sorting. Narrow index outputs own compact storage instead of keeping
an entire permutation alive. Top-K tie identities are unspecified, as in native
Torch; derivatives follow the indices returned. Stable sorting preserves original
equal-key order. Supplying `stable` selects the native keyword-dimension overload;
explicit `dim=None`, bool dimensions and invalid flags are rejected.

## NaN-aware reductions

`Tensor.nansum/nanmean` and `torch.nansum/nanmean` support real inputs,
integer or multiple axes, `keepdim` and explicit output `dtype`. `nansum`
also accepts integer/boolean input; `nanmean` requires floating input even
with an explicit floating output dtype. Unlike ordinary `sum/mean`, these
two functions allow `keepdim` without specifying `dim`.

```python
x = torch.tensor([[1., float('nan'), 3.]], requires_grad=True)
torch.nanmean(x, dim=1).backward()
print(x.grad.tolist())  # [[0.5, 0.0, 0.5]]
```

Empty/all-NaN sums are zero; empty/all-NaN means are NaN. Reverse-mode
NaN masks multiply the seed, preserving IEEE behavior for infinite/NaN
seeds. Float32/float64 JVP and reverse differentiation of its tangent use
the separate native forward-AD mask rule, including exact zero primal
gradients for `nanmean`. In-place saved-input mutation is checked; changing
the reduction output does not invalidate its pullback. Complex CPU kernels,
half/complex JVP and output buffers retain explicit errors.

Half `nanmean` follows native's sum-then-divide composition: its intermediate
sum can overflow, unlike the opmath CPU `mean` below. See the
[native reduction implementation](https://github.com/pytorch/pytorch/blob/v2.14.1/aten/src/ATen/native/ReduceOps.cpp#L1411-L1429)
and [autograd definitions](https://github.com/pytorch/pytorch/blob/v2.14.1/tools/autograd/derivatives.yaml#L1549-L1551).

## Means and explicit dtype

`Tensor.mean` and `torch.mean` accept an explicit `dtype`, including floating
outputs for integer/boolean inputs. Integer axes, axis tuples/lists, scalar and
empty tensors, `keepdim` and empty-axis all-dimension reductions are supported.
Boolean/fractional axes, duplicates and out-of-range axes are rejected; passing
`keepdim` requires an explicit dimension, including `dim=None`. Ordinary
`sum` uses the same dimension and boolean `keepdim` checks.

```python
x = torch.tensor([[1., 3.], [5., 7.]], requires_grad=True)
y = torch.mean(x, dim=0, dtype=torch.float64)
y.sum().backward()
```

Half outputs use float32 sum/division before output rounding, matching
[native CPU mean](https://github.com/pytorch/pytorch/blob/v2.14.0/aten/src/ATen/native/ReduceOps.cpp#L1336).
This avoids overflowing the intermediate sum or denominator for large half
reductions. Pullbacks round in the output dtype before the input-gradient cast.
Float32/float64 JVP and reverse differentiation of its tangent support casts
between those dtypes. Half/complex JVP, complex autograd and output buffers retain
explicit unsupported errors. Existing complex forward means remain supported.

## Statistics

`Tensor.var/std`, `torch.var/std` and `torch.var_mean/std_mean` support real
floating inputs, scalar and multi-axis reductions, `keepdim`, fractional
`correction` and the legacy boolean `unbiased` overload. Paired operations reuse
the moments calculation. Their float32 accumulator follows the tested native CPU
paired behavior, which can round differently from separately requested moments.

The rules cover empty inputs, insufficient degrees of freedom, constant standard
deviations, reverse gradients and JVP for float32/float64. Native comparisons also
cover reverse differentiation of the JVP and storage/dtype conversion after the
forward pass. Complex statistics and output buffers retain explicit unsupported
errors. Floating reductions are tolerance-checked, not promised bitwise identical
across platforms.

## NumPy imports and Tensor construction

`torch.tensor(array)` copies storage. `torch.from_numpy(array)` and same-dtype
`torch.as_tensor(array)` share it, including positive-stride non-contiguous,
Fortran, broadcast and unaligned views. A dtype-changing `as_tensor` copies.
`owner.new_tensor(data)` copies into an independent leaf using the owner's dtype
unless explicitly overridden. Tensor inputs are detached when copied.

These four factory entry points reject external NumPy string/object/date/record
and extended-precision dtypes, foreign byte order, negative strides and strides
that are not whole multiples of the element size, before a cast/copy can hide
the invalid input. Read-only imports warn once; copying `tensor`/`new_tensor`
does not warn. Imported and exported arrays share data while keeping independent
shape/dtype/stride/writeability descriptors. Writes through read-only backing storage remain outside the
supported mutation contract. The NumPy-oriented `Tensor` class constructor and
internal views retain their existing layout conveniences; it is not the legacy
PyTorch positional shape allocator. Storage itself must have a supported numeric
or boolean dtype. Numeric storage acceptance does not promise every kernel or
autograd rule for every dtype; the computational bounds below still apply.

Builtin scalar defaults start at float32/int64/bool. Out-of-range Python integers
fail when constructing default int64 tensors; weak integer operands of existing
integer tensors follow the receiving dtype's arithmetic instead. Factory casts
isolate NumPy floating-point error settings and restore the caller's settings.
As in native Torch, out-of-range ndarray floating-to-integer result values are
platform dependent. Sequence scalar storage instead checks integer range and
Python numeric protocols, following native construction rather than array casts.

### Default floating dtype (0.55)

`torch.get_default_dtype()` and `torch.set_default_dtype(torch.float64)` control
Python floating/complex inference, default floating factories, wrapped scalar
promotion, integer true division and transcendental outputs, and subsequent
neural-network parameter/buffer initialization. The initial default is float32;
float16, float32 and float64 dtype constants are accepted. Invalid settings fail
without changing the previous default. Integer/boolean construction and explicit
dtypes remain independent. NumPy inputs preserve their own dtype; like/new
factories inherit the existing tensor's dtype. The NumPy-oriented raw `Tensor`
class constructor keeps its existing NumPy storage inference conventions.

Existing tensors, models, saved forward/reverse/JVP tapes and checkpoint dtype
metadata retain their dtypes. CPU Adam/AdamW initialize scalar step counters in
float64 when the current default is float64, otherwise float32; tensor-valued
step counters preserve their checkpoint dtype on restore.

The setting is shared by namespaces from the same installed distribution,
including `as_torch()` aliases, and is not thread-local. The public package and
PaperLeet's private copy keep independent settings. Changing it never changes an
installed native PyTorch default. Leaving `as_torch()` restores imports, not the
floating dtype setting.

Real float16 random factories now follow the PyTorch 2.14.1 CPU opmath sampling
profile, including cached scalar normals, incomplete vector tails and uniform
values rounded to the upper endpoint. Stream-position checks include subsequent
float64/float32 draws. Samples are compared within dtype-appropriate tolerances;
cross-platform bitwise identity is not promised. This adds sampling/storage
support; float16 neural-network kernels and forward AD remain unsupported.
Parameterless normalization and pretrained embeddings still accept existing
float32/64 inputs or weights when an unused default is float16.
Inferred complex32 storage is unavailable in NumPy and fails explicitly. Explicit
float32/64 or complex64/128 operands can still absorb a complex wrapped scalar;
native `full` uses complex64 for complex fills with a float16 default.

### Triangular distance tiles (0.60)

Larger `pdist` forwards pack only diagonal tiles; off-diagonal tiles keep efficient
broadcast differences. Every computed pair belongs to the condensed output.
Forward uses a separate two-work-array model within the existing 8 MiB workspace
budget. Backward retains its eight-work-array block plan and accumulation order.
Feature chunk boundaries are unchanged. The 0.59 medium-input fast path remains.
This trades a larger bounded forward working set for fewer tile operations;
saved tape state and required gradient allocations follow the existing contract.

### Condensed distance forward scheduling (0.59)

Medium `pdist` inputs calculate only the packed upper-triangle pairs when all
forward work fits the existing 8 MiB workspace model. Backward still uses bounded
block reductions, avoiding scatter accumulation over every packed pair. Larger
requests retain the tiled path. Pair indices and differences are temporary;
saved state and mutation guards follow the same contract as 0.58.

### Direct pairwise distances (0.58)

`torch.cdist`, `torch.pdist` and `torch.nn.functional.pdist` support nonnegative
Minkowski orders, including zero, fractional orders and infinity, for float32/64.
Cdist broadcasts batch dimensions; pdist returns the condensed upper triangle.
Analytical first-order pullbacks share pair-direction work and accumulate
broadcast contributions without allocating expanded input gradients. Equal
coordinates and infinity-norm ties follow the native CPU subgradient conventions.
Other forward/higher derivative modes remain explicitly unsupported.

Large problems use bounded point/batch/feature tiles instead of retaining a
point-pair-by-feature difference tensor. Small problems bypass tile scheduling
when the same workspace bound proves a direct calculation fits. Saved state
contains required input arrays, output distances and metadata; non-leaf Tensor
objects can be collected, and saved arrays are released with the graph.
Original-input and saved-distance version guards remain active. No result or
cross-backward gradient cache is introduced.

The private workspace planner budgets 8 MiB for work arrays; this excludes input,
output, requested gradients and general runtime overhead and is not a process
RSS limit. Outputs are capped at 16 million elements and pair-feature work at
512 million. This permits feature-heavy workloads rejected by the former full
pair-feature allocation limit, while oversized requests fail before allocation.

Cdist validates all three native compute-mode names and preserves the library's
stable direct-difference numerical policy. It does not implement the native
matrix-multiplication rounding path or its AD dispatch. Half/complex kernels,
output buffers and GPU execution are outside this contract. Long fractional
norm gradients can be sensitive to reduction order; per-element native checks
and separately labeled scalar FP64 qualifications are recorded independently.

### Bounded integer ranges (0.57)

`arange` avoids repeated generic scalar parsing for ordinary Python integer
controls defining `0, 1, ..., n-1`, when `0 <= n <= 16,000,000`.
All native positional/keyword overloads retain argument binding, dtype/device,
option and gradient-leaf validation. Each call owns fresh contiguous storage.
Float32/64 and integer unit indices need no floating error-mode context;
float16 retains its scoped context for overflow-to-infinity storage.
Other ranges, scalar subclasses, NumPy scalars and Tensor/forward-AD controls
continue through the full existing conversion path. No value-dependent result
or argument-object cache is added.

### Sequence factories (0.56)

`arange`, `linspace` and `logspace` share bounded scalar parsing and preserve
factory leaf semantics: Tensor endpoints are scalar controls, not reverse-AD
parents. Ordinary Tensor controls requiring gradients are rejected by `arange`,
matching native scalar conversion. `arange` supports native positional/keyword overloads and
`requires_grad=True` for floating outputs. Its size uses the original bounds,
while integer output values use truncated start/step. Incorrect step direction,
zero/NaN steps and non-finite endpoints fail explicitly. Native 2.14.1 empty
floating ranges with an infinite step are preserved. Scalar-control arange JVP
is supported; Tensor-endpoint linspace/logspace JVP remains unsupported, matching
native 2.14.1. Constant factories inside JVP still produce zero tangents.

Integer linspace casts endpoints and truncates interpolated samples toward zero;
NumPy's integer linspace floor rule is not used. Finite linspace values are
evaluated from both ends using one owned working array. Float16 intermediates
are float32, following the current upstream CPU opmath design; this deliberately
avoids native 2.14.1's half-step overflow on otherwise finite wide intervals.
Floating values are tolerance-checked rather than promised bitwise identical
across CPU vector widths or fused arithmetic implementations.
Four fixed unit intervals with at most 256 elements use a 64-entry immutable
byte-template cache (at most 128 KiB of numeric bytes plus Python metadata).
Each result receives a writable independent copy; arbitrary endpoint values or
Tensor objects are not retained in this cache.

`logspace` supports real bases and real/complex floating, signed integer and
uint8 outputs. Zero/one-step behavior follows native scalar fill semantics, including
half opmath conversion and the difference between singleton and multi-element
integer exponent conversion. Complex non-finite components and zero bases are
checked separately. Boolean and wider unsigned storage support zero/one-step
linspace/logspace but have no multi-element generation kernel.

Every result owns independent contiguous storage, including empty stride `(1,)`.
The limit is 16 million output elements; output buffers, pinned memory and
non-default layouts remain unsupported. Explicit dtype/device and invalid
argument checks happen before allocation. Native integer overflow results
outside representable conversion ranges remain platform-dependent.

Sequences recursively infer supported Python/NumPy scalar and Tensor dtypes.
NumPy float64 and small-integer values retain their precision and widths:

```python
import numpy as np

torch.tensor([np.float64(1 + 2**-40)]).dtype  # torch.float64
torch.tensor([np.int16(2), np.int16(3)]).dtype  # torch.int16
torch.promote_types(torch.int64, torch.float16)  # torch.float16
```

`as_tensor` of a Python sequence owns a copy. Tensor elements are single-value
leaves; constructing from them copies and detaches. `new_tensor` uses its owner's
dtype unless overridden. Nested arrays are materialized as sequence values;
direct ndarray imports retain the stricter layout checks described above.
Native scalar conversion also distinguishes NumPy integer objects from Python
integers for uint64 storage: an inferred NumPy uint64 scalar raises TypeError,
while a direct uint64 ndarray import or explicit uint64 Python integers work.

`promote_types` follows supported Torch category/width rules, including integer
to floating promotion. Unsupported wider-unsigned integer/complex combinations
fail explicitly. Common dtype aliases (`half`, `short`, `char`, `cfloat`,
`cdouble`, and the ordinary unsigned widths) are available; storage aliases do
not extend every kernel or differentiation rule to those dtypes.

Small homogeneous builtin float/int/bool sequences avoid intermediate arrays.
Primitive vectors/rows use batch inference and checking; other nested or mixed
inputs retain ordered recursive inference and scalar storage. Global default
dtype changes, broader array-protocol inference and the full legacy Tensor
constructor remain separate compatibility work. Default floats remain float32.

## Storage views and layout queries

`Tensor.view(*shape)` and `view(size=shape)` create a storage-sharing view;
`view_as(other)` uses only the other Tensor's shape. Incompatible transposed,
stepped or broadcast chunks raise instead of silently copying. Use `reshape`
when a copy is acceptable. One inferred dimension (`-1`), scalar/empty shapes
and singleton axes follow the validated native shape-only view rules.

`stride()` returns the actual underlying strides in element units; `stride(dim)`
supports negative dimensions. `element_size()` reports bytes per element.
Views/detached aliases share storage versions, including NumPy's stride wrappers,
so alias writes still invalidate upstream saved values. General computed layouts
can differ from native Torch; these queries describe the NumPy backing storage.
Fresh empty factories use conventional contiguous strides while imported NumPy
arrays retain their original metadata.

Shape-only views retain reverse gradients and JVP reverse tapes. A JVP can copy
its tangent into the primal's nonoverlapping layout before viewing; compatible
matching layouts share storage. Materializing a differently laid-out tangent
into potentially overlapping or negative-stride storage remains explicitly
unsupported. This does not add dtype-reinterpreting views, arbitrary as_strided,
storage-offset queries, channels-last execution or general differentiable
in-place mutation. Read-only backing storage retains its existing write boundary.
See the [0.20 validation record](../../docs/validation/torch-runtime-iteration-20.md)
for source identities, native/scalar contracts and measured compatibility costs.

The 0.21 candidate also validates dimension types, ranges and duplicate axes for
`flatten`, `permute`, `transpose`, `unsqueeze`, `squeeze` and `expand`, including
their functional entries. Scalar/empty shapes and singleton strides retain their
native contracts. A flatten of one axis returns its input; squeezing a
non-singleton dimension returns a distinct view. Shape JVPs preserve the tested
primal/tangent layout rules; empty tangents retain their independent strides.

Views created under `no_grad` can report `requires_grad=True` without having a
reverse edge. Graphless views follow subsequent base flag changes, while explicit
`view.requires_grad_()` promotes an independent leaf accumulator. Mutating the
base flag does not reconnect a previously graphless view. These rules are checked
against native PyTorch; they do not establish every native view-mutation rule.
Direct `view.requires_grad = True` also promotes a leaf. Attribute assignment
requires a bool and a leaf; `requires_grad_(True)` may retain a non-leaf's flag,
while disabling it remains invalid even after its backward graph is released.
The candidate's measured costs and pending integration gates are in
the [0.21 record](../../docs/validation/torch-runtime-iteration-21.md).

## Selection and joining tensors

`torch.result_type` follows the category priority of dimensioned Tensors,
zero-dimensional Tensors and scalar Number arguments. `where` uses these rules
instead of NumPy's integer-to-float widening; `cat`/`stack` promote all input
Tensor dtypes, preserving complex Tensor values:

```python
x = torch.tensor([1., 2.], dtype=torch.float32)
y = torch.tensor([3, 4], dtype=torch.int64)
mask = torch.tensor([True, False])
assert torch.result_type(x, y) == torch.float32
assert torch.where(mask, x, y).dtype == torch.float32
assert torch.where(mask, x, 0).dtype == torch.float32
indices = torch.where(mask)  # tuple of int64 index Tensors
joined = torch.concat([x, y])  # torch.cat / torch.concatenate aliases
```

`Tensor.where`, `cat`/`stack`'s `axis` alias and native argument keywords are
supported. Empty one-dimensional cat inputs may be skipped for shape purposes
while contributing dtype and receiving empty gradients. `out=None` works for
three-argument selection/joins; real output buffers remain unsupported. NumPy
has no native complex32 storage, so promotion needing ComplexHalf fails
explicitly. Complex storage support does not add complex reverse-mode gradients.

Number argument parsing follows native protocols rather than Tensor constructor
inference. In particular NumPy bool/complex64 Numbers use their float protocol
(the latter warns and drops imaginary parts); complex Tensor inputs preserve
both components. Scalar overflow checks apply before JVP execution as well.
Boolean conditions are supported; deprecated uint8 selection conditions warn once.
One-argument `where` on wide unsigned conditions remains unsupported.

Where backward retains only its condition and shape metadata, releasing branch
Tensor storage when other graph operators do not need it. Mutating a condition
invalidates backward, while mutating a leaf branch or where output does not
invalidate where's derivative. Upstream operators retain their own mutation
rules. Saved-state violations use the library's typed unsupported-operation
policy. Default and retained backward, mixed-dtype/broadcast gradients and the
validated forward-mode subset are covered by independent CPU/WASM contracts.

## Core arithmetic

`torch.add/sub/mul/div`, their ordinary Tensor methods and
`subtract/multiply/divide/true_divide` aliases accept Tensor or Number operands.
`add` and `sub` support checked `alpha`; `div` accepts `rounding_mode="floor"`
or `"trunc"`. Promotion respects dimensioned Tensor, scalar Tensor and Number
priority rather than NumPy's integer-to-float widening. Integer true division
produces float32, while rounded int64 division keeps integer precision.

```python
x = torch.tensor([3., -3.], requires_grad=True)
y = torch.add(x, 2., alpha=0.5)
assert y.tolist() == [4., -2.]
assert torch.div(x, 2., rounding_mode="floor").tolist() == [1., -2.]
y.sum().backward()
assert x.grad.tolist() == [1., 1.]
```

`torch.reciprocal(x)` and `x.reciprocal()` preserve floating/complex storage;
integer and boolean inputs produce float32. At complex zero, the library returns
a real infinity retaining the real zero's sign and a NaN imaginary component.
Native Torch's real infinity sign at this singularity differs between CPU
platforms; this signed policy is the library's portable convention. Real
reciprocal signed zeros and all nonzero complex inputs retain their existing
validation. Python `number / tensor` follows
native reciprocal-then-multiply order, while functional `torch.div(number,
tensor)` performs direct division. This distinction matters at overflow,
underflow and signed-zero boundaries. Rounded division has zero derivatives;
ordinary arithmetic and real reciprocal support the validated reverse/forward
AD subset. Complex reverse graphs, ComplexHalf and wider-unsigned kernels that
have not been validated fail explicitly. `out=None` works; actual output buffers
remain unsupported.

Arithmetic kernels, reverse accumulation and gradient-seed casts isolate NumPy
floating-point error settings, returning IEEE non-finite values as native Torch
does. User Tensor hooks run with the caller's NumPy policy; hook changes to
`seterr`/`seterrcall` and hook exceptions persist through the reverse traversal.
This does not add floating-point trapping to WASM: its behavior depends on the
underlying NumPy runtime.

## Models and in-memory checkpoints

`nn.Module` registers parameters, child modules and buffers, with deduplicated
`named_parameters()`, `named_buffers()` and `named_modules()` traversal. Shared
layers remain shared. `train()`, `eval()`, `requires_grad_()`, `zero_grad()` and
`apply()` recurse through registered children. `Sequential`, `ModuleList`,
`ModuleDict`, `ParameterList` and `ParameterDict` support model composition.

```python
import copy

model.eval()
best = copy.deepcopy(model.state_dict())
# Continue training, then restore into the same parameter objects:
model.load_state_dict(best)
model.train()
```

`state_dict()` returns shallow tensor references by default; `deepcopy` takes an
independent snapshot while preserving buffer views. Persistent buffers, shared
parameter paths, metadata and subclass extra state are included. Loading copies
values into existing parameters, preserving optimizer references, and reports
missing/unexpected keys. Shape/type errors still fail with `strict=False`.

File checkpoint data can use `torch.save/load`, described below. Backward/global
hooks and `load_state_dict(assign=True)` remain unsupported. `ModuleList.insert` supports indices from zero through its length;
insertion into a named `Sequential` is unsupported. Copying a model clears copied
parameter gradients, as native PyTorch does. Copying non-leaf tensors or exotic
non-dense NumPy backing allocations fails explicitly.

## File checkpoints

`torch.save` and `torch.load` (also `torch.serialization.save/load`) exchange CPU
checkpoint data with native PyTorch's modern ZIP format. NumPy remains the only
third-party runtime dependency. Paths and seekable binary streams work in CPython;
Pyodide paths refer to its virtual filesystem. For browser downloads/uploads,
pass bytes through a `BytesIO` buffer and the host's file interface.

```python
import io

stream = io.BytesIO()
torch.save({"model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict()}, stream)
stream.seek(0)
checkpoint = torch.load(stream, weights_only=True, map_location="cpu")
# Construct the destination model, optimizer and scheduler first.
model.load_state_dict(checkpoint["model"])
scheduler.load_state_dict(checkpoint["scheduler"])
optimizer.load_state_dict(checkpoint["optimizer"])
```

The data subset includes ordinary Python scalars/containers, `OrderedDict`
metadata, tensor dtype/device tokens, dense tensors and `nn.Parameter`. It retains
repeated tensor identity, shared storage, offsets, nonnegative strides, empty and
expanded views, and `requires_grad`. Loaded tensors are fresh leaves; gradients,
graphs and executable hooks are not serialized. Supported tensor dtypes are bool,
signed 8/16/32/64-bit integers, unsigned 8/16/32/64-bit integers, float16/32/64 and
complex64/128. Untyped storage covers the wider unsigned types. Mixed-dtype views
of one buffer and negative tensor strides are unsupported.

The default writer uses pickle protocol2 for native weights-only interoperability.
The reader accepts protocols2–5; native weights-only readers may reject higher
protocol opcodes. The loader always uses a fixed data-global allowlist and rejects
pickle extensions, including cached host extensions. It supports `weights_only`
`None`/`True`; executable object checkpoints and `weights_only=False` fail
explicitly. Whole modules, arbitrary classes/tensor subclasses/custom tensor
attributes, TorchScript, legacy non-ZIP files, mmap, custom pickle modules, lazy
conjugate metadata and callable storage mapping are outside this contract.

Storage loads on CPU. CUDA-tagged data requires an explicit CPU `map_location`
(string, device or string mapping); this does not implement CUDA execution. Typed
storage handles byte-order conversion; non-native-endian untyped storage is
unsupported. Native/StrideTorch model, AdamW and StepLR state exchange is tested
by resuming actual updates in both directions.

## Local model callbacks

Ordinary Python attribute checks work on models: absent attributes raise
`AttributeError`, so `hasattr(module, "weight")` and `getattr(module, "weight", None)`
can drive generic initialization. Registered parameters, buffers and children
appear in `dir(module)`; computed submodule properties work with `get_submodule`.
Known unavailable execution methods still fail with `UnsupportedOperation` when
called.

`register_forward_pre_hook` and `register_forward_hook` support ordered callbacks,
`prepend`, `with_kwargs`, input/output replacements and removable handles.
Forward hooks can opt into `always_call=True` for exception cleanup; cleanup
failures warn while preserving the original exception. Direct `model.forward(...)`
bypasses callbacks. Replacements use ordinary Tensor operations and keep their
normal reverse/forward AD rules.

```python
with layer.register_forward_hook(lambda module, args, output: output * 0.5):
    prediction = layer(x)
# The context removes the callback, including on exceptions.
```

Public `register_state_dict_pre_hook`, `register_state_dict_post_hook`,
`register_load_state_dict_pre_hook` and `register_load_state_dict_post_hook`
support checkpoint transformation, local metadata and shared missing/unexpected
key lists. Post-hooks modify state/key lists in place and must return `None`.
Load pre-hooks receive the copied state mapping; shallow Tensor storage retains
the existing checkpoint sharing rules. Register callbacks again when rebuilding
a model: they are Python behavior and are not serialized in `state_dict`.

Handles are also available as `torch.utils.hooks.RemovableHandle`. Removal is
idempotent; handles weakly reference callback registries and do not retain the
model. Deep-copying a model with a handle attribute connects the copied handle to
its copied registrations. Callback dictionaries allocate lazily; models without
callbacks use a fast call path, restored after the first call following removal.
Module backward hooks, process-global registration hooks, private checkpoint-hook
interfaces, compilation and native file serialization remain unsupported.

## Reverse graphs and requested gradients

`Tensor.backward`, `torch.autograd.backward` and `torch.autograd.grad` honor
`retain_graph`. A default reverse pass frees the saved numerical state of
executed operations. Keep the graph explicitly when another pass needs that
state; reusing freed state raises RuntimeError. Operations with no native saved
Tensor requirement, such as addition, basic slicing and shape/reduction paths,
can still repeat without retention. This policy is operator-specific: even
multiplication by a constant can require saved state.

```python
loss.backward(retain_graph=True)
# Inspect or accumulate a second pass:
loss.backward()

gradients = torch.autograd.grad(
    (first_loss, second_loss), dict(model.named_parameters()),
    allow_unused=True,
)
```

Flat multiple outputs, explicit matching gradient seeds, duplicate roots/inputs,
unused inputs and named input dictionaries are supported. `autograd.grad` returns
values without accumulating leaf `.grad`; `materialize_grads=True` returns zeros
for unused inputs, following the validated Torch 2.14 `create_graph=False` rule.
Use `intermediate.retain_grad()` to accumulate a non-leaf `.grad`, including
during `autograd.grad`. `backward(inputs=...)` limits accumulation to selected
inputs. It still executes a selected non-leaf's gradient function, while `grad`
can stop there; native-reference tests cover the distinction after graph release.

Graph history remains non-leaf after release. Lightweight graph metadata retains
parent relationships without retaining every intermediate Tensor's storage;
saved VJPs release their captured arrays after execution. Explicit retention
keeps that state. Ordinary gradient buffers preserve their existing identities
and mutation guards. Integer scalar selections also preserve zero-dimensional
view storage and shared version counters.

General higher-order reverse graphs (`create_graph=True`), batched reverse seeds
and arbitrary GradientEdge inputs remain unsupported. Validated forward AD and
reverse-over-forward rules remain available. `retain_grad` inside an active
forward AD transform is unsupported; import aliases and numerical state remain
process-wide.

## Tensor gradient callbacks and copying

`Tensor.register_hook` observes or replaces a computed gradient before retained
or leaf `.grad` accumulation. Return a matching-shape/dtype Tensor or `None`.
Callbacks run in registration order; each phase snapshots its callbacks. They
also run for requested inputs of `autograd.grad`, which does not accumulate leaf
gradients. Handles use the same weak, removable/context-manager interface as
model callbacks.

```python
with layer.weight.register_hook(lambda gradient: gradient.clamp(-1., 1.)):
    loss.backward()
    optimizer.step()
```

`register_post_accumulate_grad_hook` accepts leaf tensors and runs after their
`.grad` updates during backward. It receives the parameter, may update/clear its
gradient or perform an inline update, and must return `None`. It does not run in
`autograd.grad`. Callback execution starts in no-grad mode; use `enable_grad`
explicitly when needed. Mode changes are scoped to that reverse node, including
on exceptions. A hook directly returned at a requested gradient preserves its
Tensor object and any explicitly constructed graph. General higher-order reverse
execution remains unsupported.

Ordinary graphs allocate no callback dictionaries. Non-leaf callback state is
created only for retention or registration, weakly references the Tensor wrapper,
and keeps hooks attached to history after that wrapper is collected. Copying a
Tensor/Parameter drops its Tensor callbacks. Existing leaf hooks participate in
reverse-over-forward; registration inside an active forward transform is
unsupported. Module/global backward and raw graph-node hooks remain separate work.

`target.copy_(source, non_blocking=False)` copies Tensor values into existing CPU
storage, preserving target shape/dtype/identity and sharing with views. Boolean/numeric source
broadcasting and casts are supported; non-numeric storage fails before mutation. Use `no_grad` for parameter initialization
and copies involving tensors that require gradients. Shared version counters are
updated; existing saved-state mutation guards still apply. Exact self/view copies
are accepted, while partial source/destination overlap and expanded destinations
with repeated elements fail. Complex-to-real copies warn once; complex-to-bool
keeps ordinary complex truth values. CPU copies are synchronous even when
`non_blocking=True`. Differentiable in-place copy graphs, copies inside forward AD
and mutation of read-only backing allocations remain unsupported.

## CPU model conversion and gradient modes

Existing CPU selection and dtype-conversion code works with the import aliases:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device=device, dtype=torch.float64)
x = x.to(device=device, dtype=torch.float64)
with torch.no_grad():
    prediction = model(x)
```

`torch.cuda.is_available()` is false for this NumPy backend. Device objects can
describe other devices, but execution on CUDA/MPS/meta remains unsupported.
CPU index metadata is accepted; actual tensors report the logical `cpu` device.
`Module.to`, `.cpu`, `.float`, `.double` and `.type` recurse through registered
children, preserving Parameter objects and existing gradient objects so an
already-created optimizer retains its parameter references. `.to` casts floating
and complex buffers while preserving integer/bool buffers; `.type` also casts
integer buffers. Differentiable model conversions support float32/float64.
Existing optimizer state is not converted by model conversion: restore a state
dictionary into a matching optimizer when its algorithm needs matching dtypes.

`Tensor.to` accepts dtype/device/other-Tensor overloads and `copy`; no-op dtype
conversions, `.cpu()` and already-contiguous `.contiguous()` preserve identity.
`clone`/`to` support preserve/contiguous memory formats; channels-last formats
remain unsupported. CPU-index conversions force a copy, matching native Torch.
Use `.contiguous()` or `to(copy=True, memory_format=torch.contiguous_format)`
to pack a transposed tensor; a same-dtype/device `.to` can return it unchanged.

`no_grad`, `enable_grad` and `set_grad_enabled` support contexts and decorators,
including recursive calls and generators. Generator execution sets the requested
mode only while it runs, restoring the caller's mode between yields and on errors.
Factory `requires_grad` arguments keep their native exception to `no_grad`.
Gradient mode belongs to the isolated runtime; it is not thread-local.

Convert models before building a training graph. Native backward behavior after
conversion depends on which inputs, views or outputs each operator saves; the
covered cases retain those distinctions, including mixed-dtype errors. This is
not a guarantee for arbitrary live graphs. Version guards remain linked across
storage rebinding and reject later in-place mutations. Private `_apply` callbacks
must preserve shapes; non-leaf gradient storage rebinding remains unsupported.

## Neural-network building blocks

In addition to linear/activation/container modules, the library provides
`LayerNorm`, `RMSNorm`, `Embedding`, `GELU`, `Dropout`, `Identity`, `Flatten`,
`Tanh`, `Sigmoid`, `MSELoss`, `SmoothL1Loss`, `CrossEntropyLoss` and `NLLLoss`.
Standard class submodule aliases such as `torch.nn.parameter` and
`torch.nn.modules.normalization`, `.activation`, `.sparse` and `.loss` are provided.

```python
from torch import nn

model = nn.Sequential(
    nn.LayerNorm(16), nn.Linear(16, 32), nn.GELU(approximate="tanh"),
    nn.Dropout(0.1), nn.Linear(32, 4),
)
loss_fn = nn.MSELoss()
```

Normalization supports multidimensional trailing shapes, optional affine
parameters, resets, empty inputs and float32/float64 analytical gradients. Its
JVP and reverse-over-forward paths include parameter tangents; LayerNorm follows
native cached-statistic reverse behavior. Ill-conditioned float32 LayerNorm rows
use float64 centering/variance before returning float32. This avoids cancellation
at large offsets, but results can differ from native CPU float32 accumulation on
those rows. Native CPU results themselves depend on reduction order; exact
bitwise equality on ill-conditioned inputs is not promised.

GELU offers exact `math.erf` and explicit `approximate="tanh"` paths, with VJP/JVP
and curvature rules. The exact path calls scalar erf through NumPy and has a
different cost from the array-only tanh path; no approximation is silently used.
Dropout follows recursive `train()`/`eval()`, scales kept values by `1/(1-p)` and
shares its mask across primal/tangent/backward computations. Disabled, p=0, p=1
and empty cases do not draw random values. Masks follow the Bernoulli distribution
and repeat under this library's seed; their bit stream is not native Torch's
dropout stream. In-place dropout is limited to frozen tensors.

Embedding registers a float32/float64 weight, initializes normal samples and a
zero padding row, supports frozen/shared `from_pretrained` weights, frequency
scaling, dense gradients and max-norm renormalization. Pretrained padding values
are preserved, matching native Torch. Max-norm updates only selected rows under
no-grad and maintains mutation guards; renormalization during JVP and sparse
gradients remain unsupported. Ordinary embedding weight JVP is supported.

Loss modules retain their functional boundaries: matching-shape/dtype MSE and
smooth L1, weighted/spatial cross entropy and negative log likelihood. Legacy loss options,
half/complex neural-network kernels and arbitrary live graphs through storage
rebinding remain outside this contract. Normalization/GELU reject backward after
storage rebinding explicitly; convert model storage before constructing a graph.

## Classification losses

`nn.CrossEntropyLoss` / `nn.functional.cross_entropy` accept unbatched `[C]`,
`[N,C]` and spatial `[N,C,...]` real float32/float64 inputs. Int64 class targets
support `ignore_index`, class weights and label smoothing; mean reduction divides
by the selected valid class-weight sum. Floating probability targets have the
input shape, may be unnormalized, and support target/weight gradients, mixed
float32/float64 promotion and smoothing; their mean divides by the number of
samples/spatial positions. Native-compatible nonfinite values, zero/negative
weights, all-ignored and empty inputs are checked separately.

`nn.NLLLoss` / `nn.functional.nll_loss` expose the weighted class-index linear
loss without applying log-softmax. Both loss modules register class weights as
buffers, with standard module aliases, dtype conversion and in-memory state.
Class-index weights are nondifferentiable, including during JVP. CE and NLL
input JVPs, and probability-label CE JVPs across all three operands, support
reverse-over-forward. Zero-class probability JVP and legacy `size_average` /
`reduce` options remain explicitly unsupported. Saved target/weight mutations
are checked according to the active native tape; CE input/output mutations do
not invalidate the loss itself, while NLL saves its input version.



## Bounded zero-padding for regular convolution (0.49)

Regular Conv1d/2d/3d can gather actual samples without materializing the full
zero-padded activation. Clipping geometry and gathered sampling are shared with transposed convolution;
its ordinary forward scatter stays inline. Auto keeps the existing padded path for
modest spans; an excessive padded span selects packed gathering, or per-tap
execution when a full gather would exceed its 32 MiB logical-array limit.
Selection considers both a 32 MiB padded-span threshold and the ratio to actual
sampled patches. A bounded 64-entry cache stores immutable geometry decisions,
including workspace limits, and never stores tensor/seed values. Existing reflect/replicate/circular padding paths are unchanged.

Forward, input/weight/bias VJPs and bilinear JVPs retain their supported dtype,
grouping, asymmetric-same, stride and dilation behavior. Bounded input gradients
scatter directly into the real input shape. Expanded gathered patches are not
retained on the backward tape; weight gradients regather from saved input data.
Version guards and bias-only object release remain in effect. Implicit padded
zeros preserve nonfinite products such as `0 * Inf`; finite tap paths skip only
mathematically zero regions. These are per-buffer selection estimates and
allocation policies, not a guarantee of process/BLAS peak memory. Reflect,
replicate and circular padding do not gain these zero-padding bounds.


## Transposed convolution

`nn.ConvTranspose1d/2d/3d`, `nn.functional.conv_transpose1d/2d/3d` and their
`torch.conv_transpose1d/2d/3d` aliases support dense real float32/float64 on CPU
and Pyodide WASM. Batched/unbatched inputs, groups, per-axis stride/padding/dilation,
output padding, bias, module `output_size`, state dictionaries and dtype conversion
are supported. Ordinary `torch.nn.modules.conv` imports expose the modules and
`_ConvTransposeNd`/`_ConvNd` bases. Weight layout is `(in_channels,
out_channels // groups, *kernel)`; reset bounds use output channels per group.

Direct clipped scatter and bounded matrix contraction share analytical input,
weight and bias VJPs and bilinear JVPs. Reverse differentiation of JVPs retains
both bilinear terms, including explicit zero tangents and nonfinite behavior.
Input and weight versions are saved even for bias-only tapes; detached bias or
output edits do not invalidate this operation's own backward. Empty batch
outputs and tangents have native conventional strides; zero channels and
nonempty zero-spatial inputs are rejected. Batched 2D channels-last input and
2D channels-last weight layouts select the checked native output layout.

Output padding must be smaller than either stride or dilation on each axis;
it can exceed stride when dilation is larger. Module `output_size` accepts the
spatial sizes or a full output shape and resolves only the native stride-sized
range. Only zero padding mode is supported. Constructor padding strings retain
native metadata but are invalid for forward execution.

Optional matrix workspace uses a conservative 32 MiB logical-array estimate,
not measured process peak memory. Large theoretical padding does not allocate
that full activation span: matrix backward gathers into packed order, and the
direct fallback evaluates tap slices. Implicit padded zeros retain native
`0 * Inf` backward behavior. CPU/WASM float16/complex convolution, general
non-leaf mutation, storage rebinding across saved graphs and GPU execution are
outside this version's validated support.


## Instance normalization

`nn.InstanceNorm1d`, `nn.InstanceNorm2d`, `nn.InstanceNorm3d`,
`nn.functional.instance_norm` and the low-level `torch.instance_norm` entry
support dense float32/float64 on NumPy CPU and Pyodide WASM. Modules accept
batched and unbatched inputs, optional affine parameters (`bias=False` included),
train/eval, running statistics, dtype conversion and version-2 checkpoints.
Ordinary `torch.nn.modules.instancenorm` imports and `_InstanceNorm`/`_NormBase`
base aliases work. BatchNorm also accepts the keyword-only `bias` option.

Training normalizes each sample/channel over its spatial axes. Running mean
and unbiased running variance are averaged over samples. Unlike BatchNorm,
`momentum=None` freezes the running values, `num_batches_tracked` stays zero,
and tracked training JVP is supported. Statistics buffers are nondifferentiable;
their dual tangents are ignored. Automatic training updates increase each
running buffer's version, including empty inputs. An empty batch produces NaN
running statistics; an empty spatial extent preserves their values. Native
empty-batch/channel affine errors and the functional spatial-singleton check
are retained.

Analytical input/weight/bias VJPs, JVPs and saved-statistic reverse-over-forward
rules avoid the temporary repeated parameters and graph nodes of a BatchNorm
composition. Affine parameters and evaluation statistics use forward snapshots,
so subsequent mutations preserve the saved backward values. Contiguous input
storage is version-guarded; noncontiguous input is copied before normalization.
Detached output mutations do not invalidate this operator's backward. General
non-leaf in-place edits and float16/complex normalization remain unsupported.


## Group and batch normalization

`nn.GroupNorm` and `nn.functional.group_norm` normalize each sample independently
across channel groups and spatial dimensions. Standard
`torch.nn.modules.normalization.GroupNorm` imports work. Float32/float64 inputs,
optional per-channel affine parameters, non-contiguous arrays, empty batches,
analytical VJPs and JVPs are supported. Train/eval produce the same behavior;
there are no running buffers. Module conversion, reset and in-memory checkpoints
use the common model lifecycle.

GroupNorm's JVP statistics remain differentiable on the reverse tape, matching
native PyTorch; this differs from LayerNorm/BatchNorm's saved-statistic rules.
Input/weight mutations invalidate saved graphs; bias/output changes do not.
Ill-conditioned float32 groups use selective float64 centering, following the
LayerNorm numerical policy. Centered formulas avoid native CPU affine-expansion
cancellation, so tiny groups and large offsets need not be bitwise identical to
native float32. Zero channels, empty spatial dimensions and singleton validation
follow the checked boundaries; float16/complex normalization is unsupported.

`nn.BatchNorm1d`, `nn.BatchNorm2d`, `nn.BatchNorm3d` and
`nn.functional.batch_norm` support dense real float32/float64 inputs. Ordinary
`torch.nn.modules.batchnorm` imports and its `_BatchNorm` base alias work too.

```python
layer = torch.nn.BatchNorm1d(128, momentum=0.1)
normalized = layer(torch.randn(32, 128))
layer.eval()  # Subsequent calls use the saved running statistics.
```

Training normalizes with biased batch variance and updates running variance
with the unbiased estimate. `momentum=None` uses a cumulative average. Affine
parameters, untracked statistics, train/eval changes, resets, dtype conversion,
version-2 checkpoints and legacy missing batch counters are supported. Empty
inputs preserve running statistics and yield zero affine gradients; a rejected
single-observation training call still counts as an attempted batch, as in the
native module. Functional evaluation requires both running buffers.

Input/weight/bias VJPs and JVPs are checked independently, including native
saved-statistic reverse-over-forward behavior. Running buffers are
nondifferentiable: their forward tangents are ignored and strict independence
checks remain effective. Functional training updates their primal storage.
Tracked training modules reject captured batch-counter mutation during JVP
before changing state; use evaluation or untracked modules for that transform.
Automatic running-statistic updates preserve native storage-version behavior;
explicit writes to saved input/weight/statistic storage invalidate backward.

Float32 output uses the centered normalization expression. Finite channels
with a large mean relative to their variance are selectively recomputed with
float64 statistics; inverse scale is rounded once into the input dtype.
This improves agreement with high-precision centered mathematics, and can
differ from native CPU's expanded affine expression on ill-conditioned input.
Ordinary conditioned inputs retain strict native-reference comparisons. Storage-
dtype reduction overflow and non-finite inputs can still yield non-finite
results; this is not a bitwise equivalence promise across numerical backends.
See the [native CPU transformation](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cpu/batch_norm_kernel.cpp)
and the independent numerical/gradient contracts in `tests/test_batchnorm.py`.
The [0.16 validation record](../../docs/validation/torch-runtime-iteration-16.md)
records supported behavior, complete release gates and measured costs, including
the eval-backward tradeoff.

## Convolution

`nn.Conv1d`, `Conv2d`, `Conv3d` and their `nn.functional.conv1d/2d/3d`
functions support real float32/float64, batched and unbatched inputs, groups
(including depthwise), stride, dilation, integer/tuple padding and `same`/`valid`.
Modules also support `reflect`, `replicate` and `circular` padding. Parameters,
dtype conversion and in-memory checkpoints use the common Module lifecycle.
Standard imports such as `from torch.nn.modules.conv import Conv2d` work.

```python
model = nn.Sequential(nn.Conv2d(3, 8, 3, padding="same"), nn.SiLU())
x = torch.randn(2, 3, 16, 16, requires_grad=True)
model(x).square().mean().backward()
```

Analytical input/weight/bias gradients and bilinear JVPs include validated
reverse-over-forward tapes. Saved input and weight versions are guarded even
on a bias-only tape. Empty batches/channels and nonfinite values have separate
contracts. `same` padding requires unit stride; reflection padding must be
smaller than its axis and circular padding cannot wrap more than once.

Larger finite workloads use packed NumPy matrix products when the estimated
logical workspace fits 32 MiB. Small/pointwise, nonfinite and larger-workspace
cases use direct contractions. Backward repacks inputs instead of retaining
expanded im2col activations. The estimate is a dispatch policy, not a peak-memory
limit for NumPy or BLAS. Transposed convolution, complex/float16 kernels, GPU
execution and general higher-order reverse AD remain outside this contract.
The [0.19 validation record](../../docs/validation/torch-runtime-iteration-19.md)
separates native/scalar correctness, CPU/WASM timing and application regression.

## Optimizers and training checkpoints

`torch.optim.Optimizer`, `SGD`, `Adam` and `AdamW` share parameter groups,
`add_param_group`, gradient zeroing and `state_dict`/`load_state_dict`. Adam and
AdamW support AMSGrad, maximization, per-group decay and lazily initialized
moments; parameters without gradients acquire no state. SGD supports momentum,
dampening and Nesterov. Group learning rates can be updated between steps.

```python
optimizer = torch.optim.AdamW([
    {"params": [layer.weight], "lr": 0.01},
    {"params": [layer.bias], "lr": 0.02, "weight_decay": 0},
], amsgrad=True)

checkpoint = copy.deepcopy({
    "model": layer.state_dict(),
    "optimizer": optimizer.state_dict(),
})
# Restore after constructing a matching model and optimizer:
layer.load_state_dict(checkpoint["model"])
optimizer.load_state_dict(checkpoint["optimizer"])
```

Optimizer states map to parameters by group/order, including optional saved
names; names do not change the mapping. Loading preserves the current parameter
objects and casts moment tensors to their dtype. State dictionaries are shallow:
use `deepcopy` for a snapshot. Same-dtype loaded state tensors share storage with
the supplied checkpoint, matching native PyTorch. Group count/size mismatches
fail before replacing current groups. `zero_grad(set_to_none=False)` reuses
existing gradient buffers and preserves absent gradients. Optional step closures
run with gradients enabled and return their result, even inside `no_grad`.

Standard submodule imports such as `from torch.optim.adam import Adam` and
`from torch.optim.optimizer import Optimizer` work under the host aliases. The
original adapter's single-group `lr`/`params` access remains available; use
`param_groups` and `state` for new code. Optimizer hooks, enabled foreach/fused/
capture/differentiable execution, trainable hyperparameters and zeroing non-leaf
gradient buffers remain unsupported. CPU scalar Tensor learning rates are supported.

## Learning-rate schedules

Ordinary `from torch.optim.lr_scheduler import ...` imports work with the aliases.
The supported schedules are `StepLR`, `MultiStepLR`, `ExponentialLR`,
`CosineAnnealingLR`, `LinearLR`, `ConstantLR`, `PolynomialLR`, `LambdaLR`,
`MultiplicativeLR`, `SequentialLR`, `ChainedScheduler` and `ReduceLROnPlateau`.
`LRScheduler` supports custom subclasses, including iterable learning-rate results.

```python
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

scheduler = SequentialLR(optimizer, [
    LinearLR(optimizer, start_factor=0.2, total_iters=5),
    CosineAnnealingLR(optimizer, T_max=95),
], milestones=[5])

# Each epoch: compute loss and gradients, then:
optimizer.step()
scheduler.step()
```

Construction applies the initial schedule immediately; call subsequent `step()`
after the optimizer update. `get_last_lr()` reports the applied rates. Plateau
scheduling uses `scheduler.step(validation_loss)` with min/max, relative/absolute
thresholds, patience, cooldown, per-group minima and epsilon behavior. It cannot
be a child of sequential/chained schedules because it needs a metric. Some nested
compositions have no native closed-form transition and raise `NotImplementedError`,
as native Torch does. Cyclic/one-cycle/warm-restart schedules are not provided yet.

Save a deep copy of model, optimizer **and scheduler** state for a training
snapshot. After constructing the new model, optimizer and scheduler, load model
state, scheduler state and then optimizer state. Scheduler loading retains the
destination optimizer and restores counters/child states; it does not restore
the optimizer's current learning rates. Lambda functions must be reconstructed in
code; callable objects save their attribute dictionaries, not their executable
code. The supported state dictionaries can also be stored in native-compatible
ZIP `.pt` files with `torch.save/load`.

Frozen CPU scalar Tensor learning rates retain their group object identity while
updates fill their values; initial/base/reported rate tensors are independent
copies. Current-rate schedules can follow dynamically added optimizer groups.
Base-rate/lambda/floor metadata may need explicit adjustment when groups change;
construct all groups before the scheduler when possible. Explicit epoch stepping
is deprecated and emits a warning, matching native Torch.
Differentiating schedule hyperparameters, including passing a forward-AD learning
rate through scheduler construction, remains explicitly unsupported.

## Modules and compatibility

The implementation is divided into tensor/storage, analytical autodiff and JVP,
operators, RNG, neural network functions/modules, optimizers, linear algebra,
operator capability registration, and explicit import integration. The original
bounded schedules profile remains separate so unsupported autograd still fails.

`torch._browser_capabilities` is an immutable mapping of registered functions to
reverse-mode policy, forward-mode support, unsupported keywords and signature.
It describes entry-point validation, not every dtype/shape combination or every
Tensor method. Unsupported paths raise `UnsupportedOperation` (also retained as
`BrowserUnsupported` for PaperLeet). There is no native PyTorch fallback inside
the library; the host decides whether to rerun work on a server.

This is a teaching subset, not a full PyTorch implementation: CPU/NumPy only,
bounded allocations, validated float32/float64 differentiation, selected frozen
complex operations, and selected forward-mode rules. CUDA, general higher-order
reverse-mode differentiation, arbitrary `torch` APIs, and unsupported in-place
graphs remain outside the contract. Existing error messages mentioning a server
are retained for compatibility with PaperLeet's fallback diagnostics.

## Pyodide and offline hosting

Install the pure-Python wheel with NumPy in a compatible Pyodide runtime, or use
`stridetorch.runtime.source_files()` to include the library's Python modules
in a static runtime bundle. PaperLeet exports those sources once per backend,
under its existing content-addressed runtime assets. The independent wheel has
only a NumPy runtime dependency. It contains no problem catalog, judge, pytest,
PyTorch binaries, or browser application code.

## Experimental browser GPU integration

The installed distribution includes the explicit
`stridetorch.experimental.webgpu` namespace. It provides `MatmulOffload`,
`ResidentMatmulOffload`, `Matrix`, `MatrixProgram` and `ProgramResult`; callers no
longer need Python files from the benchmark directory. Importing this namespace
works in CPython without Pyodide and leaves native `torch` imports untouched.
Actual browser transport requires Pyodide and a compatible attached host client.

Export the installed JS service modules and notices into one adjacent host
directory:

```python
from pathlib import Path
from stridetorch.experimental.webgpu import service_sources

assets = Path('webgpu-assets')
assets.mkdir(exist_ok=True)
for name, source in service_sources().items():
    (assets / name).write_text(source, encoding='utf-8')
```

The host creates a module Worker from `gpu-service.mjs`, delivers a MessagePort
to it with `{type: 'connect'}`, and sends the other port to the Python Worker.
There, import `MatrixRpc` from `rpc-client.mjs`, construct it with that port,
await `client.ready` and set `self.matrix_rpc = client`. Enter Python through
`runPythonAsync` or another JSPI-compatible entry point. The host owns Worker
shutdown and its application isolation boundary.

After that explicit setup:

```python
from stridetorch import torch
from stridetorch.experimental.webgpu import ResidentMatmulOffload

with ResidentMatmulOffload() as offload:
    a = torch.ones(256, 256, dtype=torch.float32, requires_grad=True)
    b = torch.ones(256, 256, dtype=torch.float32, requires_grad=True)
    output = offload(a, b)
    output.square().mean().backward()
```

Values and gradients remain CPU Tensors. Eligible finite float32 2D products
and ordinary first-order pullbacks use the bounded GPU service. Unsupported
inputs, `no_grad` and AD transforms retain their CPU paths. Transport errors
propagate. Protocol/capability checks refuse unavailable input retention before
upload; they do not silently retry failed GPU work on CPU. The resident cache
owns at most eight buffers and 8 MiB of logical payload by default. Close the
session explicitly or use its context manager.

`MatmulOffload(client=...)` and `ResidentMatmulOffload(client=...)` borrow an
explicit host client. A resident session fixes its default client at construction;
a stateless offload fixes it at its first GPU submission. Replacing the host's
global client does not retarget an existing offload. These wrappers release their
own buffers and leave client/Worker shutdown to the host.

For offline Pyodide hosts, `stridetorch.experimental.webgpu.source_files()`
returns the complete optional package payload, including the JS assets and
redistribution notices. The normal `stridetorch.runtime.source_files()` export
remains unchanged and excludes this opt-in integration. This is an experimental
API, not an enabled torch device or automatic `torch.matmul` replacement.

See [the packaging qualification](../../docs/validation/torch-webgpu-package.md)
for installed wheel/sdist and real-browser checks. The repository's old Python
benchmark import names and JS module URLs remain compatibility entries when
the package/source tree is available.

## Development

From the PaperLeet repository:

```sh
python -m pip install -e .
python -m pip install -e ./packages/stridetorch
python -m pytest packages/stridetorch/tests tests/test_browser_*.py -q
paperleet export
cd web
npm run check
npm run build
npm run check:browser-wasm
```

The main PaperLeet distribution includes this same source tree under the private
`paperleet._vendor.torch` namespace, so current `pip install .` users do not need
an unpublished dependency or a sibling checkout. The two distributions own
separate files: uninstalling the standalone library cannot remove PaperLeet's
bundled runtime, and installing it cannot silently change PaperLeet's backend.
The subproject builds separately with `pip wheel --no-deps ./packages/stridetorch`.
`paperleet.browser.numpy_autograd` and `schedules_numpy` retain compatibility
facades; source changes should be made in this library.

The 0.13.1 patch uses NumPy contractions for linear-layer pullbacks on the pinned
WASM backend, retaining the existing forward calculation and native NumPy/BLAS
path. Three alternating fresh-process Node/WASM measurements of the unchanged
2,000-step Flow Matching lesson reduced median full-suite execution from 33.90
seconds to 23.87 seconds (29.6%). All six runs produced identical recorded loss
curves, MMD and coverage. This is a local workload measurement, not a latency
promise for other devices or browser downloads. The
[measurement record](benchmarks/results/linear-wasm-0.13.1.json) identifies the
source, runtime, samples and scope. Empty/strided shapes, independent scalar and
native gradients, repeated backward seeds, JVP and mutation guards are covered.

The 0.15 release candidate extends direct WASM contractions to measured large
linear forward cases, preserving CPU BLAS and small/strided-input paths. Four
fresh-process measurements of the unchanged complete Flow Matching suite in
each runtime reduced median execution time by 27.0% on Node/WASM and 21.1% in
Chrome Workers versus 0.14. All eight suites passed. Floating-point reduction
order changes slightly; mode coverage/outlier proportions were identical and
the maximum sample-coordinate difference in the recorded Node pair was
`1.79e-6`. These local measurements do not guarantee remote browser deadlines.
The [0.15 validation record](../../docs/validation/torch-runtime-iteration-15.md)
and [raw measurements](benchmarks/results/iteration-15.json) retain samples,
source identities, numerical differences and current release gates.

The [0.14 validation record](../../docs/validation/torch-runtime-iteration-14.md)
tracks arithmetic promotion, native numerical order, callback policies and
separately preserved optimization phases, including the successful native/WASM
quality gate, deployment and live runtime verification.

The [0.13 validation record](../../docs/validation/torch-runtime-iteration-13.md)
tracks mixed-dtype selection/joins, where graph retention and CPU/WASM/Chrome
performance, including unresolved process variance. The
[0.12 record](../../docs/validation/torch-runtime-iteration-12.md) covers sequence
inference and checked scalar storage, and the
[0.11 record](../../docs/validation/torch-runtime-iteration-11.md) covers NumPy
import validation and metadata isolation.
See the [0.10 validation record](../../docs/validation/torch-runtime-iteration-10.md)
for Tensor gradient callbacks, CPU copies, complete training and runtime costs.
The [0.9 validation record](../../docs/validation/torch-runtime-iteration-9.md)
for graph lifetime, requested gradients, native retention policies and memory/latency tradeoffs.
The [0.8 validation record](../../docs/validation/torch-runtime-iteration-8.md)
for model introspection, local callbacks, checkpoint transforms and dispatch costs.
The [0.7 validation record](../../docs/validation/torch-runtime-iteration-7.md)
for neural-network layers, normalization accuracy, encoder training and memory.
The [0.6 validation record](../../docs/validation/torch-runtime-iteration-6.md)
for scheduling, training continuation and CPU/WASM overhead measurements.
The [0.5 validation record](../../docs/validation/torch-runtime-iteration-5.md)
for model conversion, gradient modes, storage guards and CPU/WASM measurements.
The [0.4 validation record](../../docs/validation/torch-runtime-iteration-4.md)
for optimizer/checkpoint parity, lazy-state memory, wheel isolation and real WASM/
Chrome measurements. The [0.3 record](../../docs/validation/torch-runtime-iteration-3.md)
retains model lifecycle results; the [0.2 record](../../docs/validation/torch-runtime-iteration-2.md)
retains attention, graph and import-alias results.
The [extraction record](../../docs/validation/lightweight-torch-extraction.md)
documents the original baseline.

Ongoing development and release gates are tracked in [ROADMAP.md](ROADMAP.md).
The [benchmark tools](benchmarks/README.md) cover CPU/WASM kernels and an
experimental WebGPU matmul probe. The probe is separate from the supported
CPU/NumPy torch runtime.

## License

This standalone library is licensed under [Apache-2.0](LICENSE); attribution is
in [NOTICE](NOTICE). Installed packages and `runtime.source_files()` include both
files, including the smaller schedules profile. PaperLeet's private bundled copy
carries the same notices. This license applies to `packages/stridetorch/`;
it does not set a license for the rest of the application or its dependencies.


### Math attention, grouped heads and forward AD (0.50)

`torch.nn.functional.scaled_dot_product_attention` supports float32/float64
math attention, boolean/additive broadcast masks, causal masking, custom scales,
GQA, dropout, analytical reverse AD and `torch.func.jvp`. K and V head counts may
differ; each must divide the query head count. Grouped contractions broadcast
K/V heads inside their query groups instead of materializing repeated K/V arrays.
The ordinary path keeps a shared fused NumPy pullback. Q/K are scaled before the
dot product, avoiding premature overflow in tested large finite float32/float64
cases, including negative and zero scales. A bounded 64-entry cache
stores immutable shape checks only, with no tensor, activation or RNG state.

Both primal and tangent channels support reverse differentiation, including mask
primals/tangents and shared operands. Fully masked rows have zero output and
zero derivatives through mixed AD. This stable behavior differs from NaNs in
PyTorch 2.14.1 math SDPA's fully-masked mixed-AD corner; independent finite
differences and a native compositional safe-softmax reference check that rule.
Ordinary values, JVPs and first derivatives are also compared against native SDPA.

Dropout is applied whenever `dropout_p > 0`, including evaluation code; pass zero
explicitly for evaluation. Each invocation samples one Bernoulli multiplier shared
by the primal, tangent and retained backward traversals. It follows this library's
RNG stream and distribution, not native Torch's seeded dropout bit stream. Valid
probabilities are in `[0,1]`; negative and NaN values are rejected. A causal flag
and explicit mask cannot be combined. This is CPU/Pyodide math execution, with
no FlashAttention, GPU or native PyTorch binary dependency.


### Attention validation and retained-tape buffers (0.51)

Attention's bounded 64-entry validation cache stores shapes, canonical dtype
strings and configuration only. It does not retain NumPy dtype metadata, tensor
objects, activations, gradients or RNG state. Scalar options and tensor types are
checked on every call; matching shapes with new values do not reuse results.
Single K/V heads use ordinary matrix broadcasting in both regular and forward AD.
Balanced signed Q/K prescaling and the existing fully-masked-row rule remain.

The ordinary fused VJP saves values only for the gradients that existed when the
graph was built. With attention dropout and all gradients active, it retains two
score-shaped arrays (probabilities and one multiplier) instead of three; it
regenerates their product only for the V pullback. A V-only tape retains just that
product and releases unneeded scaled Q/K arrays from the pullback. Active leaf
storage can still remain live for accumulation. A Q-only or K-only tape saves
only the needed opposite scaled operand and V values. Repeated backwards use the
original multiplier without consuming random draws. Saved-version guards still
cover original inputs and masks.

These are retained logical buffer counts, not process peak-memory measurements.
The compositional JVP/mixed-AD tape does not gain that two-buffer bound. Independent
scalar math, weak-reference lifetime checks, dtype-metadata collection and cache
warm-up tests validate the selective rules. The benchmark includes actual-library
forward/reverse/JVP and complete encoder SGD flows, with and without dropout.

### Multihead attention (0.52)

`torch.nn.MultiheadAttention` and
`torch.nn.functional.multi_head_attention_forward` support float32/float64
CPU/Pyodide math execution, sequence-first and batch-first inputs, unbatched
inputs, separate K/V dimensions, boolean/additive attention and padding masks,
per-head or averaged weights, bias K/V tokens, zero-attention tokens and
functional static K/V. Self-attention shares one packed Q/K/V projection;
cross-attention with identical K/V shares their packed projection. Input and
parameter gradients, JVPs and reverse-over-JVP use the existing tensor engine.
Module state, hooks, dtype conversion and optimizer/checkpoint workflows apply.

```python
layer = torch.nn.MultiheadAttention(64, 4, batch_first=True, dropout=0.1)
x = torch.randn(2, 16, 64, requires_grad=True)
y, weights = layer(x, x, x, need_weights=False)  # weights is None
y.square().mean().backward()
```

MHA boolean masks use `True` for blocked positions. Evaluation disables the
module's attention dropout; training follows this library's reproducible stream,
not native Torch's seeded dropout bits. `is_causal=True` requires an explicit
mask, following native MHA's math fallback. With no padding mask and
`need_weights=False`, the causal hint selects SDPA's causal mask; callers must
supply a correct hint. With padding or returned weights, the explicit mask is
used. Returning weights requires the final additive mask dtype to match the
query; the SDPA path also accepts float32 masks for float64 queries.

Fully blocked rows follow the native math path: `need_weights=True` propagates
softmax NaNs; `need_weights=False` uses SDPA's zero attention result before the
output projection, so a nonzero output bias remains. This does not emulate
native fused inference kernels, nested tensors, GPU kernels or quantization.
Positive non-boolean integer dimensions and finite dropout in `[0,1]` are
validated eagerly, including during construction.

### Dense Transformer modules (0.53)

`nn.TransformerEncoderLayer`, `nn.TransformerDecoderLayer`,
`nn.TransformerEncoder`, `nn.TransformerDecoder` and `nn.Transformer` compose
the existing attention, Linear, LayerNorm and dropout implementations. They
support sequence-first/batch-first and unbatched inputs, pre/post normalization,
ReLU/GELU or callable activation, biasless/zero-width feedforward networks,
attention/padding masks, causal hints and automatic square-causal-mask detection,
optional final normalization, independent cloned layers and custom full-model
encoders/decoders. `Transformer.generate_square_subsequent_mask` creates a
boolean, floating or complex mask. MHA itself still requires real float32/64.

The encoder accepts `enable_nested_tensor` and `mask_check` as optimization
preferences but always sets `use_nested_tensor=False`: this backend executes
dense math. Native fused Transformer inference and actual NestedTensor inputs
are not implemented. Empty stacks construct but raise IndexError when forwarded,
matching the dense native path. Layer counts and feedforward widths must be
nonnegative non-boolean integers. Correct causal hints remain the caller's
responsibility; explicit false hints disable automatic mask detection.

State keys, independent layer/parameter cloning, initialization streams and
versions, hooks, dtype conversion, pickle/deepcopy, optimizer and scheduler
checkpoint continuation are verified. `Linear.reset_parameters()` is available;
Linear and LayerNorm constructors now perform genuine initialization writes.
Outputs and parameters retain their requested public dtype.

As of 0.54, Transformer layers also preserve directly supplied registered
operators, such as `torch.sin`, `torch.relu` and `torch.nn.functional.silu`,
through Python pickle and deepcopy. Pickle resolves these operators within
the owning distribution, without importing or replacing the host's `torch`.
Live execution keeps the original callable and its capability guards; deepcopy
keeps callable identity while independently cloning parameters. This applies
to direct operator activations, not arbitrary closures or nested callable
objects. Full-model Python pickle requires trusted input; it is separate from
the weights-only `torch.load` interface.

Float32 LayerNorm statistics accumulate in float64 before rounding back. For
finite inputs/results with finite epsilon at least float32 machine epsilon,
LayerNorm tangents and their reverse tape also use float64 intermediates before
returning float32. This reduces accumulated mixed-AD rounding in deep normalized
models; it uses additional temporary/intermediate storage. Tiny/non-positive
epsilon and non-finite cases use the existing tangent formula. The native-style
frozen mean/inverse convention for LayerNorm's mixed AD remains unchanged.

Native comparisons cover float32/64 outputs, first derivatives, JVPs and mixed
AD. Five float32 mixed arrays in the current deep-model corpus differ from native
elementwise tolerances; both backends are additionally checked against an exact-
input float64 oracle with max error normalized by max(1, oracle max magnitude)
at most 1e-5. Float64 mixed AD remains checked elementwise. This is numerical
compatibility with documented floating-point bounds, not bitwise equivalence.

Version 0.66 shares one upstream-gradient matrix product between inverse JVP
reverse branches on Pyodide/WASM when matrix cores are at least 32×32 and both
inputs require gradients. CPU and smaller/one-input cases keep the independent
reverse rules. Targeted gradient requests can prune unused shared branches;
repeated requests never reuse a cached upstream gradient. Saved-value mutation
guards remain in force even when a reverse branch is pruned.

Empty inverse-JVP directions keep the original direction's version guard,
including interleaved empty batches: native empty tangent canonicalization does
not create an independent saved copy. Joint reverse rules validate and route
returned gradients in one pass. Existing rules retain construction-time needs;
only explicitly validated rules opt into target-dependent pruning.


Version 0.67 releases consumed joint backward callbacks before predecessor hooks
run, so their saved payloads do not remain live through unrelated reverse work.
Retained graphs keep those callbacks until their final traversal. All-active
joint gradients use a direct validated routing path, while filtered/frozen-input
rules preserve the general route. Leaf traversal and callback-free joint release
avoid unnecessary Python work. Shared inverse-JVP reverse rules accumulate a
private matrix temporary in place instead of allocating a second sum buffer.
These changes preserve public APIs, target selection, hooks, repeated parents,
dtype/shape checks, storage guards and retained-backward behavior.

Version 0.68 clears traversal-local references to consumed joint cotangent arrays
before predecessor work. Retained callbacks and saved-tensor version checks keep
their existing lifetime rules; gradient values, hooks and target selection are
unchanged. This removes temporary retention without caching gradient results.


## Functional vector-Jacobian products

Version 0.69 adds `torch.autograd.functional.vjp(func, inputs, v=None,
create_graph=False, strict=False)` for the supported real first-order operators.
It calls `func` once with distinct input objects, preserves the caller's gradient
flags and existing `.grad` buffers, and returns detached outputs and derivatives.
A Tensor input produces a Tensor derivative; an input tuple produces a tuple.
Output tuples retain their structure. An omitted seed requires one output with
one element. Unused inputs receive zeros, or raise an error with `strict=True`.

```python
x = torch.tensor([2., 3.])
value, gradient = torch.autograd.functional.vjp(lambda a: a.square().sum(), x)
assert value.item() == 13.0
assert gradient.tolist() == [4.0, 6.0]
assert x.grad is None
```

Inside a recorded functional VJP callback, `numpy()` and `np.asarray` reject an
implicit graph escape. `detach().numpy()`, an explicit `no_grad()` scope, and
`numpy(force=True)` deliberately disconnect the value and can therefore produce
a zero derivative. On this CPU/NumPy backend, `force=True` exports the existing
array with its dtype, strides and shared storage; it implies no GPU transfer.
The flag is keyword-only and accepts a Python bool. Existing direct-adapter
exports outside this transform keep their earlier contract.

Higher-order `create_graph=True`, complex inputs and nesting inside forward AD
retain explicit unsupported errors. This does not add `torch.func.vjp` or a GPU
backend. Numerical kernels and the 0.68 cotangent lifetime improvements remain
in use.

Registered operator attributes on `nn.Module`, including functional VJP, now
round-trip through pickle using references to this distribution's operators.
Shallow and deep copies retain live callable identity; Transformer activation
restoration keeps its legacy defaults. Arbitrary user lambdas retain ordinary
Python pickle limitations.


## Functional Jacobians

Version 0.70 adds `torch.autograd.functional.jacobian(func, inputs,
create_graph=False, strict=False, vectorize=False, strategy='reverse-mode')`
for supported real first-order operators. It runs the callback once, isolates
each explicit input from repeated arguments and captured originals, and returns
detached Jacobians shaped as `output.shape + input.shape`, with each input's
dtype. Tensor/tuple nesting follows the input and output structure. Caller
gradient flags, existing `.grad` buffers and original tensor hooks are preserved.
The callback can update its own model buffers, as in native Torch; a training
BatchNorm callback runs once and updates its running statistics once.

```python
x = torch.tensor([2., 3.])
jacobian = torch.autograd.functional.jacobian(lambda a: a.square(), x)
assert jacobian.tolist() == [[4., 0.], [0., 6.]]
```

`strict=True` checks each output against each input, including outputs that are
independent of only one argument. Constant or unused paths produce zeros in
default mode. Implicit NumPy exports inside the recorded callback raise an
error; `detach()`, `no_grad()` and `numpy(force=True)` deliberately disconnect
the graph. Empty inputs with nonempty output can yield empty Jacobians; native
empty-output and empty-tuple error boundaries are preserved.

Dense Jacobians allocate one final buffer per input and fill gradient rows
directly. Their inherent output size still scales with output elements times
input elements. Higher-order graphs, complex inputs, vectorized/forward-mode
Jacobians and nesting in forward AD remain explicitly unsupported. This API
does not add `torch.func.jacrev`, `torch.func.jacfwd` or a GPU backend.


## Functional Hessian-vector products

Version 0.71 adds `torch.autograd.functional.hvp(func, inputs, v=None,
create_graph=False, strict=False)` for the supported real float32/float64
operators. The callback returns one Tensor with one element. The function
returns its detached value and a detached Hessian-vector product with the
input shapes and dtypes, preserving Tensor/tuple input structure. It executes
the callback once, isolates repeated arguments and captured originals, and
preserves caller gradient flags, existing gradient buffers and original hooks.
Vectors are detached and converted to their corresponding input dtypes.
An omitted vector is allowed only for one input Tensor containing one element.

```python
x = torch.tensor([2., 3.])
value, product = torch.autograd.functional.hvp(
    lambda a: a.pow(3).sum(), x, torch.ones_like(x))
assert value.item() == 35.
assert product.tolist() == [12., 18.]
```

The existing implementation combines a forward direction with a reverse pass,
without constructing a dense Hessian. Version 0.79's vector-norm graphs use
three reverse traversals when every active pullback has a recorded Tensor rule;
other existing graphs retain their previous method. Normalization statistics and matrix factors stay
differentiable on this HVP path; the existing native-compatible JVP rules keep
their saved-statistic/factor behavior. Training BatchNorm may update its own
buffers once. Determinants use an adjugate path near singularity and compute
its polynomial directional derivative without singular-value-gap divisions.
This deliberately returns finite mathematical curvature at repeated singular
values where native SVD-based higher-order tapes can return backend-dependent
NaNs. Empty determinant
backwards remain independent in strict mode.

Default mode returns zero products for unused or constant derivatives;
`strict=True` raises when a required derivative is independent. Implicit NumPy
exports reject, while `detach()`, `no_grad()` and `numpy(force=True)` deliberately
disconnect the graph. Higher-order output graphs, complex/other floating
inputs, nested forward transforms and unvalidated forward rules remain
explicitly unsupported. This API does not add a general Hessian API or a GPU
backend. Registered HVP operator attributes survive model copy and pickle.


Scalar clamp gradients and forward directions preserve the production Torch 2.6
inclusive-boundary contract, including `clamp_min` and `clamp_max`. This is checked
against an isolated native Torch 2.6.0 installation; native 2.14.1 uses a different
boundary convention.
The generic absolute-value JVP retains an explicit zero reverse derivative
for a differentiable primal; it is distinct from an unused input returning
`None` from `autograd.grad(..., allow_unused=True)`.


## Functional Jacobian-vector products

Version 0.72 adds `torch.autograd.functional.jvp(func, inputs, v=None,
create_graph=False, strict=False)` for the supported real float32/float64
operators. Inputs and outputs may be a Tensor or a flat tuple of Tensors.
Vectors match their corresponding input shapes and are converted to the input
dtypes. An omitted vector requires one input containing one element.

```python
x = torch.tensor([2., 3.], requires_grad=True)
value, direction = torch.autograd.functional.jvp(
    lambda a: a.pow(3), x, torch.ones_like(x), create_graph=True)
assert value.tolist() == [8., 27.]
assert direction.tolist() == [12., 27.]
curvature = torch.autograd.grad(direction.sum(), x)[0]
assert curvature.tolist() == [12., 18.]
```

Default mode returns detached values and directions. `create_graph=True`
allows ordinary reverse queries on the returned graphs. The callback runs
once, repeated inputs become distinct callback objects, and original gradient
flags, existing gradient buffers and hooks remain isolated from internal work.
Captured originals remain separate from the processed arguments. Reverse-based
functional JVP respects `no_grad()`, `detach()` and `numpy(force=True)` boundaries;
ordinary `torch.func.jvp` retains its forward-mode behavior.

Strict mode checks dependencies for every input and output, including zero
derivatives. With `create_graph=True`, a constant Jacobian must still have a
natural differentiable dependency; private vector leaves cannot satisfy that
check. Differentiable captured weights are valid dependencies. Empty sums and
empty determinants preserve their different native strict/graph behavior.

Normalization and matrix-factor graphs remain differentiable when graph mode
needs them. Default normalization JVP avoids building those higher-order graphs.
Training BatchNorm may update its own buffers once. Singular determinant JVPs
use cofactor directions, retaining polynomial curvature rather than adopting
native saved-LU zeros or SVD-tape NaNs. The validation record distinguishes
direct native comparisons from independently checked mathematical derivatives
for the documented small determinant and fused-normalization fixtures.

This API does not enable general `autograd.grad(create_graph=True)` support,
arbitrary higher-order reverse graphs, complex differentiation, nested forward
transforms or unvalidated operator JVP rules. Registering gradient hooks inside
the callback is explicitly unsupported: such hooks can change the reverse-based
JVP and cannot be silently evaluated as an ordinary forward direction.
The registered functional JVP operator survives model copy and pickle.


Version 0.73 preserves the functional-JVP API while sharing its validation
helpers and dtype descriptors across calls. Single-Tensor validation avoids a
temporary generator, and non-strict calls avoid allocating strict-mode tracking
sets. Private base-Tensor leaves receive their final gradient metadata at
construction; Tensor subclasses retain their `detach`/`requires_grad_` dispatch.
The existing strict checks, caller isolation, cast behavior and returned reverse
graphs remain covered by the SDK and numeric inventories. Performance results
are scoped to the workflows and runtimes recorded in the validation report.

Version 0.74 uses a direct NumPy contraction for measured large two-dimensional
float32/float64 matmul calls on Pyodide, including their two pullbacks. Native
CPU calls retain NumPy matmul. The WASM route requires both output dimensions
at least 32, a reduction dimension at least 64, at least 1,048,576 multiply-add
terms, and a captured left input with at least 4,096 elements. Vector, batch,
integer, complex, small and invalid-shape products retain the existing operation.
Nonfinite and exceptional finite ranges also fall back: nonzero input magnitudes
must lie between `sqrt(finfo.tiny)` and `sqrt(finfo.max / reduction_size) / 2`.
This protects exceptional product ranges; cancellation and floating-point
roundoff still depend on the backend. Output storage remains C-contiguous, and
opposite-operand saved-value guards retain their existing behavior.

The contraction keeps NumPy's traversal choice and normalizes the output layout
afterward. Explicit `einsum(order='C')` lost the measured WASM speedup even when
the output already had C layout. Temporary range-scan allocations add bounded
peak memory; input buffers are not cached. The independent native-Torch numeric
controls, warm-call A/A/B measurements, precision limits and application gates
are recorded in the [iteration 76 validation](../../docs/validation/torch-runtime-iteration-76.md).

Version 0.75 extends the measured WASM contraction path to batched and broadcast
matmul and to the numerical products inside fused SDPA. Native CPU retains
NumPy matmul. Batched cores require `M,N >= 32`, `K >= 64`, at least 262,144
multiply-add terms per core and at least 1,048,576 across broadcast batches.
The existing caller-size gate and exceptional-range fallback remain in place.
Ranks above 34 keep the original operation because the tested shape-only
broadcast helper has a lower rank limit than NumPy matmul.

Contiguous reduction rows avoid the slow traversal observed with unprepared
ellipsis contractions. Packing preserves shared zero-stride batch axes, and
range checks inspect unique values instead of expanding those repeats. The
output remains C-contiguous; packed data are temporary and are not retained
between calls. Grouped SDPA keeps its existing unrepeated K/V representation.
Packing increases temporary allocation, and small SDPA calls have measured
overhead; timing and lifetime controls are in the
[iteration 77 validation](../../docs/validation/torch-runtime-iteration-77.md).

Limits queries use canonical scalar types. A first call with a NumPy dtype
carrying arbitrary metadata previously allowed NumPy's global `finfo` cache
to retain that metadata after all tensors were released. Cold-cache lifetime
regressions cover both the existing 2D route and the new batched route.


### Strict matrix products (0.76)

`torch.mm(input, mat2)` and `input.mm(mat2)` accept exactly two matrices.
`torch.bmm(input, mat2)` and `input.bmm(mat2)` accept exactly two rank-three
batches with equal batch counts; they do not broadcast batch dimensions.
Both use the existing CPU/NumPy and qualified WASM matmul paths, including
float32/float64 reverse AD, JVP and functional HVP. Normal numeric dispatch
requires matching dtypes. Empty `bmm` follows the tested native Torch 2.6
result-dtype behavior, including its mixed-dtype empty path and original
saved-storage checks; `mm` still requires matching dtypes when empty.
Empty `bmm` reads shape/dtype metadata only: it does not cast either input,
including a nonempty operand paired with a zero output dimension.

Float16, supported integer and complex inputs retain the existing inference
boundaries. Nonempty bool and uint16/uint32/uint64 products are rejected, as
in the native CPU controls. `out=None` is accepted by top-level calls; a real
output buffer remains an explicit unsupported operation. Tensor methods do
not accept `out`. Existing torch aliases and Module operator serialization
also expose these names. See the [0.76 validation](../../docs/validation/torch-runtime-iteration-78.md)
for native, minimum-wheel and real WASM qualification.

### Vector and general matrix products (0.77)

`torch.dot(input, tensor)`, `torch.mv(input, vec)` and
`torch.matmul(input, other)` are available with corresponding Tensor methods.
`dot` takes two vectors; `mv` takes a matrix and a vector; `matmul` supports
vector promotion and broadcast matrix batches. Public functions require Tensor
operands and use native keyword names. Top-level `out=None` is accepted;
actual output buffers raise `UnsupportedOperation`. Tensor methods do not
accept `out`.

```python
x = torch.tensor([1., 2., 3.], requires_grad=True)
matrix = torch.arange(6, dtype=torch.float32).reshape(2, 3)
matrix.mv(x).sum().backward()
assert x.grad.tolist() == [3., 5., 7.]
assert torch.dot(input=x, tensor=x).item() == 14.
```

The real float32/float64 paths support reverse gradients, JVP and functional
HVP. Empty products follow the native Torch 2.6 shape, dtype and gradient
boundary for vector, folded and broadcast dispatch. These rules account for
layout and gradient flags. Mixed-dtype empty operations compute their zero
results and derivatives from metadata, avoiding operand casts and physical
broadcast expansion. Saved-storage checks account for whether the native
intermediate would share input storage. Empty outputs have contiguous element
strides. An integer or boolean result cannot silently discard an active
floating gradient.

Boolean and uint16/uint32/uint64 vector dots are rejected even when empty;
nonempty matrix products also reject those native CPU kernel dtypes. Existing
complex-autograd and nonleaf in-place mutation boundaries remain explicit.
Python `@` reflection returns `NotImplemented` for foreign operand types.
NumPy's reflected matmul inference can return a Tensor; it requires detached
inputs when gradients are enabled.

### Affine matrix and vector products (0.78 development candidate)

`torch.addmm(input, mat1, mat2, *, beta=1, alpha=1)` and
`torch.addmv(input, mat, vec, *, beta=1, alpha=1)` compute
`beta * input + alpha * product`, with corresponding Tensor methods.
The implementation performs the affine operation on private NumPy output
storage and records one analytical reverse node. Real float32/float64 paths
support reverse gradients, direct JVP, functional JVP and functional HVP.

```python
bias = torch.tensor([1., 2.], requires_grad=True)
matrix = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)
vector = torch.tensor([2., 1.], requires_grad=True)
result = torch.addmv(bias, matrix, vector, alpha=2, beta=.5)
assert result.tolist() == [8.5, 21.]
result.sum().backward()
assert vector.grad.tolist() == [8., 12.]
```

`beta=0` ignores nonfinite input-bias values. Scalar conversion and empty-input
behavior follow the pinned native Torch 2.6 controls. In particular, an empty
`addmv` with nonzero beta can retain the bias's original shape; this native
behavior does not guarantee a valid JVP for every broadcast bias. Half-precision
inference retains the tested native nonfinite-product behavior at `alpha=0`.
Complex reverse AD and gradients for active float16 inputs remain unsupported.
Top-level `out=None` is accepted. A Tensor output buffer reports the explicit
unsupported boundary; a non-Tensor output argument raises `TypeError`.

The candidate also preserves wider direct-forward tangents through scalar
addition and softmax, supports empty softmax/log-softmax inputs, and avoids
unused Linear inference graph setup and a separate bias-result allocation.
Reverse-based JVP also preserves cat's ignored one-dimensional empty-operand
boundary, including its detached zero Jacobian contraction.
The [iteration 80 work record](../../docs/validation/torch-runtime-iteration-80.md)
tracks CPU evidence and remaining browser, packaging and application gates.
This development version has not been adopted on main or published.

Partial direct JVP preserves the shape/dtype contribution of undefined tangents
and their zero reverse dependencies. Chained affine calls retain a wider
tangent. Integer affine results reject active floating gradients; `no_grad`
inference remains available. Empty matmul JVP retains its forward-selected
kernel's dtype rules. The [iteration 81 record](../../docs/validation/torch-runtime-iteration-81.md)
adds complete Node/Chrome SDK and isolated installed-wheel qualification.

### Vector norms (introduced in 0.79)

`torch.linalg.vector_norm(x, ord=2, dim=None, keepdim=False, *, dtype=None,
out=None)` performs direct NumPy reductions, with analytical real reverse AD,
direct JVP, functional JVP and functional HVP. It accepts selected or multiple
dimensions, empty axis lists, scalar/strided input, real orders including zero
and positive/negative infinity, and non-narrowing dtype conversion. Complex and
half inputs retain inference support and their existing autograd boundaries.
Non-None output buffers remain unsupported.

Constant sign derivatives preserve logical-zero cotangents separately from
numerical zero arrays and unused gradients. Validated rules propagate these
zeros while retaining mutation guards, hooks and graph-lifetime behavior;
undeclared rules materialize the cotangent at their boundary.

For supported vector-norm HVP graphs, private scoped Tensor pullbacks follow
native reverse evaluation order. Recorded rules cover elementwise arithmetic,
powers, reciprocal, exponential, sine/cosine, tanh, absolute value, ReLU, shape
changes and sums. The callback still executes once. Caller-owned tensor types
and gradient buffers remain unchanged; temporary graph owners are released on
success and failure. This does not enable public `autograd.grad(create_graph=True)`
or differentiable HVP outputs, and unrecorded existing graphs retain their old
HVP path. Version 0.80 has completed standalone CPU/WASM, minimum-NumPy,
installed-package and local PaperLeet application checks. The qualification
record below tracks main adoption separately.

Float32 public scalar powers retain their storage-precision arithmetic. Only
private Tensor pullbacks inside a recorded norm HVP use double scalar exponent
opmath, through a buffered ufunc loop writing into a float32 output. This does
not retain a full double-sized result array. Float32/64 square, cube, square root,
reciprocal square root and reciprocal expressions use one private result buffer.
NumPy scalar exponents preserve half-Tensor result storage. Version 0.80 lets
nonempty nonscalar C/F-contiguous real arrays allocate their result in the first
ufunc. Other layouts retain the explicit buffer path to preserve output strides.

The compatibility target remains native Torch 2.6.0. Torch 2.14 changes some
empty and mixed-dtype matrix contracts; those differences remain explicit.
Floating-point results are not bitwise portable across NumPy, native CPU Torch
and WASM. One retained float32 large-input norm case differs by one ULP before
sine, causing a large phase/HVP difference. Native equivalent expressions and a
controlled native replay reproduce the WASM result. This variance and the
remaining exceptional scalar-power differences are not hidden by changing the
original point-comparison gates. Version 0.80 retains this known boundary;
a passing standalone SDK does not imply universal native numerical parity.


### Higher-order correctness and scalar-power efficiency (0.80 candidate)

Captured Tensor numerators retain their reverse history through division in
functional HVPs. Reverse-based functional JVP/HVP preserve `no_grad()` and view
disconnections, isolate caller gradient buffers, and permit a callback to switch
between polynomial and vector-norm branches. Scoped Tensor pullback recording
visits only the actual output ancestors and releases temporary owners after
success or failure. Public `autograd.grad(create_graph=True)` remains unsupported.

Primitive arithmetic constructs its numerical reverse node and Tensor rule
together, avoids redundant callbacks only while the scoped Tensor reverse pass
is active, and reuses dtype descriptors. One first-gradient traversal reuses its
validated graph order within that invocation; there is no cross-call graph cache.

Against the frozen 0.77 baseline, three fresh processes in each CPU, Node WASM
and Chrome WASM runtime measured 24 ordinary flows and 12 HVP flows. All eight
preexisting HVP flows and all ordinary flows stayed within the 10% per-process
regression threshold, with A/A controls within 5%. For the measured 2048-element
scalar-power forward workload, CPU latency fell 61–67% and WASM latency fell
50–70%. This is workload-specific evidence, not a library-wide speed claim.
New vector-norm HVPs remain approximately 2.8–4.5 times as expensive as the old
baseline's equivalent expression; their cost remains optimization work.

See the [0.80 qualification record](../../docs/validation/torch-runtime-0.80-qualification.md)
for exact baselines, numerical boundaries, package isolation, GPU experiment
scope and current application-adoption status.

### Gradient clipping (0.82)

`torch.nn.utils.clip_grad_norm_`, `clip_grad_value_` and the deprecated
`clip_grad_norm` alias are available through ordinary `torch` imports, including
`torch.nn.utils.clip_grad`. They accept a Tensor or a parameter iterable and
modify existing gradient buffers under `no_grad`, preserving their identities,
layouts and existing history. Normal saved-value checks still apply when that
history needs a value changed by clipping. The returned total norm is detached;
`error_if_nonfinite=True` rejects a nonfinite norm before changing any gradient.

Norm clipping uses the existing floating/complex vector-norm kernels, including
half storage. Value clipping supports real floating gradients. `foreach=None`,
`False` and `True` use per-gradient NumPy kernels; accepting the hint does not
provide a fused or GPU implementation. Empty norm clipping returns zero in the
default floating dtype. Value clipping without any gradients raises an error,
matching the pinned Torch 2.6 behavior.

Multiplication and direct division now protect only the input values used by
their active pullbacks. Their output can be changed when backward does not read
it; required inputs remain protected and saved-graph retention rules still
apply. Attention keeps its existing conservative original-input guards.
