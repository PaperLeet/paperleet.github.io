# StrideTorch

A lightweight, bounded `torch`-compatible Python library backed by NumPy.
It runs in CPython and Pyodide without installing native PyTorch. Analytical
reverse-mode gradients and a validated forward-mode subset come from PaperLeet's
existing browser adapter. Core API fixes and performance improvements are
developed here and checked against independent native PyTorch results.

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
leading minor. Version0.32 remains a development candidate until its validation
and application integration gates pass; see its
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
intermediate; finite narrow JVPs associate products through the RHS while keeping
the full native factor/direction rules. These are saved graph values, not
cross-call caches. Output
buffers, half factorization and complex AD remain explicit unsupported cases.
Version0.33 validation and integration are tracked in its
[validation report](../../docs/validation/torch-runtime-iteration-33.md).

Factorization failures are checked across numerical runtimes. Cholesky verifies
its returned diagonal because some WASM NumPy builds return NaN factors without
raising. General solve/inverse use explicit singular-pivot diagnostics when the
runtime cannot report floating-point error flags; empty-RHS solves also check
the coefficient. Triangular solves retain their documented nonfinite-result
behavior for zero pivots. Native linalg makes no general cross-backend guarantee
for NaN inputs; StrideTorch consistently rejects a non-positive/NaN Cholesky
factor diagonal while continuing to ignore unused input triangles.

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

Builtin scalar defaults remain float32/int64/bool. Out-of-range Python integers
fail when constructing default int64 tensors; weak integer operands of existing
integer tensors follow the receiving dtype's arithmetic instead. Factory casts
isolate NumPy floating-point error settings and restore the caller's settings.
As in native Torch, out-of-range ndarray floating-to-integer result values are
platform dependent. Sequence scalar storage instead checks integer range and
Python numeric protocols, following native construction rather than array casts.

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
