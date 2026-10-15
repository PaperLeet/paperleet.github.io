# paperleet-torch

A lightweight, bounded `torch`-compatible Python library backed by NumPy.
It runs in CPython and Pyodide without installing native PyTorch. Analytical
reverse-mode gradients and a validated forward-mode subset come from PaperLeet's
existing browser adapter. Core API fixes and performance improvements are
developed here and checked against independent native PyTorch results.

```sh
python -m pip install ./packages/paperleet-torch
paperleet-torch train.py --epochs 10
# Equivalent: python -m paperleet_torch train.py --epochs 10
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
from paperleet_torch import install_as_torch
install_as_torch()

import torch
from torch.nn import Linear
import torch.nn.functional as F
```

This enables `torch.nn`, `torch.nn.functional`, `torch.optim`, `torch.linalg`,
`torch.autograd`, `torch.func`, `torch.testing` and `torch.random`. Plain
`import paperleet_torch` leaves native PyTorch imports untouched. Explicit usage
via `from paperleet_torch import torch` also remains available. To scope aliases:

```python
from paperleet_torch import as_torch

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

This is in-memory state support. Model hooks, `load_state_dict(assign=True)`
and `torch.save/load` remain
unsupported. `ModuleList.insert` supports indices from zero through its length;
insertion into a named `Sequential` is unsupported. Copying a model clears copied
parameter gradients, as native PyTorch does. Copying non-leaf tensors or exotic
non-dense NumPy backing allocations fails explicitly.

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
`paperleet_torch.runtime.source_files()` to include the library's Python modules
in a static runtime bundle. PaperLeet exports those sources once per backend,
under its existing content-addressed runtime assets. The independent wheel has
only a NumPy runtime dependency. It contains no problem catalog, judge, pytest,
PyTorch binaries, or browser application code.

## Development

From the PaperLeet repository:

```sh
python -m pip install -e .
python -m pip install -e ./packages/paperleet-torch
python -m pytest packages/paperleet-torch/tests tests/test_browser_*.py -q
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
The subproject builds separately with `pip wheel --no-deps ./packages/paperleet-torch`.
`paperleet.browser.numpy_autograd` and `schedules_numpy` retain compatibility
facades; source changes should be made in this library.

See the [0.5 validation record](../../docs/validation/torch-runtime-iteration-5.md)
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
