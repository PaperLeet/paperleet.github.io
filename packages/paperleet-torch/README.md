# paperleet-torch

A lightweight, bounded `torch`-compatible Python library backed by NumPy.
It runs in CPython and Pyodide without installing native PyTorch. Analytical
reverse-mode gradients and a validated forward-mode subset come from PaperLeet's
existing browser adapter. This first extraction preserves its numerical behavior.

```sh
python -m pip install ./packages/paperleet-torch
```

```python
from paperleet_torch import torch

x = torch.tensor([2., 3.], requires_grad=True)
x.square().sum().backward()
print(x.grad.tolist())  # [4.0, 6.0]

layer = torch.nn.Linear(2, 1)
optimizer = torch.optim.Adam(layer.parameters(), lr=0.01)
```

`torch.nn`, `torch.nn.functional`, `torch.optim`, `torch.linalg`, `torch.autograd`
and `torch.func` are attributes of this explicit namespace. Importing the library
does not change `sys.modules['torch']` or `torch.*`. For unchanged teaching code:

```python
from paperleet_torch import as_torch

with as_torch():
    import torch
    from torch.nn.functional import linear
    # Existing torch imports now use the lightweight implementation.
# Pre-existing native torch modules are restored, including after exceptions.
```

Import aliases are process-wide: use this context in an isolated process or
browser Worker. Namespace objects share gradient mode, storage tracking and RNG
state; they are not independent execution engines. Create a fresh Worker for each
untrusted submission, as PaperLeet does.

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
python -m pytest packages/paperleet-torch/tests tests/test_browser_*.py -q
paperleet export
cd web
npm run check
npm run build
npm run check:browser-wasm
```

The main PaperLeet distribution includes this same source tree so current
`pip install .` users do not need an unpublished dependency or a sibling checkout.
The subproject builds separately with `pip wheel --no-deps ./packages/paperleet-torch`.
`paperleet.browser.numpy_autograd` and `schedules_numpy` retain compatibility
facades; source changes should be made in this library.
