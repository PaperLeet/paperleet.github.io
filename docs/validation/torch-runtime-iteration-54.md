# Torch runtime iteration 54: registered Transformer activation serialization

Version: 0.54.0. Baseline main: `dc6570257c654d3a8751b12bec3890d7dd7a3601`.

Transformer models using direct registered functions such as `torch.sin`,
`torch.relu`, `torch.tanh`, `torch.sigmoid`, `F.silu` or `F.softplus` previously
executed correctly but Python pickle failed on a local registry closure.
Encoder/decoder layers now serialize a distribution-local operator reference.
Encoder/decoder stacks and full Transformers inherit this behavior. Restoring
the model keeps the operator's original capability guards and never imports or
replaces the host's `torch`. No numerical kernel or live operator call changed.

The weak registry index does not retain discarded namespaces/functions.
Live layers keep their original function; shallow copies share parameters and
the original callable, while deep copies preserve callable identity and clone
parameters independently. The pickle reference is confined to serialized state.
Default/string/functional/module activation behavior from 0.53 remains supported.

## Validation

- CPU repository and library regression: **40,326 passed, 1 skipped**.
- Node Pyodide and Chrome Worker: **6,696 passed each**, 50 suites, all 57
  source/license hashes checked against the frozen candidate.
- Isolated installed wheel: **6,709 passed** on Python 3.10.19 / NumPy 1.26.4,
  without native Torch. Public/private wheels have disjoint installed paths and
  identical 55 Python modules plus LICENSE/NOTICE.
- Public/private coinstallation, independent uninstall, cross-process model
  restoration and fresh-process lazy resolution with a preexisting host `torch`
  module all passed. The private copy continues to work with the public package
  uninstalled.
- **61 new tests**, **293 Transformer tests total**: five module forms, six
  direct operator activations, pickle protocols 2 and the interpreter's highest
  protocol, shallow/deep copy, outputs, VJPs, JVPs, mixed derivatives and alias
  isolation. Existing independent equations and lifecycle tests also pass.
- Native PyTorch MATH attention differential: **60 cases / 4,008 float64 arrays**
  covering restored outputs, VJPs, JVPs and mixed derivatives; rtol `3e-10`,
  atol `3e-11`, maximum absolute difference
  `4.796163466380676e-14`. The first native probe
  selected CPU flash attention, which lacks forward AD; the successful probe
  explicitly selects MATH attention, matching the dense adapter.
- Fresh exported browser catalog: **78/78 problems**, **342 mutants**, all
  independent contracts passed, no browser budget exceeded. The previous native
  catalog result is reused after checking all **1,152 native input file hashes**.
- Five frontend checks/builds passed. All **11,493 previously published immutable
  assets** remain unchanged; 158 new reachable assets and three mutable indices
  select the current runtime. All seven canonical release-policy files remain
  byte-identical to the established policy baseline.

The numerical regression completed before upstream main advanced to
`dc6570257c654d3a8751b12bec3890d7dd7a3601` (company-comparison practice batches).
All ten incoming UI files were preserved by fast-forward. Numerical source,
contract tests, native catalog inputs and exported resources were unchanged;
their completed evidence was retained. All five frontend checks/builds were
rerun on the latest UI, with the earlier logs retained in the raw harness.

## Scope and limits

This fixes direct registered activation functions. Arbitrary closures and nested
callable objects keep their existing Python serialization restrictions. Full-model
Python pickle requires trusted input and is separate from weights-only `torch.load`.
This is a lifecycle/API improvement; no kernel speedup or new WebGPU backend is
claimed. The 0.53 dense CPU/WASM scope and documented float32 mixed-AD rounding
qualification carry forward.

## Reproducibility

Machine-readable result:
`packages/stridetorch/benchmarks/results/iteration-54.json`.
Raw harness, frozen source/tests, oracle arrays, package artifacts and terminal
receipts: `packages/stridetorch/benchmarks/results/iteration-54-harness.tar.gz`
(3269933 bytes, SHA-256 `6da71cee3bd8629a07d6deab88175c50042bd181929c91b947c586a6dcf3a4ec`).

Validation ran locally. No tag, workflow dispatch, release, package upload or
deployment was initiated. The broader independent-library goal remains active.

## Main adoption

Validated source `a626945bd13dc9e0b3539dd0a84f9d3d52adf456` was adopted on remote main by ordinary non-force push.
The remote SHA matched exactly, with zero observed Actions runs for that SHA.
All 57 source/license resources, 161 active exports, 11,651 immutable payloads
and seven canonical release-policy files were rechecked. A separate proof commit
records this observation; its final main identity is checked outside the repository.
No tag, workflow dispatch, release, PyPI upload or deployment was initiated.
The broader library efficiency/core-coverage/backend objective remains active.
