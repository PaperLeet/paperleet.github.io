# StrideTorch 0.37: Python feature discovery with typed fallback

Version 0.37 lets ordinary Python feature checks work across the torch-compatible
namespace, Tensors and optimizers. Previously a missing public symbol made both
`hasattr(value, name)` and `getattr(value, name, default)` raise an unsupported
backend exception. They now return False/default, as native Python/PyTorch code
expects. This supports optional code paths without rewriting them around a
library-specific exception. The independent NumPy-only package remains named
`stridetorch` pending the user's browser-oriented naming decision.

## Lookup, import and failure boundaries

All **32** declared torch module namespaces use an AttributeError subtype that
also matches the existing UnsupportedOperation/NotImplementedError handlers.
The same applies to missing public Tensor/Parameter/optimizer attributes.
Normal `nn.Module` attributes already followed native AttributeError behavior
and retain that behavior. Live missing-symbol errors carry the actual owner in
`obj` and the requested key in `name`. Private/protocol lookup remains ordinary
AttributeError. Existing values, including dynamically added/deleted attributes,
are honored; there is no negative-name cache or fake callable placeholder.

An unavailable `from torch import ...` follows normal Python ImportError
semantics. PaperLeet's existing structured exception classifier recognizes
ImportError and UnsupportedOperation, so both direct and import-based failures
retain server fallback. The application test now executes missing root symbols,
missing `torch.nn` symbols and missing submodules through the real harness.
Text inside unrelated assertions still does not establish an unsupported API.

Actual unsupported options/AD modes keep their existing NotImplementedError
classification and are not AttributeError. Genuine descriptor ValueError and
TypeError also propagate. The change is restricted to lookup; it does not turn
every backend failure into an attribute default. Attribute existence is only a
structural check, not proof that every dtype, gradient or device mode works.

The dual error bases deliberately preserve both Python lookup handling and
existing backend catches. Python's [exception documentation](https://docs.python.org/3/library/exceptions.html#inheriting-from-built-in-exceptions)
advises against multiple built-in exception bases because layouts can differ or
change. Actual gates cover Python 3.10.19, 3.12.11 and Pyodide's 3.13.2; they do
not establish compatibility with untested interpreters or future layouts.
No new dependency or global interpreter patch is introduced. The
[module lookup protocol](https://peps.python.org/pep-0562/) remains Python's own.

## Both runtime profiles and object ownership

The small Schedules profile uses the same error definitions and explicitly
exports the shared errors module. Its package resources grow from five to six,
including both redistribution notices. An isolated subprocess starts from only
those exported files, exercises values/defaults/typed failures and verifies that
the full `_api` and `_tensor` modules are not imported. Schedules still rejects
autograd; missing-attribute compatibility does not enable unsupported modes.

Standard copy/pickle behavior is checked against the running interpreter's own
AttributeError. The first Python 3.10 test incorrectly required `name` to survive
serialization; native Python 3.10 drops it, while newer tested runtimes retain
it. The corrected test follows that standard behavior and verifies that the
owner is not serialized. The original three failed assertions are retained.
An early inheritance-order probe also rejected keyword metadata; the selected
AttributeError-first form passes actual metadata, copy and catch tests.

Repeated optional probes release temporary Tensors and alias namespaces after
collection. Public/private installed layouts perform the same tests, including
error serialization and standard imports, and survive independent uninstall.
The existing explicit `import torch` alias/CLI lifecycle is preserved.

## Numerical source and local validation

Source hashes identify **40 unchanged resources**. AST comparison checks
**248 unchanged existing function definitions** in Tensor,
optimizer and Schedules code outside lookup helpers, plus unchanged numerical
install statements and non-lookup operator registration. The generic operation
unsupported helper remains unchanged. This source evidence complements runtime
tests; it does not establish all third-party-library compatibility.

- Full merged Python suite: **32,549 passes, 1 optional skip,
  34 warnings, 253.10s**.
- Focused native feature, dispatch/lifecycle, standalone and actual import
  fallback contracts: **124 passes**.
- Node Pyodide and actual Chrome Worker: **1,581 checks in 36 suites each**
  on pinned Pyodide 0.29.4 / NumPy 2.2.5 without native Torch.
- Independent Python 3.10.19 / NumPy 1.26.4 wheel: **1,594 passes**.
  All 47 core/license resources agree across public/private distributions;
  aliases, CLI, ZIP resume, lookup/import/pickle and uninstall ownership pass.
- Original native application catalog: **78 problems, 10,218 executions,
  342 mutants, 3 repeats, 133.06s**, unchanged budgets.
- Actual exported WASM: **78/78 modules, 342 mutants,
  292054ms**, concurrency 2, no deadline exceeded.
- Full frontend check and both typed builds pass. All **8,807** prior immutable
  assets remain byte-identical; **158** new payloads are reachable and all
  **161** active index/payload files are SHA-256 frozen.

The first full WASM run passed 77/78 modules. Its attention task also runs all
independent SDK suites, including the new Schedules-specific lookup tests;
that VM originally contained only the autograd export and lacked `_legacy.py`.
The maintained test driver now loads that one module from the actual exported
Schedules bundle, after the original learner/reference/mutant checks, verifies
every shared resource agrees and records both profile URLs and the supplemental
source hash. All **1,581 checks in 36 suites** then pass in the complete 78-module
rerun. No SDK test was skipped, production bundle enlarged, checkout runtime
substituted or deadline increased. The original failed run and targeted
diagnostics remain in the integration reproduction archive.

## Complete-flow measurements

Eight sequential children cover two dtypes and two fresh processes per CPU/WASM
after heavy gates. The 21 workflows use nine ABBA/BAAB blocks, warmups, GC enabled,
actual SGD updates and value/dtype/gradient/state checks. Both namespaces use
the same inverse and native-ordered JVP APIs. Optional-SGD and optional Schedules
inference compare the new Python default lookup with an explicit 0.36
UnsupportedOperation-catching workaround. Unmodified 0.36 would fail the same
optional probe, so this is not described as a speed comparison with working
unmodified feature detection. The initial 19-flow and both-profile 21-flow pilots
remain available.

| Whole workflow | CPU time reduction vs 0.36 | WASM time reduction vs 0.36 |
| --- | ---: | ---: |
| tiny-inference | -1.2 to 0.3% | -0.3 to 2.2% |
| small-inference | -4.3 to 0.5% | -1.0 to 0.7% |
| large-inference | -4.1 to -0.1% | -0.7 to 1.3% |
| upper-inference | -0.9 to 0.3% | -0.6 to 0.3% |
| batch-inference | -0.8 to 0.3% | -0.9 to 1.0% |
| tiny-backward | -2.0 to 1.0% | 1.8 to 4.9% |
| small-backward | -1.5 to 0.7% | -0.1 to 1.8% |
| large-backward | 0.3 to 1.1% | -4.0 to 1.0% |
| sgd-factor | -1.1 to 0.0% | 0.6 to 2.7% |
| optional-sgd | -3.1 to 0.1% | 3.2 to 4.2% |
| spd-chain | -0.8 to 2.6% | -1.3 to 2.1% |
| jvp | -4.2 to 0.5% | -3.7 to 3.0% |
| large-jvp | -1.7 to 2.1% | 0.5 to 1.4% |
| jvp-backward | -0.7 to 1.5% | -1.3 to 1.6% |
| large-jvp-backward | -2.2 to 3.5% | -0.0 to 1.0% |
| fixed-factor-jvp | -1.3 to 0.5% | -0.1 to 0.5% |
| legacy-inference | -1.0 to 0.5% | -0.5 to 0.5% |
| legacy-optional-inference | -1.4 to 0.5% | 0.3 to 1.2% |
| generic-control | -2.8 to 0.9% | -0.8 to 2.9% |
| cholesky-control | -1.7 to 3.3% | -1.1 to 1.3% |
| unary-control | -11.4 to 0.2% | -0.9 to 1.1% |

Negative values mean slower. This version makes no uniform numerical speedup
claim. An initial CPU unary control sample shows -11.4% despite unchanged
numerical code. Four additional complete paired CPU children run after the full
WASM gate, without source edits or discarded original samples. Their unary
control range is **-1.4 to 1.7%**, optional-SGD
**-1.5 to 0.6%**, and Schedules optional inference
**-1.2 to 0.5%**. A separate 50-iteration unary
probe counts actual missing-lookup calls: baseline
`{}` and candidate
`{}`; its gradients agree.
The follow-up remains separate from the original table.

Cold CPU namespace import medians (four fresh alternating processes per set):
NumPy included 83.63 →
82.68ms; preloaded
60.35 →
60.43ms. Source bytes
525692 → 525960; local
gzip estimates 123816 →
123901. These shared-host results
request thread limits but do not verify vendor thread counts. They are not
statistical guarantees, browser wire sizes or native-Torch/CPU-WASM throughput
comparisons. No WebGPU backend is enabled by this change.

## Reproduction, adoption and remaining scope

[`iteration-37.json`](../../packages/stridetorch/benchmarks/results/iteration-37.json)
and [`iteration-37-integration.json`](../../packages/stridetorch/benchmarks/results/iteration-37-integration.json)
contain exact hashes, receipts, probes, pilots and hashed reproduction archives.
Application gates use upstream `7b6e36c023e1ce3ebac940dac1e70e062de11ceb`, including its current catalog
in the rebuilt private wheel. Later main synchronization/adoption is recorded
separately. Native-reference tests are oracles, not runtime dependencies.

The [explicit-release policy](../releasing.md) remains unchanged. No CI, tag,
workflow dispatch, package upload or online publication is requested. The user
will tag later. Local results do not establish remote Linux CI or deployment.
The actual export inventory still lacks max_pool2d/MaxPool2d, transpose
convolution and instance normalization; this lookup repair does not advertise
them. Broader core numerical coverage and prior measured performance gaps remain
active work toward independent-library readiness.

Final main synchronization preserves all 15 incoming UI/docs/check files through
`5859d6a1d669a1e91cbe27e71da0b714ca38aa67` byte-for-byte. The complete frontend check and both
typed builds pass again, including community code blocks and resilient sharing.
All 47 computational resources and 161 exports remain unchanged; see
`iteration-37-main-sync.json`. The full exported WASM retry ran after these
checks against this same source and catalog.

Validated source was adopted non-force on remote `main` at `724b7e8a7eabf76b2369c21fe5492add166e3652`.
A read-only query found zero Actions runs for this commit. See
`iteration-37-main-adoption.json`; no tag, dispatch or online release was created.
