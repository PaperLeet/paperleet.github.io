# Lightweight torch 0.2 validation

This version adds unchanged-script imports, fixes the first runtime iteration's attention memory regression, and makes the standalone and application wheels own disjoint installed files. It remains a bounded CPU/NumPy implementation; model/optimizer lifecycle coverage and a resident WebGPU backend remain open.

## Implementation

- `paperleet-torch script.py`, `python -m paperleet_torch script.py` and host `install_as_torch()` preserve ordinary `torch` imports. Plain package imports leave native PyTorch untouched; scoped aliases restore prior modules after errors. CLI arguments, sibling imports and exit codes are tested from real wheels outside the checkout.
- Basic slices use unique-destination gradient updates, while advanced/boolean indices retain duplicate-safe scatter addition.
- Attention preserves float32 outputs, shares one multi-input analytical pullback per traversal, does not retain logits, and releases the temporary probability gradient before constructing input gradients. Ordinary operators keep their direct backward loop to avoid per-edge dispatch overhead.
- Joint pullbacks validate gradient counts, required gradients and shapes, retain frozen/saved operand version checks, and accumulate repeated input roles. Native tests include one tensor used as query/key/value, a differentiable mask view of that tensor, fully masked rows and repeated seeds.
- Adam uses per-parameter update counts, supports coupled weight decay and preserves absent/zeroed gradient buffers. Parameter groups and checkpoint APIs remain outside this release.
- PaperLeet bundles the same source under `paperleet._vendor.torch`; only the standalone distribution owns `paperleet_torch`. Both wheels work alone and uninstalling either cannot remove the other runtime. NumPy is the sole standalone runtime dependency.

## Final checks

- Python infrastructure, native-reference numerical contracts and standalone tests: **1,463 passed**, one optional Docker test skipped, one existing upstream `torch.jit.script` deprecation warning.
- Focused graph/kernel/forward-mode run: **242 passed**.
- `paperleet ci --budget 600`: **69 modules, three repetitions, 8,829 test executions, 295 mutants caught**, no issues.
- Pinned real Pyodide 0.29.4 / NumPy 2.2.5: five complete reference/mutant/learner suites, the existing multiple-test-file regression, and additional alias, dtype, indexing, delayed-Adam and finite-difference attention checks.
- Fresh Chrome 155.0.8059.40 Workers: the same five final exported bundles, **35 original tests and all 20 mutants**.
- Independently built 0.2.0 wheel: **18 standalone tests** under isolated Python outside the checkout with NumPy 1.26.4 and no native PyTorch.
- Full web checks passed. An unchanged focus test failed in the first concurrent run and passed independently and in the final full run; no UI code was changed. Final exported-source checks and the production build passed. Historical tracked assets are retained; unpublished intermediate exports were removed.
- Native/wheel/source checks and raw benchmark records identify this exact package source. No PyPI package publication or production GPU backend is part of this version. The repository has no license file; an independent public package license remains undecided.

The first Pages run exposed an application-build test importing the optional
standalone distribution. That test now reads the application's vendored runtime.
The exact publication check and catalog export passed after actually uninstalling
the standalone library. A further built-wheel regression exports Attention with
standalone and native Torch imports blocked; all nine packaging tests passed.
This keeps application-only installation independent of the optional library.

## Timings

Measurements are sequential in fresh processes/Workers: three warmup calls, nine batches of 60 iterations, seeded inputs, fixed workloads. Native CPU uses Python 3.12.11 / NumPy 2.5.4; Node/browser WASM uses the pinned runtime above. The baseline is the fetched remote `main` commit `7f236b3`; the intermediate source is the frozen first iteration.

| Runtime / dtype | Remote attention (ms) | Iteration 1 (ms) | 0.2 (ms) | 0.2 speedup vs remote |
| --- | ---: | ---: | ---: | ---: |
| CPython / float32 | 0.4432 | 0.2318 | 0.2282 | 1.94× |
| CPython / float64 | 0.4083 | 0.3260 | 0.3246 | 1.26× |
| Node WASM / float32 | 3.5486 | 2.7193 | 2.8060 | 1.26× |
| Node WASM / float64 | 3.4944 | 2.7533 | 2.7477 | 1.27× |
| Chrome WASM / float32 | 3.5167 | 2.7233 | 2.7133 | 1.30× |
| Chrome WASM / float64 | 3.4550 | 2.7500 | 2.7450 | 1.26× |

The old remote float32 attention incorrectly promoted to float64 under NumPy 2.x; that speed comparison includes the dtype correction. These results do not establish an across-library speedup. Final Chrome linear/indexed controls stayed within about 2% of the first iteration; earlier noisy samples were not used as final evidence.

## Memory

CPython tracemalloc records retained/peak allocations for the 2×4×64×32 attention fixture with an additive mask. It includes NumPy allocations registered with that runtime and is not browser RSS.

| Dtype / bytes | Remote | Iteration 1 | 0.2 |
| --- | ---: | ---: | ---: |
| float32 / forward retained | 399,510 | 337,046 | 201,910 |
| float32 / backward peak | 1,195,952 | 1,033,366 | 636,134 |
| float64 / forward retained | 398,192 | 664,344 | 398,920 |
| float64 / backward peak | 1,391,568 | 2,047,976 | 1,258,152 |

The float64 logits-retention regression is removed: forward retained allocations return to approximately the pre-iteration level, and backward peak is lower than both baselines. Frozen-operand retention, larger sequence lengths, complete training graphs and browser heap measurements remain further profiling work.

## Evidence and next work

Raw samples, source hashes, wheel identity, memory and native-gate summaries are in [`iteration-2.json`](../../packages/torchwasm/benchmarks/results/iteration-2.json). Reproduction tools are in the package's `benchmarks/` directory. Complete local logs and frozen baselines are retained in `paperleet-wasm-evaluation-20261015/torch-iteration-2`.

Core model/optimizer lifecycle, additional tensor/autograd coverage, selective saved operands and a resident GPU storage/graph prototype remain in [the roadmap](../../packages/torchwasm/ROADMAP.md). This version is a repository integration milestone, not completion of the independent-library objective.
