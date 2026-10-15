# Lightweight torch 0.10 validation

This milestone adds Tensor gradient callbacks and CPU/no-grad copying for ordinary
training code. The NumPy-only package keeps ordinary torch aliases and its bounded
first-order reverse, forward AD and reverse-over-forward contracts.

## Behavior

`Tensor.register_hook` observes a computed gradient or returns a replacement
Tensor of matching shape/dtype. Multiple callbacks run in registration order.
They execute before retained/non-leaf or leaf gradient accumulation, including
for requested inputs of `autograd.grad`. Shared branches combine contributions
before a Tensor's callback runs. Unneeded or disconnected inputs do not fire.

`register_post_accumulate_grad_hook` applies to leaves. It receives the Tensor
after its `.grad` update and may modify/clear gradients or perform an inline
parameter update. It must return None. It does not fire during `autograd.grad`.
Callback phases snapshot their registrations; changes take effect on the next
phase traversal. A regular callback can install the first post-accumulation
callback for the same backward pass.

Each reverse node starts callback execution in no-grad mode. Mode changes flow
between that node's callbacks, including regular→post, and are restored before
the next node or caller, also on exceptions. Explicitly created hook Tensor
objects survive when directly returned at a requested gradient target, including
their independently constructed graphs. General `create_graph=True` reverse
execution remains unsupported. Existing leaf callbacks work during reverse over
forward AD; registration inside an active forward transform is unsupported.

Weak removable/context-manager handles share the existing `torch.utils.hooks`
class. Non-leaf hooks stay attached to history after the Tensor wrapper is
collected. Copying Tensor/Parameter objects drops their Tensor callbacks.
Module/global backward and raw graph-node callbacks remain separate work.

`copy_` supports frozen CPU tensors and no-grad copies involving tensors requiring
gradients. It preserves target identity/shape/dtype and view storage, supports
source broadcasting/casts and updates shared mutation versions. Exact self/view
copies are allowed, including exact expanded self copies. Partial overlap,
expanded writes with repeated destinations and incompatible broadcasts fail
without changing data or versions. Non-numeric NumPy storage is rejected at the
copy boundary before writes, preventing an observed partial object-conversion
write with an unchanged version. The existing frozen Tensor factories can still
admit such non-numeric arrays, unlike native Torch; factory dtype validation is a
separate core follow-up. Dtype-changing copies isolate NumPy's floating-point
error mode, so IEEE overflow/underflow/NaN copying does not fail midway under
`np.errstate(all="raise")` or leave stale versions. Same-dtype copies avoid that
context. Complex-to-real casts warn once; complex bool
truth includes imaginary components. CPU non-blocking requests remain synchronous.
Differentiable in-place copy graphs, forward-transform mutation and writes into
read-only backing allocations remain explicit unsupported boundaries.

Native-reference tests target installed Torch 2.14.1. Public interfaces are
[Tensor hooks](https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.register_hook.html),
[post-accumulation hooks](https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.register_post_accumulate_grad_hook.html)
and [copy_](https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.copy_.html).
Ordering, mode, lifetime, identity, mutation and error details also come from
independent native executions retained in the validation evidence.

## Efficiency approach

Plain graphs allocate no hook dictionaries and no extra graph-node slots.
Callback metadata shares the optional retention slot. A weakref subclass stores
an optional hook registry without a separate owner-ref wrapper. Retained-only
nodes bypass callback dispatch when no hooks are active. Backward accumulates
directly and skips derivative records that only `autograd.grad` returns.

On CPython 3.12, `sys.getsizeof` measures the optional retained-owner weakref at
80 bytes in 0.9, the prototype wrapper plus owner weakref at 48+80 bytes, and the
final weakref subclass at 88 bytes. This saves 40 bytes against the prototype
while adding 8 bytes against 0.9, excluding Tensor storage and optional hook maps.

The first prototype's retained-only path invoked the callback dispatcher and
allocated a separate wrapper plus owner weakref. It measured about 5.5–6% slower
on Chrome than 0.9 in that fixture. Its sources and all CPU/Node/shared-browser
samples remain in local `initial-profiling`. The final comparison also measures
that prototype under the same fresh-browser protocol as the optimized source.


## Measurements and limits

The final frozen comparison uses 0.9 commit
`8865b4ccb7f7bac29161c27fb0111b6535165f87`, both float32/float64, 11 batches and
three warmups. Core cases use 120 calls per batch; Tensor callback/copy cases use
1,200, with registration/removal using 12,000. Correctness and build producers
were terminal before timing; all timing runs execute sequentially.

CPU uses Python 3.12.11/NumPy 2.5.4 on macOS arm64. Its BLAS build is Apple
Accelerate. The new `bench_cpu.py` sets OMP/MKL/OpenBLAS/VECLIB environment controls
before importing NumPy and selects Accelerate's single-thread BLAS policy with
`BLASSetThreading(1)`. Every fresh process records success and queries policy 1
before and after samples. The signatures/enum come from the installed Apple SDK
`thread_api.h`; the [public BLAS overview](https://developer.apple.com/documentation/accelerate/blas-library)
lists the threading interfaces. Earlier milestone measurements only established
the recorded OMP/MKL/OpenBLAS environment settings, not Accelerate's pool policy;
those historical raw samples are preserved and their documentation clarified.

Two fresh CPU processes per variant/dtype/suite run in baseline/candidate/
candidate/baseline order. Mean process-median changes for attention are −0.01% /
−1.75%, MLP training −0.29% / −4.60%, and plain graph backward −2.13% / −3.59%
(float32/float64). Retained-gradient backward is −0.78% / −2.04%. Float64 slicing
is 1.82% slower. These small fixtures and process ranges do not establish a broad
kernel speedup.

Real WASM uses Pyodide 0.29.4 and NumPy 2.2.5. Node runs one process per
variant/dtype/suite, and its complete samples remain in the record. Node attention
is 5.4–6.0% slower; float64 linear and leaf backward are 11.4% / 15.4% slower,
while several float32 cases are faster. Single-process comparisons remain
sensitive to runtime/process variation and are not combined with Chrome samples.

Chrome 155 uses a fresh entire browser process **and** Worker for each sample,
with baseline/candidate/prototype/prototype/candidate/baseline order for each
dtype (12 processes). The prototype is reconstructed from final sources with
[the reverse patch](../../packages/paperleet-torch/benchmarks/results/iteration-10-prototype.patch),
including its original numeric-copy behavior and old resource loader. All 32
reconstructed Python hashes are verified. Earlier shared-browser samples are
kept separately and are not pooled into this protocol.

| Chrome case | 0.9 float32 µs | 0.10 float32 µs | 0.9 float64 µs | 0.10 float64 µs |
| --- | ---: | ---: | ---: | ---: |
| Attention backward | 2,949–2,973 | 2,975–3,025 | 2,998–3,034 | 2,981–3,006 |
| MLP training step | 140.8–147.5 | 145.0–146.7 | 135.0–141.7 | 135.0–135.8 |
| Plain graph backward | 53.0–53.2 | 53.9–54.8 | 48.3–51.9 | 48.6–50.8 |
| Retained-gradient backward | 55.5–55.7 | 59.9 | 51.1–54.0 | 52.3–56.1 |

The final retained-gradient fixture remains **7.8% / 3.1% slower** on average
than 0.9, despite the lazy dispatch optimization. The same-protocol prototype
ranges are 59.3–65.5 / 54.3–56.9 µs, with substantial process variation; this is
not proof that every optimization pays back its added compatibility cost.
Plain float32 graph backward is 2.4% slower; ordinary float32 attention/MLP are
about 1.3% / 1.2% slower on average, while float64 is near parity or slightly
faster in these runs.

A leaf observation callback adds roughly 1.4–2.1 µs over each corresponding
plain leaf run. Replacement with `gradient * .5` totals 15.1–17.4 µs, including
Tensor arithmetic/allocation. Post-accumulation observation totals 6.3–7.3 µs;
registration/removal is 1.4–1.6 µs. Frozen 64×128 contiguous copies take
7.9–8.8 µs and strided copies 8.7–9.2 µs. These are complete operation timings,
not isolated callback/transfer kernel costs. Sub-clock-resolution identity
conversion values are retained without percentage claims.

Float64 AdamW baseline process medians are 422.5 and 302.5 µs, versus candidate
330.0 and 315.0 µs. Similar large AdamW variation appeared before this milestone;
its cause remains undetermined. All samples remain visible, and the averaged
ratio is not presented as an optimizer speedup.

The retained-loss memory fixture (16 stages, 64×128) is byte-identical to 0.9:
released float32/float64 losses retain 54,570 / 86,550 traced bytes after backward,
with peaks 668,690 / 1,290,502 bytes. This fixture does not call `retain_grad` and
therefore does not measure callback-holder overhead.

## License and source provenance

The user selected Apache-2.0 for the independent library. The subproject has the
standard [LICENSE](../../packages/paperleet-torch/LICENSE), [NOTICE](../../packages/paperleet-torch/NOTICE)
and SPDX license expression in package metadata. Both notices accompany the
standalone wheel/sdist, the application's private copy and both browser profiles.
This scope applies to `packages/paperleet-torch/`, not the remaining application
or third-party dependencies. No PyPI publication is made.

The license decision arrived after frozen timings. Only `runtime.py` changed
among the 32 Python files: source export now includes LICENSE/NOTICE. The other
31 files, including all numerical/graph/callback code, remain byte-identical.
The timing suites do not import this resource loader. Raw timings keep their
original exact hashes; final packaging/correctness results use separate release
hashes. The source delta is explicit in the raw record instead of silently
relabeling earlier measurements.

## Validation scope

The complete maintained Python suite passes **2,378 tests**, one optional Docker
test skipped and six native Torch warnings, 197.43 seconds. This includes 51 new
native-reference Tensor callback cases, 57 copying/boundary cases, 12 independent
NumPy-only workflows and two runtime-notice profile checks. The native lesson
gate passes **69 modules × three repeats, 8,829 executions and all 295 mutants**,
118.79 seconds. It precedes final copy/license refinements, which do not alter
original solutions, maintained lesson tests or mutants.

Real WASM runs six complete representative original/reference/mutant suites and
**94 lifecycle contracts** across eight files. Six fresh Chrome Workers pass
45 original tests, catch all 29 mutants and run those 94 contracts, asserting
version 0.10.0, the actual library import location and installed LICENSE/NOTICE.

The installed independent wheel passes **114 tests** outside the checkout with
NumPy 1.26.4 and no native Torch. Its 32 Python source hashes match the final
release snapshot. It is **104,496 bytes**, SHA-256
`ef338f9ac987f8e29dc26f820923cb182513556605730b83433391e4c0404b01`. Both wheel and sdist preserve the canonical license/notice
contents. Application and standalone wheels have disjoint ownership and retain
their runtime resources after the other distribution is removed.

The post-license frontend gate exposed a real integration failure: the public
bundle validator previously allowed only Python/YAML files. It now admits only
the two exact library notice paths. Checks cover all 69 current hydrated exports,
old runtimes without notices, manifest override rejection, traversal and unrelated
notice paths. Full web checks and production build pass after the fix.

Raw samples, frozen and release source hashes, exact browser harnesses, allocation
audits, copy-cast observations and gate results are recorded in
[iteration-10.json](../../packages/paperleet-torch/benchmarks/results/iteration-10.json).
Full logs, interrupted-run evidence, earlier variants and source snapshots remain
under `paperleet-wasm-evaluation-20261015/torch-iteration-10`. The previous 0.9
release has successful exact-commit quality/Pages runs and both live runtime byte
hashes verified before this milestone.

The wider independent-library goal remains active. Next core audits include
non-numeric frozen factory rejection and differentiable seed/hook Tensor payloads
through metadata-only ancestor VJPs. General higher-order reverse graphs,
module/global backward callbacks, broader NN coverage and a production resident
GPU backend remain separate work.

## Main integration

The integration preserves concurrent upstream starter contracts, compact and
legacy catalog formats, UI updates and the new MoCo lesson (70 current modules).
The 69 existing changed starters have identical executable ASTs after stripping
docstrings; solutions, maintained lesson tests and mutants remain unchanged.
The 130 catalog/config/browser/installed-wheel checks pass after regeneration.
The final full Python suite above includes the current upstream catalog.

MoCo additionally passes a native three-repeat gate: 300 executions and all nine
mutants caught. This supplements the preceding unchanged 69-module native gate;
it is not represented as one fresh combined local run. The maintained WASM gate
now includes MoCo's actual ten-test suite and all nine mutants. Six fresh Chrome
Workers validate the final current export, including MoCo, with the counts above.
Final web checks/build and asset-retention/backfill checks pass. All 32 release
source hashes remain unchanged through integration.

A final concurrent UI update through `012dbc98b6120233de4fafa70d26aa8142f8101d` changes only 13
frontend files. Library, Python/test, lesson, catalog and runtime trees remain
unchanged. The newly affected full frontend check and production build pass
again; numerical and curriculum gates above remain applicable.

## Released verification

Commit `ac716bac9bfcd50e0d3ab530dc72deb264f50b2d` is on main. Its exact-commit
[quality gate](https://github.com/PaperLeet/paperleet.github.io/actions/runs/38779719998)
and [Pages build/deploy](https://github.com/PaperLeet/paperleet.github.io/actions/runs/38779720016)
both complete successfully. The remote quality log records 2,378 passed, one
optional skip and six warnings (423.41 seconds), then a fresh 70-module native
gate with 9,129 executions across three repeats (382.08 seconds). All 304 mutants
are caught. Its six maintained pinned WASM suites also pass. The full Python gate
includes every exported module's complete NumPy-adapter reference/learner suite
and all mutants, independently of the native gate.

The live compact index on paperleet.com matches all 70 source starters and every
content-addressed module manifest. Both live runtime payloads match local bytes:
autograd `f1e24e96d9e09f33cdfc24f95daca402a87ef79030c2a1fe78dc4126e6ccdf4d`
(323,619 bytes) and schedules
`7f71fa4d2765e4197434c4547ccdf4e16e56e8bb24b47406e9e1bbd81011edb4`
(35,895 bytes). Canonical LICENSE/NOTICE are present in both profiles. The wider
independent-library goal continues in subsequent milestones.
