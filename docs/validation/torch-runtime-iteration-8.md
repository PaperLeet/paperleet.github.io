# Lightweight torch 0.8 validation

This milestone makes common model introspection and local callbacks work through
ordinary torch aliases. The standalone runtime remains NumPy-only. Forward hooks
can transform a model's inputs/outputs without changing its implementation, and
public checkpoint hooks can migrate an in-memory checkpoint format while
preserving the existing parameter objects and optimizer references.

## Behavior

- Missing Module attributes raise AttributeError. Generic initialization with
  `hasattr(module, "weight")` or `getattr(module, "weight", None)` works across
  Linear, normalization, activation and container modules. Registered names
  appear in `dir`, excluding numeric container indices. Registration respects
  subclass attribute/property collisions; computed Module properties work with
  `get_submodule`. Known unavailable methods remain callable interfaces that
  fail explicitly with UnsupportedOperation.
- `register_forward_pre_hook` supports prepend order, positional argument
  replacement and kwargs-aware `(args, kwargs)` replacement.
  `register_forward_hook` supports prepend, kwargs-aware observation and arbitrary
  output replacement, including structured outputs. A forward callable is
  captured before pre-hooks, so replacing `module.forward` during a pre-hook
  affects the next call. Calling `forward` directly bypasses hooks.
- Normal forward dispatch snapshots each phase's registrations. New post-hooks
  registered by a pre-hook can run in the same call; callbacks added during the
  post phase wait for the next call. Removal does not erase an already captured
  callback in that phase. Keyword/always-call flags follow their live registration
  maps. Native-reference mutation tests cover these distinctions.
- `always_call=True` executes uncalled forward hooks after pre/forward/post
  exceptions, with the current arguments and last available output (or None).
  A cleanup callback failure warns and preserves the original exception.
  A hook that already ran is not repeated. BaseException subclasses such as
  SystemExit bypass exception cleanup, matching the covered native behavior.
- Public state-dict pre/post hooks run around local save/child traversal and
  receive prefixes, keep_vars and mutable local metadata. Post-hooks mutate the
  shared destination and return None. Public load pre-hooks receive the copied
  state mapping, metadata, strict=True and shared key/error lists; post-hooks run
  after child traversal and may edit the shared incompatible-key lists before
  the final strict check. Shape/copy errors are still errors with strict=False.
- Root load mappings are OrderedDict copies with the original metadata attached;
  child mappings retain native filtered-dict behavior. Hook key changes do not
  rewrite the caller's mapping; Tensor storage remains shallow under the
  established checkpoint contract. Hooks are Python behavior, not state_dict
  entries: register them again when rebuilding a model.
- Checkpoint-hook dictionary traversal remains live. Mutating registrations
  during that traversal can raise RuntimeError, matching native OrderedDict
  behavior. Incorrect save/load post-hook return values fail explicitly.
- Handles support idempotent removal and context-manager cleanup. They weakly
  reference registration maps and do not retain discarded models. Deep-copying a
  model that stores its handle reconnects the copied handle to copied maps.
  The same class is exposed through `torch.utils.hooks.RemovableHandle`.

Hook replacements use normal Tensor/autograd operations. Tests cover gradients,
JVP and reverse-over-forward through a hooked nonlinear model. Internal flag
updates bypass subclass attribute setters so callbacks also work on models that
lock ordinary attribute assignment after construction.

Native-reference contracts target installed Torch 2.14.1 and the
[public Module callback API](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Module.html).
Local forward and public checkpoint callbacks are covered here. Backward/global
hooks, private checkpoint callback interfaces, compilation, assign-based loading
and native file serialization remain outside the runtime contract. This is not a
full native Torch event/compilation engine or an unrestricted serialization API.

## Efficiency

Callback OrderedDicts allocate only when registration needs them. An instance
flag keeps the ordinary module call path short. Removal weakly edits registries;
the next call observes an empty registry, resets the flag and restores the
unhooked path. A class-default flag prototype saved construction work but slowed
hot calls in fresh-Worker ABBA measurements; it was rejected. Empty checkpoint
paths skip unused dictionary/iterator and incompatible-key object allocations.

Timings compare frozen main `89580984f64b6832cf3adbad8b1d946b489efc94` / 0.7.0
with 0.8.0. All 30 candidate source files and benchmark scripts are hashed;
baselines have 29 sources. CPU uses Python 3.12.11 / NumPy 2.5.4 with one
OMP/MKL/OpenBLAS thread. Node v25.9.0 and Chrome 155.0.8059.40 use Pyodide
0.29.4 / NumPy 2.2.5. Runs are sequential in fresh processes/Workers without
local correctness jobs. CPU uses baseline→candidate→candidate→baseline order.

Each case uses three warmups and eleven batches. Module/checkpoint cases use
1,200 calls per batch; small dispatch/construction/registration cases use 12,000
to reduce timer quantization, with the actual count saved per result. Six common
operator controls and the complete attention encoder use 120 calls per batch.
Imports/fixture creation are excluded except the named construction/registration
cases. Active callbacks in dispatch/checkpoint timings are passive observers or
identity replacements, so those costs are framework overhead, not application
logging, tensor transforms or I/O. Float32/float64 fixtures are both measured.

| Common workload, float32 (µs) | CPU 0.7 → 0.8 | Node WASM 0.7 → 0.8 | Chrome WASM 0.7 → 0.8 |
| --- | ---: | ---: | ---: |
| Identity call | 0.108 → 0.111 | 0.272 → 0.280 | 0.267 → 0.258 |
| Sixteen Identity modules | 1.662 → 1.742 | 4.229 → 4.488 | 4.200 → 4.308 |
| Frozen linear inference | 5.438 → 5.417 | 14.018 → 14.207 | 13.417 → 13.167 |
| Construct Identity | 0.322 → 0.386 | 0.825 → 0.953 | 0.833 → 0.925 |
| Save MLP state | 3.089 → 3.290 | 7.798 → 8.310 | 7.833 → 8.250 |
| Load MLP state | 8.262 → 8.908 | 19.576 → 20.856 | 19.500 → 20.167 |

The MLP is 32→64→4 with Tanh and two parameterized layers. Ordinary call overhead
is small but measurable in the longer empty-module chain. Single-call browser
differences of a few nanoseconds are noisy and are not a claimed speedup. The
extra instance flag costs about 0.06 µs CPU / 0.09–0.13 µs Chrome at construction;
checkpoint compatibility also retains an overhead versus 0.7.

| Callback workload, Chrome WASM (µs) | float32 | float64 |
| --- | ---: | ---: |
| Identity with pre + post observers | 0.983 | 1.000 |
| Identity with kwargs-aware pre + post | 1.225 | 1.233 |
| Identity after removal, steady state | 0.258 | 0.275 |
| Register + remove one kwargs/always-call hook | 3.183 | 3.217 |
| Save MLP state with root callbacks | 8.417 | 8.333 |
| Load MLP state with root callbacks | 21.083 | 21.083 |

The first empty call after removal is excluded from steady-state timing. The
initial 0.8 checkpoint loader measured 22.7–22.9 µs on Chrome; skipping unused
objects reduced it to 20.2–20.4 µs in the final source. Initial samples are retained.
The class-default flag prototype measured about 0.84 µs construction versus
0.97 µs for the retained instance flag, but hot calls rose from 0.278 to
0.297–0.300 µs and sixteen-module chains from about 4.51 to 4.82–4.94 µs. Those
fresh-Worker ABBA samples explain why that prototype was not adopted.

CPython tracemalloc for 1,000 live Identity modules records 523,608 retained bytes
in 0.7 and 539,240 bytes in unhooked 0.8 (about 3% more); neither allocates hook
registries. Giving each model a kwargs pre-hook and kwargs/always-call post-hook
retains 3,402,552 bytes, including new callback functions and five lazy maps per
model. Removing handles retains 2,865,760 bytes because the empty maps/capacity
remain cached on live models. Removal releases callback references and restores
dispatch speed; it is not a promise to recover constructor-only memory. These
are traced Python allocations, not browser RSS or tensor storage. Reclaiming
temporary-hook map capacity is a remaining efficiency opportunity.

Chrome attention/linear/MLP/slicing/AdamW controls remain within ±3.4% in this
measurement. The complete encoder step is 2.280 ms float32 and 2.329 ms float64,
versus 2.347 / 2.339 ms in the baseline. These small differences are not claimed
as kernel speedups; module callbacks extend behavior rather than tensor kernels.

Final timings, retained-memory samples and complete gate results are recorded in
[iteration-8.json](../../packages/paperleet-torch/benchmarks/results/iteration-8.json).
Use [bench_module_hooks.py](../../packages/paperleet-torch/benchmarks/bench_module_hooks.py)
or the WASM runner's `--benchmark module_hooks` option.

## Validation scope

The complete maintained Python suite passes **2,116 tests**, one optional Docker
test skipped and five warnings originating in native Torch, 149.70 seconds.
The native lesson gate passes **69 modules, three repetitions, 8,829 executions
and all 295 mutants caught**, 118.91 seconds. Web checks and the production build
pass. All 30 final library source hashes match the timing freeze.
The final integration rebases through UI-only main commit `bb9af9a`; all Python,
test, catalog and library sources remain unchanged. Affected web checks and the
production build are rerun on that combined tree before adoption.
The Pages asset-retention checker passes all 15 cases; tracked published assets
remain available for cached pages.

Real pinned WASM runs five complete lesson reference/mutant/learner suites plus
the existing numerical/fallback regressions and **72 lifecycle contracts** across
six files. Five fresh Chrome Workers run **35 original tests, catch all 20
mutants and pass the same 72 additional contracts**, asserting version 0.8.0 and
the actual `/judge/paperleet_torch` import location.

The independent installed wheel outside the checkout passes **90 tests** with
NumPy 1.26.4 and no native Torch. It is **88,489 bytes**, SHA-256
`a9bdc35669921a986b3c1c065e4ab9a95c8feb4103a8685032c1e2891dc729f9`.
Application/vendor and standalone packaging/launcher isolation remains covered
by the maintained suite, including unchanged torch-importing scripts.

New coverage includes 55 native-reference cases and 10 NumPy-only workflows,
alongside all previous tensor, layer, optimizer, schedule and model tests. The
independent workflows include an analytical NumPy gradient/SGD trajectory,
momentum SGD continuation, checkpoint key packing/unpacking,
JVP derivatives, exception cleanup, kwargs/order/removal, weak handle lifetime
and model introspection. They run in the minimum NumPy environment and real WASM.

Full logs, source freezes, initial/final measurements and fresh-Worker scripts
remain under local `paperleet-wasm-evaluation-20261015/torch-iteration-8` evidence.
The previous 0.7 release has successful quality/Pages runs and its live two
runtime files match the released byte hashes. The broader independent-library,
core/efficiency and optional resident GPU goals remain active. No license has
been selected and no PyPI release is made.
