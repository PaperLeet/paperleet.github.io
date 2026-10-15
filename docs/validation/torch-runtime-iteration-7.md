# Lightweight torch 0.7 validation

This milestone adds registered neural-network building blocks and complete
attention-model training workflows. NumPy remains the only runtime dependency.
Existing scripts keep `import torch`, `from torch import nn` and functional or
class-submodule imports when run with `paperleet-torch train.py`. Python/Pyodide
hosts can activate the same aliases once with `install_as_torch()` before loading
user code. Plain installation/import does not automatically replace native Torch.

## Behavior and boundaries

- New modules: LayerNorm, RMSNorm, Embedding, GELU, Dropout, Identity, Flatten,
  Tanh, Sigmoid, MSELoss, SmoothL1Loss and CrossEntropyLoss. Parameters participate
  in model traversal, dtype conversion, optimizer updates and state dictionaries.
  Resetting parameters retains their objects. Deep imports such as
  `torch.nn.parameter.Parameter`, `torch.nn.modules.normalization.LayerNorm` and
  `torch.nn.modules.sparse.Embedding` share the same classes as `torch.nn`.
- Normalization supports multidimensional trailing shapes, empty inputs,
  optional affine parameters, analytical VJPs and forward AD with affine tangents.
  LayerNorm's reverse-over-forward path retains the native saved-statistic rules;
  RMSNorm retains its differentiable composite rule. Input, weight and bias
  gradients are requested through one joint pullback with active-input caches.
- Ill-conditioned float32 LayerNorm rows use float64 centering and variance,
  returning float32 results. A rounded float32 mean can otherwise erase much of
  the signal. This improves agreement with an independent float64 formula, but
  can differ from native CPU float32 Welford/reduction order on those rows.
  Bitwise native agreement on ill-conditioned inputs is not promised. RMSNorm's
  derivative preserves native composite overflow/underflow evaluation order.
- GELU has exact scalar `math.erf` and explicit `approximate="tanh"` implementations,
  with analytical VJP, JVP and reverse-over-forward curvature. Typed coefficients
  preserve scalar float32 on both NumPy 1.x and 2.x. Exact mode never silently
  switches to tanh. Curvature is computed only for forward AD.
- Dropout honors recursive train/eval and shares a scaled Bernoulli mask across
  primal, tangent and backward. Its seeded stream repeats within this library;
  it is not native Torch's dropout bit stream. Disabled, p=0, p=1 and empty cases
  do not draw randomness. Nonfinite/signed-zero behavior and frozen in-place
  updates are covered; differentiable in-place dropout is explicitly unsupported.
- Embedding supports normal initialization, padding, dense/frequency-scaled
  gradients, weight JVP and frozen/shared `from_pretrained` weights. Pretrained
  padding values are preserved. Max-norm renormalization updates selected rows
  under no-grad and preserves storage version guards, including no-change calls.
  Sparse gradients and max-norm changes during JVP remain unsupported.
- Loss classes use the existing functional contracts. Cross entropy still
  requires finite `[N,C]` logits and int64 `[N]` targets. Weighted/smoothed,
  probability-target or spatial cross entropy is not newly implemented here.
  Neural-network kernels remain dense CPU float32/float64. Normalization/GELU
  backward after storage rebinding fails explicitly; convert models before
  constructing a training graph.

Native-reference tests include a deterministic attention encoder, all parameter
gradients, SGD/LinearLR updates and checkpoint continuation. An independent
NumPy-only stochastic attention workflow uses AdamW/AMSGrad, StepLR, saved RNG
contexts and save-rebuild-load-resume. Normalization and GELU also have independent
formula and finite-difference checks. Native LayerNorm JVP cache semantics were
checked against [PyTorch's derivative implementation](https://github.com/pytorch/pytorch/blob/main/torch/csrc/autograd/FunctionsManual.cpp).

## Performance and memory

The baseline is frozen `254b653` / 0.6.0; candidate is frozen 0.7.0. All 29 Python
source hashes match the benchmark snapshot. CPU uses Python 3.12.11 / NumPy 2.5.4
with one OMP/MKL/OpenBLAS thread. Node v25.9.0 and Chrome 155.0.8059.40 use
Pyodide 0.29.4 / NumPy 2.2.5. Timings run sequentially in fresh processes/Workers,
without local correctness jobs. Each case has three warmups and eleven batches
of 120 calls. CPU uses baseline→candidate→candidate→baseline order; the table
averages the two process medians per variant. Imports are excluded.

| Workload (ms) | CPU 0.6 → 0.7 | Node WASM 0.6 → 0.7 | Chrome WASM 0.6 → 0.7 |
| --- | ---: | ---: | ---: |
| LayerNorm backward, float32 | 0.216839 → 0.237914 | 0.604956 → 0.636839 | 0.597500 → 0.627500 |
| RMSNorm backward, float32 | 0.158113 → 0.173982 | 0.447735 → 0.468563 | 0.436667 → 0.471667 |
| LayerNorm bias only, float32 | 0.122401 → 0.139407 | 0.333065 → 0.360256 | 0.330833 → 0.360000 |
| Embedding backward, float32 | 0.527740 → 0.129891 | 0.934764 → 0.278358 | 0.977500 → 0.286667 |
| LayerNorm backward, float64 | 0.362770 → 0.367410 | 0.718520 → 0.736288 | 0.686667 → 0.713333 |
| RMSNorm backward, float64 | 0.262065 → 0.276520 | 0.451865 → 0.483231 | 0.451667 → 0.488333 |
| Embedding backward, float64 | 0.143931 → 0.143451 | 0.285777 → 0.282326 | 0.290000 → 0.295000 |

The embedding fixture uses a 256×32 table, 16×32 indices and frequency scaling.
Casting frequency counts to the weight dtype avoids a float64 gradient temporary
for float32 weights: latency improves approximately 75% CPU and 70% WASM here.
Float64 is essentially unchanged. Norms use 16×32×128 inputs and 128 affine
parameters. Added validation, numerical handling and graph guards have measurable
cost: float32 full backward is about 10% slower on CPU and 5–8% on Chrome, while
float64 LayerNorm is about 1%/4% slower. The richer norm path is not claimed as a
general kernel speedup. Existing attention, linear, MLP training, slicing and
AdamW controls remain approximately flat; sub-microsecond browser calls can
round to zero and cannot support throughput conclusions.

An initial equivalent LayerNorm gradient expression summed two projection terms
before subtracting. It measured about 1.0 ms for float64 on Chrome versus 0.68 ms
in 0.6. A fresh-Worker ABBA control reproduced the regression. Restoring the
continuous subtractive order reduced the candidate to approximately 0.71 ms;
correctness tests still pass. Multiplication/square and unbroadcast prototypes
showed no stable benefit and were not adopted. Initial samples and prototype
controls remain available; the exact internal WASM cause is not established.

| New workload, Chrome WASM (ms) | float32 | float64 |
| --- | ---: | ---: |
| Exact GELU backward | 0.952500 | 0.968333 |
| Tanh GELU backward | 0.177500 | 0.195000 |
| Dropout backward | 0.127500 | 0.177500 |
| Attention encoder training step | 2.262500 | 2.334167 |

GELU uses the same normalization-sized input; exact erf is scalar Python math
through NumPy, explaining its different cost from the array-only approximation.
The encoder uses an 8×16 token batch, width 32, four attention heads, a 32→64→32
feed-forward block, two residual/norm paths, tanh GELU, p=0.1 dropout and cross
entropy. A step includes zeroing, forward, backward and momentum SGD. These new
module workloads have no 0.6 module baseline and are not cross-library claims.

For frozen 64×128 input/weight with trainable bias, active-input cache refinement
reduces traced retained allocations from 73,151 to 41,015 bytes (float32) and
135,344 to 69,872 bytes (float64), approximately 44–48% versus the initial 0.7
implementation. Input allocations precede tracing. Peak forward allocations are
approximately unchanged (171 KB / 332 KB); this is a retained-tape saving, not
peak-memory reduction. Frozen input wrappers are released and bias gradients
remain correct. A permanent weak-reference contract checks that behavior.

## Validation

After rebasing through UI-only main commit `1507c19`, the maintained Python
suite (`pytest tests packages/paperleet-torch/tests -q`) passes: **2,051 passed**,
one optional Docker test skipped and five native Torch warnings, 137.83 seconds.
The lesson gate passes **69 modules, three repetitions, 8,829 executions and all
295 mutants caught**, 115.05 seconds. The UI-only rebase preserves all 29 library
source hashes; web checks and the production build pass after that rebase.
The Pages asset-retention suite also passes all 15 cases.

An earlier invocation incorrectly added raw `problems` tests without the judge's
execution plan and produced 548 setup errors alongside the same 2,051 passing
maintained tests. That log is retained; it is not counted as a successful full
run. The correct maintained invocation above then completes with no errors.

Final gate results and source identities are recorded in
[iteration-7.json](../../packages/paperleet-torch/benchmarks/results/iteration-7.json).
The native NN suite has 122 parameterized cases; 25 new NumPy-only NN contracts
run alongside the existing model/optimizer/conversion/scheduler lifecycle suites.
The maintained real-WASM checker now runs all five lifecycle files against the
exported runtime. Five fresh Chrome Workers assert the exported version and real
`/judge/paperleet_torch` module location, run 35 lesson tests, catch all 20 lesson
mutants and run 62 additional lifecycle contracts.

The independent wheel is tested outside the checkout with NumPy 1.26.4 and no
native Torch: 80 tests pass. It is 85,398 bytes with SHA-256
`159883974bfbdd9ae63fee8c7401578d43496184e19b99e1bc17e6f3c2a86c6b`.
Packaging and launcher tests also build both application/vendor and standalone
wheels, check disjoint installed paths, and run unchanged torch-importing scripts
from outside the checkout with argument and exit-code preservation.

Use [bench_nn.py](../../packages/paperleet-torch/benchmarks/bench_nn.py), or the
WASM runner's `--benchmark nn` option. Full logs, source freezes, memory samples
and browser/Node harnesses remain in local
`paperleet-wasm-evaluation-20261015/torch-iteration-7` evidence. Tracked published
assets are retained. The production backend remains CPU/NumPy; WebGPU is still an
experiment. No license has been selected and no PyPI release is made. The wider
independent-library/core/efficiency goal remains active in the roadmap.
