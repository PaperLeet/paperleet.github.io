# Torch runtime iteration 50 — Math attention, grouped heads and forward AD

Version 0.50.0 expands functional SDPA with grouped-query attention, dropout and
forward automatic differentiation while retaining a fused ordinary NumPy VJP.
The broader independent-library, core coverage, efficiency and optional GPU goal
remains active. Main adoption is recorded separately after local validation.

## Implementation

- Extract `_attention.py`; preserve the existing private oracle entry and the
  public `torch.nn.functional.scaled_dot_product_attention` signature.
- Grouped matrix products reshape query groups and broadcast original K/V heads.
  They do not first materialize K/V repeated to every query head. K and V head
  counts may differ, provided each divides Q heads. Backward reduces the grouped
  products into the original K/V shapes, including batch broadcasting.
- Pre-scale Q/K by the square root of the scale magnitude, with the sign
  applied to Q. This avoids overflowing the dot product before a reducing scale
  is applied. Selected large finite float32/64 cases cover positive, negative
  and zero scales through values, VJPs, JVPs and mixed derivatives.
- Keep one shared analytical pullback for ordinary reverse AD. Only active
  gradients are calculated; repeated seeds do not reuse cached adjoints.
- Add exact JVPs and reverse differentiation of both primal and tangent channels,
  including additive-mask primals/tangents, shared Q/K/V operands and mask views.
- Boolean/additive masks, rectangular causal masks, custom scales, float32/64,
  strided inputs and empty sequence/feature/head cases have explicit tests.
  A float32 additive mask is accepted with double Q/K/V and retains mask-gradient
  dtype. Explicit masks and causal mode remain mutually exclusive.
- Dropout draws one multiplier per invocation, shared by values, tangents and
  retained backwards. Probability zero/one does not consume random draws. The
  library RNG supports replay and checkpoint continuation; its seeded dropout
  bit stream is not claimed to match native Torch.
- A bounded 64-entry LRU caches immutable shape validation only. Tests cover its
  cap and prove that it retains no tensor objects or RNG state.

Attention scores/probabilities still grow with query length × key length.
Avoiding repeated K/V buffers is not a peak-RSS guarantee or FlashAttention.
This version has no GPU path, float16/complex attention, `nn.MultiheadAttention`
wrapper, nested JVP or general `create_graph=True` reverse AD support.

## Numerical evidence and the fully-masked mixed-AD boundary

| Check | Result |
|---|---|
| Complete repository/library CPU suite | 39,682 passed, 1 skipped; 365.17s |
| New independent attention suite | 161 passed |
| Native math attention comparison | 204 cases, 2,592 output/gradient arrays |
| Selected extreme finite-logit native comparison | 32 cases, 208 arrays, float32/64 values and selected VJP/JVP/mixed paths |
| Native complete encoder comparison | 200 arrays, four SGD steps per dtype, checkpoint/momentum continuation |
| Node/Pyodide 0.29.4 / NumPy 2.2.5 | 6,053 contracts, 47 suites |
| Actual Chrome 155.0.8059.82 Worker | 6,053 contracts, exact source/test identities |
| Installed wheel: Python 3.10.19 / NumPy 1.26.4 | 6,066 passed, native Torch absent |
| Public/private wheel and source archives | Exact 55 source/license resources; independent ownership/uninstall |
| Original native PaperLeet catalog | 78 problems, 10,218 executions, 342 mutants, three repeats; 468.378s / 600s |
| Complete exported WASM catalog | 78/78, 342 mutants, 6,053 contracts; 294.038s |
| Frontend and both build profiles | Five successful local receipts |
| Release policies | Seven files byte-identical to canonical `13d2782a494415ba72bc604a0ef7e4c4f875ba25` |

The native oracle is PyTorch 2.14.1 in separate processes, explicitly using its
math SDPA backend. Native CPU flash forward AD is not generally implemented;
this report does not claim equivalence to every native backend.

All official SDPA values, JVPs and first VJPs match the tested moderate finite
inputs, masks, scales, empty cases and deterministic dropout endpoints. Official
mixed derivatives match directly outside the fully-masked-row corner. In 496
mixed-gradient arrays across 100 cases, official math SDPA contains NaNs where
this library produces finite values; every finite official entry is also checked.
The complete 2,592 arrays match a separate native compositional math reference
that replaces fully-masked logits before softmax and then zeros their probabilities.
Original official outputs and the initial failing differential are retained.
These are explicit, classified differences rather than a claim that every array
matches official SDPA directly.

The stable rule is zero output/derivatives for completely masked rows with finite
operands/directions. Independent finite differences cover boolean and additive
fully-masked rows, all primal/direction inputs, shared operands and mask views.
The implementation retains that well-defined extension instead of reproducing
native mixed-AD NaNs. Negative or NaN dropout probabilities are rejected, although
the tested native math backend silently ignores them; valid probabilities are
restricted to [0,1]. Dropout distribution/replay is checked with independent fixed
Bernoulli masks; no native seeded-stream equivalence is claimed.

The complete encoder uses grouped causal attention with K/V head counts 1/2,
linear projections, residuals and LayerNorm. Its loss depends on the primal and
JVP. Native comparisons include all parameter/input/direction gradients, losses,
updated weights and momentum continuation after a checkpoint rebuild. A second
independent local flow adds attention dropout and compares grouped execution with
explicit head expansion through four mixed-AD training steps and checkpoint resume.

Float32 native tolerances are rtol 3e-5/atol 2e-5 for primitive arrays and
5e-5/4e-6 for full training; float64 uses 1e-11/8e-12 and 2e-11/2e-12.
No bitwise or universal extreme-logit/dtype compatibility is asserted.

## Performance

The portable `bench_attention.py` uses independent actual 0.49 and 0.50 snapshots.
Old-library grouped workloads keep singleton-head broadcasting and repeat only
non-singleton K/V heads that need expansion before invoking the existing fused
SDPA. A same-source fully repeated-head control is retained. Small/ordinary self
attention, grouped self/cross attention, longer grouped sequences and a complete
projected causal encoder with residual/GELU/SGD cover both dtypes. Values, every
VJP and three matching optimizer updates are checked before any timing.

Four fresh processes per CPU/Node/Chrome alternate runtime and method order.
Each runs five samples of eight calls over 22 flows; all samples and negative
results are retained. JVP is validated separately and is not a speed comparison
against an old fused feature that did not exist. Positive numbers mean less time.

| Runtime | Workload | Median reduction vs 0.49 | Range across flows/dtypes | Median vs same-source repeated control |
|---|---|---:|---:|---:|
| NumPy CPU | ordinary | -6.5% | -17.8% to 3.4% | -0.8% |
| NumPy CPU | GQA | -1.1% | -5.8% to 2.1% | 1.3% |
| NumPy CPU | encoder SGD | -4.5% | -5.2% to -3.7% | -0.4% |
| Node/Pyodide | ordinary | -4.0% | -14.9% to 3.2% | 1.8% |
| Node/Pyodide | GQA | -0.3% | -4.4% to 4.6% | 1.6% |
| Node/Pyodide | encoder SGD | -2.8% | -3.0% to -2.7% | 0.9% |
| Chrome Worker | ordinary | -0.3% | -14.3% to 2.0% | -0.3% |
| Chrome Worker | GQA | 0.1% | -7.1% to 3.5% | 0.6% |
| Chrome Worker | encoder SGD | -3.2% | -4.0% to -2.3% | -1.2% |

The measured GQA medians are approximately flat. Ordinary-attention medians are
0.3%–6.5% slower, and complete encoder SGD medians are 2.8%–4.5% slower. The worst
small ordinary CPU flow is 17.8% slower. These regressions are retained explicitly;
this revision expands local functionality and fixes numerical failures without a
general speedup claim. A separate fixed-input cProfile diagnostic is archived to
guide reductions in Python validation/setup overhead while retaining stable scaling.

Shared-host timing is not universal speed, native Torch/GPU speed, cold-start or
peak-memory evidence. The initial CPU pilot ran before the metadata cache and
under shared-host load; it is preserved but does not support final timing claims.

## Source, packaging and integration identity

There are 53 Python modules plus LICENSE/NOTICE. Fifty-one old resources are
unchanged; `__init__.py`, `_api.py` and `_nn.py` change and `_attention.py` is added.
Source/license bytes change from 647,717
to 655,234; local concatenated gzip
changes from 150,068 to
151,531. These source estimates
are not wheel/network bytes or measured memory. Existing dependency versions and
the lockfile are unchanged. Fourteen pinned Pyodide runtime files were rechecked.

All 10,861 published immutable payloads are preserved; 158 new reachable payloads
yield 11,019 total. The 161 active export identities match the frozen source.
Installed public/private namespaces execute grouped attention primal/JVP/VJP and
survive independent uninstall without native Torch.
Concurrent upstream personal-profile work was fast-forwarded as nine exact Git
blobs and rechecked with all five frontend/build gates. Its source hash record is
in `upstream_syncs`; frozen numerical sources/tests/exports are unchanged across
that UI-only update.

A first frozen run passed all source/package/frontend/CPU gates before an
extreme-input probe exposed the pre-existing post-dot scaling overflow. That run
and the probe are retained. Final source gates are rerun after pre-scaling was
added; the original native Torch catalog receipt is reused with all 1,152
catalog/runner/test/config input identities unchanged, because it executes native
Torch independently of the adapter.

The first Chrome invocation failed before Python execution because its local
fixture HTTP server was unavailable. After restoring that fixture, Chrome passed
on the same frozen source. An extra coinstall check was initially outside its
embedded child script; fixing that harness and rerunning all installation phases
passed. Both failure logs/receipts are retained; successful unrelated gates were
not repeated. Numerical source/test identities did not change for those retries.

Evidence is in `packages/stridetorch/benchmarks/results/iteration-50.json`,
`iteration-50-integration.json` and four matching raw harness archives. They retain
exact sources/tests, official and compositional oracles, original failures,
allocation-policy tests, training outputs, local command receipts and all timings.
No tag, workflow dispatch, release, PyPI upload or deployment is initiated here.
