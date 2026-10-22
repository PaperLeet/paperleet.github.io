# Lightweight torch 0.17: GroupNorm

This iteration adds `nn.GroupNorm`, `nn.functional.group_norm` and the ordinary
`torch.nn.modules.normalization.GroupNorm` alias. NumPy remains the only runtime
dependency. Apache-2.0 redistribution resources and the standard import launcher
remain part of both independent and application distributions.

## Behavior and numerical policy

The shared NumPy pullback computes active input/weight/bias gradients once per
traversal. It supports real float32/float64, ranks two and above, per-channel
optional affine parameters, Fortran/strided inputs, empty batches, retained
backward and frozen parameters. Input/weight storage versions are guarded;
bias/output mutations do not invalidate a saved graph. Saved normalized values
are separate from mutable output storage. Storage rebinding remains an explicit
unsupported boundary.

GroupNorm has no running buffers: train/eval share the same computation.
Registered affine parameters support reset, dtype conversion, state dictionaries
and Adam checkpoint/resume. Zero channels, empty spatial dimensions, invalid
channel/group divisibility and Python functional singleton validation are checked.
Float16/complex normalization and general differentiable in-place mutation remain
outside the supported contract.

Native PyTorch 2.14.1 audits found an important forward-mode distinction: GroupNorm
**differentiates the mean and variance** on the reverse-over-forward tape.
Reshaping into the existing LayerNorm kernel gives correct JVP values but wrong
input gradients through that JVP. The initial native/composite audit had a maximum
input-gradient difference of 8.3304. An independent full-statistics formula
matched the native reverse tape to 9.77e-15. The implementation and independent
finite-difference tests use the full-statistics rule.

Float32 centering selectively widens ill-conditioned groups using the established
LayerNorm policy. The inverse rounds once from channel/group statistics; outputs
and gradients keep the input dtype. Centered formulas intentionally avoid native
CPU affine-expansion cancellation. Seed1712 audits against independent centered
float64 mathematics record the differences rather than relaxing every native test:

| Input | Library output / input-gradient max error | Native output / input-gradient max error |
| --- | --- | --- |
| `(3,6)`, 3 groups | 2.58e-6 / 1.07e-4 | 2.70e-6 / 4.15e-4 |
| `(2,6)`, 6 groups | 0 / 0 | 2.35e-5 / 1.53e-5 |
| `(2,6,4)`, 3 groups, offset1e5 | 2.85e-7 / 1.71e-6 | .1487 / .9777 |

The first native run retained 14 failures: twelve float32 tiny-group comparisons
and two test-helper attempts to call native `.numpy()` while gradients were
active. The helper was corrected; ordinary native comparisons retain strict
1e-5 float32 / 8e-12 float64 tolerances, while independent tiny-group math checks
bound subtraction error by input precision, inverse scale and seed magnitude.
The original failed log, test file and numerical audit are retained in the local
iteration evidence and reproducibility archive.

## Verification

- 137 new native-reference checks and 47 independent NumPy-only contracts pass.
- Full infrastructure: **17658 passed, 1 optional Docker skip, 7 warnings**;
  local elapsed203.05s. These are correctness observations, not a speed comparison.
- Full standalone collection plus new native checks:477 passed.
- Actual Node/Pyodide0.29.4 and Chrome155.0.8059.82 Worker: **320 contracts** each,
  NumPy2.2.5, source/test hashes checked inside the VM.
- Isolated installed wheel: Python3.10.19 / NumPy1.26.4, no native Torch,
  **340 standalone tests**; wheel and sdist contain all37 modules and both notices.
  Independent and private vendored package paths are disjoint and all source bytes
  match the browser/installed artifacts.
- Frontend checks and production build pass; initial JS graph remains
  404405bytes raw /130081bytes gzip before merging later upstream UI changes.
  Complete checks/build reran after inbox UI commit9ca363a2; final initial graph
  is404439bytes raw /130109bytes gzip. Numerical/runtime inputs stayed identical.

- Native catalog: **77 problems / 9987 executions / 3 repetitions /336 mutants**,
  no issues;139.727s. Complete real WASM: **77/77**,412.841s, no budget excess.
- All77 visualization exports passed. All5494 previously published immutable
  assets remain byte-identical; all156 new hashed assets are reachable.
- Full local proof, original probes, every raw performance sample and package/
  browser identities are in [iteration-17.json](../../packages/torchwasm/benchmarks/results/iteration-17.json).
  Original runners and frozen sources are in `iteration-17-harness.tar.gz`.

Exact-commit remote adoption subsequently passed as recorded below. The new
runtime bundle is8400bytes larger raw and2099bytes larger gzip than0.16, measured
with identical gzip settings; this is transfer size, not cold-start latency.

## Performance method

`bench_groupnorm.py` compares the shared first-order NumPy kernel with an explicit
Tensor composition using the **same library**. It covers tiny/MLP/vision/volume
training, affine-only and bias-only backward, and inference. It validates outputs
and every active gradient before timing. CPU and real WASM use balanced fresh
processes in composite/joint/joint/composite order, after all local correctness
and build producers terminate. Source, notices and harness files are frozen and
hashed before/after every process. All batches and process medians are retained;
there is no native-Torch, previous-release or full-application speedup claim.

| Case | CPU float32 | CPU float64 | WASM float32 | WASM float64 |
| --- | ---: | ---: | ---: | ---: |
| tiny_train | 37.5% | 41.8% | 40.0% | 43.2% |
| mlp_train | 22.9% | 22.0% | 21.3% | 21.4% |
| vision_train | 16.8% | 25.8% | 24.5% | 25.6% |
| volume_train | 30.7% | 28.9% | 31.9% | 34.3% |
| mlp_affine_only | 16.2% | 16.1% | 14.7% | 17.8% |
| mlp_bias_only | 12.1% | 14.4% | 12.3% | 16.9% |
| mlp_forward | 20.9% | 23.4% | 20.7% | 28.1% |

Cells show reduction in the median of fresh-process median elapsed times. Each
process retains9batches,80base iterations,3warmups; tiny cases multiply iterations
by4 and vision cases divide by8. CPU and WASM results are separate series. All
local correctness/build producers were terminal before the16measurement processes;
source, resources and harness hashes remained invariant throughout. No sample was
discarded. The extra browser transfer size and unmeasured cold-start/JVP costs
remain separate considerations.

The previous0.16release is now fully remote-verified; its proof is carried with
this implementation rather than a separate documentation-only CI cycle.

Final integration preserves upstream inbox changes from `9ca363a2`.
[Integration evidence](../../packages/torchwasm/benchmarks/results/iteration-17-integration.json)
records source/hash invariance and the complete post-rebase frontend rerun.

## Exact remote adoption

[Remote proof](../../packages/torchwasm/benchmarks/results/iteration-17-remote-adoption.json)
records commit `cea90293d861b299a2779fe5283a02c813f15e2d`, successful
Quality38843241977 and Pages38843241942, and all77 live starters/details/manifests
plus156 byte-identical immutable assets. Both runtime profiles contain0.17 and
Apache-2.0 notices. Remote infrastructure:17658 passed,1optional Docker skip,
10warnings in370.49s. Native catalog:77/9987/3repetitions/336mutants,302.558s.
Actual WASM:77/77,908.086s, no budget excess. Flow Matching calibration222.668s,
learner44.519s, total271.224s remain within240/60/300s. These are correctness and
shared-runner budget observations, not controlled source performance comparisons.
