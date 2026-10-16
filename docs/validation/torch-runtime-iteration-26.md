# StrideTorch 0.26 candidate: indexing and value selection

This candidate adds functional `torch.gather`, fixes dense gather's scalar,
empty-index and dtype/dimension behavior, and extends gather/sort/topk and
argsort through validated float32/64 JVP. Named outputs retain their type and
integer zero tangents. Sparse gradients, complex/half JVP and output buffers
retain explicit boundaries; a sparse flag is accepted for unrecorded forward.

Native 2.14.1 probes expose a concrete half-gradient bug: 1,000 duplicate indices
with half seed 0.1 produce gradient 100 natively, versus 105.1875 previously.
The new pullback accumulates half scatter-add in float32 and rounds once.
Index values are checked again during backward because raw NumPy writes can
bypass Tensor version counters; negative indices must not silently wrap.

Gather retains native input/index version guards but does not guard its output.
Sort/topk guard only saved indices, allowing supported input/output mutation.
Their pullbacks retain shape/dtype and selected indices, without retaining
intermediate input values solely for derivatives. Gather coordinates use
broadcast vectors, replacing full rank-by-selected-size grids. Weak-reference
tests check intermediate release; narrow topk index storage owns exactly its
selected elements. These checks do not establish process peak memory.

Sort/TopK namedtuple classes are cached. Wide, small-K topk uses partitioning
followed by sorting only the selected values when requested. Dense/small axes
retain full sorting. Tie identities are unspecified; VJP/JVP follow returned
indices. Stable sort preserves equal-key order, including NaNs and unsigned or
minimum signed values. Stable overloads require keyword dimensions; explicit
None/bool dimensions and invalid flags are rejected.

The independent implementation follows installed native probes and the primary
[gather documentation](https://docs.pytorch.org/docs/2.14/generated/torch.gather.html),
[topk contract](https://docs.pytorch.org/docs/2.14/generated/torch.topk.html),
[gather and selection derivatives](https://github.com/pytorch/pytorch/blob/v2.14.1/tools/autograd/derivatives.yaml).
No native Torch is installed in the NumPy-only or browser validation environments.

## Correctness and distributions

Final source passes 26,803 infrastructure tests (one optional Docker skip,
35 warnings, 183.76s). Earlier source passes the same count in 178.68s; it is
retained separately. Focused native/indexing/model-conversion tests pass 520 on
each optimization trial. Coverage includes scalar/empty/strided inputs, int32/64
indices, duplicate and NaN selections, strict overloads, input/output/index
mutation, raw index writes, dtype conversion, retained replay, named JVP outputs
and reverse-over-tangent. Independent tests check directional differences,
adjoint duality, compact owners and storage lifetime.

Final-source Node and Chrome each pass 821 contracts on Pyodide 0.29.4 / NumPy
2.2.5. A fresh Python 3.10.19 / NumPy 1.26.4 wheel passes 833 contracts with
native Torch absent. All 43 Python modules and LICENSE/NOTICE match the frozen
source. Public/private wheels coinstall, execute the new API and survive
uninstalling the other owner without any file changes.

## Four retained timing phases

Four balanced phases preserve all 64 fresh sequential children, each using ABBA
order across CPU/WASM and float32/64, 11 batches and three warmups. Matching
valid method workloads use 500 small calls or 25 row/column/reverse calls per
batch, with 4x8192 and 8192x4 arrays and K=8/4096. Repaired half/invalid cases,
new functional gather and JVP are excluded from comparisons against 0.25.
Post-timing fixture checks also verify small/dense/unsorted values and indices.

The first correct candidate regresses gather by 10–27%. A module-reference/dtype
cache trial removes repeated import work, and a header-cache trial reduces
repeated metadata reads, but rows still regress. The final adopted direct
coordinate indexing path keeps every validation check while avoiding the
already-validated general NumPy wrapper. Every trial is retained rather than
replacing failed/regressed measurements with a later record.

Final results relative to 0.25: small sort improves 79–82%, small topk 71–75%,
wide small-K topk 67–74% on CPU and 89–91% on WASM, including unsorted output.
Dense-K topk improves 7–14%; topk reverse improves 59% CPU / 81–82% WASM, and sort
reverse 23–27%. Small gather improves 4–15%, row gather 7–12%. Gather reverse
ranges from 2.6% slower to 8.1% faster; this remaining CPU float32 regression is
retained. Unchanged exp varies from 2.0% slower to 3.0% faster, so small changes
are not causal claims. CPU requests two BLAS threads; actual threading is
unverified on the shared host. No local tests/builds/audits overlap timing.
These are steady-state library measurements, not application, native Torch,
cold-start or peak-memory claims.

[Evidence](../../packages/stridetorch/benchmarks/results/iteration-26.json) and
the [reproducibility archive](../../packages/stridetorch/benchmarks/results/iteration-26-harness.tar.gz)
retain frozen sources, all phases, original failed Native fixture calls,
corrected tests and distribution gates.

Application catalog/export/history, latest upstream parallel-CI integration,
complete exported WASM, non-force main adoption and deployment remain gates.
An existing broader leaf-gradient layout difference remains engine work;
this release does not claim full memory-format compatibility. Main's deployed
0.25 remains independently verified while this candidate is validated.
