# Torch runtime 0.82 development candidate

This is the historical development-stage record. The subsequent
[0.82 qualification](torch-runtime-0.82-qualification.md) resolves the pending
application/export/catalog gates below while preserving these original results.

This candidate adds ordinary `torch.nn.utils.clip_grad_norm_`,
`clip_grad_value_` and the deprecated `clip_grad_norm` alias, including the
`torch.nn.utils.clip_grad` import path. Existing host alias setup remains
unchanged. The candidate is local: main adoption, final application export and
catalog qualification remain pending. No CI, tag, publication or deployment is
requested. The terminal-HVP optimization is not included.

## Behavior and saved values

Clipping accepts a Tensor or parameter iterable, ignores missing gradients and
modifies existing gradient buffers under `no_grad`. Shared entries are processed
without deduplication. Non-contiguous storage and existing gradient history are
preserved. Nonfinite norm rejection precedes all gradient writes. Value-bound
conversion precedes consuming a generator. The three foreach hints use
per-gradient NumPy kernels; no fused or GPU backend is added.

Multiplication and direct division now guard only the inputs read by their
active pullbacks. Their public output is not a saved value. Required inputs and
saved-graph consumption remain protected. Actual Torch 2.6 controls expose 17
overconservative checks in the previous implementation; all 36 controls match
the candidate. Attention explicitly retains its existing conservative
original-input mutation contract. This does not claim native SDPA mutation
parity: a separate native observation permits changes that this runtime guards.

## Completed functional checks

- An independent actual Torch 2.6 oracle covers 478 clipping scenarios. Both
  native NumPy environments and actual Node/Chrome WASM match all asserted
  numerical, dtype, shape, mutation, ownership, version and state checks.
- Native SDK: 12,298 passed with NumPy 2.5.4; 12,295 passed and three existing
  skips with NumPy 1.26.4. These counts include 71 new explicit tests and two
  dynamically collected feature checks for the new module namespaces.
- Actual Pyodide 0.29.4 / NumPy 2.2.5: 12,285 SDK tests pass in Node 22.22.1
  and Chrome 155.0.8059.82 Worker contexts, with no skipped or failed tests.
- Six public/private wheel, sdist and rebuilt-wheel artifacts contain the same
  66 runtime resources. Public metadata retains Apache-2.0, NumPy >=1.26 as its
  sole runtime dependency and the exact staged README. Both NumPy environments
  pass coinstalled-package clipping, HVP, packed-matmul and alias restoration
  checks.

The isolated WASM candidate still has a 0.81 version literal. Canonical 0.82
differs only in that literal; all other runtime resource bytes match. Canonical
0.82 SDK and installed-artifact checks use the final version. Final generated
application Worker/export checks remain pending.

The initial Node harness expected 12,283 tests and failed that count check despite
pytest returning zero with 12,285 passes. Collection comparison identifies the
two additional namespace feature checks, with no removed cases. Experiment 447
reconciles the complete existing result without repeating it. The initial
failure remains preserved; Chrome uses the corrected count from its first run.

## Secondary native version boundary

The same 478-case oracle on actual Torch 2.14.1 has three error-class differences:
unsupported complex value clipping raises `NotImplementedError` there and
`RuntimeError` in the pinned 2.6 contract implemented here. The other 475 cases
match every asserted check; there are no numerical or gradient-state
differences in this comparison. The three class differences remain recorded,
so this is not a claim of complete 2.14 parity.

## Application gate and test resource recovery

Running `pytest tests problems` passes all 33,634 application tests and skips
one optional Docker test, but the command fails on 721 exercise setup errors:
710 missing workspace implementations and 11 syntactically invalid workspace
test-fixture implementations. Those exercise tests need the reference/catalog
harness. This failed invocation is preserved, and the normal composite
regression command, `pytest tests packages/stridetorch/tests`, remains pending.

The terminated run's temporary data has a verified recoverable thin backup:
11,498 entries and 978,434,075 original bytes, backed by a 3,884,975-byte archive
and recorded immutable Git blobs. Failed partial backup evidence is retained.
The wheel fixture now omits raw benchmark archives that are not wheel inputs,
while retaining benchmark programs and all distributable sources. It also
checks every runtime resource byte in both installed namespaces. All 12
packaging/import integration tests pass with a 28 MiB generated fixture tree.
No test cases or artifact-ownership assertions are removed.

## Evidence

[Development results](../../packages/torchwasm/benchmarks/results/gradient-utilities-0.82-development.json)
and [verified raw archive](../../packages/torchwasm/benchmarks/results/gradient-utilities-0.82-development-harness.tar.gz)
retain the independent oracle, frozen candidates, failures, test sources,
SDK reports and six package artifacts. The archive contains 789 files,
289 unique payloads, 3,020,479 bytes; SHA-256
`bfbc7dd49416829da415f157f0015b4fb2ec2f75a6b122796124a3d9826f0a3e`.
Its replay manifest records external runtimes and path adaptation requirements.
