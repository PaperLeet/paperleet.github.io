# StrideTorch 0.28: matrix and axis views

This iteration expands ordinary matrix/axis APIs and audits their shared view
primitive against native PyTorch2.14.1. The baseline is0.27 at
`f1b33b0be7b47bb9e06ff92ba60a5032fb8cea06`. The independent distribution remains
`stridetorch` under Apache-2.0, with unchanged ordinary `import torch` aliases.
No independent repository/package publication is implied.

## Supported behavior

`Tensor.t()` and `torch.t` support scalar/vector views and matrix transposes;
higher ranks are rejected. `Tensor.mT` swaps the last two matrix axes of a batch.
A scalar `.mT` preserves object identity with a deprecation warning; vectors are
invalid. Transposes do not conjugate complex values. Native guidance is available
in the [t API](https://docs.pytorch.org/docs/2.14/generated/torch.t.html) and
[movedim API](https://docs.pytorch.org/docs/2.14/generated/torch.movedim.html);
the executable native oracle also records overload details beyond those summaries.

`movedim/moveaxis` and `swapaxes/swapdims` are present as both methods and functional
entries. `swapaxes` uses `axis0/axis1`, whereas `swapdims` uses `dim0/dim1`.
Movement requires two scalar integer forms or two integer sequences; mixed
forms, unequal lengths and duplicate normalized axes fail. Sequence elements
retain native index conversion; the scalar overload does not accept arbitrary
`__index__` objects or NumPy arrays. Bool/non-scalar integer Tensors are rejected
as scalar axes but follow the native sequence-element rules. Signed-int64
overflow and type-versus-bounds exception priority are independently compared.
Only numeric axes are in scope; named/symbolic tensor systems are not introduced.

An audit found that the existing `transpose` primitive accepted several scalar
axis forms rejected by native Torch. The fix validates both uncommon scalar
arguments before bounds checks. Ordinary in-range Python integer axes use an
inline path that reads rank once, avoiding the prior repeated properties/helpers.
This tightens invalid-input behavior without copying values or changing valid
permutations. New aliases reuse that primitive, including its view/version rules.

All ordinary results share storage and mutation versions. Real float16/32/64
reverse gradients and float32/64 JVP/reverse-over-tangent use the existing view
engine. Tangents may be materialized according to its layout rules. Complex
inference transposes work; complex differentiation, conjugate view bits, general
higher-order graphs and new device backends remain outside this change.
User lists and integer-Tensor axis arguments become immutable metadata at forward
time and are not retained by the tape. Detached view writes preserve mutation
guards for value-saving computations; pure permutation pullbacks save no values.

## Metadata/performance choices

The selected inverse-permutation cache stores at most32 mappings between tuples
of plain integer axes. It holds no tensors, shapes, input values, seeds, gradients
or graph nodes. Public parsers normalize user objects before cache access. Tests
exceed the capacity with120 permutations, verify its bound, release argument
payloads, vary input shapes and reuse different backward seeds. The ordinary
`permute` forward path stays identical to the baseline.

Initial eager per-node inversion improved some repeated VJPs but regressed
recorded forward time9–20%. Lazy per-node caching still introduced extra forward
work and closure state. Both implementations and all raw results are retained;
the bounded shared geometry cache avoids those per-node changes.

Five fresh sequential ABBA phases retain80 CPU/WASM timing children across
float32/64. The final source includes the stricter/inline transpose path and
measures only APIs present in the baseline: `permute` and `transpose`. New API
names are not assigned speedup claims. Inputs/seeds/derivatives assert actual
dtypes, and every child checks45 source/resource hashes. Final extended batches
use100 warmup calls,13 batches,1500 metadata/VJP calls and800 small/200 large
square/sum/backward calls. CPU requests two BLAS threads, actual BLAS threading
is unverified. No owned tests/builds overlap timing; the shared host is otherwise
uncontrolled. These are latency observations, not cold-start/application claims.

Final fresh-process transpose forward observes13.2–17.6% lower time without
recording and mostly9.2–12.3% lower time with recording. One WASM float64 rank-six
recorded-forward sample instead regresses5%. Permute retained VJPs improve about
1.6–4.9% in these samples; this does not establish an overall training speedup.
Complete square/backward results are mostly near controls. All earlier regressions
remain visible, including10.8/23.2% short-run training regressions and a47.8%
fresh-process WASM float64 rank-six no-grad permute regression. The latter path's
implementation is unchanged; that fact alone does not discard its measurement.

To investigate those anomalies, eight additional fresh CPU/WASM processes load
the two packages in separate namespaces sharing one NumPy interpreter. Each
targeted case uses1000 warmups and13 paired ABBA/BAAB blocks of2500 calls. The
rank-six no-grad permute medians range0.0–0.8% lower time; rank-six recorded
transpose10.0–11.3% lower; rank-three no-grad transpose13.2–15.8% lower. Exp
controls vary about−0.9–1.9%. The large permute anomaly is not reproduced in this
targeted paired check. This is additional evidence rather than a replacement
for the original fresh-process results, and no uniform performance guarantee
or causal attribution to host noise is made.

## Validation and release gates

Native inference covers seven dtypes (including complex), scalar/vector/matrix/
batch/empty inputs and C/F/gapped layouts. Linear VJPs after detached storage
mutation, repeated backward, gradient identity, JVP layout, reverse-over-tangent,
no-grad view flags, scalar object aliases and schema/exception priority are
compared independently. Additional rank-four/five cases verify multiple unmoved
batch axes and tangent layout. NumPy-only tests exercise updates, inner-product
adjoints, cache bounds/ownership and intermediate value release.

An initial test draft produced NumPy complex scalars instead of ndarrays and
attempted directly mutating differentiable non-leaves, an existing unsupported
mode. The fixtures were corrected to ndarrays and detached shared-storage writes;
the187-failure draft log is retained. Those fixes do not extend in-place autograd.
The final focused shape/axis corpus passes3741 tests. The initial complete source
run passes29103/one optional Docker skip. After the transpose-schema fix, the
complete final source passes29402 tests, with one optional Docker skip and34
warnings in190.93s. Official original native catalog CI passes78 modules,10218
test executions,342 mutants and three repeats in115.91s without issues. It uses
native Torch; actual shim integration is a separate exported-WASM gate.

The final actual wheel installs in Python3.10.19/NumPy1.26.4 without native Torch
and passes893 tests. Node and headless Chrome155 execute pinned
Pyodide0.29.4/NumPy2.2.5 against the same45 source/license hashes, each passing881
tests in28 suites. Public/private wheels have disjoint ownership and each remains
usable after uninstalling the other; both namespaces execute the new matrix/axis
APIs and retained gradient workflows. Earlier pre-transpose-fix proofs remain in
their own `proof` folder; the adopted ones are in `proof-final`.

Original native catalog, immutable export/full exported WASM, latest-main frontend
and application distributions, non-force adoption and actual remote Quality/
eligible Pages/live verification remain separate integration gates. Passing the
independent library corpus alone is not used to claim those broader outcomes.

[Core evidence](../../packages/stridetorch/benchmarks/results/iteration-28.json)
includes all prototype/adopted sources,88 timing children, native probes,
schema-fixture corrections, independent tests, minimum distributions and actual
Node/Chrome receipts. Adopted source hashes refer only to `trial-transpose` and
`proof-final`; earlier variants are explicitly retained as trials.


## Application integration

The original native catalog gate passes78 modules/10218 executions/342 mutants
with three repetitions in115.91s. After merging main73c777f9, all33 incoming
UI/docs/build/data files are preserved byte-for-byte. Its interview catalog adds
the policy-distillation route;107 affected catalog/interview/browser-judge tests
pass in115.49s. Full final core tests before this merge pass29402 with one optional
Docker skip. No upstream problem/test/judge Python inputs changed.

The exported shim passes all78 modules/342 mutants in219709ms with concurrency2,
including881 standalone checks in28 suites with frozen source hashes. Original
per-case and aggregate budgets remain unchanged and none is exceeded. These
catalog durations establish budget compatibility rather than speedup. All7230
published immutable payloads remain byte-identical;158 new reachable payloads
plus three mutable indexes are frozen for later live verification. Both runtime
profiles carry Apache-2.0 LICENSE/NOTICE. The autograd runtime is479313 raw bytes
(local gzip estimate111120); the schedule profile is37343/11550.

The application wheel/sdist contain the same45 core/resource hashes as the
independent0.28 wheel and the latest application catalog/server bytes. Their
installed Python paths remain disjoint. Owned dependencies reuse the successful
iteration27 npm-ci tree: lockfile and all14 Pyodide files are rechecked exactly.

Complete frontend checks pass57 bilingual preparation guides/171 prompts,
reading tabs, inbox and math-security contracts. Default and actual hosted
Pages-profile builds each have three initial JS assets, respectively402515 raw /
130018 gzip and403917 raw /130481 gzip bytes, within existing budgets. The actual
profile comes from the repository variable. No manual visual browser review is
claimed. The React checklist review preserves upstream account-key isolation,
listener cleanup, lazy loading and keyboard guards without unrelated UI edits.

[Integration evidence](../../packages/stridetorch/benchmarks/results/iteration-28-integration-gates.json)
and its205-file archive retain exports, current artifacts and logs. An initial
nonexistent pytest path, receipt-verifier working-directory/syntax corrections
and truncated terminal receipts are recorded explicitly. The affected-test log
records success; no missing command exit code is invented. Non-force adoption,
actual exact-main Quality and eligible Pages/live verification remain separate.


## Actual remote adoption

Main0fb0bdc57732aadb8233d3ec9d246570ce088c58 passes all four required jobs in
[Quality38919796621](https://github.com/PaperLeet/paperleet.github.io/actions/runs/38919796621).
The infrastructure log reports29402 passed/one optional Docker skip in318.06s.
Original native CI passes78 modules/10218 executions/342 mutants/three repetitions
in430.66s; exported WASM passes78/342 with no budget violations in522141ms. Its
28 independent suites/881 checks have the frozen test hashes and every case
retains its source, packages, test counts, mutant counts and original budgets.

The same exact source commit completes actual deployment steps in
[Pages38921426204](https://github.com/PaperLeet/paperleet.github.io/actions/runs/38921426204).
Live verification passes all78 decoded starters/current detail and manifest
references and158 immutable payloads, byte-for-byte, including both runtime
profiles and Apache-2.0 LICENSE/NOTICE. No equivalent-descendant substitution is
needed for this release. Version0.28 is positively verified on the site.

[Remote adoption evidence](../../packages/stridetorch/benchmarks/results/iteration-28-remote-adoption.json)
and its13-file archive retain actual job/deploy steps, downloaded reports,
infrastructure output, frozen hashes and the live checker. A first artifact-name
typo is recorded; the actual plural browser-wasm-reports artifact was discovered
through the API and downloaded successfully. Runner durations are compatibility
and budget evidence rather than a cross-run performance comparison.
