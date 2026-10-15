# Lightweight torch 0.13 validation

This milestone fixes mixed-dtype selection/concatenation, exposes `result_type`,
adds ordinary join aliases and removes unnecessary branch storage retention from
`where` backward graphs. The library remains NumPy-only and keeps existing torch
imports through the host/launcher aliases.

## Selection, promotion and graph contracts

`result_type` and three-argument `where` distinguish dimensioned Tensors,
zero-dimensional Tensors and wrapped Number arguments, following native category
priority instead of NumPy widening. For example float32 Tensor + int64 Tensor in
`where` stays float32. `cat`/`stack` promote every input, preserving imaginary
components of complex Tensor inputs rather than casting everything to the first
input's dtype. Supported native overload keywords, `axis` aliases and `out=None`
are checked; real output buffers remain explicitly unsupported.

Scalar Number argument conversion differs from sequence construction. Native
NumPy bool/complex64 Number arguments use their float protocol, including the
complex-to-real warning; constructing complex Tensors still preserves imaginary
values. Scalar bounds/overflow are checked before both forward- and reverse-mode
execution. Casts isolate and restore the caller's NumPy floating-point settings.
NumPy has no ComplexHalf storage: promotion requiring native complex32 raises an
explicit unsupported-operation error. This is not emulated by silently widening.
The public dtype support does not imply complex reverse-mode differentiation.

One-argument `where` returns int64 index tuples, including zero-dimensional and
empty conditions. Boolean conditions and deprecated uint8 conditions are checked,
with a once-per-runtime warning for uint8. Wide unsigned one-argument indices
remain an explicit unsupported boundary, consistent with native CPU nonzero.
`Tensor.where`, `torch.concat` and `torch.concatenate` are ordinary aliases.
Empty one-dimensional cat inputs participate in promotion but may be skipped in
shape concatenation and receive correctly shaped empty gradients.

Where VJPs save condition data and shape metadata, without closing over branch
Tensor values. Only the condition version is checked. Mutating a leaf branch or
where output does not invalidate where's derivative; mutating the condition does.
Existing upstream operators retain their separate conservative mutation guards.
Default backward frees saved state; retained backward and selected reverse-over-
forward cases are tested. Mutation violations keep the library's existing typed
unsupported-operation fallback policy rather than claiming native exception parity.

Native-reference tests use Torch 2.14.1 and compare dtype, values, argument/error
behavior, VJP, JVP, duplicate edges, broadcasting and lifecycle effects. Independent
NumPy-only workflows also train a masked two-layer model with standard torch
imports against manually computed NumPy gradients and updates. Course tests are
integration checks, not the sole oracle for these core APIs.

## Performance protocol

Baseline is released 0.12 commit `9b1822ee0e307e5b1d5e5cafc160514bd54a3a9b`.
All 33 baseline and 34 candidate Python sources and two license resources are
hashed. Harness hashes are checked after measurement. Local correctness/build
producers are terminal before sequential timings. CPU and Node both use balanced
baseline→candidate→candidate→baseline order per dtype/suite (32 processes each).
There are eight separate memory processes and eight completely fresh Chrome
processes/Workers in the same balanced order.

CPU uses Python 3.12.11/NumPy 2.5.4/macOS arm64/Apple Accelerate, with thread controls
set before NumPy imports and Accelerate's single-thread policy queried before and
after. WASM uses pinned Pyodide 0.29.4/NumPy 2.2.5; Chrome is version 155. There are
11 batches, three warmups and 120 calls for selection/core cases (30 for the
16-layer where chain). Prior 24 construction cases and five graph/hook controls
retain their own recorded counts. Full samples, counts and outliers are retained.
Fixtures/assertions run outside timing, while operation/allocation/cleanup runs
inside. Changed mixed-dtype semantics are validated separately, not compared as
an alleged speedup. A float scalar, rather than the previously mis-promoted int
scalar, ensures the two measured where outputs have identical dtype and shape.

The initial and optimized exploratory CPU protocols remain separate from final
measurements. All raw samples and sources are preserved. Timing noise and clock
resolution limit small percentage comparisons; this is not a broad speedup claim.

## Validation and distribution

The initial full suite passes **8,850 tests**, one optional Docker skip and seven
native warnings in 154.65 seconds, including 71 original modules and their mutants.
Real WASM passes 116 standalone contracts across 11 files, seven full maintained
reference/mutant/learner suites and seven fresh Chrome Workers. Chrome checks
54 original tests, all 37 representative mutants and the 116 contracts, alongside
runtime version/import location/source/resource hashes. Web checks and production
build pass.

The independent wheel installed outside the repository passes **136 tests** on
NumPy 1.26.4 without native Torch. All 34 installed/wheel/sdist source hashes and
both canonical Apache-2.0 LICENSE/NOTICE copies are checked. Initial wheel identity
is 113,838 bytes, SHA-256
`13957fc83f0a38749bc78757abaefcdc7d48ca553676792a07c4ca85e4213355`.
Documentation metadata may change at integration; runtime source identity is the
34-file frozen hash set.

## Measured benefits and costs

In the 16-stage where CPU fixture, forward retained traced allocations drop from
1,149,994 to 73,514 bytes for float32 (−93.6%) and 2,230,806 to 105,750 for float64
(−95.3%). Default-backward peaks drop from 1,226,362 to 205,304 and 2,372,718 to
369,144 bytes. Explicit retain_graph and second-pass release also preserve the
same values/gradients. The separate standard graph fixture remains byte-identical
to baseline. These are CPU tracemalloc allocations, with input/condition storage
allocated before tracing, not browser memory or process RSS.

CPU where forward changes 5.18→5.58 µs / 5.93→6.14 µs (+7.9%/+3.4%). Float-scalar
where changes +4.5%/−1.0%, where backward −0.8%/−1.9%, and the 16-stage backward
chain +0.6%/−2.5%. Cat forward changes −7.4%/−4.1%, stack forward −4.2%/+0.3%,
cat backward +3.7%/+3.5% and stack backward +1.0%/+0.8% (float32/float64).

Chrome where forward changes +8.3%/0.0%, where backward +2.4%/+4.4%, and where
chain backward 0.0%/+4.3%. Cat/stack forward improve about 3–7%; their backward
calls cost about 1–6% more in the original series. These small timings have coarse
clock resolution. Node where forward changes +6.1%/+10.2%, while other selection
cases vary around parity or improve. These are compatibility/memory improvements
with explicit common-call costs, not a broad latency speedup.

CPU attention, MLP, AdamW and graph controls stay within about 2%. Original Chrome
float32 attention/MLP/AdamW change +0.3%/+0.3%/−1.1%. Chrome float64 attention has
a material original +22.7% change: baseline process medians 2.812/2.760 ms versus
candidate 3.350/3.485 ms. Four additional fresh Chrome processes repeat the exact
immutable Worker in reverse balanced candidate→baseline→baseline→candidate order:
candidate 2.764/2.763 ms and baseline 2.808/2.863 ms. The latter series does not
reproduce the slowdown. Both series and all batches are retained separately;
no sample is filtered or pooled and no resolved cause is claimed. Original Node
float32 baseline attention/MLP and both versions' float64 AdamW also have process
excursions. Further variance investigation remains useful.

The corrected initial native audit covers 784 where combinations, 196 cat,
392 stack and 784 result_type cases. Cat/stack match all tested native results;
where/result_type each retain four explicit ComplexHalf boundaries. The prior
310/70/140 where/cat/stack mismatches are removed in that audit. Final focused
and full tests independently cover the optimized source. The initial audit is
labeled with its own source hashes and reproducible intermediate patch.

All final/prototype samples, source/resource/harness hashes, test/harness source,
checks and reconstruction proofs are in
[iteration-13.json](../../packages/paperleet-torch/benchmarks/results/iteration-13.json).
The initial 34-file prototype can be reconstructed from final source with the
[verified reverse patch](../../packages/paperleet-torch/benchmarks/results/iteration-13-prototype.patch).
The optimized prototype is byte-identical to final source; its empty diff is
explicitly recorded. Full local frozen artifacts/logs remain in
`paperleet-wasm-evaluation-20261015/torch-iteration-13`.

The independent-library goal remains active. Remaining operator-wide promotion,
large-sequence costs, broader NN coverage and resident WebGPU need separate work.
This milestone adds no GPU execution mode and makes no PyPI publication.

## Integration with current main

The implementation is rebased onto
`abab5acde7731f0c1ba2552665b1ec1250d06cdc`. Upstream adds bilingual community
writing guidance and six browser-first training/generation exercises. All 34
timed source hashes remain unchanged. Catalog conflicts are resolved by canonical
export, preserving every upstream exercise and all published hashed assets.

The combined 77-module full suite passes **8,856 tests**, one optional Docker skip
and seven native warnings in 157.34 seconds. Native CI runs all 77 modules with
three repetitions, 9,987 executions and 336 mutants, passing its reference/
mutation/determinism/CPU-budget gate in 118.10 seconds.
The maintained real WASM gate now passes **13** complete module suites. Thirteen
fresh Chrome Workers pass every reference/calibration suite, retaining all 116
standalone contracts. Combined web checks and production build pass.

After documentation updates, the independent wheel is rebuilt and reinstalled
outside the checkout. All 34 runtime hashes, canonical Apache LICENSE/NOTICE and
136 minimum-NumPy tests pass again. Final metadata wheel: **114,679 bytes**,
SHA-256 `89e44f0c587b4ebde631bfac16a67814957a452c3f7a10e5f2499fd0dd267edf`. Earlier wheel identity remains labeled above.

Published asset retention and the exact local integration receipts are included
in the raw record. Remote CI/Pages and exact live-asset verification are recorded
after the main push; local green checks do not imply a completed deployment.

A subsequent UI-only main update is integrated at
`d47b6d6256d5d00a3af9931b57ab5f77e90152bf`. It changes interview preparation
prompts and sign-in/navigation returns; numerical/application runtime sources,
problem catalog and benchmark harnesses remain unchanged. Canonical export,
25 affected catalog tests, all frontend checks and production build pass again.
The original 77-module numerical gates remain tied to their earlier base above.
All **4,390** upstream published hashed assets are byte-identical; no unpublished
unreachable assets require removal. Official Pages retention/backfill checks pass.
Thirteen Chrome suites cover 95 original tests and catch 61 representative mutants,
in addition to 116 standalone contracts.
