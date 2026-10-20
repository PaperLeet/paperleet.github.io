# Lightweight torch 0.20: real storage views and layout queries

This candidate separates shape-only `Tensor.view`/`view_as` from `reshape`, adds
`stride`/`element_size`, and preserves shared version/copy behavior across NumPy
stride wrappers. NumPy remains the only runtime dependency. Ordinary Torch
imports through aliases/launcher and Apache resources retain their contract.

## Contract and mathematical checks

`view(*shape)`/`view(size=shape)` infer at most one -1 and create storage-sharing
metadata only when the original stride chunks permit the target layout. Incompatible
transposes, stepped axes and broadcasts reject instead of silently copying;
`reshape` remains allowed to copy. `view_as` uses only the other Tensor's shape.
Strides report actual NumPy backing metadata in element units; general computed
layouts need not match every native Torch layout choice. Imported empty strides
are preserved, while fresh supported dense factories use conventional empty
strides, including zero-element randperm without advancing RNG state.

Reverse view derivatives reshape logical seeds and do not guard values they do
not need. Detached view writes still bump the shared storage version, invalidating
upstream operators that saved it. Wrapper bases resolve to the original allocation
for version tracking, deepcopy and module rebinding. Independent checkpoints copy
an allocation once and preserve view aliases inside the copy.

JVPs keep analytical reverse tapes. Compatible tangent layouts share storage;
otherwise a nonoverlapping positive primal layout can receive a differentiable
copy before viewing. Materializing a different tangent into potentially overlapping
or negative-stride storage remains explicitly unsupported. Dtype-reinterpreting
views, public arbitrary as_strided/storage-offset APIs, channels-last and general
higher-order/in-place autograd are not added by this iteration.

The current collection includes1431 new native-reference cases and55 independent
NumPy/scalar contracts. An initial55-case audit found20 silent-copy mismatches;
the corrected prototype matched1028 native geometry/value/stride observations.
Expanded native cases include inferred dimensions, dtype and argument rejection,
empty/singleton/Fortran/stepped/broadcast inputs, retained seeds, JVP/reverse and
version guards. Independent coordinate formulas, scalar derivatives, checkpoint
aliases and object lifetime supplement the native oracle. The focused collection
with existing conversion/input tests passes1868 cases.

Actual Node/Pyodide0.29.4 and Chrome155.0.8059.82 Worker each pass677 contracts
with in-VM source/test hashes. A standalone wheel passes697 tests on isolated
Python3.10.19/NumPy1.26.4 without native Torch. These sources contain40 Python
modules plus Apache LICENSE/NOTICE. Final application/private-distribution gates
are recorded below after integration; focused success alone is not release proof.

## Performance and retained candidates

All four comparative phases retain every process/batch sample, frozen source,
license and harness hash. Each CPU/WASM × float32/float64 series uses balanced
baseline/candidate/candidate/baseline fresh processes, eleven batches and1200base
calls, one tenth as many for reverse/JVP, after three warmups. Baseline is the
published0.19 implementation; only mutually successful, equivalent view/alias/
value/gradient workloads are compared. Previously wrong copying/rejection cases
and mismatched-layout tangent copies are excluded. CPU usesNumPy2.5.4; WASM uses
Pyodide0.29.4/NumPy2.2.5. Environment variables request two CPU threads without
claiming a measured BLAS count. Known local producers were terminal before each
series; this remains a local shared-machine measurement, not full-app/native-Torch
speed, cold start or peak memory.

The first correct candidate increased simple view time3.6–4× and common allocation
cost1–6%; it was held from main. General stride wrappers and generic owner traversal
were replaced on common paths. A second phase retained33–89% bare-view overhead,
a third32–64%. The current measured source adds an inline validated two-integer
path and removes dynamic imports from matching JVP calls. Common constructors
are near prior latency, while strict view checking retains the costs below.
Positive cells mean higher elapsed time; negative cells mean lower elapsed time.

| Case | CPU float32 | CPU float64 | WASM float32 | WASM float64 |
| --- | ---: | ---: | ---: | ---: |
| scalar_tensor | -1.2% | -4.2% | +0.6% | +0.5% |
| array_import | -3.0% | -0.8% | +0.7% | +4.1% |
| array_copy | -0.5% | +1.2% | +4.9% | -1.0% |
| view_contiguous | +10.1% | +11.4% | +19.9% | +16.6% |
| view_stepped | +8.6% | +12.8% | +21.3% | +21.1% |
| view_transposed_shape | +7.0% | +11.8% | +28.1% | +29.5% |
| common_reverse | -1.5% | +2.6% | +1.4% | +3.2% |
| view_reverse | +1.0% | +2.1% | +1.1% | +2.2% |
| view_jvp_matching | +6.6% | +4.3% | +8.3% | +11.2% |

These final microbenchmarks retain bare-view7–30% and matching-JVP4–11% overhead;
complete view-reverse calls retain1–2% overhead. This is an explicit compatibility
cost, not a claimed application speedup. All rejected stages and noisy controls
remain in the evidence. No test tolerances were relaxed.

## Retained failures

The first focused matrix retained15 failures:13 empty-factory stride differences
were repaired;2 fixtures attempted already-unsupported differentiable nonleaf
mutation and now write through detached aliases to test the same storage guards.
A broad text replacement briefly caused an indentation/collection failure; original
source is retained. The initial suite emitted640warnings, including deprecated NumPy2.5 stride
assignment; that deprecated path was eliminated using validated metadata views and proper wrapper-owner tracking.
The first minimum-version run retained3 failures/693passes: NumPy1.26 promotes an
empty random intermediate before Tensor's float32 cast. Empty factory metadata is
now applied after that cast. Another shape shortcut retained1 failure/1537passes
because bare view() must reject missing arguments with TypeError rather than treat
it as scalar reshape. The args guard was fixed and all focused/VM/minimum gates
rerun. A failed audit read Torch-namespace __version__ instead of package metadata;
its original script/log is retained. These are separate failures, not discarded
performance samples.

## Integration gates

The previous shared-owner/metadata snapshot passed20939 infrastructure cases,
1optional Docker skip and10warnings in171.64s. The current inline snapshot has
expanded tests and needs its own final full gate, complete catalog/WASM/export,
private distribution and post-upstream frontend checks before main adoption.
The exact0.19 source and its UI-only descendant were fully verified through
Quality38853290375/Pages38854836181 and158 frozen live payloads; that predecessor
proof is carried with this substantive iteration instead of a docs-only deploy.

## Final local integration

The final40-module inline source passes **21269 infrastructure tests,1 optional
Docker skip,10warnings**,194.84s. Native catalog passes **78 problems /10218
executions /3 repeats /342 mutants**,138.899s. Complete real WASM
passes **78/78**,400.136s, without browser budget excess.
Node/Chrome each pass677 contracts; the final installed standalone wheel passes
697 on Python3.10.19/NumPy1.26.4 without native Torch. All final standalone/private
wheel and sdist module/license bytes match the source and VM snapshots, and their
installed Python paths are disjoint. The private distributions also contain the
updated interview curriculum JSON byte-for-byte.

Upstream `be1c9e3d` supplies UI refinements, normalization study navigation and
Transformer/efficiency prompts. No numerical library inputs changed on rebase;
complete final native/catalog/WASM and frontend/build/Pages policy gates use the
merged source. Initial JS graph is404521raw /130144gzip bytes, including upstream
UI. Existing native visualization inputs did not change, so no new local whole-viz
generation is claimed; complete browser T3 workflows reran, and remote native CI
regenerates visualization data when this commit is published.

All **5966 published immutable assets** remain byte-identical; all158new assets
are reachable. Autograd runtime is `runtime-84ec068fea120c2d35f6.json`, schedules
`runtime-84aabb0b7a555fa6f247.json`. Autograd transfer size is441023raw /101583gzip
bytes, +10556raw /2590gzip versus0.19 under identical gzip settings. Memory and
cold-start behavior were not measured.

[Full evidence](../../packages/torchwasm/benchmarks/results/iteration-20.json)
retains all64 timed fresh-process reports, earlier failures, source/VM/wheel
identities and final course gates. `iteration-20-harness.tar.gz` archives944
runner/source/test/log files.161 frozen expected deployment files include158
immutable payloads and three catalog representations. Exact remote Quality,
eligible Pages deployment and live verification follow main publication.
