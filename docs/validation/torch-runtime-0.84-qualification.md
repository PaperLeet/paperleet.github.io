# StrideTorch 0.84 CPU/WASM qualification

Version 0.84 adds `index_select`, `index_add` and `scatter_add`, their Tensor
methods, guarded in-place accumulation and analytical derivative rules. The
[indexed implementation record](torch-indexed-cpu-wasm.md) describes values,
half accumulation, selective guards and allocation measurements. The final
candidate additionally rejects a scalar Tensor with `requires_grad=True` as
`index_add` alpha, including under `no_grad`, matching native PyTorch. Rejected
updates leave destination values and versions intact; detached scalar alpha
continues to work.

The scalar-alpha repair changes only `_alpha`; all other indexed function ASTs,
including the measured allocation kernel, remain identical to the measured
candidate. Its four additional SDK cases and four native comparison cases are
preserved with the original failing probe.

| Verification | Final repaired candidate |
| --- | --- |
| Installed wheel SDK plus native indexed comparisons, NumPy 2.5.4 / Torch 2.14.1 | 13,032 passed |
| Same, NumPy 1.26.4 / Torch 2.6.0 | 13,029 passed, 3 existing skips |
| Public/private wheel, sdist and wheel rebuilt from sdist | Six artifacts inspected; both installed NumPy baselines pass |
| PaperLeet package isolation and gradient utilities | 12 passed |
| Actual installed wheel in Node and Chrome Worker Pyodide | 12,354 passed each, no skips |
| Actual exported CPU VM independent SDK | 12,343 passed across 93 suites |
| Full exported PaperLeet WASM catalog | 89 modules passed, 406 mutants caught, no budget violations |
| Actual nested problem page and opaque judge Worker | 7/7 tests, indexed gradients/JVP/HVP and scalar-alpha refusal passed |
| Generated Worker and both local/Pages builds | Passed, exact 63-source runtime identities checked |

The native catalog's 89 cases and three repetitions produced 12,360 successful
records. Problem, native runner and reference bytes are unchanged across the
adapter repair. The earlier complete application/SDK run passed 46,000 cases
with one existing Docker skip; the version literal and new alpha validation are
covered by subsequent installed, package-boundary and real WASM checks rather
than being described as another complete application run.

The installed package has 78 resources, its private PaperLeet runtime 67, and
default CPU embedding 63. The schedules profile remains six files. The optional
complete export includes `runtime.py`; importing the root package alone does
not create a `runtime` attribute. A previous harness explanation conflated
that lazy submodule import with the intentional omission from the separate CPU
embedding profile. The cold-import diagnosis and correct explicit import are
now preserved and verified, including fresh offline Node/Chrome interpreters.

The default exported VM omits the 13 host CLI/packaging cases and the optional
WebGPU package suite. Those cases run in the native installed checks. The full
optional wheel in Pyodide executes the portable WebGPU package tests as well,
excluding only three host subprocess cases. No numerical test was removed.
The earlier 88/89 export check failed when those host cases entered the CPU VM;
the upstream harness fix is retained and the full catalog was rerun afterward.

The [summary](../../packages/stridetorch/benchmarks/results/qualification-0.84.json)
and [raw archive](../../packages/stridetorch/benchmarks/results/qualification-0.84-harness.tar.gz)
contain final package sources, tests, native results, full catalog records,
exact export bytes, Worker probes, failed predecessors and replay dependencies.
The archive contains 1,152 virtual files and 667 unique payloads, with all
payload hashes verified. SHA-256:

`225c0b500ced12851c5a826d0166ddea0e7d558d5813d50af4cb408d1561690d`

The CPU allocation result remains scoped to the 24 measured contexts: roughly
25%–50% lower traced reverse peak, identical gradient hashes, no process-RSS,
WASM-heap or latency claim. Automatic WebGPU dispatch remains disabled.

Core qualification used source base `9e4fe511` with separately hash-bound fresh
exports. Later main changes confined to frontend controls and learning guides
are integrated and verified separately before adoption. No tag, Actions dispatch
or publication was performed by this iteration; main adoption requires the
fresh guarded audit of the final combined tree.
