# Indexed CPU/WASM candidate

This is the historical precursor recorded with 0.83.0 metadata. The
[0.84 qualification](torch-runtime-0.84-qualification.md) records the final
package version, scalar-alpha repair and subsequent browser verification.

This local, unreleased candidate adds `torch.index_select`, `torch.index_add`
and `torch.scatter_add`, their Tensor methods, and guarded `index_add_` /
`scatter_add_` updates. The package metadata remains 0.83.0 while the next core
release is prepared. The browser catalog exports and main have not adopted it.

The implementation is in `_indexed.py`; it uses NumPy in native Python and
Pyodide/WASM. It adds no automatic WebGPU dispatch. The library name remains
StrideTorch pending a separate naming decision.

## Behavior and storage

Dense forward computation covers bool, uint8, signed integers, float16/32/64
and complex64/128. Real reverse gradients and float32/64 JVP, tangent reverse,
functional Jacobian/HVP and the norm-HVP Tensor rules are covered. Complex
autograd, non-None output buffers and differentiable mutation retain explicit
unsupported boundaries.

Repeated float16 indices follow the observed native CPU accumulation paths.
`index_add` widens to float32 for multidimensional first/last-axis updates with
int64 indices and original `alpha == 1`; other paths round each half update,
including the scaled source. `index_select` reverse uses the same routing.
`scatter_add` widens half accumulation regardless of rank. Both tested PyTorch
versions agree on these paths. The native
[CPU implementation](https://github.com/pytorch/pytorch/blob/v2.6.0/aten/src/ATen/native/TensorAdvancedIndexing.cpp)
also explains the routing condition. This is a tested CPU profile, not a promise
of identical rounding on every PyTorch device or kernel implementation.

Int32 and int64 selection/add indices are supported. Scatter accepts both in
line with tested PyTorch 2.14 behavior; PyTorch 2.6 rejects int32 scatter indices.
Scalar/empty-index exceptions, scalar alpha conversion and the real-source
gradient error for complex alpha are checked against native behavior.

Pullbacks save version counters selectively: selection guards its indices;
index-add guards indices/source only for an active source; scatter-add guards
indices only for an active source. They do not retain intermediate values merely
to construct a derivative. Raw negative/out-of-range index mutations are checked
again at reverse time rather than silently wrapping NumPy indices. Empty/scalar
scatter pullbacks reduce to the source shape when native autograd permits it.

In-place updates preserve destination storage, aliases and version increments,
including successful empty updates. Overlapping sources/indices and expanded
destinations are rejected. Invalid indices are checked before writing, keeping
failed updates atomic; native CPU can partially modify an invalid destination
without incrementing its version, and that behavior is deliberately not copied.

## Verification

| Check | Result |
| --- | --- |
| Native indexed differential cases | 658 passed against each of PyTorch 2.14.1 and 2.6.0 |
| Final source SDK plus those differential cases, NumPy 2.5.4 | 13,024 passed |
| Same, NumPy 1.26.4 | 13,021 passed, 3 existing skips |
| Installed public wheel, SDK extracted from sdist | 12,366 passed / 12,363 passed + 3 existing skips |
| Actual Node Pyodide 0.29.4 / NumPy 2.2.5 | 12,350 passed, no skips |
| Actual Chrome 155 Worker, same Pyodide/NumPy | 12,350 passed, no skips |
| PaperLeet application plus SDK | 46,000 passed, 1 existing Docker skip |
| Public/private wheel + sdist + rebuilt wheel | Six artifacts inspected; native installed smoke checks pass on both NumPy baselines |

The WASM selection excludes the 13 host CLI/packaging tests and three tests
requiring host subprocesses. Those tests pass in both installed native suites.
A separate fresh Pyodide instance verifies exported offline sources, import
isolation, unchanged `import torch` alias installation and indexed gradients.
The installed public package contains 78 resources; the private runtime has 67;
default offline embedding has 63; the schedules profile remains unchanged.

The 44 new NumPy-only SDK cases cover adjoints, finite-difference nonlinear
norm-HVPs, aggregation conservation, aliases, invalid-write atomicity, index
mutation, intermediate array release and half accumulation profiles. Native
comparison tests and their failed predecessors are retained in the raw record.

## Allocation improvement

Selection reverse now allocates its zero accumulator directly, removing an
otherwise unused dense zero array and copy. Across 24 CPU dtype/shape/NumPy
contexts, three traced VJPs per candidate produced identical gradient hashes.
Median traced peak allocation fell 24.77%–49.94%. For float32 `(1024, 512)` axis-0
selection on NumPy 2.5.4, it fell from 4,205,308 to 2,108,060 bytes.

These are Python/NumPy allocations during retained-graph reverse, excluding
preallocated inputs. They are not process RSS, WASM heap measurements or a
latency claim. Scatter coordinate vectors also avoid full rank-by-size grids.

## Evidence and release work

[Summary](../../packages/torchwasm/benchmarks/results/indexed-cpu-wasm.json) and
[raw evidence archive](../../packages/torchwasm/benchmarks/results/indexed-cpu-wasm-harness.tar.gz)
preserve the candidate sources, SDK/native tests, package artifacts, logs,
allocation probes, failed candidates and corrected harnesses. The archive uses
content-addressed payloads with a complete virtual-path manifest and replay
dependencies. SHA-256:

`5707a1a3a708aca14c2d8e1e8752ddf29c906987d1152203bc2fc25fc98603ac`

The first application run failed after duplicate, unrelated benchmark copies
in a build snapshot exhausted disk. Exact-hash canonical references were saved
before removing those redundant copies; the same application source then passed
in full. The successful package build stages only its actual inputs.

Before main adoption: bump the core version and metadata, regenerate and qualify
the active browser exports with the new source set, exercise the actual nested
browser judge and catalog budgets, then perform a fresh source audit and guarded
main push. This record does not claim those release checks have happened. No CI,
tag, publication or deployment was triggered.
