# Torch runtime iteration 0.42: direct 1D max pooling and reachable scans

This candidate adds `nn.MaxPool1d`, functional value/index entry points, and
root `torch.max_pool1d` / `torch.max_pool1d_with_indices` aliases. Ordinary
Torch imports continue through the existing host alias/launcher. Numerical
selection and scatter kernels are shared with 2D without constructing extra
Tensor view graph nodes. Main adoption is recorded separately after validation.

## Compatibility and computation

CL/NCL inputs support stride, padding, dilation, ceil mode and optional indices.
Float32/float64 support reverse AD, JVP and reverse-over-forward; float16
inference is supported. The native CPU target supports integer 1D inference
only with returned indices: the adapter preserves that boundary explicitly.
Existing integer 2D inference remains supported. Valid 1D kernels above eight
million taps require the server; invalid argument/geometry checks run first.
Native Torch 2.14.1 supplies the reference implementation used by these tests.

MaxPool1d and MaxPool2d share a private Module base and remain public sibling
types. This preserves dimensional `isinstance` checks, repr, state dictionaries,
pickle and ordinary import aliases. Saved input/index/tangent mutation and
output edits are compared with native paths, including hidden-index forward AD.
Empty values and indices preserve conventional contiguous strides.

A paired block handles 1×2 windows, including the existing 2D API; 2×2 blocks
retain their behavior. Finite ties choose the first actual sample; multiple
NaNs choose the last, including payload/route selection. Window packing retains
the conservative 32 MiB workspace estimate. It is separate from output
allocations and total process memory.

The scan fallback bounds each kernel-axis loop by offsets that can reach any
real sample between the first and last output. Globally unreachable taps are
discarded before iteration; remaining taps keep their original order. No
reassociation, cached user values or new input copies are introduced. Million-
tap padding over seven input values is checked against native values, indices,
VJP, JVP and reverse tangents. These tests do not time the slow old million-tap
scan. Dilated windows without any actual sample retain their inference values
and indices but explicitly require the server for derivatives.

## Validation

- Focused native/standalone max and average pooling: 4,379 passed.
- Full repository/library CPU suite: 36,930 passed, one optional skip; 317.48 s.
- Actual Node/Pyodide and Chrome Worker: 3,302 independent checks each, 41 suites; Pyodide 0.29.4 / NumPy 2.2.5.
- Isolated Python 3.10.19 / NumPy 1.26.4 standalone wheel: 3,315 passed, native Torch absent.
- Four internal routes repeat 360 seeded 1D/2D float32/64 geometries: 1,440 numeric/geometry trials. Invalid outputs and explicitly unsupported empty-window derivatives are classified separately from valid VJP/JVP comparisons.
- Native parameter outcomes, index/IEEE selection, noncontiguous and empty layouts, selective saved versions, Module type/pickle and Conv1d–SiLU–max–adaptive-average–linear momentum-SGD checkpoint continuation are covered.
- Public/private wheels and source archives share all 49 source/license identities. Coinstallation, normal Torch aliases, large-window tapes, training/checkpoint resume and independent uninstall pass in minimal environments.
- Original native catalog: 78 problems, 10,218 executions, 342 mutants, three repeats; 169.957 s within the unchanged 600 s aggregate budget.
- Exported WASM catalog: 78 problems and 342 mutants; 276828 ms, concurrency 2, no exceeded browser budgets. Exported attention includes all 3,302 independent checks against actual exported source.
- Full frontend checks, publishing-policy checks and both typed production profiles pass. All 9,597 prior immutable payloads are unchanged; 158 new reachable payloads and 161 active identities are frozen.

## Performance

The supported baseline is the adopted 0.41 2D composition. It never invokes
a missing old 1D API. Both candidate composition and direct 1D API are measured:
the former controls the shared pair-block kernel, the latter also removes
Tensor view/graph dispatch. All values, indices, derivatives and a complete
training update are compared before timing. Each CPU, Node/Pyodide and Chrome
Worker runtime has four fresh processes, alternating variant order, seven
samples and 40 calls per sample (five for complete training), across 52
float32/float64 forward/strided/index/VJP/JVP and complete CNN/SGD flows.

The table gives reductions in elapsed time against the old composition. Ranges
span the two dtypes after taking the median of four process results.

| Flow | CPU | Node/WASM | Chrome Worker/WASM |
| --- | ---: | ---: | ---: |
| Pair forward | 71.6–72.3% | 68.1–71.1% | 65.0–73.1% |
| Pair with returned indices | 52.5–52.8% | 53.5–57.5% | 55.6–56.5% |
| Pair full backward | 29.7–30.0% | 27.7–29.4% | 31.2–34.5% |
| Large pair forward | 52.4–52.9% | 52.7–54.7% | 54.6–56.2% |
| Overlapping forward | 15.0–15.4% | 16.7–18.5% | 16.3–17.4% |
| Dilated JVP | 16.8–17.6% | 16.6–18.6% | 14.5–15.6% |
| Complete CNN/SGD step | 9.6–11.2% | 7.9–11.4% | 6.0–10.9% |

Negative entries denote regressions. Shared-host variance, all samples,
same-candidate composition comparisons and control drift remain in the JSON
and harness. This does not establish a uniform speedup, native Torch speed
comparison, GPU behavior or cross-runtime throughput.

A separate six-process comparison uses the same explicitly forced scan
algorithm in both namespaces, seven samples and four calls, on 1,024/4,096-tap
kernels with seven input values. Its 12 forward/VJP/JVP flows verify exact
values and derivatives before timing. These target the fallback, not automatic
dispatch or application throughput. The 4,096-tap results below combine two
fresh processes per runtime; ranges span the two dtypes.

| Forced scan flow | CPU | Node/WASM | Chrome Worker/WASM |
| --- | ---: | ---: | ---: |
| Forward | 94.3–94.4% | 94.6–94.7% | 94.2–94.3% |
| Full backward | 84.5–84.8% | 85.1–85.5% | 84.4–84.8% |
| JVP | 86.6–87.2% | 86.8–87.5% | 86.0–87.2% |

Eight additional paired CPU/WASM children retain 21 existing library
workflows; sixteen alternating fresh CPU processes measure namespace import
with NumPy included/preloaded separately. Source/gzip estimates do not measure
browser transport, installation or total resident memory. Earlier prototype
measurements and tests are retained separately under `before-scan`; final
claims above use the frozen 0.42 source only.

## Evidence and release policy

- `packages/stridetorch/benchmarks/results/iteration-42.json` and `iteration-42-harness.tar.gz`.
- `packages/stridetorch/benchmarks/results/iteration-42-integration.json` and `iteration-42-integration-harness.tar.gz`.
- Reusable tools: `bench_max_pooling_1d.py` and `bench_pooling_scan.py` under the package benchmarks directory.

Only `__init__.py`, `_api.py` and `_pooling.py` change among the 49 runtime/
license resources; 46 remain byte-identical. The canonical quality gate remains
tag/manual triggered, with the existing current-main Pages release chain.
Ordinary main pushes do not request new CI or deployment. No tag, dispatch,
package upload or online publication is requested by this iteration. The user
will release later. WebGPU remains a separate backend investigation requiring
resident storage and end-to-end transfer/graph evidence.

Latest upstream `fa4d3d3239ba4236bbbd8880fc2d82ec14092d8a` adds 11 frontend/check/style/documentation
changes. They were retained by fast-forward, and the full latest frontend suite,
both typed builds and every changed check script pass. All 49 source/license
and 161 active-export identities remain exact; existing numerical/native/full-
WASM evidence retains unchanged inputs. See `iteration-42-main-sync.json`
and `iteration-42-main-sync-harness.tar.gz`.

Validated source was adopted non-force on remote `main` at `6d7dde478235a78277c779cc8631cdb2bb54bbc2`.
A read-only query found zero Actions runs for this commit. See
`iteration-42-main-adoption.json`; no tag, dispatch or online release was created.
