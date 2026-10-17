# Torch runtime iteration 0.43: volumetric max pooling

Add `nn.MaxPool3d`, functional values/indices, and the value-only root `torch.max_pool3d` alias. Existing host alias/launcher behavior supports ordinary Torch imports. MaxPool1d/2d/3d are public sibling Module classes. Main adoption is recorded after the validated source is actually pushed.

## Compatibility and implementation

CDHW/NCDHW inputs support stride, padding, dilation, ceil mode, empty batches and per-channel flattened int64 indices. Float16 and the supported integer dtypes allow both return forms; float32/float64 support VJP, JVP and reverse differentiation of the tangent. This differs from the native value-only 1D integer dispatch boundary. Native Torch 2.14.1 supplies the local differential oracle.

Positive strided NumPy arrays with an NDHWC layout preserve native value/index strides. Forward AD canonicalizes ambiguous C=1 or single-spatial-point tangent layouts as native gather does. Dense leaf gradient layouts, strided/F-order inputs, empty outputs, first-finite ties, last-NaN selection and signed-zero payload selection are covered. This does not add general channels_last_3d conversion APIs.

A separate three-dimensional numerical engine shares the existing graph/scatter logic. A 1/2-edge block kernel avoids general packing for cube/slab reductions. Ordered scan loops skip globally unreachable taps; packed windows retain a conservative 32 MiB temporary-workspace estimate. That bound excludes output allocations and is not a peak-memory measurement. Returned indices share the actual storage retained by derivative closures, including after output layout conversion. Input/index/tangent version guards remain active.

Valid kernel/output counts above eight million, or flattened spatial indices beyond the runtime pointer range, explicitly require the server. Invalid arguments/output geometry are rejected first. Dilated windows containing no actual sample reproduce native inference values/indices, but derivatives remain explicitly unsupported. Float16 AD, complex/bfloat16 kernels, higher-order graph construction and GPU device execution remain outside this contract.

## Validation

- Full repository/library CPU suite: 37,450 passed, one optional skip; 583.72 s. Other local work overlapped, so this duration is not a performance comparison.
- Independent 3D region/scatter/derivative/layout/import/training/checkpoint suite: 520 passed. Existing max/average pooling regression subset: 1,720 passed.
- Four dispatch settings repeat 236 native-reference trials each (944 total): 166 valid random geometries, 54 invalid geometries and 16 dtype/return-form probes per setting. Of the random cases, 162 execute VJP/JVP/reverse tangent; empty-sampling derivatives are excluded explicitly. Four trials reproduce inference only. Values, indices and exact layout strides are compared where applicable.
- Actual Node/Pyodide and Chrome Worker: 3,822 independent tests each, 42 suites, Pyodide 0.29.4 / NumPy 2.2.5. All 50 source/license resource hashes match the frozen candidate.
- Isolated Python 3.10.19 / NumPy 1.26.4 wheel: 3,835 passed, native Torch absent. Public/private wheel source identities, alias imports, independent uninstalls, serialization and 3D VJP/JVP pass in a fresh NumPy-only environment.
- Original native catalog: 78 problems, 10,218 executions, 342 caught mutants, three repeats; 294.014 s within the unchanged 600 s budget.
- Complete exported WASM catalog: 78/78 and 342 mutants; 442.859 s, concurrency two, no exceeded browser budgets. The attention path runs all 3,822 contracts against the actual exported library, with only the pre-existing legacy-profile supplementation.
- Full frontend checks and both typed production builds pass. The eight upstream School-label/source-check changes and seventeen subsequent frontend navigation/practice-entry files were retained by fast-forward. Full frontend checks and both builds passed after that 17-file state. A subsequent 5-file feed-filter UI delta passed 10 affected suites and both production builds, retaining the earlier snapshot identities against their original Git blobs; a further 5-file reading-toolbar/viewport delta passed 10 affected suites and both production builds; a final six-file interview/shared-search delta passed ten affected suites and both builds; all 50 math/license resources and 161 exports remained identical.
- Prior 9,755 immutable export payloads are preserved; 158 new payloads are reachable, with all 161 active export identities frozen. Public and private distributions agree on all 50 core/license resources. The 46 resources outside the three changed files and one new engine are unchanged from 0.42.

Source/license footprint rises from 574,320 to 586,643 raw bytes; concatenated local gzip estimates rise from 134,976 to 136,875 bytes. This is not an actual network-transfer, wheel-size, cold-start or peak-memory measurement.

Empty-batch regression tests forbid positive-length coordinate ranges for all four dispatch settings. On a zero-element `(0, 1, 200, 200, 200)` index-returning call, traced NumPy/Python peak allocations fell from about 64.46 MB to 9–10 KB versus the initial unreleased 0.43 engine. VJP/JVP still return the expected empty shapes. This single-call tracemalloc probe is separate from timing and does not establish RSS or WASM memory bounds.

## Complete-flow performance

The public `bench_max_pooling_3d.py` compares the supported 0.42 separable 2D/1D composition with direct 3D calls. The baseline includes index recombination and output layout restoration; an identical composition in the candidate controls changes in shared code. No absent old 3D API is timed. Each of 62 float32/float64 cases verifies its outputs and applicable gradients/indices or a complete training update before timing.

Twelve sequential children use four fresh CPU, Node/Pyodide and Chrome Worker processes each, alternating variant order, seven samples and forty calls (five training) per sample. Timing starts after all owned correctness/build processes finish. This is a shared host, with every sample retained. The table gives time reductions from medians of the four child medians; negative values mean slower.

| Flow | dtype | CPU | Node/WASM | Chrome/WASM |
|---|---|---:|---:|---:|
| Cube forward | float32 | 11.1% | 6.8% | 13.8% |
| Cube forward | float64 | 9.7% | 15.5% | 14.3% |
| Cube VJP | float32 | 26.8% | 25.2% | 25.6% |
| Cube VJP | float64 | 24.9% | 26.7% | 26.9% |
| Larger cube forward | float32 | 16.6% | 19.8% | 21.4% |
| Larger cube forward | float64 | 10.5% | 21.0% | 15.7% |
| Overlap forward | float32 | -36.3% | -36.6% | -38.5% |
| Overlap forward | float64 | -37.3% | -42.2% | -38.9% |
| Dilated JVP | float32 | 41.0% | 41.8% | 41.8% |
| Dilated JVP | float64 | 43.5% | 42.0% | 41.7% |
| Conv3d + SGD step | float32 | 7.3% | 8.2% | 8.1% |
| Conv3d + SGD step | float64 | 7.8% | 13.6% | 12.0% |

The complete case ranges and composition controls are retained rather than claiming every configuration improves:

- cpu: direct reductions -42.5% to 66.0%; composition-control reductions -9.7% to 3.0%.
- node: direct reductions -47.4% to 65.6%; composition-control reductions -9.2% to 8.1%.
- chrome: direct reductions -41.9% to 65.8%; composition-control reductions -7.7% to 19.0%.

These measurements compare library flows, including full Tensor calls and the stated training step. They do not establish native-Torch, GPU, cross-runtime, cold-import, browser-transfer or whole-PaperLeet throughput improvements. WebGPU remains a separate storage/backend evaluation; these results establish the NumPy CPU/Pyodide path only.

## Evidence and release controls

Exact source/test hashes, all child samples/receipts, native oracle inputs, runtime harnesses and export identities are retained in `packages/stridetorch/benchmarks/results/iteration-43*.json` and the companion harness archives. Initial package-evidence path, UI-runner working-directory and WASM benchmark helper-search-path errors, and their artifact-copy/cwd/path-only repairs, remain in the raw archives. Failed preliminary timing children are excluded from the final twelve-child dataset. The initial and final unreleased 0.43 empty-batch proof generations are retained separately. All required final source/build/test gates completed successfully.

All seven canonical release-policy files match the retained baseline. This work creates no tag, dispatch, release, PyPI upload or online deployment. A separately started manual quality gate is observed before adopting main; main is not moved during an active quality/Pages chain. The standalone library still uses StrideTorch until the user selects a replacement name.

## Main adoption

Validated source `669ef9ad700619b15a6fed119d1b4b4420fd68b3` was adopted on remote main by ordinary non-force push. The remote SHA was verified exactly; zero Actions runs were observed for this source SHA. All 50 source/license resources, 161 active export identities, 9,913 immutable payloads and seven canonical release-policy files were rechecked. The separate proof commit is verified outside the repository after its push. No tag,dispatch,release,PyPI upload or online deployment was initiated by this work.
