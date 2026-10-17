# Torch runtime iteration 0.44: reuse spatial maxima across depth windows

Optimize the existing MaxPool3d API by reusing each HW reduction across overlapping depth windows. The 0.43 measurements identified a 36–42% overlap-forward regression against separable public-API composition. This version instead compares the same supported direct 3D APIs on both releases, with the old numerical path retained as a same-source control. Main adoption is recorded after the validated source is actually pushed.

## Numerical route and compatibility

Automatic dispatch compares nominal kernel/output work and requires at least a 35% estimated reduction before selecting the split path. Index/gradient calls also require at least 16,384 estimated saved visits, retaining the original tiny-tape route when fixed overhead dominates. Cube/slab blocks keep their original route. The HW pass reuses the existing ordered 2D scan/window/block kernels, operating on raw NumPy arrays. An ordered depth scan carries the selected HW indices directly and adds their input-depth offsets. No additional Tensor view nodes, depth-index gather arrays or user-value caches are introduced.

First finite ties and last NaNs retain their selected values/indices. The final flattened indices are the actual storage retained by the shared VJP/JVP closures, including after channels-last output conversion. Integer/half inference, float32/64 VJP/JVP/reverse tangent, CDHW/NCDHW, positive strided/F-order/NDHWC arrays, padding, dilation, ceil mode, empty batches, singleton output strides, mutation guards, aliases, serialization and checkpointed training retain their existing rules. Native Torch 2.14.1 supplies the independent differential oracle.

The selector reserves live intermediate values/indices, masks, coordinate vectors and index-merge space before choosing this route. Each packed HW stage must fit the remaining part of the existing 32 MiB workspace estimate; otherwise it scans. If even the reserve does not fit, the original 3D route remains available. This is a temporary-workspace estimate, excluding output allocations and total process memory. Metadata-only rejection is tested on a three-million-depth expanded view without allocating that intermediate. Empty batches continue to construct no positive-length coordinate grids.

Dilated windows without any actual sample retain the existing inference values/indices and explicit derivative boundary; such windows do not enter the split route. The earlier eight-million kernel/output bounds and runtime-pointer index bounds remain in force. This adds no GPU execution, half AD, general higher-order graphs, or general channels_last_3d conversion APIs.

## Validation

- Full repository/library CPU suite: 37,605 passed, one optional skip; 286.62 s. Correctness runs overlap other local work and are not performance comparisons.
- Focused max/average pooling: 2,395 passed, including 675 independent 3D region/scatter/layout/derivative/import/training checks. The fifth dispatch setting forces the split route where its sampling/workspace guards permit it.
- Five settings repeat 236 native-reference trials each (1,180 total). Each includes 166 valid random geometries, 54 invalid geometries and 16 dtype/return-form probes. Of the random trials, 162 execute VJP/JVP/reverse tangent; four retain inference-only empty-sampling coverage. Values, indices and applicable exact strides are compared.
- Actual Node/Pyodide and Chrome Worker: 3,977 independent tests each, 42 suites; Pyodide 0.29.4 / NumPy 2.2.5. All 50 source/license identities match the frozen candidate.
- Isolated Python 3.10.19 / NumPy 1.26.4 public wheel: 3,990 passed, native Torch absent. Public/private source identities, aliases, checkpoint resume, independent uninstalls and both namespaces pass in a fresh NumPy-only environment.
- Original native catalog: 78 problems, 10,218 executions, 342 caught mutants, three repeats; 387.490 s within the unchanged 600 s budget.
- Complete exported WASM catalog: 78/78, 342 caught mutants, 263.427 s, concurrency two, no exceeded browser budgets. The actual exported library runs all 3,977 independent contracts, with only the established legacy-profile supplementation.
- Full frontend checks and both typed production builds pass. A final six-file interview-browse delta is checked with ten affected canonical suites and both production builds. Later upstream curriculum/UI changes are retained with exact per-commit identities. The changed curriculum publishability test is rerun separately, and the private wheel/sdist and coinstall are rebuilt and rechecked against the latest metadata. The full CPU count remains the frozen-runtime run; supplemental metadata/UI receipts cover the later delta.
- All 9,913 prior immutable payloads are retained. The 158 new payloads are reachable, with all 161 active export identities frozen. Only __init__.py, _pooling.py and _pooling3d.py change among 50 source/license resources; 47 are unchanged from 0.43.

The concatenated source/license footprint grows from 586,643 to 590,273 bytes; local gzip estimates grow from 136,875 to 137,747 bytes. These are not actual download, installed-size, peak-memory or cold-start measurements.

## Complete-flow performance

The public bench_pooling_separable.py compares baseline direct 0.43 calls, candidate direct 0.44 calls and a separately loaded candidate namespace with split dispatch disabled. All 76 float32/float64 flows verify outputs and applicable indices/gradients or one complete training update before timing. Cases retain cube/slab/dilated/strided/NDHWC inputs and add larger overlap windows plus complete overlap-pool Conv3d–SiLU–linear–loss–momentum-SGD updates.

Twelve sequential children run four fresh CPU, Node/Pyodide and Chrome Worker processes each, alternating variant order. Seven samples use forty ordinary calls, five training calls, or four larger-overlap calls. Timing starts after owned correctness/build processes finish. Every sample is retained on this shared host. The table shows time reductions from ratios of medians across four fresh processes; negative values mean slower.

| Flow | dtype | CPU | Node/WASM | Chrome/WASM |
|---|---|---:|---:|---:|
| Overlap forward | float32 | 43.4% | 41.0% | 46.7% |
| Overlap forward | float64 | 44.2% | 40.2% | 44.4% |
| Overlap + indices | float32 | 13.2% | 19.1% | 20.4% |
| Overlap + indices | float64 | 12.9% | 2.2% | 18.5% |
| Overlap VJP | float32 | 7.5% | 15.0% | 9.8% |
| Overlap VJP | float64 | 9.6% | 14.8% | 14.7% |
| Overlap JVP | float32 | 8.8% | 20.7% | 10.8% |
| Overlap JVP | float64 | 10.8% | 13.9% | 19.9% |
| Larger overlap forward | float32 | 42.1% | 46.6% | 44.3% |
| Larger overlap forward | float64 | 43.5% | 43.5% | 44.1% |
| Larger overlap VJP | float32 | 47.9% | 44.2% | 30.7% |
| Larger overlap VJP | float64 | 57.6% | 39.4% | 45.3% |
| Overlap Conv3d + SGD | float32 | -1.2% | -0.2% | 1.7% |
| Overlap Conv3d + SGD | float64 | -2.1% | -5.1% | 0.9% |
| Cube Conv3d + SGD | float32 | 5.3% | 4.4% | -1.0% |
| Cube Conv3d + SGD | float64 | -1.4% | -1.4% | -3.1% |

Complete case ranges and disabled-route controls are retained rather than implying uniform speedups:

- cpu: direct reductions -30.3% to 62.2%; same-source disabled-route control reductions -17.3% to 23.2%.
- node: direct reductions -9.7% to 55.9%; same-source disabled-route control reductions -14.5% to 16.1%.
- chrome: direct reductions -13.3% to 56.2%; same-source disabled-route control reductions -28.4% to 8.7%.

The heuristic, metadata checks and remaining index/scatter work still have costs. Unsupported calls and empty/no-work cases are excluded from speed claims. These results do not establish native-Torch, GPU, cross-runtime, import, transfer or whole-PaperLeet throughput improvements. WebGPU and broader core coverage remain active work beyond this NumPy CPU/Pyodide optimization.

## Evidence and release controls

Exact sources, test hashes, native oracle inputs, all child samples/receipts and export identities are retained in packages/stridetorch/benchmarks/results/iteration-44*.json and the companion harness archives. Exploratory prototypes/smoke measurements and the measured tiny-tape regression remain separate from final version evidence. The initial timing tree was stopped after CPU/Node measurements identified overhead; all final runtime-dependent gates and twelve timing children use the revised selector. The completed native-Torch catalog oracle is retained with unchanged native input identities rechecked. One unchanged frontend workspace reading-return test hit its wait bound in the first tuned full check; its isolated retry passed with unchanged inputs, and the complete canonical frontend check was rerun. Initial failure, isolated retry and frontend input identities are retained. All required final source/test/package/catalog/frontend gates completed successfully.

All seven canonical release-policy files match the retained baseline. Main is not moved during an active quality/Pages chain. This work initiates no tag, workflow dispatch, release, package upload or deployment. StrideTorch remains the package name pending a user-selected replacement.

## Main adoption

Validated source `046e65e0209d8e2aead5706641bde7cd509b1b95` was adopted on remote main by ordinary non-force push. The remote SHA was verified exactly; zero Actions runs were observed for this source SHA. All 50 source/license resources, 161 active export identities, 10,071 immutable payloads and seven canonical release-policy files were rechecked. The separate proof commit is verified outside the repository after its push. No tag,dispatch,release,PyPI upload or online deployment was initiated by this work.
