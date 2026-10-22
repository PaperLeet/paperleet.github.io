# Integer index bounds without full-size comparison buffers

These measurements describe the development candidate based on `08636f5e`,
whose package version was still `0.84.0`. Its mathematical source is now included
in the [0.85 qualification](torch-runtime-0.85-qualification.md), with fresh
installed-package, catalog, frontend and actual Worker checks. Main adoption is
recorded separately. No CI, tag, publication or deployment was triggered by
these measurements. The package has not been renamed.

`gather`, `index_select`, `index_add`, `scatter_add` and selection pullbacks now
share a checked integer-range predicate. Two integer reductions replace the
two index-sized boolean comparison arrays. Empty arrays short-circuit before
reductions. Non-integer arrays retain the previous comparison semantics,
including NaNs that could otherwise mask an invalid saved index after raw NumPy
metadata changes. Each caller retains its existing exception type and message.
Validation is still performed before an in-place update can modify storage.

Only `_indexing.py` and `_indexed.py` differ between the old and new runtime
resource sets. Package resources, numerical kernels, derivative rules and
optional GPU dispatch are otherwise identical. The default CPU payload still
contains 63 resources, and automatic GPU dispatch remains disabled.

The [machine-readable record](../../packages/torchwasm/benchmarks/results/index-bounds.json)
binds the exact implementation/test hashes to the
[raw evidence archive](../../packages/torchwasm/benchmarks/results/index-bounds-harness.tar.gz).

| Functional verification | Result |
| --- | --- |
| Full SDK and native indexed comparisons, NumPy 2.5.4 / Torch 2.14.1 | 13,144 passed |
| Same, NumPy 1.26.4 / Torch 2.6.0 | 13,141 passed, 3 existing skips |
| Full portable SDK in Node Pyodide | 12,466 passed |
| Same in a real Chrome Worker | 12,466 passed |
| Native selection and public/private wheel isolation, modern environment | 331 passed |
| Public/private wheel isolation, minimum environment | 9 passed |

The 112 additional SDK cases check weighted adjoints and values with int32/int64
indices, positive-stride slices, transposed and broadcast views, descending
values, empty axes, both signed integer extremes and atomic invalid mutations.
The full Pyodide suite retains its established exclusions for host CLI and
three subprocess-only cases. Fresh interpreters also verify offline exports,
ordinary imports, Torch alias activation and indexed gradients.

The actual previous `_bounds` function definitions are compiled from the frozen
baseline source and installed into the unchanged surrounding call chains for
timing. The candidate uses the actual new functions, including its additional
shared-helper call. All other runtime resources are hash-identical. This avoids
giving the old code an artificial helper-call overhead. The earlier temporary
predicate-wrapper exploration is retained separately and superseded.

Each of four environments executes 344 predicate equivalence checks, 18 helper
measurements and 64 complete float32 Tensor flows. Complete flows cover the
four indexing operations, forward and forward/VJP, both index dtypes, tiny and
large vectors, row and column selection/aggregation and broadcast index views.
Outputs and gradients must match exactly before timing. Seven rotated rounds
retain every sample for baseline, an identical-baseline A/A control and candidate;
calibration targets approximately 30 ms baseline batches. Timing runs do not
overlap tests or other benchmarks.

| Environment | Complete-flow candidate/baseline median range | A/A range |
| --- | --- | --- |
| NumPy 2.5.4 | 0.600–1.036 | 0.954–1.042 |
| NumPy 1.26.4 | 0.636–1.005 | 0.947–1.029 |
| Node / Pyodide / NumPy 2.2.5 | 0.662–1.073 | 0.903–1.021 |
| Chrome Worker / Pyodide / NumPy 2.2.5 | 0.653–0.993 | 0.966–1.033 |

The initial Node run contains a 7.27% slower int64 row `index_add` forward
context. Its baseline samples also drift from approximately 800 to 900 us.
The original result remains in the record. Three fresh Node processes repeat
16 aggregation contexts with nine rotated rounds, 75 ms calibration and varied
initial ordering. The original context's ratios are 0.975, 0.996 and 0.962;
the maximum ratio across all 16 contexts in each repeat is 1.021, 1.003 and 1.013.
The initial slowdown is therefore not reproduced by these repeats. These are
warm library observations on this machine, without a universal speedup claim.

Allocation runs start after input construction and record three traced peaks
per arm. For 1,048,576 dense int64 indices, the standalone check's peak falls
from approximately 1 MiB to 176 bytes on NumPy 2.5, 920 bytes on NumPy 1.26 and
492 bytes in Node/Chrome Pyodide. Complete-call savings depend on the workload:

| Forward aggregation: 262,144 int64 indices into 1,024 float32 values | Old peak bytes | New peak bytes |
| --- | --- | --- |
| NumPy 2.5, `index_add` | 268,613 | 12,044 |
| NumPy 2.5, `scatter_add` | 268,041 | 11,256 |
| NumPy 1.26, `index_add` | 269,902 | 14,552 |
| NumPy 1.26, `scatter_add` | 268,758 | 13,560 |
| Node/Chrome Pyodide, `index_add` | 267,782 | 40,960 |
| Node/Chrome Pyodide, `scatter_add` | 267,538 | 40,592 |

These are Python/NumPy traced allocations. Process RSS and WASM linear-memory
capacity are outside this measurement. Output allocation dominates some other
flows, whose whole-call peaks remain effectively unchanged.

Failed predecessors are retained. The first new fixtures imported negative
NumPy strides, which native Torch and this library reject; corrected fixtures
use descending copies and supported positive-stride views. An additional
minimum-environment application run encountered missing setuptools/PyYAML and
33 native Torch 2.6 selection-comparison differences. The same 33 failure IDs
and messages occur with the baseline, including int32 gather and axis exception
differences. Adding only setuptools 84.0.0 and PyYAML 6.0.3 to the task-owned
environment leaves every existing dependency version unchanged and allows all
nine package-boundary cases to pass. No numerical tests were removed to obtain
the complete SDK results above.

The archive contains 968 logical files and 445 unique SHA-256-addressed payloads.
Its SHA-256 is:

`a43993285405e1963d86b7870c088c8a331bf120c0283a95fdf4eeb0e01ce4ca`

`manifest.json` maps logical paths to `payloads/<sha256>` members. Materialize
those paths under a replay directory, verifying each payload hash first. The
native benchmark then runs independently of the repository:

```sh
PYTHONDONTWRITEBYTECODE=1 python -B replay/torch-iteration-668/benchmark.py result.json
```

It needs NumPy and the restored bundled candidate sources; native Torch is not
imported by this benchmark. Node/Chrome runners also need the recorded external
Pyodide 0.29.4 runtime cache and their host executable paths adjusted locally.
