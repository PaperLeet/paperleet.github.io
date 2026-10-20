# TorchWasm 0.91 local qualification

TorchWasm 0.91 adds the checked cumulative-product implementation and its
zero-safe gradients, direct and reverse-derived JVP, and composed HVP. Medium
measured inference flows improve relative to the preceding API development;
large CPU inference costs and the quadratic differentiable zero-input path
remain explicit. Production WebGPU remains disabled.

The exact numerical source passes 14,009 SDK tests on modern NumPy; the minimum
NumPy environment passes 14,006 with three existing skips for ranks 34, 35
and 64, which that NumPy version cannot construct. Node/Pyodide and a
fresh Chrome Worker each pass all 13,993 selected SDK cases. WASM excludes the 13-case standalone/CLI file and three subprocess tests.
Aliases and offline export smokes are separately exercised in each WASM run;
the CLI is checked through the native SDK and installed-package runs. The full native application
passes 33,653 tests with the existing opt-in Docker submission-image test
skipped. All six public/private wheel, sdist
and rebuilt-wheel artifacts pass byte/resource and installed-package checks.

After merging main, the first complete frontend attempt exposed a render-time
`location.href` read in the report-practice search hook. The fix permits absent
or undefined locations during server rendering, while events still compare the
live browser URL and keep their scope/IME leases. The original failing log is
retained. Both the new SSR cases and the existing report study-list test pass.
The subsequent complete 248-command frontend run, both builds and their asset
and metadata checks pass. Actual Chrome loads the nested attention page, edits
the canonical solution plus API probes through CodeMirror, and passes all seven
judge tests using the exact 65-resource 0.91 runtime. Its probes include zero
factors, JVP/HVP, long scans, ownership and nonfinite derivatives.

All 89 PaperLeet modules pass fresh Node/Pyodide execution of their tests and
mutants, with no browser-budget excess. These checks use Pyodide 0.29.4,
Python 3.13.2 and NumPy 2.2.5. Production WebGPU remains disabled.

The qualification archive uses a self-contained SHA-256 blob manifest. Every
first-party input/source/test snapshot, harness, raw log/result and installation
artifact is present. Original cumulative-product development archives are also
included. No ancestor archive, omitted development directory or Git object is
needed to recover these bytes. Third-party runtimes and dependencies remain
separately installed; their observed versions and dependency loader are retained.
The SDK snapshot includes a portable native replay helper.

The [machine record](../../packages/torchwasm/benchmarks/results/qualification-0.91-cumprod.json)
binds the current source and all local gates. Archive SHA-256:
`5f3329a82610cb442408e1756d0f27e1f142b7ac5ddf095f47a14458ab1eb37f`.
Its 16,291,144-byte v1 archive contains 7,845 logical files
from 2,856 unique blobs, with zero inherited archive or Git blobs. The public
verifier passes integrity and current-source checks.

The [restoration receipt](../../packages/torchwasm/benchmarks/results/qualification-0.91-cumprod-replay.json)
records an actual complete extraction outside the repository and independent
hash readback of all 7,845 files. The recovered SDK then passes 669 selected
cases on each native environment: all 656 cumulative-product cases plus 13
standalone/import/CLI cases. This is a focused portability check; it does not
claim another full SDK run. Its separate current replay archive retains the
complete SDK inputs, portable tool, raw logs and XML; SHA-256:
`c7d4886a267ceb61fe150842c7b0e4f2852fb1d2576d2a416811b2f45e09594f`.

To check and recover the primary archive from this checkout:

```sh
python packages/torchwasm/benchmarks/verify_qualification.py packages/torchwasm/benchmarks/results/qualification-0.91-cumprod.json --check-current-source
python packages/torchwasm/benchmarks/restore_qualification.py packages/torchwasm/benchmarks/results/qualification-0.91-cumprod.json packages/torchwasm/benchmarks/results/qualification-0.91-cumprod-harness.tar.gz /tmp/torchwasm-091-replay sdk/
python /tmp/torchwasm-091-replay/sdk/replay-native-sdk.py
```

The output directory must not already exist. The last command runs the complete
SDK and needs NumPy and pytest in the selected Python environment. Add
`-k 'cumprod or standalone'` to repeat the focused 669-case check. The original
browser harnesses and dependency loader remain in the primary archive; their
pinned third-party runtime dependencies are separately installed.

Qualified source base: `e2b07350eb102b62a2f9ec7f93f7d84c74b8edc2`.
Merged main snapshot: `a00abfd83f503984265b8fff2f592fc5f5fee10a`.

The package wheels retain the README captured when qualification began, which
says qualification is in progress. This record closes those local gates; a
later documentation update must rebuild its package artifacts. The early SDK
plan's prose says 467 added cases; collected/executed totals and the complete
656-case cumulative-product module are authoritative and correctly bound.

This record qualifies the stated local source and merged-main snapshot. Main
adoption remains pending. The complete development branch and all prior proofs
remain intact. A separate delivery selection can retain every existing main
immutable asset and proof while adding the current runtime and self-contained
qualification. It needs its own exact-tree checks and a fresh guarded Git grant.
No CI dispatch, tag, publication or deployment was initiated.
