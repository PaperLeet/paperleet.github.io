# Installed experimental WebGPU integration

The independent distribution now includes
`stridetorch.experimental.webgpu`. Consumers can import the Python offload API
and export its complete JS service assets from an installed wheel or sdist,
without borrowing files from PaperLeet's benchmark directory. The namespace
remains experimental and requires explicit host setup; it does not enable a
Torch GPU device or automatic dispatch.

## Package and compatibility contract

The namespace exports `MatmulOffload`, `ResidentMatmulOffload`, `Matrix`,
`MatrixProgram`, `ProgramResult`, `service_sources` and `source_files`.
Importing it in CPython requires no Pyodide/JS bindings and leaves native Torch
imports untouched. Browser bindings load only when needed. Missing activation
reports that Pyodide and an attached, ready MatrixRpc client are required.

`service_sources()` exports four adjacent modules and LICENSE/NOTICE. All JS
module dependencies resolve within that set. `source_files()` explicitly
exports the complete optional package payload for an offline Pyodide FS.
The installed public package has 77 resources: the existing 64 root Python
modules and two notices, seven experimental Python modules and four JS assets.
Only NumPy remains a required dependency. Apache-2.0 notices and the CLI remain
present in the built metadata.

The original CPU root's 66 resources are byte-identical. The compact default
autograd export still has 62 entries, and schedules has six; neither includes
the optional integration. All 183 active PaperLeet exports remain byte-identical.
The application wheel retains its separate private 66-resource CPU package.
The development version remains 0.83.0; no new release has been published.

The old Python benchmark import names alias the canonical modules, including
their shared reentry gate. Old JS entry points re-export the package sources
when the full repository tree is served. Independent hosts should export
`service_sources()` instead of copying those thin entries. Setup and usage are
documented in the [package README](../../packages/torchwasm/README.md#experimental-browser-gpu-integration).

## Client ownership

Both offloads accept an explicit `client=`. A resident session fixes its default
client at construction; a stateless offload fixes it at its first GPU submission.
Replacing the host's global `matrix_rpc` cannot retarget an existing offload.
The wrappers borrow the connection: resident close frees its own leases but
does not close the host's client or Worker. The host retains that lifecycle.
Providing both an injected transport/backend and a client is rejected.

An initial lazy-binding candidate could follow a replacement global client.
The preserved negative test reproduces this on that candidate. Final native
and browser checks verify default and explicit client binding, forward values
and real gradients after global replacement, no requests on the unrelated
legacy connection, and a still-open borrowed connection after wrapper cleanup.

## Installed and browser checks

Six final public/private artifacts were built locally: wheels, sdists and wheels
rebuilt from each sdist. Their actual installed package payloads match the
qualified 77 public and 66 private resources. The public sdist preserves all 94
SDK Python source files. The wheel metadata carries the exact final README,
Apache-2.0 license expression/notices, CLI and sole NumPy dependency.

| Gate | Final result |
| --- | --- |
| Installed SDK, NumPy 2.5.4 / native Torch 2.14.1 | 12,322 passed |
| Installed SDK, NumPy 1.26.4 / native Torch 2.6 | 12,319 passed, three existing skips |
| Targeted package/benchmark tests, each environment | 150 passed |
| Root packaging, CLI, co-install/uninstall and private export | 12 passed |
| Portable protocol/resource checks | 154 passed |

The targeted set contains 136 prior benchmark/compatibility cases and 14 new
package/import/resource/client-binding cases. It overlaps the full SDK suite
and is not an additional unique-test total. The root tests also rebuild from a
full frozen repository layout and verify the final public/private payloads.

Chrome 155 / Pyodide 0.29.4 / NumPy 2.2.5 installs the actual final wheel into
its FS. Its host serves assets exported from that installed wheel. No Python
core source checkout payload is loaded. All 77 imported resources and six asset
exports/notices match the wheel. The original 51 independent native numerical
contexts and 31 state/ownership controls pass with unchanged `rtol=1e-5,
atol=1e-6`. Seven compatibility/binding contexts cover mixed client/service
versions, early capability refusal and pinned borrowed connections.
All four real hosts end closed with zero tracked allocations, handles and bytes.
The explicitly incompatible host still performs no computation or GPU allocation.

One earlier test harness copied ROOT-relative packaging tests into an incomplete
layout, producing nine setup errors. Those records remain intact. The restored
snapshot layout passes all 12 original tests, and the final package is checked
again after the client-binding fix. No product test was removed or weakened.

## Resident-call overhead

Six fresh Chrome processes measure sizes 256 and 512, three per size, with
single-node repeats, two-node repeats and changing right operands. The previous
flat negotiating Python implementation and the packaged implementation use
byte-identical JS, shader, planner and protocol on three independently bounded
host sessions. Timing includes autograd, transfers, owned copies, cache lookup,
eviction and the new lazy bindings. Setup and final cleanup remain outside the
clock. Normal GC is enabled; all native reference and ownership checks pass.

The timing fixture binds each provider to its own client by changing only the
program constructor's client argument. The final implementation uses exactly
that constructor argument. The source audit verifies identical non-constructor
resident paths and identical matrix/bridge helpers against the measured binding.
The later explicit-client constructor fix does not change that resident path.
Stateless transport has numerical/VJP coverage, with no new latency claim.

| Size and flow | Unpaired packaged/previous ratio | Paired packaged/previous ratio |
| --- | ---: | ---: |
| 256, repeated single | 1.0039 | 1.0002 |
| 256, repeated chain | 1.0186 | 1.0108 |
| 256, changing right operand | 1.0098 | 1.0159 |
| 512, repeated single | 1.0274 | 1.0158 |
| 512, repeated chain | 1.0277 | 1.0137 |
| 512, changing right operand | 0.9851 | 0.9766 |

Each ratio is a median of three process estimates. All six contexts pass the
frozen 5% overhead gate, aggregate A/A within 5%, every individual A/A within
10% and batch floors of at least 30 ms. The largest aggregate overhead is 2.77%
across estimators. One individual unpaired A/A remains flagged: 1.0593 in the
256 chain, cycle 0; its paired estimate is 1.0013. Both aggregate controls pass.
Minimum batches range from 46.0 to 65.1 ms across contexts. No favorable process
was rerun or replaced. These results qualify bounded overhead, not a speedup.

## Evidence and remaining scope

The [machine-readable qualification](../../packages/torchwasm/benchmarks/results/webgpu-package.json)
links the [verified raw archive](../../packages/torchwasm/benchmarks/results/webgpu-package-harness.tar.gz):
2,629 files, 433 unique payloads, SHA-256
`1f68b93e2acca88f2fc509a86db6f6ce58dbd86141f3c4abfaddcbf73907a1f9`.
It preserves both package candidates, the negative binding test, failed layout
gate, final artifacts and SDK results, real browser probes, all timing samples,
source-identity audit and replay dependencies. Generated temporary trees remain
available externally; required Pyodide runtime assets and native environments
are external replay dependencies.

This moves the integration into the independent distribution. Stable GPU API
commitments, wider operations/dtypes, public resident Tensor storage, task
isolation and default dispatch remain future work. No CI, tag, publication or
deployment is enabled. The full runtime improvement goal remains active.

## Main adoption

Commit `23db63ad55090488eb407666ebd4387c5df1a723` adopts the qualified package
and compatibility entries on main. Direct remote-tree verification covers all
27 feature paths, all 77 public package resources, the unchanged 66 CPU root
resources, 183 active exports, seven release policies and 243 qualified input
files. Four targeted frontend checks pass after integrating current main.

Both feature and integration SHAs have zero workflow runs. Pages still uses the
legacy `gh-pages` branch. No tag, publication or deployment was made.
The [main adoption receipt](../../packages/torchwasm/benchmarks/results/webgpu-package-main-adoption.json)
and [raw verification evidence](../../packages/torchwasm/benchmarks/results/webgpu-package-main-adoption-harness.tar.gz)
record the guarded ordinary push and exact remote checks.
