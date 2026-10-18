# Iteration 78: strict matrix APIs and experimental GPU failure recovery

Version 0.76 adds `torch.mm` / `Tensor.mm` and `torch.bmm` / `Tensor.bmm`. Normal products reuse the qualified 0.75 CPU/WASM engine. The separate GPU experiment fixes error rollback and cancellation; this change does not add a Torch GPU device or claim an additional speedup.

## Matrix contracts and empty-result work

`mm` requires two rank-two inputs. `bmm` requires two rank-three inputs with equal batch counts; it does not broadcast batches. Ordinary products require matching dtypes and valid inner dimensions. Nonempty bool and uint16/uint32/uint64 products are rejected consistently with the native CPU controls. Float16, integer and complex inference retain the existing library boundaries. Reverse AD, JVP and functional HVP are qualified for float32/float64. Real output buffers remain explicitly unsupported; top-level `out=None` is accepted and Tensor methods do not accept `out`. The ordinary torch aliases and registered-operator Module serialization expose the new names.

Native Torch 2.6 treats empty bmm differently from mm: bmm can choose mat2’s dtype before numeric-dispatch checks, while mm still requires equal dtypes. The selected implementation handles empty bmm using shape/dtype metadata and zero-valued differential rules. It does not cast or read either input. Opposite-operand storage guards remain attached to original inputs, including mixed-dtype empty products. The new regression tests also bound the traced allocation of a zero-output call below 64 KiB when a source-sized cast would allocate 512 KiB of data. This is a tracked operation-allocation criterion, not process RSS.

The discarded first candidate cast an operand to reuse matmul. This preserved tested values but could copy a large nonempty input for an empty result, and could introduce irrelevant complex/nonfinite conversion behavior. Its exact source and diagnostics are retained. The selected metadata-only path is requalified independently; no source-equivalence claim is made between those implementations.

## Independent and installed-package validation

Native Torch 2.6 generates 140 inference/error cases and 48 ordinary plus eight mixed-empty AD fixtures, retaining 576 full arrays. Controls cover values, dtypes/shapes, VJPs, JVP and HVP, including Fortran/strided storage and qualified large WASM contractions. Separate native comparisons cover 108 saved-storage mutations and all 14 × 14 dtype pairs across four empty conditions (784 cases), including nonfinite/complex source data. All empty outputs match without conversion warnings. The original oracle program and payload hashes are retained.

The selected source passes 2,181 new matrix contracts. Node and real Chrome/Pyodide each pass all 11,183 core contracts, every full-array native control, the 108 storage checks and 784 dtype cases. Pyodide uses the pinned 0.29.4 runtime and NumPy 2.2.5.

Complete native application/library suite: 44825 passed, 1 skipped, 34 warnings in 275.35s (0:04:35).

The independent wheel on Python 3.10.19 / NumPy 1.26.4, without native Torch or PaperLeet installed, reports: 11193 passed, 3 skipped, 41 warnings in 31.22s. The three skipped cases are the pre-existing ranks that NumPy 1.26 cannot construct. Public and private wheels contain the exact 62 selected code/notice resources; both source archives are also rebuilt independently and produce wheels with those exact resources and current READMEs.

After co-installation, the public package, PaperLeet private namespace and installed `stridetorch` launcher all execute unchanged `import torch` matrix code and gradients. The CLI preserves its arguments. No native Torch is installed in that environment. The app sdist intentionally stores the canonical source under packages/stridetorch/src; its build backend creates the private wheel namespace. This layout is verified by rebuilding, rather than assuming the sdist directly contains the private wheel layout.

## Experimental WebGPU recovery

Primitive upload/matmul/read failures wait for possibly issued work before destroying temporary buffers and reclaiming their byte accounting. If draining fails, the device closes and all resident handles become invalid. Acknowledged cancellation now takes precedence over a later execution error while the service remains open; subsequent requests wait for rollback to complete. Both the original early destruction and the cancellation/error race are reproduced in retained diagnostics.

Portable checks cover 39 primitive failure/lifetime cases, four cancellation/failure races, 40 program checks and nine protocol checks. A shared injected fixture supports the three engine-check suites. Real Chrome additionally passes 13 controlled host-fault tests using real GPU buffers/queues and numerical recovery, plus four lifecycle tests. The observed adapter is Apple Metal 3 with fallback=false; headless Chrome uses no additional launch flags. Explicit host exceptions, delayed completion and device destruction are test injections, not spontaneous hardware-fault coverage or physical VRAM evidence.

The unchanged native GPU oracle covers 36 finite float32 2D matrix/analytical-VJP cases, all transpose combinations and independent float64 controls. The final service passes those fixtures, 108 program executions in three ownership modes, 13 primitive and four program boundaries, and cancellation/ownership/isolation checks. Shader and default limits are byte-identical to the previous prelude. GPU code remains an explicit Matrix/RPC experiment; it is not automatic Torch GPU autograd.

## Application and release boundary

Regenerated exports pass 89/89 real WASM modules and 406 mutants, plus complete frontend, policy, and both build-profile gates.

All 15,139 prior immutable resources are preserved; 180 new reachable assets and 183 live export/index identities represent 0.76. The application execution base is local commit `0ed2aa3a95d9e7febf683874b5dafd1284af5f1f`, which contains the previously qualified but unpushed 0.75 work. Remote main was `7e7b64a5a5cfd6f296b76a7f6e740fd2f0900889` at execution. No 0.75 or 0.76 main adoption is implied by local qualification.

One early CPU attempt ended with exit 120 while local disk space was nearly exhausted and never produced a pytest summary. It is preserved as incomplete, not counted as passing. Completed task-owned temporary trees were removed, source/evidence preserved, and the selected source received the successful fresh suite above. Other failed harness attempts and the unselected casting implementation remain distinguishable from qualified results.

Release policies are unchanged. No CI, tag, package publication or deployment was initiated. [Machine record](../../packages/stridetorch/benchmarks/results/iteration-78.json).

[Raw evidence](../../packages/stridetorch/benchmarks/results/iteration-78-raw.tar.gz) — 7,835,131 bytes; SHA-256 `3d4320960c219dace85493cbdbe2d27cbd528d6f1fb3098ae698402c38eb2d28`.

## Rebased UI validation

Remote main advanced by three UI commits to `58ff4d8ba738c0d7929bca7e80d4743d99f83adb` while the 0.76 application checks ran. Both tested unpublished commits were preserved at `refs/codex/checkpoints/torch-iteration-78-before-ui-rebase` and rebased without conflicts. The rebased tree `5a7e8d7a0a4405bf4b99e6de73b3819bbe9ddc12` retains all 26 external UI/validation changes. All 62 core resources, 78 test files, 1,426 application inputs, live exports and release-policy identities match the original qualification. Existing frontend dependencies and check commands are preserved; two new frontend checks are included.

The complete frontend checks, default/current-Pages-profile builds, release-policy checks and exact generated-worker verification were rerun on that rebased tree and passed. Numerical CPU/WASM runs retain their original execution SHA; no numerical rerun on the rebased Git parent is claimed. The combined qualification ties unchanged compute/application inputs to the new tested frontend.

## Path-routing integration

A further 106 upstream paths were reconciled at `3f56fa6863b38db12ea0b2153ce396b3e8396672`. The actual frontend qualification ran on `170e134d722bac4aa39b8d32e0732aa823f39a43`: complete frontend checks, release-policy checks, default build, path-routing production-profile build and exact generated-worker verification passed. The production build verified all 166 static HTTP 200 routes, metadata, no-JS content, root assets and real missing-page 404s. Upstream added `VITE_PATH_ROUTING=1` to the CI/Pages build environment; release triggers and SHA gates were preserved. All 62 core resources, 78 tests, 1,426 application inputs, live exports and 15,139 prior immutable resources retained their qualified identities. Original numerical executions retain their original SHA.

A fresh real Chrome session opened `/problems/architecture/attention/`, edited the canonical solution through CodeMirror, and executed the real judge Worker with Pyodide/NumPy WASM. All 7/7 Attention tests passed, together with explicit `mm` gradient, nonempty `bmm`, and mixed-dtype empty `bmm` assertions. The hosted health and anonymous-account responses were trusted test fixtures; all other external requests, including writes, were blocked. This verifies the local browser execution flow, without claiming live backend/API integration. The first harness attempt stopped at the expected read-only health precondition and is retained separately.

Upstream then added a three-file static-asset retention fix at `891f620921791775082e2e290ff5caf8e06e5b72`. It normalizes root-relative asset references and adds local validation; the exact workflow delta only adds two validation commands. The resulting local tree `4042315876503fbaaf3fe2e6f5f793feb3b7d829` preserves all other tested frontend/build and numerical inputs. Twenty asset-retention tests, a read-only validation of the actual 253-asset production build, release-gate checks and backfill checks passed. Prior browser-served build payloads were rechecked byte-for-byte. No CI, tag, publication or deployment was initiated.

## Concurrent UI recovery

An ordinary guarded push after the existing release chain ended failed with a transport broken pipe; no adoption is inferred from that attempt. Main independently advanced to `a16a0e54605fba6cea4d06603814ae664b12e4da` by two UI commits (19 paths), preserved at reconciliation. The rebased execution tree `830b1318afbb942f9ce7ba9021ceead6b9bf8b8e` retains those interview/admin changes, unchanged frontend dependencies, both newly added checks and all original check commands. Core/test/application/export and current release-policy identities are unchanged.

Complete frontend checks, both default and production path-routing builds, asset-retention checks, release policies and exact worker regeneration passed on the new tree. The fresh nested Attention page again passed 7/7 real Worker/Pyodide tests plus explicit mm/bmm assertions, with the same documented trusted health/anonymous-account fixtures and blocked external writes. Original full numerical test SHAs are preserved. No CI, tag, package publication or deployment was initiated.
