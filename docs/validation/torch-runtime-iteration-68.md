# Torch runtime iteration 68

Version **0.68.0** releases traversal-local references to consumed joint cotangents before predecessor work. Previously, the returned gradient tuple and the final loop/conversion locals could keep arrays alive after pending predecessor gradients had replaced them. The pending map still owns gradients that remain necessary. Retained callbacks, storage guards, hooks, target selection, duplicate parents and analytical derivatives keep their existing rules.

Eight new lifetime regressions fail on adopted 0.67 and pass on 0.68. A rejected first prototype deleted an unbound conversion local on filtered nonleaf backward targets; its source/failing log remain in the raw archive. The selected implementation clears that local safely. The alternative consumed-seed reuse implementation provided similar allocation gains and was retained as an unadopted experiment.

## Validation

- Full repository Python regression: **41,716 passed, 1 skipped**, 34 warnings, 580.24 seconds. This includes the existing native-Torch API comparisons. Python/application/numerical inputs retain the proved hashes through the disjoint upstream UI updates.
- Actual standalone wheel/source archive: **8,092 installed tests** on Python 3.10.19 / NumPy 1.26.4 without native Torch. Final README metadata and all 61 source/license resources match the integrated package. Standalone/private-wheel ownership, uninstall, aliases, checkpoint resume and fresh-process pickle activation pass.
- Real Node and Chrome SDK checks: **8,079 cases** each, against the exact versioned 0.68 source and Pyodide 0.29.4 / NumPy 2.2.5.
- Full exported WASM catalog: **85 modules / 378 mutants**, no reported browser budget exceedance. Its independent 63-suite SDK run matches all published test hashes.
- Complete current frontend checks and both Node 22.23.3 build profiles pass locally, including the production Pages environment read from the repository setting. Reviewed upstream UI baseline: `b44bab6b1e225e9ae8fa3616443d020597d14f3c`. The upstream startup-budget repair is included without changing the budgets.
- All **13,735** previous immutable assets remain unchanged; **172** new assets are reachable through the **175** active export/index identities. Both runtime profiles match versioned source. Pinned runtime resources and all seven release-policy files retain their reviewed hashes.

## Allocation and latency evidence

The complete self-attention training-step probe includes input gradients, loss, backward, gradient clearing and SGD. Native Torch 2.14.1 outputs, gradients and updated parameters are checked before tracing. For batch 2, length 64, dimension 256, tracked incremental CPU allocation peaks are approximately **15% lower** in both float32 and float64. Smaller cases vary; all six MLP training-step negative controls show **no peak reduction**. These measurements are not RSS, browser heap or GPU memory.

A controlled synthetic joint-gradient graph shows approximately 20% lower tracked allocation at larger widths. That original prototype has identical graph/numerical bytes to 0.68 and a different version literal; the samples are preserved with their original identity.

Twelve serial fresh timing processes cover 30 complete flows across CPU, Node and Chrome. Initial GroupNorm outliers include about 3% slower medians and remain in the results. Eighteen further fresh A/A/B processes rotate all six module-load orders and use 12 balanced common batches of at least 50 ms for five selected outliers. Their candidate/baseline median range is **0.9957–1.0023**. The longer controls do not reproduce the initial GroupNorm slowdown; this is not a uniform speedup claim.

## WebGPU feasibility

Three fresh Chrome processes complete **24** bounded float32 matrix multiplications through a secure-host GPU service, requested by Pyodide in an opaque-origin `allow-scripts` Worker that still exposes no GPU. Nine malformed operation/shape/size requests are rejected. Results match NumPy, including strided inputs packed for transfer. Asynchronous `runPythonAsync` plus JSPI makes this bridge possible on the tested browser.

This is a separate developer fixture: **no torch GPU backend is enabled**. It does not qualify gradients, batched or zero-sized operations, mixed precision, cancellation/device-loss recovery, production CSP/resource policy, browser-wide support or throughput. CPU tests were running concurrently, so no GPU performance inference is made. The separate functional VJP prototype is not included in 0.68.

## Evidence and release

Machine-readable evidence: [`iteration-68.json`](../../packages/stridetorch/benchmarks/results/iteration-68.json). Raw scripts, deterministic fixtures, original samples, failures, logs and package artifacts: [`iteration-68-harness.tar.gz`](../../packages/stridetorch/benchmarks/results/iteration-68-harness.tar.gz). SHA-256: `0e13d5d5928ef49a02760f3e2acffa375fca6a22997dd196c1801c8ed85e03aa`; 4,443,832 bytes. Every logical archive payload is verified; identical payloads with matching permissions/ownership use tar hardlinks. Filesystem timestamps are not comparison inputs. Replaying original scripts requires installing the recorded dependencies and adapting original workspace paths.

Local validation does not itself claim main adoption or deployment. Main adoption is recorded separately. Ordinary main pushes must continue to use the checked release-chain guard. No workflow dispatch, tag, package publication or deployment is requested by this iteration.

## Current-main frontend extension

After source validation, upstream `bf6915ba5a5a731b206f9abcc24c02ad8eadbd32` added deferred community tag discovery. Merge `847c45286c1196a2ec64f72b7ace7fdca472bbdf` preserves all original 0.68 numerical, export, package and report/archive payloads. The complete **149-command** frontend suite, WASM driver, Pages policy and both Node 22.23.3 builds pass again. Current production startup graph: **408,681 raw / 132,849 gzip** bytes. No budgets or release triggers were relaxed. [Machine-readable current-UI qualification](../../packages/stridetorch/benchmarks/results/iteration-68-current-ui.json).

## Main adoption

Version 0.68.0 was adopted on main at `c13563b0cfb66af2f993e89a95b21314bcfc4960`, including implementation `c84a3210e8235108e182d5e70e61dc66b5398116` and the qualified current UI. A guarded ordinary push passed with no active release chain. The remote ref and 313 Git blobs were checked; all 13,735 earlier immutable asset tree entries remain identical. The adopted SHA had zero Actions runs at verification. No workflow dispatch, tag, package publication or deployment was initiated. [Adoption receipt](../../packages/stridetorch/benchmarks/results/iteration-68-main-adoption.json).
