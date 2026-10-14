# Eight more CPU/WASM lessons — 2026-10-14

Browser judging expands from **50/60 to 58/60 modules**, covering **447 maintained tests and 235 maintained mutants**. This iteration adds Block Diffusion, PKPO, DiffusionNFT, FD population statistics, JiT, Drifting, DMD and DMD2. Maintained problems, tests, reference implementations, mutants, experiment loops and thresholds are unchanged. MeanFlow and FD Loss continue to use the native server; unsupported learner APIs keep the existing server fallback.

## Compatibility work

- Signed floor division constructs the actual Block Diffusion mask, retaining negative-number floor semantics, broadcasting and dtype promotion. Integer zero division is rejected, overflowing integer division is bounded, and differentiable floor division explicitly requires the server rather than inventing a zero derivative.
- Matrix trace supports rectangular, empty and strided matrices, native integer accumulation and an analytical diagonal VJP. ReLU modules reuse the elementwise analytical VJP and reject in-place operation.
- The bounded `torch.linalg.norm` entry point supports Euclidean vectors and Frobenius reductions, including zero/empty inputs, dimension selection and retained dimensions. Explicit matrix 2-norms are spectral and require the server. Other orders, narrowing casts and output buffers are not silently approximated.
- Optional-low `randint` forms retain the existing CPU random stream, including generator state, scalar/empty shapes and supplied dtypes. Bounds outside the output integer dtype and excessive allocations are rejected. Named tensor comparisons reuse the existing precise scalar comparisons, allowing DMD's leaking-gradient mutant to fail on semantics instead of being classified as an unsupported API.

The contracts were checked against official [floor division](https://docs.pytorch.org/docs/2.14/generated/torch.floor_divide.html), [trace](https://docs.pytorch.org/docs/2.14/generated/torch.trace.html), [norm](https://docs.pytorch.org/docs/2.14/generated/torch.linalg.norm.html) and [random integer](https://docs.pytorch.org/docs/2.14/generated/torch.randint.html) documentation and installed native PyTorch 2.14.1. The production Docker PyTorch 2.6.0 image was not exercised. New operators have **75 independent native differential checks**, including analytical gradients and explicit unsupported boundaries.

## Complete experiment validation

| Module | Browser tests / caught mutants | Unchanged experiment evidence |
| --- | --- | --- |
| Block Diffusion | 7/7; 4/4 | Original 12×12 concatenated mask and all individual token perturbations; mask mismatches and forbidden influence both zero |
| PKPO | 9/9; 4/4 | 2,048 groups of 16 Gaussian samples, 15 means and k=1/4/8; gradient identity error 1.33e-15; optima 0.875 and 0.975 |
| DiffusionNFT | 9/9; 4/4 | All 60 velocity updates; final/analytic loss both 1.17, optimum error 3.52e-9, neutral velocity exactly zero |
| FD population statistics | 9/9; 4/4 | All 40 alternating microbatches; gradient error zero, batch/EMA moment errors 30 and 0.01599 |
| JiT | 6/6; 3/3 | Three original 160-update x/epsilon/v networks; x evaluation loss 0.21443 → 0.11271, unchanged 16D embedding and 40-step generation |
| Drifting | 7/7; 4/4 | Both original 260-update generators; Swiss-roll/checkerboard MMD 0.5629/0.75403 → 0.009206/0.013855 |
| DMD | 7/7; 4/4 | 220 joint updates plus all three 600-update bimodal strategies; matching and matching+regression each cover both modes |
| DMD2 | 6/6; 4/4 | Exactly 1,000 critic and 200 generator updates; mean error 0.021261 |

Each row ran through the actual built application in **Browser only** mode, with self-hosted Pyodide 0.29.4 and NumPy 2.2.5, per-run reference/mutant calibration, and the maintained T0–T3 suite. The experiment tab displayed each run's actual artifacts and current-source identity. These are the repository's bounded teaching experiments, not full-paper image/language-model benchmarks. No WebGPU or GPU path was used.

An additional disposable FD-statistics submission requested a matrix spectral norm. Browser only showed the unsupported-API/server option and marked the previous successful report as belonging to older source. It did not fabricate a norm, accept the new source or submit a native judge job.

DMD inspection also exposed a presentation bug: the legacy `comparison_loss_curves` object was recursively displayed as separate 2D point clouds. It now renders one labeled multi-series loss chart, preserving each strategy's actual step coordinates; malformed curves remain readable raw data. Dedicated rendering checks cover this distinction. The refreshed DMD experiment was checked against its persisted run.

## Checks

- `.venv/bin/python -m pytest tests/test_browser*.py -q`: **813 passed**, including all maintained browser reference/mutant calibrations, the 75 new operator comparisons and prior isolation/source/fallback checks.
- `.venv/bin/paperleet ci` with the eight modules, `--repeats 2 --budget 180`: **588 native test cases across two runs passed; all 31 new mutants caught**.
- `cd web && npm run check`: passed, including 58 unchanged-source exports, 351 compile-only starter/reference/mutant sources, browser/server filtering, plot presentation, account/draft boundaries and existing judging checks.
- `cd web && npm run build`: passed; initial JavaScript is **371,175 raw / 118,906 gzip bytes**, across three assets, within the existing homepage budget. The adapter/runtime work remains outside the homepage import graph.

Active content-addressed test/runtime assets total **1,170,324 bytes**, compared with **8,284,153 bytes** for equivalent self-contained bundles (**85.9% less**). This excludes Python/wheels and retained historical hashes. Existing hashes remain available for cached clients. Validation used a disposable loopback database and agent-owned tabs; no development/production account or database was used. Request logs recorded **zero native judge POSTs and zero HTTP errors**. The temporary tab and server were closed and the temporary database was removed. Production deployment and additional browser engines are not claimed.
