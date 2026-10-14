# Differentiable FD Loss in CPU/WASM — 2026-10-14

FD Loss now runs the unchanged nine-test suite and original affine-generator experiment in the browser. Coverage increases from **58/60 to 59/60 modules**, with **456 maintained tests and 239 maintained mutants**. MeanFlow continues to use the native server. No problem, reference, mutant, experiment loop, numerical threshold or covariance convention was changed.

## Analytical runtime behavior

- Real batched `eigh`/`eigvalsh` retain ascending spectra, the requested forward triangle, float32/float64 precision, strided inputs and empty dimensions. Separate analytical VJPs differentiate eigenvalues and eigenvectors. The backward follows native PyTorch's symmetric-matrix convention, including when the forward selected one triangle of nonsymmetric input.
- Eigenvalue VJPs use spectral projectors and do not divide by spectral gaps. Repeated positive spectra retain finite gradients for smooth spectral objectives with equal derivatives inside each eigenspace. Basis-dependent repeated-eigenvalue derivatives and degenerate/numerically unresolved eigenvector derivatives explicitly require the server. Complex eigendecomposition and output buffers remain unsupported.
- Input and both decomposition outputs retain the existing shared storage-version protection. A no-grad edit through a detached alias invalidates an old graph; the runtime does not differentiate mutated saved values.
- Determinant gradients use SVD cofactor products, including singular/rank-deficient, scalar-matrix, empty and batched cases. They never divide by a determinant or singular value. A separate minor-expansion polynomial checks singular derivatives independently of native PyTorch.
- `diagonal` supports offsets, batch dimensions and arbitrary distinct axes. It is a real storage-sharing view with an analytical scatter VJP and version invalidation, rather than a copied diagonal or a frozen gradient.

The maintained FD tests retain independent non-eigen 2×2 covariance formulas, a correlated reference covariance, repeated positive spectra, reference/weight freezing, stopped denominator gradients, symmetry/rotation/translation checks and all **200** training updates. Differentiable eigenvectors and diagonal extraction are also needed by maintained *wrong* implementations: the compatibility calibration rejects unsupported APIs rather than counting them as caught mistakes.

Contracts were checked against official [eigh](https://docs.pytorch.org/docs/2.14/generated/torch.linalg.eigh.html) and [eigvalsh](https://docs.pytorch.org/docs/2.14/generated/torch.linalg.eigvalsh.html) documentation and installed native PyTorch 2.14.1. **52 new checks** compare values and VJPs with native PyTorch, verify independent symmetric finite differences and polynomial derivatives, and cover repeated spectra, signs, triangle selection, empty/strided shapes and alias mutation. The production Docker PyTorch 2.6.0 image and additional browser engines were not exercised.

## Checks and browser evidence

- `.venv/bin/python -m pytest tests/test_browser*.py -q`: **866 passed**, including every maintained browser reference/mutant calibration and existing source, timeout, isolation and fallback checks.
- `.venv/bin/paperleet ci --problem diffusion/fd_loss --repeats 2 --budget 180`: **90 native test cases passed across two runs; all four mutants caught**.
- `cd web && npm run check`: passed, including **59 source-faithful exports** and **357 compile-only starter/reference/mutant sources**.
- `cd web && npm run build`: passed; the initial graph remains **371,175 raw / 118,906 gzip bytes** across three JavaScript assets. Python and the compatibility adapter still load on demand.
- Actual in-app browser, self-hosted Pyodide **0.29.4** and NumPy **2.2.5**, **Browser only**: reference **9/9**, calibration **9 reference tests / 4 mutants**. The submitted-run lab displayed its actual 0–200-step curves and current-source identity.
- The independent raw FD fell from **9.2661 to 0.0073302**; final EMA FD was **0.0065923**, mean error **0.049628**, covariance error **0.14498** and minimum covariance eigenvalue **0.60069**. Normalized objective values are shown separately; no raw-distance clipping or diagonal jitter was added.
- Submitting the maintained unfrozen-reference mutant returned **8/9**, with a genuine reference-gradient failure rather than an unsupported-API outcome.
- A disposable source requested eigenvector gradients at an identity matrix. It showed the unsupported-API/server option and retained the preceding report with an older-source label. No synthetic finite gradient or successful new grade was returned. Restoring the unchanged reference and running again returned 9/9 with full calibration.

Active content-addressed browser assets total **1,204,990 bytes**, compared with **8,676,079 bytes** for equivalent self-contained bundles (**86.1% less**), excluding Python/wheels and retained historical hashes. Validation used a temporary loopback database and agent-owned tab; no development/production account or database was used. Request logs confirmed the final manifest/runtime were fetched, with zero native judge POSTs and zero HTTP errors. The temporary tab and server were closed and the temporary database was removed. No production deployment or full-paper image-generation benchmark is claimed.
