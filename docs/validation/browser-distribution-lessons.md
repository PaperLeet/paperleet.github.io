# DAPS, SIGReg and VISReg in CPU/WASM — 2026-10-14

Three additional maintained modules now execute their complete tests, reference/mutation calibration and original experiments in the browser. Coverage increases from **44/60 to 47/60 modules**, comprising **362 maintained tests and 192 maintained mutants**. No problem source, reference, mutant, experiment loop or numerical threshold was edited.

## Runtime changes

- Real float32/float64 matrix inverses support batches and use the analytical inverse VJP. Singular/non-square matrices fail; complex inverse and output buffers retain server execution.
- Sort returns actual values and int64 source indices; its VJP scatters through those indices. Native comparisons cover scalar, empty and strided arrays, stable ascending/descending ties and NaNs, plus default sorting away from ties. Unstable tie identities have no unique ordering guarantee.
- Trapezoidal integration retains gradients through both values and irregular coordinate nodes, with one-dimensional and supported broadcast coordinates. Empty integration returns detached zeros, as in the tested native CPU. Unsupported ranks/dtypes are explicit fallbacks.
- Inverse error function uses `statistics.NormalDist` lower-tail quantiles without loading another package. A central erf correction preserves tiny nonzero inputs whose quantile probability rounds to 0.5. It preserves dtype, signed zero, NaN/out-of-domain behavior, infinite endpoints and analytical first derivatives. Ordinary values/gradients are compared with native PyTorch. Near ±1, the installed native CPU's erf-based Newton correction loses measurable accuracy; those tails instead have an independent `math.erfc` round-trip check. The browser implementation does not imitate that rounding error.
- Frozen complex numerical arithmetic preserves dtype promotion and imaginary values in the original SIGReg full-grid characteristic-function oracle. Native comparisons cover mixed real/complex dtypes, array/scalar ranks, arithmetic, exponentials, means, absolute values and integration. Complex autograd and lossy complex-to-real casts, along with unvalidated paths such as comparisons, variance, sigmoid and softmax, explicitly require the server. The learner's SIGReg gradient path remains real and fully differentiable.
- `linspace(..., requires_grad=True)` produces an actual differentiable leaf. SIGReg's supplied node tensor can then be detached by the implementation rather than being silently constructed without gradients.

API contracts were checked against the official [inverse](https://docs.pytorch.org/docs/2.14/generated/torch.linalg.inv.html), [sort](https://docs.pytorch.org/docs/2.14/generated/torch.sort.html), [trapezoid](https://docs.pytorch.org/docs/2.14/generated/torch.trapezoid.html) and [inverse error function](https://docs.pytorch.org/docs/2.14/generated/torch.erfinv.html) documentation, within the bounded teaching subset. Native differential checks used installed PyTorch 2.14.1; the production Docker PyTorch 2.6.0 image was not exercised.

## Maintained experiments

| Module | Complete suite and calibration | Original experiment |
| --- | --- | --- |
| DAPS | 9/9; all 4 mutants caught | All 240 Langevin updates on 4,096 samples, analytic Gaussian posterior/finite-step covariance, independent re-noising |
| SIGReg | 9/9; all 4 mutants caught | All 100 Adam updates on 64 embeddings and 16 directions; independent full symmetric-grid complex oracle and same-moment distribution comparison |
| VISReg | 9/9; all 4 mutants caught | All 80 updates on three 24-by-2 views; sorted shape discrepancy, fixed normalization gradients and shared global-view anchor |

## Checks and UI evidence

- `.venv/bin/python -m pytest tests/test_browser*.py -q`: **638 passed**. This includes all browser suites/mutants plus **102 new numerical/operator checks**, unsupported-path behavior, and existing source/protocol/version protections.
- `.venv/bin/paperleet ci --problem diffusion/daps --problem representation/sigreg --problem representation/visreg --repeats 2 --budget 180`: **3 problems, 270 test cases, 2 runs** passed; all 12 native mutants caught.
- `cd web && npm run check`: passed, including all 47 source-faithful bundles, 286 compile-only source checks, runtime filtering, account ownership, queue fallback, callback deadlines, source identity, and experiment rendering.
- `cd web && npm run build`: passed TypeScript and size budgets; initial graph **371,092 bytes raw / 118,875 bytes gzip**, across three JavaScript assets. No new JavaScript dependency or runtime wheel was added.
- Real in-app browser, self-hosted loopback production build, Browser only: DAPS, SIGReg and VISReg each passed **9/9** with **9 reference tests and all 4 mutants**. Native judge POST count and HTTP error count were both **zero**. The maintained T3 tests and their original generated charts were inspected in My run results.
- Observed DAPS browser metrics: mean error ≈0.013547, ULA covariance error ≈0.028918, re-noising variance error ≈0.00022775, measurement MSE 2.25 →0.28798. Original acceptance thresholds are unchanged.
- Observed SIGReg browser objective: 27.989 →0.15205; minimum feature standard deviation ≈0.99664. Its full-grid oracle passed the original absolute-error threshold `<1e-10`.
- A signed-out hosted-style guest completed VISReg without a server job. Objective was 0.84483 →0.027846; mean feature standard deviation 0.15 →0.81674; squared feature center 0.88667 →0.000092961. The browser result explicitly remained local and unverified for account statistics.
- SIGReg with a differentiable complex expression explicitly returned the unsupported Python/PyTorch API notice in Browser only mode. It submitted no native job and preserved the prior report; restoring the exact reference reran full calibration and returned 9/9.
- A separate SIGReg QA submission executed additional assertions inside the actual WASM Python runtime: inverse-error-function values at ±1e-300 and ±1e-20 retained relative accuracy, and the float64 value immediately below one passed an `erfc` tail round trip. Its complete maintained suite still passed 9/9. These assertions were added only to the disposable QA submission, never to a maintained problem file.
- The right-hand Experiment Lab previously forced reference curves even after a successful submission. It now uses automatic source selection. Real UI checks confirmed a new guest sees reference curves before running, then My run results after a successful run; an existing local SIGReg report also opens My run results. Explicit reference/concept entry points remain available.

Screenshots of all three actual-run labs were inspected. The temporary servers and test tabs were closed after verification. Browser execution used a temporary database and a server that rejected native judge submissions. This validation does not claim full PyTorch/complex-autograd support, full-paper pretraining, additional browser-engine coverage or production deployment.
